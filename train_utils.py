import logging
import pathlib
from typing import List, Union

import os, sys
import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import importlib
try:
    importlib.import_module("apex.optimizers")
    from apex.optimizers import FusedAdam, FusedLAMB
except Exception as e:
    pass
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from train.dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, Logger, WandbLogger
# from transformers import AdamW
import curtsies.fmtfuncs as cf
import torchmetrics

# https://fairscale.readthedocs.io/en/latest/
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper #anytime
from fairscale.experimental.nn.offload import OffloadModel #Single-GPU
from fairscale.optim.adascale import AdaScale #DDP
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP #Sharding
from fairscale.optim.oss import OSS #Sharding
import shutil
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.wrap import (
					# default_auto_wrap_policy,
					enable_wrap,
					wrap,
					)

def save_state(model: nn.Module, optimizer: Optimizer, scheduler_groups: "list of schedulers", epoch: int, val_loss: int, path_and_name: Union[pathlib.Path, str]):
    """ Saves model, optimizer and epoch states to path (only once per node) 
    Only local rank=0!"""
    if get_local_rank() == 0:
        scheduler_kwargs = {name : sch.state_dict() for name, sch in zip(["step_scheduler", "epoch_scheduler"], scheduler_groups)}
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        checkpoint = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            **scheduler_kwargs
        }

        torch.save(checkpoint, str(path_and_name))
        print(cf.on_yellow(f"Save a model in rank {get_local_rank()}!"))

def load_state(model: nn.Module, optimizer: Optimizer, scheduler_groups: "list of schedulers", path_and_name: Union[pathlib.Path, str], model_only=False, use_artifacts=False, logger=None, name=None):
    """ Loads model, optimizer and epoch states from path
    Across multi GPUs
    logger and name kwargs are only needed when use_artifacts is turned on"""
    if use_artifacts and opt.log: 
        model_name = path_and_name.split("/")[-1]
        name = model_name.split(".")[0]
        if get_local_rank() == 0:
            prefix_dir = logger.download_artifacts(f"{name}_model_objects")
            shutil.copy(os.path.join(prefix_dir, model_name), path_and_name + ".artifacts") #copy a file from artifcats dir to save dir! add .adtifacts at the end!
        if dist.is_initialized(): 
            dist.barrier(device_ids=[get_local_rank()]) 
#         path_and_name = os.path.join(prefix_dir, model_name)
        path_and_name += ".artifacts"
	
    ckpt = torch.load(path_and_name, map_location={'cuda:0': f'cuda:{get_local_rank()}'}) #model, optimizer, scheduler
    try:
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(ckpt['model']) #If DDP is saved...
        else:
            model.load_state_dict(ckpt["model"])
        if not model_only:
            optimizer.load_state_dict(ckpt["optimizer"])
            step_scheduler = scheduler_groups[0]
            epoch_scheduler = scheduler_groups[1]
            step_scheduler.load_state_dict(ckpt["step_scheduler"])
            epoch_scheduler.load_state_dict(ckpt["epoch_scheduler"])
            val_loss = ckpt["val_loss"]
            epoch = ckpt["epoch"]
        if model_only:
            epoch = 0
            val_loss = 1e20
    except Exception as e:
        print(e)
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(ckpt) #If DDP is saved...
        else:
            model.load_state_dict(ckpt)
        epoch = 0
        val_loss = 1e20
    finally:
        print(f"Loaded a model from rank {get_local_rank()}!")
    return epoch, val_loss

def single_train(args, model, loader, loss_func, epoch_idx, optimizer, scheduler, grad_scaler, local_rank, logger: WandbLogger, tmetrics):
    #add grad_scaler, local_rank,
    model = model.train()
    losses = []
    path_and_name = os.path.join(args.load_ckpt_path, "{}.pth".format(args.name))
    _loss = 0.
    _loss_metrics = 0.

    pbar = tqdm(enumerate(loader), total=len(loader), unit='batch',
                         desc=f'Training Epoch {epoch_idx}', disable=(args.silent or local_rank != 0))
    for step, packs in pbar:
        if args.gpu and args.use_tensors:
            assert args.backbone in ["cphysnet","cschnet","cgcnn","ctorchmdnet", "calignn", "graph_transformer"], "Wrong data format for a given backbone model!"
            pack, names = packs[0], packs[1]
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, targetE = pack.x, pack.edge_attr, pack.edge_index, pack.cif_id, pack.batch, pack.edge_weight, pack.y
            pack =  atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE
            # print(atom_fea, targetE)
            atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE = to_cuda(pack)
        else:
            print("Significant error in dataloader!")
            break
		
        with torch.cuda.amp.autocast(enabled=args.amp):
            preds = model(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch) 
            loss_mse = loss_func(args, preds, targetE) #get_loss_func
            loss_metrics = tmetrics(preds.view(-1,).detach().cpu(), targetE.view(-1,).detach().cpu()) #LOG energy only!
            
        if args.log:
            logger.log_metrics({'rank0_specific_train_loss_mse': loss_mse.item()})
            logger.log_metrics({'rank0_specific_train_loss_mae': loss_metrics})

        loss = loss_mse
        
        grad_scaler.scale(loss).backward()
        # gradient accumulation
        if (step + 1) % args.accumulate_grad_batches == 0 or (step+ 1) == len(loader):
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            model.zero_grad(set_to_none=True)
            #scheduler.step() #stepwise (self.last_epoch is called (as a step) internally)  
#         losses.append(loss)
        _loss += loss.item()
        _loss_metrics += loss_metrics.item()
        #if step % 10 == 0: save_state(model, optimizer, scheduler, epoch_idx, path_and_name) #Deprecated
        pbar.set_postfix(mse_loss=loss.item(), mae_loss=loss_metrics.item() if hasattr(loss_metrics, "item") else loss_metrics)

#     return torch.cat(losses, dim=0).mean() #Not MAE
    return _loss/len(loader), _loss_metrics/len(loader) #mean loss; Not MAE


def single_val(args, model, loader, loss_func, optimizer, scheduler, logger: WandbLogger, tmetrics):
    model = model.eval()
    _loss = 0
    _loss_metrics = 0.

    with torch.inference_mode():  
        pbar = tqdm(enumerate(loader), total=len(loader), unit='batch',
                            desc=f'Validation', disable=(args.silent or get_local_rank() != 0))
        for step, packs in pbar:
            pack, names = packs[0], packs[1]

            if args.gpu and args.use_tensors:
                assert args.backbone in ["cphysnet","cschnet","cgcnn","ctorchmdnet", "calignn", "graph_transformer"], "Wrong data format for a given backbone model!"
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, targetE = pack.x, pack.edge_attr, pack.edge_index, pack.cif_id, pack.batch, pack.edge_weight, pack.y
                pack =  atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE
                atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE = to_cuda(pack)	
            else:
                print("Significant error in dataloader!")
                break
		
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds = model(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch) 
                loss_mse = loss_func(args, preds, targetE) #get_loss_func
                loss_metrics = tmetrics(preds.view(-1,).detach().cpu(), targetE.view(-1,).detach().cpu()) #LOG energy only!

            if args.log:
                logger.log_metrics({'rank0_specific_val_loss_mse': loss_mse.item()})
                logger.log_metrics({'rank0_specific_val_loss_mae': loss_metrics})

            loss = loss_mse
            _loss += loss.item()
            _loss_metrics += loss_metrics.item()

    return _loss/len(loader), _loss_metrics/len(loader) #mean loss; Not MAE
                
def single_test(args, model, loader, loss_func, optimizer, scheduler, logger: WandbLogger, tmetrics):
    model = model.eval()
    _loss = 0
    _loss_metrics = 0.

    with torch.inference_mode():  
        pbar = tqdm(enumerate(loader), total=len(loader), unit='batch',
                            desc=f'Test', disable=(args.silent or get_local_rank() != 0))
        for step, packs in pbar:
            pack, names = packs[0], packs[1]

            if args.gpu and args.use_tensors:
                assert args.backbone in ["cphysnet","cschnet","cgcnn","ctorchmdnet", "calignn", "graph_transformer"], "Wrong data format for a given backbone model!"
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, targetE = pack.x, pack.edge_attr, pack.edge_index, pack.cif_id, pack.batch, pack.edge_weight, pack.y
                pack =  atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE
                atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE = to_cuda(pack)	
            else:
                print("Significant error in dataloader!")
                break
		
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds = model(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch) 
                loss_mse = loss_func(args, preds, targetE) #get_loss_func
                loss_metrics = tmetrics(preds.view(-1,).detach().cpu(), targetE.view(-1,).detach().cpu()) #LOG energy only!

            if args.log:
                logger.log_metrics({'rank0_specific_test_loss_mse': loss_mse.item()})
                logger.log_metrics({'rank0_specific_test_loss_mae': loss_metrics})

            loss = loss_mse
            _loss += loss.item()
            _loss_metrics += loss_metrics.item()

    return _loss/len(loader), _loss_metrics/len(loader) #mean loss; Not MAE
	
def train(model: nn.Module,
          get_loss_func: _Loss,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
	  test_dataloader: DataLoader,
          logger: Logger,
          args):
    """Includes evaluation and testing as well!"""

    #DDP options
    #Init distributed MUST be called in run() function, which calls this train function!

    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    tmetrics = torchmetrics.MeanAbsoluteError()
    
    #DDP Model
    if dist.is_initialized() and not args.shard:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()
        print(f"DDP is enabled {dist.is_initialized()} and this is local rank {local_rank} and sharding is {args.shard}!!")    
        model.train()
        if args.log: logger.start_watching(model) #watch a model!
    elif dist.is_initialized() and args.shard:
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=100
        )
        torch.cuda.set_device(local_rank)
        model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    #Grad scale
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    #Optimizer
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay)
        base_optimizer = FusedAdam
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay)
        base_optimizer = FusedLAMB
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                    weight_decay=args.weight_decay)
        base_optimizer = torch.optim.SGD
        base_optimizer_arguments = dict(lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'torch_adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay, eps=1e-8)
        base_optimizer = torch.optim.Adam
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    elif args.optimizer == 'torch_adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay, eps=1e-8)
        base_optimizer = torch.optim.AdamW
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    elif args.optimizer == 'torch_sparse_adam':
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              eps=1e-8)
        base_optimizer = torch.optim.SparseAdam
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

#     if args.shard:
#         optimizer = OSS(
#             params=model.parameters(),
#             optim=base_optimizer,
#             **base_optimizer_arguments)
#         model = ShardedDDP(model, optimizer) #OOS optimizer and Single-gpu model to ShardedDDP

    #SCHEDULER
    total_training_steps = len(train_dataloader) * args.epoches
    warmup_steps = total_training_steps // args.warm_up_split
    scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_training_steps) #can be used for every step (and epoch if wanted); per training step?
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9, patience=3, verbose=True) #needs a validation metric to reduce LR (perhaps mainly for epoch-wise eval)
    #Load state (across multi GPUs)

    path_and_name = os.path.join(args.load_ckpt_path, "{}.pth".format(args.name))
    scheduler_groups = [scheduler, scheduler_re] #step and epoch schedulers
    epoch_start, best_loss = load_state(model, optimizer, scheduler_groups, path_and_name, use_artifacts=args.use_artifacts, logger=logger, name=args.name) if args.resume else (0, 1e20)
    
    best_loss = best_loss
    #DDP training: Total stats (But still across multi GPUs)
    init_start_event.record()
    for epoch_idx in range(epoch_start, args.epoches):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)
        
        ###TRAINING
        train_epoch = single_train
        #DDP training: Individual stats (i.e. train_epoch; also still across multi GPUs)
        loss, loss_metrics = train_epoch(args, model, train_dataloader, get_loss_func, epoch_idx, optimizer, scheduler, grad_scaler, local_rank,
                           logger, tmetrics) #change to single_train with DDP ;; model is AUTO-updated...

        #ZERO RANK LOGGING
        if dist.is_initialized():
            loss = torch.tensor(loss, dtype=torch.float, device=device)
            loss_metrics = torch.tensor(loss_metrics, dtype=torch.float, device=device)

            """
            #Works!
            #https://github.com/open-mmlab/mmdetection/blob/482f60fe55c364e50e4fc4b50893a25d8cc261b0/mmdet/apis/test.py#L160
            #only on local rank 0
	    device = torch.cuda.current_device()
            losses = [torch.tensor(0., device=device) for _ in range(world_size)] #list of tensors: must put "device=cuda"
            torch.distributed.all_gather(losses, loss) #Gather to losses!
	    losses = torch.tensor([_ for _ losses]).to(device)
            #losses = loss.new_zeros(len(losses)).data.copy_(torch.tensor(losses)) #tensor of size world_size
            """

            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM) #Sum to loss
            loss = (loss / world_size).item()
            #print(f"Sanity check: all_reduced GPU mean loss {loss} AND all_gather GPU mean loss {losses.mean()}...")
            logging.info(f'Train loss: {loss}')
            torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
            loss_metrics = (loss_metrics / world_size).item()
            #print(f"Sanity check: all_reduced GPU mean loss {loss} AND all_gather GPU mean loss {losses.mean()}...")
            logging.info(f'Train MAE: {loss_metrics}')
	
        if args.log: 
            logger.log_metrics({'ALL_REDUCED_train_loss': loss}, epoch_idx) #zero rank only
            logger.log_metrics({'ALL_REDUCED_train_MAE': loss_metrics}, epoch_idx) #zero rank only
        #mae_reduced = tmetrics.compute() #Synced and reduced autometically!
        #logger.log_metrics({'ALL_REDUCED_train_mae_loss': mae_reduced.item()}, epoch_idx) #zero rank only
        tmetrics.reset()
        
        ###EVALUATION
        evaluate = single_val
        val_loss, loss_metrics = evaluate(args, model, val_dataloader, get_loss_func, optimizer, scheduler, logger, tmetrics) #change to single_val with DDP
        if dist.is_initialized():
            val_loss = torch.tensor(val_loss, dtype=torch.float, device=device)
            loss_metrics = torch.tensor(loss_metrics, dtype=torch.float, device=device)
            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
            val_loss = (val_loss / world_size).item()
            torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
            loss_metrics = (loss_metrics / world_size).item()
        if args.log: 
            logger.log_metrics({'ALL_REDUCED_val_loss': val_loss}, epoch_idx)
            logger.log_metrics({'ALL_REDUCED_val_MAE': loss_metrics}, epoch_idx) #zero rank only
#zero rank only
        #mae_reduced = tmetrics.compute() #Synced and reduced autometically!
        #logger.log_metrics({'ALL_REDUCED_val_mae_loss': mae_reduced.item()}, epoch_idx) #zero rank only
        tmetrics.reset()
        #scheduler_re.step(val_loss) #Not on individual stats but the total stats
        scheduler.step()

        scheduler_groups = [scheduler, scheduler_re] #step and epoch schedulers
        if val_loss < best_loss:
            path_and_name = os.path.join(args.load_ckpt_path, "{}.pth".format(args.name))
            save_state(model, optimizer, scheduler_groups, epoch_idx, val_loss, path_and_name)
            if args.log: logger.log_artifacts(name=f"{args.name}_model_objects", dtype="pytorch_models", path_and_name=path_and_name) #version will be appended to name; path_and_name is model(.pt)
            best_loss = val_loss
		
        ###TESTING
        test_loss, loss_metrics = single_test(args, model, test_dataloader, get_loss_func, optimizer, scheduler, logger, tmetrics) #change to single_val with DDP
        if dist.is_initialized():
            test_loss = torch.tensor(test_loss, dtype=torch.float, device=device)
            loss_metrics = torch.tensor(loss_metrics, dtype=torch.float, device=device)
            torch.distributed.all_reduce(test_loss, op=torch.distributed.ReduceOp.SUM)
            test_loss = (test_loss / world_size).item()
            torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
            loss_metrics = (loss_metrics / world_size).item()
        if args.log: 
            logger.log_metrics({'ALL_REDUCED_test_loss': test_loss}, epoch_idx) #zero rank only
            logger.log_metrics({'ALL_REDUCED_test_MAE': loss_metrics}, epoch_idx) #zero rank only
        #mae_reduced = tmetrics.compute() #Synced and reduced autometically!
        #logger.log_metrics({'ALL_REDUCED_test_mae_loss': mae_reduced.item()}, epoch_idx) #zero rank only
        tmetrics.reset()
	
        model.train()	
    init_end_event.record()

    if local_rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
	
    print(cf.on_yellow("Training is OVER..."))
    epoch_start, best_loss = load_state(model, optimizer, scheduler_groups, path_and_name, use_artifacts=False, logger=logger, name=args.name) 
    save_state(model, optimizer, scheduler_groups, epoch_idx, val_loss, path_and_name)
    if args.log: logger.log_artifacts(name=f"{args.name}_model_objects", dtype="pytorch_models", path_and_name=path_and_name) #version will be appended to name; path_and_name is model(.pt)
    #DONE: SAVE ONE MORE TIME before ending???
