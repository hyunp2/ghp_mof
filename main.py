import os, argparse
import pandas as pd
import numpy as np
from typing import *
import warnings

import torch
from torch_scatter import scatter
  
from train.train_utils import load_state
from train.dist_utils import get_local_rank, init_distributed, increase_l2_fetch_granularity, WandbLogger
from train.data_utils import DataModuleCrystal
from train.loss_utils import get_loss_func_crystal
from train.dist_utils import *
from models import cgcnn
from configs import BACKBONES, BACKBONE_KWARGS

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--ensemble_names', nargs="*", type=str, default=None)
    parser.add_argument('--model_filename', type=str, default=None, help="GAN Model")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpus', action='store_true')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--log', action='store_true') #only returns true when passed in bash
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--use-artifacts', action='store_true', help="download model artifacts for loading a model...") 
    parser.add_argument('--which_mode', type=str, help="which mode for script?", default="train", choices=["train","infer","explain"]) 

    # data
    parser.add_argument('--train_test_ratio', type=float, default=0.02)
    parser.add_argument('--train_val_ratio', type=float, default=0.03)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--warm_up_split', type=int, default=5)
    parser.add_argument('--batches', type=int, default=160)
    parser.add_argument('--test_samples', type=int, default=5) # -1 for all
    parser.add_argument('--test_steps', type=int, default=100)
    parser.add_argument('--sync_batch', action='store_true', help="sync batchnorm") #normalize energy???
    parser.add_argument('--data_norm', action='store_true') #normalize energy???
    parser.add_argument('--dataset', type=str, default="cifdata", choices=["cifdata"])
    parser.add_argument('--data_dir', type=str, default="/Scr/hyunpark/ArgonneGNN/ligand_data")
    parser.add_argument('--ase_save_dir', type=str, default="/Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/ase_run")
    # parser.add_argument('--data_dir_crystal', type=str, default="/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax")
    parser.add_argument('--data_dir_crystal', type=str, default="/Scr/hyunpark/ArgonneGNN/hMOF/cifs/")
    parser.add_argument('--task', type=str, default="homo")
    parser.add_argument('--pin_memory', type=bool, default=True) #causes CUDAMemory error;; asynchronously reported at some other API call
    parser.add_argument('--use_artifacts', action="store_true", help="use artifacts for resuming to train")
    parser.add_argument('--use_tensors', action="store_true") #for data, use DGL or PyG formats?
    parser.add_argument('--crystal', action="store_true") #for data, use DGL or PyG formats?
    parser.add_argument('--make_data', action="store_true", help="force making data") 
    parser.add_argument('--save_to_pickle', type=str, default=None, help="whether to save CIFDataset")
    parser.add_argument('--num_oversample', type=int, default=0, help="number of oversampling for minority") # -1 for all
    parser.add_argument('--custom_dataloader', default=None, help="custom dataloader obj")
    parser.add_argument('--truncate_above', type=float, default=None, help="property of Crystal data truncation cutoff...")

    # train
    parser.add_argument('--epoches', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128) #Per GPU batch size
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate','-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--distributed',  action="store_true")
    parser.add_argument('--low_memory',  action="store_true")
    parser.add_argument('--amp', action="store_true", help="floating 16 when turned on.")
    parser.add_argument('--loss_schedule', '-ls', type=str, choices=["manual", "lrannealing", "softadapt", "relobralo", "gradnorm"], help="how to adjust loss weights.")
    parser.add_argument('--with_force', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='adam', choices=["adam","lamb","sgd","torch_adam","torch_adamw","torch_sparse_adam"])
    parser.add_argument('--gradient_clip', type=float, default=None) 
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 
    parser.add_argument('--shard', action="store_true", help="fairscale ShardedDDP") #fairscale ShardedDDP?
    parser.add_argument(
        "--not_use_env",
        default=False,
        action="store_false",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )

    # inference
    parser.add_argument('--inference_df_name', type=str, default=None)

    # model
    parser.add_argument('--backbone', type=str, default='cgcnn', choices=["cgcnn"])
    parser.add_argument('--load_ckpt_path', type=str, default="models")
    parser.add_argument('--explain', type=bool, default=False, help="gradient hook for CAM...") #Only for Schnet.Physnet.Alignn WIP!
    parser.add_argument('--dropnan', action="store_true", help="drop nan smiles... useful for ligand model! during inference!")

    # explain options
    parser.add_argument('--n_components', type=int, default=2, help="Dimension reduction", choices=[2,3])
    parser.add_argument('--clust_num', type=int, default=8, help="cluster numbers")
    parser.add_argument('--clust_algo', type=str, default="kmeans", help="which algo for cluster", choices=["kmeans","dbscan","bgm"])
    parser.add_argument('--proj_algo', type=str, default="umap", help="which algo for reduction", choices=["umap","tsne","pca"])
    parser.add_argument('--color_scheme', type=str, default="pred", help="which value to color on scatter", choices=["real","pred","clust"])
    # parser.add_argument('--render_option', type=str, default="plotly", help="how to save")
    parser.add_argument('--infer_for', type=str, default="ligand", help="infer_for crystal or ligand", choices=["ligand","crystal"])
    parser.add_argument('--which_explanation', default="projection", help="what to highlight", choices=["projection", "embedding", "interactive", "decomposition"])
    parser.add_argument('--models_to_explain', default=["physnet", "schnet", "mpnn", "megnet"], help="list of models to explain", nargs="*")

    opt = parser.parse_args()

    return opt
    
def call_model(opt: argparse.ArgumentParser, mean: float, std: float, logger: WandbLogger, return_metadata=False):
    #Model 
    model = BACKBONES.get(opt.backbone, cgcnn.CrystalGraphConvNet) #Uninitialized class
    model_kwargs = BACKBONE_KWARGS.get(opt.backbone, None) #TorchMDNet not yet!

    model_kwargs.update({"explain": opt.explain})  
    
    if opt.backbone in ["cgcnn"]:
        model_kwargs.update({"mean":mean, "std":std})
        model = model(**model_kwargs) 
        radius_cutoff = model_kwargs.get("cutoff", 10.)
        max_num_neighbors = model_kwargs.get("max_num_neighbors", 32)
	
    if opt.gpu:
        model = model.to(torch.cuda.current_device())
    model.eval()
    torch.backends.cudnn.enabled=False

    path_and_name = os.path.join(opt.load_ckpt_path, "{}.pth".format(opt.name))

    load_state(model, optimizer=None, scheduler_groups=None, path_and_name=path_and_name, model_only=True, use_artifacts=False, logger=logger, name=None)
    if torch.__version__.startswith('2.0'): 
        model = torch.compile(model)
        print("PyTorch model has been compiled...")
    if not return_metadata:
        return model
    else:
        return model, radius_cutoff, max_num_neighbors

def call_loader(opt: argparse.ArgumentParser):
    #Distributed Sampler Loader

    if opt.dataset in ["cifdata"]:
        datamodule = DataModuleCrystal(opt=opt) #For jake's bias exists; for hMOF, bias is None...
	
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    mean = datamodule.mean
    std = datamodule.std

    return train_loader, val_loader, test_loader, mean, std

def run():
    """
    train() -> train_nvidia -> train_epoch
    
    This function must define a Normal Model, DistSampler etc.
    Then, INSIDE train_nvidia, DDP Model, DDP optimizer, set_epoch for DistSampler, GradScaler etc. (and all_gather etc) are fired up.
    Then inside train_epoch, Loss/Backprop is done
    """
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    opt = get_parser()
    
    if opt.log:
        logger = WandbLogger(name=None, entity="argonne_gnn", project='internship')
        # logger = WandbLogger(name=None, entity="hyunp2", project='ArgonneGNN')
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")
    else:
        logger = None
	
    print('Backbone {} With_force {}'.format(opt.backbone, opt.with_force))

    train_loader, val_loader, test_loader, mean, std = call_loader(opt)

    #Model 
    model = BACKBONES.get(opt.backbone, physnet.Physnet) #Uninitialized class
    model_kwargs = BACKBONE_KWARGS.get(opt.backbone, None) #TorchMDNet not yet!

    model_kwargs.update({"explain": opt.explain})  
    
    if opt.backbone in ["cgcnn"]:
        model_kwargs.update({"mean":mean, "std":std})
        model = model(**model_kwargs) 
    print("mean", mean, "std", std)

    if opt.gpu:
        model = model.to(torch.cuda.current_device())
    
    #Dist training
    if is_distributed:         
        nproc_per_node = torch.cuda.device_count()
        affinity = set_affinity(local_rank, nproc_per_node)
    increase_l2_fetch_granularity()

    if opt.crystal and opt.dataset in ["cifdata"]:
        train_crystal(model=model,
          train_dataloader=train_loader,
          val_dataloader=val_loader,
          test_dataloader=test_loader,
          logger=logger,
          get_loss_func=get_loss_func_crystal,
          args=opt)

def explain():
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    opt = get_parser()
    opt.explain = True #Force turn-on explainer
	
    assert opt.log, "Explain mode must enable W&B logging..."
    logger = WandbLogger(name=None, entity="argonne_gnn", project='internship')
    # logger = WandbLogger(name=None, entity="hyunp2", project='ArgonneGNN')
    os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
    os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")

    train_loader, val_loader, test_loader, mean, std = call_loader(opt)

    device = torch.device("cpu")	
    if opt.gpu:
        device = torch.cuda.current_device()

    if opt.dataset in ["cifdata"]:
        infer_for_method = infer_for_crystal

    if opt.which_explanation in ["embedding"]:
        """CAM based explainability: DONE for ligand, Not yet for MOF"""
        if not opt.dataset in ["cifdata", "gandata", "cifdata_original"]: #, "Visualization for EMBEDDING method for CRYSTALS are not supported yet..."
            """BELOW: Only for ligands"""
            model, radius_cutoff, max_num_neighbors = call_model(opt, mean, std, logger, return_metadata=True)
            if not opt.dropnan:
                print("Embedding explanation requires non-Nan SMILES, reverting to dropnan...")
                opt.dropnan = True
            else:
                pass
            if smiles_list[0] == "None":
                print("Smiles list is not provided, reverting to using test set!")
                df = pd.DataFrame()
                smiles_list, *_ = preprocess(infer_for_method, df, test_loader, model, return_vecs=True) #contains nans
                smiles_list = smiles_list[:100].tolist() #Due to sheer number of smiles list...
            else:
                print("Smiles list is provided!")
            logger.log_explainer(model=model, smiles_list=smiles_list, device=device, radius_cutoff=radius_cutoff, max_num_neighbors=max_num_neighbors)
        else:
            """BELOW: Only for MOFs"""
            model, radius_cutoff, max_num_neighbors = call_model(opt, mean, std, logger, return_metadata=True)
            from plotlyMOF import viz_mof_cif_v2
            # cifsos.listdir(opt.data_dir_crystal)
            fig = viz_mof_cif_v2(os.path.join(opt.data_dir_crystal, "DB12-ODODIW_clean.cif"))
            path_html = "plotly_visualization_output.html" #overwrite is ok...
            fig.write_html(path_html, auto_play = False)
            logger.log_html(path_html) #For each model column, get multiple index rows of color schemes
            

def infer_for_crystal(opt, df, dataloader, model, return_vecs=False):
    if return_vecs: final_conv_acts_list=[]
    # data_names_all, energies_all, y_all = [], [], []
    df_list = []
    print(dataloader)
    for one_data_batch in dataloader:
        data_batch = one_data_batch[0] #Get DATA instance
        data_names = one_data_batch[1] #Get CIF names
        print(data_batch)
        # print(data_names)
        data_batch = data_batch.to(torch.cuda.current_device())
        # print(data_batch)
        # x, edge_attr, edge_idx, edge_weight, cif_id, batch = to_cuda((data_batch.x, data_batch.edge_attr, data_batch.edge_index, data_batch.edge_weight, data_batch.cif_id, data_batch.batch))

        if opt.ensemble_names is not None:
            e, s = model(data_batch.x, data_batch.edge_attr, data_batch.edge_index, data_batch.edge_weight, data_batch.cif_id, data_batch.batch)
            energies = e
            stds = s
        else:
            e = model(data_batch.x, data_batch.edge_attr, data_batch.edge_index, data_batch.edge_weight, data_batch.cif_id, data_batch.batch)
            energies = e
        y = data_batch.y
        print(data_names, energies, y)
        # print(np.array(data_names).reshape(-1,1).shape, np.array(data_names).reshape(-1,1))
        if return_vecs: final_conv_acts_list.append(scatter(src=model.final_conv_acts, index=data_batch.batch, dim=0, reduce="mean").detach().cpu().numpy())
		
        # data_names_all.extend(data_names) #[List[str], List[str] ...]
        # energies_all.append(energies) #[tensor, tensor, tensor]
        # y_all.append(y) #[tensor, tensor, tensor]

        if opt.ensemble_names is not None:
            df_list = df_list + [pd.DataFrame(data=np.concatenate([np.array(data_names).reshape(-1,1), energies.detach().cpu().numpy().reshape(-1,1), 
								   stds.detach().cpu().numpy().reshape(-1,1), y.detach().cpu().numpy().reshape(-1,1)], axis=1), 
					      			   columns=["name","pred","std","real"])]
        else:
            df_list = df_list + [pd.DataFrame(data=np.concatenate([np.array(data_names).reshape(-1,1), energies.detach().cpu().numpy().reshape(-1,1), 
								   y.detach().cpu().numpy().reshape(-1,1)], axis=1), columns=["name","pred","real"])]
        
        # df = pd.concat([df, pd.DataFrame(data=np.concatenate([np.array(data_names).reshape(-1,1), energies.detach().cpu().numpy().reshape(-1,1), y.detach().cpu().numpy().reshape(-1,1)], axis=1), columns=["name","pred","real"])], axis=0, ignore_index=True)
    # print(df)
    # print(energies, y)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    
    select_nans = np.where(df.name.values == "nan")[0] #only rows
    select_nonans = np.where(df.name.values != "nan")[0] #only rows
    df = df.drop(index=select_nans.tolist()).reset_index().drop(columns="index") if opt.dropnan else df
    if return_vecs: 
        final_conv_acts_list = np.concatenate(final_conv_acts_list, axis=0)
        final_conv_acts_list = final_conv_acts_list[select_nonans] if opt.dropnan else final_conv_acts_list
        assert df.shape[0] == final_conv_acts_list.shape[0], "Dataframe and Latents must match in sample numbers!"
    # print(df)
    if not return_vecs:
        return df
    else:
        return df, final_conv_acts_list

def infer(opt=None):
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    opt = get_parser() if opt is None else opt
    
    if opt.explain:
        assert opt.log, "Explain mode must enable W&B logging..."
        logger = WandbLogger(name=None, entity="argonne_gnn", project='internship')
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")
    else:
        if opt.log:
            logger = WandbLogger(name=None, entity="argonne_gnn", project='internship')
            os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
            os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
            os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")        
        else:
            logger = None
            
    if not opt.custom_dataloader:
        train_loader, val_loader, test_loader, mean, std = call_loader(opt)  
        print("mean", mean, "std", std)
    else: 
        #Use custom dataset and loader if option is off
        train_loader, test_loader = opt.custom_dataloader, None
        mean, std = None, None

    if opt.ensemble_names is not None:
        models = []
        for name in opt.ensemble_names:
            opt.name = name
            model = call_model(opt, mean, std, logger) 
            models.append(model)
        model = lambda *inp : (torch.cat([models[0](*inp), models[1](*inp), models[2](*inp)], dim=-1).mean(dim=-1), 
			       torch.cat([models[0](*inp), models[1](*inp), models[2](*inp)], dim=-1).std(dim=-1))
    else:
        model = call_model(opt, mean, std, logger) 

    df = pd.DataFrame()
    dataloader = test_loader if opt.train_frac != 1.0 else train_loader

    if opt.dataset in ["cifdata"]:
        df = infer_for_crystal(opt, df, dataloader, model)

    torch.backends.cudnn.enabled=True
	
    pathlib.Path(os.path.join(os.getcwd(), "publication_figures")).mkdir(exist_ok=True)
    if opt.inference_df_name is None:
        if opt.ensemble_names is not None: 
            opt.name = "ensemble"
        df.to_csv(os.path.join(os.getcwd(), "publication_figures", f"{opt.name}_property_prediction.csv"))
    else:
        df.to_csv(os.path.join(os.getcwd(), "publication_figures", f"{opt.inference_df_name}"))

    print(f"Property is predicted and saved as {opt.name}_property_prediction.csv ...")

if __name__ == "__main__":
    warnings.simplefilter("ignore")	
	
    opt = get_parser()
    if opt.which_mode in ["train"]:
        run()
    elif opt.which_mode in ["explain"]:
        explain()
    elif opt.which_mode in ["infer"]:
        infer(opt=opt)
    # python -m main --which_mode infer --backbone cgcnn --load_ckpt_path models --name cgcnn_pub_hmof_0.1 --gpu --data_dir_crystal /Scr/hyunpark/ArgonneGNN/hMOF/cifs 
