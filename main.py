import os, abc, sys
import pdb
import glob
import copy
import wandb
import h5py
import random
import string
import argparse
import pandas as pd
import numpy as np
import scipy.stats
import importlib
from scipy.special import softmax
from PIL import Image
from typing import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib
import functools
import ase
import shutil
import pickle
import io
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdDepictor
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from skimage.io import imread
from cairosvg import svg2png, svg2ps
import collections
import re
import ray

from fast_ml.model_development import train_valid_test_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch_geometric
from torch_geometric.data import Batch
from torch_scatter import scatter
from torch_cluster import radius_graph

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from models import cgcnn

import captum

try:
    importlib.import_module("apex.optimizers")
    from apex.optimizers import FusedAdam, FusedLAMB
except Exception as e:
    pass
  
from train.training_functions_pub import train as train_molecule
from crystals.training_functions_original_cgcnn_pub import train as train_cgcnn_original
from train.training_functions_pub import load_state, save_state, single_test
from train.dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, WandbLogger
from train.dataloader import DataModuleEdge, DataModuleOthers
from crystals.dataloader import DataModuleCrystal
from crystals.cgcnn_original import DataModuleCrystal as DataModuleCrystal_original
from train.loss_utils_pub import get_loss_func, get_loss_func_crystal
from train.gpu_affinity import *
from data.smiles_parser import from_smiles
from train.ase_pub import AseInterface, CustomCalculator
from train.plotly_pub import *

def get_parser():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--name', type=str, default=''.join(random.choice(string.ascii_lowercase) for i in range(10)))
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--model_filename', type=str, default=None, help="GAN Model")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpus', action='store_true')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--log', action='store_true') #only returns true when passed in bash
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--use-artifacts', action='store_true', help="download model artifacts for loading a model...") 
    parser.add_argument('--which_mode', type=str, help="which mode for script?", default="train", choices=["train","infer","tuning","explain","md","relax","ase2mda"]) 
    parser.add_argument('--zeo_exec', type=str, help="Zeo++ executable", default="/Scr/hyunpark/zeo++-0.3/network") 
    parser.add_argument('--zeo_path', type=str, help="Zeo++ file path", default="/Scr/hyunpark/ArgonneGNN/") 

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
    parser.add_argument('--dataset', type=str, default="qm9edge", choices=["qm9", "md17", "ani1", "ani1x", "qm9edge", "s66x8", "moleculenet","cifdata","gandata","cifdata_original","redox"])
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
    parser.add_argument('--num_oversample', type=int, default=1000, help="number of oversampling for minority") # -1 for all
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
    parser.add_argument('--backbone', type=str, default='physnet', choices=["schnet","physnet","torchmdnet","alignn","dimenet","dimenetpp","cgcnn","cgcnn_original","calignn","cphysnet","cschnet","ctorchmdnet","megnet","mpnn", "graph_transformer"])
    parser.add_argument('--load_ckpt_path', type=str, default="/Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub/")
    parser.add_argument('--explain', type=bool, default=False, help="gradient hook for CAM...") #Only for Schnet.Physnet.Alignn WIP!
    parser.add_argument('--dropnan', action="store_true", help="drop nan smiles... useful for ligand model! during inference!")

    # hyperparameter optim
    parser.add_argument('--resume_hp', action="store_true", help="resume hp search if discontinued...")
    parser.add_argument('--hp_savefile', type=str, default="results.csv", help="resume hp search from this file...")
    parser.add_argument('--hyper_inputs', type=str, default="hyper_inputs.yaml", help="tuning inputs..")
    parser.add_argument('--save_hyper_dir', type=str, default=os.getcwd(), help="where to save hyper results.csv")

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

    #md options
    parser.add_argument('--heat_bath', type=float, default=None, help="Temperature in Kelvin... None is NVE; float is NVT")
    parser.add_argument('--timestep', type=float, default=0.1, help="in fs unit of intergration time")
    parser.add_argument('--md_steps', type=int, default=200, help="how many frames (aka md integration steps)")
    parser.add_argument('--md_name', type=str, default="test", help="saved md traj name")
    parser.add_argument('--mda_format', type=str, default="dcd", help="save to a different traj format and return MDA Universe...")
    parser.add_argument('--smiles_list', default=["CC(=O)O", "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O", "Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O"], nargs="*", help="list of SMILES")

    opt = parser.parse_args()

    return opt
    
def call_model(opt: argparse.ArgumentParser, mean: float, std: float, logger: WandbLogger, return_metadata=False):
    #Model 
    model = BACKBONES.get(opt.backbone, physnet.Physnet) #Uninitialized class
    model_kwargs = BACKBONE_KWARGS.get(opt.backbone, None) #TorchMDNet not yet!

    if opt.backbone not in ["calignn", "alignn"]: model_kwargs.update({"explain": opt.explain})  
    else: model_kwargs.explain = opt.explain #only for alignn net (due to pydantic)
    
    if opt.backbone in ["schnet", "physnet", "dimenet", "dimenetpp","cschnet","cphysnet","cgcnn","cgcnn_original","megnet","mpnn"]:
        model_kwargs.update({"mean":mean, "std":std})
        if "transformer" in opt.name and "mpnn" in opt.name: 
            model_kwargs.update({"nlp": "transformer"})
        elif not "transformer" in opt.name and "mpnn" in opt.name: 
            model_kwargs.update({"nlp": "gru"})
        else: 
            pass
        model = model(**model_kwargs) 
        radius_cutoff = model_kwargs.get("cutoff", 10.)
        max_num_neighbors = model_kwargs.get("max_num_neighbors", 32)
    elif opt.backbone in ["graph_transformer"]:
        preencoder = BACKBONES.get("cphysnet")
        preencoder_config = BACKBONE_KWARGS.get("graph_transformer")
        preencoder = preencoder(**preencoder_config) #physnet
        model = BACKBONES.get(opt.backbone)
        model = model(opt, preencoder)
    elif opt.backbone in ["alignn","calignn"]:
        model_kwargs.mean = mean
        model_kwargs.std = std
        radius_cutoff = model_kwargs.cutoff
        model = model(model_kwargs) #Accounting for alignn net
    elif opt.backbone in ["torchmdnet","ctorchmdnet"]:
        model_kwargs.update({"mean":mean, "std":std})
        radius_cutoff = model_kwargs.get("cutoff_upper", 5.)
        max_num_neighbors = model_kwargs.get("max_num_neighbors", 64)
        model = torchmdnet.create_model(model_kwargs, mean=mean, std=std) if opt.backbone=="torchmdnet" else ctorchmdnet.create_model(model_kwargs)

    if opt.gpu:
        model = model.to(torch.cuda.current_device())
    model.eval()
    torch.backends.cudnn.enabled=False

    path_and_name = os.path.join(opt.load_ckpt_path, "{}.pth".format(opt.name))

    load_state(model, optimizer=None, scheduler_groups=None, path_and_name=path_and_name, model_only=True, use_artifacts=False, logger=logger, name=None)
    if not return_metadata:
        return model
    else:
        return model, radius_cutoff, max_num_neighbors

def call_loader(opt: argparse.ArgumentParser):
    #Distributed Sampler Loader
    if opt.dataset in ["qm9edge"]:
        datamodule = DataModuleEdge(opt=opt)
    elif opt.dataset in ["qm9", "md17", "ani1", "ani1x", "moleculenet","s66x8","redox"]:
        datamodule = DataModuleOthers(hparams=opt)
    elif opt.dataset in ["cifdata","gandata"]:
        datamodule = DataModuleCrystal(opt=opt) #For jake's bias exists; for hMOF, bias is None...
    elif opt.dataset in ["cifdata_original"]:
        datamodule = DataModuleCrystal_original(opt=opt)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    if not opt.dataset in ["gandata"]:
        mean = datamodule.mean
        std = datamodule.std
    else:
        mean = None
        std = None
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

    if opt.backbone not in ["calignn", "alignn"]: model_kwargs.update({"explain": opt.explain})  
    else: model_kwargs.explain = opt.explain #only for alignn net (due to pydantic)
    
    if opt.backbone in ["schnet", "physnet", "dimenet", "dimenetpp","cschnet","cphysnet","cgcnn","cgcnn_original","megnet","mpnn"]:
        model_kwargs.update({"mean":mean, "std":std})
        model = model(**model_kwargs) 
    elif opt.backbone in ["graph_transformer"]:
        preencoder = BACKBONES.get("cphysnet")
        preencoder_config = BACKBONE_KWARGS.get("graph_transformer")
        preencoder = preencoder(**preencoder_config) #physnet
        model = BACKBONES.get("graph_transformer")
        model = model(opt, preencoder)
    elif opt.backbone in ["alignn","calignn"]:
        model_kwargs.mean = mean
        model_kwargs.std = std
        model = model(model_kwargs) #Accounting for alignn net
    elif opt.backbone in ["torchmdnet","ctorchmdnet"]:
        model_kwargs.update({"mean":mean, "std":std})
#         print("loader", mean, std)
        model = torchmdnet.create_model(model_kwargs, mean=mean, std=std) if opt.backbone=="torchmdnet" else ctorchmdnet.create_model(model_kwargs)
    print("mean", mean, "std", std)

    if opt.gpu:
        model = model.to(torch.cuda.current_device())
    
    #Dist training
    if is_distributed:         
        nproc_per_node = torch.cuda.device_count()
        affinity = set_affinity(local_rank, nproc_per_node)
    increase_l2_fetch_granularity()

    if not opt.crystal:
        train_molecule(model=model,
          train_dataloader=train_loader,
          val_dataloader=val_loader,
          test_dataloader=test_loader,
          logger=logger,
          get_loss_func=get_loss_func,
          args=opt)
    elif opt.crystal and opt.dataset in ["cifdata", "gandata"]:
        train_crystal(model=model,
          train_dataloader=train_loader,
          val_dataloader=val_loader,
          test_dataloader=test_loader,
          logger=logger,
          get_loss_func=get_loss_func_crystal,
          args=opt)
    elif opt.crystal and opt.dataset in ["cifdata_original"]:
        train_cgcnn_original(model=model,
          train_dataloader=train_loader,
          val_dataloader=val_loader,
          test_dataloader=test_loader,
          logger=logger,
          get_loss_func=get_loss_func_crystal,
          args=opt)

# Ligand:    python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main_pub --log --backbone physnet --gpu --name physnet_pub_qm9edgelumo --epoches 1000 --batch_size 512 --optimizer torch_adam --data_dir /Scr/hyunpark/ArgonneGNN/argonne_gnn/data --use_tensors --load_ckpt_path /Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub --dataset qm9edge --task lumo --which_mode train
# Crystal:   python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module main_pub --log --backbone calignn --gpu --name calignn_pub --epoches 1000 --batch_size 4 --optimizer torch_adam  --data_dir_crystal /Scr/hyunpark/ArgonneGNN/DATA/diverse_metals/cifs/ --use_tensors --load_ckpt_path /Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub --dataset cifdata --crystal --which_mode train --smiles_list None

def hyper():
    from train import hyper_opt_tuning_pub
#     python -m main_pub --log --backbone torchmdnet --gpu --name torchmdnet_pub --epoches 1000 --batch_size 512 --optimizer torch_adam --data_dir /Scr/hyunpark/ArgonneGNN/argonne_gnn/data --use_tensors --load_ckpt_path /Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub --dataset md17 --task ethanol --save_hyper_dir [your_directory] --which_mode tuning

def mols_to_wimgs_func(i, mol):
    # mol = AllChem.AddHs(mol)
    # Chem.AllChem.Compute2DCoords(mol)
    # Chem.AllChem.GenerateDepictionMatching2DStructure(mol, mol)
    # img = Chem.Draw.MolToImage(mol, size=(300, 300))
    # from skimage.io import imread
    # img.save("tmp.png")
    # img = imread('tmp.png')
    # os.remove('tmp.png')
    
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg2png(bytestring=svg, write_to=f'tmp{i}.png', dpi=100)
    img = imread(f'tmp{i}.png')
    os.remove(f'tmp{i}.png')
    return wandb.Image(img)

def most_commom_substructure(opt: argparse.ArgumentParser, summary: collections.namedtuple, logger: WandbLogger):
    """
    Maybe do something with Fragments as well??
    https://github.com/igashov/DiffLinker/blob/main/data/geom/generate_geom_multifrag.py
    BRICS vs MMPA
    """
    assert not opt.dataset in ["cifdata","gandata","cifdata_original"], "MCS cannot be acquired for crystals..."
    clust_num = opt.clust_num
    list_of_idx = list(map(lambda inp: np.where(summary.clustered == inp)[0], np.arange(clust_num) ))
    smiles_to_mols = np.array(list(map(lambda inp: Chem.MolFromSmiles(inp), summary.names ))) #to ndarray for indexing
    mcs_list = list(map(lambda inp: rdFMCS.FindMCS(smiles_to_mols[np.array(inp)].tolist()).smartsString, list_of_idx )) #list of smartsstring...
    mcs_to_mols = list(map(lambda inp: Chem.MolFromSmarts(inp), mcs_list ))
    mols_to_wimgs = list(map(lambda i, inp: mols_to_wimgs_func(i, inp), np.arange(len(mcs_to_mols)), mcs_to_mols))
    
    centroids = summary.centroids
    dicts = {"SMARTS": mcs_list, "Images": mols_to_wimgs}
    df = pd.DataFrame.from_dict(dicts)
    return df

def explain(smiles_list: List[str] = ["CC(=O)O", "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O", "Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O"]):
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

    if opt.dataset not in ["cifdata","gandata","cifdata_original"]:
        infer_for_method = infer_for_ligand 
    elif opt.dataset in ["cifdata","gandata","cifdata_original"]:
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
            
    elif opt.which_explanation in ["projection"]:
        """Kepler based interactive graph"""
        ##WIP: Maybe deprecate?
        model = call_model(opt, mean, std, logger)
        from kmapper import KeplerMapper, Cover
        from kmapper.visuals import colorscale_from_matplotlib_cmap
        from sklearn.manifold import TSNE
        from sklearn.cluster import DBSCAN, KMeans
        from umap import UMAP
        df = pd.DataFrame()
        df, X = infer_for_method(df, test_loader, model, return_vecs=True)     #df and numpy
        mapper = KeplerMapper()
        clusters = KMeans().fit_transform(X)
        X_projected = mapper.fit_transform(X, projection=[UMAP(n_components=2)])
        ###SPEED-UP: 1. less Cover cubes (=computation);; 2. less Cover overlap (=edges)
        graph = mapper.map(X_projected, X, clusterer=KMeans(), cover=Cover(10, 0.5)) #REDUCED dimension: X_projected accelerates graph generation
        f = open("mapper_graph_output.pickle","wb")
        pickle.dump(graph, f)
        # print(X_projected, X)
        path_html = "mapper_visualization_output.html"
        print(df.pred.values.shape, df.real.values.shape, clusters.shape, X.shape)
        html = mapper.visualize(graph, color_values=np.stack([df.pred.values.reshape(-1,), df.real.values.reshape(-1,), clusters.argmax(axis=-1).reshape(-1,), clusters.argmin(axis=-1).reshape(-1,)], axis=1), color_function_name=['pred', 'real', 'cluster_max','cluster_min'], path_html=path_html, colorscale = colorscale_from_matplotlib_cmap(plt.cm.jet, nbins=255), custom_tooltips=df.name.values.reshape(-1,), include_searchbar=True) 
        logger.log_html(path_html)
        
    elif opt.which_explanation in ["interactive"]:
        """Plotly based 2D 3D dimension reduction"""
        ##WIP: change how to log results!
        html_dict = collections.defaultdict(list)
        rogi_dict = collections.defaultdict(list)
        color_scheme_list = ["real","clust"]
        mcs_df_list = []
        for model_name in opt.models_to_explain:
            opt.name = model_name
            backbone = re.split('_pub_|_pub', model_name)[0] if "pub" in model_name else model_name.split("_")[0] #WIP: must change at some point! split no either patterns
            opt.backbone = backbone
            model = call_model(opt, mean, std, logger)
            df = pd.DataFrame()
            # df, X = infer_for_ligand(df, test_loader, model, return_vecs=True)
            # print(df, X)
            names, lat_vecs, preds, reals = preprocess(infer_for_method, df, test_loader, model, return_vecs=True)
            for color_scheme in color_scheme_list:
                fig, summary = update_graph(names, lat_vecs, preds, reals, opt.clust_num, opt.clust_algo, opt.proj_algo, color_scheme, opt.n_components, model_name)
                path_html = "plotly_visualization_output.html" #overwrite is ok...
                fig.write_html(path_html, auto_play = False)
                html_dict[model_name].append(wandb.Html(open(path_html))) #For each model column, get multiple index rows of color schemes
                if color_scheme == "clust" and not opt.dataset in ["cifdata","gandata","cifdata_original"]: mcs_df_list.append(most_commom_substructure(opt, summary, logger)) #append a DF
                else: pass
            ri_means, ri_stds, ri_metrics = roughness_index(names, lat_vecs, preds, reals, opt.clust_num, opt.clust_algo, opt.proj_algo, color_scheme, opt.n_components, model_name)
            rogi_dict[model_name].extend(ri_means.tolist())
            rogi_dict[model_name].extend(ri_stds.tolist())
        rogi_dict["Rogi Metric and Statistics"].extend([ri_met + "_mean" for ri_met in ri_metrics])
        rogi_dict["Rogi Metric and Statistics"].extend([ri_met + "_std" for ri_met in ri_metrics])
        dataframe = pd.DataFrame.from_dict(html_dict) #ROWS: prop&clust&ROGImean&ROGIstd
        dataframe["color_scheme"] = color_scheme_list #+ ["ROGI_mean", "ROGI_std"]
        dataframe = dataframe.set_index("color_scheme")
        logger.log_dataframe(dataframe, "dimension reduction")
        dataframe = pd.DataFrame.from_dict(rogi_dict) #ROWS: prop&clust&ROGImean&ROGIstd
        # dataframe["ROGIs"] = ["ROGI_mean", "ROGI_std"]
        # dataframe = dataframe.set_index("ROGIs")
        logger.log_dataframe(dataframe, "ROGI")
        if not opt.dataset in ["cifdata","gandata","cifdata_original"]:
            mcs_df = pd.concat(mcs_df_list, axis=0).reset_index().drop(columns="index")
            logger.log_dataframe(mcs_df, "mcs")
        else:
            pass

    elif opt.which_explanation in ["decomposition"]:
        """CVXPY based decomposing atom contributions"""
        #WIP!
        import cvxpy as cp
        from train.explainer import plot_explanations
        if opt.dropnan: 
            print("drop nan must be disabled... changing to non dropnan")
            opt.dropnan = False
        model, radius_cutoff, max_num_neighbors = call_model(opt, mean, std, logger, return_metadata=True)
        df = pd.DataFrame()
        smiles_list, *_ = preprocess(infer_for_method, df, test_loader, model, return_vecs=True) #contains nans
        assert len(smiles_list) == len(test_loader.dataset)
        #WIP: match the smiles!
        for i, data_batch in enumerate(test_loader):
            temp_fwd = model(z=data_batch["z"].to(torch.cuda.current_device()), pos=data_batch["pos"].to(torch.cuda.current_device()), batch=data_batch["batch"].to(torch.cuda.current_device()))[0] #forward flush! and get Energy
            temp_fwd.backward(gradient=torch.ones_like(temp_fwd)) #forward flush!
            batch_unique, batch_counts = data_batch["batch"].to(torch.cuda.current_device()).unique(return_counts=True)
            batch_counts_cumsum = batch_counts.cumsum(dim=0)
            edge_index = radius_graph(
                data_batch["pos"].to(torch.cuda.current_device()),
                r=radius_cutoff,
                batch=data_batch["batch"].to(torch.cuda.current_device()),
                loop=True,
                max_num_neighbors=max_num_neighbors,
            )
            smiles_chunk = smiles_list[ batch_unique.detach().cpu().numpy() + i*len(data_batch["batch"])]
            print(len(smiles_chunk), len(data_batch["batch"]))
            metadata_list = list(map(lambda batch_index, smiles, end_idx: plot_explanations(model, data_batch["batch"].to(torch.cuda.current_device()), batch_index, smiles, end_idx, edge_index, return_metadata=True),  batch_unique, smiles_list[ batch_unique.detach().cpu().numpy() + i*len(batch_unique) ], batch_counts_cumsum )) 
        # df = pd.DataFrame.from_records(metadata_list) #saliency and ugradcam columns
        print(metadata_list)
    torch.backends.cudnn.enabled=True

#     python -m main_pub --log --backbone torchmdnet --gpu --name torchmdnet_pub --epoches 1000 --batch_size 512 --optimizer torch_adam --data_dir /Scr/hyunpark/ArgonneGNN/argonne_gnn/data --use_tensors --load_ckpt_path /Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub --dataset md17 --task ethanol --which_mode explain --which_explanation projection
# python -m main_pub --log  --gpu --batch_size 64 --data_dir /Scr/hyunpark/ArgonneGNN/argonne_gnn/data --use_tensors --load_ckpt_path /Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub --dataset cifdata --which_mode explain --which_explanation interactive --models_to_explain cphysnet_pub_oversample cschnet_pub_oversample cgcnn_pub_oversample

def infer_for_crystal(opt, df, dataloader, model, return_vecs=False):
    if return_vecs: final_conv_acts_list=[]

    for one_data_batch in dataloader:
        data_batch = one_data_batch[0] #Get DATA instance
        data_names = one_data_batch[1] #Get CIF names
        # print(data_names)
        data_batch = data_batch.to(torch.cuda.current_device())
        # print(data_batch)
        # x, edge_attr, edge_idx, edge_weight, cif_id, batch = to_cuda((data_batch.x, data_batch.edge_attr, data_batch.edge_index, data_batch.edge_weight, data_batch.cif_id, data_batch.batch))

        e = model(data_batch.x, data_batch.edge_attr, data_batch.edge_index, data_batch.edge_weight, data_batch.cif_id, data_batch.batch)
        energies = e
        y = data_batch.y
        # print(np.array(data_names).reshape(-1,1).shape, np.array(data_names).reshape(-1,1))
        if return_vecs: final_conv_acts_list.append(scatter(src=model.final_conv_acts, index=data_batch.batch, dim=0, reduce="mean").detach().cpu().numpy())

        df = pd.concat([df, pd.DataFrame(data=np.concatenate([np.array(data_names).reshape(-1,1), energies.detach().cpu().numpy().reshape(-1,1), y.detach().cpu().numpy().reshape(-1,1)], axis=1), columns=["name","pred","real"])], axis=0, ignore_index=True)
    # print(df)
    
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

def infer_for_ligand(opt, df, dataloader, model, return_vecs=False):
    if return_vecs: final_conv_acts_list=[]
    for data_batch in dataloader:
        batch = data_batch["batch"].to(torch.cuda.current_device())
        z = data_batch["z"].to(torch.cuda.current_device())
        pos = data_batch["pos"].to(torch.cuda.current_device())
        # print(batch.shape, z.shape, pos.shape)
        valid_batch = []
        # valid_idx = []
        data_names = []

        for idx, bat in enumerate(batch.unique()):
            try:
                one_smiles = Chem.MolToSmiles(dict2rdkitmol({"z": z[bat == batch], "pos": pos[bat == batch]}))
                valid_batch.append(bat)
                print(one_smiles)
                # valid_idx.append(idx)
                data_names.append(one_smiles)
            except Exception as e:
                print("No smiles")
                # one_smiles = Chem.MolToSmiles(dict2rdkitmol({"z": z[bat == batch], "pos": pos[bat == batch]}))
                valid_batch.append(bat)
                # valid_idx.append(idx)
                data_names.append("nan")
        data_names = np.array(data_names).reshape(-1,1)
        valid_batch_location = torch.isin(batch, torch.LongTensor(valid_batch).to(batch)) #Filtered out batch (num_atoms)
        valid_idx_location = torch.LongTensor(valid_batch).to(batch) #Filtered out batch (num_mols)
        # print(valid_idx_location)
        batch = batch[valid_batch_location]
        _, unique_counts = batch.unique(return_counts=True)
        batch = torch.repeat_interleave(torch.arange(valid_idx_location.size(0)).to(batch), unique_counts) #IMPORTANT!: CORRECT way of getting reduced batches!
        # print(batch.unique().shape)
        z = z[valid_batch_location] #choose valid atoms (hence, molecules) (num_atoms)
        pos = pos[valid_batch_location] #choose valid atoms (hence, molecules) (num_atoms)
        # print(batch.unique().shape, z.shape, pos.shape)

        y = data_batch["E"].to(torch.cuda.current_device())[valid_idx_location] # (num_mols)
        e_and_f = model(z=z, pos=pos, batch=batch)
        energies = e_and_f[0]
        forces = e_and_f[1]
        if return_vecs: final_conv_acts_list.append(scatter(src=model.final_conv_acts, index=batch, dim=0, reduce="sum").detach().cpu().numpy())
        # print(final_conv_acts_list)
        # print(energies, data_names.shape, len(energies), len(y))
        df = pd.concat([df, pd.DataFrame(data=np.concatenate([data_names, energies.detach().cpu().numpy().reshape(-1,1), y.detach().cpu().numpy().reshape(-1,1)], axis=1), columns=["name","pred","real"])], axis=0, ignore_index=True)

    select_nans = np.where(df.name.values == "nan")[0] #only rows
    select_nonans = np.where(df.name.values != "nan")[0] #only rows
    df = df.drop(index=select_nans.tolist()).reset_index().drop(columns="index") if opt.dropnan else df
    if return_vecs: 
        final_conv_acts_list = np.concatenate(final_conv_acts_list, axis=0)
        final_conv_acts_list = final_conv_acts_list[select_nonans] if opt.dropnan else final_conv_acts_list
        assert df.shape[0] == final_conv_acts_list.shape[0], "Dataframe and Latents must match in sample numbers!"
    if not return_vecs:
        return df
    else:
        return df, final_conv_acts_list

def infer(smiles_list: List[str] =["CC(=O)O", "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O", "Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O"], opt=None):
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

    model = call_model(opt, mean, std, logger)

    if not smiles_list[0] == "None":
        assert isinstance(smiles_list, (tuple, list)) and isinstance(smiles_list[0], str), "SMILES strings are not provides!"
        data_batch = list(map(from_smiles, smiles_list)) #torch_geometric list of Data instance
        data_batch = Batch.from_data_list(data_batch) #Batch instance
        if opt.gpu:
            data_batch = data_batch.to(torch.cuda.current_device())
        e_and_f = model(z=data_batch.z, pos=data_batch.pos, batch=data_batch.batch) #forward flush! and get Energy
        energies = e_and_f[0]
        forces = e_and_f[1]
        df = pd.DataFrame(data=energies.detach().cpu().numpy().reshape(-1,1), columns=["property"])

    else:
        df = pd.DataFrame()
        ###DONE: WIP: temporary for test_loader:
        # dataset = torch.utils.data.ConcatDataset([test_loader.dataset, val_loader.dataset])
        # loader_attr = datamodule.dataloader_kwargs
        # dataloader = torch_geometric.loader.DataLoader(dataset, **loader_attr)
        dataloader = test_loader if opt.train_frac != 1.0 else train_loader
        
        if opt.dataset in ["cifdata", "gandata", "cifdata_original"]:
            df = infer_for_crystal(opt, df, dataloader, model)
        else:
            df = infer_for_ligand(opt, df, dataloader, model)
    torch.backends.cudnn.enabled=True

    if opt.inference_df_name is None:
        df.to_csv(os.path.join(os.getcwd(), "publication_figures", f"{opt.name}_property_prediction.csv"))
    else:
        df.to_csv(os.path.join(os.getcwd(), "publication_figures", f"{opt.inference_df_name}"))

    print(f"Property is predicted and saved as {opt.name}_property_prediction.csv ...")

    from publication_figures.plot import log_replot_to_wandb
    if opt.log: 
        log_replot_to_wandb(opt, logger) #Log all CSVs to WandB
        print(f"Logging all csv files to WandB...")
    else:
        print(f"Skipping WandB...")


# Ligand:      python -m main_pub --log --backbone torchmdnet --gpu --name torchmdnet_pub --epoches 1000 --batch_size 512 --optimizer torch_adam --data_dir /Scr/hyunpark/ArgonneGNN/argonne_gnn/data --use_tensors --load_ckpt_path /Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub --dataset md17 --task ethanol --which_mode infer
# Crystal:     python -m main_pub --log --backbone cphysnet --gpu --name cphysnet_pub --epoches 1000 --batch_size 64 --smiles_list None --optimizer torch_adam  --data_dir_crystal /Scr/hyunpark/ArgonneGNN/DATA/diverse_metals/cifs/ --data_dir /Scr/hyunpark/ArgonneGNN/argonne_gnn/data --use_tensors --load_ckpt_path /Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub --dataset cifdata --crystal --task ethanol --which_mode infer

def md(smiles_list: List[str]):
    #https://github.com/hyunp2/EBM/blob/main/Legacy/testing_adp_md.py#:~:text=atoms%20%3D%20Atoms,1%3DH%2C%20etc.)
#     molecule = from_smiles(smiles) #torch_geometric Data instance
#     atoms: ase.Atoms = Atoms(positions=molecule.pos.detach().cpu().numpy(), 
# 		  numbers=molecule.z.view(-1,).detach().cpu().numpy(), pbc=True)  # positions in A, numbers in integers (1=H, etc.)
    opt = get_parser()
    assert len(smiles_list) == 1, "List is an input, but has one smiles string!"
    smiles = smiles_list[0]

    if opt.log:
        logger = WandbLogger(name=None, entity="argonne_gnn", project='internship')
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache/wandb")
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".config/wandb")        
    else:
        logger = None
	
    train_loader, val_loader, test_loader, mean, std = call_loader(opt)

    model = call_model(opt, mean, std, logger)

    if not os.path.exists(opt.ase_save_dir):
        os.makedirs(opt.ase_save_dir)
    from_smiles(smiles, write_to_xyz=f"{opt.md_name}.xyz") #write a xyz
    shutil.move(f"./{opt.md_name}.xyz", os.path.join(opt.ase_save_dir, f"{opt.md_name}.xyz")) #move to ase_run directory
    ase_run = AseInterface(molecule_path=f"{opt.md_name}.xyz", ml_model=model, working_dir=opt.ase_save_dir) #opt.ase_save_dir is ase_run directory

    if opt.which_mode in ["md"]:
        ase_run.init_md(name=opt.md_name, time_step=opt.timestep, temp_bath=opt.heat_bath)
        ase_run.run_md(steps=opt.md_steps)
    elif opt.which_mode in ["relax"]:
        ase_run.optimize(steps=opt.md_steps)
    elif opt.which_mode in ["ase2mda"]:    
        u = ase_run.convert_to_mdanalysis(ase_traj=os.path.join(opt.ase_save_dir, f"{opt.md_name}.traj"), slices=slice(0,None,100), save_format=opt.mda_format )
        print(u.trajectory)
#     python -m main_pub --log --backbone torchmdnet --gpu --name torchmdnet_pub --epoches 1000 --batch_size 512 --optimizer torch_adam --data_dir /Scr/hyunpark/ArgonneGNN/argonne_gnn/data --use_tensors --load_ckpt_path /Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/save_pub --dataset md17 --task ethanol --which_mode ase2mda --md_name acetic_acid --smiles_list "CC(=O)O" "O=C1NC=CC(=O)N1"

def dict2rdkitmol(input_dict):
	#import pickle, io
	#moldict = None
	#with io.open("./argonne_gnn_gitlab/for_rdkit_conversion.pickle", "rb") as f:
		#moldict = pickle.load(f)
	import pandas as pd
	df = pd.DataFrame([input_dict["z"].tolist()] + input_dict["pos"].T.tolist()).T
	df.columns = ["atmNum", "x", "y", "z"]
	df = df.astype({"atmNum": int, "x": float, "y": float, "z": float})
	periodic_table = ["XXXX", \
	"H", "He", \
	"Li", "Be", "B", "C", "N", "O", "F", "Ne", \
	"Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", \
	"K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", \
	"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", \
	"Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", \
	"Fr", "Ra",  "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
	df["element"] = [periodic_table[x] for x in df["atmNum"]]
	df["q"] = 0. 
	#with io.open("tmp.xyz", "w", newline="\n") as f:
		#f.write(str(len(df)) + "\n\n" + df[["element", "x", "y", "z"]].to_string(header=None, index=None))
	pdb_str = ""
	for i in df.index:
		pdb_str = pdb_str + "HETATM" + \
							"{:5d}".format(i) + \
							" " + \
							df.at[i, "element"].rjust(4, " ") + \
							" " + \
							"UNL" + \
							"  " + \
							"{:4d}".format(1) + \
							"    " + \
							"{:.3f}".format(df.at[i, "x"]).rjust(8, " ") + \
							"{:.3f}".format(df.at[i, "y"]).rjust(8, " ") + \
							"{:.3f}".format(df.at[i, "z"]).rjust(8, " ") + \
							"{:.2f}".format(1).rjust(6, " ") + \
							"{:.2f}".format(0).rjust(6, " ") + \
							"      " + \
							"0".ljust(4, " ") + \
							df.at[i, "element"].rjust(2, " ") + \
							"{:.0f}".format(df.at[i, "q"]).rjust(2, " ") + "\n"

	lines = ""
	pdb_str = pdb_str + lines + "END"
	# print(pdb_str)
	# with io.open("tmp.pdb", "w", newline="\n") as f:
	# 	f.write(pdb_str)
	from rdkit import Chem
	return Chem.MolFromPDBBlock(pdb_str)

if __name__ == "__main__":
    opt = get_parser()
    if opt.which_mode in ["train"]:
        run()
    elif opt.which_mode in ["tuning"]:
        hyper()
    elif opt.which_mode in ["explain"]:
        explain(opt.smiles_list)
    elif opt.which_mode in ["infer"]:
        infer(opt.smiles_list)
    elif opt.which_mode in ["md","relax","ase2mda"]:
#         md("CC(=O)O")
        md(opt.smiles_list)
#         md("c1ccccc1")

#     opt.log = True
#     opt.backbone = "torchmdnet"
#     opt.gpu = True
#     opt.use_tensors = True
#     opt.load_ckpt_path = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/save_pub"
#     opt.dataset = "qm9"
#     opt.task = "homo"
#     # opt.which_mode = "md"
#     # opt.name = "torchmdnet_pub"
	
# #     if opt.log:
# #         logger = WandbLogger(name=None, entity="argonne_gnn", project='internship')
# #         os.environ["WANDB_CACHE_DIR"] = os.getcwd()
# #     else:
# #         logger = None
	
# #     if opt.dataset in ["qm9edge"]:
# #         datamodule = DataModuleEdge(opt=opt)
#     if opt.dataset in ["qm9", "md17", "ani1", "ani1x", "moleculenet"]:
#         datamodule = DataModuleOthers(hparams=opt)
# #     if opt.dataset in ["cifdata","gandata"]:
# #         datamodule = DataModuleCrystal(opt=opt)
#     # train_loader = datamodule.train_dataloader()
#     # val_loader = datamodule.val_dataloader()
#     test_loader = datamodule.test_dataloader()
# #     mean = datamodule.mean
# #     std = datamodule.std
#     data = iter(test_loader).next()
#     bat = data["batch"]
#     z = data["z"][bat==0]
#     pos = data["pos"][bat==0]
#     f = open("for_rdkit_conversion.pickle", "wb")
#     pickle.dump({"z": z, "pos": pos}, f)

#     #Model 
#     model = BACKBONES.get(opt.backbone, physnet.Physnet) #Uninitialized class
#     model_kwargs = BACKBONE_KWARGS.get(opt.backbone, None) #TorchMDNet not yet!

#     if opt.backbone not in ["calignn", "alignn"]: model_kwargs.update({"explain": opt.explain})  
#     else: model_kwargs.explain = opt.explain #only for alignn net (due to pydantic)
    
#     if opt.backbone in ["schnet", "physnet", "dimenet", "dimenetpp","cschnet","cphysnet","cgcnn"]:
#         model_kwargs.update({"mean":mean, "std":std})
#         model = model(**model_kwargs) 
#     elif opt.backbone in ["alignn","calignn"]:
#         model_kwargs.mean = mean
#         model_kwargs.std = std
#         model = model(model_kwargs) #Accounting for alignn net
#     elif opt.backbone in ["torchmdnet","ctorchmdnet"]:
#         model_kwargs.update({"mean":mean, "std":std})
#         model = torchmdnet.create_model(model_kwargs, mean=mean, std=std) if opt.backbone=="torchmdnet" else ctorchmdnet.create_model(model_kwargs)

#     if opt.gpu:
#         model = model.to(torch.cuda.current_device())
	
#     path_and_name = os.path.join(opt.load_ckpt_path, "{}.pth".format(opt.name))
#     load_state(model, optimizer=None, scheduler_groups=None, path_and_name=path_and_name, model_only=True, use_artifacts=False, logger=logger, name=None)

#     from ase.io import read, write
#     molecule = read("ase_run/somemol.xyz")
#     molecule.calc = CustomCalculator(model)
    
