# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import argparse
import ctypes
import logging
import os
import random
from functools import wraps#, partial
# import functools
from typing import Union, List, Dict
import wandb
import pandas as pd
import PIL

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
import pathlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import *
# from wandb.sdk.interface._dtypes import AnyType

# from torch_geometric.data import Batch
# from torch_cluster import radius_graph
# import copy

# https://github.com/whitead/molcloud for visualization of LIGANDS!

def aggregate_residual(feats1, feats2, method: str):
    """ Add or concatenate two fiber features together. If degrees don't match, will use the ones of feats2. """
    if method in ['add', 'sum']:
        return {k: (v + feats1[k]) if k in feats1 else v for k, v in feats2.items()}
    elif method in ['cat', 'concat']:
        return {k: torch.cat([v, feats1[k]], dim=1) if k in feats1 else v for k, v in feats2.items()}
    else:
        raise ValueError('Method must be add/sum or cat/concat')


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


def unfuse_features(features: Tensor, degrees: List[int]) -> Dict[str, Tensor]:
    return dict(zip(map(str, degrees), features.split([degree_to_dim(deg) for deg in degrees], dim=-1)))


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_cuda(x):
    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
    if isinstance(x, Tensor):
        return x.cuda(non_blocking=True)
    elif isinstance(x, tuple):
        return (to_cuda(v) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=torch.cuda.current_device())


def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', 0))


def init_distributed() -> bool:
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        if backend == 'nccl':
            torch.cuda.set_device(get_local_rank())
        else:
            logging.warning('Running on CPU only!')
        assert torch.distributed.is_initialized()
    return distributed

def init_distributed_spawn() -> bool:
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://', rank=local_rank, world_size=world_size)
        if backend == 'nccl':
            torch.cuda.set_device(get_local_rank())
        else:
            logging.warning('Running on CPU only!')
        assert torch.distributed.is_initialized()
        
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    return distributed

def increase_l2_fetch_granularity():
    # maximum fetch granularity of L2: 128 bytes
    _libcudart = ctypes.CDLL('libcudart.so')
    # set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def using_tensor_cores(amp: bool) -> bool:
    major_cc, minor_cc = torch.cuda.get_device_capability()
    return (amp and major_cc >= 7) or major_cc >= 8

class Logger(ABC):
    @rank_zero_only
    @abstractmethod
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    @abstractmethod
    def log_metrics(self, metrics, step=None):
        pass

    @staticmethod
    def _sanitize_params(params):
        def _sanitize(val):
            if isinstance(val, Callable):
                try:
                    _val = val()
                    if isinstance(_val, Callable):
                        return val.__name__
                    return _val
                except Exception:
                    return getattr(val, "__name__", None)
            elif isinstance(val, pathlib.Path) or isinstance(val, Enum):
                return str(val)
            return val

        return {key: _sanitize(val) for key, val in params.items()}
    
    @rank_zero_only
    @abstractmethod
    def log_deephyper(self, dataframe: pd.DataFrame, step: Optional[int] = None) -> None:
        pass
    
    @staticmethod
    @abstractmethod
    def mol_to_pil_image(molecule: str, width: int = 300, height: int = 300) -> "PIL.Image":
        pass
    
    @staticmethod
    @abstractmethod
    def rdkit_dataframe(smiles_string: str):
        pass
    
    
    
class WandbLogger(Logger):
    def __init__(
            self,
            name: str,
            save_dir: pathlib.Path=None,
            id: Optional[str] = None,
            project: Optional[str] = None,
            entity: Optional[str] = None
    ):
        super().__init__()
        if not dist.is_initialized() or dist.get_rank() == 0:
    #             save_dir.mkdir(parents=True, exist_ok=True)
            self.experiment = wandb.init(name=name,
                                         project=project,
                                         entity=entity,
                                         settings=wandb.Settings(start_method="fork"),
                                         id=id,
                                         dir=None,
                                         resume='allow',
                                         anonymous='must')

    @rank_zero_only
    def start_watching(self, model):
        wandb.watch(model)

    @property
    @rank_zero_only
    def finish(self, ):
        self.experiment.finish()

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        params = self._sanitize_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_html(self, html_file: str, name: str=None) -> None:
        if name == None : self.experiment.log({"html": wandb.Html(open(html_file))})
        else: self.experiment.log({name: wandb.Html(open(html_file)) }) 

    @rank_zero_only
    def log_image(self, image: Union[np.ndarray], name: str=None) -> None:
        if name == None : self.experiment.log({"image": wandb.Image(image)})
        else: self.experiment.log({name: wandb.Image(image)}) 

    @rank_zero_only
    def log_dataframe(self, df: Union[pd.DataFrame], name: str=None) -> None:
        table = wandb.Table(dataframe=df)
        if name == None : self.experiment.log({"dataframe": table}) 
        else: self.experiment.log({name: table}) 

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is not None:
            self.experiment.log({**metrics, 'epoch': step})
        else:
            self.experiment.log(metrics)
            
    @rank_zero_only
    def log_artifacts(self, name: str, dtype: str, path_and_name: Union[str, pathlib.Path]) -> None:
        """https://docs.wandb.ai/guides/artifacts/api"""
        artifact = wandb.Artifact(name=name, type=dtype)
        artifact.add_file(str(path_and_name)) #which directory's file to add; when downloading it downloads directory/file
        self.experiment.log_artifact(artifact)

    @rank_zero_only    
    def download_artifacts(self, name: str) -> None:
        """https://docs.wandb.ai/guides/artifacts/api"""
        if not ":" in name:
            name += ":latest"
        artifact = self.experiment.use_artifact(name)
        return artifact.download() #returns directory/file
            
            
