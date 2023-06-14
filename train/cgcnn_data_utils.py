from __future__ import print_function, division
import abc, sys

import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader #Can this handle DDP? yeah!

import torch.distributed as dist 
from train.dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, WandbLogger
from torch.utils.data import DistributedSampler
from typing import *

__all__ = ["_get_split_sizes", "get_dataloader", "GaussianDistance", "AtomInitializer", 
           "AtomCustomJSONInitializer", "CIFData", "DataModuleCrystal"]

def _get_split_sizes(train_frac: float, full_dataset: Dataset) -> Tuple[int, int, int]:
    """DONE: Need to change split schemes!"""
    len_full = len(full_dataset)
    len_train = int(len_full * train_frac)
    len_test = int(0.1 * len_full)
    len_val = len_full - len_train - len_test
    return len_train, len_val, len_test  
  
def get_dataloader(dataset: Dataset, shuffle: bool, collate_fn: callable=None, **kwargs):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    loader = DataLoader(dataset, shuffle=(shuffle and sampler is None), sampler=sampler, collate_fn=collate_fn, **kwargs)
    return loader

class GaussianDistance(torch.nn.Module):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
#         self.filter = np.arange(dmin, dmax+step, step)
        super().__init__()
        self.register_buffer("filter", torch.arange(dmin, dmax+step, step).type(torch.float32))
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return torch.exp(-(distances[..., None] - self.filter)**2 /
                      self.var**2)
      
    def forward(self, distances):
        return self.expand(distances)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = torch.from_numpy(np.array(value)).type(torch.float32)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        super().__init__()
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!' # e.g. imax
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv') #for target
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file) #Embedding dict!
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step) #nn.Module

    def len(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def get(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id+'.cif')) #idx is a 1-cif file index
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal.sites[i].specie.number)
                              for i in range(len(crystal))]) #Embed atoms of crystal
        atom_fea = torch.from_numpy(atom_fea) #tensorfy
        
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True) #list has num_atoms elements; each element has variable length neighbors! 
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs] #For each atom of crystal, sort each atomic neighbors based on distance!
        nbr_fea_idx_row, nbr_fea_idx_col, nbr_fea = [], [], []
#         for nbr in all_nbrs:
#             if len(nbr) < self.max_num_nbr:
#                 warnings.warn('{} not find enough neighbors to build graph. '
#                               'If it happens frequently, consider increase '
#                               'radius.'.format(cif_id))
#                 nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
#                                    [0] * (self.max_num_nbr - len(nbr))) #index
#                 nbr_fea.append(list(map(lambda x: x[1], nbr)) +
#                                [self.radius + 1.] * (self.max_num_nbr -
#                                                      len(nbr))) #distance
#             else:
#                 nbr_fea_idx.append(list(map(lambda x: x[2],
#                                             nbr[:self.max_num_nbr])))
#                 nbr_fea.append(list(map(lambda x: x[1],
#                                         nbr[:self.max_num_nbr])))
        for idx, nbr in enumerate(all_nbrs):
            if len(nbr) < self.max_num_nbr:
               warnings.warn('{} not find enough neighbors to build graph. '
                          'If it happens frequently, consider increase '
                          'radius.'.format(cif_id))   
            nbr_fea_idx_row.extend([idx]*len(nbr)) #num_edges
            nbr_fea_idx_col.extend(list(map(lambda x: x[2], nbr))) #num_edges
            nbr_fea.extend(list(map(lambda x: x[1], nbr))) #num_edges

        nbr_fea_idx_row, nbr_fea_idx_col, nbr_fea = np.array(nbr_fea_idx_row), np.array(nbr_fea_idx_col), torch.from_numpy(np.array(nbr_fea)) #(n_i, M), (n_i, atom_fea_len) --> (edges=n_i*M,), (edges=n_i*M,), (edges=n_i*M, atom_fea_len)  
        dists = nbr_fea.type(torch.float32) #edges,
        nbr_fea = self.gdf.expand(nbr_fea) #(n_i, M, nbr_fea_len) --> (edges, nbr_fea_len)
        atom_fea = torch.tensor(atom_fea).type(torch.float32) #(natoms, atom_fea_len)
        nbr_fea = torch.tensor(nbr_fea).type(torch.float32) #(edges, nbr_fea_len)
        nbr_fea_idx_row = torch.LongTensor(nbr_fea_idx_row).type(torch.long) #edges,
        nbr_fea_idx_col = torch.LongTensor(nbr_fea_idx_col).type(torch.long) #edges,
        nbr_fea_idx = torch.stack((nbr_fea_idx_row, nbr_fea_idx_col), dim=0) #(2,edges)
        target = torch.Tensor([float(target)]).type(torch.float32) #(1,) so that (B,1) when batched!
        return Data(x=atom_fea, edge_attr=nbr_fea, edge_index=nbr_fea_idx, edge_weight=dists, y=target, cif_id=cif_id) #PyG type dataset
      
class DataModuleCrystal(abc.ABC):
    """ Abstract DataModule. Children must define self.ds_{train | val | test}. """

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        self.opt = opt = dataloader_kwargs.pop("opt")

        if get_local_rank() == 0:
            self.prepare_data()
            print(f"{get_local_rank()}-th core is parsed!")
#             self.prepare_data(opt=self.opt, data=self.data, mode=self.mode) #torch.utils.data.Dataset; useful when DOWNLOADING!

        # Wait until rank zero has prepared the data (download, preprocessing, ...)
        if dist.is_initialized():
            dist.barrier(device_ids=[get_local_rank()]) #WAITNG for 0-th core is done!
        
        root_dir = self.opt.data_dir_crystal
        full_dataset = CIFData(root_dir)
        
        self.dataloader_kwargs = {'pin_memory': opt.pin_memory, 'persistent_workers': dataloader_kwargs.get('num_workers', 0) > 0,
                                 'batch_size': opt.batch_size}
        self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(full_dataset, _get_split_sizes(self.opt.train_frac, full_dataset),
                                                                generator=torch.Generator().manual_seed(0))
        
    def prepare_data(self, ):
        """ Method called only once per node. Put here any downloading or preprocessing """
        root_dir = self.opt.data_dir_crystal
        full_dataset = CIFData(root_dir)
        
    def train_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print(cf.on_red("Not applicable for crystal..."))
        return get_dataloader(self.ds_train, shuffle=True, collate_fn=None, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print(cf.on_red("Not applicable for crystal..."))
        return get_dataloader(self.ds_val, shuffle=False, collate_fn=None, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print(cf.on_red("Not applicable for crystal..."))
        return get_dataloader(self.ds_test, shuffle=False, collate_fn=None, **self.dataloader_kwargs)    

