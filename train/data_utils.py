from __future__ import print_function, division
import abc, sys
import collections
from torch_geometric.data import Data, Dataset
import csv
from curtsies import fmtfuncs as cf
import functools
import json
import os
import argparse
import warnings
import numpy as np
import torch
from pymatgen.core.structure import Structure
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader #Can this handle DDP? yeah!
import torch.distributed as dist 
from train.dist_utils import get_local_rank
from torch.utils.data import DistributedSampler
from typing import *
# from train.cgcnn_data_utils import * #get_dataloader func, _get_split_sizes etc.
import pickle
from pymatgen.optimization.neighbors import find_points_in_spheres

import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory

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
    cif_id: str or int --> CHANGED to ATOM SPECIES NUMBER shape: (num_nodes) and LongTensor
    """
    def __init__(self, root_dir=os.path.join(os.getcwd(), "cif_files"), max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, original=True, truncate_above: float=None):
        super().__init__()
        self.root_dir = root_dir
        self.original = original
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!' # e.g. imax
        
        id_prop_file_ = os.path.join(self.root_dir, 'id_prop.csv') #for target
        assert os.path.exists(id_prop_file_), 'id_prop.csv does not exist!'
        # id_prop_file_ = os.path.join(self.root_dir, 'id_prop_5000.csv') #for target
        # id_prop_file_ = os.path.join(self.root_dir, 'id_prop_0.1.csv') #for target

        if os.path.exists(id_prop_file_):
            # print('id_prop_5000.csv will be used for speedy training!')
            id_prop_file = id_prop_file_

        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader if row[1] < truncate_above] if truncate_above is not None else [row for row in reader]
        print(cf.on_yellow(f"Truncate_above is {truncate_above is not None}..."))
        # random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file) #Embedding dict!
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step) #nn.Module

    def len(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def get(self, idx):
        cif_id, target = self.id_prop_data[idx]
        cif_name = cif_id
        try:
            if ".cif" in cif_id:
                cif_id = os.path.splitext(cif_id)[0]
            crystal = Structure.from_file(os.path.join(self.root_dir,
                                                    cif_id+'.cif')) #idx is a 1-cif file index
            atom_idx = [crystal.sites[i].specie.number for i in range(len(crystal))]
            atom_fea = np.vstack([self.ari.get_atom_fea(crystal.sites[i].specie.number)
                                for i in range(len(crystal))]) #Embed atoms of crystal
            atom_fea = torch.from_numpy(atom_fea) #tensorfy
            
            #Original:
            if self.original:
                all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True) #list has num_atoms elements; each element has variable length neighbors! 
                all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs] #For each atom of crystal, sort each atomic neighbors based on distance!
                nbr_fea_idx_row, nbr_fea_idx_col, nbr_fea = [], [], []

                for idx, nbr in enumerate(all_nbrs):
                    if len(nbr) < self.max_num_nbr:
                        warnings.warn('{} not find enough neighbors to build graph. '
                                    'If it happens frequently, consider increase '
                                    'radius.'.format(cif_id))   
                    nbr_fea_idx_row.extend([idx]*len(nbr)) #num_edges
                    nbr_fea_idx_col.extend(list(map(lambda x: x[2], nbr))) #num_edges
                    nbr_fea.extend(list(map(lambda x: x[1], nbr))) #num_edges
            else:
                #BELOW: https://github.com/materialsvirtuallab/m3gnet/blob/main/m3gnet/graph/_structure.py#:~:text=cart_coords%2C,)
                lattice_matrix = np.ascontiguousarray(np.array(crystal.lattice.matrix), dtype=float)
                pbc = np.array([1, 1, 1], dtype=int)
                cart_coords = np.ascontiguousarray(np.array(crystal.cart_coords), dtype=float)
                numerical_tol = 1e-8
                center_indices, neighbor_indices, images, distances=find_points_in_spheres(cart_coords, cart_coords, r=self.radius, pbc=pbc,
                                                                                        lattice=lattice_matrix, tol=numerical_tol)
                exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
                nbr_fea_idx_row, nbr_fea_idx_col, images_offset, nbr_fea = list(map(lambda inp: inp[exclude_self], (center_indices, neighbor_indices, images, distances) ))

            nbr_fea_idx_row, nbr_fea_idx_col, nbr_fea = np.array(nbr_fea_idx_row), np.array(nbr_fea_idx_col), torch.from_numpy(np.array(nbr_fea)) #(n_i, M), (n_i, atom_fea_len) --> (edges=n_i*M,), (edges=n_i*M,), (edges=n_i*M, atom_fea_len)  
            dists = nbr_fea.type(torch.float32) #edges,
            nbr_fea = self.gdf.expand(nbr_fea) #(n_i, M, nbr_fea_len) --> (edges, nbr_fea_len)
            atom_fea = torch.tensor(atom_fea).type(torch.float32) #(natoms, atom_fea_len)
            nbr_fea = torch.tensor(nbr_fea).type(torch.float32) #(edges, nbr_fea_len)
            nbr_fea_idx_row = torch.LongTensor(nbr_fea_idx_row).type(torch.long) #edges,
            nbr_fea_idx_col = torch.LongTensor(nbr_fea_idx_col).type(torch.long) #edges,
            nbr_fea_idx = torch.stack((nbr_fea_idx_row, nbr_fea_idx_col), dim=0) #(2,edges)
            target = torch.Tensor([float(target)]).type(torch.float32) #(1,) so that (B,1) when batched!
            atom_idx = torch.from_numpy(np.array(atom_idx)).type(torch.long) #nodes,
            return Data(x=atom_fea, edge_attr=nbr_fea, edge_index=nbr_fea_idx, edge_weight=dists, y=target, cif_id=atom_idx, ), cif_name #PyG type dataset
            
        except Exception as e:
            print(e)

def data_augmentation(opt: argparse.ArgumentParser, full_dataset: Dataset):
    """A function to oversample minority"""
    # indices: Union[np.array, torch.LongTensor] 
    assert os.path.exists(opt.data_dir_crystal), 'root_dir does not exist!' # e.g. imax
    id_prop_file = os.path.join(opt.data_dir_crystal, 'id_prop.csv') #for target
    assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = np.array([row for row in reader])
    target_values = torch.from_numpy(id_prop_data[:,1].astype(float)).view(-1,)

    # torch.quantile(target_values, 0.99)
    target_top_mofs = target_values >= 4

    all_indices = torch.arange(target_values.size(0))
    top_indices = torch.where(target_top_mofs==True)[0] #only top mof indices
    non_top_indices = torch.where(target_top_mofs==False)[0] #rest (nontop) mof indices
    rand_indices_for_nontop = torch.randperm(non_top_indices.size(0), generator=torch.Generator(device=torch.cuda.current_device()).manual_seed(42), device=torch.cuda.current_device()) #shuffle rest (nontop) mof indices
    non_top_indices = non_top_indices[rand_indices_for_nontop] #shuffle the indicies for train-val-test split! (NONTOP only!)
    split_lengths = torch.LongTensor([int(len(non_top_indices)*opt.train_frac), (len(non_top_indices)*(1-opt.train_frac)//2), (len(non_top_indices)-(len(non_top_indices)*(1-opt.train_frac)//2))])
    split_lengths_cumsum = torch.cat([torch.tensor([0]), split_lengths.cumsum(dim=0)], dim=0)
    np.random.seed(42)
    #BELOW: https://github.com/scikit-learn-contrib/imbalanced-learn/blob/9f8830e/imblearn/over_sampling/_random_over_sampler.py#L23:~:text=target_class_indices%20%3D%20np.flatnonzero(y%20%3D%3D%20class_sample),)
    top_indices_repeat = torch.from_numpy(np.random.choice(top_indices.detach().cpu().numpy(), size=opt.num_oversample, replace=True)) #

    train_idx, val_idx, test_idx = list(map(lambda start, end: non_top_indices[start : end], split_lengths_cumsum[:-1], split_lengths_cumsum[1:] ))
    train_idx = torch.cat([train_idx, top_indices_repeat], dim=0) #nontop train + top train!
    trainset, valset, testset = list(map(lambda indices: torch.utils.data.Subset(full_dataset, indices), [train_idx, val_idx, test_idx] ))
    
    train_idx_keys = np.array(list(collections.Counter(train_idx.detach().cpu().numpy()).keys()))
    train_idx_values = np.array(list(collections.Counter(train_idx.detach().cpu().numpy()).values()))
    print(cf.on_yellow(f"Originally {len(top_indices)} Top-MOFs. Replicated {np.sort(train_idx_values[train_idx_values > 1]).sum()} times..."))
    print(cf.on_yellow(f"Data enriched by {100 * (np.sort(train_idx_values[train_idx_values > 1]).sum()) / (len(all_indices))} percent..."))
    assert np.all (np.sort(top_indices) == np.sort(train_idx_keys[train_idx_values > 1]) ), "duplicated samples must exist in training set"
    # BELOW: Returns the unique values in t1 that are not in t2.
    # https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors#:~:text=Returns%20the%20unique%20values%20in%20t1%20that%20are%20not%20in%20t2.
    # t1 = torch.unique(all_indices)
    # t2 = torch.unique(top_indices)
    # non_top_indices = t1[(t1[:, None] != t2).all(dim=1)]
    return trainset, valset, testset

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
        
        root_dir = self.opt.data_dir_crystal #change to data_dir and DEPRECATE this command
        print("Root dir chosen is", root_dir)
        if self.opt.dataset in ["cifdata"]:
            if self.opt.save_to_pickle == None:
                full_dataset = CIFData(root_dir, truncate_above=self.opt.truncate_above)
            else:
                pickle_data = os.path.splitext(self.opt.save_to_pickle)[0]
                pickle_data += ".pickle"
                if not os.path.exists(pickle_data):
                    print("Saving a pickle file!")
                    full_dataset = CIFData(root_dir)
                    with open(pickle_data, "wb") as f:
                        pickle.dump(full_dataset, f)
                else:
                    print("Loading a saved pickle file!")
                    class CustomUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if name == 'CIFData':
                                return CIFData
                            return super().find_class(module, name)
                    full_dataset = CustomUnpickler(open(pickle_data,"rb")).load()
                    
            
        self.dataloader_kwargs = {'pin_memory': opt.pin_memory, 'persistent_workers': dataloader_kwargs.get('num_workers', 0) > 0,
                                 'batch_size': opt.batch_size} if not self.opt.dataset in ["gandata"] else {'pin_memory': opt.pin_memory, 'persistent_workers': dataloader_kwargs.get('num_workers', 0) > 0,
                                 'batch_size': opt.sample_size}
        if not self.opt.dataset in ["cifdata"]:
            self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(full_dataset, _get_split_sizes(self.opt.train_frac, full_dataset),
                                                                generator=torch.Generator().manual_seed(0))
        else:
            if not opt.num_oversample == 0:
                self.ds_train, self.ds_val, self.ds_test = data_augmentation(opt, full_dataset)
            else:
                self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(full_dataset, _get_split_sizes(self.opt.train_frac, full_dataset),
                                                                generator=torch.Generator().manual_seed(0))
        
        # energy_batch_data = torch_geometric.data.Batch.from_data_list(self.ds_train)
        # if "y" in energy_batch_data:
        # self._mean = 1.0873 #energy_batch_data.y.mean(dim=0) #pass to _standardize; 1.0873
        # self._std = 0.9749 #energy_batch_data.y.std(dim=0) #pass to _standardize; 0.9749
        self._mean = None
        self._std = None
        # else:
        #     self._mean = 0.
        #     self._std = 1.

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std
    
    def prepare_data(self, ):
        """ Method called only once per node. Put here any downloading or preprocessing """
        root_dir = self.opt.data_dir_crystal
        if self.opt.dataset in ["cifdata"]:
            full_dataset = CIFData(root_dir)
        elif self.opt.dataset in ["gandata"]:
            full_dataset = GANData(root_dir, make_data=self.opt.make_data)
        elif self.opt.dataset in ["cdvaedata"]:
            full_dataset = CDVAEData(root_dir)
            
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
    
if __name__ == "__main__":
#     root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax"
    root_dir = "/Scr/hyunpark/ArgonneGNN/cubicgan_modified/data/trn-cifs/"

    dataset = GANData(root_dir, make_data=False)
    print(dataset[3])
#     print(dataset[30].aux_data, dataset[3].arr_coords)

#     dataset = CDVAEData(root_dir)
#     print(dataset[3])

#     dataset = CIFData(root_dir, original=False)
#     print(dataset[3])
