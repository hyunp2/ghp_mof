from __future__ import print_function, division
import abc, sys
import collections
import torch_geometric
from torch_geometric.data import Data, Dataset
import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory
import csv
from curtsies import fmtfuncs as cf
import functools
import json
import os
import argparse
import random
import warnings
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile,CifParser
from pymatgen.core.lattice import Lattice
from pymatgen.core import Element, Composition
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader #Can this handle DDP? yeah!
import torch.distributed as dist 
from train.dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, WandbLogger
from torch.utils.data import DistributedSampler
from typing import *
from crystals.cgcnn_data_utils import * #get_dataloader func, _get_split_sizes etc.
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import multiprocessing as mp
import torch.distributed as dist
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.symmetry.groups import SpaceGroup
from p_tqdm import p_umap
import wandb
from pymatgen.optimization.neighbors import find_points_in_spheres
from pccc_mof.featurizer import *

def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
#     crystal = Structure.from_str(crystal_str, fmt='cif')
    crystal = Structure.from_file(crystal_str)
    
    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal

def abs_cap(val, max_abs_val=1):
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
def build_crystal_graph(crystal: Structure, graph_method='crystalnn'):
    """
    """

    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms

def preprocess_tensors(crystal_array_list, niggli, primitive, graph_method):
    def process_one(batch_idx, crystal_array, niggli, primitive, graph_method):
        frac_coords = crystal_array[0]
        atom_types = crystal_array[1]
        lengths = crystal_array[2]
        angles = crystal_array[3]
        crystal = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False)
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {
            'batch_idx': batch_idx,
            'graph_arrays': graph_arrays,
        }
        return result_dict

    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_array_list))),
        crystal_array_list,
        [niggli] * len(crystal_array_list),
        [primitive] * len(crystal_array_list),
        [graph_method] * len(crystal_array_list),
        num_cpus=30,
    )
    ordered_results = list(
        sorted(unordered_results, key=lambda x: x['batch_idx']))
    return ordered_results

class CDVAEData(Dataset):
    def __init__(self, root_dir, random_seed=123, make_data=False):
        super().__init__()
        self.root_dir = root_dir
        assert os.path.exists(root_dir), 'root_dir does not exist!' # e.g. imax
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv') #for target
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
#         random.shuffle(self.id_prop_data)
        valid_files = list(filter(lambda inp: os.path.splitext(inp)[1] == ".cif", os.listdir(root_dir) )) #only CIF list
        valid_files = list(map(lambda inp: os.path.join(root_dir, inp), valid_files )) #only CIF list with root_dir
        crystals = list(map(build_crystal, valid_files)) #List[Structure]
        self.results = list(map(build_crystal_graph, crystals))
        
    def len(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def get(self, idx):
        (frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms) = self.results[idx]
            
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data
    
LaAc = ['Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']
short_LaAc = ['Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
cubics_ratio = {'Fm-3m': 186344, 'F-43m': 184162,'Pm-3m':5243}
sp2id = {'Fm-3m':0,'F-43m':1,'Pm-3m':2} #WIP: USE PGCGM extension!

class sp_lookup:
    def __init__(self, device, sp_dict):
        self._affine_matrix_list = []
        self._affine_matrix_range = []
        self.device = device

        #mapping the heck
        d = {}
        for i in range(1,231):
            if i == 222:
                d['Pn-3n'] = i
                continue
            if i == 224:
                d['Pn-3m'] = i
                continue
            if i == 227:
                d['Fd-3m'] = i
                continue
            if i == 228:
                d['Fd-3c'] = i
                continue
            if i == 129:
                d['P4/nmm'] = i
                continue
            symbol = SpaceGroup.from_int_number(i).symbol
            if symbol.endswith("H"):
                symbol = symbol.replace("H", "")
            d[symbol] = i
        
        sp_list = []
        for symbol,i in sp_dict.items():
            sp_list.append((i, d[symbol], symbol))
        # exit(sp_list)
        for i,spid,_ in sp_list:
            symops = SpaceGroup.from_int_number(spid).symmetry_ops

            for op in symops:
                tmp = op.affine_matrix.astype(np.float32)
                
                if np.all(tmp == -1.0):
                    print(tmp)
                self._affine_matrix_list.append(tmp)
            
            self._affine_matrix_range.append((len(self._affine_matrix_list) - len(symops),\
             len(self._affine_matrix_list)))

    @property
    def affine_matrix_list(self):
        return torch.Tensor(self._affine_matrix_list).to(self.device)
    
    @property
    def affine_matrix_range(self):
        return torch.Tensor(self._affine_matrix_range).type(torch.int64).to(self.device)

    @property
    def symm_op_collection(self):
        arr = []
        for r0,r1 in self._affine_matrix_range:
            ops = np.array(self._affine_matrix_list)[r0:r1]
            zeros = np.zeros((192-len(ops), 4, 4))
            ops = np.concatenate((ops, zeros),0)
            arr.append(ops)
        arr = np.stack(arr, 0)

        return torch.Tensor(arr).type(torch.float32).to(self.device)

def atom_embedding(d_elements: Dict[str, int]):
#     d_elements #Structure.sites instance
    features = np.zeros((len(d_elements), 23))
    for k in d_elements:
        # i: index; k: Structure.site
#         e = k.specie
#         print(k)
        i = d_elements[k]
        e = Element(k)
        features[i][0] = e.Z
        features[i][1] = e.X
        features[i][2] = e.row
        features[i][3] = e.group
        features[i][4] = e.atomic_mass
        features[i][5] = float(e.atomic_radius)
        features[i][6] = e.mendeleev_no
        # features[i][7] = sum(e.atomic_orbitals.values())
        features[i][7] = float(e.average_ionic_radius)
        features[i][8] = float(e.average_cationic_radius)
        features[i][9] = float(e.average_anionic_radius)
        features[i][10] = sum(e.ionic_radii.values())
        features[i][11] = e.max_oxidation_state
        features[i][12] = e.min_oxidation_state
#         features[i][13] = np.nan_to_num(sum(e.oxidation_states)/len(e.oxidation_states), nan=0.0)
#         features[i][14] = np.nan_to_num(sum(e.common_oxidation_states)/len(e.common_oxidation_states), nan=0.0)
        features[i][13] = np.divide(sum(e.oxidation_states), len(e.oxidation_states), out=np.zeros(1), where=len(e.oxidation_states)!=0)
        features[i][14] = np.divide(sum(e.common_oxidation_states), len(e.common_oxidation_states), out=np.zeros(1), where=len(e.common_oxidation_states)!=0)
        features[i][15] = float(e.is_noble_gas)
        features[i][16] = float(e.is_transition_metal)
        features[i][17] = float(e.is_post_transition_metal)
        features[i][18] = float(e.is_metalloid)
        features[i][19] = float(e.is_alkali)
        features[i][20] = float(e.is_alkaline)
        features[i][21] = float(e.is_halogen)
        features[i][22] = float(e.molar_volume)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features

def extract_function(file):
    cif = CifFile.from_file(file)
    data = cif.data
    formula = list(data.keys())[0]
    block = list(data.values())[0]
    bases = block['_atom_site_type_symbol']
    #remove materials with La AND Ac rows
    #remove materials having more than three base atoms 
    occu = np.array(block['_atom_site_occupancy']).astype(float)

    if len(bases)==3 and len(set(bases))==3 and all(occu == 1.0):
        xs = np.array(block['_atom_site_fract_x']).reshape((-1,1))
        ys = np.array(block['_atom_site_fract_y']).reshape((-1,1))
        zs = np.array(block['_atom_site_fract_z']).reshape((-1,1))
        coords = np.hstack([xs,ys,zs]).astype(float)
        lengths = np.array([block['_cell_length_a'],block['_cell_length_b'],block['_cell_length_c']]).astype(float)
        angles = np.array([block['_cell_angle_alpha'],block['_cell_angle_beta'],block['_cell_angle_gamma']]).astype(float)
        lattice = Lattice.from_parameters(
                lengths[0],
                lengths[1],
                lengths[2],
                angles[0],
                angles[1],
                angles[2],
            )
        matrix = lattice.matrix

        a = [
            formula,
            block['_symmetry_Int_Tables_number'],
            block['_symmetry_space_group_name_H-M'],
            bases,
            coords,
            matrix,
            lengths,
            angles
        ]

        b = [
            file.replace('.cif',''),
            formula,
            block['_symmetry_Int_Tables_number'],
            block['_symmetry_space_group_name_H-M'],
            ]
        return a,b

class GANData(Dataset):
    def __init__(self, root_dir, random_seed=123, make_data=False):
        super().__init__()
        self.root_dir = root_dir
        assert os.path.exists(root_dir), 'root_dir does not exist!' # e.g. imax
#         id_prop_file = os.path.join(self.root_dir, 'id_prop.csv') #for target
#         assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
#         with open(id_prop_file) as f:
#             reader = csv.reader(f)
#             self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
#         random.shuffle(self.id_prop_data)
        
        prob_sp = {k:cubics_ratio[k]/sum(cubics_ratio.values()) for k in cubics_ratio}
        prob = np.zeros(len(prob_sp))
        for k in prob_sp:
            prob[sp2id[k]] = prob_sp[k]
        self.prob = prob
        
        if make_data:
            filenames = os.listdir(os.getcwd())
            if np.isin(filenames, ['ternary-dataset-pool.pkl', 'ternary-lable-records.csv','cubic-elements-dict.json','cubic-elements-features.npy']).any():
                idx = np.isin(filenames, ['ternary-dataset-pool.pkl', 'ternary-lable-records.csv','cubic-elements-dict.json','cubic-elements-features.npy']).tolist().index(True)
                assert filenames[idx] in ['ternary-dataset-pool.pkl', 'ternary-lable-records.csv','cubic-elements-dict.json','cubic-elements-features.npy'], "not a file to delete..."
                os.unlink(filenames[idx])
                
            if (not os.path.exists(os.path.join(os.getcwd(), 'ternary-dataset-pool.pkl'))) or (not os.path.exists(os.path.join(os.getcwd(), 'ternary-lable-records.csv'))) or (not os.path.exists(os.path.join(os.getcwd(), 'cubic-elements-dict.json'))) or (not os.path.exists(os.path.join(os.getcwd(), 'cubic-elements-features.npy'))):
                if get_local_rank() == 0: 
                    valid_files = list(filter(lambda inp: os.path.splitext(inp)[1] == ".cif", os.listdir(root_dir) )) #only CIF list
                    valid_files = list(map(lambda inp: os.path.join(root_dir, inp), valid_files )) #only CIF list with root_dir
#                     pool = mp.Pool(processes=24)
#                     results = [pool.apply_async(extract_function, args=(file,)) for file in valid_files]
                    results = list(map(extract_function, valid_files))
#                     print(results[0].get())
                    d = {}
                    ternary_uniq_sites = []
                    for p in results:
#                         if p.get() is not None:
#                         a,b = p.get()
                        a,b = p
                        d[b[0]] = a
                        ternary_uniq_sites.append(b)
#                         print(a,b)
                    with open(os.path.join(os.getcwd(), 'ternary-dataset-pool.pkl'),'wb') as f:
                        pickle.dump(d, f, protocol=4)
                    df = pd.DataFrame(np.array(ternary_uniq_sites), columns=['id','formula','spid','spsym'])
                    df.to_csv(os.path.join(os.getcwd(), 'ternary-lable-records.csv'), index=False)

                    """
                    #Method 1 for element
#                     elements = Element.__dict__["_member_names_"] #List[str]; str is an element; NOT every element has a property!
                    elements = list(set(LaAc) - set(short_LaAc)) #comment out?
#                     elements = list(filter(lambda inp: Element(inp).atomic_radius, elements))
                    """
                    #Method 2 for element
                    values = df.values
                    ids,formulas = [],[]
                    for row in values:
                        ix,comp,_,symbol = row
                        if symbol in cubics_ratio:
                            ids.append(ix)
                            formulas.append(comp)
                    ids = np.array(ids).astype(str)
#                     np.random.shuffle(ids)
                    elements = []
                    for f in formulas:
                        elements += list(Composition(f).as_dict().keys())
                    elements = list(set(elements))
                    elements.sort()
                    
                    d_elements = {}
                    for i,e in enumerate(elements):
                        d_elements[e]=i #dict
                    with open(os.path.join(os.getcwd(), 'cubic-elements-dict.json'), 'w') as f:
                        json.dump(d_elements, f, indent=2)
                    embedding = atom_embedding(d_elements)
                    np.save(os.path.join(os.getcwd(), 'cubic-elements-features'), embedding)
                    
        if dist.is_initialized():
            dist.barrier(device_ids=[get_local_rank()]) #WAITNG for 0-th core is done!
            

        df = pd.read_csv(os.path.join(os.getcwd(), 'ternary-lable-records.csv'))
        with open(os.path.join(os.getcwd(), 'cubic-elements-dict.json'),'r') as f:
            d_elements = json.load(f)
        embedding = np.load(os.path.join(os.getcwd(), 'cubic-elements-features.npy'))
        with open(os.path.join(os.getcwd(), 'ternary-dataset-pool.pkl'),'rb') as f:
            d = pickle.load(f)
        
        self.d = d #AUX_DATA
        self.df = df #DATA
        self.d_elements = d_elements
        self.embedding = embedding
                    
        values = df.values
        ids,formulas = [],[]
        for row in values:
            ix,comp,_,symbol = row
            if symbol in cubics_ratio:
                ids.append(ix)
        ids = np.array(ids).astype(str)
        
#         print(ids)
        arr_sp = []
        arr_element = []
        arr_coords = []
        arr_lengths = []
        arr_angles = []
        for idx in ids:
            _,_,sp,e,coords,_,abc,angles=d[idx]
            tmp = np.rint(np.array(coords)/0.125)
            h = np.rint(np.array(angles)/30.0)
            if not np.any(np.isin(tmp, [1.0, 3.0, 5.0, 7.0])):
                arr_sp.append(sp2id[sp]) #not necessary
                arr_element.append([d_elements[key] for key in e])
                arr_coords.append(coords)
                arr_lengths.append(abc[0]) #must change!
                arr_angles.append(angles)
        arr_sp = np.array(arr_sp).astype(int)
        arr_coords = np.stack(arr_coords, axis=0).astype(float)
        m_coord_scales = np.amax(arr_coords, axis=0)/2.0
        arr_lengths = np.stack(arr_lengths, axis=0)#.reshape(len(ids),9).astype(float)
        maximum_lengths = np.amax(arr_lengths, axis=0)/2.0
        arr_angles = np.stack(arr_angles, axis=0)
        maximum_angles = np.amax(arr_angles, axis=0)/2.0
        arr_element = np.stack(arr_element, axis=0).astype(int)

        arr_coords = (arr_coords-m_coord_scales)/m_coord_scales
        arr_lengths = (arr_lengths-maximum_lengths)/maximum_lengths
        arr_angles = (arr_angles-maximum_angles)/maximum_angles
        arr_coords = np.nan_to_num(arr_coords, posinf=0, neginf=0) if np.isnan(arr_coords).any() else arr_coords
        arr_lengths = np.nan_to_num(arr_lengths, posinf=0, neginf=0) if np.isnan(arr_lengths).any() else arr_lengths
        arr_angles = np.nan_to_num(arr_angles, posinf=0, neginf=0) if np.isnan(arr_angles).any() else arr_angles
        
        if np.isin(arr_coords, [np.nan, np.inf, -np.inf]).any():
            warnings.warn(f"There are invalid numbers in normalized coords: {np.isin(arr_coords, [np.nan, np.inf, -np.inf]).any()}")
        else:
            print("No errors in Dataloader...")
#         tmp = np.zeros_like(arr_coords)
#         np.divide((arr_coords-m_coord_scales), m_coord_scales, out=tmp, where=m_coord_scales!=0.)
#         arr_lengths = np.divide((arr_lengths-maximum_lengths), maximum_lengths, out=np.zeros_like(arr_lengths), where=maximum_lengths!=0.)
#         arr_angles = np.divide((arr_angles-maximum_angles), maximum_angles, out=np.zeros_like(arr_angles), where=maximum_angles!=0.)
        
#         print(type(maximum_lengths), type(maximum_angles),type(m_coord_scales),type(prob))
        maximum_angles,m_coord_scales,prob = list(map(lambda inp: torch.from_numpy(inp), (maximum_angles,m_coord_scales,prob) ))
        arr_sp, arr_element,arr_coords,arr_lengths,arr_angles = list(map(lambda inp: torch.from_numpy(inp), (arr_sp,arr_element,arr_coords,arr_lengths,arr_angles) ))
        self.AUX_DATA, self.DATA = AUX_DATA, DATA = (len(d_elements),len(sp2id),maximum_lengths, maximum_angles,m_coord_scales,sp2id,prob), (arr_sp,arr_element,arr_coords,arr_lengths,arr_angles)
        
    def len(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def get(self, idx):
        arr_sp, arr_element, arr_coords, arr_lengths, arr_angles = self.DATA
        return Data(aux_data=self.AUX_DATA, arr_sp=arr_sp[idx], arr_element=arr_element[idx], arr_coords=arr_coords[idx], arr_lengths=arr_lengths[idx], arr_angles=arr_angles[idx]) #PyG type dataset
#         arr_element, arr_coords, arr_lengths, arr_angles = self.DATA
#         DATA = (arr_element[idx], arr_coords[idx], arr_lengths[idx], arr_angles[idx])
#         return Data(aux_data=self.AUX_DATA, data=DATA)
    

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
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
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
                    
        elif self.opt.dataset in ["gandata"]:
            full_dataset = GANData(root_dir)
        elif self.opt.dataset in ["cdvaedata"]:
            full_dataset = CDVAEData(root_dir)
            
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
