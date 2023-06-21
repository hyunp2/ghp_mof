import os
import shutil
import numpy as np
import warnings
import multiprocessing as mproc
import subprocess
from rdkit import Chem
from rdkit import RDLogger
from subprocess import PIPE
from pymatgen.core.periodic_table import Specie, Element
from pymatgen.core.structure import Molecule

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

os.makedirs('output_for_assembly',exist_ok='True')

nodes = [i.split('_')[1].split('.sdf')[0] for i in os.listdir('data/conformers') if 'conformers' in i]
nodes.remove('V')

NCPUS = int(0.9*os.cpu_count())

def add_Hs(file):
    if not file.startswith('.'):
        mol_num = file.split('_')[1]
        sample_num = file.split('_')[2]
        # generate smile strings
        result = subprocess.run(f'obabel {os.path.join(base_dir,file)} -osmi', shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True).stdout.split()[0]
        # remove square brackets
        result = result.replace('[','').replace(']','')
        smiles.append(result)
        if '@' not in result:
            # generate 3D coordinates (with hydrogens)
            target_xyz_path = os.path.join(xyz_H_dir,node,f'mol_{mol_num}_{sample_num}.xyz')
            subprocess.run(f'obabel -:"{result}" --gen3D -O {target_xyz_path}', shell=True)
            # remove invalid structures
            info = ''.join(open(target_xyz_path).readlines())
            if 'nan' in info:
                os.remove(target_xyz_path)

def generate_dummy_atoms(file):
    if not file.startswith('.'):
        try:
            # get smiles string of molecule
            mol_smiles = subprocess.run(f'obabel {os.path.join(xyz_H_dir,node,file)} -osmi', shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True).stdout.split()[0]
            mol_smiles = mol_smiles.replace('[','').replace(']','')
            aromatic_N = Chem.MolFromSmarts('[n]') # check aromatic N atoms
            carboxylic = Chem.MolFromSmarts('[O;H,-]C=O') # check carboxylic group
            Chem.SetGenericQueriesFromProperties(aromatic_N)
            Chem.SetGenericQueriesFromProperties(carboxylic)
            list_N_containing_rings = Chem.MolFromSmiles(mol_smiles).GetSubstructMatches(aromatic_N)
            list_carboxylic = Chem.MolFromSmiles(mol_smiles).GetSubstructMatches(carboxylic)
            has_heterocyclic_ring = True if len(list_N_containing_rings)>=2 and len(list_carboxylic)==0 else False
            if not has_heterocyclic_ring: # molecule does NOT have heterocyclic rings
                molecule = Molecule.from_file(os.path.join(xyz_H_dir,node,file))
                O_id_remove = [] # ids of O atoms to be removed
                H_id_remove = [] # ids of H atoms attached to O atom in -COOH to be removed
                for site_id in range(len(molecule)):
                    if molecule[site_id].specie.symbol == "C":
                        neighbor_sites = molecule.get_neighbors(molecule[site_id],1.5)
                        neighbor_elements = []
                        for site in neighbor_sites:
                            neighbor_elements.append(site.specie.symbol)
                        if set(neighbor_elements) == set(['C','O','O']):
                            # substitute -C by -At
                            molecule[site_id].species = Element('At')
                            # append O_id_remove
                            for s in neighbor_sites:
                                if s.specie.symbol == 'O':
                                    for i,site in enumerate(molecule.sites):
                                        if list(site.coords) == list(s.coords):
                                            O_id_remove.append(i)
                if len(O_id_remove) == 4:
                    # append H_id_remove
                    for O_id in O_id_remove:
                        neighbor_site_O = molecule.get_neighbors(molecule[O_id],1.2)
                        for i,site in enumerate(molecule.sites):
                            try:
                                if list(site.coords) == list(neighbor_site_O[0].coords) and site.specie.symbol == 'H':
                                    H_id_remove.append(i)
                            except:
                                pass
                    molecule.remove_sites(O_id_remove+H_id_remove)
                    # write to new xyz
                    if len(O_id_remove) == 4 and len(H_id_remove) == 2:
                        molecule.to(os.path.join(xyz_X_dir,node,file))
            else: # molecule HAS heterocyclic strings
                molecule = Molecule.from_file(os.path.join(xyz_H_dir,node,file))
                terminal_N_id = []
                for site_id in range(len(molecule)):
                    # find terminal N atoms
                    if molecule[site_id].specie.symbol == "N":
                        neighbor_sites = molecule.get_neighbors(molecule[site_id],1.4)
                        neighbor_elements = []
                        for site in neighbor_sites:
                            neighbor_elements.append(site.specie.symbol)
                        if set(neighbor_elements) == set(['C','C']):
                            for i,site in enumerate(molecule.sites):
                                if list(site.coords) == list(molecule[site_id].coords):
                                    terminal_N_id.append(i)
                if len(terminal_N_id) == 2:
                    # add dummy atom Fr
                    for id in terminal_N_id: # for each terminal N atom
                        N_site = molecule[id]
                        neighbor_sites = molecule.get_neighbors(N_site,2.87)
                        # find the C atom that is most far away from the N atom within the cutoff radius
                        neighbor_site_distances = []
                        for site in neighbor_sites:
                            neighbor_site_distances.append(molecule[id].distance(site))
                        C_site = neighbor_sites[neighbor_site_distances.index(max(neighbor_site_distances))]
                        # find the vector pointing from the carbon atoms to terminal N atom
                        vector_C_N = N_site.coords - C_site.coords
                        len_vector_C_N = np.linalg.norm(vector_C_N)
                        cos_alpha = vector_C_N[0]/len_vector_C_N
                        cos_beta = vector_C_N[1]/len_vector_C_N
                        cos_gamma = vector_C_N[2]/len_vector_C_N
                        if node == 'CuCu':
                            len_metal_N = 1.85
                        elif node == 'ZnZn':
                            len_metal_N = 2
                        translate_X = len_metal_N*cos_alpha
                        translate_Y = len_metal_N*cos_beta
                        translate_Z = len_metal_N*cos_gamma
                        dummy_X =  N_site.coords[0] + translate_X
                        dummy_Y =  N_site.coords[1] + translate_Y
                        dummy_Z =  N_site.coords[2] + translate_Z
                        molecule.append(Specie('Fr',oxidation_state=None),np.array([dummy_X,dummy_Y,dummy_Z]))
                    #output molecule
                    molecule.to(os.path.join(xyz_X_heterocyclic_dir,node,file))
        except:
           pass
    
for node in nodes:
    for n_atoms in range(5,11):
        print(f'Now on n_atom: {n_atoms} node: {node}')
        base_dir = f'output/n_atoms_{n_atoms}/{node}/'
        xyz_H_dir = f'output_for_assembly/n_atoms_{n_atoms}/xyz_h/'
        os.makedirs(xyz_H_dir,exist_ok=True)
        xyz_X_dir = f'output_for_assembly/n_atoms_{n_atoms}/xyz_X/'
        os.makedirs(xyz_X_dir,exist_ok=True)
        xyz_X_heterocyclic_dir = f'output_for_assembly/n_atoms_{n_atoms}/xyz_X_heterocyclic/'
        os.makedirs(xyz_X_heterocyclic_dir,exist_ok=True)

        if len(os.listdir(base_dir)) > 0: # results not empty
            if not os.path.exists(os.path.join(xyz_H_dir,node)):
                # add hydrogens
                print(f'Adding Hs...')
                smiles = []
                os.makedirs(os.path.join(xyz_H_dir,node),exist_ok=True)
                with mproc.Pool(NCPUS) as mp:
	                mp.map_async(add_Hs, os.listdir(base_dir)).get()
                
            # add dummy atoms
            print(f'Adding dummy atoms...')
            os.makedirs(os.path.join(xyz_X_dir,node),exist_ok=True)
            os.makedirs(os.path.join(xyz_X_heterocyclic_dir,node),exist_ok=True)
            with mproc.Pool(NCPUS) as mp:
               mp.map_async(generate_dummy_atoms, os.listdir(os.path.join(xyz_H_dir,node))).get()