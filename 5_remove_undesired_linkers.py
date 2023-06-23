import os
import glob
import shutil
import functools
import subprocess
import pandas as pd
from utils.sascorer import *
from utils.scscorer import *
from multiprocessing import Pool, Manager

linker_base_dir = 'output_for_assembly/'

NCPUS = int(0.9*os.cpu_count())


if __name__ == '__main__':
    # remove linkers with S, P and I        
    for n_atom in [5]:
    # change to the line below to reproduce paper result
    #for n_atom in range(5,11):
        os.makedirs(os.path.join(linker_base_dir,f'n_atoms_{n_atom}','linkers_removed'),exist_ok=True)

    print('Removing linkers with S, P and I ...')
    for n_atoms in sorted(glob.glob(os.path.join(linker_base_dir,'*'))):
        print(f'Now working on n_atoms: {n_atoms}')
        if 'n_atoms' in n_atoms:
            for type in glob.glob(os.path.join(n_atoms,'xyz_*')):
                for sys in os.listdir(os.path.join(n_atoms,'xyz_h')):
                    target_dir = os.path.join(n_atoms,'linkers_removed',type.split('/')[-1],sys)
                    os.makedirs(target_dir,exist_ok=True)
                    mol_path = os.path.join(type,sys)
                    for mol in os.listdir(mol_path):
                        data = ''.join(open(os.path.join(mol_path,mol)).readlines())
                        if 'S' in data or 'P' in data or 'I' in data:
                            shutil.move(os.path.join(mol_path,mol),target_dir)

    for n_atoms in sorted(os.listdir(linker_base_dir)):
        if os.path.isdir(os.path.join(linker_base_dir,n_atoms)) and 'n_atoms' in n_atoms:
            for sys in os.listdir(os.path.join(linker_base_dir,n_atoms,'xyz_h')):
                print(f'Now on {n_atoms} - {sys}')
                dir_xyz_X = f'output_for_assembly/{n_atoms}/xyz_X/{sys}'
                dir_xyz_X_heterocyclic = f'output_for_assembly/{n_atoms}/xyz_X_heterocyclic/{sys}'

                print('Copying all linkers to target dir')
                linkers_dir = f'linker_xyz/{sys}'
                linkers_heterocyclic_dir = f'linker_heterocyclic_xyz/{sys}'
                os.makedirs(linkers_dir,exist_ok=True)
                os.makedirs(linkers_heterocyclic_dir,exist_ok=True)
                # copy all carboxylic linkers to target dir and rename
                for file in os.listdir(dir_xyz_X):
                    shutil.copy(os.path.join(dir_xyz_X,file),os.path.join(linkers_dir,file.split('.')[0]+n_atoms+'.xyz'))
                # gather all heterocyclic linkers to target dir and rename
                for file in os.listdir(os.path.join(dir_xyz_X_heterocyclic)):
                    shutil.copy(os.path.join(dir_xyz_X_heterocyclic,file),os.path.join(linkers_heterocyclic_dir,file.split('.')[0]+n_atoms+'.xyz'))
