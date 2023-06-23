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

                # generate smiles
                print('Generating SMILES ...')
                os.makedirs(os.path.join(linker_base_dir,n_atoms,'smiles'),exist_ok=True)
                if '.' not in sys:

                    def append_smile(smiles_all, mol):
                        smiles = subprocess.run(f"obabel -ixyz {os.path.join(linker_base_dir,n_atoms,'xyz_h',sys,mol)} -osmi",shell=True, capture_output=True, text=True).stdout
                        smiles_all.append(smiles)

                    manager = Manager()
                    smiles_all = manager.list()

                    with Pool(NCPUS) as p:
                        p.map(functools.partial(append_smile, smiles_all),os.listdir(os.path.join(linker_base_dir,n_atoms,'xyz_h',sys)))
                    with open(os.path.join(linker_base_dir,n_atoms,'smiles',sys+'_smi.txt'),'w+') as f:
                        for smi in smiles_all:
                            f.write(smi)
                            
                # calculate SAscore and SCscore
                smiles = [i.split()[0] for i in open(os.path.join(linker_base_dir,n_atoms,'smiles',sys+'_smi.txt')).readlines()]
                print('Calculating SAscore and SCscore ...')
                df_sa = processMols_sa(smiles)
                df_sc = processMols_sc(smiles)
                df_sa_sc = df_sa.merge(df_sc,how='outer')
                os.makedirs(os.path.join(linker_base_dir,n_atoms,'sc_sa_score'),exist_ok=True)
                df_sa_sc.to_csv(os.path.join(linker_base_dir,n_atoms,'sc_sa_score',f'{sys}.csv'),index=False)

                # merge info
                os.makedirs(os.path.join(linker_base_dir,n_atoms,'info'),exist_ok=True)
                print('Merging info ...')
                sa_scores = []
                sc_scores = []
                lines = open(os.path.join(linker_base_dir,n_atoms,'smiles',sys+'_smi.txt')).readlines()
                df_sa_sc = pd.read_csv(os.path.join(linker_base_dir,n_atoms,'sc_sa_score',sys+'.csv'))
                SMILES = [l.split()[0] for l in lines]
                files = [l.split()[1] for l in lines]
                for SMI in SMILES:
                    sa_score = df_sa_sc[df_sa_sc.smiles==SMI].sa_score.values[0]
                    sa_scores.append(sa_score)
                    sc_score = df_sa_sc[df_sa_sc.smiles==SMI].sc_score.values[0]
                    sc_scores.append(sc_score)
                df = pd.DataFrame({'file':files,'smiles':SMILES,'sa_score':sa_scores,'sc_score':sc_scores})
                df.to_csv(os.path.join(linker_base_dir,n_atoms,'info',sys+'_info.csv'),index=False)

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
