import os
import pandas as pd

for n_atom in [i for i in os.listdir('output_for_assembly') if 'n_atoms' in i]:
    for sys in os.listdir(os.path.join('output_for_assembly',n_atom,'xyz_X')):
        print(f'n_atom:{n_atom} sys:{sys}')
        SMILES = []
        SA_Scores = []
        SC_Scores = []
        info = pd.read_csv(os.path.join('output_for_assembly',n_atom,'info',f'{sys}_info.csv'))
        for file in os.listdir(os.path.join(n_atom,'xyz_X',sys)):
            line = info[info.file.str.contains(file)]
            SMILES.append(line.smiles.values[0])
            SA_Scores.append(line.sa_score.values[0])
            SC_Scores.append(line.sc_score.values[0])
        df = pd.DataFrame({'smiles':SMILES,'sa_score':SA_Scores,'sc_score':SC_Scores})
        df.to_csv(os.path.join('output_for_assembly',n_atom,'sc_sa_score',sys+'_linker.csv'),index=False)
