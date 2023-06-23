import os
import moses
import pandas as pd
from tqdm import tqdm

for n_atoms in [i for i in os.listdir('output_for_assembly')if 'n_atoms' in i]:
		for sys in tqdm([i.split('_')[0] for i in os.listdir(os.path.join('output_for_assembly',n_atoms,'smiles')) if '.txt' in i]):
			smiles_all = list(pd.read_csv(os.path.join(n_atoms,'info',sys+'_info.csv')).smiles)
			metrics = moses.get_all_metrics(smiles_all)
			os.makedirs(os.path.join('output_for_assembly',n_atoms,'metrics'),exist_ok=True)
			with open(os.path.join('output_for_assembly',n_atoms,'metrics',sys+'.txt'),'w+') as f:
				f.write(str(metrics))