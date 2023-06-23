import os
import moses
import pandas as pd
from tqdm import tqdm

for n_atoms in os.listdir('.'):
	if 'n_atoms' in n_atoms:
		for sys in tqdm([i.split('_')[0] for i in os.listdir(os.path.join(n_atoms,'smiles')) if '.txt' in i]):
			smiles_all = list(pd.read_csv(os.path.join(n_atoms,'info',sys+'_info_sa_sc_filter.csv')).smiles)
			metrics = moses.get_all_metrics(smiles_all)
			os.makedirs(os.path.join(n_atoms,'metrics'),exist_ok=True)
			with open(os.path.join(n_atoms,'metrics',sys+'.txt'),'w+') as f:
				f.write(str(metrics))