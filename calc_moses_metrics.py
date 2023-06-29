import os
import moses
import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

for n_atoms in os.listdir('output_for_assembly'):
	if 'n_atoms' in n_atoms:
		for sys in tqdm([i.split('_')[0] for i in os.listdir(os.path.join('output_for_assembly',n_atoms,'smiles')) if '.txt' in i]):
			smiles_all = []
			info = pd.read_csv(os.path.join('output_for_assembly',n_atoms,'info',f'{sys}_info.csv'))
			for file in os.listdir(os.path.join('output_for_assembly',n_atoms,'xyz_X',sys)):
				line = info[info.file.str.contains(file)]
				smiles_all.append(line.smiles.values[0])
			metrics = moses.get_all_metrics(smiles_all)
			os.makedirs(os.path.join('output_for_assembly',n_atoms,'metrics'),exist_ok=True)
			with open(os.path.join('output_for_assembly',n_atoms,'metrics',sys+'.txt'),'w+') as f:
				f.write(str(metrics))