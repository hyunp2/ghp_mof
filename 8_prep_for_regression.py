import os
import shutil
import pandas as pd

# copy atom_init.json to target dir
shutil.copy(os.path.join('utils','atom_init.json'),os.path.join('MOFs','MOFs_all'))
print('copied atom_init.json ...')

# generate atom_init.json to target dir
names = []
wc = []

for file in os.listdir(os.path.join('MOFs','MOFs_all')):
	if '.cif' in file:
		name = file.split('.')[0]
		names.append(name)
		wc.append(0)
df = pd.DataFrame({'name':names,'wc':wc})
df.to_csv(os.path.join('MOFs','MOFs_all','id_prop.csv'),header=None,index=False)
print('generated id_prop.csv ...')