import os
import shutil
import subprocess
import warnings
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
import multiprocessing as mproc

warnings.filterwarnings('ignore')

NCPUS = int(0.9*os.cpu_count())

# gather cat0 MOFs
print(f'Gathering cat0 MOFs...')
os.makedirs(os.path.join('MOFs','MOFs_all_cat0'),exist_ok=True)
def cp(mof):
	shutil.copy(os.path.join('newMOFs',mof,'mofCart.cif'),os.path.join('MOFs','MOFs_all_cat0',mof+'.cif'))

with mproc.Pool(NCPUS) as mp:
	mp.map_async(cp, os.listdir('newMOFs')).get()


os.makedirs(os.path.join('MOFs','temp_ori'),exist_ok=True)
# generate cat1 MOFs
def cat1(mof):
	try:
		cif = Structure.from_file(os.path.join('MOFs','MOFs_all_cat0',mof))
		# output original cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_ori',mof))
		# apply origin shift
		cif.translate_sites(indices=range(len(cif)),vector=(0.5,0.5,0.5),frac_coords=True)
		# output origin shifted cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_shifted_cat1',mof))
		# concatenation the coordinates
		ori_cif = open(os.path.join('MOFs','temp_ori',mof)).readlines()
		new_cif = open(os.path.join('MOFs','temp_shifted_cat1',mof)).readlines()
		new_cif_coords = [i for i in new_cif if len(i.split()) == 7 and '_' not in i]
		# merge coordinates
		cif_cat = ori_cif
		cif_cat += new_cif_coords
		with open(os.path.join('MOFs','MOFs_all_cat1',mof.split('.')[0]+'_cat1.cif'),'w+') as f:
			for line in cif_cat:
				f.write(line)
	except:
		pass
	
print(f'Generating cat1 MOFs...')
os.makedirs(os.path.join('MOFs','MOFs_all_cat1'),exist_ok=True)
os.makedirs(os.path.join('MOFs','temp_shifted_cat1'),exist_ok=True)
with mproc.Pool(NCPUS) as mp:
	mp.map_async(cat1, os.listdir(os.path.join('MOFs','MOFs_all_cat0'))).get()


# generate cat2 MOFs
def cat2(mof):
	try:
		cif = Structure.from_file(os.path.join('MOFs','MOFs_all_cat0',mof))
		# output original cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_ori',mof))
		# apply origin shift 1
		cif.translate_sites(indices=range(len(cif)),vector=(0.333,0.333,0.333),frac_coords=True)
		# output origin shifted cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_shifted_cat2_loc1',mof))
		# apply origin shift 2
		cif.translate_sites(indices=range(len(cif)),vector=(0.333,0.333,0.333),frac_coords=True)
		# output origin shifted cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_shifted_cat2_loc2',mof))
		# concatenation the coordinates
		ori_cif = open(os.path.join('MOFs','temp_ori',mof)).readlines()
		new_cif_1 = open(os.path.join('MOFs','temp_shifted_cat2_loc1',mof)).readlines()
		new_cif_1_coords = [i for i in new_cif_1 if len(i.split()) == 7 and '_' not in i]
		new_cif_2 = open(os.path.join('MOFs','temp_shifted_cat2_loc2',mof)).readlines()
		new_cif_2_coords = [i for i in new_cif_2 if len(i.split()) == 7 and '_' not in i]
		# merge coordinates
		cif_cat = ori_cif
		cif_cat += new_cif_1_coords
		cif_cat += new_cif_2_coords
		with open(os.path.join('MOFs','MOFs_all_cat2',mof.split('.')[0]+'_cat2.cif'),'w+') as f:
			for line in cif_cat:
				f.write(line)
	except:
		pass

print(f'Generating cat2 MOFs...')
os.makedirs(os.path.join('MOFs','MOFs_all_cat2'),exist_ok=True)
os.makedirs(os.path.join('MOFs','temp_shifted_cat2_loc1'),exist_ok=True)
os.makedirs(os.path.join('MOFs','temp_shifted_cat2_loc2'),exist_ok=True)
with mproc.Pool(NCPUS) as mp:
	mp.map_async(cat2, os.listdir(os.path.join('MOFs','MOFs_all_cat0'))).get()


# generate cat3 MOFs
def cat3(mof):
	try:
		cif = Structure.from_file(os.path.join('MOFs','MOFs_all_cat0',mof))
		# output original cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_ori',mof))
		# apply origin shift 1
		cif.translate_sites(indices=range(len(cif)),vector=(0.25,0.25,0.25),frac_coords=True)
		# output origin shifted cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_shifted_cat3_loc1',mof))
		# apply origin shift 2
		cif.translate_sites(indices=range(len(cif)),vector=(0.25,0.25,0.25),frac_coords=True)
		# output origin shifted cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_shifted_cat3_loc2',mof))
		# apply origin shift 3
		cif.translate_sites(indices=range(len(cif)),vector=(0.25,0.25,0.25),frac_coords=True)
		# output origin shifted cif
		CifWriter(cif).write_file(os.path.join('MOFs','temp_shifted_cat3_loc3',mof))

		# concatenation the coordinates
		ori_cif = open(os.path.join('MOFs','temp_ori',mof)).readlines()
		new_cif_1 = open(os.path.join('MOFs','temp_shifted_cat3_loc1',mof)).readlines()
		new_cif_2 = open(os.path.join('MOFs','temp_shifted_cat3_loc2',mof)).readlines()
		new_cif_3 = open(os.path.join('MOFs','temp_shifted_cat3_loc3',mof)).readlines()
		new_cif_1_coords = [i for i in new_cif_1 if len(i.split()) == 7 and '_' not in i]
		new_cif_2_coords = [i for i in new_cif_2 if len(i.split()) == 7 and '_' not in i]
		new_cif_3_coords = [i for i in new_cif_3 if len(i.split()) == 7 and '_' not in i]
		# merge coordinates
		cif_cat = ori_cif
		cif_cat += new_cif_1_coords
		cif_cat += new_cif_2_coords
		cif_cat += new_cif_3_coords
		with open(os.path.join('MOFs','MOFs_all_cat3',mof.split('.')[0]+'_cat3.cif'),'w+') as f:
			for line in cif_cat:
				f.write(line)
	except:
		pass

print(f'Generating cat3 MOFs...')
os.makedirs(os.path.join('MOFs','MOFs_all_cat3'),exist_ok=True)
os.makedirs(os.path.join('MOFs','temp_shifted_cat3_loc1'),exist_ok=True)
os.makedirs(os.path.join('MOFs','temp_shifted_cat3_loc2'),exist_ok=True)
os.makedirs(os.path.join('MOFs','temp_shifted_cat3_loc3'),exist_ok=True)
with mproc.Pool(NCPUS) as mp:
	mp.map_async(cat3, os.listdir(os.path.join('MOFs','MOFs_all_cat0'))).get()

# gather all MOFs
print(f'Gathering all MOFs...')
os.makedirs(os.path.join('MOFs','MOFs_all'))
subprocess.run('find MOFs/MOFs_all_cat* -maxdepth 3 -name *.cif -type f | xargs cp -t MOFs/MOFs_all',shell=True)

# remove temporary dirs
subprocess.run('rm -r MOFs/temp_*',shell=True)