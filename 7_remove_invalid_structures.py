import os
import shutil
from tqdm import tqdm
import multiprocessing as mproc
from pymatgen.core.structure import Structure

os.makedirs(os.path.join('MOFs','MOFs_invalid'),exist_ok=True)

NCPUS = int(0.9*os.cpu_count())

print(f'Number of CPUs: {NCPUS}')

def exam_cif(mof):
    try:
        Structure.from_file(os.path.join(os.path.join('MOFs','MOFs_all'),mof))
    except:
        print(f'removed {mof}')
        shutil.move(os.path.join('MOFs','MOFs_all',mof),os.path.join('MOFs','MOFs_invalid'))

with mproc.Pool(NCPUS) as mp: 
    mp.map_async(exam_cif,os.listdir(os.path.join('MOFs','MOFs_all'))).get()