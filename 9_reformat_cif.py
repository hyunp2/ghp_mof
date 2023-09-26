import os
import glob
import multiprocessing as mproc
from pymatgen.core import IStructure

NCPUS = int(0.9*os.cpu_count())

def fix_label_cif(incif):
    pmg_structure = IStructure.from_file(incif)
    label_dict = {}
    for i in range(0, len(pmg_structure.sites)):
        curr_symbol = pmg_structure.sites[i].specie.symbol
        if curr_symbol in label_dict.keys():
            label_dict[curr_symbol] = label_dict[curr_symbol] + 1
        else:
            label_dict[curr_symbol] = 0
        pmg_structure.sites[i].label = curr_symbol + "%d" % label_dict[curr_symbol]
    outcif=incif
    pmg_structure.to(filename=outcif, fmt="cif")

if __name__ == '__main__':
    with mproc.Pool(NCPUS) as mp:
        mp.map_async(fix_label_cif,glob.glob(os.path.join('MOFs','MOFs_all','*'))).get()