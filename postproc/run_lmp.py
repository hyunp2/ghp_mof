import re, sys, os, io, shutil, warnings, subprocess
import pandas as pd
import numpy as np
#import multiprocessing
from ray.util import multiprocessing
import matplotlib.pyplot as plt

# using the code from rytheranderson/cif2lammps at: https://github.com/rytheranderson/cif2lammps
def preprocessing(cif_name):
    cif_dir = "../newMOFs_20230518"
    cif_path = os.path.join(cif_dir, cif_name)
    
    from main_conversion import single_conversion
    lmp_dir = cif_dir + "_lmp"
    lmp_path = os.path.join(lmp_dir, cif_name.replace(".cif", ""))
    os.makedirs(lmp_path, exist_ok=True)
    from UFF4MOF_construction import UFF4MOF
    print("\n\n")
    print("\n\n")
    print(lmp_path)
    try:
        single_conversion(cif_path, 
            force_field=UFF4MOF, 
            ff_string='UFF4MOF', 
            small_molecule_force_field=None, 
            outdir=lmp_path, 
            charges=False, 
            parallel=False, 
            replication='2x2x2', 
            read_cifs_pymatgen=True, 
            add_molecule=None, 
            small_molecule_file=None)
        in_file_name = [x for x in os.listdir(lmp_path) if x.startswith("in.") and not x.startswith("in.lmp")][0]
        data_file_name = [x for x in os.listdir(lmp_path) if x.startswith("data.") and not x.startswith("data.lmp")][0]
        in_file_rename = "in.lmp"
        data_file_rename = "data.lmp"
        print("Reading data file for element list: " + os.path.join(lmp_path, data_file_name))
        with io.open(os.path.join(lmp_path, data_file_name), "r") as rf:
            df = pd.read_csv(io.StringIO(rf.read().split("Masses")[1].split("Pair Coeffs")[0]), sep=r"\s+", header=None)
            element_list = df[3].to_list()
        
        
        with io.open(os.path.join(lmp_path, in_file_rename), "w") as wf:
            print("Writing input file: " + os.path.join(lmp_path, in_file_rename))
            with io.open(os.path.join(lmp_path, in_file_name), "r") as rf:
                print("Reading original input file: " + os.path.join(lmp_path, in_file_name))
                wf.write(rf.read().replace(data_file_name, data_file_rename) + """

# simulation

fix             fxnpt all npt temp 300.0 300.0 100.0 tri 1.0 1.0 800.0
variable        Nevery equal 1000

thermo          ${Nevery}
thermo_style    custom step cpu dt time temp press pe ke etotal density xlo xhi ylo yhi zlo zhi cella cellb cellc cellalpha cellbeta cellgamma
thermo_modify   flush yes

minimize        1.0e-10 1.0e-10 10000 1000000
reset_timestep  0

dump            trajectAll all custom ${Nevery} dump.lammpstrj.all.0 id type element x y z q
dump_modify     trajectAll element """ + " ".join(element_list) + """


timestep        0.5
run             100000
timestep        1.0
run             2000000
undump          trajectAll
write_restart   relaxing.*.restart
write_data      relaxing.*.data

""")
        os.remove(os.path.join(lmp_path, in_file_name))
        shutil.move(os.path.join(lmp_path, data_file_name), os.path.join(lmp_path, data_file_rename))
        print("Success!!\n\n")
        print("###############################################")
        print("###############################################")
        print("###############################################")
        print("###############################################")
        print("\n\n")
        print("\n\n")

    except Exception as e:
        print(e)
        shutil.rmtree(lmp_path)

def run_lmp_simulation(input_dict):
    Ncpus_per_job = input_dict["Ncpus_per_job"]
    lmp_job_path = input_dict["lmp_job_path"]
    #if "relaxing.2100000.restart" not in os.listdir(lmp_job_path):
    #print("######\nRunning LAMMPS simulation in " + lmp_job_path + " with " + str(Ncpus_per_job) + " CPUs...\n######\n\n")
    # keep the "--bind-to none" in mpirun to ensure concurrent execution!!!
    CompletedProcess = subprocess.run("cd " + lmp_job_path + " && mpirun -np " + str(int(Ncpus_per_job)) + " --bind-to none lmp_mpi -in in.lmp", 
                                        shell=True, capture_output=True)
    with io.open(os.path.join(lmp_job_path, "stdout.txt"), "w", newline="\n") as wf:
        wf.write(str(CompletedProcess.stdout, 'UTF-8'))
    with io.open(os.path.join(lmp_job_path, "stderr.txt"), "w", newline="\n") as wf:
        wf.write(str(CompletedProcess.stderr, 'UTF-8'))
    #print("$$$$$$\nLAMMPS simulation exited in " + lmp_job_path + "\n$$$$$$\n\n")
    return
  
  
  
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    cif_dir = "../newMOFs_20230518"
    lmp_dir = cif_dir + "_lmp"
    cif_names = [x for x in os.listdir(cif_dir) if x.endswith(".cif")]

    # create lammps inputs    
    print("Createing CPU pool with " + str(int(os.cpu_count()*0.9)) + " slots for writing simulation files...\n\n")
    # with multiprocessing.Pool(int(os.cpu_count()*0.8)) as mpool:
    #     mpool.map_async(preprocessing, cif_names).get()
    # with multiprocessing.Pool(1) as mpool:
    #     mpool.map_async(preprocessing, cif_names).get()

    # run lammps
    Ncpus_per_job = 4
    Npool = int(os.cpu_count() / Ncpus_per_job)

    lmp_jobs = [os.path.join(lmp_dir, x) for x in os.listdir(lmp_dir)]
    input_dicts = [{"Ncpus_per_job": Ncpus_per_job, 
                    "lmp_job_path": lmp_job} for lmp_job in lmp_jobs if "relaxing.2100000.restart" not in os.listdir(lmp_job)]
    print("Createing CPU pool with " + str(Npool) + " slots for simulations...\n\n")
    mpool = multiprocessing.Pool(Npool)
    mpool.map_async(run_lmp_simulation, input_dicts).get()
    mpool.close()
    
    
