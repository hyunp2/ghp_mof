# GHP-MOFassemble

Official repository of the paper "GHP-MOFassemble: Diffusion modeling, high throughput screening, and molecular dynamics for rational discovery of novel metal-organic frameworks for carbon capture at scale".

Authors: Hyun Park, Xiaoli Yan, Ruijie Zhu

## Diffusion Model Accelerates Computational Design of MOF Structures For Carbon Capture

This computational framework enables generation of novel MOF structures with DiffLinker-generated linkers and desinated node-topology pair. 

### Prerequisite
```
pip install -r requirements.txt
```

### Dataset
`data/hMOF_CO2_info.csv` contains the MOF name, MOFid, MOFkey, and isotherm data of 137,652 hypothetical MOF (hMOF) structures. It is a starting point of our framework.

### MOF linker generation
New MOF linkers are generated using a diffusion model named [DiffLinker](https://github.com/igashov/DiffLinker) with the molecular fragments parsed from high-performing MOFs in the hMOF dataset.

Workflow:
1. High-performing MOF structures (with CO2 capacity larger than 2 mmol/g @ 0.1 bar) are selected from the hMOF database
2. The MOFids of high-performing MOFs are parsed to yield the SMILES strings of MOF linkers
3. Matched Molecular Pair Algorithm (MMPA) is used to fragment unique linkers into their corresponding molecular fragments
4. DiffLinker is used to sample new MOF linkers with varying number of sampled atoms
5. The generated linkers are assembled with one of the three pre-selected nodes into new MOFs in pcu topology

### Note
The following files in the *utils* folder were obtained from the [DeLinker](https://github.com/oxpig/DeLinker) package (under the *analysis* or *data* folder):

- prepare_data_from_sdf.py
- fpscores.pkl.gz
- frag_utils.py
- sascorer.py
- wehi_pains.csv

The following files in the *utils* folder were obtained from the [DiffLinker](https://github.com/igashov/DiffLinker) package (under the *data/zinc* folder):

- filter_and_merge.py
- prepare_dataset.py
- prepare_dataset_parallel.py

### License
This framework is released under the Creative Commons (CC) License.