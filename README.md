# GHP-MOFassemble

Official repository of the paper "GHP-MOFassemble: Diffusion modeling, high throughput screening, and molecular dynamics for rational discovery of novel metal-organic frameworks for carbon capture at scale".

Authors: Hyun Park, Xiaoli Yan, Ruijie Zhu & Eliu Huerta

This novel framework combines generative AI, graph modeling, and high performance computing to enable high-throughput generation of novel pcu MOF structures with DiffLinker-generated linkers and designated nodes.

## Prerequisite

A list of required python packages can be found in `ghp-mof.ipynb`.

## Dataset

`data/hMOF_CO2_info.csv` contains MOF name, MOFid, MOFkey, and isotherm data of 137,652 [hypothetical MOF (hMOF)](https://mof.tech.northwestern.edu/databases) structures.

## Workflow
To see how this framework works, please execute cells in `ghp-mof.ipynb`, which involves the following steps:
1. High-performing MOF structures (with CO2 capacity larger than 2 mmol/g @ 0.1 bar) are selected from the hMOF database
2. The MOFids of these high-performing MOFs are parsed to yield the SMILES strings of MOF linkers
3. [Matched Molecular Pair Algorithm (MMPA)](https://www.rdkit.org/docs/source/rdkit.Chem.rdMMPA.html) implemented in RDKit is used to fragment the unique linkers into their corresponding molecular fragments
4. [DiffLinker](https://github.com/igashov/DiffLinker) is then used to sample new MOF linkers with number of sampled atoms varying from 5 to 10
5. The generated linkers are assembled with one of three pre-selected nodes into MOFs in the pcu topology
6. The modified [CGCNN model](https://iopscience.iop.org/article/10.1088/2632-2153/acd434) proposed in our previous work is used to infer the CO2 capacities of the AI-generated MOF structures

## Example high-performing MOF structures

18 predicted high-performing MOF structures that passed the molecular dynamics simulation density change criteria (<1%) are included in the `high_performing_MOF_cifs` folder.

## License

This computational framework is released under the CC BY 4.0 Licence.
