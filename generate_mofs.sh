print(f'Step 1 - Fragmenting linkers in high-performing hMOF structures into molecular fragments (~30 mins)')
python 1_fragmentation.py

print(f'Step 2 - Generating molecular fragment comformers')
python 2_generate_frag_sdf.py

print(f'Step 3 - Sampling new MOF linkers using DiffLinker')
python 3_difflinker.py

print(f'Step 4 - Converting to all-atomistic molecules and identify dummy atoms')
python 4_xyz2assemble.py

print('Step 5 - Removing linkers with S, P and I elements')
python 5_remove_undesired_linkers.py

print('Step 6 - Assemblying cat0 MOFs')
python 6_assemble.py

print('Step 7 - Generating catenated MOFs and gathering all MOFs')
python 7_gather_mofs_catenate.py

print('Step 7 - Removing structures that cannot be read by pymatgen')
python 8_remove_invalid_structures.py

print('Step 9 - Preparing necessarly files for making predictions')
python 9_prep_for_regression.py