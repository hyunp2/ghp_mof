echo Step 1 - Fragmenting linkers in high-performing hMOF structures into molecular fragments
echo ----------------------------------------------------------------------------------------
python 1_fragmentation.py

echo Step 2 - Generating molecular fragment comformers
echo -------------------------------------------------
python 2_generate_frag_sdf.py

echo Step 3 - Sampling new MOF linkers using DiffLinker
echo --------------------------------------------------
python 3_difflinker.py

echo Step 4 - Converting to all-atomistic molecules and identifying dummy atoms
echo -----------------------------------------------------------------------
python 4_xyz2assemble.py

echo Step 5 - Removing linkers with S, P and I elements
echo --------------------------------------------------
python 5_remove_undesired_linkers.py

echo Step 6 - Assemblying cat0 MOFs
echo ------------------------------
python 6_assemble.py

echo Step 7 - Generating cat1/cat2/cat3 MOFs and gathering all MOFs
echo --------------------------------------------------------------
python 7_gather_mofs_catenate.py

echo Step 8 - Removing structures that cannot be read by pymatgen
echo ------------------------------------------------------------
python 8_remove_invalid_structures.py

echo Step 9 - Preparing necessarly files for making predictions
echo ----------------------------------------------------------
python 9_prep_for_regression.py