echo 
echo ----------------------------------------------------------------------------------------
python 1_fragmentation.py

echo X
echo -------------------------------------------------
python 2_generate_frag_sdf.py

echo X
echo --------------------------------------------------
python 3_difflinker.py