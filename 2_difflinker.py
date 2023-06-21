import os
import argparse
import subprocess

nodes = [i.split('_')[1].split('.sdf')[0] for i in os.listdir('data/conformers') if 'conformers' in i]

for n_atoms in range(5,11):
    print(f'Sampling {n_atoms} atoms...')
    for node in nodes:
        if node != 'V':
            print(f'Now on node: {node}')
            OUTPUT_DIR = f'output/n_atoms_{n_atoms}/{node}'
            os.makedirs(OUTPUT_DIR,exist_ok=True)
            subprocess.run(f'python -W ignore utils/difflinker_sample_and_analyze.py --linker_size {n_atoms} --fragments data/fragments_all/{node}/hMOF_frag.sdf --model models/geom_difflinker.ckpt --output {OUTPUT_DIR} --n_samples 20',shell=True)