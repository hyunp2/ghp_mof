import argparse
import os
import time
import numpy as np
from tqdm import tqdm

import torch
from rdkit import Chem

from src import const
from src.datasets import collate_with_fragment_edges, get_dataloader, parse_molecule
from src.lightning import DDPM
from src.linker_size_lightning import SizeClassifier
from src.visualizer import save_xyz_file, visualize_chain
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fragments', action='store', type=str, required=True,
    help='Path to the file with input fragments'
)
parser.add_argument(
    '--model', action='store', type=str, required=True,
    help='Path to the DiffLinker model'
)
parser.add_argument(
    '--linker_size', action='store', type=str, required=True,
    help='Either linker size (int) or path to the GNN for size prediction'
)
parser.add_argument(
    '--output', action='store', type=str, required=False, default='./',
    help='Directory where sampled molecules will be saved'
)
parser.add_argument(
    '--n_samples', action='store', type=int, required=False, default=5,
    help='Number of linkers to generate'
)
parser.add_argument(
    '--n_steps', action='store', type=int, required=False, default=None,
    help='Number of denoising steps'
)
parser.add_argument(
    '--anchors', action='store', type=str, required=False, default=None,
    help='Comma-separated indices of anchor atoms '
         '(according to the order of atoms in the input fragments file, enumeration starts with 1)'
)
parser.add_argument(
    '--samples_dir', action='store', type=str, required=False, default='./',
    help='Directory where sampled molecules ANIMATION will be saved'
)

def read_molecules(path):
    if path.endswith('.pdb'):
        return Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol'):
        return Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol2'):
        return Chem.MolFromMol2File(path, sanitize=False, removeHs=True)
    elif path.endswith('.sdf'):
        return Chem.SDMolSupplier(path, sanitize=False, removeHs=True)
    raise Exception('Unknown file extension')


def generate_animation(ddpm, chain_batch, node_mask,n_mol):

    batch_size = chain_batch.size(1) #Batch size
    batch_indices = torch.arange(batch_size)
    for bi, mi in zip(batch_indices):
        chain = chain_batch[:, bi, :, :] #(FLD)
        name = f'mol_{n_mol}_{mi}'
        #name = f'mol_{n_mol}'
        chain_output = os.path.join(ddpm.samples_dir, name)
        os.makedirs(chain_output, exist_ok=True)

        one_hot = chain[:, :, 3:-1] if ddpm.include_charges else chain[:, :, 3:] #FLD
        positions = chain[:, :, :3] #FL3
        chain_node_mask = torch.cat([node_mask[bi].unsqueeze(0) for _ in range(ddpm.FRAMES)], dim=0) #FL
        names = [f'{name}_{j}' for j in range(ddpm.FRAMES)]

        save_xyz_file(chain_output, one_hot, positions, chain_node_mask, names=names, is_geom=ddpm.is_geom)
        visualize_chain(chain_output, wandb=None, mode=name, is_geom=ddpm.is_geom) #set wandb None for now!

def main(input_path, model, output_dir, n_samples, n_steps, linker_size, anchors):
    print(f'linker_size: {linker_size}')
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    if linker_size.isdigit():
        print(f'Generating linkers with {linker_size} atoms')
        linker_size = int(linker_size)

        def sample_fn(_data):
            return torch.ones(_data['positions'].shape[0], device=device, dtype=const.TORCH_INT) * linker_size
    else:
        print(f'Will generate linkers with sampled numbers of atoms')
        size_nn = SizeClassifier.load_from_checkpoint(linker_size, map_location=device).eval().to(device)

        def sample_fn(_data):
            out, _ = size_nn.forward(_data, return_loss=False)
            probabilities = torch.softmax(out, dim=1)
            distribution = torch.distributions.Categorical(probs=probabilities)
            samples = distribution.sample()
            sizes = []
            for label in samples.detach().cpu().numpy():
                sizes.append(size_nn.linker_id2size[label])
            sizes = torch.tensor(sizes, device=samples.device, dtype=const.TORCH_INT)
            return sizes

    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)

    if n_steps is not None:
        ddpm.edm.T = n_steps #otherwise, ddpm.edm.T = 1000 default

    if ddpm.center_of_mass == 'anchors' and anchors is None:
        print(
            'Please pass anchor atoms indices '
            'or use another DiffLinker model that does not require information about anchors'
        )
        return

    # Reading input fragments
    extension = input_path.split('.')[-1]
    if extension not in ['sdf', 'pdb', 'mol', 'mol2']:
        print('Please upload the file in one of the following formats: .pdb, .sdf, .mol, .mol2')
        return

    try:
        molecules = read_molecules(input_path)
        molecules = [Chem.RemoveAllHs(i) for i in molecules]
        name = '.'.join(input_path.split('/')[-1].split('.')[:-1])
    except Exception as e:
        return f'Could not read the molecule: {e}'
    for n_mol,molecule in enumerate(molecules):
        positions, one_hot, charges = parse_molecule(molecule, is_geom=ddpm.is_geom)
        fragment_mask = np.ones_like(charges)
        linker_mask = np.zeros_like(charges)
        anchor_flags = np.zeros_like(charges)
        if anchors is not None:
            for anchor in anchors.split(','):
                anchor_flags[int(anchor) - 1] = 1

        dataset = [{
            'uuid': '0',
            'name': '0',
            'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
            'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
            'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
            'anchors': torch.tensor(anchor_flags, dtype=const.TORCH_FLOAT, device=device),
            'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
            'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
            'num_atoms': len(positions),
        }] * n_samples
        batch_size = min(n_samples, 64) #TRY to make sure n_samples < 64
        dataloader = get_dataloader(dataset, batch_size=batch_size, collate_fn=collate_with_fragment_edges)

        # Sampling
        for batch_i, data in tqdm(enumerate(dataloader)):
            #print('Sampling only the FINAL output...')
            chain, node_mask = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
            x = chain[0][:, :, :ddpm.n_dims]
            h = chain[0][:, :, ddpm.n_dims:]
            hsize = h.size()

            # Put the molecule back to the initial orientation
            com_mask = data['fragment_mask'] if ddpm.center_of_mass == 'fragments' else data['anchors']
            pos_masked = data['positions'] * com_mask
            N = com_mask.sum(1, keepdims=True)
            mean = torch.sum(pos_masked, dim=1, keepdim=True) / N
            x = x + mean * node_mask

            offset_idx = batch_i * batch_size
            names = [f'mol_{n_mol}_{i}' for i in range(batch_size)]
            save_xyz_file(output_dir, h, x, node_mask, names=names, is_geom=ddpm.is_geom, suffix='')


if __name__ == '__main__':
    start_time = time.time()
    args = parser.parse_args()
    main(
        input_path=args.fragments,
        model=args.model,
        output_dir=args.output,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        linker_size=args.linker_size,
        anchors=args.anchors,
    )
    print("--- %s seconds ---" % (time.time() - start_time))
    # python -W ignore difflinker_sample_and_analyze.py --fragments <YOUR_PATH> --model models/geom_difflinker.ckpt --linker_size models/geom_size_gnn.ckpt --output_dir <YOUR_PATH> --samples_dir <YOUR_PATH>