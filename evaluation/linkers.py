import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolStandardize
# from src import metrics
import sys, os
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent #main direct
sys.path.append(ROOT_DIR)
from evaluation.delinker_utils import sascorer, calc_SC_RDKit
from evaluation.moses_utils import npscorer
from evaluation.scscore_utils.scscorer import SCScorer
from tqdm import tqdm
from pdb import set_trace
from typing import *
from overloading import overload
import argparse
import pandas as pd
import curtsies.fmtfuncs as cf
from joblib import Parallel, delayed
import psutil
from rdkit.Chem import QED, AllChem, rdMolAlign


parser = argparse.ArgumentParser()
parser.add_argument(
    '--filename', 
    help='Path to your CSV file for', default=None
)
parser.add_argument(
    '--save_result', 
    help='Save the result of evaluation', action='store_true', 
)

def get_valid_as_in_delinker(data, progress=False):
    valid = []
    generator = tqdm(enumerate(data), total=len(data)) if progress else enumerate(data)
    for i, m in generator:
        pred_mol = Chem.MolFromSmiles(m['pred_mol_smi'], sanitize=False)
        true_mol = Chem.MolFromSmiles(m['true_mol_smi'], sanitize=False)
        frag = Chem.MolFromSmiles(m['frag_smi'], sanitize=False)

        pred_mol_frags = Chem.GetMolFrags(pred_mol, asMols=True, sanitizeFrags=False)
        pred_mol_filtered = max(pred_mol_frags, default=pred_mol, key=lambda mol: mol.GetNumAtoms())

        try:
            Chem.SanitizeMol(pred_mol_filtered)
            Chem.SanitizeMol(true_mol)
            Chem.SanitizeMol(frag)
        except:
            continue

        if len(pred_mol_filtered.GetSubstructMatch(frag)) > 0:
            valid.append({
                # 'pred_mol': m['pred_mol'],
                # 'true_mol': m['true_mol'],
                'pred_mol_smi': Chem.MolToSmiles(pred_mol_filtered),
                'true_mol_smi': Chem.MolToSmiles(true_mol),
                'frag_smi': Chem.MolToSmiles(frag)
            })

    return valid


def extract_linker_smiles(molecule, fragments):
    match = molecule.GetSubstructMatch(fragments)
    elinker = Chem.EditableMol(molecule)
    for atom_id in sorted(match, reverse=True):
        elinker.RemoveAtom(atom_id)
    linker = elinker.GetMol()
    Chem.RemoveStereochemistry(linker)
    try:
        linker = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(linker))
    except:
        linker = Chem.MolToSmiles(linker)
    return linker


def compute_and_add_linker_smiles(data, progress=False):
    data_with_linkers = []
    generator = tqdm(data) if progress else data
    for m in generator:
        pred_mol = Chem.MolFromSmiles(m['pred_mol_smi'], sanitize=True)
        true_mol = Chem.MolFromSmiles(m['true_mol_smi'], sanitize=True)
        frag = Chem.MolFromSmiles(m['frag_smi'], sanitize=True)

        pred_linker = extract_linker_smiles(pred_mol, frag)
        true_linker = extract_linker_smiles(true_mol, frag)
        data_with_linkers.append({
            **m,
            'pred_linker': pred_linker,
            'true_linker': true_linker,
        })

    return data_with_linkers


def compute_uniqueness(data, progress=False):
    mol_dictionary = {}
    generator = tqdm(data) if progress else data
    for m in generator:
        frag = m['frag_smi']
        pred_mol = m['pred_mol_smi']
        true_mol = m['true_mol_smi']

        key = f'{true_mol}.{frag}'
        mol_dictionary.setdefault(key, []).append(pred_mol)

    total_mol = 0
    unique_mol = 0
    for molecules in mol_dictionary.values():
        total_mol += len(molecules)
        unique_mol += len(set(molecules))

    return unique_mol / total_mol


def compute_novelty(data, progress=False):
    novel = 0
    # true_linkers = set([m['true_linker'] for m in data])
    true_linkers = set([m['true_mol_smi'] for m in data])

    generator = tqdm(data) if progress else data
    for m in generator:
        # pred_linker = m['pred_linker']
        pred_linker = m['pred_mol_smi']
        if pred_linker in true_linkers:
            continue
        else:
            novel += 1

    return novel / len(data)


def compute_recovery_rate(data, progress=False):
    total = set()
    recovered = set()
    generator = tqdm(data) if progress else data
    for m in generator:
        pred_mol = Chem.MolFromSmiles(m['pred_mol_smi'], sanitize=True)
        Chem.RemoveStereochemistry(pred_mol)
        pred_mol = Chem.MolToSmiles(Chem.RemoveHs(pred_mol))

        true_mol = Chem.MolFromSmiles(m['true_mol_smi'], sanitize=True)
        Chem.RemoveStereochemistry(true_mol)
        true_mol = Chem.MolToSmiles(Chem.RemoveHs(true_mol))

        # true_link = m['true_linker']
        true_link = m['true_mol_smi']
        total.add(f'{true_mol}.{true_link}')
        if pred_mol == true_mol:
            recovered.add(f'{true_mol}.{true_link}')

    return len(recovered) / len(total)


def calc_sa_score_mol(mol):
    if mol is None:
        return None
    return sascorer.calculateScore(mol)

def calc_np_score_mol(mol):
    if mol is None:
        return None
    return npscorer.scoreMol(mol)

def calc_sc_score_mol(mol):
    if mol is None:
        return None
    # root_dir = pathlib.Path(__file__).parent
    WEIGHTS_FILE = os.path.join(ROOT_DIR, 'evaluation', 'scscore_utils', 'scscore_1024uint8_model.ckpt-10654.as_numpy.json.gz')
    model = SCScorer()
    model.restore(WEIGHTS_FILE)
    smi = Chem.MolToSmiles(mol)
    return model.get_score_from_smi(smi)[1]


def check_ring_filter(linker):
    check = True
    # Get linker rings
    ssr = Chem.GetSymmSSSR(linker)
    # Check rings
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    check = False
    return check


def check_pains(mol, pains_smarts):
    for pain in pains_smarts:
        if mol.HasSubstructMatch(pain):
            return False
    return True


def check_qed(mol):
    if mol is None:
        return None
    qed = QED.qed(mol)
    return qed


def check_logP(mol):
    '''
    https://github.com/aspuru-guzik-group/curiosity/blob/779733b2f1878ae9567a60cf3eb9652ab62de197/score_functions.py#L62:~:text=def%20get_logP(mol,MolLogP(mol)
    Calculate logP of a molecule 
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object, for which logP is to calculates
    
    Returns:
    float : logP of molecule (mol)
    '''
    return Chem.Descriptors.MolLogP(mol)


def calc_2d_filters(toks: dict[str], pains_smarts):
    pred_mol = Chem.MolFromSmiles(toks['pred_mol_smi'])
    frag = Chem.MolFromSmiles(toks['frag_smi'])
    # linker = Chem.MolFromSmiles(toks['pred_linker'])

    result = [False, False, False, False, False]
    if len(pred_mol.GetSubstructMatch(frag)) > 0:
        sa_score = False
        ra_score = False
        pains_score = False
        np_score = False
        sc_score = False

    try:
        sa_score = calc_sa_score_mol(pred_mol) < calc_sa_score_mol(frag)
    except Exception as e:
        print(f'Could not compute SA score: {e}')
    try:
        # ra_score = check_ring_filter(linker)
        ra_score = check_ring_filter(pred_mol)
    except Exception as e:
        print(f'Could not compute RA score: {e}')
    try:
        pains_score = check_pains(pred_mol, pains_smarts)
    except Exception as e:
        print(f'Could not compute PAINS score: {e}')
    try:
        np_score = calc_np_score_mol(pred_mol) < calc_np_score_mol(frag)
    except Exception as e:
        print(f'Could not compute NP score: {e}')
    try:
        sc_score = calc_sc_score_mol(pred_mol) < calc_sc_score_mol(frag)
    except Exception as e:
        print(f'Could not compute SC score: {e}')
    try:
        qed_score = check_qed(pred_mol) 
    except Exception as e:
        print(f'Could not compute QED score: {e}')

    result = [sa_score, ra_score, pains_score, np_score, sc_score, qed_score]

    return result


def calc_filters_2d_dataset(data):
    """SA, RA, PAIN, NP, SC, QED utils"""
    with open(os.path.join(ROOT_DIR, 'evaluation', 'delinker_utils', 'wehi_pains.csv'), 'r') as f:
        pains_smarts = [Chem.MolFromSmarts(line[0], mergeHs=True) for line in csv.reader(f)]

    pass_all = pass_SA = pass_RA = pass_PAINS = pass_NP = pass_SC = pass_QED = 0
    # for m in data:
    #     filters_2d = calc_2d_filters(m, pains_smarts)
    #     pass_all += filters_2d[0] & filters_2d[1] & filters_2d[2] & filters_2d[3] & filters_2d[4]
    #     # pass_all += filters_2d[1] & filters_2d[2] 
    #     pass_SA += filters_2d[0]
    #     pass_RA += filters_2d[1]
    #     pass_PAINS += filters_2d[2]
    #     pass_NP += filters_2d[3]
    #     pass_SC += filters_2d[4]
    #     pass_QED+= filters_2d[5]

    with Parallel(n_jobs=psutil.cpu_count(), backend='multiprocessing') as parallel:
        results = parallel(delayed(calc_2d_filters)(toks, pains_smarts) for count, toks in enumerate(data)) #List[List]
    result = np.array(results) #(nmols, nprops)
    pass_SA, pass_RA, pass_PAINS, pass_NP, pass_SC, pass_QED = result.sum(axis=0)
    pass_all = np.all(result, axis=1).sum()

    return pass_all / len(data), pass_SA / len(data), pass_RA / len(data), pass_PAINS / len(data), pass_NP / len(data), pass_SC / len(data), pass_QED / len(data)

def calc_basic_2d_dataset(data):
    try:
        novelty_score = compute_novelty(data)
    except Exception as e:
        print(f'Could not compute novelty score: {e}')
    try:
        uniqueness_score = compute_uniqueness(data)
    except Exception as e:
        print(f'Could not compute uniqueness score: {e}')
    try:
        recovery_score = compute_recovery_rate(data)
    except Exception as e:
        print(f'Could not compute recovery score: {e}')
    return 


def calc_sc_rdkit_full_mol(gen_mol, ref_mol):
    try:
        score = calc_SC_RDKit.calc_SC_RDKit_score(gen_mol, ref_mol)
        return score
    except:
        return -0.5


def sc_rdkit_score(data):
    scores = []
    for m in data:
        # score = calc_sc_rdkit_full_mol(m['pred_mol'], m['true_mol'])
        gen_mol, ref_mol = Chem.MolFromSmiles(m['pred_mol_smi']), Chem.MolFromSmiles(m['true_mol_smi'])

        ### 1 Conformer ###
        # [AllChem.EmbedMolecule(mol, randomSeed=0xf00d) for mol in [gen_mol, ref_mol]] #3D with Hs
        # [AllChem.MMFFOptimizeMolecule(mol) for mol in [gen_mol, ref_mol]]  #optimized geom
        # # conformer = mol.GetConformer()

        # # https://rdkit.readthedocs.io/en/latest/Cookbook.html#:~:text=%23%20Align%20them%20with%20OPEN3DAlign%0ApyO3A%20%3D%20rdMolAlign.GetO3A(mol1%2C%20mol2)%0Ascore%20%3D%20pyO3A.Align()
        # # Align them
        # # rms = rdMolAlign.AlignMol(mol1, mol2)
        # # print(rms)
        # # Align them with OPEN3DAlign
        # pyO3A = Chem.rdMolAlign.GetO3A(gen_mol, ref_mol)
        # score = pyO3A.Align()

        ### Multiple conformers ###
        [AllChem.EmbedMultipleConfs(mol, 
                                        clearConfs=True, 
                                        numConfs=10, 
                                        pruneRmsThresh=1,
                                        numThreads=0) for mol in [gen_mol, ref_mol]] 
        # rmslist = []
        [AllChem.AlignMolConformers(mol, RMSlist=[]) for mol in [gen_mol, ref_mol]]

        score = calc_sc_rdkit_full_mol(gen_mol, ref_mol)

        scores.append(score)

    return np.mean(scores)


##### 3D Metrics ?????#####
def rmsd_frag_mol(gen_mol, ref_mol, start_pt):
    try:
        # Delete linker - Gen mol
        du = Chem.MolFromSmiles('*')
        clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])

        fragmented_mol = get_frags(gen_mol, clean_frag, start_pt)
        if fragmented_mol is not None:
            # Delete linker - Ref mol
            clean_frag_ref = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])
            fragmented_mol_ref = get_frags(ref_mol, clean_frag_ref, start_pt)
            if fragmented_mol_ref is not None:
                # Sanitize
                Chem.SanitizeMol(fragmented_mol)
                Chem.SanitizeMol(fragmented_mol_ref)
                # Align
                pyO3A = rdMolAlign.GetO3A(fragmented_mol, fragmented_mol_ref).Align()
                rms = rdMolAlign.GetBestRMS(fragmented_mol, fragmented_mol_ref)
                return rms #score
    except:
        return 100 # Dummy RMSD

def rmsd_frag_scores(gen_mols):
    return [rmsd_frag_mol(gen_mol, ref_mol, start_pt) for (gen_mol, ref_mol, start_pt) in gen_mols]

def get_delinker_metrics(pred_molecules, true_molecules, true_fragments):
    default_values = {
        'DeLinker/validity': 0,
        'DeLinker/uniqueness': 0,
        'DeLinker/novelty': 0,
        'DeLinker/recovery': 0,
        'DeLinker/2D_filters': 0,
        'DeLinker/2D_filters_SA': 0,
        'DeLinker/2D_filters_RA': 0,
        'DeLinker/2D_filters_PAINS': 0,
        'DeLinker/SC_RDKit': 0,
    }
    if len(pred_molecules) == 0:
        return default_values

    data = []
    for pred_mol, true_mol, true_frag in zip(pred_molecules, true_molecules, true_fragments):
        data.append({
            'pred_mol': pred_mol,
            'true_mol': true_mol,
            'pred_mol_smi': Chem.MolToSmiles(pred_mol),
            'true_mol_smi': Chem.MolToSmiles(true_mol),
            'frag_smi': Chem.MolToSmiles(true_frag)
        })

    # Validity according to DeLinker paper:
    # Passing rdkit.Chem.Sanitize and the biggest fragment contains both fragments
    valid_data = get_valid_as_in_delinker(data)
    validity_as_in_delinker = len(valid_data) / len(data)
    if len(valid_data) == 0:
        return default_values

    # Compute linkers and add to results
    valid_data = compute_and_add_linker_smiles(valid_data)

    # Compute uniqueness
    uniqueness = compute_uniqueness(valid_data)

    # Compute novelty
    novelty = compute_novelty(valid_data)

    # Compute recovered molecules
    recovery_rate = compute_recovery_rate(valid_data)

    # 2D filters
    pass_all, pass_SA, pass_RA, pass_PAINS, pass_NP, pass_SC = calc_filters_2d_dataset(valid_data)

    # 3D Filters
    sc_rdkit = sc_rdkit_score(valid_data)

    return {
        'DeLinker/validity': validity_as_in_delinker,
        'DeLinker/uniqueness': uniqueness,
        'DeLinker/novelty': novelty,
        'DeLinker/recovery': recovery_rate,
        'DeLinker/2D_filters': pass_all,
        'DeLinker/2D_filters_SA': pass_SA,
        'DeLinker/2D_filters_RA': pass_RA,
        'DeLinker/2D_filters_PAINS': pass_PAINS,
        'DeLinker/SC_RDKit': sc_rdkit,
    }

# @overload
def get_delinker_metrics_v2(data: List[Dict]):
    default_values = {
        'DeLinker/validity': 0,
        'DeLinker/uniqueness': 0,
        'DeLinker/novelty': 0,
        'DeLinker/recovery': 0,
        'DeLinker/2D_filters': 0,
        'DeLinker/2D_filters_SA': 0,
        'DeLinker/2D_filters_RA': 0,
        'DeLinker/2D_filters_PAINS': 0,
        'DeLinker/SC_RDKit': 0,
        'DeLinker/2D_filters_NP': 0,
        'DeLinker/2D_filters_SC': 0,
        'DeLinker/2D_filters_RA_QED': 0,

    }
    if len(pred_molecules) == 0:
        return default_values

    # Validity according to DeLinker paper:
    # Passing rdkit.Chem.Sanitize and the biggest fragment contains both fragments
    valid_data = get_valid_as_in_delinker(data)
    print(f'number valid: {len(valid_data)}')
    validity_as_in_delinker = len(valid_data) / len(data)
    if len(valid_data) == 0:
        return default_values

    # Compute linkers and add to results
    valid_data = compute_and_add_linker_smiles(valid_data)

    # Compute uniqueness
    uniqueness = compute_uniqueness(valid_data)

    # Compute novelty
    novelty = compute_novelty(valid_data)

    # Compute recovered molecules
    recovery_rate = compute_recovery_rate(valid_data)

    # 2D filters
    pass_all, pass_SA, pass_RA, pass_PAINS, pass_NP, pass_SC, pass_QED = calc_filters_2d_dataset(valid_data)

    # 3D Filters
    sc_rdkit = sc_rdkit_score(valid_data)

    return {
        'DeLinker/validity': validity_as_in_delinker,
        'DeLinker/uniqueness': uniqueness,
        'DeLinker/novelty': novelty,
        'DeLinker/recovery': recovery_rate,
        'DeLinker/2D_filters': pass_all,
        'DeLinker/2D_filters_SA': pass_SA,
        'DeLinker/2D_filters_RA': pass_RA,
        'DeLinker/2D_filters_PAINS': pass_PAINS,
        'DeLinker/SC_RDKit': sc_rdkit,
        'DeLinker/2D_filters_NP': pass_NP,
        'DeLinker/2D_filters_SC': pass_SC,
        'DeLinker/2D_filters_RA_QED': pass_QED

    }

if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.filename

    if filename is not None:
        df = pd.read_csv(filename)
        true_molecules = df.true_molecules.values.tolist()
        pred_molecules = df.pred_molecules.values.tolist()
        frag_molecules = df.frag_molecules.values.tolist()
    else:
        true_molecules = ['[O]C(=O)C1=[C][C]=C([C]=C1F)[C]C#CC1=[C][C]=C(C(=O)[O])[C]=C1F',
                    '[O]C(=O)C1=C([C]=C([C]=[C]1)C(=O)[N]C1=C(F)C(=C(C(=O)[O])[C]=[C]1)F)F',
                    'O=C([O])C#C[C]1OC(=O)O[C]1C#CC(=O)[O]']
        pred_molecules = ['[O]C(=O)C1=[C][C]=C([C]=C1F)[C]C#CC1=[C][C]=C(C(=O)[O])[C]=C1F',
                    '[O]C(=O)C1=C([C]=C([C]=[C]1)C(=O)[N]C1=C(F)C(=C(C(=O)[O])[C]=[C]1)F)F',
                    'O=C([O])C#C[C]1OC(=O)O[C]1C#CC(=O)[O]']
        frag_molecules = ['[O]C(=O)C1=[C][C]=C([C]=C1F)[C]C#CC1=[C][C]=C(C(=O)[O])[C]=C1F',
                    '[O]C(=O)C1=C([C]=C([C]=[C]1)C(=O)[N]C1=C(F)C(=C(C(=O)[O])[C]=[C]1)F)F',
                    'O=C([O])C#C[C]1OC(=O)O[C]1C#CC(=O)[O]']

    assert len(true_molecules) == len(pred_molecules) and len(true_molecules) == len(frag_molecules), "Input size does not match..."

    data = []
    for true_mol, pred_mol, frag_mol in zip(true_molecules, pred_molecules, frag_molecules):
        data.append({
            'true_mol_smi': true_mol,
            'pred_mol_smi': pred_mol,
            'frag_smi': frag_mol 
                })

    #####DEPRECATED#####
    # valid = []
    # generator = tqdm(enumerate(data), total=len(data)) 
    # for i, m in generator:
    #     pred_mol = Chem.MolFromSmiles(m['pred_mol_smi'], sanitize=False)
    #     pred_mol_frags = Chem.GetMolFrags(pred_mol, asMols=True, sanitizeFrags=False)
    #     pred_mol_filtered = max(pred_mol_frags, default=pred_mol, key=lambda mol: mol.GetNumAtoms())
    #     try:
    #         Chem.SanitizeMol(pred_mol_filtered)
    #     except:
    #         continue
    #     valid.append({
    #         'pred_mol_smi': Chem.MolToSmiles(pred_mol_filtered),
    #                 })

    # pass_all, pass_SA, pass_RA, pass_PAINS, pass_NP, pass_SC = calc_filters_2d_dataset(valid)
    # print(pass_all, pass_SA, pass_RA, pass_PAINS, pass_NP, pass_SC)
    ####################

    #df = pd.DataFrame(columns=['true_mol_smi','pred_mol_smi','frag_smi'])
    #for d in data:
        #result = get_delinker_metrics_v2(d) #dict
        #df_line = pd.DataFrame([result])
    result = get_delinker_metrics_v2(data) #dict
    df = pd.DataFrame([result])
    #df = df.append(df_line)
    filename_root, ext = os.path.splitext(args.filename)
    df.to_csv(filename_root + "_result" + ext)


    #Validity, SA, RA, PAINS, NP, SC (only predictions), Novelty, Uniquenesss, rdkitScore, compute_recovery_rate (predictions, GT, fragments)
    #TODO: Pass real frag & smiles & 3D mols 