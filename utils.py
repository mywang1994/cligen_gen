import os
import math
import random
from tqdm import tqdm


import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from vina import Vina

from openbabel import pybel as pyb
from openbabel import openbabel

from rdkit import Chem
from rdkit.Chem import rdChemReactions, Descriptors, rdMolTransforms, rdMolDescriptors, rdmolops




def split_molecule(smiles):

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:  
        return []
    
    Chem.SanitizeMol(molecule) 
    synthons = [molecule]
    amide_smarts = "*NC(*)=O"
    triazole_smarts = "*c1cn(*)nn1"

    changed = True
    while changed:
        changed = False
        new_synthons = []
        for synthon in synthons:
            amide_bonds = synthon.GetSubstructMatches(Chem.MolFromSmarts(amide_smarts))
 
            triazole_rings = synthon.GetSubstructMatches(Chem.MolFromSmarts(triazole_smarts))

            if amide_bonds or triazole_rings:
                changed = True
                if amide_bonds:
                    reaction_smarts = "[*:1]C(=O)N[*:2]>>[*:1]C(=O)O[3*].[*:2]N[3*]"
                elif triazole_rings:
                    reaction_smarts = r"[*:1]c1cn([*:2])nn1>>[1*]/C=C(\[2*])[*:1].[2*]/N=N\N([1*])[*:2]"
                reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
                try: 
                    products = reaction.RunReactants((synthon,))
                    for product in products:
                        for mol in product:
                            new_synthons.append(mol)
                except:
                    pass
            else:
                new_synthons.append(synthon)
        synthons = new_synthons

    final_synthons = [Chem.MolToSmiles(mol) for mol in synthons if mol is not None]
    return final_synthons


def synthon_prepare(smi_path):
    all_synthons = set() 
    data=pd.read_csv(smi_path)
    for smiles in tqdm(data['SMILES']):
        synthons = split_molecule(smiles)
        all_synthons.update(synthons)
    pd.DataFrame({'Synthons': list(all_synthons)}).to_csv('synthons.csv', index=False)


def tokened(file_path):
    df = pd.read_csv(file_path,usecols=[0])
    df.columns = ['SMILES']
    df['SMILES'] = df['SMILES'].str.replace('Cl', 'X').replace('Br', 'Y').replace('[nH]', 'Z')
    all_tokens = set()
    for smiles in df['SMILES']:
        tokens = set(smiles) 
        all_tokens.update(tokens)
    
    return all_tokens

def ind2smi (output, idx_to_char):
    # Get the indices of the maximum values along the last dimension
    indices = torch.argmax(output, dim=-1)
    
    # Convert indices to characters
    smiles_list = []
    for seq in indices:
        smiles = ''.join(idx_to_char[idx.item()] for idx in seq)
        smiles_list.append(smiles)
    return smiles_list

def make_layers(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True, activation=True, is_relu=False):
    layer = []
    layer.append(nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if norm:
        layer.append(nn.InstanceNorm1d(out_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)  

def make_layers_transpose(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True, activation=True, is_relu=False):
    layer = []
    layer.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if norm:
        layer.append(nn.InstanceNorm1d(out_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def split_molecule(mol):
    def get_num_atoms(fragment):
        return fragment.GetNumAtoms()

    if mol is None:
        raise ValueError("Invalid molecule input.")

    cut_bonds = []
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cut_bonds.append(bond.GetIdx())

    def atom_diff(frag_atoms):
        return max(frag_atoms) - min(frag_atoms)

    best_fragments = None
    smallest_diff = float('inf')
    best_cut_bonds = None

    # Try to find the best cut
    for i in range(len(cut_bonds)):
        for j in range(i + 1, len(cut_bonds)):
            try:
                frags = Chem.FragmentOnBonds(mol, [cut_bonds[i], cut_bonds[j]], addDummies=True, dummyLabels=[(0, 0), (1, 1)])
                frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)

                if len(frags) == 3:
                    frag_atoms = [get_num_atoms(frag) for frag in frags]
                    diff = atom_diff(frag_atoms)
                    if diff < smallest_diff:
                        smallest_diff = diff
                        best_fragments = frags
                        best_cut_bonds = [cut_bonds[i], cut_bonds[j]]
            except Exception as e:
                print(f"Error processing bonds {cut_bonds[i]} and {cut_bonds[j]}: {e}")
                continue

    if best_fragments is None:
        raise ValueError("Could not find a suitable cut to split the molecule into three parts.")

    # Determine which fragment is left, middle, and right
    atom_indices = [frag.GetAtoms()[0].GetIdx() for frag in best_fragments]
    sorted_indices = sorted(range(len(atom_indices)), key=lambda k: atom_indices[k])

    left = best_fragments[sorted_indices[0]]
    middle = best_fragments[sorted_indices[1]]
    right = best_fragments[sorted_indices[2]]

    return left, middle, right


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, char_to_idx):
        self.smiles_list = smiles_list
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = smiles_to_mol(smiles)
        
        left, middle, right = split_molecule(mol)
        
        left_smiles = Chem.MolToSmiles(left)
        right_smiles = Chem.MolToSmiles(right)
        middle_smiles = Chem.MolToSmiles(middle)

        
        left_indices = smiles_to_indices(left_smiles, self.char_to_idx)
        right_indices = smiles_to_indices(right_smiles, self.char_to_idx)
        middle_indices = smiles_to_indices(middle_smiles, self.char_to_idx)



        left_indices = pad_sequence(left_indices)
        right_indices = pad_sequence(right_indices)
        middle_indices = pad_sequence(middle_indices)
        
        return (torch.tensor(left_indices, dtype=torch.long),
                torch.tensor(right_indices, dtype=torch.long),
                torch.tensor(middle_indices, dtype=torch.long))
 
def pad_sequence(seq):
    seq += [0] * (100 - len(seq))
    return seq[:100]
def smiles_to_indices(smiles, char_to_idx):    
    return [char_to_idx[char] for char in smiles]


def split_input_synthons(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")

    # Find potential cutting points (bonds not in rings)
    cut_bonds = []
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cut_bonds.append(bond.GetIdx())

    if len(cut_bonds) < 2:
        raise ValueError("Not enough non-ring bonds to cut")

    # Randomly select two bonds to cut
    cut1, cut2 = random.sample(cut_bonds, 2)
    while cut2 <= cut1:
        cut1, cut2 = random.sample(cut_bonds, 2)

    # Cut the molecule
    frags = Chem.FragmentOnBonds(mol, [cut1, cut2], addDummies=True, dummyLabels=[(0, 0), (1, 1)])
    frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)
    
    if len(frags) != 3:
        raise ValueError("Failed to split SMILES into three parts")

    return frags

def process_synthons_dataset(smi, char_to_idx, vocab_size):

    try:
        frags = split_input_synthons(smi)
        left, middle, right = [Chem.MolToSmiles(frag) for frag in frags]
    except ValueError as e:
        print(f"Skipping SMILES {smi}: {e}")
        return None, None, None

    left_indices = smiles_to_indices(left, char_to_idx)
    middle_indices = smiles_to_indices(middle, char_to_idx)
    right_indices = smiles_to_indices(right, char_to_idx)

    left_tensor= pad_sequence(left_indices)
    middle_tensor= pad_sequence(middle_indices)
    right_tensor= pad_sequence(right_indices)

    left_tensor = torch.tensor(left_tensor)
    middle_tensor = torch.tensor(middle_tensor)
    right_tensor = torch.tensor(right_tensor)

    left_final = nn.utils.rnn.pad_sequence([left_tensor], batch_first=True, padding_value=vocab_size)
    middle_final = nn.utils.rnn.pad_sequence([middle_tensor], batch_first=True, padding_value=vocab_size)
    right_final = nn.utils.rnn.pad_sequence([right_tensor], batch_first=True, padding_value=vocab_size)

    return left_final, middle_final, right_final





def filter_invalid_molecules(smiles_list):

    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
    return valid_smiles

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

def cos_function_weight(batchSize, imgSize, device):
    weight = torch.ones((imgSize, imgSize)) 
    for i in range(imgSize):
        weight[:, i] = (1. + math.cos(math.pi * i / float(imgSize-1))) * 0.5
    weight = weight.view(1,1,imgSize,imgSize).repeat(batchSize,1,1,1)
    return Variable(weight).cuda(device)

def gaussian_weight(size1, size2, device=0):
    weight = torch.ones((size1, size2)) 
    var = (size2/4)**2
    for i in range(size2):
        weight[:, i] = math.exp(-(float(i))**2/(2*var))
    weight = weight.view(size1,size2)
    return Variable(weight).cuda(device)

def gaussian_bias(size, device=0):
    bias = torch.ones((size)) 
    var = (size/4)**2
    for i in range(size):
        bias[i] = math.exp(-(float(i))**2/(2*var))
    return Variable(bias).cuda(device)



def padding_smi(smiles_seq):
    seq += [0] * (100 - len(smiles_seq)) 
    padded_seq = seq[:100]  
    return torch.tensor(padded_seq, dtype=torch.long)



def assemble_smiles(left, middle, right):

    left_mol = Chem.MolFromSmiles(left)
    middle_mol = Chem.MolFromSmiles(middle)
    right_mol = Chem.MolFromSmiles(right)
    

    if left_mol is None or middle_mol is None or right_mol is None:
        raise ValueError("One of the SMILES strings could not be converted to a molecule.")
    

    left_frag = Chem.MolToSmiles(left_mol, isomericSmiles=True)
    middle_frag = Chem.MolToSmiles(middle_mol, isomericSmiles=True)
    right_frag = Chem.MolToSmiles(right_mol, isomericSmiles=True)
    

    combined_frag = left_frag + '.' + middle_frag + '.' + right_frag
    combined_mol = Chem.MolFromSmiles(combined_frag)
    

    combined_mol = rdmolops.CombineMols(left_mol, middle_mol)
    combined_mol = rdmolops.CombineMols(combined_mol, right_mol)

    assembled_smiles = Chem.MolToSmiles(combined_mol, isomericSmiles=True)
    return assembled_smiles

def assemble_smiles_with_symbols(smiles1, smiles2, smiles3):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    mol3 = Chem.MolFromSmiles(smiles3)
    
    # Combine molecules without modifying them
    combined = Chem.CombineMols(mol1, mol2)
    combined = Chem.CombineMols(combined, mol3)

    # Manual assembly with mark replacement
    combined_smiles = smiles1.replace("[0*]", "[0*]") + smiles2.replace("[0*]", "").replace("[1*]", "") + smiles3.replace("[1*]", "[1*]")
    
    return combined_smiles

def disassemble_smiles_with_symbols(combined_smiles):
    # Use the markers to find the split points
    parts = combined_smiles.split('[0*]')
    left = parts[0] + '[0*]'
    remaining = parts[1].split('[1*]')
    middle = '[0*]' + remaining[0] + '[1*]'
    right = '[1*]' + remaining[1]

    return left, middle, right

def pdb2pdbqt(input_pdb, output_pdbqt):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_pdb)
    obConversion.WriteFile(mol, output_pdbqt)


def vina_dock(lig,save_path='./log/docked.pdbqt'):
    v = Vina(sf_name='vina')
    v.set_receptor('./data/protein.pdbqt')
    mymol = pyb.readstring("smi", lig)
    mymol.make3D()
    mymol.write(format='pdbqt', filename='./log/lig.pdbqt',overwrite=True)
    v.set_ligand_from_file('./log/lig.pdbqt')
    v.compute_vina_maps(center=[15.190, 53.903, 16.917], box_size=[20, 20, 20])
    v.dock(exhaustiveness=32, n_poses=20)
    v.write_poses(save_path, n_poses=1, overwrite=True)
    return  float(os.popen('v.write_poses(save_path, n_poses=1, overwrite=True)')[1].split(' (kcal/mol)')[0].split()[0])