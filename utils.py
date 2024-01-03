import os
import torch
import random
import numpy as np
import random
import string
from vina import Vina
from openbabel import pybel as pyb
import pandas as pd

import math
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm
import torch
import torch.nn as nn


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
    df = pd.read_csv(file_path)

    df['SMILES'] = df['SMILES'].str.replace('Cl', 'X').replace('Br', 'Y').replace('[nH]', 'Z')

    all_tokens = set()
    for smiles in df['SMILES']:
        tokens = set(smiles) 
        all_tokens.update(tokens)
    return all_tokens

def make_layers(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True, activation=True, is_relu=False):
    layer = []
    layer.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if norm:
        layer.append(nn.InstanceNorm2d(out_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)  

def make_layers_transpose(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True, activation=True, is_relu=False):
    layer = []
    layer.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if norm:
        layer.append(nn.InstanceNorm2d(out_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)



def cos_function_weight(batchSize, imgSize, device):
    weight = torch.ones((imgSize, imgSize)) 
    for i in range(imgSize):
        weight[:, i] = (1. + math.cos(math.pi * i / float(imgSize-1))) * 0.5
    weight = weight.view(1,1,imgSize,imgSize).repeat(batchSize,1,1,1)
    return Variable(weight).cuda(device)

def gaussian_weight(batchSize, imgSize, device=0):
    weight = torch.ones((imgSize, imgSize)) 
    var = (imgSize/4)**2
    for i in range(imgSize):
        weight[:, i] = math.exp(-(float(i))**2/(2*var))
    weight = weight.view(1,1,imgSize,imgSize).repeat(batchSize,1,1,1)
    return Variable(weight).cuda(device)

def padding_smi(smiles_seq):
    seq += [0] * (100 - len(smiles_seq)) 
    padded_seq = seq[:100]  
    return torch.tensor(padded_seq, dtype=torch.long)


def extract_section(smiles_seq, ksizes, strides, rates, padding='same'):

    assert len(smiles_seq.size()) == 4
    assert padding in ['same', 'valid']


    smiles_seq = padding_smi(smiles_seq, ksizes, strides, rates)


    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    section = unfold(smiles_seq)
    return section 

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x
    






def vina_dock(lig,save_path='./log/docked.pdbqt'):
    v = Vina(sf_name='vina')
    v.set_receptor('./data/protein.pdb')
    v.set_ligand_from_file(lig)
    v.compute_vina_maps(center=[15.190, 53.903, 16.917], box_size=[20, 20, 20])
    v.dock(exhaustiveness=32, n_poses=20)
    v.write_poses(save_path, n_poses=1, overwrite=True)
    return  float(os.popen('v.write_poses(save_path, n_poses=1, overwrite=True)')[1].split(' (kcal/mol)')[0].split()[0])