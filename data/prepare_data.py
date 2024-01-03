from utils import split_molecule

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Prepare synthons ')


parser.add_argument('--input_path', type=str,  default='[3*]NC1CCCC(N[3*])CC1',help='input smiles path',)
parser.add_argument('--output_path', type=str, default='./data/synthons/synthons.csv', help='the path of synthons library')




args = parser.parse_args()

all_synthons=set()

data=pd.read_csv(args.input_path)

for smiles in tqdm(data['SMILES']):
    synthons = split_molecule(smiles)
    all_synthons.update(synthons)




pd.DataFrame({'Synthons': list(all_synthons)}).to_csv(args.output_path, index=False)