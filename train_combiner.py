import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from utils import *
from model import combiner,pre_dataset_combiner


parser = argparse.ArgumentParser(description='Traning the reaction-based combiner..........')
parser.add_argument('--mol_p', type=str, help='the path of molecular dataset', default='./data/synthons/data.csv')
parser.add_argument('--syn_p', type=str, help='the path of synthons library',default='./data/synthons/synthons.csv')

parser.add_argument('--num_p', type=int, help='number of positive_samples', default=100)
parser.add_argument('--num_n', type=int, help='number of negative_samples',default=1000)
parser.add_argument('--lr',type=float,help='learning rate',default=1e-4)
parser.add_argument('--epoch',type=int,help='training epoches',default=80)

args = parser.parse_args()


#build molecular dataset
df_m = pd.read_csv(args.mol_p)
df_m['SMILES'] = df_m['SMILES'].str.replace('Cl', 'X').replace('Br', 'Y').replace('[nH]', 'Z')
smiles_db = df_m["SMILES"].tolist()
smiles_db=filter_invalid_molecules(smiles_db)

#build synthon dataset
df_s = pd.read_csv(args.syn_p)
synthon_db = df_s["Synthons"].tolist()  

char_set=tokened(args.mol_p)
char_to_idx = {char: idx for idx, char in enumerate(char_set)}  



def contains_functional_group(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    return mol.HasSubstructMatch(patt)

def decompose_molecule(mol, decomp_rules):
    frags = []
    for rule in decomp_rules:
        rxn = AllChem.ReactionFromSmarts(rule)
        ps = rxn.RunReactants((mol,))
        for products in ps:
            frags.append(products)
    return frags

def calc_tanimoto(mol1, mol2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def apply_reaction(smarts_reaction, reactants):
    rxn = AllChem.ReactionFromSmarts(smarts_reaction)
    products = rxn.RunReactants(reactants)
    return [Chem.MolToSmiles(product[0]) for product in products]


#generate positive_samples and negative_samples
def generate_samples(mol, synthon_db, threshold_positive=0.7, threshold_negative=0.4, n_positive=100,n_negative=1000):
    positive_samples = []
    negative_samples = []
    for synthon in synthon_db:
        synthon_mol = Chem.MolFromSmiles(synthon)
        similarity = calc_tanimoto(mol, synthon_mol)
        if similarity >= threshold_positive:
            positive_samples.append(synthon)
        elif similarity <= threshold_negative:
            negative_samples.append(synthon)
    return random.sample(positive_samples, n_positive), random.sample(negative_samples, n_negative)




def train_combiner(model, data_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader)}')








def main():
    embedding_dim = 128
    hidden_dim = 256
    synthon_hidden_dim = 128
    vocab_size = len(char_to_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = combiner(vocab_size, embedding_dim, hidden_dim, synthon_hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    decomp_rules_amide = ["[*:1]C(=O)N[*:2]>>[*:1]C(=O)O[3*].[*:2]N[3*]"]
    decomp_rules_triazole = ["[*:1]c1cn([*:2])nn1>>[1*]/C=C(\\[2*])[*:1].[2*]/N=N\\N([1*])[*:2]"]
    amide_reaction = '[*:1]C(=O)O[3*].[*:2]N[3*]>>[*:1]C(=O)N[*:2]'
    triazole_reaction = '[1*]/C=C(\\[2*])[*:1].[2*]/N=N\\N([1*])[*:2]>>[*:1]c1cn([*:2])nn1'

    for e in range(args.epoch):  
        selected_smiles = random.sample(smiles_db, 10000)
        training_smiles = []
        training_labels = []
        for smile in selected_smiles:
            mol = Chem.MolFromSmiles(smile)
            if contains_functional_group(mol, "C(=O)N") or contains_functional_group(mol, "c1cnnn1"):
                fragments = decompose_molecule(mol, decomp_rules_amide + decomp_rules_triazole)
                for fragment in fragments:
                    frag_smile = Chem.MolToSmiles(fragment[0])
                    positive_samples, negative_samples = generate_samples(fragment[0], synthon_db)
                    for sample in positive_samples:
                        sample_mol = Chem.MolFromSmiles(sample)
                        if contains_functional_group(fragment[0], "C(=O)O"):
                            combined_smile = apply_reaction(amide_reaction, (fragment[0], sample_mol))
                        elif contains_functional_group(fragment[0], "/C=C/"):
                            combined_smile = apply_reaction(triazole_reaction, (fragment[0], sample_mol))
                        if combined_smile:
                            training_smiles.extend(combined_smile)
                            training_labels.extend([1] * len(combined_smile))
                    for sample in negative_samples:
                        sample_mol = Chem.MolFromSmiles(sample)
                        if contains_functional_group(fragment[0], "C(=O)O"):
                            combined_smile = apply_reaction(amide_reaction, (fragment[0], sample_mol))
                        elif contains_functional_group(fragment[0], "/C=C/"):
                            combined_smile = apply_reaction(triazole_reaction, (fragment[0], sample_mol))
                        if combined_smile:
                            training_smiles.extend(combined_smile)
                            training_labels.extend([0] * len(combined_smile))

        dataset = pre_dataset_combiner(training_smiles, training_labels, char_to_idx)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
        train_combiner(model, data_loader, criterion, optimizer, device, num_epochs=1)  
    torch.save(model.state_dict(), './data/combiner.pth')
if __name__ == "__main__":
    main()
