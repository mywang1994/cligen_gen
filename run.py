import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils import *
from rdkit import Chem, AllChem
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import numpy as np
from openbabel.pybel import *
import argparse
from model import inpainting, combiner,pre_dataset_combiner



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description='ClickGen: Directed Exploration of Synthesizable Chemical Space Leading to the Rapid Synthesis of Novel and Active Lead Compounds via Modular Reactions and Reinforcement Learning')



parser.add_argument('--syn_p', type=str, default='./data/synthons/synthons.csv', help='the path of synthons library')
parser.add_argument('--input', type=str, help='start fragment', default='[3*]NC1CCCC(N[3*])CC1',required=False)
parser.add_argument('--protein', type=str, default='./data/protein.pdbqt', help='protein, PARP1 PDBQT format')
parser.add_argument('--inpainting', type=str2bool, nargs='?', const=True, default=True, help="Inpainting mode (default: False)")
parser.add_argument('--num_sims', type=int, default=10000, help='Number of simulation steps',required=False)
parser.add_argument('--embed_dim ', type=int, default=64, help='embedding_dim',required=False)
parser.add_argument('--hid_dim ', type=int, default=256, help='hidden_dim',required=False)
parser.add_argument('--syn_dim ', type=int, default=128, help='synthon_hidden_dim',required=False)




args = parser.parse_args()







embedding_dim = args.embed_dim 
hidden_dim = args.hid_dim    
synthon_hidden_dim = args.syn_dim
syn_path=args.syn_p
char_set=tokened(syn_path)
char_to_idx = {char: idx for idx, char in enumerate(char_set)}
vocab_size = len(char_to_idx)
idx_to_char = {v: k for k, v in char_to_idx.items()}


def smi_to_seq(smi, char_to_idx):
    smis = smi.replace("Cl", "X").replace("Br", "Y").replace("[nH]", "Z")
    sequence = []
    for char in smis:
        if char in char_to_idx:
            sequence.append(char_to_idx[char])
        else:
            print(f"Unrecognized character in SMILES: {char}")
    return sequence







def try_react(syn1, syn2, reaction_smarts):
    mol1 = Chem.MolFromSmiles(syn1)
    mol2 = Chem.MolFromSmiles(syn2)
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    products = rxn.RunReactants((mol1, mol2))
    if products:
        return Chem.MolToSmiles(products[0][0])
    return None

def combine_syns(input,num,module):
    df = pd.read_csv(syn_path)
    sample_syns = random.sample(df['Synthons'].tolist(), min(num, len(df)))
    if module:
       processed_smi_list = []
       for smi in sample_syns:
            l, m, r = process_synthons_dataset(smi)
            smi_symbols=assemble_smiles_with_symbols(l, m, r)
            processed_smi_list.append(smi_symbols)
       sample_syns=processed_smi_list 
    successful_products = []
    for syn_smis in sample_syns:
        product = try_react(input, syn_smis,'[*:1]C(=O)O[3*].[*:2]N[3*]>>[*:1]C(=O)N[*:2]')
        if product:
            successful_products.append(product)
        else:
            product = try_react(input, syn_smis,'[1*]/C=C(\[2*])[*:1].[2*]/N=N\N([1*])[*:2]>>[*:1]c1cn([*:2])nn1')
            if product:
                successful_products.append(product)
    return successful_products

def predict_syns(input,module):
    module=args.inpainting
    if module:
        l_m, _m, r_m = process_synthons_dataset(input,char_to_idx,vocab_size)
        m_m,_f1,_f2= inpainting(l_m,r_m)
        l_smi,m_smi,r_smi=ind2smi(l_m)[0],ind2smi(m_m)[0],ind2smi(r_m)[0]
        input = assemble_smiles(l_smi,m_smi,r_smi)
    smi_list=combine_syns(input,1000,module)
    combine_model = combiner()
    combine_model.load_state_dict(torch.load('./data/model/combiner.pth'))
    combine_model.eval()
    inpainting_model = inpainting(vocab_size=vocab_size)
    inpainting_model.load_state_dict(torch.load('./data/model/inpainting.pth'))
    inpainting_model.eval()

    test_dataset = pre_dataset_combiner(smi_to_seq(smi_list), [0] * len(smi_to_seq(smi_list)), char_to_idx)  
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    smiles_probabilities = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if module:
               inputs=disassemble_smiles_with_symbols(inputs)
               inputs,_f1,_f2=inpainting(inputs[0],inputs[2])
               m_smi=ind2smi(inputs)[0]
               inputs = assemble_smiles(inputs[0],m_smi,inputs[0])
            outputs = combine_model(inputs)
            probability = outputs.item() 
            smiles_probabilities.append((smi_list[i], probability))
    return smiles_probabilities

def smi_to_sdf(smi,out_path):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    sdf_filename = out_path  
    writer = Chem.SDWriter(sdf_filename)
    writer.write(mol)
    writer.close()

def roulette(select_list):
    '''
    roulette algorithm
    '''
    sum_val = sum(select_list)
    random_val = random.random()
    probability = 0
    if sum_val != 0:
        for i in range(len(select_list)):
            probability += select_list[i] / sum_val
            if probability >= random_val:
                return i
            else:
                continue
    else:
        return random.choice(range(len(select_list)))


DIC=[]
SOC=[]
N_IDX=0

class State():

    def __init__(self, input, cho=None, sta=[], choices=[],start=True):
        self.input = input
        self.start = start
        self.score=0
        self.states = sta + [self.score]
        self.choices = choices + [self.cho]

    def is_terminal(self,sdf):
        suppl = Chem.SDMolSupplier(sdf)
        for mol in suppl:
            if mol is not None:
                non_hydrogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
                if non_hydrogen_count > 1:
                    return True
                else:
                    return False
    def next_state(self):
        if self.start:
           syn_pro=predict_syns(input,1000)
           smiles_scores = [(smiles, vina_dock(smiles)) for smiles in 
                            [item[0] for item in sorted(syn_pro, key=lambda x: x[1], reverse=True)[:10]]]
           smi_to_sdf(max(smiles_scores, key=lambda item: item[1])[0],'./log/output.sdf')

           DIC.append(smiles_scores)
        else:
           input = Chem.MolToSmiles(input,10) 
           syn_pro=predict_syns(input)
           smiles_scores = [(smiles, vina_dock(smiles)) for smiles in 
                            [item[0] for item in sorted(syn_pro, key=lambda x: x[1], reverse=True)[:1]]]
           smi_to_sdf(max(smiles_scores, key=lambda item: item[1])[0],'./log/output.sdf')
           DIC.append(smiles_scores)
        

class Node():
    def __init__(self, state, parent=None,  reward=0):
        self.visits = 0
        self.reward = reward
        self.state = state
        self.children = []
        self.parent = parent
        self.longest_path = 0


    def add_child(self, child_state, node_id):
        child = Node(child_state, node_id=node_id, parent=self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self, num_moves_lambda):
        num_moves = len(DIC)
        if num_moves_lambda != None:
            num_moves = num_moves_lambda(self)
        if len(self.children) == num_moves:
            return True
        return False




        


            

def UCTSEARCH(budget, root, start_score=0, num_moves_lambda=None):
    # Begin the MCTS
    for iter in range(int(budget)):
        front = TREEPOLICY(root, start_score, num_moves_lambda)
        BACKUP2(front)


def TREEPOLICY(node, start_score):
    # Choose whether to expand the node based on the status of the current node
    while node.state.is_terminal('./log/output.sdf') == False:
        if len(node.children) == 0:
            node = EXPAND(node, start_score)
        else:
            node = BESTCHILD(node, start_score)
    return node


def EXPAND(node, start_score):

    # Get the children of a node and add them to the tree
    if node.state:
        for nextmove in node.state.h1s_avail:
            next = State(state_type=1, sdf=rf'{node.state.sdf}', h1=nextmove, sta=node.state.states)
            N_IDX += 1
            node.add_child(next, node_id=N_IDX, bestscore=start_score)
        return node.children[-1]
    else:
        new_states = node.state.next_state()
        if len(new_states) == 0:
            return node
        else:
            scores = []
            for nextmove in new_states:
                next = State(state_type=0, Frag_Deg=nextmove[:2], sco=nextmove[2], sta=node.state.states)
                N_IDX += 1
                best_score = min(start_score, nextmove[2])
                scores.append(abs(nextmove[4]))
                node.add_child(next, node_id=N_IDX, bestscore=best_score, qed=abs(nextmove[4]))
            return node.children[roulette(scores)]
        

def BESTCHILD(node, start_score):

    # Select child nodes based on the node's UCB
    scores = []
    for c in node.children:

        exploit = start_score - c.best_score
        explore = math.sqrt(2.0 * math.log(node.visits + 0.000001) / float(c.visits + 0.000001))

        score = exploit + 1 / (2 * math.sqrt(2.0)) * explore
        scores.append(score)
    if True:
        idx = roulette(scores)

    else:
        idx = random.choice(range(len(scores)))
    return node.children[idx]


def DEFAULTPOLICY(node):
    state = node.state
    num_states = 0

    while state.is_terminal('./log/output.sdf') == False:
        state = state.next_state()
        num_states += 1
    if state.type == 1:
        if num_states != 0:
            num_states -= 1
    num_nodes = len(state.states) - num_states
    print(state.type)
    return state.states, num_nodes, num_states


def BACKUP2(node):

    parent_node = node
    while parent_node != None:
        parent_node.visits += 1
        if len(parent_node.children) == 0:
            x = parent_node
            parent_node = node.parent
            son_node = x
        else:
            if parent_node.best_score > son_node.best_score:
                parent_node.best_score = son_node.best_score
            x = parent_node
            parent_node = parent_node.parent
            son_node = x


def BACKUP(node, states, num_nodes):
    i = 1
    if node.longest_path == 0:
        node.longest_path = len(states)
    while node != None:
        node.visits += 1
        best_score = min(states[num_nodes - i:])
        i += 1
        if best_score < node.best_score:
            node.best_score = best_score
            reward = max(best_score, 0)
        else:
            reward = 0
        if best_score < np.mean(DIC[:,0]):
            SOC.append(best_score)
        node.reward += reward
        node = node.parent
    return

if __name__ == "__main__":

    
    frag = args.input
    score = vina_dock(frag)
    ipts = [args.input, score]
    current_node = Node(State(), reward=ipts[1])
    result = UCTSEARCH(args.num_sims, current_node, start_score=ipts[1])







