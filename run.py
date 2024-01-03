import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from dl_model import ContextualAttention, identity_block, convolutional_block,BCT_P,GRB,SHC
from utils import vina_dock, tokened, make_layers, make_layers_transpose
from rdkit import Chem, AllChem
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import numpy as np
from openbabel.pybel import *
import argparse


parser = argparse.ArgumentParser(description='ClickGen: Directed Exploration of Synthesizable Chemical Space Leading to the Rapid Synthesis of Novel and Active Lead Compounds via Modular Reactions and Reinforcement Learning')


parser.add_argument('--input', type=str, help='start fragment', default='[3*]NC1CCCC(N[3*])CC1')
parser.add_argument('--syn_p', type=str, default='./data/synthons/synthons.csv', help='the path of synthons library')
parser.add_argument('--protein', type=str, default='./data/parp1.pdb', help='protein, PDB format')

parser.add_argument('--num_sims', type=int, default=1000000, help='Number of simulation steps')
parser.add_argument('--embed_dim ', type=int, default=64, help='embedding_dim')
parser.add_argument('--hid_dim ', type=int, default=256, help='hidden_dim')
parser.add_argument('--syn_dim ', type=int, default=128, help='synthon_hidden_dim')



args = parser.parse_args()








embedding_dim = args.embed_dim 
hidden_dim = args.hid_dim    
synthon_hidden_dim = args.syn_dim
syn_path=args.syn_p
char_set=tokened(syn_path)
char_to_idx = {char: idx for idx, char in enumerate(char_set)}
vocab_size = len(char_to_idx)


class inpainting(nn.Module):
    def __init__(self, device=0, skip=[0,1,2,3,4], attention=[0,1,2,3,4]):
        super(inpainting, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.skip = skip
        self.attention = attention
        self.CA = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True, two_input=False, use_cuda=True, device_ids=device)         
        self.encoder_stage1_conv1 = make_layers(3, 64, kernel_size=4, stride=2, padding=1, bias=False, norm=False, activation=True, is_relu=False)
        self.encoder_stage1_conv2 = make_layers(64, 128, kernel_size=4, stride=2, padding=1, bias=False, norm=False, activation=True, is_relu=False)       
        self.encoder_stage2 = nn.Sequential(convolutional_block([128, 64, 64, 256], norm=False),identity_block([256, 64, 64, 256], norm=False),identity_block([256, 64, 64, 256], norm=False))     
        self.encoder_stage3 = nn.Sequential(convolutional_block([256, 128, 128, 512]),identity_block([512, 128, 128, 512]),identity_block([512, 128, 128, 512]),identity_block([512, 128, 128, 512]))        
        self.encoder_stage4 = nn.Sequential(convolutional_block([512, 256, 256, 1024]),identity_block([1024, 256, 256, 1024]),identity_block([1024, 256, 256, 1024]),identity_block([1024, 256, 256, 1024]))
        self.encoder_stage5 = nn.Sequential(convolutional_block([1024, 512, 512, 1024]),identity_block([1024, 512, 512, 1024]),identity_block([1024, 512, 512, 1024]),identity_block([1024, 512, 512, 1024]),identity_block([1024, 512, 512, 1024])
        )
        self.feature_in = make_layers(1024, 512, kernel_size=1, stride=1, padding=0, norm=False, activation=True, is_relu=False)
        
        self.BCT = BCT_P(device=device)
        
        self.feature_out = make_layers(512, 1024, kernel_size=1, stride=1, padding=0, norm=False, activation=True, is_relu=False)
        
        self.GRB5 = GRB(1024,1)
        self.decoder_stage5 = nn.Sequential(identity_block([1024, 512, 512, 1024], is_relu=True),identity_block([1024, 512, 512, 1024], is_relu=True),make_layers_transpose(1024, 1024, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True, is_relu=True)
        )
        
        self.SHC4 = SHC(1024)
        if 4 in self.skip:
            self.SHC4_mid = SHC(1024)
        self.skip4 = nn.Sequential(
            nn.InstanceNorm2d(1024, affine=True),
            nn.ReLU()
        )
        self.GRB4 = GRB(1024,2)
        self.decoder_stage4 = nn.Sequential(identity_block([1024, 256, 256, 1024], is_relu=True),identity_block([1024, 256, 256, 1024], is_relu=True), identity_block([1024, 256, 256, 1024], is_relu=True), make_layers_transpose(1024, 512, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True, is_relu=True)
        )
        self.SHC3 = SHC(512)
        if 3 in self.skip:
            self.SHC3_mid = SHC(512)
        self.skip3 = nn.Sequential(
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU()
        )
        self.GRB3 = GRB(512,4)
        self.decoder_stage3 = nn.Sequential(
            identity_block([512, 128, 128, 512], is_relu=True),
            identity_block([512, 128, 128, 512], is_relu=True),
            identity_block([512, 128, 128, 512], is_relu=True),
            make_layers_transpose(512, 256, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True, is_relu=True)
        )
        
        self.SHC2 = SHC(256, norm=False)
        if 2 in self.skip:
            self.SHC2_mid = SHC(256, norm=False)
        self.skip2 = nn.ReLU()
        self.GRB2 = GRB(256, 4, norm=False)
        self.decoder_stage2 = nn.Sequential(
            identity_block([256, 64, 64, 256], is_relu=True, norm=False),
            identity_block([256, 64, 64, 256], is_relu=True, norm=False),
            identity_block([256, 64, 64, 256], is_relu=True, norm=False),
            identity_block([256, 64, 64, 256], is_relu=True, norm=False),
            make_layers_transpose(256, 128, kernel_size=4, stride=2, padding=1, bias=False, norm=False, activation=True, is_relu=True)
        )
        
        self.SHC1 = SHC(128, norm=False)
        if 1 in self.skip:
            self.SHC1_mid = SHC(128, norm=False)
        self.skip1 = nn.ReLU()
        self.decoder_stage1 = make_layers_transpose(128, 64, kernel_size=4, stride=2, padding=1, bias=False, norm=False, activation=True, is_relu=True)
        
        self.SHC0 = SHC(64, norm=False)
        if 0 in self.skip:
            self.SHC0_mid = SHC(64, norm=False)
        self.skip0 = nn.ReLU()
        self.decoder_stage0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.fc1=nn.Linear(64, vocab_size)
        
    def encode(self, x):
        shortcut = []
        x = self.encoder_stage1_conv1(x)
        shortcut.append(x)
        x = self.encoder_stage1_conv2(x)
        shortcut.append(x)
        x = self.encoder_stage2(x)
        shortcut.append(x)
        x = self.encoder_stage3(x)
        shortcut.append(x)
        x = self.encoder_stage4(x)
        shortcut.append(x)
        x = self.encoder_stage5(x)
        shortcut.append(x)
        x = self.feature_in(x)
        
        return x, shortcut

    def decode(self, x, shortcut):

        out = self.GRB5(x)
        out = self.decoder_stage5(out)
        
        if 4 in self.skip:
            out = list(torch.split(out, 8, dim=3))
            if (4 in self.attention): 
                sc_l = [shortcut[4][0]]
                sc_r = [shortcut[4][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip4(self.SHC4_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip4(self.SHC4(torch.cat((out[0],shortcut[4][0]),1), shortcut[4][0]))
            out[2] = self.skip4(self.SHC4(torch.cat((out[2],shortcut[4][1]),1), shortcut[4][1]))
            out = torch.cat((out),3)
            out = self.GRB4(out)
        out = self.decoder_stage4(out)
        
        if 3 in self.skip:
            out = list(torch.split(out, 16, dim=3))
            if (3 in self.attention): 
                sc_l = [shortcut[3][0]]
                sc_r = [shortcut[3][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip3(self.SHC3_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip3(self.SHC3(torch.cat((out[0],shortcut[3][0]),1), shortcut[3][0]))
            out[2] = self.skip3(self.SHC3(torch.cat((out[2],shortcut[3][1]),1), shortcut[3][1]))
            out = torch.cat((out),3)
            out = self.GRB3(out)
        out = self.decoder_stage3(out)
        
        if 2 in self.skip:
            out = list(torch.split(out, 32, dim=3))
            if (2 in self.attention): 
                sc_l = [shortcut[2][0]]
                sc_r = [shortcut[2][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip2(self.SHC2_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip2(self.SHC2(torch.cat((out[0],shortcut[2][0]),1), shortcut[2][0]))
            out[2] = self.skip2(self.SHC2(torch.cat((out[2],shortcut[2][1]),1), shortcut[2][1]))
            out = torch.cat((out),3)
            out = self.GRB2(out)
        out = self.decoder_stage2(out)
        
        if 1 in self.skip:
            out = list(torch.split(out, 64, dim=3))
            if (1 in self.attention): 
                sc_l = [shortcut[1][0]]
                sc_r = [shortcut[1][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip1(self.SHC1_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip1(self.SHC1(torch.cat((out[0],shortcut[1][0]),1), shortcut[1][0]))
            out[2] = self.skip1(self.SHC1(torch.cat((out[2],shortcut[1][1]),1), shortcut[1][1]))
            out = torch.cat((out),3)
        out = self.decoder_stage1(out)
        
        if 0 in self.skip:
            out = list(torch.split(out, 128, dim=3))
            if (0 in self.attention): 
                sc_l = [shortcut[0][0]]
                sc_r = [shortcut[0][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip0(self.SHC0_mid(torch.cat((out[1],sc_m[0]),1), out[1]))
            out[0] = self.skip0(self.SHC0(torch.cat((out[0],shortcut[0][0]),1), shortcut[0][0]))
            out[2] = self.skip0(self.SHC0(torch.cat((out[2],shortcut[0][1]),1), shortcut[0][1]))
            out = torch.cat((out),3)
        out = self.fc1(self.decoder_stage0(out))    
        return out

    def forward(self, x1, x2, only_encode=False):     
        shortcut = [[] for i in range(6)]
        x1, shortcut_x1 = self.encode(x1)
        for i in range(6):
            shortcut[i].append(shortcut_x1[i])
        if only_encode:
            return x1

        x2, shortcut_x2 = self.encode(x2)
        for i in range(6):
            shortcut[i].append(shortcut_x2[i])
        f_out, f1, f2 = self.BCT(x1, x2)
        out = self.feature_out(f_out)       
        out = torch.cat((shortcut[5][0],out,shortcut[5][1]),3)
        out = self.decode(out, shortcut)
        
        return out, f_out, f1, f2

def smi_to_seq(smi, char_to_idx):
    smis = smi.replace("Cl", "X").replace("Br", "Y").replace("[nH]", "Z")
    sequence = []
    for char in smis:
        if char in char_to_idx:
            sequence.append(char_to_idx[char])
        else:
            print(f"Unrecognized character in SMILES: {char}")
    return sequence


class Combiner(nn.Module):
    def __init__(self):
        super(Combiner, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # SynthonSelectionNet
        self.fc1 = nn.Linear(hidden_dim, synthon_hidden_dim)
        self.fc2 = nn.Linear(synthon_hidden_dim, synthon_hidden_dim)
        self.fc3 = nn.Linear(synthon_hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        x = h_n[-1]

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
class Pre_Dataset(Dataset):
    def __init__(self, smiles, labels, char_to_idx):
        self.smiles = smiles
        self.labels = labels
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_seq = smiles_to_seq(self.smiles[idx], self.char_to_idx)
        # 进行填充
        padded_seq = self.pad_sequence(smiles_seq)
        label = self.labels[idx]
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

    def pad_sequence(self, seq):
        seq += [0] * (100 - len(seq))  # 假设0是padding值
        return seq[:100]  # 确保序列不超过最大长度





def try_react(syn1, syn2, reaction_smarts):
    mol1 = Chem.MolFromSmiles(syn1)
    mol2 = Chem.MolFromSmiles(syn2)
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    products = rxn.RunReactants((mol1, mol2))
    if products:
        return Chem.MolToSmiles(products[0][0])
    return None

def combine_syns(input,num):
    df = pd.read_csv(syn_path)
    sample_syns = random.sample(df['Synthons'].tolist(), min(num, len(df)))
    
    successful_products = []
    for syn_smis in sample_syns:
        product = try_react(input, syn_smis, '[*:1]C(=O)O[3*].[*:2]N[3*]>>[*:1]C(=O)N[*:2]')
        if product:
            successful_products.append(product)
        else:
            product = try_react(input, syn_smis, '[1*]/C=C(\[2*])[*:1].[2*]/N=N\N([1*])[*:2]>>[*:1]c1cn([*:2])nn1')
            if product:
                successful_products.append(product)
    return successful_products

def predict_syns_0(input):
    smi_list=combine_syns(input,1000)
    model = Combiner()
    model.load_state_dict(torch.load('./data/model/combiner.pth'))
    model.eval()

    test_dataset = Pre_Dataset(smi_to_seq(smi_list), [0] * len(smi_to_seq(smi_list)), char_to_idx)  
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    smiles_probabilities = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            outputs = model(inputs)
            probability = outputs.item() 
            smiles_probabilities.append((smi_list[i], probability))
    return smiles_probabilities

def predict_syns_1(input,num):
    smi_list=combine_syns(input,num)
    model = Combiner()
    model.load_state_dict(torch.load('./data/model/combiner.pth'))
    model.eval()

    test_dataset = Pre_Dataset(smi_to_seq(smi_list), [0] * len(smi_to_seq(smi_list)), char_to_idx)  
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    smiles_probabilities = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            outputs = model(inputs)
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
           syn_pro=predict_syns_0(input,1000)
           smiles_scores = [(smiles, vina_dock(smiles)) for smiles in 
                            [item[0] for item in sorted(syn_pro, key=lambda x: x[1], reverse=True)[:10]]]
           smi_to_sdf(max(smiles_scores, key=lambda item: item[1])[0],'./log/output.sdf')

           DIC.append(smiles_scores)
        else:
           input = Chem.MolToSmiles(input,10) 
           syn_pro=predict_syns_0(input)
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
        new_states = node.state.next_states()
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







