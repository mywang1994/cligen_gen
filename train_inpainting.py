import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import SMILESDataset, tokened, gaussian_weight
import argparse
import pandas as pd
from model import inpainting
from utils import *



parser = argparse.ArgumentParser(description='Training the inpainting model.....')
parser.add_argument('--mol_p', type=str, help='the path of molecular dataset', default='./data/synthons/data.csv')
parser.add_argument('--embed_dim ', type=int, default=64, help='embedding_dim')
parser.add_argument('--hid_dim ', type=int, default=256, help='hidden_dim')
parser.add_argument('--skip_connection', type=int,help='skip connection', nargs='+', default=[0,1,2,3,4])
parser.add_argument('--attention', type=int,help='attention mechanism', nargs='+', default=[1])
parser.add_argument('--lr',type=float,help='learning rate',default=1e-4)
parser.add_argument('--epoch',type=int,help='training epoches',default=80)


args = parser.parse_args()






#build training dataset
df_m = pd.read_csv(args.mol_p)
df_m['SMILES'] = df_m['SMILES'].str.replace('Cl', 'X').replace('Br', 'Y').replace('[nH]', 'Z')
smiles_db = df_m["SMILES"].tolist()
smiles_db=filter_invalid_molecules(smiles_db)
char_set=tokened(args.mol_p)
char_to_idx = {char: idx for idx, char in enumerate(char_set)}  
vocab_size = len(char_to_idx)


def collate_fn(batch, vocab_size):
    lefts, rights, targets = zip(*batch)
    lefts = nn.utils.rnn.pad_sequence(lefts, batch_first=True, padding_value=vocab_size)
    rights = nn.utils.rnn.pad_sequence(rights, batch_first=True, padding_value=vocab_size)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=vocab_size)
    return lefts, rights, targets



# Training
def train(model, train_loader):
    
    model.train()
    mse = nn.MSELoss(reduction = 'none').cuda(0)
    
    rec_loss = 0
    cons_loss = 0
    optimizer= optim.Adam(model.parameters(), lr=args.lr)
  
    for batch_idx, (l_mol, r_mol, m_mol) in enumerate(train_loader):
                 
        batchSize = l_mol.shape[0]
        mol_len = l_mol.shape[1]            
        l_mol, r_mol, m_mol = Variable(l_mol).cuda(0), Variable(r_mol).cuda(0), Variable(m_mol).cuda(0)
        
        ## Generate mid-molecules
        mol_pred, F_lmol, F_rmol = model(l_mol, r_mol)
          
        # Reconstruction Loss
        weight = gaussian_weight(batchSize, mol_len, device=0)
        mask = weight + weight.flip(3)
        rec_loss = mask * mse(mol_pred, m_mol).mean() * batchSize
        
        #Consistency Loss
        cons_loss = (mse(F_lmol[0], F_rmol[0]) + mse(F_lmol[1], F_rmol[1]) + mse(F_lmol[2], F_rmol[2])).mean() * batchSize        
        
        gen_loss = rec_loss + cons_loss
        rec_loss += rec_loss.data
        cons_loss += cons_loss.data
        

        if (batch_idx % 3) != 0:
            optimizer.zero_grad()
            gen_loss.backward()
            optimizer.step()

        
def main():
    model =inpainting(vocab_size, embedding_dim=args.embed_dim, hidden_dim=args.hid_dim,skip=args.skip_connection, attention=args.attention).cuda(0)
    
    dataset = SMILESDataset(smiles_db, char_to_idx)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=lambda x: collate_fn(x, vocab_size))

    for epoch in range(args.epoch):        
        train(model, train_loader)
    torch.save(model.state_dict(), './data/inpainting.pth')
        

if __name__ == '__main__':
    main()
    
   


   