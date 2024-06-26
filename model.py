import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
from utils import *
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd


#Reaction-based combiner 
class combiner(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, synthon_hidden_dim):
        super(combiner, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
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

class pre_dataset_combiner(Dataset):
    def __init__(self, smiles, labels, char_to_idx):
        self.smiles = smiles
        self.labels = labels
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_seq = self.smiles_to_seq(self.smiles[idx])
        padded_seq = self.pad_sequence(smiles_seq)
        label = self.labels[idx]
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

    def smiles_to_seq(self, smile):
        return [self.char_to_idx[char] for char in smile]

    def pad_sequence(self, seq):
        seq += [0] * (100 - len(seq))
        return seq[:100]


#inpainting generator 
class inpainting(nn.Module):
    def __init__(self, vocab_size=39, embedding_dim=64, hidden_dim=512, de_hidden_dim=128,latent_dim=512,device=0, skip=[0,1,2,3,4], attention=[0,1,2,3,4] ):
        super(inpainting, self).__init__()
        self.device=device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.skip = skip
        self.attention = attention
        self.CA = ContextualAttention(ksize=3, stride=1, rate=2, softmax_scale=10, two_input=False, use_cuda=True, device_ids=device)         
        self.encoder_stage1_conv1 = make_layers(100, 64, kernel_size=1, stride=1, padding=0, bias=False, norm=False, activation=True, is_relu=False)
        self.encoder_stage1_conv2 = make_layers(64, 128, kernel_size=1, stride=1, padding=0, bias=False, norm=False, activation=True, is_relu=False)       
        self.encoder_stage2 = nn.Sequential(convolutional_block([128, 64, 64, 256], norm=False),identity_block([256, 64, 64, 256], norm=False),identity_block([256, 64, 64, 256], norm=False))     
        self.encoder_stage3 = nn.Sequential(convolutional_block([256, 128, 128, 512]),identity_block([512, 128, 128, 512]),identity_block([512, 128, 128, 512]),identity_block([512, 128, 128, 512]))        
        self.encoder_stage4 = nn.Sequential(convolutional_block([512, 256, 256, 1024]),identity_block([1024, 256, 256, 1024]),identity_block([1024, 256, 256, 1024]),identity_block([1024, 256, 256, 1024]))
        self.encoder_stage5 = nn.Sequential(convolutional_block([1024, 512, 512, 1024]),identity_block([1024, 512, 512, 1024]),identity_block([1024, 512, 512, 1024]),identity_block([1024, 512, 512, 1024]),identity_block([1024, 512, 512, 1024])
        )
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.rnn1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden_to_mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc0 = nn.Linear(1024 * 512, latent_dim) 
        self.latent_to_hidden = nn.Linear(latent_dim, de_hidden_dim)
        self.deconv1 = make_layers_transpose(latent_dim, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = make_layers_transpose(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = make_layers_transpose(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = make_layers_transpose(64, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.lstm = nn.LSTM(embedding_dim, de_hidden_dim, batch_first=True)
        self.hidden_to_vocab = nn.Linear(de_hidden_dim, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.reduction = nn.Linear(1024, de_hidden_dim)
        self.de_hidden_dim = de_hidden_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8192, 1024)  # Adjust the input size to match the flattened output size
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, vocab_size)  
        self.BCT = BCT_P(device=device)
        self.decoder_stage2_conv1 = make_layers(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, norm=False, activation=True, is_relu=False)
        self.decoder_stage2_conv2 = make_layers(512, 1024, kernel_size=1, stride=1, padding=0, bias=False, norm=False, activation=True, is_relu=False)

        self.feature_out = make_layers(512, 1024, kernel_size=1, stride=1, padding=0, norm=False, activation=True, is_relu=False)

        self.GRB5 = GRB(1024,1)
        self.decoder_stage5 = nn.Sequential(identity_block([1024, 512, 512, 1024], is_relu=True),identity_block([1024, 512, 512, 1024], is_relu=True),make_layers_transpose(1024, 1024, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True, is_relu=True)
        )

        self.linear = nn.Linear(1024, 512)
        self.SHC4 = SHC(1024)
        if 4 in self.skip:
            self.SHC4_mid = SHC(1024)
        self.skip4 = nn.Sequential(
            nn.InstanceNorm1d(1024, affine=True),
            nn.ReLU()
        )
        self.GRB4 = GRB(1024,2)
        self.decoder_stage4 = nn.Sequential(identity_block([1024, 256, 256, 1024], is_relu=True),identity_block([1024, 256, 256, 1024], is_relu=True), identity_block([1024, 256, 256, 1024], is_relu=True), make_layers_transpose(1024, 512, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True, is_relu=True)
        )
        self.SHC3 = SHC(512)
        if 3 in self.skip:
            self.SHC3_mid = SHC(512)
        self.skip3 = nn.Sequential(
            nn.InstanceNorm1d(512, affine=True),
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
            nn.ConvTranspose1d(64, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def encode(self, x):
        shortcut = []

        
        x = self.embedding(x)

  
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
        x, h_n= self.rnn(x)

        shortcut.append(x)
        mean = self.hidden_to_mean(h_n)
        logvar = self.hidden_to_logvar(h_n)

        
        return x, shortcut, mean,logvar

    def decode_smi(self, x, shortcut):

        out = self.GRB5(x)
        
        out = self.decoder_stage5(out)
        if 4 in self.skip:
            out = torch.split(out, 32, dim=2)
            out = list(out)
            if (4 in self.attention): 
                sc_l = [shortcut[4][0]]
                
                sc_r = [shortcut[4][1]]
                
                
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r])
                out[1] = self.skip4(self.SHC4_mid(torch.cat((out[1],sc_m[0]),2), out[1]))
            out[0] = self.skip4(self.SHC4(torch.cat((out[0],shortcut[4][0]),2), shortcut[4][0]))
            out[2] = self.skip4(self.SHC4(torch.cat((out[2],shortcut[4][1]),2), shortcut[4][1]))
            out = torch.cat((out),2)
            
            out = self.GRB4(out)
        out = self.decoder_stage4(out)
        
        if 3 in self.skip:
            out = list(torch.split(out, 64,dim=2))
            if (3 in self.attention): 
                sc_l = [shortcut[3][0]]
                sc_r = [shortcut[3][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip3(self.SHC3_mid(torch.cat((out[1],sc_m[0]),2), out[1]))
            out[0] = self.skip3(self.SHC3(torch.cat((out[0],shortcut[3][0]),2), shortcut[3][0]))
            out[2] = self.skip3(self.SHC3(torch.cat((out[2],shortcut[3][1]),2), shortcut[3][1]))
            out = torch.cat((out),2)
            out = self.GRB3(out)
        out = self.decoder_stage3(out)
        
        if 2 in self.skip:
            out = list(torch.split(out, 128,dim=2))
            if (2 in self.attention): 
                sc_l = [shortcut[2][0]]
                sc_r = [shortcut[2][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip2(self.SHC2_mid(torch.cat((out[1],sc_m[0]),2), out[1]))
            out[0] = self.skip2(self.SHC2(torch.cat((out[0],shortcut[2][0]),2), shortcut[2][0]))
            out[2] = self.skip2(self.SHC2(torch.cat((out[2],shortcut[2][1]),2), shortcut[2][1]))
            out = torch.cat((out),2)
            out = self.GRB2(out)
        out = self.decoder_stage2(out)
        
        if 1 in self.skip:
            out = list(torch.split(out, 256,dim=2))
            if (1 in self.attention): 
                sc_l = [shortcut[1][0]]
                sc_r = [shortcut[1][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip1(self.SHC1_mid(torch.cat((out[1],sc_m[0]),2), out[1]))
            out[0] = self.skip1(self.SHC1(torch.cat((out[0],shortcut[1][0]),2), shortcut[1][0]))
            out[2] = self.skip1(self.SHC1(torch.cat((out[2],shortcut[1][1]),2), shortcut[1][1]))
            out = torch.cat((out),2)
        out = self.decoder_stage1(out)
        
        if 0 in self.skip:
            out = list(torch.split(out, 512,dim=2))
            if (0 in self.attention): 
                sc_l = [shortcut[0][0]]
                sc_r = [shortcut[0][1]]
                sc_m = self.CA(out[0], out[2], out[1], [sc_l, sc_r]) 
                out[1] = self.skip0(self.SHC0_mid(torch.cat((out[1],sc_m[0]),2), out[1]))

            out[0] = self.skip0(self.SHC0(torch.cat((out[0],shortcut[0][0]),2), shortcut[0][0]))
            out[2] = self.skip0(self.SHC0(torch.cat((out[2],shortcut[0][1]),2), shortcut[0][1]))
            out = torch.cat((out),2)
        out = self.decoder_stage0(out) 

        return out

    def forward(self, x1, x2, only_encode=False):     
        shortcut = [[] for i in range(6)]
        x1, shortcut_x1, mean,logvar = self.encode(x1)
        for i in range(6):
            shortcut[i].append(shortcut_x1[i])
        if only_encode:
            return x1

        x2, shortcut_x2, mean,logvar = self.encode(x2)
        for i in range(6):
            shortcut[i].append(shortcut_x2[i])


        out, f1, f2 = self.BCT(x1, x2)

        out = shortcut[5][0] + out + shortcut[5][1]

        out = self.decode_smi(out, shortcut)


        #weight = gaussian_weight(out.size(1),out.size(2))
        #bias = gaussian_bias(out.size(1))
        weight = torch.randn(int(out.size(1)/2),out.size(2)).cuda(self.device)
        bias = torch.randn(int(out.size(1)/2)).cuda(self.device)
        out = F.linear(out,weight,bias)
        out = self.decoder_stage2_conv1(out)
        out = self.decoder_stage2_conv2(out)
        out = out.reshape((out.size(0), -1))
        out = self.fc0(out)  
        out = out.unsqueeze(2)  
        
        h = self.deconv1(out)         
        h = self.deconv2(h) 
        h = self.deconv3(h) 
        h = self.deconv4(h)  
        h = h.permute(0, 2, 1)  
        h_flat = h.contiguous().view(out.size(0), -1) 


 
        h0 = h_flat.view(1, out.size(0), -1) 
        h0=self.reduction(h0)
        c0 = torch.zeros_like(h0)  
        inputs = torch.zeros(out.size(0), 100, dtype=torch.long, device=self.device)
        inputs = self.embedding(inputs)  
        output, _ = self.lstm(inputs, (h0, c0))
        logits = self.hidden_to_vocab(output)  

        return logits, f1, f2





class identity_block(nn.Module):
    def __init__(self, channels, norm=True, is_relu=False):
        super(identity_block, self).__init__()
        
        self.conv1 = make_layers(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv2 = make_layers(channels[1], channels[2], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv3 = make_layers(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=False)
        self.output = nn.ReLU() if is_relu else nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + shortcut

        x = self.output(x)
        return x  
    
class convolutional_block(nn.Module):
    def __init__(self, channels, norm=True, is_relu=False):
        super(convolutional_block, self).__init__()
        
        self.conv1 = make_layers(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv2 = make_layers(channels[1], channels[2], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv3 = make_layers(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=False)
        self.shortcut_path = make_layers(channels[0], channels[3], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=False)
        self.output = nn.ReLU() if is_relu else nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.shortcut_path(shortcut)
        x = x + shortcut
        x = self.output(x)
        return x    
    
class SHC(nn.Module):
    def __init__(self, channel, norm=True):
        super(SHC, self).__init__()

        self.conv1 = make_layers(channel, int(channel/2), kernel_size=1, stride=1, padding=0, norm=norm, activation=True, is_relu=True)
        self.conv2 = make_layers(int(channel/2), int(channel/2), kernel_size=3, stride=1, padding=1, norm=norm, activation=True, is_relu=True)
        self.conv3 = make_layers(int(channel/2), channel, kernel_size=1, stride=1, padding=0, norm=norm, activation=False)

    def forward(self, x, shortcut):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat((x, shortcut),2)
        return x
    
class GRB(nn.Module):
    def __init__(self, channel, dilation, norm=True):
        super(GRB, self).__init__()

        self.path1 = nn.Sequential(
            make_layers(channel, int(channel/2), kernel_size=1, stride=1, padding=0, dilation=dilation, norm=norm, activation=True, is_relu=True),
            make_layers(int(channel/2), channel, kernel_size=1, stride=1, padding=0, dilation=dilation, norm=norm, activation=False)
        )
        self.path2 = nn.Sequential(
            make_layers(channel, int(channel/2), kernel_size=1, stride=1, padding=0, dilation=dilation, norm=norm, activation=True, is_relu=True),
            make_layers(int(channel/2), channel, kernel_size=1, stride=1, padding=0, dilation=dilation, norm=norm, activation=False)
        )
        self.output = nn.ReLU()

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        
        x = x + x1 + x2
        x = self.output(x)
        return x

class BCT_P(nn.Module):
    def __init__(self, size=[512, 4], split=4, pred_step=4, device=0):
        super(BCT_P, self).__init__()

        self.channel, self.width = size
        self.height = 1  

        self.LSTM_encoder_1 = nn.LSTM(512, 512, num_layers=2, batch_first=True)
        self.LSTM_decoder_1 = nn.LSTM(512, 512, num_layers=2, batch_first=True)
        self.LSTM_decoder_2 = nn.LSTM(512, 1024, num_layers=2, batch_first=True)



        self.dec_feat = make_layers(128, 1024, kernel_size=1, stride=1, padding=0, norm=False, activation=True, is_relu=False)
        self.split = split
        self.pred_step = pred_step
        self.device = device
    def forward(self, x1, x2):

        batch_size = x1.size(0)
        init_hidden = (
            Variable(torch.zeros(2, batch_size, 512)).cuda(self.device),
            Variable(torch.zeros(2, batch_size, 512)).cuda(self.device)
        )
        init_hidden_1 = (
            Variable(torch.zeros(2, batch_size, 128)).cuda(self.device),
            Variable(torch.zeros(2, batch_size, 128)).cuda(self.device)
        )

        # Split the input tensors along the channel dimension
        
        split_size = self.channel // self.split

        x1_splits = torch.split(x1, split_size, dim=1)
        x2_splits = torch.split(x2, split_size, dim=1)

        x1_split_reversed = [split.flip(dims=[1]) for split in x1_splits]
        x2_split_reversed = [split.flip(dims=[1]) for split in x2_splits]
        # Encode feature from x2 (left->right)
        en_hidden = init_hidden
        for i in range(self.split):
            split_input = x2_splits[i]

            
            en_out, en_hidden = self.LSTM_encoder_1(split_input, en_hidden)
        hidden_x2 = en_hidden

       

        # Encode feature from x1 (right->left)
        en_hidden = init_hidden
        for i in reversed(range(self.split)):
            split_input = x1_split_reversed[i]
            en_out, en_hidden = self.LSTM_encoder_1(split_input, en_hidden)
        hidden_x1_reversed = en_hidden

        # Decode feature from x1 (left->right)
        de_hidden = init_hidden
        for i in range(self.split):
            split_input = x1_splits[i]
            de_out, de_hidden = self.LSTM_decoder_1(split_input, de_hidden)  
       

        de_out, de_hidden = self.LSTM_decoder_1(de_out, de_hidden)
        de_out=self.dec_feat(de_out) 

        x1_out = x1 + de_out




        de_hidden = hidden_x1_reversed
        for i in reversed(range(self.split)):
            split_input = x2_split_reversed[i]
            de_out, de_hidden = self.LSTM_decoder_1(split_input, de_hidden)
        de_out=self.dec_feat(de_out) 
        x2_out = de_out + x2


        for i in range(self.split):
            de_out, de_hidden = self.LSTM_decoder_1(de_out, de_hidden)  
        x2_out = de_out + x2_out
             
        out = x1_out + x2_out
        


        return out, x1_out, x2_out

class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, softmax_scale=10, two_input=True, weight_func='cos', use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate

        self.softmax_scale = softmax_scale

        self.two_input = two_input
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        if weight_func == 'cos':
            self.weight_func = cos_function_weight
        elif weight_func == 'gaussian':
            self.weight_func = gaussian_weight

    def forward(self, left, right, mid, shortcut, mask=None):
        
        if self.two_input == False:
            
            left = torch.cat((left,right),2)
            for i in range(len(shortcut[0])):
                shortcut[0][i] = torch.cat((shortcut[0][i], shortcut[1][i]),2)
        

        raw_int_ls = list(shortcut[0][0].size())   
        raw_int_ms = list(shortcut[1][0].size())

        if self.two_input:
            raw_int_rs = list(shortcut[1][0].size()) 

        raw_l = [item[0] for item in shortcut]

        
        if self.two_input:
            raw_r = raw_l = [item[1] for item in shortcut]
        
        raw_l = [raw_l[i].view(raw_int_ls[0], raw_int_ls[1], -1) for i in range(len(raw_l))]         
        raw_l_groups = [torch.split(raw_l[i], 1, dim=2) for i in range(len(raw_l))]

        if self.two_input:
            raw_r = [raw_r[i].view(raw_int_rs[0], raw_int_rs[1], -1) for i in range(len(raw_r))]   
            raw_r_groups = [torch.split(raw_r[i], 1, dim=2) for i in range(len(raw_r))]


        left = F.interpolate(left, scale_factor=1, mode='nearest')

        if self.two_input:
            right = F.interpolate(right, scale_factor=1, mode='nearest')
        mid = F.interpolate(mid, scale_factor=1, mode='nearest')
        int_ls = list(left.size())    
        if self.two_input:
            int_rs = list(right.size())
        int_mids = list(mid.size())
        mid_groups = torch.split(mid, 2, dim=2) 
        
     
        left = left.view(int_ls[0], int_ls[1], -1) 
        l_groups = torch.split(left, 2, dim=2)
        if self.two_input:
            right = right.view(int_rs[0], int_rs[1], -1)
            r_groups = torch.split(right, 2, dim=2)
        batch = [i for i in range(raw_int_ls[0])]

        y_l = [[] for i in range(len(shortcut[0]))]
        y_r = [[] for i in range(len(shortcut[0]))]
        y = [[] for i in range(len(shortcut[0]))]


        weight = self.weight_func(raw_int_ls[0], raw_int_ls[2], device=self.device_ids)
        scale = self.softmax_scale   
   
 
        if self.two_input == False:
            r_groups = l_groups


        
        for xi, li, ri, batch_idx in zip(mid_groups, l_groups, r_groups, batch):
        
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda(self.device_ids)

            yi = []
            xi = F.pad(xi, (1, 0)) 
            yi.append(F.conv1d(xi, li, stride=1)) 

            if self.two_input:
                yi.append(F.conv1d(xi, ri, stride=1))
                

            yi = [F.softmax(yi[i]*scale, dim=1) for i in range(len(yi))]


            for i in range(len(shortcut[0])):
                li_center = raw_l_groups[i][batch_idx]
                
                current_dim = yi[0].shape[1]
                target_dim = li_center.shape[1]
                pad_size = target_dim - current_dim


                yi[0] = F.pad(yi[0], (0, 0, 0, pad_size), 'constant', 0)


                if self.two_input:
                    ri_center = raw_r_groups[i][batch_idx]
                y_l[i].append(torch.cat((yi[0], li_center), dim=2))
                if self.two_input:
                    y_r[i].append(torch.cat((yi[0], ri_center), dim=2))

        for i in range(len(shortcut[0])):

            
            y_l[i] = torch.cat(y_l[i], dim=2).contiguous()
            pad_size=raw_int_ms[2]-y_l[i].shape[2]
            y_l[i] = F.pad(y_l[i], (0, pad_size), 'constant', 0) if pad_size > 0 else y_l[i]
            

            if self.two_input:
                y_r[i] = torch.cat(y_r[i], dim=2).contiguous()
                y_r[i] = F.pad(y_r[i], (0, pad_size), 'constant', 0) if pad_size > 0 else y_r[i]
            else:
                y[i]=y_l[i]

        return y
    
def collate_fn(batch, vocab_size):
    lefts, rights, targets = zip(*batch)
    lefts = nn.utils.rnn.pad_sequence(lefts, batch_first=True, padding_value=vocab_size)
    rights = nn.utils.rnn.pad_sequence(rights, batch_first=True, padding_value=vocab_size)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=vocab_size)
    return lefts, rights, targets


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std




def smiles_to_indices(smiles, char_to_idx):
    return [char_to_idx[char] for char in smiles]







