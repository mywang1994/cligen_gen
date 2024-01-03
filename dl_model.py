import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
from utils import make_layers, cos_function_weight, gaussian_weight, padding_smi, extract_section, reduce_sum





class identity_block(nn.Module):
    def __init__(self, channels, norm=True, is_relu=False):
        super(identity_block, self).__init__()
        
        self.conv1 = make_layers(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv2 = make_layers(channels[1], channels[2], kernel_size=3, stride=1, padding=1, bias=False, norm=norm, activation=True, is_relu=is_relu)
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
        self.conv2 = make_layers(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False, norm=norm, activation=True, is_relu=is_relu)
        self.conv3 = make_layers(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False, norm=norm, activation=False)
        self.shortcut_path = make_layers(channels[0], channels[3], kernel_size=1, stride=2, padding=0, bias=False, norm=norm, activation=False)
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

        self.conv1 = make_layers(channel*2, int(channel/2), kernel_size=1, stride=1, padding=0, norm=norm, activation=True, is_relu=True)
        self.conv2 = make_layers(int(channel/2), int(channel/2), kernel_size=3, stride=1, padding=1, norm=norm, activation=True, is_relu=True)
        self.conv3 = make_layers(int(channel/2), channel, kernel_size=1, stride=1, padding=0, norm=norm, activation=False)

    def forward(self, x, shortcut):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + shortcut
        return x
    
class GRB(nn.Module):
    def __init__(self, channel, dilation, norm=True):
        super(GRB, self).__init__()

        self.path1 = nn.Sequential(
            make_layers(channel, channel, kernel_size=(3,1), stride=1, padding=(dilation,0), dilation=dilation, norm=norm, activation=True, is_relu=True),
            make_layers(channel, channel, kernel_size=(1,7), stride=1, padding=(0,3*dilation), dilation=dilation, norm=norm, activation=False)
        )
        self.path2 = nn.Sequential(
            make_layers(channel, channel, kernel_size=(1,7), stride=1, padding=(0,3*dilation), dilation=dilation, norm=norm, activation=True, is_relu=True),
            make_layers(channel, channel, kernel_size=(3,1), stride=1, padding=(dilation,0), dilation=dilation, norm=norm, activation=False)
        )
        self.output = nn.ReLU()

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = x + x1 + x2
        x = self.output(x)
        return x

class BCT_P(nn.Module):
    def __init__(self, size=[512,4,4], split=4, pred_step=4, device=0):

        super(BCT_P, self).__init__()

        self.channel, self.height, self.width = size
        self.lstm_size = self.channel * self.height * int(self.width/split)
        self.LSTM_encoder = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.LSTM_decoder = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.output = make_layers(2*self.channel, self.channel, kernel_size=1, stride=1, padding=0, norm=False, activation=True, is_relu=False)
        self.split = split
        self.pred_step = pred_step
        self.device = device

    def forward(self, x1, x2):
        init_hidden = (Variable(torch.zeros(2, x1.shape[0], self.lstm_size)).cuda(self.device), Variable(torch.zeros(2,x1.shape[0], self.lstm_size)).cuda(self.device))
        x1_out = x1
        x2_out = x2
        x1_split = torch.stack(torch.split(x1, int(self.width/self.split), dim=3)).view(self.split, -1, 1, self.lstm_size)
        x1_split_reversed = torch.stack(torch.split(x1, int(self.width/self.split), dim=3)).flip(dims=[4]).view(self.split, -1, 1, self.lstm_size)
        x2_split = torch.stack(torch.split(x2, int(self.width/self.split), dim=3)).view(self.split, -1, 1, self.lstm_size)
        x2_split_reversed = torch.stack(torch.split(x2, int(self.width/self.split), dim=3)).flip(dims=[4]).view(self.split, -1, 1, self.lstm_size)
        
        # Encode feature from x2 (left->right)
        en_hidden = init_hidden
        for i in range(self.split):
            en_out, en_hidden = self.LSTM_encoder(x2_split[i], en_hidden)
        hidden_x2 = en_hidden

        # Encode feature from x1 (right->left)
        en_hidden = init_hidden
        for i in reversed(range(self.split)):
            en_out, en_hidden = self.LSTM_encoder(x1_split_reversed[i], en_hidden)
        hidden_x1_reversed = en_hidden
        
        # Decode feature from x1 (left->right)
        de_hidden = hidden_x2
        for i in range(self.split):
            de_out, de_hidden = self.LSTM_decoder(x1_split[i], de_hidden) # f_1^2 ~ f_1^5
        x1_out = torch.cat((x1_out, de_out.view(-1, self.channel, self.height, int(self.width/self.split))), 3)
        for i in range(self.pred_step + self.split - 1):
            de_out, de_hidden = self.LSTM_decoder(de_out, de_hidden) # f_1^6 ~ f_1^12
            x1_out = torch.cat((x1_out, de_out.view(-1, self.channel, self.height, int(self.width/self.split))), 3)
        
        # Decode feature from x2 (right->left)
        de_hidden = hidden_x1_reversed
        for i in reversed(range(self.split)):
            de_out, de_hidden = self.LSTM_decoder(x2_split_reversed[i], de_hidden) # f_2^11' ~ f_2^8'
        x2_out = torch.cat((de_out.view(-1, self.channel, self.height, int(self.width/self.split)).flip(dims=[3]), x2_out), 3)
        for i in range(self.pred_step + self.split - 1):
            de_out, de_hidden = self.LSTM_decoder(de_out, de_hidden) # f_2^7' ~ # f_2^1'
            x2_out = torch.cat((de_out.view(-1, self.channel, self.height, int(self.width/self.split)).flip(dims=[3]), x2_out), 3)

        x1_out = (x1_out[:,:,:,:self.width], x1_out[:,:,:,self.width:-self.width], x1_out[:,:,:,-self.width:])
        x2_out = (x2_out[:,:,:,:self.width], x2_out[:,:,:,self.width:-self.width], x2_out[:,:,:,-self.width:])
        out = self.output(torch.cat((x1_out[1], x2_out[1]),1))
        
        return out, x1_out, x2_out
    
class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10, fuse=False, two_input=True, weight_func='cos', use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.two_input = two_input
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        if weight_func == 'cos':
            self.weight_func = cos_function_weight
        elif weight_func == 'gaussian':
            self.weight_func = gaussian_weight

    def forward(self, left, right, mid, shortcut, mask=None):
        
        if self.two_input == False:
            left = torch.cat((left,right),3)
            for i in range(len(shortcut[0])):
                shortcut[0][i] = torch.cat((shortcut[0][i], shortcut[1][i]),3)
        
  
        raw_int_ls = list(shortcut[0][0].size())   
        raw_int_ms = list(shortcut[1][0].size())   
        if self.two_input:
            raw_int_rs = list(shortcut[1][0].size()) 

        kernel = 2 * self.rate
        raw_l = [extract_section(shortcut[0][i], ksizes=[kernel, kernel], strides=[self.rate*self.stride, self.rate*self.stride], rates=[1, 1]) for i in range(len(shortcut[0]))] # [N, C*k*k, L]
        if self.two_input:
            raw_r = [extract_section(shortcut[1][i], ksizes=[kernel, kernel], strides=[self.rate*self.stride, self.rate*self.stride], rates=[1, 1]) for i in range(len(shortcut[1]))] # [N, C*k*k, L]
        
 
        raw_l = [raw_l[i].view(raw_int_ls[0], raw_int_ls[1], kernel, kernel, -1) for i in range(len(raw_l))]
        raw_l = [raw_l[i].permute(0, 4, 1, 2, 3) for i in range(len(raw_l))]    
        raw_l_groups = [torch.split(raw_l[i], 1, dim=0) for i in range(len(raw_l))]
        if self.two_input:
            raw_r = [raw_r[i].view(raw_int_rs[0], raw_int_rs[1], kernel, kernel, -1) for i in range(len(raw_r))]
            raw_r = [raw_r[i].permute(0, 4, 1, 2, 3) for i in range(len(raw_r))]   
            raw_r_groups = [torch.split(raw_r[i], 1, dim=0) for i in range(len(raw_r))]


        left = F.interpolate(left, scale_factor=1./self.rate, mode='nearest')
        if self.two_input:
            right = F.interpolate(right, scale_factor=1./self.rate, mode='nearest')
        mid = F.interpolate(mid, scale_factor=1./self.rate, mode='nearest')
        int_ls = list(left.size())    
        if self.two_input:
            int_rs = list(right.size())
        int_mids = list(mid.size())
        mid_groups = torch.split(mid, 1, dim=0) 
        
     
        left = extract_section(left, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1])
        left = left.view(int_ls[0], int_ls[1], self.ksize, self.ksize, -1)
        left = left.permute(0, 4, 1, 2, 3)   
        l_groups = torch.split(left, 1, dim=0)
        if self.two_input:
            right = extract_section(right, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1])

            right = right.view(int_rs[0], int_rs[1], self.ksize, self.ksize, -1)
            right = right.permute(0, 4, 1, 2, 3) 
            r_groups = torch.split(right, 1, dim=0)

        batch = [i for i in range(raw_int_ls[0])]
        y_l = [[] for i in range(len(shortcut[0]))]
        y_r = [[] for i in range(len(shortcut[0]))]
        y = [[] for i in range(len(shortcut[0]))]

        weight = self.weight_func(raw_int_ls[0], raw_int_ls[2], device=self.device_ids)
        k = self.fuse_k
        scale = self.softmax_scale   
        fuse_weight = torch.eye(k).view(1, 1, k, k)  
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda(self.device_ids)
        if self.two_input == False:
            r_groups = l_groups

        for xi, li, ri, batch_idx in zip(mid_groups, l_groups, r_groups, batch):
        
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda(self.device_ids)
            li = li[0] 
            max_li = torch.sqrt(reduce_sum(torch.pow(li, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            li_normed = li / max_li
            if self.two_input:
                ri = ri[0] 
                max_ri = torch.sqrt(reduce_sum(torch.pow(ri, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
                ri_normed = ri / max_ri


            xi = padding_smi(xi)
            yi = []
            yi.append(F.conv2d(xi, li_normed, stride=1)) 
            if self.two_input:
                yi.append(F.conv2d(xi, ri_normed, stride=1))
 
            if self.fuse:
  
                for i in range(len(yi)):
                    yi[i] = yi[i].view(1, 1, int_ls[2]*int_ls[3], int_mids[2]*int_mids[3]) 
                    yi[i] = padding_smi(yi[i])
                    yi[i] = F.conv2d(yi[i], fuse_weight, stride=1) 
                    yi[i] = yi[i].contiguous().view(1, int_ls[2], int_ls[3], int_mids[2], int_mids[3]) 
                    yi[i] = yi[i].permute(0, 2, 1, 4, 3)
                    yi[i] = yi[i].contiguous().view(1, 1, int_ls[2]*int_ls[3], int_mids[2]*int_mids[3])
                    yi[i] = padding_smi(yi[i])
                    yi[i] = F.conv2d(yi[i], fuse_weight, stride=1)
                    yi[i] = yi[i].contiguous().view(1, int_ls[3], int_ls[2], int_mids[3], int_mids[2])
                    yi[i] = yi[i].permute(0, 2, 1, 4, 3).contiguous()
            yi = [yi[i].view(1, int_mids[2] * int_ls[3], int_mids[2], int_mids[3]) for i in range(len(yi))] 
            yi = [F.softmax(yi[i]*scale, dim=1) for i in range(len(yi))]


            for i in range(len(shortcut[0])):
                li_center = raw_l_groups[i][batch_idx][0]
                if self.two_input:
                    ri_center = raw_r_groups[i][batch_idx][0]
 
                y_l[i].append(F.conv_transpose2d(yi[0], li_center, stride=self.rate, padding=1) / 4.)
                if self.two_input:
                    y_r[i].append(F.conv_transpose2d(yi[1], ri_center, stride=self.rate, padding=1) / 4.)

        for i in range(len(shortcut[0])):
            y_l[i] = torch.cat(y_l[i], dim=0).contiguous().view(raw_int_ms)  
            if self.two_input:
                y_r[i] = torch.cat(y_r[i], dim=0).contiguous().view(raw_int_ms)
                y[i] = weight * y_l[i] + weight.flip(3) * y_r[i]
            else:
                y[i]