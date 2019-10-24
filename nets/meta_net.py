from __future__ import print_function
import torch.nn as nn
import torch
from torch.nn import functional as F
from utils.funcs import complex_mul_taps, complex_conv_transpose

class meta_Net_DNN(nn.Module):
    def __init__(self, if_relu): # it only gets paramters from other network's parameters
        super(meta_Net_DNN, self).__init__()
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()

    def forward(self, x, var, if_bias, h, device, noise_dist, if_RTN):
        idx_init = 0
        if if_bias:
            gap = 2
        else:
            gap = 1
        idx = idx_init
        while idx < len(var):
            if idx > idx_init: # no activation from the beginning
                if idx == gap * 2+idx_init: # after last layer of encoder
                    pass
                else:
                    x = self.activ(x)
            if idx == idx_init:
                if if_bias:
                    w1, b1 = var[idx], var[idx + 1] # weight and bias
                    x = F.linear(x, w1, b1)
                    idx += 2
                else:
                    w1 = var[idx] # weight
                    x = F.linear(x, w1)
                    idx += 1
            elif idx == gap * 1+idx_init:
                if if_bias:
                    w2, b2 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w2, b2)
                    idx += 2
                else:
                    w2 = var[idx]  # weight and bias
                    x = F.linear(x, w2)
                    idx += 1
            elif idx == gap * 2+idx_init:
                #### now we need to normalize and then pass the channel
                x_norm = torch.norm(x, dim=1)
                x_norm = x_norm.unsqueeze(1)
                x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm
                x = complex_mul_taps(h, x)
                x = x.to(device)
                # noise
                n = torch.zeros(x.shape[0], x.shape[1])
                for noise_batch_ind in range(x.shape[0]):
                    n[noise_batch_ind] = noise_dist.sample()
                n = n.type(torch.FloatTensor).to(device)
                x = x + n

                if if_RTN:
                    if if_bias:
                        w_rtn_1, b_rtn_1 = var[idx], var[idx+1]
                        h_inv = F.linear(x, w_rtn_1, b_rtn_1)
                        h_inv = self.tanh(h_inv)
                        w_rtn_2, b_rtn_2 = var[idx+2], var[idx + 3]
                        h_inv = F.linear(h_inv, w_rtn_2, b_rtn_2)
                        h_inv = self.tanh(h_inv)
                        w_rtn_3, b_rtn_3 = var[idx + 4], var[idx + 5]
                        h_inv = F.linear(h_inv, w_rtn_3, b_rtn_3)
                        rtn_gap = 6
                    else:
                        w_rtn_1 = var[idx]
                        h_inv = F.linear(x, w_rtn_1)
                        h_inv = self.tanh(h_inv)
                        w_rtn_2 = var[idx+1]
                        h_inv = F.linear(h_inv, w_rtn_2)
                        h_inv = self.tanh(h_inv)
                        w_rtn_3 = var[idx+2]
                        h_inv = F.linear(h_inv, w_rtn_3)
                        rtn_gap = 3
                    x = complex_conv_transpose(h_inv, x)
                    x = x.to(device)
                else:
                    rtn_gap = 0
                ############## from now, demodulator
                if if_bias:
                    w3, b3 = var[idx+ rtn_gap], var[idx + rtn_gap + 1]  # weight and bias
                    x = F.linear(x, w3, b3)
                    idx += (2 + rtn_gap)
                else:
                    w3 = var[idx + rtn_gap]  # weight
                    x = F.linear(x, w3)
                    idx += (1 + rtn_gap)
            elif idx == gap * 3+rtn_gap+idx_init:
                if if_bias:
                    w4, b4 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w4, b4)
                    idx += 2
                else:
                    w4 = var[idx]  # weight
                    x = F.linear(x, w4)
                    idx += 1
        return x

def meta_dnn(**kwargs):
    net = meta_Net_DNN(**kwargs)
    return net
