import torch
import numpy as np

def message_gen(k, mb_size):
    tot_message_num = pow(2,k)
    m = torch.zeros(mb_size, tot_message_num)
    label = torch.zeros(mb_size)
    for ind_mb in range(mb_size):
        if ind_mb % tot_message_num == 0:
            rand_lst = torch.randperm(tot_message_num)
        ind_one_rand_lst = ind_mb % tot_message_num
        ind_one = rand_lst[ind_one_rand_lst]
        m[ind_mb, ind_one] = 1
        label[ind_mb] = ind_one
    return m, label

def channel_set_gen(num_channels, tap_num, if_toy):
    channel_list = []
    for ind_channels in range(num_channels):
        if if_toy:
            assert tap_num == 1
            if ind_channels % 2 == 0:
                h_toy = torch.zeros(2 * tap_num)
                h_toy[0] = 1 * np.cos(np.pi/4)
                h_toy[1] = 1 * np.sin(np.pi/4)
            else:
                h_toy = torch.zeros(2 * tap_num)
                h_toy[0] = 1 * np.cos((3*np.pi) / 4)
                h_toy[1] = 1 * np.sin((3*np.pi) / 4)
            channel_list.append(h_toy)
        else:
            chan_var = 1 / (2 * tap_num)  # since we are generating real and im. part indep. so 1/2 and we are considering complex, -> 2L generated
            Chan = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2 * tap_num),
                                                                              chan_var * torch.eye(2 * tap_num))
            h = Chan.sample()
            channel_list.append(h)
    return channel_list





