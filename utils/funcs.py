import torch

def complex_mul(h, x): # h fixed on batch, x has multiple batch
    if len(h.shape) == 1:
        # h is same over all messages (if estimated h, it is averaged)
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[0] - x[:, 1] * h[1]
        y[:, 1] = x[:, 0] * h[1] + x[:, 1] * h[0]
    elif len(h.shape) == 2:
        # h_estimated is not averaged
        assert x.shape[0] == h.shape[0]
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[:, 0] - x[:, 1] * h[:, 1]
        y[:, 1] = x[:, 0] * h[:, 1] + x[:, 1] * h[:, 0]
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError
    return y


def complex_mul_taps(h, x_tensor):
    if len(h.shape) == 1:
        L = h.shape[0] // 2  # length/2 of channel vector means number of taps
    elif len(h.shape) == 2:
        L = h.shape[1] // 2  # length/2 of channel vector means number of taps
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError
    y = torch.zeros(x_tensor.shape[0], x_tensor.shape[1], dtype=torch.float)
    assert x_tensor.shape[1] % 2 == 0
    for ind_channel_use in range(x_tensor.shape[1]//2):
        for ind_conv in range(min(L, ind_channel_use+1)):
            if len(h.shape) == 1:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(h[2*ind_conv:2*(ind_conv+1)], x_tensor[:, (ind_channel_use-ind_conv)*2:(ind_channel_use-ind_conv+1)*2])
            else:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(
                    h[:, 2 * ind_conv:2 * (ind_conv + 1)],
                    x_tensor[:, (ind_channel_use - ind_conv) * 2:(ind_channel_use - ind_conv + 1) * 2])

    return y

def complex_conv_transpose(h_trans, y_tensor): # takes the role of inverse filtering
    assert len(y_tensor.shape) == 2 # batch
    assert y_tensor.shape[1] % 2 == 0
    assert h_trans.shape[0] % 2 == 0
    if len(h_trans.shape) == 1:
        L = h_trans.shape[0] // 2
    elif len(h_trans.shape) == 2:
        L = h_trans.shape[1] // 2
    else:
        print('h shape length need to be either 1 or 2')

    deconv_y = torch.zeros(y_tensor.shape[0], y_tensor.shape[1] + 2*(L-1), dtype=torch.float)
    for ind_y in range(y_tensor.shape[1]//2):
        ind_y_deconv = ind_y + (L-1)
        for ind_conv in range(L):
            if len(h_trans.shape) == 1:
                deconv_y[:, 2*(ind_y_deconv - ind_conv):2*(ind_y_deconv - ind_conv+1)] += complex_mul(h_trans[2*ind_conv:2*(ind_conv+1)] , y_tensor[:,2*ind_y:2*(ind_y+1)])
            else:
                deconv_y[:, 2 * (ind_y_deconv - ind_conv):2 * (ind_y_deconv - ind_conv + 1)] += complex_mul(
                    h_trans[:, 2 * ind_conv:2 * (ind_conv + 1)], y_tensor[:, 2 * ind_y:2 * (ind_y + 1)])
    return deconv_y[:, 2*(L-1):]


