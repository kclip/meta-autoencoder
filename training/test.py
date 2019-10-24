import torch
from data_gen.data_set import message_gen

def test_per_channel_per_snr(args, h, net_for_testtraining, test_snr, actual_channel_num, PATH_after_adapt, if_val):
    if torch.cuda.is_available():
        net_for_testtraining.load_state_dict(torch.load(PATH_after_adapt))
    else:
        net_for_testtraining.load_state_dict(torch.load(PATH_after_adapt, map_location = torch.device('cpu')))


    batch_size = args.test_size
    success_test = 0
    Eb_over_N_test = pow(10, (test_snr / 10))
    R = args.bit_num / args.channel_num
    noise_var_test = 1 / (2 * R * Eb_over_N_test)

    Noise_test = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(actual_channel_num),
                                                                            noise_var_test * torch.eye(
                                                                                actual_channel_num))
    m_test, label_test = message_gen(args.bit_num, batch_size)
    m_test = m_test.type(torch.FloatTensor).to(args.device)
    label_test = label_test.type(torch.LongTensor).to(args.device)

    out_test = net_for_testtraining(m_test, h, Noise_test, args.device, args.if_RTN)
    for ind_mb in range(label_test.shape[0]):
        assert label_test.shape[0] == batch_size
        if torch.argmax(out_test[ind_mb]) == label_test[ind_mb]:  # means correct classification
            success_test += 1
        else:
            pass
    accuracy = success_test / label_test.shape[0]
    if not if_val:
        print('for snr: ', test_snr, 'acccuracy: ', 1 - accuracy)

    return 1 - accuracy


