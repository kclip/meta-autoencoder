import torch
import argparse
from data_gen.data_set import channel_set_gen
from training.train import test_training
from training.test import test_per_channel_per_snr
from nets.auto_encoder import dnn
from torch.utils.tensorboard import SummaryWriter
from training.meta_train import multi_task_learning
import pickle
import scipy.io as sio
import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description='end_to_end-meta')

    # bit num (k), channel uses (n), tap number (L), number of pilots (P), Eb/N0
    parser.add_argument('--bit_num', type=int, default=4, help='number of bits')
    parser.add_argument('--channel_num', type=int, default=4, help='number of channel uses')
    parser.add_argument('--tap_num', type=int, default=3, help='..')
    parser.add_argument('--mb_size', type=int, default=16, help='minibatch size')
    parser.add_argument('--mb_size_meta_train', type=int, default=16,
                        help='minibatch size during meta-training (this can be useful for decreasing pilots)')
    parser.add_argument('--mb_size_meta_test', type=int, default=16,
                        help='minibatch size for query set (this can be useful for decreasing pilots)')
    parser.add_argument('--Eb_over_N_db', type=float, default=15,
                        help='energy per bit to noise power spectral density ratio')

    # paths
    parser.add_argument('--path_for_common_dir', dest='path_for_common_dir',
                        default='default_folder/default_subfolder/', type=str)
    parser.add_argument('--path_for_meta_training_channels', dest='path_for_meta_training_channels', default=None,
                        type=str)
    parser.add_argument('--path_for_test_channels', dest='path_for_test_channels', default=None, type=str)
    parser.add_argument('--path_for_meta_trained_net', dest='path_for_meta_trained_net', default=None, type=str)

    # neural network architecture (number of neurons for hidden layer)
    parser.add_argument('--num_neurons_encoder', type=int, default=None, help='number of neuron in hidden layer in encoder')
    parser.add_argument('--num_neurons_decoder', type=int, default=None, help='number of neuron in hidden layer in decoder')
    # whether to use bias and relu (if not relu: tanh)
    parser.add_argument('--if_not_bias', dest='if_bias', action='store_false', default=True)
    parser.add_argument('--if_not_relu', dest='if_relu', action='store_false', default=True)
    # RTN
    parser.add_argument('--if_RTN', dest='if_RTN', action='store_true', default=False)
    # in case of running on gpu, index for cuda device
    parser.add_argument('--cuda_ind', type=int, default=0, help='index for cuda device')

    # experiment details (hyperparameters, number of data for calculating performance and for meta-training
    parser.add_argument('--lr_testtraining', type=float, default=0.001, help='lr for adaptation to new channel')
    parser.add_argument('--lr_meta_update', type=float, default=0.01, help='lr during meta-training: outer loop (update initialization) lr')
    parser.add_argument('--lr_meta_inner', type=float, default=0.1, help='lr during meta-training: inner loop (local adaptation) lr')
    parser.add_argument('--test_size', type=int, default=1000000, help='number of messages to calculate BLER for test (new channel)')
    parser.add_argument('--num_channels_meta', type=int, default=100, help='number of meta-training channels (K)')
    parser.add_argument('--num_channels_test', type=int, default=20, help='number of new channels for test (to get average over BLER)')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=20, help='number of meta-training channels considered in one meta-update')
    parser.add_argument('--num_meta_local_updates', type=int, default=1, help='number of local adaptation in meta-training')
    parser.add_argument('--num_epochs_meta_train', type=int, default=10000,
                        help='number epochs for meta-training')

    # if run for joint training, if false: meta-learning
    parser.add_argument('--if_joint_training', dest='if_joint_training', action='store_true', default=False) # else: meta-learning for multi-task learning
    # whether to use Adam optimizer to adapt to a new channel
    parser.add_argument('--if_not_test_training_adam', dest='if_test_training_adam', action='store_false',
                        default=True)
    # if run on toy example (Fig. 2 and 3)
    parser.add_argument('--if_toy', dest='if_toy', action='store_true',
                        default=False)
    # to run on a more realistic example (Fig. 4)
    parser.add_argument('--if_RBF', dest='if_RBF', action='store_true',
                        default=False)
    parser.add_argument('--test_per_adapt_fixed_Eb_over_N_value', type=int, default=15,
                        help='Eb/N0 in db for test')
    # desinged for maml: sgd during args.num_meta_local_updates with args.lr_meta_inner and then follow Adam optimizer with args.lr_testtraining
    parser.add_argument('--if_adam_after_sgd', dest='if_adam_after_sgd', action='store_true',
                        default=False)

    args = parser.parse_args()

    args.device = torch.device("cuda:" + str(args.cuda_ind) if torch.cuda.is_available() else "cpu")
    if args.num_neurons_encoder == None: # unless specified, set number of hidden neurons to be same as the number of possible messages
        args.num_neurons_encoder = pow(2,args.bit_num)
    if args.num_neurons_decoder == None:
        args.num_neurons_decoder = pow(2, args.bit_num)

    if args.if_test_training_adam == False:
        args.if_adam_after_sgd = False

    if args.if_toy == True:
        print('running for toy scenario')
        args.bit_num = 2
        args.channel_num = 1
        args.tap_num = 1
        args.mb_size = 4
        args.mb_size_meta_train = 4
        args.mb_size_meta_test = 4
        args.num_channels_meta = 20
        args.num_neurons_encoder = 4
        args.num_neurons_decoder = 4
    elif args.if_RBF == True:
        print('running for a more realistic scenario')
        args.bit_num = 4
        args.channel_num = 4
        args.tap_num = 3
        args.mb_size = 16
        args.mb_size_meta_train = 16
        args.mb_size_meta_test = 16
        args.num_channels_meta = 100
        args.num_neurons_encoder = 16
        args.num_neurons_decoder = 16
    else:
        print('running on custom environment')
    print('Running on device: {}'.format(args.device))
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)


    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    common_dir = './' + args.path_for_common_dir + curr_time + '/'

    PATH_before_adapt = common_dir + 'saved_model/' + 'before_adapt/' + 'init_net'
    PATH_meta_intermediate = common_dir + 'saved_model/' + 'during_meta_training/' + 'epochs/'

    os.makedirs(common_dir + 'saved_model/' + 'before_adapt/')
    os.makedirs(common_dir + 'saved_model/' + 'after_adapt/')
    os.makedirs(PATH_meta_intermediate)

    os.makedirs(common_dir + 'meta_training_channels/')
    os.makedirs(common_dir + 'test_channels/')
    os.makedirs(common_dir + 'test_result/')

    dir_meta_training = common_dir + 'TB/' + 'meta_training'
    writer_meta_training = SummaryWriter(dir_meta_training)
    dir_during_adapt = common_dir + 'TB/' + 'during_adapt/'

    test_Eb_over_N_range = [args.test_per_adapt_fixed_Eb_over_N_value]
    test_adapt_range = [0, 1, 2, 5, 10, 100, 200, 1000, 10000]

    if len(test_Eb_over_N_range) > 1:
        assert len(test_adapt_range) == 1
    if len(test_adapt_range) > 1:
        assert len(test_Eb_over_N_range) == 1

    test_result_all_PATH = common_dir + 'test_result/' + 'test_result.mat'
    save_test_result_dict = {}

    actual_channel_num = args.channel_num * 2

    net = dnn(M=pow(2, args.bit_num), num_neurons_encoder=args.num_neurons_encoder, n=actual_channel_num, n_inv_filter = args.tap_num,
              num_neurons_decoder=args.num_neurons_decoder, if_bias=args.if_bias, if_relu=args.if_relu, if_RTN=args.if_RTN)
    if torch.cuda.is_available():
        net = net.to(args.device)
    net_for_testtraining = dnn(M=pow(2, args.bit_num), num_neurons_encoder=args.num_neurons_encoder, n=actual_channel_num, n_inv_filter = args.tap_num,
              num_neurons_decoder=args.num_neurons_decoder, if_bias=args.if_bias, if_relu=args.if_relu, if_RTN=args.if_RTN)
    if torch.cuda.is_available():
        net_for_testtraining = net_for_testtraining.to(args.device)

    Eb_over_N = pow(10, (args.Eb_over_N_db/10))
    R = args.bit_num/args.channel_num
    noise_var = 1 / (2 * R * Eb_over_N)
    Noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(actual_channel_num), noise_var * torch.eye(actual_channel_num))
    if args.path_for_meta_training_channels is None:
        print('generate meta-training channels')
        h_list_meta = channel_set_gen(args.num_channels_meta, args.tap_num, args.if_toy)
        h_list_meta_path = common_dir + 'meta_training_channels/' + 'training_channels.pckl'
        f_meta_channels = open(h_list_meta_path, 'wb')
        pickle.dump(h_list_meta, f_meta_channels)
        f_meta_channels.close()
    else:
        print('load previously generated channels')
        h_list_meta_path = args.path_for_meta_training_channels + '/' + 'training_channels.pckl'
        f_meta_channels = open(h_list_meta_path, 'rb')
        h_list_meta = pickle.load(f_meta_channels)
        f_meta_channels.close()

    if args.path_for_meta_trained_net is None:
        if args.if_joint_training:
            print('start joint training')
        else:
            print('start meta-training')
        multi_task_learning(args, net, h_list_meta, writer_meta_training, Noise)
        torch.save(net.state_dict(), PATH_before_adapt)
    else:
        print('load previously saved autoencoder')
        PATH_before_adapt = args.path_for_meta_trained_net

    if args.path_for_test_channels is None:
        print('generate test channels')
        h_list_test = channel_set_gen(args.num_channels_test, args.tap_num, args.if_toy)
        h_list_test_path = common_dir + 'test_channels/' + 'test_channels.pckl'
        f_test_channels = open(h_list_test_path, 'wb')
        pickle.dump(h_list_test, f_test_channels)
        f_test_channels.close()
    else:
        print('load previously generated channels')
        h_list_test_path = args.path_for_test_channels + '/' + 'test_channels.pckl'
        f_test_channels = open(h_list_test_path, 'rb')
        h_list_test = pickle.load(f_test_channels)
        f_test_channels.close()

    if len(h_list_test) > args.num_channels_test:
        h_list_test = h_list_test[:args.num_channels_test]
    print('used test channels', h_list_test)

    dir_test = common_dir + 'TB/' + 'test'
    writer_test = SummaryWriter(dir_test)

    print('start adaptation with test set')
    if_val = False
    total_block_error_rate = torch.zeros(args.num_channels_test, len(test_Eb_over_N_range), len(test_adapt_range))
    ind_adapt_steps = 0
    for adapt_steps in test_adapt_range:
        print('curr adaptation: ', adapt_steps)
        os.mkdir(common_dir + 'saved_model/' + 'after_adapt/' + str(adapt_steps) + '_adapt_steps/')
        os.mkdir(common_dir + 'test_result/' + str(adapt_steps) + '_adapt_steps/')
        test_result_per_adapt_steps = common_dir + 'test_result/' + str(adapt_steps) + '_adapt_steps/' + 'test_result.mat'
        save_test_result_dict_per_adapt_steps = {}

        block_error_rate = torch.zeros(args.num_channels_test, len(test_Eb_over_N_range))
        ind_h = 0
        for h in h_list_test:
            print('current channel ind', ind_h)
            PATH_after_adapt = common_dir + 'saved_model/' + 'after_adapt/' + str(adapt_steps) + '_adapt_steps/'+ str(ind_h) + 'th_adapted_net'
            writer_per_test_channel = []
            test_training(args, h, net_for_testtraining, Noise, PATH_before_adapt, PATH_after_adapt, adapt_steps)
            # test
            ind_snr = 0
            for test_snr in test_Eb_over_N_range:
                block_error_rate_per_snr_per_channel = test_per_channel_per_snr(args, h, net_for_testtraining, test_snr, actual_channel_num, PATH_after_adapt, if_val)
                block_error_rate[ind_h, ind_snr] = block_error_rate_per_snr_per_channel
                total_block_error_rate[ind_h, ind_snr, ind_adapt_steps] = block_error_rate_per_snr_per_channel
                ind_snr += 1
            ind_h += 1
        ind_snr = 0
        save_test_result_dict_per_adapt_steps['block_error_rate'] = block_error_rate.data.numpy()
        sio.savemat(test_result_per_adapt_steps, save_test_result_dict_per_adapt_steps)
        writer_test.add_scalar('average (h) block error rate per adaptation steps', torch.mean(block_error_rate[:, :]), adapt_steps)
        ind_adapt_steps += 1

    save_test_result_dict['block_error_rate_total'] = total_block_error_rate.data.numpy()
    sio.savemat(test_result_all_PATH, save_test_result_dict)