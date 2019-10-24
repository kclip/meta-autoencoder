import torch
from data_gen.data_set import message_gen
from nets.meta_net import meta_dnn

def multi_task_learning(args, net, h_list_meta, writer_meta_training, Noise):
    meta_optimiser = torch.optim.Adam(net.parameters(), args.lr_meta_update)
    h_list_train = h_list_meta[:args.num_channels_meta]

    for epochs in range(args.num_epochs_meta_train):
        first_loss = 0
        second_loss = 0
        iter_in_sampled_device = 0  # for averaging meta-devices
        for ind_meta_dev in range(args.tasks_per_metaupdate):
            # during this, meta-gradients are accumulated
            channel_list_total = torch.randperm(len(h_list_train)) # sampling with replacement
            current_channel_ind = channel_list_total[ind_meta_dev]
            current_channel = h_list_train[current_channel_ind]
            if args.if_joint_training:
                iter_in_sampled_device, first_loss_curr, second_loss_curr = joint_training(args, iter_in_sampled_device,
                                                                                 net, current_channel, Noise)
            else: # maml
                iter_in_sampled_device, first_loss_curr, second_loss_curr = maml(args, iter_in_sampled_device,
                                                                                 net, current_channel, Noise)
            first_loss = first_loss + first_loss_curr
            second_loss = second_loss + second_loss_curr
        first_loss = first_loss / args.tasks_per_metaupdate
        second_loss = second_loss / args.tasks_per_metaupdate
        writer_meta_training.add_scalar('first loss', first_loss, epochs)
        writer_meta_training.add_scalar('second loss', second_loss, epochs)
        # meta-update
        meta_optimiser.zero_grad()
        for f in net.parameters():
            f.grad = f.total_grad.clone() / args.tasks_per_metaupdate
        meta_optimiser.step()  # Adam

def maml(args, iter_in_sampled_device, net, current_channel, Noise):
    net.zero_grad()
    para_list_from_net = list(map(lambda p: p[0], zip(net.parameters())))
    net_meta_intermediate = meta_dnn(if_relu = args.if_relu)

    for inner_loop in range(args.num_meta_local_updates):
        if inner_loop == 0:
            m, label = message_gen(args.bit_num, args.mb_size_meta_train)
            m = m.type(torch.FloatTensor).to(args.device)
            label = label.type(torch.LongTensor).to(args.device)
            out = net_meta_intermediate(m, para_list_from_net, args.if_bias, current_channel, args.device, Noise, args.if_RTN)
            loss = torch.nn.functional.cross_entropy(out, label)
            first_loss_curr = float(loss)
            grad = torch.autograd.grad(loss, para_list_from_net, create_graph=True)
            intermediate_updated_para_list = list(map(lambda p: p[1] - args.lr_meta_inner * p[0], zip(grad, para_list_from_net)))
        else:
            m, label = message_gen(args.bit_num, args.mb_size_meta_train)
            m = m.type(torch.FloatTensor).to(args.device)
            label = label.type(torch.LongTensor).to(args.device)
            out = net_meta_intermediate(m, intermediate_updated_para_list, args.if_bias, current_channel,
                                        args.device, Noise, args.if_RTN)
            loss = torch.nn.functional.cross_entropy(out, label)
            grad = torch.autograd.grad(loss, intermediate_updated_para_list, create_graph=True)
            intermediate_updated_para_list = list(map(lambda p: p[1] - args.lr_meta_inner * p[0], zip(grad, intermediate_updated_para_list)))
            ###########
    #### meta-update
    m, label = message_gen(args.bit_num, args.mb_size_meta_test)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)
    out = net_meta_intermediate(m, intermediate_updated_para_list, args.if_bias, current_channel,
                                args.device, Noise, args.if_RTN)
    loss = torch.nn.functional.cross_entropy(out, label)
    second_loss_curr = float(loss)
    para_list_grad = torch.autograd.grad(loss, para_list_from_net, create_graph=False)
    ind_f_para_list = 0
    for f in net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = para_list_grad[ind_f_para_list].data.clone()
        else:
            f.total_grad = f.total_grad + para_list_grad[ind_f_para_list].data.clone()
        ind_f_para_list += 1
    iter_in_sampled_device = iter_in_sampled_device + 1
    return iter_in_sampled_device, first_loss_curr, second_loss_curr


def joint_training(args, iter_in_sampled_device, net, current_channel, Noise):
    net.zero_grad()
    para_list_from_net = list(map(lambda p: p[0], zip(net.parameters())))
    net_meta_intermediate = meta_dnn(if_relu = args.if_relu)

    m, label = message_gen(args.bit_num, args.mb_size_meta_test)
    m = m.type(torch.FloatTensor).to(args.device)
    label = label.type(torch.LongTensor).to(args.device)
    out = net_meta_intermediate(m, para_list_from_net, args.if_bias, current_channel, args.device, Noise, args.if_RTN)
    loss = torch.nn.functional.cross_entropy(out, label)
    first_loss_curr = float(loss)
    grad = torch.autograd.grad(loss, para_list_from_net, create_graph=False)
    second_loss_curr = first_loss_curr

    ind_f_para_list = 0
    for f in net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = grad[ind_f_para_list].data.clone()
        else:
            f.total_grad = f.total_grad + grad[ind_f_para_list].data.clone()
        ind_f_para_list += 1
    iter_in_sampled_device = iter_in_sampled_device + 1
    return iter_in_sampled_device, first_loss_curr, second_loss_curr