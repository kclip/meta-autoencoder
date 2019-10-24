import torch
from data_gen.data_set import message_gen

def test_training(args, h, net_for_testtraining, Noise, PATH_before_adapt, PATH_after_adapt, adapt_steps): #PATH_before_adapt can be meta-learneds
    # initialize network (net_for_testtraining) (net is for meta-training)
    if torch.cuda.is_available():
        net_for_testtraining.load_state_dict(torch.load(PATH_before_adapt))
    else:
        net_for_testtraining.load_state_dict(torch.load(PATH_before_adapt, map_location = torch.device('cpu')))
    if args.if_test_training_adam and not args.if_adam_after_sgd:
        testtraining_optimiser = torch.optim.Adam(net_for_testtraining.parameters(), args.lr_testtraining)
    else:
        pass

    num_adapt = adapt_steps
    for epochs in range(num_adapt):
        m, label = message_gen(args.bit_num, args.mb_size)
        m = m.type(torch.FloatTensor).to(args.device)
        label = label.type(torch.LongTensor).to(args.device)
        for f in net_for_testtraining.parameters():
            if f.grad is not None:
                f.grad.detach()
                f.grad.zero_()

        out = net_for_testtraining(m, h, Noise, args.device, args.if_RTN)
        loss = torch.nn.functional.cross_entropy(out, label)
        # grad calculation
        loss.backward()
        ### adapt (update) parameter
        if args.if_test_training_adam:
            if args.if_adam_after_sgd:
                if epochs < args.num_meta_local_updates:
                    for f in net_for_testtraining.parameters():
                        if f.grad is not None:
                            f.data.sub_(f.grad.data * args.lr_meta_inner)
                elif epochs == args.num_meta_local_updates:
                    testtraining_optimiser = torch.optim.Adam(net_for_testtraining.parameters(),
                                                              args.lr_testtraining)
                    testtraining_optimiser.step()
                else:
                    testtraining_optimiser.step()
            else:
                testtraining_optimiser.step()
        else:
            for f in net_for_testtraining.parameters():
                if f.grad is not None:
                    f.data.sub_(f.grad.data * args.lr_testtraining)
    # saved adapted network for calculate BLER
    torch.save(net_for_testtraining.state_dict(), PATH_after_adapt)


