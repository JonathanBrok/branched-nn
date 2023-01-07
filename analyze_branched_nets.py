import json
import argparse
import torch
import matplotlib.pyplot as plt

# from train_resnet_on_cifar10 import test as test_cifar10
# from train_resnet_on_cifar10 import get_data_loaders as get_cifar10_loaders

from train_branched_models_on_mnist import test as test_mnist
from train_branched_models_on_mnist import get_data_loaders as get_mnist_loaders
from train_branched_models_on_mnist import get_net_criterion as get_net_criterion_mnist

from core.branch_moore_penrose_functions import moore_penrose_analysis_for_branched_net

from utils.other_utils import MyMSELOSS


def analyze_branched_net(net, device, testloader, criterion, dir_name, restricted_num_samples=None):
    net.eval()
    test_loss = 0
    correct = 0
    tot = 0
    out_branches_list_all = []
    for data, target in testloader:
        data, target = data.to(device), target.to(device)

        if net.num_classes == 1:  # i.e. args.two_class is set True
            target = target.float()

        sort_ind = torch.argsort(target)
        data, target = data[sort_ind], target[sort_ind]
        tot_new = tot + len(target)
        if tot_new > restricted_num_samples:
            break
        tot = tot_new


        output, out_branches_list = net(data, return_branch_output=True)
        if len(out_branches_list_all) == 0:
            out_branches_list_all = out_branches_list
        else:
            out_branches_list_all = [torch.concatenate([o1, o2], dim=0) for o1, o2 in zip(out_branches_list_all, out_branches_list)]




        cur_loss = criterion(output, target)

        test_loss += cur_loss.item()  # sum up batch loss

        if net.num_classes == 1:  # i.e. args.two_class is set True
            if (args.loss == 'ce') and not(args.softmax_output):
                pred = output > 0
                pred = pred.float()
            else:
                raise NotImplementedError
        else:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        correct += pred.eq(target.view_as(pred)).sum().item()

    spec, corr = specialization(out_branches_list_all)

    plt.figure()
    plt.imshow(corr)
    plt.title('out correlation. spec. ' + str(spec))
    plt.savefig(dir_name + '/corr.png')

    plt.figure()
    out_branches_mat = torch.concatenate(out_branches_list, dim=1)
    max_val = torch.max(out_branches_mat)
    min_val = torch.min(out_branches_mat)

    for i, branch_out in enumerate(out_branches_list):
        plt.subplot(1, net.num_branches, i + 1)
        plt.imshow(branch_out.cpu().data)

        plt.clim(min_val, max_val)
    plt.colorbar()
    plt.title('branch No. ' + str(i + 1))
    plt.xlabel('class')
    plt.ylabel('data')
    plt.savefig(dir_name + '/branch_out_batch.png')

    plt.figure()
    out_branches_mat_all = torch.concatenate(out_branches_list_all, dim=1)
    max_val = torch.max(out_branches_mat_all)
    min_val = torch.min(out_branches_mat_all)
    for i, branch_out in enumerate(out_branches_list_all):
        plt.subplot(1, net.num_branches, i + 1)
        plt.imshow(branch_out.cpu().data)

        plt.clim(min_val, max_val)
    plt.colorbar()
    plt.title('branch No. ' + str(i + 1))
    plt.xlabel('class')
    plt.ylabel('data')
    plt.savefig(dir_name + '/branch_out_many_batches.png')

    is_linearized_model = False
    if is_linearized_model:
        f0_list, grads_times_lin_list, grads_mat_list = net.branched_forward_core(data)

        plt.figure()
        max_val = -1e5
        min_val = 1e5
        for i, f0 in enumerate(f0_list):
            plt.subplot(1, net.num_branches, i + 1)
            plt.imshow(f0.cpu().data)
            max_val = max(max_val, torch.max(f0))
            min_val = min(min_val, torch.min(f0))
            plt.clim(min_val, max_val)
        plt.colorbar()
        plt.title('branch No. ' + str(i+1))
        plt.xlabel('class')
        plt.ylabel('data')


        plt.figure()
        max_val = -1e5
        min_val = 1e5
        for i, grads_times_lin in enumerate(grads_times_lin_list):
            plt.subplot(1, net.num_branches,  i + 1)
            plt.imshow(grads_times_lin.cpu().data)
            max_val = max(max_val, torch.max(grads_times_lin))
            min_val = min(min_val, torch.min(grads_times_lin))
            plt.clim(min_val, max_val)
        plt.colorbar()
        plt.title('branch No. ' + str(i + 1))
        plt.xlabel('class')
        plt.ylabel('data')

        plt.figure()
        max_val = -1e5
        min_val = 1e5
        for i, (f0, grads_times_lin) in enumerate(zip(f0_list, grads_times_lin_list)):
            branch_out = f0 + grads_times_lin
            plt.subplot(1, net.num_branches, i + 1)
            plt.imshow(branch_out.cpu().data)
            max_val = max(max_val, torch.max(branch_out))
            min_val = min(min_val, torch.min(branch_out))
            plt.clim(min_val, max_val)
        plt.colorbar()
        plt.title('branch No. ' + str(i + 1))
        plt.xlabel('class')
        plt.ylabel('data')

        # plt.figure()
        # max_val = -1e5
        # min_val = 1e5
        # for i, grads_mat_list in enumerate(grads_mat_list):
        #     plt.subplot(1, net.num_branches,  i + 1)
        #     plt.imshow(grads_mat_list.data)
        #     max_val = max(max_val, torch.max(grads_mat_list))
        #     min_val = min(min_val, torch.min(grads_mat_list))
        #     plt.clim(min_val, max_val)
        # # plt.colorbar()






    test_loss /= tot
    test_acc = correct / tot
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, tot,
        100. * test_acc))
    return test_loss, test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist_branched_twoclass', action='store_true')
    parser.add_argument('--mnist_branched', action='store_true')
    parser.add_argument('--mnist_branched_linearized', action='store_true')
    parser.add_argument('--mnist_branched_fc', action='store_true')
    parser.add_argument('--mnist_branched_fc_twoclass', action='store_true')
    parser.add_argument('--mnist_branched_fc_linearized', action='store_true')
    parser.add_argument('--softmax_output', action='store_true')
    parser.add_argument('--loss', type=str, help='ce or mse')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'mse':
        criterion = MyMSELOSS()

    if args.mnist_branched_linearized:
        train_batch_size = 32
        test_batch_size = 32
        restricted_num_samples = 200
        # directories:'
        dir_name = 'models_branched_linearized_mnist/theta_model_branched_linearized_original_loss_mse_softmax_True_lr_sched_False_lr_0.1'

    elif args.mnist_branched:
        train_batch_size = 64
        test_batch_size = 200
        restricted_num_samples=1e10  # 1e10 i.e. not restricted
        dir_name = 'models_branched_linearized_mnist/theta_model_branched_original_loss_mse_softmax_True_lr_sched_False_lr_0.1'

    elif args.mnist_branched_twoclass:
        train_batch_size = 64
        test_batch_size = 200
        restricted_num_samples = 1e10  # 1e10 i.e. not restricted
        dir_name = 'mnist_runs/theta_model_branched_original_loss_ce_softmax_False_lr_sched_False_lr_0.01_twoclass_True'

    elif args.mnist_branched_fc_linearized:
        train_batch_size = 32
        test_batch_size = 32
        restricted_num_samples = 200
        dir_name = 'mnist_runs/theta_model_branched_linearized_fc_loss_ce_softmax_False_lr_sched_False_lr_0.1_twoclass_False'

    elif args.mnist_branched_fc:
        train_batch_size = 64
        test_batch_size = 200
        restricted_num_samples=1e10  # 1e10 i.e. not restricted
        dir_name = 'mnist_runs/theta_model_branched_fc_loss_ce_softmax_False_lr_sched_False_lr_0.1_twoclass_False'

    elif args.mnist_branched_fc_twoclass:
        train_batch_size = 64
        test_batch_size = 200
        restricted_num_samples=1e10  # 1e10 i.e. not restricted
        dir_name = 'mnist_runs/theta_model_branched_fc_loss_ce_softmax_False_lr_sched_False_lr_0.001_twoclass_True'

    elif args.cifar10_resnet18:
            pass  # TODO

    elif args.cifar10_linearized_resnet18:
        pass  # TODO

    else:
        raise NotImplementedError



    # load original training args
    args_path = dir_name + '/args.txt'
    args = parser.parse_args()
    with open(args_path, 'r') as f:
        args.__dict__ = json.load(f)

    Theta_path = dir_name + '/init-theta.pt'

    # Load params
    model_init_state_dict, Theta = torch.load(Theta_path, map_location=torch.device('cpu'))
    theta_end = Theta[-1]
    epochs = len(Theta) - 1

    if args.data == 'mnist':
        test = test_mnist
        trainloader, testloader = get_mnist_loaders(train_batch_size, test_batch_size, device)
        # construct model
        net, criterion = get_net_criterion_mnist(args)
    elif args.data == 'cifar10':
        pass  # TODO implement a cifar10 equivalent to above

    # assign initial weights
    for branch, state_dict in zip(net_init.branches, model_init_state_dict):
        branch.set_model(state_dict)

    # assign optimized linear weights
    for branch, state_dict in zip(net.branches, theta_end):
        branch.linear.load_state_dict(state_dict)


    # analyze pt. 1
    test_loss, test_acc = analyze_branched_net(net, device, testloader, criterion=criterion, dir_name=dir_name, restricted_num_samples=restricted_num_samples)

    # analyze pt. 2
    if net.num_classes == 1:
        compare_to_thry = False
        y, y_hat, x, y_thry_hat, x_thry, a, c, spec, act = moore_penrose_analysis_for_branched_net(net_init, testloader, compare_to_thry, device)

        plt.figure()
        plt.plot(x_thry, '.', label='theoretical equivalent')
        if compare_to_thry:
            plt.plot(x, label='pseudo inverse result')
        plt.legend()
        plt.figure()
        plt.imshow(c)
        # plt.show()




    plt.show()