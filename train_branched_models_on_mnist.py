from __future__ import print_function
import argparse

import json

import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import copy
import numpy as np
import os
import random
from utils.other_utils import MyMSELOSS

import matplotlib.pyplot as plt

from models.linearized_model import BranchedLinearizedModel
from models.models import *
from models.branched_models import *
from moore_penrose_functions import specialization


def get_net(args, num_classes, device):
    branch_constructor = get_constructor(args.model_type, num_classes, args.first_width, device, softmax_output=False)

    print('construct branched nn')
    net = NetBranched(branch_constructor, args.num_branches)

    # linearize
    if args.linearize_model:
        print('linearize')
        net = BranchedLinearizedModel(net, device)

    return net


def get_constructor(model_type, num_classes, first_width, device, softmax_output=False):
    # assign a branch constructor according to model type

    if model_type == 'linear':
        def constructor():
            return Net_lin(softmax_output=softmax_output, num_classes=num_classes).to(device)
    elif model_type == 'original':
        def constructor():
            return Net(softmax_output=softmax_output, num_classes=num_classes, first_width=first_width).to(device)
    elif model_type == 'composed_1_conv':
        def constructor():
            return Net_two_convs(softmax_output=softmax_output, num_classes=num_classes).to(device)
    elif model_type == 'two_fc':
        def constructor():
            return NetFC(softmax_output=softmax_output, num_classes=num_classes).to(device)
    elif model_type == 'cloned':
        net = Net(softmax_output=softmax_output, num_classes=num_classes, use_dropout=False).to(device)
        def constructor():
            return net

    return constructor


def get_criterion(loss, num_classes=None):
    if loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif loss == 'mse':
        assert num_classes is not None
        criterion = MyMSELOSS(num_classes)
    return criterion


def get_data_loaders(train_batch_size, test_batch_size, device, shuffle=True, fraction = 1.0, two_class=False):
    assert fraction <= 1.0

    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if device == 'cuda':
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    def two_class_only_helper(dataset, num1=0, num2=4, convert_to_0_1_labels=True):
        num1_ind = np.array(dataset.targets == num1)
        num2_ind = np.array(dataset.targets == num2)
        ind = num1_ind + num2_ind
        dataset.targets = dataset.train_labels[ind]

        if convert_to_0_1_labels:
            dataset.targets = dataset.targets == max(num1, num2)

        dataset.data = dataset.train_data[ind]
        return dataset

    def subsample_data_helper(dataset):
        inds = list(range(len(dataset)))
        num_samples = round(fraction *len(dataset))
        inds_subset =  random.sample(inds, num_samples)
        dataset = torch.utils.data.Subset(dataset, inds_subset)
        return dataset

    if two_class:
        dataset1 = two_class_only_helper(dataset1)
        dataset2 = two_class_only_helper(dataset2)

    dataset1 = subsample_data_helper(dataset1)
    dataset2 = subsample_data_helper(dataset2)




    trainloader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    testloader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return trainloader, testloader


def train(args, net, criterion, device, trainloader, optimizer, epoch):
    net.train()
    correct = 0
    num_samples = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        if net.num_classes == 1:  # i.e. args.two_class is set True
            target = target.float()
        optimizer.zero_grad()
        output = net(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if net.num_classes == 1:  # i.e. args.two_class is set True
            if (args.loss == 'ce') and not(args.softmax_output):
                pred = output > 0
                pred = pred.float()
            elif args.loss == 'mse':
                pred = torch.round(output).float()
                print('pred')
                print(pred)
                print('output')
                print(output)
                print('target')
                print(target)
            else:
                raise NotImplementedError
        else:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        correct += pred.eq(target.view_as(pred)).sum().item()
        num_samples += len(data)
        if batch_idx % 10 == 0:
            cur_acc = correct / num_samples
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccumulated Accuracy: {:.3f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item(),
                cur_acc))

            if args.dry_run:
                break


def test(net, device, testloader, criterion, dir_name='', suffix=''):
    net.eval()
    test_loss = 0
    correct = 0
    out_branches_list = []
    for i, (data, target) in enumerate(testloader):
        data, target = data.to(device), target.to(device)
        if net.num_classes == 1:  # i.e. args.two_class is set True
            target = target.float()


        output, cur_out_branches_list = net(data, return_branch_output=True)

        if i > 0:
            if prev_batch_size == data.shape[0]:
                out_branches_list += [cur_out_branches_list]
            else:
                print('skipped {}_th batch (for specialization only, not for accuracy or loss)'.format(i))
        else:
            out_branches_list += [cur_out_branches_list]

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

        prev_batch_size = data.shape[0]

    out_branches_list = [torch.cat(single_branch_list, dim=0) for single_branch_list in zip(*out_branches_list)]


    # show correlation and spcialization
    spec, c = specialization(out_branches_list, return_additional_measures=False)

    


    test_loss /= len(testloader.dataset)
    test_acc = correct / len(testloader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * test_acc))
    
    
    
    
    plt.figure()
    plt.imshow(c)
    plt.title('Specialization: {}'.format(spec))
    plt.colorbar()
    plt.savefig(dir_name + '/corr_spec' + suffix + '.png')
    plt.close()

    # show branches
    branch_output = torch.cat(out_branches_list, dim=1).data.cpu()
    select_num_max = 100  # dont show 1000 samples, 100 is enough
    select_num = min(select_num_max, len(branch_output))
    target_select = target[:select_num]
    o = branch_output[:select_num]
    o_sorted = o[torch.argsort(target_select).cpu()]

    plt.figure()
    plt.imshow(o_sorted)
    plt.title('Accuracy: {}'.format(correct / len(testloader.dataset)))
    plt.colorbar()
    plt.savefig(dir_name + '/branch_out' + suffix + '.png')
    plt.close()
    
    
    
    return test_loss, test_acc


def main(args):
    # paths
    if args.linearize_model:
        lin_model = '_linearized'
    else:
        lin_model = ''
    model_dir_name = 'mnist_runs/{}{}'.format(args.model_type, lin_model)
    if not os.path.exists(model_dir_name):
        os.mkdir(model_dir_name)
    fname = model_dir_name + '/loss_{}_softmax_{}_lr_sched_{}_lr_{}_twoclass_{}_num_branches_{}_first_width_{}'.format(args.loss, args.softmax_output, args.use_scheduler, args.lr, args.two_class, args.num_branches, args.first_width)
    os.mkdir(fname)
    dir_name = './' + fname

    # save args
    args_path = dir_name + '/args.txt'
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # technicalities
    print('use_cuda = ' + str(use_cuda))
    torch.manual_seed(args.seed)
    if args.two_class:
        num_classes = 1  # in the 2-class case there's no advantage in considering one-hot labels
    else:
        num_classes = 10  # for 3 and more classes - one-hot labeling is a must
    device = torch.device("cuda" if use_cuda else "cpu")

    # get data, model, loss and optimizer
    trainloader, testloader = get_data_loaders(train_batch_size=args.batch_size, test_batch_size=args.test_batch_size, two_class=args.two_class, device=device)
    net = get_net(args, num_classes, device)
    criterion = get_criterion(args.loss, num_classes=None)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    if args.use_scheduler:
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        arr_lr = []
    else:
        arr_lr = args.lr

    # save initialization
    # TODO replace below cases with a get_theta method, and a get_state_dict method
    if net.linearized_model:  # then theta is parameters of linear layer only (other non-theta parameters are the initial gradient and weights)
        if net.branched_model:
            theta = [branch.linear.state_dict() for branch in net.branches]
            model_init_state_dict = [branch.model.state_dict() for branch in net.branches]
        else:
            theta = net.linear.state_dict()
            model_init_state_dict = net.model.state_dict()
    else:
        model_init_state_dict = []
        if net.branched_model:
            theta = [branch.state_dict() for branch in net.branches]
        else:
            theta = net.state_dict()
    Theta = [copy.deepcopy(theta)]
    torch.save((model_init_state_dict, Theta), dir_name + '/theta-loss.pt')

    # before training, test initialized model
    print('Test set (initialized net): ')
    test(net, device, testloader, criterion, dir_name, suffix='_init')

    # Train
    print('Begin training')
    for epoch in range(1, args.epochs + 1):
        train(args, net, criterion, device, trainloader, optimizer, epoch)
        print('Test set: ')
        test(net, device, testloader, criterion, dir_name)

        # TODO replace below cases with a get_theta method, implemented for each model individually
        if net.linearized_model:  # then theta is parameters of linear layer only (other non-theta parameters are the initial gradient and weights)
            if net.branched_model:
                theta = [branch.linear.state_dict() for branch in net.branches]
            else:
                theta = net.linear.state_dict()
        else:
            if net.branched_model:
                theta = [branch.state_dict() for branch in net.branches]
            else:
                theta = net.state_dict()
        Theta += [copy.deepcopy(theta)]

        if args.use_scheduler:
            arr_lr += scheduler.get_last_lr()
            scheduler.step()  # this does not optimize weights! It merely updates the leaqrning rate for the next epocjh.

        optimizer.step()

        torch.save((model_init_state_dict, Theta), dir_name + '/theta-loss.pt')

        print('saved!')


if __name__ == '__main__':
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # arguments to play with
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--two_class', action='store_true', help='whether to train on two digits only (default is 4 and 5)')
    parser.add_argument('--model_type', type=str, default='original', help='options: linear, original')
    parser.add_argument('--linearize_model', action='store_true', help='whether to linearize the model, i,e, restrict it to the first-order taylor expansion w.r.t. its parameters')
    parser.add_argument('--num_branches', type=int, default=8, help='No. of Branches')
    parser.add_argument('--first_width', type=int, default=32, help='when possible, this affects the general width of all the model - where the width of the first layer is set by this, and the rest of the layers are set accordingly. ')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 1.0)')

    # arguments that are fixed as default in our experiments
    parser.add_argument('--loss', default='ce', type=str, help='loss criterion, either ce or mse')
    parser.add_argument('--use_scheduler', default=False, type=bool)
    parser.add_argument('--softmax_output', action='store_true', help='whether to softmax the ouput oof the model before feeding it to the loss criterion (set false if using --loss == ce')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N', help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--data', type=str, default='mnist', metavar='S', help='used for user clarity: will see in saved args that this was trained on mnist')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'mse':
        criterion = MyMSELOSS()

    main(args)


