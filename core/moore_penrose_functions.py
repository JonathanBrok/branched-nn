import numpy as np
from models.linearized_model import BranchedLinearizedModel
import torch
from matplotlib import pyplot as plt



def branch_correlation(out_branches_list):
    """
    :param out_branches_list: a list of same-size tensors or numpy arrays
    :return: correlation matrix (numpy array) between list elements
    """

    # if pytorch tensor, then convert to numpy array
    if torch.is_tensor(out_branches_list[0]):
        out_branches_list = [b.cpu().detach().numpy() for b in out_branches_list]

    out_branches_list_flat = [np.ndarray.flatten(b) for b in out_branches_list]
    out_branches_flat_mat = np.stack(out_branches_list_flat, axis=1)

    c = np.cov(out_branches_flat_mat.T)

    return c


def specialization(out_branches_list, return_additional_measures=False):
    """
    :param out_branches_list: a list of same-size pytorch-tensors or numpy-arrays
    :param return_additional_measures: whether to return extra meaures (currently unused in our scripts)
    :return: Branch Correlation as well as Specialization measures(s)
    """

    num_branches = len(out_branches_list)



    c = branch_correlation(out_branches_list)

    # new definition (same logic as old, but linearly mapped to [0, 1])
    spec = (num_branches - np.sum(c ** 2) / np.sum(np.diag(c) ** 2)) / (num_branches - 1)

    if not return_additional_measures:
        return spec, c
    else:
        # some unused measures
        local_spec = np.sqrt((np.diag(c)) ** 2 / np.sum(c ** 2, axis=0))
        importance = np.sqrt(np.diag(c) ** 2 / np.sum(np.diag(c) ** 2))
        return spec, c, local_spec, importance



def simulate_branched_linearized_mse(a, y):
    """
    Considers the Linearization of a Branched NN, optimized on MSE (hence a pseudo-inverse solution is valid, since we have a linear model and mse criterion)
    :param a: a list with l elements where each element is a m X k matrix corresponding to gradients of a different branch
    :param l: No. of Branches
    :param y: labels
    :return: x: solved parameters
             y_hat: per-branch output, concatenated
    """

    l = len(a)  # No. of branches
    m = a[0].shape[0]  # No. of samples

    # a single stacked gradient matrix for all branches
    a_total = np.concatenate(a, axis=1)

    print('a[0].shape')
    print(a[0].shape)
    print('a_total.shape')
    print(a_total.shape)
    # solve for parameters
    x = np.linalg.pinv(a_total) @ y
    p = a[0].shape[1]  # No. of parameters per branch
    # initialize branch outputs
    y_hat = []
    # solve branch outputs
    for i in range(l):
        y_hat.append(a[i] @ x[i * p:(i + 1) * p, ])
    return x, y_hat


def simulate_theoretical_branched_linearized_mse(a, y):
    """
    Considers the Linearization of a Branched NN, optimized on MSE (hence a pseudo-inverse solution is valid, since we have a linear model and mse criterion)
    :param a: a list with l elements where each element is a m X k matrix corresponding to gradients of a different branch
    :param y: labels
    :return: x: solved parameters
             y_hat: per-branch output, concatenated
    """

    l = len(a)  # No. of branches
    m = a[0].shape[0]  # No. of samples

    # pre-prepare gradient matrix and kernel
    a_total = np.concatenate(a, axis=1)  # a single stacked gradient matrix for all branches
    aa = a_total @ np.transpose(a_total)  # gradient kernel matrix

    # apply theoretical per-branch solution
    aa_inv = np.linalg.pinv(aa)  # inverse kernel. Gradient kernel (aa) is assumed to be full rank and hence the pseudo-inverse is infact an inverse
    x_thry_list = []
    for a_i in a:
        x_thry_i = np.transpose(a_i) @ aa_inv @ y
        x_thry_list += [x_thry_i]

    # branch outputs givnen above theoretical branch solution
    y_hat = []  # initialize branch outputs
    for i, (a_i, x_thry_i) in enumerate(zip(a, x_thry_list)):
        y_hat.append(a_i @ x_thry_i)

    x = np.concatenate(x_thry_list)
    return x, y_hat


def get_initial_grad_f(branched_net, data, device):
    """
    Uses our Branched Linearized Model's functions
    :param constructor: intializes architecture for each branch
    :param num_branches: No. of branches
    :return: branch gradients (list of pytorch tensors), and predictions of the initialized branches (list of tensors)
    """
    # build linearized branched net
    net = BranchedLinearizedModel(branched_net, device)
    # get grads
    grads = net.get_branch_grads(data, split_output_dim_to_list=False)
    # get initial output
    out_0 = net.get_branch_f0s(data)

    return grads, out_0


def moore_penrose_analysis_for_branched_net(branched_net, testloader, compare_to_thry, device):

    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        break  # single batch

    grads, out_0 = get_initial_grad_f(branched_net=branched_net, data=data, device=device)

    list_of_tensors_to_list_of_numpy = lambda l_t : [t.data.cpu().numpy() for t in l_t]
    target = target.float().data.cpu().numpy()
    target_minus_f0 = target
    for out_0_i in out_0:
        target_minus_f0 -= torch.squeeze(out_0_i).data.cpu().numpy()



    grads = list_of_tensors_to_list_of_numpy(grads)
    # y = target - sum(out_0)

    # mse linearized pseudo-inverse solution per-branch
    if compare_to_thry:
        x, y_hat = simulate_branched_linearized_mse(a=grads, y=target)
    else:
        x, y_hat = [], []
    x_thry, y_thry_hat = simulate_theoretical_branched_linearized_mse(a=grads, y=target)

    # evaluate specialization
    num_branches = len(grads)

    spec, c = specialization(y_hat)

    plt.figure()
    plt.plot(x, label='pseudo inverse result')
    plt.plot(x_thry, '.', label='theoretical equivalent')
    plt.legend()

    plt.figure()
    plt.imshow(c)

    return target, y_hat, x, y_thry_hat, x_thry, grads, c, spec

