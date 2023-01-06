import numpy as np
from models.linearized_model import BranchedLinearizedModel
import torch
from matplotlib import pyplot as plt

from core.branched_models import specialization


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

    spec, c, local_spec, importance, act = specialization(y_hat, return_additional_measures=True)

    return target, y_hat, x, y_thry_hat, x_thry, grads, c, spec, act


def wrapper_simulate_toy_branched_linearized(m, k, l, distribution_mode):
    """
    :param m: No. of  Data samples
    :param k: No. of Branch Parameters
    :param l: No. of Branches
    :return
        y_hat: Outputs of a linearized toy example "trained" on MSE via pseudo-inverse
        y: the labels
    """
    # toy labels
    y = np.random.normal(size=(m, 1))
    # toy gradients
    a = toy_random_gradients(m, k, l, distribution_mode=distribution_mode)
    # mse linearized pseudo-inverse solution per-branch
    x, y_hat = simulate_branched_linearized_mse(a, y)

    x_thry, y_thry_hat = simulate_theoretical_branched_linearized_mse(a, y)

    # evaluate specialization
    spec, c, local_spec, importance, act = specialization(y_hat, return_additional_measures=True)

    return y, y_hat, x, y_thry_hat, x_thry, a, c, spec, act


def my_cummult(mat, axis=1):
    """
    similarly to cumsum, but with elementwise multipication instead of summation.
    mat is assumed to be a numpy matrix.
    """
    if axis == 1:
        mat = mat.T
    elif axis != 0:
        raise ValueError
    mat_cummult = np.zeros_like(mat)
    cur_mult = np.ones(mat.shape[1])
    for r, mat_row in enumerate(mat):
        cur_mult = cur_mult * mat_row  # element-wise multipication
        mat_cummult[r, :] = cur_mult
    if axis == 1:
        mat_cummult = mat_cummult.T
    return mat_cummult


def toy_random_gradients(m, k, l, distribution_mode='mult'):
    """
    Generate random matrices, which are supposed to approximate the gradients of Branched NN at initialization
    :param m: No. of  Data samples
    :param k: No. of Branch Parameters
    :param l: No. of Branches
    :return: a: a list with l elements where each element is a m X k random matrix corresponding to "toy gradients" of a different branch
    Remark: As in practical NNs, Gradients for each branch are iid (but inter-branch gradients may be dependent, as is the case in branched NNs)
    """
    a = []
    for i in range(l):
        rand_gradient = np.random.random((m, k))
        if distribution_mode == 'summed':
            rand_gradient = np.cumsum(rand_gradient, axis=1)
        elif distribution_mode == 'mult':
            rand_gradient = my_cummult(rand_gradient, axis=1)
        elif distribution_mode != 'normal':
            raise ValueError
        # else its iid, and does not need additional processing
        a.append(rand_gradient)

    return a