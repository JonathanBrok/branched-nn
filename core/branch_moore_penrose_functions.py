import numpy as np
from models.linearized_model import BranchedLinearizedModel
import torch
from matplotlib import pyplot as plt

from core.branch_specialization_measures import get_branch_specialization_measures


def simulate_branched_linearized_mse(a, y):
    """
    Considers the Linearization of a Branched NN, optimized on MSE (hence a pseudo-inverse solution is valid, since we have a linear model and mse criterion)
    :param a: a list with l elements where each element is a m X k matrix corresponding to gradients of a different branch
    :param y: labels (each label is a scalar)  # TODO support for vector labels
    :return: x: solved parameters, as a single numpy array
             y_hat: per-branch output, as list of numpy arrays
    """
    # get sizes
    l = len(a)  # No. of branches
    p = a[0].shape[1]  # No. of parameters per branch

    # a single stacked gradient matrix for all branches
    a_total = np.concatenate(a, axis=1)

    # solve for parameters
    x = np.linalg.pinv(a_total) @ y

    # solve branch outputs
    y_hat = []  # initialize branch outputs
    for i in range(l):
        y_hat.append(a[i] @ x[i * p:(i + 1) * p, ])

    return x, y_hat


def simulate_theoretical_branched_linearized_mse(a, y):
    """
    Solves the same problem as the above function "simulate_branched_linearized_mse", via an alternative kernel=based mathematical expression (see paper)
    """

    # prepare gradient matrix and kernel
    a_total = np.concatenate(a, axis=1)  # a single stacked gradient matrix for all branches
    aa = a_total @ np.transpose(a_total)  # gradient kernel matrix

    # apply theoretical per-branch solution
    aa_inv = np.linalg.pinv(aa)  # inverse kernel. Gradient kernel (aa) is assumed to be full rank and hence the pseudo-inverse is infact an inverse
    x_list = []
    for a_i in a:
        x_i = np.transpose(a_i) @ aa_inv @ y
        x_list += [x_i]

    # obtain branch outputs theoretical branch solution
    y_hat = []  # initialize branch outputs
    for a_i, x_i in zip(a, x_list):
        y_hat.append(a_i @ x_i)

    # return
    x = np.concatenate(x_list)  # concatenate to be consistent with the return value of "simulate_branched_linearized_mse".
    return x, y_hat


def get_grads_angles(grads):
    """
    :param grads: a list of same-size numpy arrays. of length L
    :return: grads_cosine: An L x L numpy matrix, where each element is the cosine distance between two flattened elements of grads
            grads_angles: A list. for every pair of grads, there's a scalar element in the list which is the angle in degrees
    """

    num_branches = len(grads)

    # correlation
    grads_flat = [np.ndarray.flatten(g_cur) for g_cur in grads]  # L list of MX1 arrays, where M is th number of elements in each g_cur
    grads_flat_cat = np.stack(grads_flat, axis=1)  # M x L array
    grads_corr = np.transpose(grads_flat_cat) @ grads_flat_cat  # L x L array

    # cross-magnitudes
    grads_mag = [np.sqrt(np.sum(g_cur ** 2)) for g_cur in grads_flat]  # L list
    grads_mag_cat = np.asarray(grads_mag)  # L array of scalars
    grads_mag_cat_singleton = grads_mag_cat[:, None]  # L x 1 array
    grads_mag_cross_matrix = grads_mag_cat_singleton @ np.transpose(grads_mag_cat_singleton)  # L x L array

    # cosine
    grads_cosine = grads_corr / grads_mag_cross_matrix  # L x L
    print('grads_cosine.shape')
    print(grads_cosine.shape)
    get_upper_tri_as_vector = lambda a, l: a[np.triu_indices(l, k=1)]
    grads_angles = np.rad2deg(np.arccos(np.abs(get_upper_tri_as_vector(grads_cosine, num_branches))))

    return grads_cosine, grads_angles


def construct_grad_out_angle(branched_net, testloader, device):
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        break  # single batch

    grads, out = get_initial_grad_f(branched_net, data, device)
    grads = [g_cur.data.cpu().numpy() for g_cur in grads]  # from torch to numpy
    _, grads_angles = get_grads_angles(grads)
    return grads, out, grads_angles


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
        x_thry, y_thry_hat = simulate_theoretical_branched_linearized_mse(a=grads, y=target)
    else:
        x_thry, y_thry_hat = [], []

    x, y_hat = simulate_branched_linearized_mse(a=grads, y=target)


    spec, c, local_spec, importance, act = get_branch_specialization_measures(y_hat, return_additional_measures=True)

    return target, y_hat, x, y_thry_hat, x_thry, grads, c, spec, act


def moore_penrose_analysis_with_toy_gradients(m, k, l, distribution_mode):
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
    spec, c, local_spec, importance, act = get_branch_specialization_measures(y_hat, return_additional_measures=True)

    return y, y_hat, x, y_thry_hat, x_thry, a, c, spec, act


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
