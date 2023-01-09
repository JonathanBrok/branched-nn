import numpy as np
import torch

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


def calc_activated_branches(c, thresh=0.1):
    """
    :param c: numpy correlation matrix
    :param thresh: threshold of relative activity for a branch to be considered active. 0.1 is the 10% threshold mentioned in the paper, hence hard-coded
    :return: No. of active branches
    """
    importance = calc_importance(c)
    return np.sum(np.greater_equal(importance, thresh * np.max(importance)))


def calc_specialization(c):
    num_branches = c.shape[0]
    return (num_branches - np.sum(c ** 2) / np.sum(np.diag(c) ** 2)) / (num_branches - 1)


def calc_importance(c):
    return np.sqrt(np.diag(c) ** 2 / np.sum(np.diag(c) ** 2))


def calc_local_specialization(c):
    return np.sqrt((np.diag(c)) ** 2 / np.sum(c ** 2, axis=0))


def get_branch_specialization_measures(out_branches_list, return_additional_measures=False):
    """
    :param out_branches_list: a list of same-size pytorch-tensors or numpy-arrays
    :param return_additional_measures: whether to return extra meaures (currently unused in our scripts)
    :return: Branch Correlation as well as Specialization measures(s)
    """

    c = branch_correlation(out_branches_list)

    # new definition (same logic as old, but linearly mapped to [0, 1])
    spec = calc_specialization(c)

    if not return_additional_measures:
        return spec, c

    # other measures
    local_spec = calc_local_specialization(c)
    importance = calc_importance(c)
    act = calc_activated_branches(c, thresh=0.1)

    return spec, c, local_spec, importance, act
