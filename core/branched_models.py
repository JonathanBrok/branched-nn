from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
# from models.models import *


class SuperNetBranched(nn.Module):
    def __init__(self, num_branches):
        super(SuperNetBranched, self).__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList()
        self.softmax_output = None
        self.linearized_model = False
        self.branched_model = True
        # construct branches! (upon inheritance)

    def forward(self, x, return_branch_output=False):

        out_branches_list = [net(x, softmax_output=False) for net in self.branches]  # Note that branch output is never softmaxed. Softmax is performed (if it is performed) after branch summation
        out_branches_newdim = torch.transpose(torch.stack(out_branches_list), 1, 0)  # batch X branch X out_channels  X 1 X 1

        out = torch.sum(out_branches_newdim, dim=1)  # sum branches to a tensor of # batch X out_channels  X 1 X 1
        out = torch.squeeze(out)

        if self.softmax_output:
            out = torch.softmax(out, dim=-1)

        if return_branch_output:
            return out, out_branches_list
        else:
            return out


class NetBranched(SuperNetBranched):
    def __init__(self, branch_constructor, num_branches):
        super().__init__(num_branches)
        self.num_branches = num_branches
        self.branches = nn.ModuleList()

        # construct branches
        for _ in range(self.num_branches):
            cur_branch_net = branch_constructor()
            self.branches.append(cur_branch_net)

        self.softmax_output = self.branches[0].softmax_output
        self.num_classes = self.branches[0].num_classes


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

        act_thresh = 0.1  # threshold of relative activity for a branch to be considered active. 0.1 is the 10% threshold mentioned in the paper, hence hard-coded
        act = np.sum(np.greater(importance, act_thresh * np.max(importance)))

        return spec, c, local_spec, importance, act