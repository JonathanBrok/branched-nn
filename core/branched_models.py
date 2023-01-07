from __future__ import print_function

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
