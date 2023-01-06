
import copy
from functorch import make_functional_with_buffers, vmap, jacrev
import torch

from core.branched_models import SuperNetBranched

def zeroize_weights(model):
    state_dict = model.state_dict()
    w = state_dict['weight']
    state_dict['weight'] = w * 0
    model.load_state_dict(state_dict)


class LinearizedBufferModel(torch.nn.Module,):
    def __init__(self, model, num_classes, device):
        super(LinearizedBufferModel, self).__init__()
        self.linearized_model = True
        self.device=device
        # 1) initial model, frozen
        self.model = copy.deepcopy(model)  # models are usually passed by reference.. make sure its a deep copy
        self.fnet, self.params, self.buffers = make_functional_with_buffers(model)
        self.params_0 = copy.deepcopy(self.params)

        # 2) sizes
        self.num_classes = num_classes
        self.num_params = len(torch.cat([0 * param.view(-1) for param in self.model.parameters()]))

        # 3) linear learnt model
        self.linear = torch.nn.Linear(self.num_params, 1).to(device)
        # TODO 1) perhaps disregard this zero initialization? It's nice theory-wise but probably hinders performance. 2) a more elegant way of initializing with zeros
        zeroize_weights(model=self.linear)


    def forward(self, X, softmax_output=False):
        for k1, k2 in zip(self.params, self.params_0):
            assert torch.allclose(k1, k2, atol=1e-5)
        f0, grads_times_lin, _ = self.forward_core(X)
        out = grads_times_lin + f0
        if softmax_output:
            out = torch.softmax(out, dim=-1)
        return out

    def forward_core(self, X):
        f0 = self.get_f0(X)  # batch X num_classes
        grads_mat = self.grads(X)  # batch*num_classes X num_parameters  # TODO return support for device without passing through cpu
        grads_times_lin = self.get_grad_times_lin(grads_mat)  # batch X num_classes
        return f0, grads_times_lin, grads_mat

    def get_f0(self, X):
        with torch.no_grad():
            f0 = self.model.forward(X)  # batch X num_classes
        return f0

    def get_grad_times_lin(self, grads_mat):
        grads_times_lin_stacked = self.linear(grads_mat)  # batch * num_classes x 1 (vector w/ singleton dimension)
        grads_times_lin_list = torch.chunk(grads_times_lin_stacked, chunks=self.num_classes, dim=0)  # num_classes elements, each elem is batch x 1 (vector w/ singleton dimension)
        grads_times_lin = torch.cat(grads_times_lin_list, dim=1)  # batch x num_classes
        return grads_times_lin

    def set_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.fnet, self.params, self.buffers = make_functional_with_buffers(self.model)
        self.params_0 = copy.deepcopy(self.params)


    def model_single(self, params, buffers, x):
        return self.fnet(params, buffers, x.unsqueeze(0)).squeeze(0)


    def grads(self, X, split_output_dim_to_list=False):
        __grads = vmap(jacrev(self.model_single), (None, None, 0), randomness='same')  # perform jacrev, vectorized over the batch dimension. (None, 0) specifies theat batch dimension is second input's 0 dimension, where the "None"s indicate to skip the first two inputs, which are parameters and buffers

        grads_tuple_transposed_unflattened = __grads(self.params, self.buffers, X)  # each elem in tuple is a gradient for a different layer, and is a tensor of size [batch x out_dim x num_layer_params]

        # begin a quite lengthy reshaping procedure
        grads_list_transposed_unflattened = list(grads_tuple_transposed_unflattened)

        grads_list_transposed = [g.flatten(start_dim=2).detach() for g in grads_list_transposed_unflattened]  # flatten all dimensions that are not output-dim or batch-dim (and detach)

        grads_list_transposed_chunked = [torch.chunk(g, chunks=g.shape[1], dim=1) for g in grads_list_transposed] # Convert grads_list_transposed to a nested list as follows: Each elem is chunked along output dimension (output_dim)

        grads_list_chunked = list(map(list, zip(*grads_list_transposed_chunked)))  # transpose so that output_dim chunks are first dim x ension of nested lists (and second dimensions becomes layer domension)

        grads_list = [torch.cat(g, dim=-1).squeeze(1) for g in grads_list_chunked]  # un-nest the nested lists by concatenatating the layer-dimension - thus we have a output_dim length list.

        if split_output_dim_to_list:
            return grads_list  # a output_dim length list of gradients

        grads_mat = torch.cat(grads_list, dim=0)

        print_debug = False  # TODO erase (or set True to examine the reshaping procedure)
        if print_debug:
            print('len(grads_list_transposed_unflattened)')
            print(len(grads_list_transposed_unflattened))
            print('grads_list_transposed_unflattened[0].shape')
            print(grads_list_transposed_unflattened[0].shape)
            print('grads_list_transposed_unflattened[-1].shape')
            print(grads_list_transposed_unflattened[-1].shape)
            for bla in grads_list_transposed_unflattened:
                print(bla.shape[1])
            for bla in grads_list_transposed_unflattened:
                print(bla.shape[2])

            print('len(grads_list_transposed)')
            print(len(grads_list_transposed))
            print('grads_list_transposed[0].shape')
            print(grads_list_transposed[0].shape)
            print('grads_list_transposed[-1].shape')
            print(grads_list_transposed[-1].shape)
            for bla in grads_list_transposed:
                print(bla.shape[1])
            for bla in grads_list_transposed:
                print(bla.shape[2])

            print('len(grads_list_transposed_chunked)')
            print(len(grads_list_transposed_chunked))
            print('len(grads_list_chunked[0])')
            print(len(grads_list_transposed_chunked[0]))
            print('len(grads_list_chunked)')
            print(len(grads_list_chunked))
            print('len(grads_list_chunked[0])')
            print(len(grads_list_chunked[0]))
            print('len(grads_list)')
            print(len(grads_list))
            print('grads_list[0].shape')
            print(grads_list[0].shape)
            print('grads_list[-1].shape')
            print(grads_list[-1].shape)

            print('grads_mat.shape')
            print(grads_mat.shape)

        return grads_mat


class BranchedLinearizedModel(SuperNetBranched):
    """
    Linearize a branched NN. Remark: By linearity of the gradient, forward method is unchanged (hence not overridden)
    """
    def __init__(self, branched_nn, device):
        super().__init__(branched_nn.num_branches)
        self.linearized_model=True
        self.branched_model = True
        self.num_classes = branched_nn.num_classes

        # construct linearized branches
        for branch in branched_nn.branches:
            cur_branch_net = LinearizedBufferModel(branch, num_classes=branched_nn.branches[0].num_classes, device=device)
            self.branches.append(cur_branch_net)

    # additional methods for linearized models
    def get_branch_grads(self, x, split_output_dim_to_list=False):
        branch_grads = [branch.grads(x, split_output_dim_to_list=split_output_dim_to_list) for branch in self.branches]  # each element in list is batch*num_classes X num_parameters
        return branch_grads

    def get_branch_f0s(self, x):
        branch_f0s = [branch.get_f0(x) for branch in self.branches]  # each element in list is batch X num_classes
        return branch_f0s

    def get_branch_grad_times_lin(self, x):
        branch_grad_times_lins = [branch.model.get_grads_times_lin(x) for branch in self.branches]  # each element in list is batch X num_classes
        return branch_grad_times_lins

    def branched_forward_core(self, x):
        f0_list = []
        grads_times_lin_list = []
        grads_mat_list = []
        for branch in self.branches:
            f0, grads_times_lin, grads_mat = branch.forward_core(x)
            f0_list.append(f0)
            grads_times_lin_list.append(grads_times_lin)
            grads_mat_list.append(grads_mat)
        return f0_list, grads_times_lin_list, grads_mat_list

if __name__ =='__main__':
    # test linearization vs un-linearized model
    from models.models import Net
    from train_branched_models_on_mnist import get_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test linearization around initialization
    eps = 1e-3
    num_classes=10
    batch_size = 16

    trainloader, testloader = get_data_loaders(train_batch_size=batch_size, test_batch_size=batch_size, device=device)

    net = Net()
    net_lin = LinearizedBufferModel(net, num_classes=num_classes, device=device)
    print(len(net_lin.params_0))
    theta = net_lin.params_0
    p = eps * torch.ones_like(theta)  # perturbation vector
    theta_p = theta + p  # suffix _p for perturbed


    net.eval()
    test_loss = 0
    correct = 0
    for x, target in testloader:
        x, target = x.to(device), target.to(device)
        f = net_lin.fnet(theta, x)  # we use fnet (instead of net), because grads are ordered according to fnet
        assert torch.allclose(f, net(x))  # assert fnet does what its supposed to do
        f_p = net_lin.fnet(theta_p, x)
        grads_mat = net_lin.grads(x)  # batch*num_classes X num_parameters
        g_times_p_stacked = torch.matmul(grads_mat, p)  # batch*num_classes X 1
        # reshape as in get_grad_times_lin
        g_times_p = torch.cat(torch.chunk(g_times_p_stacked, chunks=num_classes, dim=0), dim=1)  # batch x num_classes

        linearization_err = torch.sum((f_p - f - g_times_p) ** 2)
        print('linearization error: {}'.format(linearization_err))


        # cur_loss = criterion(output, target)
        # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability



    # test linearization around optimized checkpoint

    # test linearization between initialization and trained

    # test linearization between initialization and zero parameters









