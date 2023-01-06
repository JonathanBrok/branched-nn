import matplotlib.pyplot as plt
from models.branched_models import NetBranched
from models.models import Net
from train_branched_models_on_mnist import get_data_loaders as get_mnist_loaders

from moore_penrose_functions import *


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




if __name__ == '__main__':

    ms =  [1024]
    l = 8
    maxlogk = 5  # 10
    distribution_mode_list = ['grad', 'normal', 'mult']  # distribution_mode_list = ['iid', 'mult']
    calc_eigenvals = False  # set False to avoid expensive calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





    def wrapper_simulate_toy_branched_linearized(m, k, l, distribution_mode='mult'):
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
        spec, c = specialization(y_hat)

        plt.figure()
        plt.plot(x, label='pseudo inverse result')
        plt.plot(x_thry, '.', label='theoretical equivalent')
        plt.legend()

        plt.figure()
        plt.imshow(c)

        return y, y_hat, x, y_thry_hat, x_thry, a, c, spec

    plt.figure(101)
    if distribution_mode_list[0] == 'grad':
        num_toy_samples = []
    for i, m in enumerate(ms):
        # set
        ks = [2 ** logk for logk in range(maxlogk)]
        rs = [1 / l for _ in range(len(ks))]
        specs = []
        eigenvals = []

        # data, target = get_mnist_45_batch(batch_size=batch_size)
        two_class = True  # only 4 and 5 digits (works with scalar output)
        trainloader, testloader = get_mnist_loaders(train_batch_size=m, test_batch_size=m, device=device, two_class=two_class)


        # analyze
        specs_grad = []
        specs_normal = []
        specs_mult = []
        for k in ks:



            # branched nn
            def constructor():
                return Net(softmax_output=False, num_classes=1, first_depth=k).to(device)
            branched_net = NetBranched(branch_constructor=constructor, num_branches=l).to(device)


            y, y_hat, x, y_thry_hat, x_thry, a, c, spec_grad_cur = moore_penrose_analysis_for_branched_net(branched_net, testloader, compare_to_thry=True, device=device)


            num_toy_params = 10 * k
            num_branch_params = a[0].shape[1]
            num_toy_samples = round(m * num_toy_params / num_branch_params)  # keep sample to param ratio the same as in the branched nn case

            print('k')
            print(k)
            print('num_branch_params')
            print(num_branch_params)
            print('num_toy_samples (not rounded)')
            print(m * k / num_branch_params)

            y, y_hat, x, y_thry_hat, x_thry, a, c, spec_normal_cur = wrapper_simulate_toy_branched_linearized(num_toy_samples, num_toy_params, l, distribution_mode='normal')

            y, y_hat, x, y_thry_hat, x_thry, a, c, spec_mult_cur = wrapper_simulate_toy_branched_linearized(num_toy_samples, num_toy_params, l, distribution_mode='mult')

            specs_grad.append(spec_grad_cur)
            specs_normal.append(spec_normal_cur)
            specs_mult.append(spec_mult_cur)

            print('l: {}, m: {}, k: {}, spec grad: {}'.format(l, m, k, spec_grad_cur))
            print('l: {}, m: {}, k: {}, spec normal: {}'.format(l, m, k, spec_normal_cur))
            print('l: {}, m: {}, k: {}, spec mult: {}'.format(l, m, k, spec_mult_cur))

            if calc_eigenvals:
                a_total = np.concatenate(a, axis=1)
                u, s, vh = np.linalg.svd(a_total)
                eigenvals += [s]

                plt.figure()
                plt.subplot(len(ms), 3, 3 * i + 2 + j)
                for k, s in zip(ks, eigenvals):
                    plt.plot(s)
                plt.legend(['# Branch Params.: ' + str(k) for k in ks])
                plt.title('Eigen Values')

        plt.figure(101)

        plt.plot(specs_grad, label='grad')
        plt.plot(specs_normal, label='normal')
        plt.plot(specs_mult, label='mult')

        plt.xticks(ticks=range(maxlogk), labels=ks)
        plt.xlabel('Branch Width Factor')
        plt.ylabel('Specialization')
        plt.legend()

    # save
    # plt.savefig('spec_normal_rand_linearized.png', dpi=600)
    plt.show()
