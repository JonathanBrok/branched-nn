from core.branched_models import NetBranched
from models.models import Net
from train_branched_models_on_mnist import get_data_loaders as get_mnist_loaders

from core.moore_penrose_functions import *


def compare_grad_normal_mult_helper(m, k, l, testloader, do_show=True, calc_eigenvals=False):
    """
    compares properties of the Branched Moore-Penrose solution for a specific value of m,, k, l  
    """
    # branched nn
    def constructor():
        return Net(softmax_output=False, num_classes=1, first_width=k).to(device)

    branched_net = NetBranched(branch_constructor=constructor, num_branches=l).to(device)

    y_grad, y_hat_grad, x_grad, y_thry_hat_grad, x_thry_grad, a_grad, c_grad, spec_grad, act_grad = moore_penrose_analysis_for_branched_net(branched_net, testloader, compare_to_thry=True, device=device)

    # get No. of parameters in branch of branched nn
    branch_0 = branched_net.branches[0]
    num_branch_params = sum(p.numel() for p in branch_0.parameters())  # or equivalently, a[0].shape[1]
    # declare No. of parameters in toy gradient
    num_toy_params = 10 * k
    # calculate No. of samples for toy, keeping sample to param ratio the same as in the branched nn case
    num_toy_samples = round(m * num_toy_params / num_branch_params)

    y_normal, y_hat_normal, x_normal, y_thry_hat_normal, x_thry_normal, a_normal, c_normal, spec_normal, act_normal = wrapper_simulate_toy_branched_linearized(num_toy_samples, num_toy_params, l, distribution_mode='normal')

    y_mult, y_hat_mult, x_mult, y_thry_hat_mult, x_thry_mult, a_mult, c_mult, spec_mult, act_mult = wrapper_simulate_toy_branched_linearized(num_toy_samples, num_toy_params, l, distribution_mode='mult')

    print('l: {}, m: {}, k: {}, spec grad: {}, act grad {}'.format(l, m, k, spec_grad, act_grad))
    print('l: {}, m: {}, k: {}, spec normal: {}, act normal {}'.format(l, m, k, spec_normal, act_normal))
    print('l: {}, m: {}, k: {}, spec mult: {}, act mult {}'.format(l, m, k, spec_mult, act_mult))

    if do_show:
        plt.figure()
        plt.subplot(3, 2, 1)
        plt.plot(x_normal, label='pseudo inverse result')
        plt.plot(x_thry_normal, '.', label='theoretical equivalent')
        plt.title('normal, l={}, k={}, m={}'.format(l, k, m))
        plt.legend()
        plt.subplot(3, 2, 2)
        plt.imshow(c_normal)
        plt.colorbar()

        plt.subplot(3, 2, 3)
        plt.plot(x_mult, label='pseudo inverse result')
        plt.plot(x_thry_mult, '.', label='theoretical equivalent')
        plt.title('mult, l={}, k={}, m={}'.format(l, k, m))
        plt.legend()
        plt.subplot(3, 2, 4)
        plt.imshow(c_mult)
        plt.colorbar()

        plt.subplot(3, 2, 5)
        plt.plot(x_grad, label='pseudo inverse result')
        plt.plot(x_thry_grad, '.', label='theoretical equivalent')
        plt.title('grad, l={}, k={}, m={}'.format(l, k, m))
        plt.legend()
        plt.subplot(3, 2, 6)
        plt.imshow(c_grad)
        plt.colorbar()

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

    return spec_grad, spec_normal, spec_mult, act_grad, act_normal, act_mult


def compare_grad_normal_mult(m_list, k_list, l_list, testloader, show_along=None):
    """
    Plots Specialization and Activation as function of the zipped values. (each of the cases normal, mult and grad has an individual plot).
    """
    if show_along is not None:
        assert show_along in ('m', 'l', 'k')

    # analyze
    specs_grad = []
    specs_normal = []
    specs_mult = []
    acts_grad = []
    acts_normal = []
    acts_mult = []
    for m, k, l in zip(m_list, k_list, l_list):
        spec_grad_cur, spec_normal_cur, spec_mult_cur, act_grad_cur, act_normal_cur, act_mult_cur = compare_grad_normal_mult_helper(m, k, l, testloader)

        specs_grad.append(spec_grad_cur)
        specs_normal.append(spec_normal_cur)
        specs_mult.append(spec_mult_cur)
        acts_grad.append(act_grad_cur)
        acts_normal.append(act_normal_cur)
        acts_mult.append(act_mult_cur)

    specs_grad = np.asarray(specs_grad)
    specs_normal = np.asarray(specs_normal)
    specs_mult = np.asarray(specs_mult)
    acts_grad = np.asarray(acts_grad)
    acts_normal = np.asarray(acts_normal)
    acts_mult = np.asarray(acts_mult)

    if show_along is not None:
        if show_along == 'm':
            xticks = np.arange(len(m_list))
            x_vals = m_list
            x_label = 'No. of Samples'
        elif show_along == 'k':
            xticks = np.arange(len(k_list))
            x_vals = k_list
            x_label = 'Branch Width Factor'
        elif show_along == 'l':
            xticks = np.arange(len(l_list))
            x_vals = l_list
            x_label = 'No. of Branches'

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(specs_grad, label='grad')
        plt.plot(specs_normal, label='normal')
        plt.plot(specs_mult, label='mult')
        plt.ylabel('Specialization')
        plt.xticks(ticks=xticks, labels=x_vals)
        plt.xlabel(x_label)

        plt.legend()


        plt.subplot(1, 2, 2)
        width = 0.1
        plt.bar(xticks, acts_grad, width, label='grad')
        plt.bar(xticks + width, acts_normal, width, label='normal')
        plt.bar(xticks + 2 * width, acts_mult, width, label='mult')
        plt.ylabel('No. of Active Branches')
        plt.xticks(ticks=xticks, labels=x_vals)
        plt.xlabel(x_label)
        plt.legend()

    return specs_grad, specs_normal, specs_mult, acts_grad, acts_normal, acts_mult



if __name__ == '__main__':

    m =  1024
    l = 8
    k = 2
    maxlogk = 4  # 5
    maxlogl = 6
    num_expr = 10

    distribution_mode_list = ['grad', 'normal', 'mult']  # distribution_mode_list = ['iid', 'mult']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data, target = get_mnist_45_batch(batch_size=batch_size)
    two_class = True  # only 4 and 5 digits (works with scalar output)
    trainloader, testloader = get_mnist_loaders(train_batch_size=m, test_batch_size=m, device=device, two_class=two_class)

    # 2) compare along increasing No. of branches
    l_list_2 = [2 ** elem for elem in range(1, maxlogl)]
    m_list_2 = [m] * len(l_list_2)
    k_list_2 = [k] * len(l_list_2)

    show_along = 'l'
    specs_grad, specs_normal, specs_mult, acts_grad, acts_normal, acts_mult = compare_grad_normal_mult(m_list_2, k_list_2, l_list_2, testloader, show_along=show_along)
    show_along = None

    specs_grad_expr = []
    specs_normal_expr = []
    specs_mult_expr = []
    acts_grad_expr = []
    acts_normal_expr = []
    acts_mult_expr = []
    for _ in range(num_expr):
        specs_grad, specs_normal, specs_mult, acts_grad, acts_normal, acts_mult = compare_grad_normal_mult(m_list_2, k_list_2, l_list_2, testloader, show_along=show_along)
        specs_grad_expr += [specs_grad]
        specs_normal_expr += [specs_normal]
        specs_mult_expr += [specs_mult]
        acts_grad_expr += [acts_grad]
        acts_normal_expr += [acts_normal]
        acts_mult_expr += [acts_mult]

    def helper_mean_var(measure_expr):
        measure_expr = np.stack(measure_expr, axis=1)
        measure_mean = np.mean(measure_expr, axis=1)
        measure_var = np.var(measure_expr, axis=1)
        return measure_mean, measure_var

    specs_mult_mean, specs_mult_var = helper_mean_var(specs_mult_expr)
    acts_mult_mean, acts_mult_var = helper_mean_var(acts_mult_expr)
    specs_grad_mean, specs_grad_var = helper_mean_var(specs_grad_expr)
    acts_grad_mean, acts_grad_var = helper_mean_var(acts_grad_expr)
    specs_normal_mean, specs_normal_var = helper_mean_var(specs_normal_expr)
    acts_normal_mean, acts_normal_var = helper_mean_var(acts_normal_expr)

    plt.figure()
    def helper_plot_with_shaded_error_region(y, y_var, label, x=None):
        error = 0.5 * np.sqrt(y_var)
        if x is None:
            x = range(y.shape[0])
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - error, y + error, alpha=0.5)

    xticks = np.arange(len(l_list_2))
    x_vals = l_list_2
    x_label = 'No. of Branches'

    plt.subplot(1, 2, 1)
    helper_plot_with_shaded_error_region(specs_grad_mean, specs_grad_var, label='grad', x=xticks)
    helper_plot_with_shaded_error_region(specs_normal_mean, specs_normal_var, label='normal', x=xticks)
    helper_plot_with_shaded_error_region(specs_mult_mean, specs_mult_var, label='mult', x=xticks)
    plt.ylabel('Specialization')
    plt.xticks(ticks=xticks, labels=x_vals)
    plt.xlabel(x_label)
    plt.legend()

    plt.subplot(1, 2, 2)
    # helper_plot_with_shaded_error_region(acts_grad_mean, acts_grad_var, label='grad', x=None)
    # helper_plot_with_shaded_error_region(acts_normal_mean, acts_normal_var, label='normal', x=None)
    # helper_plot_with_shaded_error_region(acts_mult_mean, acts_mult_var, label='mult', x=None)
    # plt.ylabel('No. of Active Branches')
    # plt.xticks(ticks=xticks, labels=x_vals)
    # plt.xlabel(x_label)
    # plt.legend()

    width = 0.1
    plt.bar(xticks, acts_grad, width, yerr=0.5*np.sqrt(acts_grad_var), label='grad')
    plt.bar(xticks + width, acts_normal, width, yerr=0.5*np.sqrt(acts_normal_var),label='normal')
    plt.bar(xticks + 2 * width, acts_mult, width, yerr=0.5*np.sqrt(acts_mult_var), label='mult')
    plt.ylabel('No. of Active Branches')
    plt.xticks(ticks=xticks, labels=x_vals)
    plt.xlabel(x_label)
    plt.legend()

    # 1) compare along increasing width of each bbanch
    k_list_1 = [2 ** elem for elem in range(maxlogk)]
    m_list_1 = [m] * len(k_list_1)
    l_list_1 = [l] * len(k_list_1)
    specs_grad, specs_normal, specs_mult,acts_grad, acts_normal, acts_mult = compare_grad_normal_mult(m_list_1, k_list_1, l_list_1, testloader, show_along='k')



    # save
    # plt.savefig('spec_normal_rand_linearized.png', dpi=600)
    plt.show()
