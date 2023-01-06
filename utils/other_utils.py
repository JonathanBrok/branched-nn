import torch
from torch.nn import functional as F


class MyMSELOSS(torch.nn.Module):
    """
    A loss criterion class to calculate MSE loss from a one-hot vector, when the input is given as indices (instead of one-hot)
    """

    def __init__(self, num_classes=10):
        super(MyMSELOSS, self).__init__()
        self.mse_criterion = torch.nn.MSELoss()
        self.num_classes = num_classes

    def forward(self, y_estim, y):
        y = F.one_hot(y, num_classes=self.num_classes).type(torch.FloatTensor).to(device)
        return self.mse_criterion(y_estim, y)


def unnormalize_im(im_batch):
    im_batch_unnormalized = torch.zeros(im_batch.shape)
    for i, im in enumerate(im_batch):
        # approximate unnormalization
        im[0] = im[0] * 0.229 + 0.485
        im[1] = im[1] * 0.224 + 0.456
        im[2] = im[2] * 0.225 + 0.406
        im_batch_unnormalized[i] = im
    return im_batch_unnormalized.clamp_(0, 1)