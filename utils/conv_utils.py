import torch

def merge_conv_kernels(k1, k2):
    """
    implementation from https://stackoverflow.com/questions/58357815/how-do-i-merge-2d-convolutions-in-pytorch
    :input k1: A tensor of shape ``(out1, in1, s1, s1)``
    :input k1: A tensor of shape ``(out2, in2, s2, s2)``
    :returns: A tensor of shape ``(out2, in1, s1+s2-1, s1+s2-1)``
      so that convolving with it equals convolving with k1 and
      then with k2.
    """
    padding = k2.shape[-1] - 1
    # Flip because this is actually correlation, and permute to adapt to BHCW
    k3 = torch.conv2d(k1.permute(1, 0, 2, 3), k2.flip(-1, -2), padding=padding).permute(1, 0, 2, 3)
    return k3


def merge_conv_layers(conv1, conv2, im_siz=None):
    """
    Uses the function merge_conv_kernels (above) to merge conv1, conv2 - assuming they are applied as conv2(conv1(x)).
    Requires:
     1) "full" padding, i.e. kernel_size-1 padding on each side.
     2) square kernels only
    """

    # get original kernels and sizes
    k1 = conv1.weight.data
    k2 = conv2.weight.data
    out1, in1, s1, s1a = k1.shape
    out2, in2, s2, s2a = k2.shape

    # assert supported case
    if  s1 != s1a or s2 != s2a:
        raise ValueError  # non-square kernels are not supported

    # merge k1, k2 to k3
    k3 = merge_conv_kernels(k1, k2)
    out3, in3, s3, _ = k3.shape

    if conv1.padding[0] == s1 - 1 and conv2.padding[0] == s2 - 1:
        padding = s3 - 1

    elif conv1.padding == 'same' and conv2.padding == 'valid':
        # This is a very specific case, where first layer is "same", and second layer prodices a completely flattens output.
        to_cut = int((s3 - s2) / 2)
        assert (s3 - s2) / 2 == to_cut
        assert s2 == im_siz[0] == im_siz[1]
        k3 = k3[:, :, to_cut:-to_cut, to_cut:-to_cut]
        padding = 'valid'

    else:
        print(conv1.padding)
        print(conv2.padding)
        raise ValueError  # only full padding, or 'same' then 'valid' is fully supported


    # use k3 to initialize a nn.conv2d layer
    conv3 = torch.nn.Conv2d(in3, out3, kernel_size=s3, stride=1, padding=padding, bias=False)
    conv3.weight = torch.nn.Parameter(k3)
    return conv3