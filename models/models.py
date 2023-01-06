'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.conv_utils import merge_conv_layers

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, softmax_output=False):
        super(ResNet, self).__init__()
        self.linearized_model = False
        self.branched_model = False
        self.softmax_output = softmax_output
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, softmax_output=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        if softmax_output is None:  # use default value unless specified. TODO: add assertion for valid values
            softmax_output = self.softmax_output  

        if softmax_output:
            out = torch.softmax(out, dim=-1)
        return out


def ResNet18(num_classes, softmax_output):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, softmax_output)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


class Net_lin(nn.Module):
    def __init__(self, softmax_output=False, num_classes=10):  
        super(Net_lin, self).__init__()
        self.linearized_model = False  # Even though it is linear, it is not linearized
        self.branched_model = False
        self.softmax_output = softmax_output
        self.conv = nn.Conv2d(1, num_classes, kernel_size=28, stride=1, bias=False)
        self.num_classes=num_classes        

    def forward(self, x, softmax_output=None):
        x = self.conv(x)
        out = x.squeeze(dim=-1).squeeze(dim=-1)  # squeeze image height and width dimensions
        
        if softmax_output is None:  # use default value unless specified. TODO: add assertion for valid values
            softmax_output = self.softmax_output
        
        if softmax_output:
            out = torch.softmax(out, dim=-1)
        return out


class Net_conv_composed(nn.Module):
    def __init__(self, softmax_output=False):
        super(Net_conv_composed, self).__init__()
        self.linearized_model = False  # Even though it is linear, it is not linearized
        self.branched_model = False
        self.softmax_output = softmax_output
        self.conv = nn.Conv2d(1, 10, kernel_size=28, stride=1, bias=False)

    def forward(self, x, softmax_output=None):
        x = self.conv(x)
        out = x.squeeze(dim=-1).squeeze(dim=-1)  # squeeze image height and width dimensions

        if softmax_output is None:  # use default value unless specified. TODO: add assertion for valid values
            softmax_output = self.softmax_output

        if softmax_output:
            out = torch.softmax(out, dim=-1)
        return out



class NetFC(nn.Module):  # similar to fully connected nn from NTK's google colab linearization in weight space notebook, available at https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/weight_space_linearization.ipynb
    def __init__(self, softmax_output=False, im_size=[1, 28, 28], num_classes=10):
        super(NetFC, self).__init__()
        if im_size[0] != 1:
            raise NotImplementedError  # TODO support rgb
        self.linearized_model = False
        self.branched_model = False
        self.num_classes = num_classes
        self.softmax_output = softmax_output
        self.conv_fc = nn.Conv2d(1, 512, kernel_size=(im_size[1], im_size[2]), stride=1, bias=False)  # Effectively a fully connected layer. This implementation is convenient for the sake of visualization of linear layers as images
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, softmax_output=None):
        x = self.conv_fc(x)
        x = torch.erf(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)  # squeeze image height and width dimensions
        out = self.fc2(x)

        if softmax_output is None:  # use default value unless specified. TODO: add assertion for valid values
            softmax_output = self.softmax_output

        if softmax_output:
            out = torch.softmax(out, dim=-1)
        return out


class Net(nn.Module):
    def __init__(self, softmax_output=False, first_width=32, im_size=[1, 28, 28], num_classes=10, use_dropout=True):
        super(Net, self).__init__()
        self.linearized_model = False
        self.branched_model = False
        self.softmax_output = softmax_output
        self.num_classes = num_classes
        self.use_dropout = use_dropout


        conv2_factor = 1  # originally 2
        fc_shared_dim = 2  # originally 128
        fc1_first_dim = int(first_width * conv2_factor * ((im_size[1] - 4) / 2) * ((im_size[2] - 4) / 2))  # default: 9216, calulated as (32 * 2 * ((28 - 4) / 2) ^ 2)
        
        self.conv1 = nn.Conv2d(im_size[0], first_width, 3, 1)
        self.conv2 = nn.Conv2d(first_width, conv2_factor * first_width, 3, 1)
        if self.use_dropout:
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(fc1_first_dim, fc_shared_dim)
        self.fc2 = nn.Linear(fc_shared_dim, self.num_classes)

    def forward(self, x, softmax_output=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.use_dropout:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        
        if softmax_output is None:  # use default value unless specified. TODO: add assertion for valid values
            softmax_output = self.softmax_output

        if softmax_output:
            x = torch.softmax(x, dim=-1)

        return x


class Net_two_convs(nn.Module):
    def __init__(self, softmax_output=False, first_width=1, im_size=[1, 28, 28], num_classes=10):
        super(Net_two_convs, self).__init__()
        self.linearized_model = False
        self.branched_model = False
        self.softmax_output = softmax_output
        self.num_classes = num_classes

        # self.conv1 = nn.Conv2d(im_size[0], 1, kernel_size=5, stride=1, padding='same', bias=False)  # out shape: 1 X im_size[0] X im_size[1]
        # self.conv2 = nn.Conv2d(1, self.num_classes, kernel_size=[im_size[1], im_size[2]], padding='valid', bias=False, stride=1)  # out shape: num_classes X 1 X 1

        conv1 = nn.Conv2d(im_size[0], first_width, kernel_size=5, stride=1, padding='same', bias=False)  # out shape: 1 X im_size[0] X im_size[1]
        conv2 = nn.Conv2d(first_width, self.num_classes, kernel_size=[im_size[1], im_size[2]], padding='valid', bias=False, stride=1)  # out shape: num_classes X 1 X 1

        self.merged_conv = merge_conv_layers(conv1, conv2, im_siz=(28, 28))

        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, x, softmax_output=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = torch.squeeze(x)

        if softmax_output is None:  # use default value unless specified. TODO: add assertion for valid values
            softmax_output = self.softmax_output

        if softmax_output:
            x = torch.softmax(x, dim=-1)

        return x


class Net_backup_before_slimming(nn.Module):
    def __init__(self, softmax_output=False, first_depth=32, im_size=[1, 28, 28], num_classes=10):
        super(Net, self).__init__()
        self.linearized_model = False
        self.branched_model = False
        self.softmax_output = softmax_output
        self.num_classes = num_classes

        fc1_first_dim = int(first_depth * 2 * ((im_size[1] - 4) / 2) * (
                    (im_size[2] - 4) / 2))  # default: 9216, calulated as (32 * 2 * ((28 - 4) / 2) ^ 2)

        self.conv1 = nn.Conv2d(im_size[0], first_depth, 3, 1)
        self.conv2 = nn.Conv2d(first_depth, 2 * first_depth, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(fc1_first_dim, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x, softmax_output=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        if softmax_output is None:  # use default value unless specified. TODO: add assertion for valid values
            softmax_output = self.softmax_output

        if softmax_output:
            x = torch.softmax(x, dim=-1)

        return x

