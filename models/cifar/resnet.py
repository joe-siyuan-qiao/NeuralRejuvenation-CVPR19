from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import math

from .. import nr_modules as nr

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def CreateBasicBlock(inplanes, planes, stride=1, downsample=None):
    block = nr.ConcatList()
    conv1 = nr.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
    relu1 = nr.ReLU(inplace=True)
    conv2 = nr.Conv2d(planes, planes, kernel_size=3, padding=1, is_fixed=True)
    long_path = nr.Sequential(conv1, relu1, conv2)
    if downsample is not None:
        short_cut = downsample
    else:
        short_cut = nr.Identity()
    block.add(long_path)
    block.add(short_cut)
    return [block, nr.CAddList(), nr.ReLU(inplace=True)]

def CreateBottleneck(inplanes, planes, stride=1, downsample=None):
    block = nr.ConcatList()
    conv1 = nr.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    relu1 = nr.ReLU(inplace=True)
    conv2 = nr.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    relu2 = nr.ReLU(inplace=True)
    conv3 = nr.Conv2d(planes, planes * 4, kernel_size=1, bias=False, is_fixed=True)
    long_path = nr.Sequential(conv1, relu1, conv2, relu2, conv3)
    if downsample is not None:
        short_cut = downsample
    else:
        short_cut = nr.Identity()
    block.add(long_path)
    block.add(short_cut)
    return [block, nr.CAddList(), nr.ReLU(inplace=True)]


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = CreateBottleneck if depth >=44 else CreateBasicBlock
        self.block_expansion = 4 if depth >=44 else 1

        self.wide_factor = 1

        self.inplanes = 16 * self.wide_factor
        conv1 = nr.Conv2d(3, 16 * self.wide_factor, kernel_size=3, padding=1,
                               bias=False, is_input=True, is_fixed=True)
        relu = nr.ReLU(inplace=True)
        layers = [conv1, relu]
        layers += self._make_layer(block, 16 * self.wide_factor, n)
        layers += self._make_layer(block, 32 * self.wide_factor, n, stride=2)
        layers += self._make_layer(block, 64 * self.wide_factor, n, stride=2)
        fc = nr.ConvLinear(64 * self.block_expansion * self.wide_factor, num_classes)
        layers += [fc]

        self.network = nr.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block_expansion:
            downsample = nr.Conv2d(self.inplanes, planes * self.block_expansion,
                    kernel_size=1, stride=stride, bias=False, is_fixed=True)
        layers = []
        layers += block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * self.block_expansion
        for i in range(1, blocks):
            layers += block(self.inplanes, planes)

        return layers

    def forward(self, x):
        return self.network(x)

    def inspect(self, flops_flag=None):
        live_neurons = torch.cuda.ByteTensor([1, 1, 1])
        live_neurons, live_params, all_params = self.network.inspect(live_neurons)
        return live_params, all_params

    def rejuvenate(self, flops_flag=None):
        live_params, all_params = self.inspect()
        expand_rate = math.sqrt(float(all_params) / float(live_params))
        self.network.rejuvenate(expand_rate)


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
