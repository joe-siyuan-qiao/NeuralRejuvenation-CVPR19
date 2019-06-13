import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from .. import nr_modules as nr
import math

__all__ = ['densenet121_nr', 'densenet169_nr']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121_nr(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169_nr(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def CreateBottleneck(inplanes, expansion=4, growthRate=12):
    planes = expansion * growthRate
    conv1 = nr.Conv2d(inplanes, planes, kernel_size=1, bias=False, split_output=True)
    relu1 = nr.ReLU(inplace=False)
    conv2 = nr.Conv2d(planes, growthRate, kernel_size=3, padding=1, bias=False, split_output=True)
    relu2 = nr.ReLU(inplace=False)
    block = nr.ConcatList()
    long_path = nr.Sequential(conv1, relu1, conv2, relu2)
    short_cut = nr.Identity()
    block.add(short_cut)
    block.add(long_path)
    return [block, nr.JoinList(1)]


def CreateTransition(inplanes, outplanes):
    conv = nr.Conv2d(inplanes, outplanes, kernel_size=1, bias=False, split_output=True)
    relu = nr.ReLU(inplace=False)
    avgp = nr.AvgPool2d(2)
    return [conv, relu, avgp]


def CreateBlock(num_layers, num_input_features, bn_size, growth_rate, drop_rate):
    layers = []
    for i in range(num_layers):
        layers += CreateBottleneck(num_input_features + i * growth_rate, bn_size, growth_rate)
    return layers


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        conv1 = nr.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3,
                            bias=False, is_input=True, split_output=True)
        relu = nr.ReLU(inplace=False)
        pool = nr.MaxPool2d(kernel_size=3, stride=2, padding=1)
        features = [conv1, relu, pool]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = CreateBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            num_features = num_features + num_layers * growth_rate
            features += block
            if i != len(block_config) - 1:
                trans = CreateTransition(num_features, num_features // 2)
                features += trans
                num_features = num_features // 2
        features.append(nr.ConvLinear(num_features, num_classes))
        self.network = nr.Sequential(*features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, '__is_linear__'):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

    def inspect(self, flops_flag):
        live_neurons = torch.cuda.ByteTensor([1, 1, 1])
        dummy, live_params, all_params = self.network.inspect(live_neurons=live_neurons, flops_flag=flops_flag)
        return live_params, all_params

    def rejuvenate(self, flops_flag):
        live_params, all_params = self.inspect(flops_flag=flops_flag)
        expand_rate = math.sqrt(float(all_params) / float(live_params))
        self.network.rejuvenate(expand_rate, flops_flag=flops_flag)

    def bn_restore(self, bn_target):
        self.network.bn_restore(bn_target=bn_target)

