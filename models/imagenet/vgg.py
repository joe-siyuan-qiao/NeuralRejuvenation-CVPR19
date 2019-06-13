'''VGG for ImageNet. FC layers are removed.
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

from .. import nr_modules as nr


__all__ = [
        'vgg_nr',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.cfg = ['I', 64, 'IM', 128, 128, 'M', 256, 256, 256, 256, 'M',
                512, 512, 512, 512, 'M', 512, 512, 512, 512]
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == 'IM':
                layers.append(nr.MaxPool2d(kernel_size=3, stride=2, padding=1))
            elif v == 'I':
                layers.append(nr.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, is_input=True))
                layers.append(nr.ReLU(inplace=True))
                in_channels=64
            elif v == 'M':
                layers.append(nr.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nr.Conv2d(in_channels, v, kernel_size=3,
                    padding=1, is_input=False))
                layers.append(nr.ReLU(inplace=True))
                in_channels = v
        layers.append(nr.ConvLinear(512, num_classes))
        self.network = nr.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        x = self.network(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, '__is_linear__'):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                    continue
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def inspect(self):
        live_neurons = torch.cuda.ByteTensor([1, 1, 1])
        live_neurons, live_params, all_params = self.network.inspect(live_neurons)
        return live_params, all_params

    def rejuvenate(self):
        live_params, all_params = self.inspect()
        expand_rate = float(all_params) / float(live_params)
        self.network.rejuvenate(math.sqrt(expand_rate))

    def bn_restore(self, bn_target):
        self.network.bn_restore(bn_target=bn_target)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(**kwargs)
    return model

def vgg_nr(**kwargs):
    return vgg19_bn(**kwargs)
