import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

from .. import nr_modules as nr


__all__ = [
    'vgg16_nr', 'vgg19_nr'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        classifier = []
        classifier.append(nr.AvgPool2d(kernel_size=14))
        classifier += [nr.Conv2d(512, 4096, kernel_size=1), nr.ReLU(inplace=True)]
        classifier += [nr.Conv2d(4096, 4096, kernel_size=1), nr.ReLU(inplace=True)]
        classifier.append(nr.ConvLinear(4096, num_classes))

        self.network = nr.Sequential(*(features + classifier))
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

    def inspect(self, flops_flag):
        live_neurons = torch.cuda.ByteTensor([1, 1, 1])
        live_neurons, live_params, all_params = self.network.inspect(live_neurons, flops_flag=flops_flag)
        return live_params, all_params

    def rejuvenate(self, flops_flag):
        live_params, all_params = self.inspect(flops_flag=flops_flag)
        expand_rate = float(all_params) / float(live_params)
        self.network.rejuvenate(math.sqrt(expand_rate), flops_flag=flops_flag)

    def bn_restore(self, bn_target):
        self.network.bn_restore(bn_target=bn_target)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    is_input = True
    for v in cfg:
        if v == 'M':
            layers += [nr.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nr.Conv2d(in_channels, v, kernel_size=3, padding=1, is_input=is_input)
            is_input = False
            layers += [conv2d, nr.ReLU(inplace=True)]
            in_channels = v
    return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def vgg16_nr(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19_nr(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

