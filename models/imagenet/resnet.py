import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from .. import nr_modules as nr


__all__ = ['resnet18_nr', 'resnet34_nr', 'resnet50_nr', 'resnet101_nr',
           'resnet152_nr']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

    def __init__(self, block, layers, block_expansion, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.block_expansion = block_expansion
        conv1 = nr.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, is_input=True, is_fixed=True)
        relu = nr.ReLU(inplace=True)
        maxpool = nr.MaxPool2d(kernel_size=3, stride=2, padding=1)
        network = [conv1, relu, maxpool]

        network += self._make_layer(block, 64, layers[0])
        network += self._make_layer(block, 128, layers[1], stride=2)
        network += self._make_layer(block, 256, layers[2], stride=2)
        network += self._make_layer(block, 512, layers[3], stride=2)
        fc = nr.ConvLinear(512 * self.block_expansion, num_classes)
        network += [fc]

        self.network = nr.Sequential(*network)

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

    def inspect(self, flops_flag):
        live_neurons, live_params, all_params = self.network.inspect(flops_flag=flops_flag)
        return live_params, all_params

    def rejuvenate(self, flops_flag):
        live_params, all_params = self.inspect(flops_flag=flops_flag)
        expand_rate = math.sqrt(float(all_params) / float(live_params))
        self.network.rejuvenate(expand_rate, flops_flag=flops_flag)

    def bn_restore(self, bn_target):
        self.network.bn_restore(bn_target=bn_target)


def resnet18_nr(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CreateBasicBlock, [2, 2, 2, 2], 1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_nr(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CreateBasicBlock, [3, 4, 6, 3], 1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_nr(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CreateBottleneck, [3, 4, 6, 3], 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_nr(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CreateBottleneck, [3, 4, 23, 3], 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_nr(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CreateBottleneck, [3, 8, 36, 3], 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
