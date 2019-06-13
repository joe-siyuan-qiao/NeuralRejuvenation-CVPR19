import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .. import nr_modules as nr

__all__ = ['densenet']


from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


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


def CreateBasicBlock(inplanes, expansion=1, growthRate=12):
    planes = expansion * growthRate
    conv = nr.Conv2d(inplanes, growthRate, kernel_size=3, padding=1, bias=False, split_output=True)
    relu = nr.ReLU(inplace=False)
    block = nr.ConcatList()
    long_path = nr.Sequential(conv, relu)
    short_cut = nr.Identity()
    block.add(short_cut)
    block.add(long_path)
    return [block, nr.JoinList(1)]


def CreateTransition(inplanes, outplanes):
    conv = nr.Conv2d(inplanes, outplanes, kernel_size=1, bias=False, split_output=True)
    relu = nr.ReLU(inplace=False)
    avgp = nr.AvgPool2d(2)
    return [conv, relu, avgp]


class DenseNet(nn.Module):

    def __init__(self, depth=22, block=CreateBottleneck,
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == CreateBasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        conv = nr.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False, is_input=True, split_output=True)
        relu = nr.ReLU(inplace=False)
        layers = [conv, relu]
        layers += self._make_denseblock(block, n)
        layers += self._make_transition(compressionRate)
        layers += self._make_denseblock(block, n)
        layers += self._make_transition(compressionRate)
        layers += self._make_denseblock(block, n)
        fc = nr.ConvLinear(self.inplanes, num_classes)
        layers += [fc]
        self.network = nr.Sequential(*layers)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, '__is_linear__'):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                    continue
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers += block(self.inplanes, growthRate=self.growthRate)
            self.inplanes += self.growthRate

        return layers

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return CreateTransition(inplanes, outplanes)


    def forward(self, x):
        return self.network(x)

    def inspect(self, flops_flag=None):
        live_neurons = torch.cuda.ByteTensor([1, 1, 1])
        dummy, live_params, all_params = self.network.inspect(live_neurons)
        return live_params, all_params

    def rejuvenate(self, flops_flag=None):
        live_params, all_params = self.inspect()
        expand_rate = int(math.sqrt(float(all_params) / float(live_params)))
        self.network.rejuvenate(expand_rate)


def densenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)

