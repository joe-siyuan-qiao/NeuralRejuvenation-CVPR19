'''
Neural Rejuvenation Layers
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

import itertools # for _load_from_state_dict
from torch.nn import Parameter


__nr_conv_ca_mode__ = True

def set_nr_ca(mode=True):
    global __nr_conv_ca_mode__
    __nr_conv_ca_mode__ = mode


class Sequential(nn.Sequential):
    r"""
    sequential with neural rejuvenation
    """

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, x):
        return super(Sequential, self).forward(x)

    def inspect(self, live_neurons=None, flops_flag=False):
        if live_neurons is None:
            live_neurons = torch.cuda.ByteTensor([1, 1, 1])
        live_params, all_params = 0, 0
        for module in self._modules.values():
            live_neurons, sub_live_params, sub_all_params = module.inspect(live_neurons, flops_flag)
            live_params = live_params + sub_live_params
            all_params = all_params + sub_all_params
        return live_neurons, live_params, all_params

    def rejuvenate(self, expand_rate=None, inp_neurons=None, flops_flag=False):
        if expand_rate is None:
            dummy, live_params, all_params = self.inspect(flops_flag=flops_flag)
            expand_rate = float(all_params) / float(live_params)
        if inp_neurons is None:
            inp_neurons = 3
        for module in self._modules.values():
            inp_neurons = module.rejuvenate(expand_rate, inp_neurons)
        return inp_neurons

    def bn_restore(self, bn_weight=None, bn_target=1.0):
        if bn_weight is None:
            bn_weight = torch.cuda.FloatTensor([1., 1., 1.])
        for module in self._modules.values():
            bn_weight = module.bn_restore(bn_weight, bn_target)
        return bn_weight


class MaxPool2d(nn.Module):
    r"""
    max pooling layer with neural rejuvenation
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
            return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size, stride, padding, dilation,
                return_indices, ceil_mode)

    def forward(self, x):
        if type(x) == list:
            return [self.layer(ele) for ele in x]
        else:
            return self.layer(x)

    def inspect(self, live_neurons, flops_flag):
        return live_neurons, 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        return inp_neurons

    def bn_restore(self, bn_weight, bn_target):
        return bn_weight


class AvgPool2d(nn.Module):
    r"""
    average pooling layer with neural rejuvenation
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
            count_include_pad=True):
        super(AvgPool2d, self).__init__()
        self.layer = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode,
                count_include_pad)

    def forward(self, x):
        if type(x) == list:
            return [self.layer(ele) for ele in x]
        else:
            return self.layer(x)

    def inspect(self, live_neurons, flops_flag):
        return live_neurons, 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        return inp_neurons

    def bn_restore(self, bn_weight, bn_target):
        return bn_weight


class _AttConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, bias=True):
        super(_AttConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias)
        self.register_buffer('live_inp', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('live_out', torch.tensor(-1, dtype=torch.long))
        self.init_cross = False
        self.__nr_conv_ca_mode__ = __nr_conv_ca_mode__

    def forward(self, x):
        weight = self.weight
        if self.live_inp.item() < 0:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
        live_out, live_inp = self.live_out.item(), self.live_inp.item()
        dead_out, dead_inp = self.out_channels - live_out, self.in_channels - live_inp
        ll_weight = weight.narrow(0, 0, live_out).narrow(1, 0, live_inp)
        rr_weight = weight.narrow(0, live_out, dead_out).narrow(1, live_inp, dead_inp)
        lr_weight = weight.narrow(0, 0, live_out).narrow(1, live_inp, dead_inp)
        rl_weight = weight.narrow(0, live_out, dead_out).narrow(1, 0, live_inp)

        live_x, dead_x = x.narrow(1, 0, live_inp), x.narrow(1, live_inp, dead_inp)
        ll_output = F.conv2d(live_x, ll_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        rr_output = F.conv2d(dead_x, rr_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        lr_output = F.conv2d(dead_x, lr_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        rl_output = F.conv2d(live_x, rl_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)

        if self.__nr_conv_ca_mode__:
            ll_output = ll_output + lr_output * 2 * F.sigmoid(ll_output)
            rr_output = rr_output + rl_output * 2 * F.sigmoid(rr_output)
        return torch.cat([ll_output, rr_output], dim=1)

    def set_live_inp_and_out(self, live_inp, live_out):
        self.live_inp.fill_(live_inp)
        self.live_out.fill_(live_out)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    if name == 'weight':
                        self.out_channels = input_param.size(0)
                        self.in_channels = input_param.size(1)
                        param.resize_(input_param.size())
                    else:
                        error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                'where the shape is {} in current model.'
                                .format(key, param.shape, input_param.shape))
                        continue

                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    if name == 'weight':
                        param.resize_(input_param.size())
                        param.copy_(input_param)
                    else:
                        error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key, input_param in state_dict.items():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

        if self.init_cross:
            live_inp, live_out = self.live_inp.item(), self.live_out.item()
            if live_inp <= 0 or live_out <= 0:
                return
            dead_out, dead_inp = self.out_channels - live_out, self.in_channels - live_inp
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            self.weight.data.narrow(0, 0, live_out).narrow(1, live_inp,
                    dead_inp).normal_(0, math.sqrt(2.0 / n))
            self.weight.data.narrow(0, live_out, dead_out).narrow(1, 0,
                    live_inp).normal_(0, math.sqrt(2.0 / n))


class BatchNorm2d(nn.BatchNorm2d):
    r''' Need to rewrite load_state_dict '''
    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    if name in ['weight', 'bias', 'running_mean', 'running_var']:
                        self.num_features = input_param.size(0)
                        param.resize_(input_param.size())
                    else:
                        error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                'where the shape is {} in current model.'
                                .format(key, param.shape, input_param.shape))
                        continue

                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key, input_param in state_dict.items():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)


class Conv2d(nn.Module):
    r"""
    2d convolutional layer with neural rejuvenation
    batch normalization is assumed
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, bias=False, is_input=False, is_fixed=False, split_output=False):
        super(Conv2d, self).__init__()
        # hyper-parameters
        self.live_thres = 1e-2
        self.all_dead_thres = 1e-3
        # real layers
        self.conv = _AttConv2d(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias)
        self.bn = BatchNorm2d(out_channels)
        self.bn.is_fixed = is_fixed
        # save configurations
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.is_input = is_input
        self.is_fixed = is_fixed
        self.split_output = split_output
        # to save in state_dict
        self.register_buffer('all_dead', torch.tensor(0, dtype=torch.long))
        self.register_buffer('is_rejuvenated', torch.tensor(0, dtype=torch.long))
        self.register_buffer('live_out', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('dead_out', torch.tensor(-1, dtype=torch.long))
        # for FLOPs calculating
        self.register_buffer('out_h', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('out_w', torch.tensor(-1, dtype=torch.long))

    def forward(self, x):
        if self.out_h.item() < 0:
            if type(x) == list:
                inp_h, inp_w = x[0].size(2), x[0].size(3)
            else:
                inp_h, inp_w = x.size(2), x.size(3)
            self.out_h.fill_(int((inp_h + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) /
                        self.conv.stride[0] + 1))
            self.out_w.fill_(int((inp_w + 2 * self.conv.padding[1] - self.conv.kernel_size[1]) /
                        self.conv.stride[1] + 1))
        if self.is_input:
            if self.is_rejuvenated.item() > 0:
                if not type(x) == list:
                    x = [x, x]
                live_weight = self.conv.weight.narrow(0, 0, self.live_out.item())
                dead_weight = self.conv.weight.narrow(0, self.live_out.item(), self.dead_out.item())
                live_output = F.conv2d(x[0], live_weight, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)
                dead_output = F.conv2d(x[1], dead_weight, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)
                output = self.bn(torch.cat([live_output, dead_output], dim=1))
                if not self.split_output:
                    return output
                live_output = output.narrow(1, 0, self.live_out.item())
                dead_output = output.narrow(1, self.live_out.item(), self.dead_out.item())
                return [live_output, dead_output]
            else:
                if type(x) == list:
                    x = x[0]
                return self.bn(self.conv(x))
        if not self.split_output:
            if self.all_dead.item() > 0:
                return None
            return self.bn(self.conv(x))
        else:
            if self.all_dead.item() > 0:
                return [None, None]
            if type(x) == list:
                x = torch.cat(x, 1)
            output = self.bn(self.conv(x))
            if not (self.is_rejuvenated.item() > 0):
                return output
            else:
                live_output = output.narrow(1, 0, self.live_out.item())
                dead_output = output.narrow(1, self.live_out.item(), self.dead_out.item())
                return [live_output, dead_output]

    def inspect(self, live_neurons, flops_flag):
        self.live_prev = live_neurons
        self.dead_prev = self.live_prev.clone().fill_(1) - self.live_prev
        weight = torch.abs(self.bn.weight.data)
        self.dead = weight < (weight.max() * self.live_thres)
        if self.live_prev.sum().item() < 1 or weight.max() < self.all_dead_thres:
            self.dead.fill_(1)
            self.all_dead.fill_(1)
        elif self.is_fixed:
            self.dead.fill_(0)
        self.live = self.dead.clone().fill_(1) - self.dead
        weight = torch.cuda.ByteTensor(self.conv.weight.size()).fill_(1)
        if self.all_dead.item() > 0:
            return self.live, 0, weight.sum()
        self.live_prev_mask = self.live_prev.clone().view(1, -1, 1, 1).expand(
                self.conv.weight.size())
        self.dead_prev_mask = self.dead_prev.clone().view(1, -1, 1, 1).expand(
                self.conv.weight.size())
        self.live_mask = self.live.clone().view(-1, 1, 1, 1).expand(
                self.conv.weight.size())
        self.dead_mask = self.dead.clone().view(-1, 1, 1, 1).expand(
                self.conv.weight.size())
        self.one_mask = self.live_prev_mask * self.live_mask
        if flops_flag:
            live_flops = self.one_mask.sum().item() * self.out_h.item() * self.out_w.item()
            all_flops = weight.sum().item() * self.out_h.item() * self.out_w.item()
            return self.live, live_flops, all_flops
        return self.live, self.one_mask.sum().item(), weight.sum().item()

    def rejuvenate(self, expand_rate, inp_neurons):
        if self.all_dead.item() > 0:
            print (' | Rejuvenation | No Survival ')
            del self.conv, self.bn
            return
        # -- expand
        self.live_inp = self.live_prev.sum().item()
        live_out = self.live.sum().item()
        self.live_out.fill_(live_out)
        # desired_inp = int(self.live_inp * expand_rate)
        desired_inp = inp_neurons
        if self.is_input:
            desired_inp = self.live_inp
        desired_out = int(live_out * expand_rate)
        self.dead_out.fill_(desired_out - live_out)
        # -- create new
        new_conv = _AttConv2d(desired_inp, desired_out, self.kernel_size, self.stride,
                self.padding, self.dilation, self.groups, self.bias)
        new_bn = BatchNorm2d(desired_out)
        new_bn.weight.data.fill_(0.5)
        new_bn.bias.data.fill_(0)
        n = new_conv.kernel_size[0] * new_conv.kernel_size[1] * desired_out
        new_conv.weight.data.normal_(0, math.sqrt(2.0 / n))
        # -- copy live neurons
        new_conv.weight.data.narrow(0, 0, live_out).narrow(1, 0, self.live_inp).copy_(
                self.conv.weight.data[self.one_mask].view(live_out, self.live_inp,
                    self.one_mask.size(2), self.one_mask.size(3)))
        new_bn.weight.data.narrow(0, 0, live_out).copy_(self.bn.weight.data[self.live])
        new_bn.bias.data.narrow(0, 0, live_out).copy_(self.bn.bias.data[self.live])
        new_bn.running_mean.narrow(0, 0, live_out).copy_(self.bn.running_mean[self.live])
        new_bn.running_var.narrow(0, 0, live_out).copy_(self.bn.running_var[self.live])
        # -- delete old and assign new
        del self.conv, self.bn
        self.conv, self.bn = new_conv, new_bn
        print (' | Rejuvenation | input {} -> {} | output {} -> {}'.format(self.live_inp,
            self.conv.in_channels, live_out, self.conv.out_channels))
        self.is_rejuvenated.fill_(1)
        # -- create shadow parameters and masks for optimizer
        if self.is_input == False:
            self.conv.weight.shadow_data = self.conv.weight.data.clone().cuda()
            self.conv.weight.shadow_zero = torch.cuda.ByteTensor(self.conv.weight.data.size()).fill_(0)
            if self.conv.weight.size(0) > live_out and self.conv.weight.size(1) > self.live_inp:
                self.conv.weight.shadow_zero[:live_out, self.live_inp:, :, :] = 1
                self.conv.weight.shadow_zero[live_out:, :self.live_inp, :, :] = 1
                self.conv.weight.data[self.conv.weight.shadow_zero.data] = 0
                self.conv.set_live_inp_and_out(self.live_inp, live_out)
        return desired_out

    def bn_restore(self, bn_weight, bn_target):
        bn_weight = bn_weight.view(1, -1, 1, 1).expand(self.conv.weight.data.size())
        self.conv.weight.data = self.conv.weight.data * bn_weight.data / bn_target
        bn_weight = self.bn.weight.data.clone().abs()
        if self.is_fixed:
            bn_weight.fill_(1.0)
            return bn_weight
        bn_weight[bn_weight > bn_target] = bn_target
        self.bn.bias.data = self.bn.bias.data / (bn_weight / bn_target)
        pos_idx = (self.bn.weight.data >= 0) * (self.bn.weight.data <= bn_target)
        neg_idx = (self.bn.weight.data <= 0) * (self.bn.weight.data >= -bn_target)
        self.bn.weight.data[pos_idx] = bn_target
        self.bn.weight.data[neg_idx] = -bn_target
        return bn_weight


class Flat(nn.Module):
    r"""
    flatten tensor to (batch_size, feature_dimension)
    equivalent to view(x.size(0), -1)
    """

    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        if type(x) == list:
            x = torch.cat(x, 1)
        return x.view(x.size(0), -1)

    def inspect(self, live_neurons, flops_flag):
        return live_neurons, 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        return inp_neurons

    def bn_restore(self, bn_weight, bn_target):
        return bn_weight


class ConvLinear(nn.Module):
    r"""
    linear layer with neural rejuvenation
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super(ConvLinear, self).__init__()
        self.linear = _AttConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,\
                padding=padding, dilation=dilation, bias=bias)
        self.kernel_size = kernel_size
        self.linear.__is_linear__ = True
        # to save in state_dict
        self.register_buffer('is_rejuvenated', torch.tensor(0, dtype=torch.long))
        self.register_buffer('live_inp', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('dead_inp', torch.tensor(-1, dtype=torch.long))

    def forward(self, x):
        if self.is_rejuvenated.item() > 0:
            if type(x) == list:
                x = torch.cat(x, dim=1)
            live_input, dead_input = x.clone(), x.clone()
            dead_input.narrow(1, 0, self.live_inp.item()).zero_()
            live_input.narrow(1, self.live_inp.item(), self.dead_inp.item()).zero_()
            live_output = self.linear(live_input).unsqueeze(2)
            dead_output = self.linear(dead_input).unsqueeze(2)
            output = torch.cat([live_output, dead_output], 2)
        else:
            output = self.linear(x).unsqueeze(2)
        return output

    def inspect(self, live_neurons, flops_flag):
        self.live_prev = live_neurons
        self.dead_prev = self.live_prev.clone().fill_(1) - self.live_prev
        self.live_mask = self.live_prev.clone().view(1, -1, 1, 1)
        self.dead_mask = self.dead_prev.clone().view(1, -1, 1, 1)
        self.live_mask = self.live_mask.expand(self.linear.weight.size())
        self.dead_mask = self.dead_mask.expand(self.linear.weight.size())
        return live_neurons, 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        self.live_inp.fill_(self.live_prev.sum().item())
        # desired_inp = int(self.live_inp.item() * expand_rate)
        desired_inp = inp_neurons
        self.dead_inp.fill_(desired_inp - self.live_inp.item())
        new_linear = _AttConv2d(desired_inp, self.linear.out_channels, kernel_size=1)
        new_linear.weight.data.normal_(0, 0.01)
        to_copy = self.linear.weight.data[self.live_mask]
        new_linear.weight.data.narrow(1, 0, self.live_inp.item()).copy_(
                to_copy.view(self.linear.out_channels, self.live_inp.item(), 1, 1))
        new_linear.bias.data.copy_(self.linear.bias.data)
        del self.linear
        self.linear = new_linear
        self.linear.__is_linear__ = True
        self.is_rejuvenated.fill_(1)
        return inp_neurons

    def bn_restore(self, bn_weight, bn_target):
        bn_weight = bn_weight.view(1, -1, 1, 1).expand(self.linear.weight.data.size())
        self.linear.weight.data = self.linear.weight.data * bn_weight
        return bn_weight


class GlobalAvgPool2d(nn.Module):

    def __init__(self, output_size):
        super(GlobalAvgPool2d, self).__init__()
        self.layer = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        # b x c x h x w
        if len(x.size()) == 4:
            return self.layer(x)
        # b x c x n x h x w
        elif len(x.size()) == 5:
            b, c, n, h, w = x.size()
            out = self.layer(x.view(b, c * n, h, w))
            return out.view(b, c, n)
        else:
            print ('Unsupported input dimension')
            print (x.size())
            raise


class ReLU(nn.Module):
    r"""
    relu layer with neural rejuvenation
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.layer = nn.ReLU(inplace)

    def forward(self, x):
        # the outputs of previous layers are dead
        if x is None:
            return None
        if type(x) == list:
            return [self.layer(ele) if (ele is not None) else None for ele in x]
        else:
            return self.layer(x)

    def inspect(self, live_neurons, flops_flag):
        return live_neurons, 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        return inp_neurons

    def bn_restore(self, bn_weight, bn_target):
        return bn_weight


class CrossEntropyLoss(nn.Module):
    r"""
    cross entropy loss layer for neural rejuvenation
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.layer = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        if len(inputs.size()) <= 2:
            return self.layer(inputs, targets)
        live_inputs = inputs.narrow(2, 0, 1).squeeze()
        if inputs.size(2) > 1:
            dead_inputs = inputs.narrow(2, 1, 1).squeeze()
            return self.layer(live_inputs, targets) + self.layer(dead_inputs, targets)
        else:
            return self.layer(live_inputs, targets)


class ConcatList(nn.Module):
    r"""
    ref: https://github.com/torch/nn/blob/master/ConcatTable.lua
    used for residual networks and densenet
    """

    def __init__(self):
        super(ConcatList, self).__init__()
        self.module_cnt = 0

    def add(self, sub_module):
        setattr(self, 'sub_module_{}'.format(self.module_cnt), sub_module)
        self.module_cnt = self.module_cnt + 1

    def forward(self, x):
        output = []
        for i in range(self.module_cnt):
            sub_module = getattr(self, 'sub_module_{}'.format(i))
            output.append(sub_module.forward(x))
        return output

    def inspect(self, live_neurons, flops_flag):
        out_live_neurons = []
        live_params, all_params = 0, 0
        for i in range(self.module_cnt):
            sub_module = getattr(self, 'sub_module_{}'.format(i))
            sub_live_neurons, sub_live_params, sub_all_params = sub_module.inspect(live_neurons, flops_flag)
            out_live_neurons.append(sub_live_neurons)
            live_params = live_params + sub_live_params
            all_params = all_params + sub_all_params
        return out_live_neurons, live_params, all_params

    def rejuvenate(self, expand_rate, inp_neurons):
        out_neurons = []
        for i in range(self.module_cnt):
            sub_module = getattr(self, 'sub_module_{}'.format(i))
            out_neurons.append(sub_module.rejuvenate(expand_rate, inp_neurons))
        return out_neurons

    def bn_restore(self, bn_weight, bn_target):
        bn_weight_out = []
        for i in range(self.module_cnt):
            sub_module = getattr(self, 'sub_module_{}'.format(i))
            bn_weight_out += [sub_module.bn_restore(bn_weight, bn_target)]
        return bn_weight_out


class CAddList(nn.Module):
    r"""
    ref: https://github.com/torch/nn/blob/master/CAddTable.lua
    used for residual networks
    """

    def __init__(self):
        super(CAddList, self).__init__()

    def forward(self, x):
        assert(type(x) == list)
        output = 0
        for i in range(0, len(x)):
            if x[i] is not None:
                output = output + x[i]
        return output

    def inspect(self, live_neurons, flops_flag):
        # CAddList cannot prune anything because of the slow speed of scatter_add_
        return live_neurons[0].clone().fill_(1), 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        return inp_neurons[0]

    def bn_restore(self, bn_weight, bn_target):
        return bn_weight[0]


class Identity(nn.Module):
    r"""
    ref: https://github.com/torch/nn/blob/master/Identity.lua
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def inspect(self, live_neurons, flops_flag):
        return live_neurons, 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        return inp_neurons

    def bn_restore(self, bn_weight, bn_target):
        return bn_weight


class JoinList(nn.Module):
    r"""
    ref: https://github.com/torch/nn/blob/master/JoinTable.lua
    used for densenet
    """

    def __init__(self, dimension):
        super(JoinList, self).__init__()
        self.dimension = dimension
        self.register_buffer('is_rejuvenated', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        assert(type(x) == list)
        if self.is_rejuvenated.item() < 1:
            valid_x = [ele for ele in x if ele is not None]
            return torch.cat(valid_x, self.dimension)
        else:
            live_valid_x = [ele[0] for ele in x if ele[0] is not None]
            dead_valid_x = [ele[1] for ele in x if ele[1] is not None]
            return [torch.cat(live_valid_x, self.dimension),
                    torch.cat(dead_valid_x, self.dimension)]

    def inspect(self, live_neurons, flops_flag):
        assert(type(live_neurons) == list)
        return torch.cat(live_neurons, 0), 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        self.is_rejuvenated.fill_(1)
        return sum(inp_neurons)

    def bn_restore(self, bn_weight, bn_target):
        return torch.cat(bn_weight, 0)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(JoinList, self)._load_from_state_dict(state_dict, prefix, metadata, False, missing_keys, unexpected_keys, error_msgs)

