import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import random as rd

class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, model, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sparsity=1e-4, zero_prb=1.0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.model = model
        self.sparsity = sparsity
        self.zero_prb = zero_prb
        self.zero_out_prev = False
        self.zero_out_next = False
        super(SGD, self).__init__(model.parameters(), defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def update_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'is_fixed') and m.is_fixed:
                    continue
                if hasattr(m, 'weight'):
                    m.weight.grad.data.add_(self.sparsity * torch.sign(m.weight.data))

    def print_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                data = m.weight.data.abs()
                print ('BN min / median / max: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(data.min(),
                    data.median(), data.max(), (data < data.max() * 1e-2).sum() / float(data.size(0))))

    def inspect(self, flops_flag=False):
        """ Performs an inpection on the utilization of the parameters
        """
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        return model.inspect(flops_flag=flops_flag)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.update_bn()
        self.zero_out_prev = self.zero_out_next
        self.zero_out_next = rd.random() <= self.zero_prb

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def eval(self):
        """ Set the model to eval mode by synchronzing parameters in data and shadow data
        """
        pass
