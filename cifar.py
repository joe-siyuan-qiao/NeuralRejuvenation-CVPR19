'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

import models.nr_modules as nr
import pdb

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


nr.set_nr_ca(False)
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Neural rejuvenation
parser.add_argument('--nr-sparsity', default=1e-4, type=float)
parser.add_argument('--nr-target', type=float, default=0.25, help='when to rejuvenate')
parser.add_argument('--nr-zero-prb', type=float, default=1.0, help='probability of zeroing out cross params')
parser.add_argument('--nr-zero-rat', type=float, default=0.03)
parser.add_argument('--nr-temp', type=float, default=1.0)
parser.add_argument('--nr-use-adv', type=int, default=0)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)


    class Jigsaw(object):
        def __call__(self, tensor):
            tensor_1 = tensor.clone()
            tensor_2 = tensor.clone()
            return torch.cat([tensor_1, tensor_2], dim=0)
            h, w = tensor.size(1), tensor.size(2)
            d = random.randint(0, h - 1)
            if d > 0:
                tensor_u, tensor_d = tensor.narrow(1, 0, d), tensor.narrow(1, d, h - d)
                tensor_1.narrow(1, 0, h - d).copy_(tensor_d)
                tensor_1.narrow(1, h - d, d).copy_(tensor_u)
            d = random.randint(0, w - 1)
            if d > 0:
                tensor_l, tensor_r = tensor.narrow(2, 0, d), tensor.narrow(2, d, w - d)
                tensor_2.narrow(2, 0, w - d).copy_(tensor_r)
                tensor_2.narrow(2, w - d, d).copy_(tensor_l)
            return torch.cat([tensor_1, tensor_2], dim=0)

        def __repr__(self):
            return self.__class__.__name__ + '()'


    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Jigsaw(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nr.CrossEntropyLoss()
    optimizer = nr.SGD(model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            sparsity=args.nr_sparsity, zero_prb=args.nr_zero_prb)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    is_rejuvenated = False
    if args.nr_target >= 0.999:
        is_rejuvenated = True
    epoch = start_epoch
    while epoch < args.epochs:
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        # optimizer.print_bn()

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, is_rejuvenated and args.nr_use_adv > 0)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, optimizer)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        epoch = epoch + 1
        if not is_rejuvenated:
            args.epochs = args.epochs + 1
            for schedule_idx in range(len(args.schedule)):
                args.schedule[schedule_idx] = args.schedule[schedule_idx] + 1
            live_params, all_params = optimizer.inspect()
            util_params = float(live_params) / float(all_params)
            print(' | Parameter utilization {}'.format(util_params))

            if util_params <= args.nr_target:
                model = model.module
                model.rejuvenate()
                model = torch.nn.DataParallel(model).cuda()
                optimizer = nr.SGD(model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        sparsity=0, zero_prb=args.nr_zero_prb)
                is_rejuvenated = True

    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, is_rejuvenated):
    # switch to train mode
    model.train()
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    live_top1 = AverageMeter()
    live_top5 = AverageMeter()
    dead_top1 = AverageMeter()
    dead_top5 = AverageMeter()
    end = time.time()

    global_avg_pool = nr.GlobalAvgPool2d(1).cuda()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        batch_size = inputs.size(0)

        # compute output
        if is_rejuvenated:
            inputs = [inputs.narrow(1, 0, 3), inputs.narrow(1, 3, 3)]
        else:
            inputs = inputs.narrow(1, 0, 3)

        conv_outputs = model(inputs)
        outputs = global_avg_pool(conv_outputs)
        loss = criterion(outputs, targets)
        if type(inputs) == list:
            inputs = inputs[0]

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        if len(outputs.size()) <= 2:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        elif outputs.data.size(2) > 1:
            prec1, prec5 = accuracy(outputs.data.narrow(2, 0, 1).squeeze(),
                    targets.data, topk=(1, 5))
            live_top1.update(prec1.item(), inputs.size(0))
            live_top5.update(prec5.item(), inputs.size(0))

            prec1, prec5 = accuracy(outputs.data.narrow(2, 1, 1).squeeze(),
                    targets.data, topk=(1, 5))
            dead_top1.update(prec1.item(), inputs.size(0))
            dead_top5.update(prec5.item(), inputs.size(0))

            outputs = outputs.data.narrow(2, 0, 1) + outputs.data.narrow(2, 1, 1)
            prec1, prec5 = accuracy(outputs.squeeze(),
                    targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        else:
            prec1, prec5 = accuracy(outputs.data.narrow(2, 0, 1).squeeze(),
                    targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, optimizer=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    live_top1 = AverageMeter()
    live_top5 = AverageMeter()
    dead_top1 = AverageMeter()
    dead_top5 = AverageMeter()

    # switch to evaluate mode
    if optimizer is not None:
        optimizer.eval()
    model.eval()
    global_avg_pool = nr.GlobalAvgPool2d(1)
    torch.set_grad_enabled(False)

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        conv_outputs = model(inputs)
        outputs = global_avg_pool(conv_outputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        losses.update(loss.data.item(), inputs.size(0))
        if len(outputs.size()) <= 2:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        elif outputs.data.size(2) > 1:
            prec1, prec5 = accuracy(outputs.data.narrow(2, 0, 1).squeeze(),
                    targets.data, topk=(1, 5))
            live_top1.update(prec1.item(), inputs.size(0))
            live_top5.update(prec5.item(), inputs.size(0))

            prec1, prec5 = accuracy(outputs.data.narrow(2, 1, 1).squeeze(),
                    targets.data, topk=(1, 5))
            dead_top1.update(prec1.item(), inputs.size(0))
            dead_top5.update(prec5.item(), inputs.size(0))

            outputs = outputs.data.narrow(2, 0, 1) + outputs.data.narrow(2, 1, 1)
            prec1, prec5 = accuracy(outputs.squeeze(),
                    targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        else:
            prec1, prec5 = accuracy(outputs.data.narrow(2, 0, 1).squeeze(),
                    targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | live_top1: {ltop1: .4f} | dead_top1: {dtop1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ltop1=live_top1.avg,
                    dtop1=dead_top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
