'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models

import models.nr_modules as nr
import pdb

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from torchvision.datasets.folder import default_loader
import ntpath

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--image-size', default=224, type=int, help='image size')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
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
parser.add_argument('--iter-size', default=1, type=int)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Neural rejuvenation
parser.add_argument('--nr-sparsity-start', default=0, type=float)
parser.add_argument('--nr-increase-thres', default=0.01, type=float)
parser.add_argument('--nr-increase-step', default=5e-5, type=float)
parser.add_argument('--nr-target', type=float, default=0.5, help='when to rejuvenate')
parser.add_argument('--nr-zero-prb', type=float, default=1.0, help='probability of zeroing out cross params')
parser.add_argument('--nr-zero-rat', type=float, default=0.03)
parser.add_argument('--nr-temp', type=float, default=1.0)
parser.add_argument('--nr-use-adv', type=int, default=0)
parser.add_argument('--nr-bn-target', type=float, default=0)
parser.add_argument('--nr-compress-only', dest='nr_compress_only', action='store_true', help='stop after NR')
parser.add_argument('--nr-flops', dest='nr_flops_flag', action='store_true')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

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
        # https://github.com/joe-siyuan-qiao/GUNN
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


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            Jigsaw(),
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(int(256.0 / 224.0 * args.image_size)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nr.CrossEntropyLoss()
    optimizer = nr.SGD(model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            sparsity=args.nr_sparsity_start, zero_prb=args.nr_zero_prb)

    # Resume
    title = 'ImageNet-' + args.arch
    is_rejuvenated = False
    util_params_old = 1.0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        is_rejuvenated = checkpoint['is_rejuvenated']
        util_params_old = checkpoint['util_params_old']
        args.epochs = checkpoint['epochs']
        args.schedule = checkpoint['schedule']
        print('==> Resumed model size: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        if args.nr_bn_target > 0:
            print('==> Restoring BN to {:.2f}'.format(args.nr_bn_target))
            model.module.bn_restore(args.nr_bn_target)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Valid 5 Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    if args.nr_target >= 0.999:
        is_rejuvenated = True
    epoch = start_epoch
    while epoch < args.epochs:
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_acc_5 = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, test_acc_5])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'is_rejuvenated': is_rejuvenated,
                'util_params_old': util_params_old,
                'epochs': args.epochs,
                'schedule': args.schedule,
                'sparsity': optimizer.sparsity,
            }, is_best, checkpoint=args.checkpoint)

        epoch = epoch + 1
        if not is_rejuvenated:
            args.epochs = args.epochs + 1
            for schedule_idx in range(len(args.schedule)):
                args.schedule[schedule_idx] = args.schedule[schedule_idx] + 1
            live_params, all_params = optimizer.inspect(flops_flag=args.nr_flops_flag)
            util_params = float(live_params) / float(all_params)
            print(' | Utilization {} at {}: {} {}'.format(util_params, optimizer.sparsity, live_params, all_params))

            if util_params <= args.nr_target:
                shutil.copyfile(os.path.join(args.checkpoint, 'checkpoint.pth.tar'), os.path.join(args.checkpoint,
                    'NR_{}_before_{}.pth.tar'.format(args.arch, args.nr_target)))
                model = model.module
                model.rejuvenate(flops_flag=args.nr_flops_flag)
                model = torch.nn.DataParallel(model).cuda()
                optimizer = nr.SGD(model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        sparsity=0, zero_prb=args.nr_zero_prb)
                is_rejuvenated = True

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'acc': test_acc,
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict(),
                        'is_rejuvenated': is_rejuvenated,
                        'util_params_old': util_params_old,
                        'epochs': args.epochs,
                        'schedule': args.schedule,
                        'sparsity': optimizer.sparsity,
                    }, False, checkpoint=args.checkpoint, filename='NR_{}_{}.pth.tar'.format(args.arch, args.nr_target))

                if args.nr_compress_only:
                    return

            if util_params_old - util_params < args.nr_increase_thres:
                optimizer.sparsity += args.nr_increase_step

            util_params_old = util_params


    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda, is_rejuvenated=False):
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

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
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
                    size=len(train_loader),
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

def test(val_loader, model, criterion, epoch, use_cuda):
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
    global_avg_pool = nr.GlobalAvgPool2d(1)
    model.eval()
    torch.set_grad_enabled(False)

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

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
                    size=len(val_loader),
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
    return (losses.avg, top1.avg, top5.avg)

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
