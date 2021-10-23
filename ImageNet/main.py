import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import IMAGENET_Data
from Models import alexnet
from Models import alexnet_bin
from Models import alexnet_snp
from Models import alexnet_snp_bin
from Models import resnet
from Models import resnet_snp
from Models import vgg
from Models import vgg_snp
import util

import warnings

warnings.filterwarnings('ignore')

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import gc

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--full', action='store_true',
                    help='use full-precision')
parser.add_argument('--snp', action='store_true',
                    help='use snp model')
parser.add_argument('--expt_num', action='store', default=10,
                    help='the num of the experiment')
parser.add_argument('--alexnet', action='store_true',
                    help='use alexnet')
parser.add_argument('--resnet', action='store_true',
                    help='use resnet')
parser.add_argument('--vgg', action='store_true',
                    help='use vgg')

def main():
    global args, best_prec1
    global type
    args = parser.parse_args()

    epochs = int(args.epochs)
    expt_num = int(args.expt_num)
    acc_list = []
    acc_list_5 = []

    if args.alexnet:
        if args.full:
            if not args.snp:
                type = 'alexnet'
            else:
                type = 'alexnet_snp'
        else:
            if not args.snp:
                type = 'alexnet_bin'
            else:
                type = 'alexnet_snp_bin'
    elif args.resnet:
        if args.full:
            if not args.snp:
                type = 'resnet'
            else:
                type = 'resnet_snp'
        else:
            if not args.snp:
                type = 'resnet_bin'
            else:
                type = 'resnet_snp_bin'
    elif args.vgg:
        if args.full:
            if not args.snp:
                type = 'vgg'
            else:
                type = 'vgg_snp'
        else:
            if not args.snp:
                type = 'vgg_bin'
            else:
                type = 'vgg_snp_bin'

    filename = 'ExpData/' + type + '.txt'

    for i in range(expt_num):

        best_prec1 = 0
        best_prec5 = 0

        # create model
        if args.alexnet:
            if not args.full:
                if not args.snp:
                    model = alexnet_bin.net(pretrained=args.pretrained)
                else:
                    model = alexnet_snp_bin.net(pretrained=args.pretrained)
            else:
                if not args.snp:
                    model = alexnet.net(pretrained=args.pretrained)
                else:
                    model = alexnet_snp.net(pretrained=args.pretrained)
        elif args.resnet:
            if not args.full:
                if not args.snp:
                    model = alexnet_bin.net(pretrained=args.pretrained)
                else:
                    model = alexnet_snp_bin.net(pretrained=args.pretrained)
            else:
                if not args.snp:
                    model = resnet.resnet18()
                else:
                    model = resnet_snp.ResNet18()
        elif args.vgg:
            if not args.full:
                if not args.snp:
                    model = alexnet_bin.net(pretrained=args.pretrained)
                else:
                    model = alexnet_snp_bin.net(pretrained=args.pretrained)
            else:
                if not args.snp:
                    model = vgg.vgg11()
                else:
                    model = vgg_snp.VGG('VGG11')

        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        # define loss function (criterion) and optimizer
        # optimizer = optim.SGD(model.parameters(), lr=args.lr,
        #                       momentum=0.9, weight_decay=5e-4)
        # BIN
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=0.00001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 2.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)
                m.bias.data = m.bias.data.zero_()

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        train_loader = IMAGENET_Data.train_loader
        val_loader = IMAGENET_Data.val_loader

        # print(model)

        # define the binarization operator
        global bin_op
        if not args.full:
            bin_op = util.BinOp(model)

        for epoch in range(epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, i)
            scheduler.step()
            # evaluate on validation set
            prec1, prec5 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                'optimizer': optimizer.state_dict(),
            }, is_best, i)

        with open(filename, 'a') as f:
            f.write('Expt {}: Best prec@1: {:.2f}%\n'.format(i, best_prec1))
            f.write('Expt {}: Best prec@5: {:.2f}%\n'.format(i, best_prec5))
        acc_list.append(best_prec1)
        acc_list_5.append(best_prec5)

    with open(filename, 'a') as f:
        f.write('Mean@1: {}\n'.format(torch.mean(acc_list)))
        f.write('Var@1: {}'.format(torch.var(acc_list)))
        f.write('Mean@5: {}\n'.format(torch.mean(acc_list_5)))
        f.write('Var@5: {}'.format(torch.var(acc_list_5)))


def train(train_loader, model, criterion, optimizer, epoch, expt_no):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # process the weights including binarization
        if not args.full:
            bin_op.binarization()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # restore weights
        if not args.full:
            bin_op.restore()
            bin_op.updateBinaryWeightGrad()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Expt{0}: Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                expt_no, epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        gc.collect()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    if not args.full:
        bin_op.binarization()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    if not args.full:
        bin_op.restore()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, expt_no):
    filename = 'Experiment/' + type + '_' + str(expt_no) + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'Experiment/model_best' + type +
                        '_' + str(expt_no) + '.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
