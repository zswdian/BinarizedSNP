from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import CIFAR_Data
from Models import nin
from Models import nin_bin
from Models import nin_snp
from Models import nin_snp_bin
from Models import resnet
from Models import resnet_bin
from Models import resnet_snp
from Models import resnet_snp_bin
from Models import vgg
from Models import vgg_snp
from Models import vgg_bin
from Models import vgg_snp_bin
import util
import argparse
from torch.autograd import Variable
import numpy as np

import warnings

warnings.filterwarnings('ignore')


def save_state(expt_no, model, acc):
    print('==> Saving model ...')
    state = {
        'best_acc': acc,
        'state_dict': model.state_dict(),
    }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)
    torch.save(state, 'Experiment/' + type + '_' + str(expt_no) + '.pth.tar')


def train(epoch, expt_no):
    model.train()

    for batch_idx, (data, target) in enumerate(trainloader):
        # binarize the weights
        if not args.full:
            bin_op.binarization()

        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)

        # backwarding
        loss = criterion(output, target)
        loss.backward()

        # restore the weights
        if not args.full:
            bin_op.restore()
            bin_op.updateBinaryWeightGrad()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Expt{}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                expt_no, epoch, batch_idx * len(data), len(trainloader.dataset),
                                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return


def test(expt_no):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    if not args.full:
        bin_op.binarization()

    with torch.no_grad():
        for data, target in testloader:
            data, target = Variable(data.cuda()), Variable(target.cuda())

            output = model(data)
            test_loss += criterion(output, target).data.item()
            predict = output.data.max(1, keepdim=True)[1]
            correct += predict.eq(target.data.view_as(predict)).cpu().sum()

    acc = 100. * correct / len(testloader.dataset)

    if best_acc < acc:
        best_acc = acc
        save_state(expt_no, model, best_acc)

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset), acc))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return


if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action='store', default='0.01', type=float,
                        help='the intial learning rate')
    parser.add_argument('--pretrained', action='store_true',
                        help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--epochs', action='store', default='0',
                        help='the start range of epoch')
    parser.add_argument('--full', action='store_true',
                        help='use full-precision')
    parser.add_argument('--snp', action='store_true',
                        help='use snp model')
    parser.add_argument('--expt_num', action='store', default=10,
                        help='the num of the experiment')
    parser.add_argument('--resnet', action='store_true',
                        help='use resnet')
    parser.add_argument('--vgg', action='store_true',
                        help='use vgg')
    parser.add_argument('--nin', action='store_true',
                        help='use nin')
    args = parser.parse_args()
    print('==> Options:', args)

    # set the random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    trainloader = CIFAR_Data.trainloader
    testloader = CIFAR_Data.testloader

    if args.nin:
        if args.full:
            if not args.snp:
                type = 'nin'
            else:
                type = 'nin_snp'
        else:
            if not args.snp:
                type = 'nin_bin'
            else:
                type = 'nin_snp_bin'
    elif args.resnet:
        if args.full:
            if not args.snp:
                type = 'resnet_data'
            else:
                type = 'resnet_data_snp'
        else:
            if not args.snp:
                type = 'resnet_data_bin'
            else:
                type = 'resnet_data_snp_bin'
    elif args.vgg:
        if args.full:
            if not args.snp:
                type = 'vgg_data'
            else:
                type = 'vgg_data_snp'
        else:
            if not args.snp:
                type = 'vgg_data_bin'
            else:
                type = 'vgg_data_snp_bin'

    epochs = int(args.epochs)
    expt_num = int(args.expt_num)
    acc_list = []

    filename = 'ExpData/' + type + '.txt'

    # start training
    for i in range(expt_num):

        # define the model
        if args.nin:
            if not args.full:
                if not args.snp:
                    model = nin_bin.Net()
                else:
                    model = nin_snp_bin.Net()
            else:
                if not args.snp:
                    model = nin.Net()
                else:
                    model = nin_snp.Net()
        elif args.resnet:
            if not args.full:
                if not args.snp:
                    model = resnet_bin.ResNet18()
                else:
                    model = resnet_snp_bin.ResNet18()
            else:
                if not args.snp:
                    model = resnet.ResNet18()
                else:
                    model = resnet_snp.ResNet18()
        elif args.vgg:
            if not args.full:
                if not args.snp:
                    model = vgg_bin.VGG('VGG11')
                else:
                    model = vgg_snp_bin.VGG("VGG11")
            else:
                if not args.snp:
                    model = vgg.VGG('VGG11')
                else:
                    model = vgg_snp.VGG('VGG11')

        # initialize the model
        if not args.pretrained:
            print('==> Initializing model parameters ...')
            best_acc = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.05)
                    m.bias.data.normal_(0, 0.0)
        else:
            print('==> Load pretrained model form', args.pretrained, '...')
            pretrained_model = torch.load('Experiment/' + type + '.pth.tar')
            best_acc = pretrained_model['best_acc']
            best_acc_output = pretrained_model['best_acc_output']
            model.load_state_dict(pretrained_model['state_dict'])

        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        # define solver and criterion
        # optimizer = optim.SGD(model.parameters(), lr=args.lr,
        #                       momentum=0.9, weight_decay=5e-4)
        # BIN
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=0.00001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # define the binarization operator
        if not args.full:
            bin_op = util.BinOp(model)

        # do the evaluation if specified
        if args.evaluate:
            test(i + 1)
            exit(0)

        best_acc = 0

        for epoch in range(1, epochs + 1):
            train(epoch, i + 1)
            test(i + 1)
            scheduler.step()

        with open(filename, 'a') as f:
            f.write('Expt {}: Best Accuracy: {:.2f}%\n'.format(i + 1, best_acc))
        acc_list.append(best_acc)

    with open(filename, 'a') as f:
        f.write('Mean: {}\n'.format(np.mean(acc_list)))
        f.write('Var: {}'.format(np.var(acc_list)))
