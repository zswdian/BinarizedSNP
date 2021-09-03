from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import MNIST_Data
from Models import net
from Models import net_binary
from Models import snp
from Models import snp_binary
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


def adjust_learning_rate(optimizer, epoch):
    update_list = [15, 30, 45]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action='store', default='0.01',
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

    args = parser.parse_args()
    print('==> Options:', args)

    # set the random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    trainloader = MNIST_Data.trainloader
    testloader = MNIST_Data.testloader

    if args.full:
        if not args.snp:
            type = 'data'
        else:
            type = 'data_snp'
    else:
        if not args.snp:
            type = 'data_bin'
        else:
            type = 'data_snp_bin'

    epochs = int(args.epochs)
    expt_num = int(args.expt_num)
    acc_list = []

    filename = 'ExpData/' + type + '.txt'

    # start training
    for i in range(expt_num):

        # define the model

        if not args.full:
            if not args.snp:
                model = net_binary.Net()
            else:
                model = snp_binary.Net()
        else:
            if not args.snp:
                model = net.Net()
            else:
                model = snp.Net()

        # initialize the model
        if not args.pretrained:
            print('==> Initializing model parameters ...')
            best_acc = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.05)
                    m.bias.data.zero_()
        else:
            print('==> Load pretrained model form', args.pretrained, '...')
            pretrained_model = torch.load('Experiment/' + type + '.pth.tar')
            best_acc = pretrained_model['best_acc']
            best_acc_output = pretrained_model['best_acc_output']
            model.load_state_dict(pretrained_model['state_dict'])

        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        # define solver and criterion
        base_lr = float(args.lr)
        param_dict = dict(model.named_parameters())
        params = []

        for key, value in param_dict.items():
            params += [{'params': [value], 'lr': base_lr, 'weight_decay': 0.00001}]

        optimizer = optim.Adam(params, lr=0.10, weight_decay=0.00001)
        criterion = nn.CrossEntropyLoss()

        # define the binarization operator
        if not args.full:
            bin_op = util.BinOp(model)

        # do the evaluation if specified
        if args.evaluate:
            test(i + 1)
            exit(0)

        best_acc = 0

        for epoch in range(1, epochs + 1):
            adjust_learning_rate(optimizer, epoch)
            train(epoch, i + 1)
            test(i + 1)

        with open(filename, 'a') as f:
            f.write('Expt {}: Best Accuracy: {:.2f}%\n'.format(i + 1, best_acc))
        acc_list.append(best_acc)

    with open(filename, 'a') as f:
        f.write('Mean: {}\n'.format(np.mean(acc_list)))
        f.write('Var: {}'.format(np.var(acc_list)))
