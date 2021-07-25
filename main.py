from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from CIFAR10 import CIFAR_Data
from CIFAR10.Models import net as cn
from CIFAR10.Models import net_binary as cnb
from CIFAR10.Models import snps as cs
from CIFAR10.Models import snps_binary as csb
from MNIST import MNIST_Data
from CIFAR10.Models import net as mn
from CIFAR10.Models import net_binary as mnb
from CIFAR10.Models import snps as ms
from CIFAR10.Models import snps_binary as msb
import util
import argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def save_state(expt_no, model, acc, output):
    print('==> Saving model ...')
    state = {
        'best_acc': acc,
        'best_output': output,
        'state_dict': model.state_dict(),
    }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)
    torch.save(state, './MNIST/Experiment/data_' + str(expt_no) + '.pth.tar')


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
    global beat_acc_output
    model.eval()
    test_loss = 0
    correct = 0
    output = []
    if not args.full:
        bin_op.binarization()

    with torch.no_grad():
        for data, target in testloader:
            data, target = Variable(data.cuda()), Variable(target.cuda())

            output = model(data)
            test_loss += criterion(output, target).data.item()
            predict = output.data.max(1, keepdim=True)[1]
            correct += predict.eq(target.data.view_as(predict)).cpu().sum()
            # correct += predict.eq(target.data.view_as(predict)).sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        best_output = output
        save_state(expt_no, model, best_acc, best_output)

    test_loss /= len(testloader.dataset)

    test_loss_list.append(test_loss)
    test_acc_list.append(acc)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset), acc))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

    return


def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


def draw(expt_no):
    x1 = x2 = range(1, epochs)
    y1 = test_acc_list
    y2 = test_loss_list
    plt.figure(expt_no)
    plt.plot(x1, y1, 'r', '-', marker='*')
    plt.plot(x2, y2, 'b', '-.', marker='*')
    plt.savefig('acc_loss' + str(expt_no) + '.jpg')
    return


if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action='store', default='0.01',
                        help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
                        help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--epochs', action='store', default='0',
                        help='the start range of epoch')
    parser.add_argument('--full', action='store', default=False,
                        help='use full-precision')
    parser.add_argument('--snps', action='store', default=False,
                        help='use snps model')
    parser.add_argument('--expt_num', action='store', default=10,
                        help='the num of the experiment')
    parser.add_argument('--cifar', action='store', default=False,
                        help='use CIFAR10')
    parser.add_argument('--mnist', action='store', default=False,
                        help='use MNIST')
    parser.add_argument('--imagenet', action='store', default=False,
                        help='use ImageNet')
    args = parser.parse_args()
    print('==> Options:', args)

    # set the random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    if args.cifar:
        trainloader = CIFAR_Data.trainloader
        testloader = CIFAR_Data.testloader
    elif args.mnist:
        trainloader = MNIST_Data.trainloader
        testloader = MNIST_Data.testloader
    # else:
    #     trainloader = MNIST_Data.trainloader
    #     testloader = MNIST_Data.testloader

    test_loss_list = []
    test_acc_list = []

    epochs = int(args.epochs)
    expt_num = int(args.expt_num)
    acc = []
    # start training
    for i in range(expt_num):
        # define the model
        if args.cifar:
            if not args.full:
                if not args.snps:
                    model = cnb.Net()
                else:
                    model = csb.Net()
            else:
                if not args.snps:
                    model = cn.Net()
                else:
                    model = cs.Net()
        elif args.mnist:
            if not args.full:
                if not args.snps:
                    model = mnb.Net()
                else:
                    model = msb.Net()
            else:
                if not args.snps:
                    model = mn.Net()
                else:
                    model = ms.Net()
        # elif args.imagenet:
        #     if not args.full:
        #         if not args.snps:
        #             model = imagenet.net_binary.Net()
        #         else:
        #             model = imagenet.snps_binary.Net()
        #     else:
        #         if not args.snps:
        #             model = imagenet.net.Net()
        #         else:
        #             model = imagenet.snps.Net()

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
            pretrained_model = torch.load('Models/net_binary.pth.tar')
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
            test(i+1)
            exit(0)

        best_acc = 0

        for epoch in range(1, epochs+1):
            adjust_learning_rate(optimizer, epoch)
            train(epoch, i+1)
            test(i+1)

        with open('mnist_data.txt', 'a') as f:
            f.write('Expt {}: Best Accuracy: {:.2f}%\n'.format(i+1, best_acc))
        acc.append(best_acc)
        # draw(i+1)

    with open('mnist_data.txt', 'a') as f:
        f.write('Mean: {}\n'.format(np.mean(acc)))
        f.write('Var: {}'.format(np.var(acc)))
