import torch
import torch.nn as nn
import torch.optim as optim
import Data
from Models import net_binary
import util
import argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
    }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'Models/net_binary.pth.tar')


def train(epoch):
    # load the weights
    if epoch is not epoch_start:
        pretrained_model = torch.load('Models/net_binary.pth.tar')
        model.load_state_dict(pretrained_model['state_dict'])

    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):

        # binarize the weights
        bin_op.binarization()

        # forwarding
        # data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)

        # backwarding
        loss = criterion(output, target)
        loss.backward()

        # restore the weights
        bin_op.restore()
        bin_op.updateBinaryWeightGrad()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return


def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()

    with torch.no_grad():
        for data, target in testloader:
            # data, target = Variable(data.cuda()), Variable(target.cuda())

            output = model(data)
            test_loss += criterion(output, target).data.item()
            predict = output.data.max(1, keepdim=True)[1]
            # correct += predict.eq(target.data.view_as(predict)).cpu().sum()
            correct += predict.eq(target.data.view_as(predict)).sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)

    test_loss /= len(testloader.dataset)

    test_loss_list.append(test_loss)
    test_acc_list.append(acc)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(testloader.dataset), acc))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return


def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


def draw():
    x1 = x2 = range(epoch_start, epoch_end)
    y1 = test_acc_list
    y2 = test_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.savefig("accuracy_loss.jpg")
    plt.show()
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
    parser.add_argument('--epoch_start', action='store', default='0',
                        help='the start range of epoch')
    parser.add_argument('--epoch_end', action='store', default='320',
                        help='the end range of epoch')
    args = parser.parse_args()
    print('==> Options:', args)

    # set the seed
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)

    # prepare the data
    trainloader = Data.trainloader
    testloader = Data.testloader

    # define classes
    classes = Data.classes

    # define the model
    model = net_binary.Net()

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
        model.load_state_dict(pretrained_model['state_dict'])

    # model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # define solver and criterion
    lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params': [value], 'lr': lr, 'weight_decay': 0.00001}]

    optimizer = optim.Adam(params, lr=0.1, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    test_loss_list = []
    test_acc_list = []

    epoch_start = int(args.epoch_start)
    epoch_end = int(args.epoch_end)

    # start training
    for epoch in range(epoch_start, epoch_end):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()

    draw()
