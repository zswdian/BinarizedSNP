import torch
import torch.nn as nn


class BinActive(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        n = input[0].nelement()
        s = input.size()
        beta = input.norm(1, 3, keepdim=True)\
            .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
        return input.sign().mul(beta.expand_as(s))

    def backward(self, grad_output):
        input, = self.saved_tensors
        n = input[0].nelement()
        s = input.size()
        grad_input = input.norm(1, 3, keepdim=True).sum(2, keepdim=True)\
            .sum(1, keepdim=True).div(n).expand(s)
        grad_input[input.lt(-1.0)] = 0
        grad_input[input.gt(1.0)] = 0
        grad_input = grad_input.mul(grad_output)
        grad_input_add = input.sign().mul(grad_output)
        grad_input_add = grad_input_add.sum(3, keepdim=True).sum(2, keepdim=True)\
            .sum(1, keepdim=True).div(n).expand(s)
        grad_input_add = grad_input_add.mul(input.sign())
        return grad_input.add(grad_input_add).mul(1.0 - 1.0 / s[1]).mul(n)


class BinConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=-1,
                 stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        # self.pRelu = nn.PReLU(0.25)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.bn(input)
        # x = self.pRelu(input)
        x = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
