import torch
import torch.nn as nn


class BinActiv(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        mean = input.abs().mean(dim=1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(1)] = 0
        grad_input[input.ge(-1)] = 0
        return grad_input


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
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding)
        # self.avg = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride,
        #                         padding=self.padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.bn(input)
        x, mean = BinActiv()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        x = self.conv(x)
        # beta = self.avg(mean)
        # x = x.mul(beta)
        x = self.relu(x)
        return x
