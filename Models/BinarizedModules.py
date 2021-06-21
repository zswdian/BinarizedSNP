import torch
import torch.nn as nn


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
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
