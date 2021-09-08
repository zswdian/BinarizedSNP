import torch.nn as nn
from BinarizedModules import BinSNPConv2d


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            BinSNPConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinSNPConv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BinSNPConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
            BinSNPConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinSNPConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BinSNPConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
            BinSNPConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), 10)
        return x