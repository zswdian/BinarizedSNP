import torch.nn as nn
from BinarizedModules import BinSNPSConv2d


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinSNPSConv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinSNPSConv2d(50 * 4 * 4, 500, kernel_size=5, stride=1, padding=0),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.net(x)
        return x