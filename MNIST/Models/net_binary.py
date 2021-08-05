import torch.nn as nn
from CIFAR10.BinarizedModules import BinConv2d


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinConv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            BinConv2d(50 * 4 * 4, 500, Linear=True),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x


