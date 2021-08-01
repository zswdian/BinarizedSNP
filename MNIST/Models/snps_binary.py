import torch.nn as nn
from BinarizedModules import BinSNPSConv2d


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinSNPSConv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            BinSNPSConv2d(50 * 4 * 4, 500, Linear=True),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x