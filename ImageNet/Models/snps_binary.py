import torch.nn as nn
import torch
from BinarizedModules import BinSNPSConv2d


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinSNPSConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinSNPSConv2d(256, 384, kernel_size=3, stride=1, padding=1),
            BinSNPSConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1),
            BinSNPSConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            BinSNPSConv2d(256 * 8 * 8, 4096, Linear=True),
            BinSNPSConv2d(4096, 4096, dropout=0.5, Linear=True),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 8 * 8)
        x = self.classifier(x)
        return x


def Net(pretrained=False, **kwargs):

    model = Net(**kwargs)
    if pretrained:
        model_path = '/Experiment/snps_bin.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model