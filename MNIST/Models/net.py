import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(50, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(50 * 4 * 4, 500, kernel_size=5, stride=1, padding=0),
            nn.PReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.net(x)
        return x