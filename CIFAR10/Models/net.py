import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.BatchNorm2d(160, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.5),
            nn.PReLU(),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.5),
            nn.PReLU(),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), 10)
        return x