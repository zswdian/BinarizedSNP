import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.BatchNorm2d(1, eps=1e-4, momentum=0.1, affine=True),
            nn.PReLU(),
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(),
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(),
            nn.BatchNorm2d(50, eps=1e-4, momentum=0.1, affine=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.PReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x