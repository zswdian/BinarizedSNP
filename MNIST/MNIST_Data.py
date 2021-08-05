import torchvision
import torchvision.transforms as transforms
import torch.utils.data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

DOWNLOAD = False

trainset = torchvision.datasets.MNIST(root='Data', train=True,
                                      download=DOWNLOAD, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='Data', train=False,
                                     download=DOWNLOAD, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=True, num_workers=2)
