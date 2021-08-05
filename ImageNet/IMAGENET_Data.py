import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

input_size = 227
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[1. / 255., 1. / 255., 1. / 255.])

train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])

train_set = datasets.ImageFolder(root='Data/train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                           shuffle=True, num_workers=8, pin_memory=True)

val_set = datasets.ImageFolder(root='Data/val', transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=256,
                                         shuffle=False, num_workers=8, pin_memory=True)
