import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

input_size = 227

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])

train_set = datasets.ImageFolder('Data/tiny-imagenet-200/train', transform=transform_train)
val_set = datasets.ImageFolder('Data/tiny-imagenet-200/val', transform=transform_val)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                           shuffle=True, num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=256,
                                         shuffle=False, num_workers=8, pin_memory=True)
