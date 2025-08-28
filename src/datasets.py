import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_cifar10(batch_size: int = 128, num_workers: int = 2):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = datasets.CIFAR10(root="~/.torch/data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root="~/.torch/data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False, persistent_workers=False)
    return train_loader, test_loader
