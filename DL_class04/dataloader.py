import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models
import torchvision.transforms


def get_loader(batch_size, num_workers):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    dataset_dir = "~/.torchvision/dataset/CIFAR10"
    train_dataset = torchvision.datasets.CIFAR10(dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(dataset_dir, train=False, transform=test_transform, download=True)
    # pin_memory: use all memory
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                              shuffle=False, pin_memory=True, drop_last=False)
    return train_loader, test_loader
