import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar10(batch_size):
    transform = torchvision.models.ResNet101_Weights.DEFAULT.transforms()

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)

    return trainloader, testloader
