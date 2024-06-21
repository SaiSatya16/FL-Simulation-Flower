from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import torch


def get_mnist():

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=tr)
    testset = MNIST(root='./data', train=False, download=True, transform=tr)
    return trainset, testset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float=0.1):
    trainset, testset = get_mnist()
    #split train set into 'num_partitions' trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(trainset, partition_len, generator=torch.Generator().manual_seed(2023))

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset in trainsets:
        mum_total = len(trainset)
        num_val = int(val_ratio * mum_total)
        num_train = mum_total - num_val
        
        for_train, for_val = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(2023))
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = DataLoader(testset, batch_size=128)
    return trainloaders, valloaders, testloader