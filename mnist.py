# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt


def get_mnist_dataset(root):
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: (x - 0.5) * 2)]
    )
    ds = MNIST(root, download=True, train=True, transform=transform)
    return ds
