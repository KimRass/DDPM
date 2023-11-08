# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torchvision.transforms as T
from torchvision.datasets.mnist import MNIST


def get_mnist_dataset(root):
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: (x - 0.5) * 2)]
    )
    ds = MNIST(root, download=True, train=True, transform=transform)
    return ds
