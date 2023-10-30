# References:
    # https://huggingface.co/blog/annotated-diffusion

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_config, get_device, image_to_grid
from data import get_mnist_dataset

CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/DDPM/config.yaml")
# CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

batch_size = 16
ds = get_mnist_dataset("/Users/jongbeomkim/Documents/datasets")
dl = DataLoader(ds, batch_size, shuffle=True)

image, _ = next(iter(dl))
grid = image_to_grid(image, 4)
grid.show()


class DDPM(nn.Module):
    def __init__(self, model, init_beta, fin_beta, n_timesteps, device):
        super().__init__()

        self.device = device
        self.model = model.to(device)

        # "We set the forward process variances to constants increasing linearly from $\beta_{1} = 10^{-4}$
        # to $\beta_{T} = 0.02$.
        self.betas = torch.linspace(init_beta, fin_beta, n_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        # Run each image through the model for each timestep t in the vector t.
        # The model returns its estimation of the noise that was added.
        return self.model(x, t)


ddpm = DDPM(
    model=model,
    init_beta=CONFIG["INIT_BETA"],
    fin_beta=CONFIG["FIN_BETA"],
    n_timesteps=CONFIG["N_TIMESTEPS"],
    device=DEVICE,
)