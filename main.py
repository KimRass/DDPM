# References:
    # https://huggingface.co/blog/annotated-diffusion

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_config, get_device, image_to_grid, show_forward
from data import get_mnist_dataset
from model import UNetForDDPM
from generate_images import generate_image


class DDPM(nn.Module):
    def __init__(self, model, init_beta, fin_beta, n_timesteps, device):
        super().__init__()

        self.model = model.to(device)
        self.n_timesteps = n_timesteps
        self.device = device

        # "We set the forward process variances to constants increasing linearly from $\beta_{1} = 10^{-4}$
        # to $\beta_{T} = 0.02$.
        self.betas = torch.linspace(init_beta, fin_beta, n_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]
        ).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more x. (We can directly skip to the desired step.)
        b, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(b, c, h, w).to(self.device)

        x = (a_bar ** 0.5).reshape(b, 1, 1, 1) * x0 + ((1 - a_bar) ** 0.5).reshape(b, 1, 1, 1) * eta
        return x

    def backward(self, x, t):
        # The model returns its estimation of the noise that was added.
        return self.model(x, t)


CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/DDPM/config.yaml")
# CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

batch_size = 16
ds = get_mnist_dataset("/Users/jongbeomkim/Documents/datasets")
dl = DataLoader(ds, batch_size, shuffle=True)

image, _ = next(iter(dl))
grid = image_to_grid(image, 4)
grid.show()


model = UNetForDDPM(n_timesteps=CONFIG["N_TIMESTEPS"])
ddpm = DDPM(
    model=model,
    init_beta=CONFIG["INIT_BETA"],
    fin_beta=CONFIG["FIN_BETA"],
    n_timesteps=CONFIG["N_TIMESTEPS"],
    device=DEVICE,
)
show_forward(ddpm=ddpm, dl=dl, device=DEVICE)

generated = generate_image(
    ddpm=ddpm,
    batch_size=16,
    frames_per_gif=100,
    img_size=28,
    device=DEVICE,
    gif_name="/Users/jongbeomkim/Downloads/test.gif"
)