# References:
    # https://huggingface.co/blog/annotated-diffusion

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_config,
    get_device,
    gather,
    image_to_grid,
    show_forward_process,
    get_noise_like,
)
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
        self.beta = torch.linspace(init_beta, fin_beta, n_timesteps, device=device) # "$\beta_{t}$"
        self.alpha = 1 - self.beta # "$\alpha_{t} = 1 - \beta_{t}$"
        self.alpha_bar = self._get_alpha_bar(self.alpha) # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"

    def _get_alpha_bar(self, alpha):
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        return torch.cumprod(alpha, dim=0)

    def _q(self, t): # $q(x_{t} \vert x_{0})$
        alpha_bar_t = gather(self.alpha_bar, t=t) # "$\bar{\alpha_{t}}$"
        mean = (alpha_bar_t ** 0.5) # $\sqrt{\bar{\alpha_{t}}}$
        var = 1 - alpha_bar_t # $(1 - \bar{\alpha_{t}})\mathbf{I}$
        return mean, var

    def _sample_from_q(self, x0, t, eps):
        if eps is None:
            eps = get_noise_like(x0) # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        mean, var = self._q(t)
        return mean * x0 + (var ** 0.5) * eps

    def forward(self, x0, t, eps=None):
        x = self._sample_from_q(x0=x0, t=t, eps=eps)
        return x

    def estimate_noise(self, x, t):
        # The model returns its estimation of the noise that was added.
        return self.model(x, t)


if __name__ == "__main__":
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
    show_forward_process(ddpm=ddpm, dl=dl, device=DEVICE)

    generated = generate_image(
        ddpm=ddpm,
        batch_size=4,
        n_frames=100,
        img_size=28,
        device=DEVICE,
        gif_name="/Users/jongbeomkim/Downloads/test.gif"
    )
    grid = image_to_grid(generated, n_cols=int(batch_size ** 0.5))
    grid.show()
