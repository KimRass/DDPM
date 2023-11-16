# References:
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
    # https://medium.com/mlearning-ai/understanding-the-diffusion-model-and-the-theory-tensorflow-cafcd5752b98

import torch
import torch.nn as nn
import numpy as np
import imageio
from tqdm import tqdm
import math
from pathlib import Path

from utils import (
    sample_noise,
    index,
    image_to_grid,
    get_linear_beta_schdule,
    save_image,
    to_batched_timesteps,
)
from model import UNet


class DDIM(nn.Module):
    def __init__(self, n_timesteps=1000, n_ddim_timesteps=250, init_beta=0.0001, fin_beta=0.02):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_ddim_timesteps = n_ddim_timesteps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.step_size = self.n_timesteps // self.n_ddim_timesteps

        beta = get_linear_beta_schdule(
            init_beta=init_beta, fin_beta=fin_beta, n_timesteps=n_timesteps,
        )
        alpha = 1 - beta # "$\alpha_{t} = 1 - \beta_{t}$")
        self.alpha_bar = self._get_alpha_bar(alpha)
        self.alpha_bar = self.alpha_bar[range(1, self.n_timesteps - self.step_size + 2, self.step_size)]
        # print(self.alpha_bar)

        self.model = UNet(n_timesteps=n_timesteps)

    def _get_alpha_bar(self, alpha):
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        return torch.cumprod(alpha, dim=0)

    def _q(self, t): # $q(x_{t} \vert x_{0})$
        alpha_bar_t = index(self.alpha_bar.to(t.device), t=t) # "$\bar{\alpha_{t}}$"
        mean = (alpha_bar_t ** 0.5) # $\sqrt{\bar{\alpha_{t}}}$
        var = 1 - alpha_bar_t # $(1 - \bar{\alpha_{t}})\mathbf{I}$
        return mean, var

    def _sample_from_q(self, x0, t, eps):
        b, c, h, _ = x0.shape
        if eps is None:
            # "$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"
            eps = sample_noise(batch_size=b, n_channels=c, img_size=h, device=x0.device)
        mean, var = self._q(t)
        return mean * x0 + (var ** 0.5) * eps

    def forward(self, x0, t, eps=None): # Forward (diffusion) process
        noisy_image = self._sample_from_q(x0=x0, t=t, eps=eps)
        return noisy_image

    def predict_noise(self, x, t):
        # The model returns its estimation of the noise that was added.
        return self.model(x, t=t)

    @torch.no_grad()
    def _sample_from_p(self, x, timestep, idx):
        self.model.eval()

        b, _, _, _ = x.shape
        batched_idx = to_batched_timesteps(timestep=idx, batch_size=b, device=x.device)
        batched_idx_tm1 = to_batched_timesteps(timestep=idx - 1, batch_size=b, device=x.device)
        # alpha_bar_t = index(self.alpha_bar.to(x.device), t=batched_idx)
        # alpha_bar_tm1 = index(self.alpha_bar.to(x.device), t=batched_idx_tm1)
        alpha_bar_t = index(self.alpha_bar.to(x.device), t=batched_idx_tm1)
        alpha_bar_tm1 = index(self.alpha_bar.to(x.device), t=batched_idx)

        t = to_batched_timesteps(timestep=timestep, batch_size=b, device=x.device)
        # tm1 = to_batched_timesteps(timestep=timestep - self.step_size, batch_size=b, device=x.device)
        pred_noise = self.predict_noise(x.detach(), t=t)

        out = (alpha_bar_tm1 ** 0.5) * (x - ((1 - alpha_bar_t) ** 0.5) * pred_noise) / (alpha_bar_t ** 0.5)
        out += ((1 - alpha_bar_tm1) ** 0.5) * pred_noise

        self.model.train()
        return out

    def sample(
        self, batch_size, n_channels, img_size, device, return_image=True,
    ): # Reverse (denoising) process
        x = sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
        # pbar = range(self.n_timesteps - 1, -1, -self.step_size)
        pbar = range(self.n_timesteps - self.step_size + 1, -1, -self.step_size)
        if device.type == "cpu":
            pbar = tqdm(pbar)
        for i, timestep in enumerate(pbar):
            idx = len(pbar) - i - 1
            # print(idx)
            # x = self._sample_from_p(x=x, timestep=timestep)
            x = self._sample_from_p(x=x, timestep=timestep, idx=idx)
            # if timestep - self.step_size < 0:
            #     break
        if not return_image:
            return x

        image = image_to_grid(x, n_cols=int(batch_size ** 0.5))
        return image
