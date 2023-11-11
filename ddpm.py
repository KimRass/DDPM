# References:
    # https://huggingface.co/blog/annotated-diffusion

import torch
import torch.nn as nn
import numpy as np
import imageio
from tqdm import tqdm

from utils import (
    sample_noise,
    index,
    image_to_grid,
    get_linear_beta_schdule,
)
from model import UNet


def _get_linear_beta_schdule(init_beta, fin_beta, n_timesteps):
    # "We set the forward process variances to constants increasing linearly."
    # return torch.linspace(init_beta, fin_beta, n_timesteps) # "$\beta_{t}$"
    return torch.linspace(init_beta, fin_beta, n_timesteps + 1) # "$\beta_{t}$"


class DDPM(nn.Module):
    def __init__(self, n_timesteps=1000, init_beta=0.0001, fin_beta=0.02):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.beta = get_linear_beta_schdule(
            init_beta=init_beta, fin_beta=fin_beta, n_timesteps=n_timesteps,
        )
        self.alpha = 1 - self.beta # "$\alpha_{t} = 1 - \beta_{t}$"
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        self.alpha_bar = self._get_alpha_bar(self.alpha)

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
    def _sample_from_p(self, x, timestep):
        self.model.eval()

        b, c, h, _ = x.shape
        t = torch.full(
            size=(b,), fill_value=timestep, dtype=torch.long, device=x.device,
        )
        alpha_t = index(self.alpha.to(x.device), t=t)
        alpha_bar_t = index(self.alpha_bar.to(x.device), t=t)
        eps_theta = self.predict_noise(x.detach(), t=t)
        # # ["Algorithm 2"] "4: $$\mu_{\theta}(x_{t}, t) =
        # \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}z_{\theta}(x_{t}, t)\Big)$$
        # + \sigma_{t}z"
        model_mean = (1 / (alpha_t ** 0.5)) * (x - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * eps_theta)

        if timestep > 0:
            beta_t = index(self.beta.to(x.device), t=t)
            eps = sample_noise(batch_size=b, n_channels=c, img_size=h, device=x.device)
            model_mean += (beta_t ** 0.5) * eps

        self.model.train()
        return model_mean

    def sample(self, batch_size, n_channels, img_size, device, n_cols=0): # Reverse process
        x = sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
        for timestep in tqdm(range(self.n_timesteps - 1, -1, -1)):
            x = self._sample_from_p(x, timestep=timestep)

        if n_cols == 0:
            n_cols = int(batch_size ** 0.5)
        image = image_to_grid(x, n_cols=n_cols)
        return image

    def _get_frame(self, x):
        b, _, _, _ = x.shape
        grid = image_to_grid(x, n_cols=int(b ** 0.5))
        frame = np.array(grid)
        return frame

    def progressively_sample(self, batch_size, n_channels, img_size, device, save_path, n_frames=100):
        with imageio.get_writer(save_path, mode="I") as writer:
            x = sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
            for timestep in tqdm(range(self.n_timesteps - 1, -1, -1)):
                x = self._sample_from_p(x, timestep=timestep)

                if timestep % (self.n_timesteps // n_frames) == 0:
                    frame = self._get_frame(x)
                    writer.append_data(frame)

    def _linearly_interpolate(self, x, y, n):
        _, b, c, d = x.shape
        lambs = torch.linspace(start=0, end=1, steps=n)
        lambs = lambs[:, None, None, None].expand(n, b, c, d)
        return ((1 - lambs) * x + lambs * y)

    def interpolate(self, image1, image2, timestep, n=10, to_image=True):
        t = torch.full(size=(1,), fill_value=timestep)
        noisy_image1 = self(image1, t=t)
        noisy_image2 = self(image2, t=t)

        x = self._linearly_interpolate(noisy_image1, noisy_image2, n=n)
        for timestep in tqdm(range(timestep - 1, -1, -1)):
            x = self._sample_from_p(x, timestep=timestep)
        x = torch.cat([image1, x, image2], dim=0)
        if not to_image:
            return x
        image = image_to_grid(x, n_cols=n + 2)
        return image

    def coarse_to_fine_interpolate(self, image1, image2, n_rows=9, n=10):
        rows = list()
        for timestep in range(self.n_timesteps, -1, - self.n_timesteps // (n_rows - 1)):
            row = self.interpolate(image1, image2, timestep=timestep, n=n, to_image=False)
            rows.append(row)
        x = torch.cat(rows, dim=0)
        image = image_to_grid(x, n_cols=n + 2)
        return image
