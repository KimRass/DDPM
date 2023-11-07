# References:
    # https://huggingface.co/blog/annotated-diffusion

import torch
import torch.nn as nn

from utils import extract, get_noise
from model import UNet


def _get_linear_beta_schdule(init_beta, fin_beta, n_timesteps):
    return torch.linspace(init_beta, fin_beta, n_timesteps) # "$\beta_{t}$"


class DDPM(nn.Module):
    def __init__(self, n_timesteps, init_beta, fin_beta):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.model = UNet(n_timesteps=n_timesteps)

        # "We set the forward process variances to constants increasing linearly from $\beta_{1} = 10^{-4}$
        # to $\beta_{T} = 0.02$.
        self.beta = _get_linear_beta_schdule(
            init_beta=init_beta, fin_beta=fin_beta, n_timesteps=n_timesteps,
        )
        self.alpha = 1 - self.beta # "$\alpha_{t} = 1 - \beta_{t}$"
        self.alpha_bar = self._get_alpha_bar(self.alpha) # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"

    def _get_alpha_bar(self, alpha):
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        return torch.cumprod(alpha, dim=0)

    def _q(self, t): # $q(x_{t} \vert x_{0})$
        alpha_bar_t = extract(self.alpha_bar.to(t.device), t=t) # "$\bar{\alpha_{t}}$"
        mean = (alpha_bar_t ** 0.5) # $\sqrt{\bar{\alpha_{t}}}$
        var = 1 - alpha_bar_t # $(1 - \bar{\alpha_{t}})\mathbf{I}$
        return mean, var

    def _sample_from_q(self, x0, t, eps):
        b, c, h, _ = x0.shape
        if eps is None:
            # "$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"
            eps = get_noise(batch_size=b, n_channels=c, img_size=h, device=x0.device)
        mean, var = self._q(t)
        return mean * x0 + (var ** 0.5) * eps

    def forward(self, x0, t, eps=None):
        noisy_image = self._sample_from_q(x0=x0, t=t, eps=eps)
        return noisy_image

    def estimate_noise(self, x, t):
        # The model returns its estimation of the noise that was added.
        return self.model(x, t)
