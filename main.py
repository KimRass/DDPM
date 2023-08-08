# References:
    # https://nn.labml.ai/diffusion/ddpm/index.html

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


# def __init__(tot_n_steps, eps_theta_model):
tot_n_steps = 100
# self.tot_n_steps = tot_n_steps # $T$
# self.eps_theta_model = eps_theta_model # $\epsilon_{\theta}(x_{t}, t)$
beta = torch.linspace(0.0001, 0.02, tot_n_steps) # $\beta_{t}$
alpha = 1 - beta # $\alpha_{t} = 1 - \beta_{t}$
alpha_bar = torch.cumprod(alpha, dim=0) # $\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$
sigma_square = beta


def q(x0, t): # $q(x_{t} \vert x_{0})$
    # x0 = torch.randn((4, 3, 200, 300))
    # t = torch.tensor([11])
    alpha_bar_t = torch.gather(alpha_bar, dim=0, index=t) # $\bar{\alpha_{t}}$
    mean = alpha_bar_t ** 0.5 * x0 # $\sqrt{\bar{\alpha_{t}}}x_{0}$
    var = 1 - alpha_bar_t # $(1 - \bar{\alpha_{t}})\mathbf{I}$
    return mean, var

def sample_from_q(x0, t, eps):
    if eps is None:
        eps = torch.randn_like(x0) # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
    mean, var = q(x0=x0, t=t, eps=None)
    return mean + var ** 0.5 * eps

def sample_from_p(xt, t):
    alpha_t = torch.gather(alpha, dim=0, index=t) # $\alpha_{t}$
    alpha_bar_t = torch.gather(alpha_bar, dim=0, index=t) # $\bar{\alpha_{t}}$
    beta_t = torch.gather(beta, dim=0, index=t) # $\beta_{t}$
    # $\frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}\epsilon_{\theta}(x_{t}, t)\Big)$
    mean = (1 / alpha_t ** 0.5) * (xt - beta_t / ((1 - alpha_bar_t) ** 0.5) * eps_theta_model(xt, t))

    sigma_square_t = torch.gather(sigma_square, dim=0, index=t) # $\sigma_{t}^{2}\mathbf{I}$
    var = sigma_square_t

    eps = torch.randn_like(xt, device=xt.device) # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
    return mean + var ** 0.5 * eps

def loss