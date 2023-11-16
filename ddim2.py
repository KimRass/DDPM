"""
---
title: Denoising Diffusion Implicit Models (DDIM) Sampling
summary: >
 Annotated PyTorch implementation/tutorial of
 Denoising Diffusion Implicit Models (DDIM) Sampling
 for stable diffusion model.
---

# Denoising Diffusion Implicit Models (DDIM) Sampling

This implements DDIM sampling from the paper
[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
"""

from typing import Optional, List

import numpy as np
import torch

from labml import monit
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.sampler import DiffusionSampler


class DDIMSampler(DiffusionSampler):
    """
    ## DDIM Sampler

    This extends the [`DiffusionSampler` base class](index.html).

    DDIM samples images by repeatedly removing noise by sampling step by step using,

    \begin{align}
    x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
            \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
            \Bigg) \\
            &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
            &+ \sigma_{\tau_i} \epsilon_{\tau_i}
    \end{align}

    where $\epsilon_{\tau_i}$ is random noise,
    $\tau$ is a subsequence of $[1,2,\dots,T]$ of length $S$,
    and
    $$\sigma_{\tau_i} =
    \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
    \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$

    Note that, $\alpha_t$ in DDIM paper refers to ${\color{lightgreen}\bar\alpha_t}$ from [DDPM](ddpm.html).
    """

    def __init__(self, model, n_ddim_timesteps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        :param n_steps: is the number of DDIM sampling steps, $S$
        :param ddim_discretize: specifies how to extract $\tau$ from $[1,2,\dots,T]$.
            It can be either `uniform` or `quad`.
        :param ddim_eta: is $\eta$ used to calculate $\sigma_{\tau_i}$. $\eta = 0$ makes the
            sampling process deterministic.
        """
        super().__init__(model)
        c = self.n_timesteps // n_ddim_timesteps
        self.time_steps = np.asarray(list(range(0, self.n_timesteps, c))) + 1

        with torch.no_grad():
            self.alpha_bar = alpha_bar[self.time_steps].clone().to(torch.float32)
            self.alpha_bar_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])

            self.ddim_sigma = (ddim_eta *
                               ((1 - self.alpha_bar_prev) / (1 - self.alpha_bar) *
                                (1 - self.alpha_bar / self.alpha_bar_prev)) ** .5)

    def get_x_prev_and_pred_x0(self, pred_noise: torch.Tensor, index: int, x: torch.Tensor, *,
                               repeat_noise: bool):
        """
        ### Sample $x_{\tau_{i-1}}$ given $\epsilon_\theta(x_{\tau_i})$
        """

        alpha_bar_t = self.alpha_bar[index]
        alpha_bar_tm1 = self.alpha_bar_prev[index]
        sigma_t = self.ddim_sigma[index]
        # sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        signal_rate = alpha_bar_tm1 ** 0.5
        signal = (x - ((1 - alpha_bar_t) ** 0.5) * pred_noise) / (alpha_bar_t ** 0.5)
        (1 - alpha_bar_tm1 - sigma_t ** 2) ** 0.5
        noise_rate = (1 - alpha_bar_tm1 - sigma_t ** 2) ** 0.5

        # if sigma_t == 0.:
        #     noise = 0.
        # # If same noise is used for all samples in the batch
        # elif repeat_noise:
        #     noise = torch.randn((1, *x.shape[1:]), device=x.device)
        # else:
        # noise = torch.randn(x.shape, device=x.device)

        # x_prev = signal_rate * signal + noise_rate * pred_noise + sigma_t * noise
        x_prev = signal_rate * signal + noise_rate * pred_noise
        return x_prev

    # @torch.no_grad()
    # def p_sample(
    #     self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, *,
    #     repeat_noise: bool = False,
    #     uncond_scale: float = 1,
    #     uncond_cond: Optional[torch.Tensor] = None
    # ):
    #     """
    #     ### Sample $x_{\tau_{i-1}}$

    #     :param x: is $x_{\tau_i}$ of shape `[batch_size, channels, height, width]`
    #     :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
    #     :param t: is $\tau_i$ of shape `[batch_size]`
    #     :param step: is the step $\tau_i$ as an integer
    #     :param index: is index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
    #     :param repeat_noise: specified whether the noise should be same for all samples in the batch
    #     """
    #     pred_noise = self.predict_noise(x, t, c,
    #                        uncond_scale=uncond_scale,
    #                        uncond_cond=uncond_cond)

    #     x_prev, pred_x0 = self.get_x_prev_and_pred_x0(
    #         pred_noise, index, x, repeat_noise=repeat_noise,
    #     )
    #     return x_prev, pred_x0, pred_noise

    @torch.no_grad()
    def sample(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               ):
        """
        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$.
            And `x_last` is then $x_{\tau_{S - i'}}$.
        """

        # Get device and batch size
        device = self.model.device
        bs = shape[0]

        # Get $x_{\tau_S}$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at $\tau_{S - i'}, \tau_{S - i' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in monit.enum('Sample', time_steps):
            index = len(time_steps) - i - 1
            ts = x.new_full((bs,), step, dtype=torch.long)
            pred_noise = self.predict_noise(
                x,
                ts,
                cond,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )
            x = self.get_x_prev_and_pred_x0(
                pred_noise, index, x, repeat_noise=repeat_noise,
            )
        return x

    # @torch.no_grad()
    # def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
    #     if noise is None:
    #         noise = torch.randn_like(x0)
    #     return ((self.alpha_bar) ** 0.5)[index] * x0 + ((1 - self.alpha_bar) ** 0.5)[index] * noise


n_timesteps = 1000
n_ddim_timesteps = 200
c = n_timesteps // n_ddim_timesteps
time_steps = np.asarray(list(range(0, n_timesteps, c))) + 1
alpha_bar = np.arange(1000)
alpha_bar[time_steps[:-1]]
alpha_bar[:1]
torch.cat([alpha_bar[0:1], alpha_bar[time_steps[:-1]]])
# [i for i in np.flip(time_steps)]
[(len(time_steps) - i - 1, step) for i, step in enumerate(np.flip(time_steps))]

step_size = n_timesteps // n_ddim_timesteps
list(range(n_timesteps -step_size + 1, -1, -step_size))
list(range(1, n_timesteps - step_size + 2, step_size))



time_steps = range(0, 1000, 5)
alpha_bar = np.arange(1000)
a1 = alpha_bar[time_steps]
a2 = np.concatenate([alpha_bar[:1], alpha_bar[time_steps[: -1]]])
a1 - a2