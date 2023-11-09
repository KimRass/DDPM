# References:
    # https://wandb.ai/wandb_fc/korean/reports/-Frechet-Inception-distance-FID-GANs---Vmlldzo0MzQ3Mzc
    # https://github.com/w86763777/pytorch-ddpm/blob/master/score/fid.py

import torch
import argparse
import numpy as np
import scipy
from tqdm import tqdm
import math

from utils import get_config, sample_noise, index_using_timestep
from inceptionv3 import InceptionV3
from generate_images import get_ddpm_from_checkpoint


def get_matrix_sqrt(x):
    sqrtm = scipy.linalg.sqrtm(np.array(x, dtype="float64"))
    if np.iscomplexobj(sqrtm):
       sqrtm = sqrtm.real
    return torch.from_numpy(sqrtm)


def get_mean_and_cov(embed):
    mu = embed.mean(dim=0).detach().cpu()
    sigma = torch.cov(embed.squeeze().T).detach().cpu()
    return mu, sigma


def get_frechet_distance(mu1, mu2, sigma1, sigma2):
    cov_product = get_matrix_sqrt(sigma1 @ sigma2)
    fd = ((mu1 - mu2) ** 2).sum() + torch.trace(sigma1 + sigma2 - 2 * cov_product)
    return fd.item()


def get_fid(embed1, embed2):
    mu1, sigma1 = get_mean_and_cov(embed1)
    mu2, sigma2 = get_mean_and_cov(embed2)
    fd = get_frechet_distance(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)
    return fd


@torch.no_grad()
def generate_image(ddpm, batch_size, n_channels, img_size, device):
    ddpm.eval()
    # Sample pure noise from a Gaussian distribution.
    # ["Algorithm 2"] "1: $x_{T} \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$"
    x = sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
    for timestep in range(ddpm.n_timesteps - 1, -1, -1):
        t = torch.full(
            size=(batch_size,), fill_value=timestep, dtype=torch.long, device=device,
        )
        eps_theta = ddpm.predict_noise(x, t=t) # "$z_{\theta}(x_{t}, t)$"

        beta_t = index_using_timestep(ddpm.beta.to(device), t=t)
        alpha_t = index_using_timestep(ddpm.alpha.to(device), t=t)
        alpha_bar_t = index_using_timestep(ddpm.alpha_bar.to(device), t=t)

        # Partially denoise image.
        # "$$\mu_{\theta}(x_{t}, t) =
        # \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}\epsilon_{\theta}(x_{t}, t)\Big)$$"
        x = (1 / (alpha_t ** 0.5)) * (x - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * eps_theta)

        if timestep > 0:
            eps = sample_noise(
                batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device,
            ) # ["Algorithm 2"] "3: $z \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$"
            x += (beta_t ** 0.5) * eps
    return x


class Evaluator(object):
    def __init__(self, n_samples, n_cpus, dl, device):

        self.n_samples = n_samples
        self.batch_size = dl.batch_size
        self.n_cpus = n_cpus
        self.dl = dl
        self.device = device

        self.inceptionv3 = InceptionV3().to(device)
        self.real_embed = self.get_real_embedding()

        self.n_channels = 3
        self.img_size = 4

    def get_real_embedding(self):
        embeds = list()
        di = iter(self.dl)
        # for _ in range(math.ceil(self.n_samples // self.batch_size)):
        for _ in tqdm(range(math.ceil(self.n_samples // self.batch_size))):
            # _, self.n_channels, self.img_size, _ = x0.shape
            x0 = next(di)
            x0 = x0.to(self.device)
            embed = self.inceptionv3(x0)
            embeds.append(embed)
        real_embed = torch.cat(embeds)[: self.n_samples]
        return real_embed

    def get_synthesized_embedding(self, ddpm, device):
        embeds = list()
        for _ in tqdm(range(math.ceil(self.n_samples // self.batch_size))):
            x0 = generate_image(
                ddpm=ddpm,
                batch_size=self.batch_size,
                n_channels=self.n_channels,
                img_size=self.img_size,
                device=device,
            )
            x0 = x0.to(self.device)
            embed = self.inceptionv3(x0)
            embeds.append(embed)
        synth_embed = torch.cat(embeds)[: self.n_samples]
        return synth_embed

    def evaluate(self, ddpm, device):
        synth_embed = self.get_synthesized_embedding(ddpm=ddpm, device=device)
        fid = get_fid(self.real_embed, synth_embed)
        return fid
