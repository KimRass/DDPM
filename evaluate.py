# References:
    # https://wandb.ai/wandb_fc/korean/reports/-Frechet-Inception-distance-FID-GANs---Vmlldzo0MzQ3Mzc
    # https://github.com/w86763777/pytorch-ddpm/blob/master/score/fid.py

import torch
import argparse
import numpy as np
import scipy
from tqdm import tqdm

from utils import get_config, get_noise, extract
from inceptionv3 import InceptionV3
from train import get_tain_dl


def _get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=32)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


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


# def get_real_embedding(dl, n):
def get_real_embedding(inceptionv3, dl):
    embeds = list()
    # cnt = 0
    for x0 in tqdm(dl):
        embed = inceptionv3(x0)
        embeds.append(embed)
    # return torch.cat(embeds)[: n]
    return torch.cat(embeds)


@torch.no_grad()
def sample_image(ddpm, batch_size, n_channels, img_size, device):
    ddpm.eval()
    # "1: $x_{T} \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$" ("Algorithm 2")
    x = get_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
    for t in range(ddpm.n_timesteps - 1, -1, -1):
        batched_t = torch.full(
            size=(batch_size,), fill_value=t, dtype=torch.long, device=device,
        )
        eps_theta = ddpm.predict_noise(x, t=batched_t) # "$z_{\theta}(x_{t}, t)$"

        beta_t = extract(ddpm.beta.to(device), t=t)
        alpha_t = extract(ddpm.alpha.to(device), t=t)
        alpha_bar_t = extract(ddpm.alpha_bar.to(device), t=t)

        # "$$\mu_{\theta}(x_{t}, t) =
        # \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}\epsilon_{\theta}(x_{t}, t)\Big)$$"
        x = (1 / (alpha_t ** 0.5)) * (x - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * eps_theta)

        if t > 0:
            eps = get_noise(
                batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device,
            ) # "3: $z \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$" ("Algorithm 2")
            x += (beta_t ** 0.5) * eps
    return x


def get_synthesized_embedding(n, ddpm, batch_size, n_channels, img_size, device):
    embeds = list()
    for _ in range(n // batch_size + 1):
        x0 = sample_image(
            ddpm=ddpm, batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device,
        )
        embed = inceptionv3(x0)
        embeds.append(embed)
    return torch.cat(embeds)[: n]


def evaluate():
    real_embed = 


if __name__ == "__main__":
    args = _get_args()
    CONFIG = get_config(args)

    inceptionv3 = InceptionV3()

    train_dl = get_tain_dl(CONFIG)
    real_embed = get_real_embedding(inceptionv3=inceptionv3, dl=train_dl, n=100)
    print(real_embed.shape)
