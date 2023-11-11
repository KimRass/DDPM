# References:
    # https://wandb.ai/wandb_fc/korean/reports/-Frechet-Inception-distance-FID-GANs---Vmlldzo0MzQ3Mzc
    # https://github.com/w86763777/pytorch-ddpm/blob/master/score/fid.py

import torch
import numpy as np
import scipy
from tqdm import tqdm
import math
import argparse

from utils import get_config
from inceptionv3 import InceptionV3
from generate_image import get_ddpm_from_checkpoint
from train import get_tain_dl


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
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


class Evaluator(object):
    def __init__(self, n_samples, n_cpus, dl, device):

        self.n_samples = n_samples
        self.batch_size = dl.batch_size
        self.n_cpus = n_cpus
        self.dl = dl
        self.device = device

        self.inceptionv3 = InceptionV3().to(device)
        self.inceptionv3.eval()
        self.real_embed = self.get_real_embedding()

    @torch.no_grad()
    def get_real_embedding(self):
        embeds = list()
        di = iter(self.dl)
        for _ in range(math.ceil(self.n_samples // self.batch_size)):
            x0 = next(di)
            _, self.n_channels, self.img_size, _ = x0.shape
            x0 = x0.to(self.device)
            embed = self.inceptionv3(x0.detach())
            embeds.append(embed)
        real_embed = torch.cat(embeds)[: self.n_samples]
        return real_embed

    @torch.no_grad()
    def get_synthesized_embedding(self, ddpm, device):
        print("Calculating embeddings for synthetic data distribution...")

        embeds = list()
        for _ in tqdm(range(math.ceil(self.n_samples // self.batch_size))):
            x0 = ddpm.sample(
                batch_size=self.batch_size,
                n_channels=self.n_channels,
                img_size=self.img_size,
                device=device,
                to_image=False,
            )
            x0 = x0.to(self.device)
            embed = self.inceptionv3(x0.detach())
            embeds.append(embed)
        synth_embed = torch.cat(embeds)[: self.n_samples]
        return synth_embed

    def evaluate(self, ddpm, device):
        synth_embed = self.get_synthesized_embedding(ddpm=ddpm, device=device)
        fid = get_fid(self.real_embed, synth_embed)
        return fid


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(args)

    train_dl = get_tain_dl(
        data_dir=CONFIG["DATA_DIR"],
        img_size=CONFIG["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        n_cpus=CONFIG["N_CPUS"],
    )

    ddpm = get_ddpm_from_checkpoint(
        ckpt_path=CONFIG["CKPT_PATH"],
        n_timesteps=CONFIG["N_TIMESTEPS"],
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
        device=CONFIG["DEVICE"],
    )

    evaluator = Evaluator(
        n_samples=CONFIG["N_EVAL_IMAGES"], n_cpus=CONFIG["N_CPUS"], dl=train_dl, device=CONFIG["DEVICE"],
    )
    fid = evaluator.evaluate(ddpm=ddpm, device=CONFIG["DEVICE"])
    print(f"Frechet instance distance: {fid:.1f}")
