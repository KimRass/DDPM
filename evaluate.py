# References:
    # https://wandb.ai/wandb_fc/korean/reports/-Frechet-Inception-distance-FID-GANs---Vmlldzo0MzQ3Mzc
    # https://github.com/w86763777/pytorch-ddpm/blob/master/score/fid.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy
from tqdm import tqdm
import math
import argparse

from utils import get_config
from inceptionv3 import InceptionV3
from generate_image import get_ddpm_from_checkpoint
from celeba import CelebADataset, ImageGridDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--real_data_dir", type=str, required=True)
    parser.add_argument("--gen_data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_eval_imgs", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def get_matrix_sqrt(x):
    conv_mean = scipy.linalg.sqrtm(x)
    if np.iscomplexobj(conv_mean):
       conv_mean = conv_mean.real
    return conv_mean


def get_mean_and_cov(embed):
    mu = embed.mean(axis=0)
    sigma = np.cov(embed, rowvar=False)
    return mu, sigma


def get_frechet_distance(mu1, mu2, sigma1, sigma2):
    cov_mean = get_matrix_sqrt(sigma1 @ sigma2)
    fd = ((mu1 - mu2) ** 2).sum() + np.trace(sigma1 + sigma2 - 2 * cov_mean)
    return fd.item()


def get_fid(embed1, embed2):
    embed1 = np.random.random(10*2048).reshape((10,2048))
    embed2 = np.random.random(10*2048).reshape((10,2048))
    mu1, sigma1 = get_mean_and_cov(embed1)
    mu2, sigma2 = get_mean_and_cov(embed2)
    fd = get_frechet_distance(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)
    return fd


class Evaluator(object):
    def __init__(self, ddpm, n_eval_imgs, batch_size, real_dl, gen_dl, device):

        self.ddpm = ddpm
        self.ddpm.eval()
        self.n_eval_imgs = n_eval_imgs
        self.batch_size = batch_size
        self.real_dl = real_dl
        self.gen_dl = gen_dl
        self.device = device

        self.inceptionv3 = InceptionV3().to(device)
        self.inceptionv3.eval()

        self.real_embed = self.get_real_embedding()
        self.gen_embed = self.get_generated_embedding()
        print(self.real_embed.shape, self.gen_embed.shape)

    def _to_embeddding(self, x):
        embed = self.inceptionv3(x.detach())
        embed = embed.squeeze()
        embed = embed.cpu().numpy()
        return embed

    @torch.no_grad()
    def get_real_embedding(self):
        embeds = list()
        di = iter(self.real_dl)
        for _ in tqdm(range(math.ceil(self.n_eval_imgs // self.batch_size))):
            x0 = next(di)
            _, self.n_channels, self.img_size, _ = x0.shape
            x0 = x0.to(self.device)
            embed = self._to_embeddding(x0)
            embeds.append(embed)
        real_embed = np.concatenate(embeds)[: self.n_eval_imgs]
        return real_embed

    @torch.no_grad()
    def get_generated_embedding(self):
        embeds = list()
        di = iter(self.real_dl)
        for _ in tqdm(range(math.ceil(self.n_eval_imgs // self.batch_size))):
            x0 = next(di)
            _, self.n_channels, self.img_size, _ = x0.shape
            x0 = x0.to(self.device)
            embed = self._to_embeddding(x0)
            embeds.append(embed)
        gen_embed = np.concatenate(embeds)[: self.n_eval_imgs]
        return gen_embed

    def evaluate(self):
        fid = get_fid(self.real_embed, self.gen_embed)
        return fid


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(args)

    real_ds = CelebADataset(data_dir=CONFIG["REAL_DATA_DIR"], img_size=CONFIG["IMG_SIZE"])
    real_dl = DataLoader(
        real_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["N_CPUS"],
        pin_memory=False,
        drop_last=False,
    )
    gen_ds = ImageGridDataset(
        data_dir=CONFIG["GEN_DATA_DIR"],
        img_size=CONFIG["IMG_SIZE"],
    )
    gen_dl = DataLoader(
        gen_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["N_CPUS"],
        pin_memory=False,
        drop_last=False,
    )

    ddpm = get_ddpm_from_checkpoint(
        ckpt_path=CONFIG["CKPT_PATH"],
        n_timesteps=CONFIG["N_TIMESTEPS"],
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
        device=CONFIG["DEVICE"],
    )

    evaluator = Evaluator(
        ddpm=ddpm,
        n_eval_imgs=CONFIG["N_EVAL_IMGS"],
        batch_size=CONFIG["BATCH_SIZE"],
        real_dl=real_dl,
        gen_dl=gen_dl,
        device=CONFIG["DEVICE"],
    )
    fid = evaluator.evaluate()
    print(f"Frechet instance distance: {fid:.2f}")
