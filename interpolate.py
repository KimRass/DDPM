import torch
import numpy as np
import scipy
from tqdm import tqdm
import argparse
from PIL import Image

from utils import get_config, sample_noise
from inceptionv3 import InceptionV3
from generate_images import get_ddpm_from_checkpoint
from celeba import get_transformer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--timestep", type=int, required=True)
    # parser.add_argument("--batch_size", type=int, required=True)
    # parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=32)
    parser.add_argument("--n_frames", type=int, required=False, default=100)

    args = parser.parse_args()
    return args


def interpolate(image1, image2, n=10):
    _, b, c, d = image1.shape
    lambs = torch.linspace(start=0, end=1, steps=n)
    lambs = lambs[:, None, None, None]
    lambs = lambs.expand(n, b, c, d)
    return (lambs * image1 + (1 - lambs) * image2)


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(args)

    ddpm = get_ddpm_from_checkpoint(
        ckpt_path=CONFIG["CKPT_PATH"],
        n_timesteps=CONFIG["N_TIMESTEPS"],
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
        device=CONFIG["DEVICE"],
    )
    # t = torch.full(size=(1,), fill_value=CONFIG["TIMESTEP"])
    # eps = sample_noise(
    #     batch_size=CONFIG["BATCH_SIZE"],
    #     n_channels=CONFIG["N_CHANNELS"],
    #     img_size=CONFIG["IMG_SIZE"],
    #     device=CONFIG["DEVICE"],
    # )

    image1 = Image.open("/Users/jongbeomkim/Documents/datasets/celeba/img_align_celeba/001700.jpg")
    image2 = Image.open("/Users/jongbeomkim/Documents/datasets/celeba/img_align_celeba/001800.jpg")

    transformer = get_transformer(img_size=CONFIG["IMG_SIZE"], hflip=False)
    x01 = transformer(image1).unsqueeze(0)
    x02 = transformer(image2).unsqueeze(0)
    gen_image = ddpm.sample_using_interpolation(x01, x02, timestep=CONFIG["TIMESTEP"])
    gen_image.show()
    # noisy_image1 = ddpm(x01, t=t, eps=eps)
    # noisy_image2 = ddpm(x02, t=t, eps=eps)
    # noisy_image = interpolate(noisy_image1, noisy_image2, n=10)

    # perform_progressive_generation(
    #     ddpm=ddpm,
    #     batch_size=CONFIG["BATCH_SIZE"],
    #     n_channels=CONFIG["N_CHANNELS"],
    #     img_size=CONFIG["IMG_SIZE"],
    #     device=CONFIG["DEVICE"],
    #     save_path=CONFIG["SAVE_PATH"],
    #     x_t=noisy_image,
    #     start_timestep=CONFIG["TIMESTEP"],
    #     image_only=True,
    # )
