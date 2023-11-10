# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from tqdm import tqdm

from utils import (
    get_config,
    sample_noise,
    index_using_timestep,
    image_to_grid,
    save_image,
)
from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=32)
    parser.add_argument("--n_frames", type=int, required=False, default=100)

    args = parser.parse_args()
    return args


def get_ddpm_from_checkpoint(ckpt_path, n_timesteps, init_beta, fin_beta, device):
    ddpm = DDPM(
        n_timesteps=n_timesteps,
        init_beta=init_beta,
        fin_beta=fin_beta,
    ).to(device)
    state_dict = torch.load(str(ckpt_path), map_location=device)
    ddpm.load_state_dict(state_dict)
    return ddpm


def _get_frame(x):
    b, _, _, _ = x.shape
    grid = image_to_grid(x, n_cols=int(b ** 0.5))
    frame = np.array(grid)
    return frame


@torch.no_grad()
def perform_progressive_generation(
    ddpm,
    batch_size,
    n_channels,
    img_size,
    device,
    save_path,
    x_t=None,
    start_timestep=999,
    image_only=True,
):
    frame_indices = np.linspace(start=0, stop=ddpm.n_timesteps, num=100, dtype="uint16")

    ddpm.eval()
    gif_path = Path(save_path).with_suffix(".gif")
    with imageio.get_writer(gif_path, mode="I") as writer:
        if x_t is not None:
            x = x_t.clone()
        else:
            x = sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
        for timestep in tqdm(range(start_timestep, -1, -1)):
            t = torch.full(
                size=(batch_size,), fill_value=timestep, dtype=torch.long, device=device,
            )
            eps_theta = ddpm.predict_noise(x, t=t)

            beta_t = index_using_timestep(ddpm.beta.to(device), t=t)
            alpha_t = index_using_timestep(ddpm.alpha.to(device), t=t)
            alpha_bar_t = index_using_timestep(ddpm.alpha_bar.to(device), t=t)

            x = (1 / (alpha_t ** 0.5)) * (x - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * eps_theta)

            if timestep > 0:
                eps = sample_noise(
                    batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device,
                )
                x += (beta_t ** 0.5) * eps

            if (not image_only) and (timestep in frame_indices):
                frame = _get_frame(x)
                writer.append_data(frame)

            if timestep == 0:
                if x_t is not None:
                    n_cols = x.shape[0]
                else:
                    n_cols = int(batch_size ** 0.5)
                grid = image_to_grid(x, n_cols=n_cols)
                save_image(grid, path=save_path)


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
    perform_progressive_generation(
        ddpm=ddpm,
        img_size=CONFIG["IMG_SIZE"],
        n_channels=CONFIG["N_CHANNELS"],
        batch_size=CONFIG["BATCH_SIZE"],
        save_path=CONFIG["SAVE_PATH"],
        device=CONFIG["DEVICE"],
    )
