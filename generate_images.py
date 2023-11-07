# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from tqdm import tqdm

from utils import load_config, get_device, get_noise, extract, image_to_grid, save_image
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


def get_ddpm_from_checkpoint(ckpt_path, device):
    state_dict = torch.load(str(ckpt_path), map_location=device)
    ddpm = DDPM(
        n_timesteps=state_dict["n_timesteps"],
        init_beta=state_dict["initial_beta"],
        fin_beta=state_dict["final_beta"],
    ).to(device)
    ddpm.load_state_dict(state_dict["model"])

    epoch = state_dict["epoch"]
    return ddpm, epoch


def _get_frame(x):
    b, _, _, _ = x.shape
    grid = image_to_grid(x, n_cols=int(b ** 0.5))
    frame = np.array(grid)
    return frame


@torch.no_grad()
def generate_images(
    ddpm, batch_size, n_channels, img_size, save_path, device, n_frames=100, image_only=False,
):
    frame_indices = np.linspace(start=0, stop=ddpm.n_timesteps, num=n_frames, dtype="uint16")

    ddpm.eval()
    gif_path = Path(save_path).with_suffix(".gif")
    with imageio.get_writer(gif_path, mode="I") as writer:
        # Sample pure noise from a Gaussian distribution.
        # "$x_{T} \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$"
        x = get_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
        for t in tqdm(range(ddpm.n_timesteps - 1, -1, -1)):
            batched_t = torch.full(
                size=(batch_size,), fill_value=t, dtype=torch.long, device=device,
            )
            eps_theta = ddpm.estimate_noise(x, t=batched_t) # "$z_{\theta}(x_{t}, t)$"

            beta_t = extract(ddpm.beta.to(device), t=t, device=device)
            alpha_t = extract(ddpm.alpha, t=t, device=device)
            alpha_bar_t = extract(ddpm.alpha_bar, t=t, device=device)

            # Partially denoise image.
            # "$$\mu_{\theta}(x_{t}, t) =
            # \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}\epsilon_{\theta}(x_{t}, t)\Big)$$"
            x = (1 / (alpha_t ** 0.5)) * (x - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * eps_theta)

            if t > 0:
                eps = get_noise(
                    batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device,
                ) # "$z \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$"
                x += (beta_t ** 0.5) * eps

            if (not image_only) and (t in frame_indices):
                frame = _get_frame(x)
                writer.append_data(frame)

            if t == 0:
                grid = image_to_grid(x, n_cols=int(args.batch_size ** 0.5))
                save_image(grid, path=save_path)


if __name__ == "__main__":
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    args = get_args()

    DEVICE = get_device()

    ddpm, _ = get_ddpm_from_checkpoint(ckpt_path=args.ckpt_path, device=DEVICE)
    generate_images(
        ddpm=ddpm,
        img_size=args.img_size,
        n_channels=CONFIG["N_CHANNELS"],
        batch_size=args.batch_size,
        save_path=args.save_path,
        device=DEVICE,
    )