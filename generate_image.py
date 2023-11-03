# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

import torch
import numpy as np
import einops
import imageio
from pathlib import Path
import argparse

from utils import load_config, get_device, get_noise, gather, image_to_grid
from model import UNetForDDPM
from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_timesteps", type=int, required=True)
    parser.add_argument("--gif_path", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def _denorm(x):
    copied = x.clone()
    copied -= copied.amin(dim=(1, 2, 3))[:, None, None, None]
    copied /= copied.amax(dim=(1, 2, 3))[:, None, None, None]
    copied *= 255 # $[0, 255]$
    return copied


def generate_image(ddpm, img_size, n_channels, batch_size, n_frames, gif_path, device):
    # batch_size=4
    # device=DEVICE
    # n_frames=100
    # n_channels=1
    # gif_path="/Users/jongbeomkim/Downloads/test.gif"
    # img_size = 28
    frame_indices = np.linspace(start=0, stop=ddpm.n_timesteps, num=n_frames, dtype="uint8")
    frames = list()

    ddpm = ddpm.to(device)
    ddpm.eval()
    with torch.no_grad():
        # Sample pure noise from a Gaussian distribution.
        # "$x_{T} \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$"
        x = get_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)

        for idx, t in enumerate(range(ddpm.n_timesteps - 1, -1, -1)):
            # Estimate noise to be removed.
            time_tensor = torch.full(
                size=(batch_size, 1), fill_value=t, dtype=torch.long, device=device,
            )
            eps_theta = ddpm.estimate_noise(x, t=time_tensor) # "$z_{\theta}(x_{t}, t)$"

            beta_t = gather(ddpm.beta, t=t)
            alpha_t = gather(ddpm.alpha, t=t)
            alpha_bar_t = gather(ddpm.alpha_bar, t=t)

            # Partially denoise image.
            x = (1 / (alpha_t ** 0.5)) * (x - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * eps_theta)
            # "$$\mu_{\theta}(x_{t}, t) =
            # \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}\epsilon_{\theta}(x_{t}, t)\Big)$$"

            if t > 0:
                random_noise = get_noise(
                    batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device,
                ) # "$z \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$"
                x += (beta_t ** 0.5) * random_noise

            # Add frames to GIF file.
            if idx in frame_indices or t == 0:
                x = _denorm(x)
                frame = einops.rearrange(
                    x, pattern="(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(batch_size ** 0.5),
                )
                frame = frame.cpu().numpy().astype("uint8")

                frames.append(frame)

    with imageio.get_writer(gif_path, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)

            if idx == len(frames) - 1:
                for _ in range(n_frames // 3):
                    writer.append_data(frames[-1])
    return x


if __name__ == "__main__":
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    args = get_args()

    DEVICE = get_device()

    model = UNetForDDPM(
        n_channels=CONFIG["N_CHANNELS"],
        n_timesteps=args.n_timesteps,
        time_embed_dim=CONFIG["TIME_EMBED_DIM"],
    )
    ddpm = DDPM(
        model=model,
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
        n_timesteps=args.n_timesteps,
        device=DEVICE,
    ).to(DEVICE)
    state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    ddpm.load_state_dict(state_dict)
    generated = generate_image(
        ddpm=ddpm,
        img_size=CONFIG["IMG_SIZE"],
        n_channels=CONFIG["N_CHANNELS"],
        batch_size=args.batch_size,
        n_frames=100,
        gif_path=args.gif_path,
        device=DEVICE,
    )
    grid = image_to_grid(generated, n_cols=int(args.batch_size ** 0.5))
    grid.show()
