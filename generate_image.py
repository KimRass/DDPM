# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from tqdm import tqdm

from utils import load_config, get_device, get_noise, gather, image_to_grid
from ddpm import DDPMForCelebA


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    # parser.add_argument("--n_timesteps", type=int, required=True)
    parser.add_argument("--gif_path", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate_image(ddpm, img_size, n_channels, batch_size, n_frames, gif_path, device):
    # batch_size=4
    # device=DEVICE
    # n_frames=100
    # n_channels=1
    # gif_path="/Users/jongbeomkim/Downloads/test.gif"
    # img_size = 28
    frame_indices = np.linspace(start=0, stop=ddpm.n_timesteps, num=n_frames, dtype="uint8")

    ddpm = ddpm.to(device)
    ddpm.eval()
    with imageio.get_writer(gif_path, mode="I") as writer:
        # Sample pure noise from a Gaussian distribution.
        # "$x_{T} \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$"
        x = get_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
        for idx, t in enumerate(tqdm(range(ddpm.n_timesteps - 1, -1, -1))):
            # Estimate noise to be removed.
            batched_t = torch.full(
                size=(batch_size, 1), fill_value=t, dtype=torch.long, device=device,
            )
            eps_theta = ddpm.estimate_noise(x, t=batched_t) # "$z_{\theta}(x_{t}, t)$"

            beta_t = gather(ddpm.beta, t=t)
            alpha_t = gather(ddpm.alpha, t=t)
            alpha_bar_t = gather(ddpm.alpha_bar, t=t)

            # Partially denoise image.
            # "$$\mu_{\theta}(x_{t}, t) =
            # \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}\epsilon_{\theta}(x_{t}, t)\Big)$$"
            x = (1 / (alpha_t ** 0.5)) * (x - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * eps_theta)

            if t > 0:
                eps = get_noise(
                    batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device,
                ) # "$z \sim \mathcal{L}(\mathbf{0}, \mathbf{I})$"
                x += (beta_t ** 0.5) * eps

            if idx in frame_indices or t == 0:
                grid = image_to_grid(x, n_cols=int(args.batch_size ** 0.5))
                frame = np.array(grid)
                writer.append_data(frame)

            # gif 파일에서 마지막 프레임을 오랫동안 보여줍니다.
            if idx == ddpm.n_timesteps - 1:
                for _ in range(ddpm.n_timesteps // 3):
                    writer.append_data(frame)
    return x


# def get_ddpm_from_checkpoint(ckpt_path, device):
#     state_dict = torch.load(ckpt_path, map_location=device)
#     ddpm = DDPMForCelebA(
#         n_timesteps=state_dict["n_timesteps"],
#         time_dim=state_dict["time_dimension"],
#         init_beta=state_dict["initial_beta"],
#         fin_beta=state_dict["final_beta"],
#     ).to(device)
#     ddpm.load_state_dict(state_dict["model"])
#     return ddpm


if __name__ == "__main__":
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    args = get_args()

    DEVICE = get_device()

    ddpm = get_ddpm_from_checkpoint(ckpt_path=args.ckpt_path, device=DEVICE)
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
