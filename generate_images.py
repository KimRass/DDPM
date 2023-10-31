# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import torch.nn as nn
import numpy as np
import einops
import imageio


def generate_image(
    ddpm, batch_size, frames_per_gif, gif_name, img_size, device, n_channels=1,
):
    # batch_size=16
    # device=DEVICE
    # frames_per_gif=100
    # gif_name="/Users/jongbeomkim/Downloads/sampling.gif"
    # c=1
    # img_size = 28
    frame_indices = np.linspace(0, ddpm.n_timesteps, frames_per_gif).astype(np.uint)
    frames = list()

    ddpm = ddpm.to(device)
    ddpm.eval()
    with torch.no_grad():
        # Starting from random noise
        x = torch.randn(batch_size, n_channels, img_size, img_size).to(device)

        for idx, t in enumerate(range(ddpm.n_timesteps - 1, -1, -1)):
            # Estimate noise to be removed.
            time_tensor = torch.full(
                size=(batch_size, 1), fill_value=t, dtype=torch.long, device=device,
            )
            eta_theta = ddpm.backward(x, time_tensor)

            beta_t = ddpm.betas[t]
            alpha_t = ddpm.alphas[t]
            alpha_bar_t = ddpm.alpha_bars[t]

            # Partially denoise image.
            # x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * eta_theta)
            x = (1 / (alpha_t ** 0.5)) * (x - beta_t / ((1 - alpha_bar_t) ** 0.5) * eta_theta)
            # "$$\mu_{\theta}(x_{t}, t) = \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}\epsilon_{\theta}(x_{t}, t)\Big)$$"

            if t > 0:
                z = torch.randn(batch_size, n_channels, img_size, img_size).to(device)
                sigma_t = beta_t.sqrt()
                # Add some more noise like in Langevin Dynamics fashion.
                x = x + sigma_t * z

            # Add frames to GIF file.
            if idx in frame_indices or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= normalized[i].min()
                    normalized[i] *= (255 / normalized[i].max())

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(
                    normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(batch_size ** 0.5),
                )
                frame = frame.cpu().numpy().astype("uint8")

                # Rendering frame
                frames.append(frame)

    # Save
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
    return x
