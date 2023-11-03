# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import argparse
from pathlib import Path
import math
from time import time

from utils import (
    load_config,
    get_device,
    image_to_grid,
    get_noise,
    sample_timestep,
    get_elapsed_time,
)
from celeba import CelebADataset
from model import UNetForDDPM
from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def save_checkpoint(ddpm, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ddpm.state_dict(), str(save_path))


if __name__ == "__main__":
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    args = get_args()

    DEVICE = get_device()

    model = UNetForDDPM(
        n_channels=CONFIG["N_CHANNELS"],
        n_timesteps=CONFIG["N_TIMESTEPS"],
        time_embed_dim=CONFIG["TIME_EMBED_DIM"],
    )
    ddpm = DDPM(
        model=model,
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
        n_timesteps=CONFIG["N_TIMESTEPS"],
        device=DEVICE,
    )
    crit = nn.MSELoss()

    train_ds = CelebADataset(
        data_dir=args.data_dir, img_size=CONFIG["IMG_SIZE"], mean=CONFIG["MEAN"], std=CONFIG["STD"],
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpus,
        pin_memory=True,
        drop_last=True,
    )

    optim = Adam(ddpm.parameters(), lr=0.0001)

    scaler = GradScaler() if DEVICE.type == "cuda" else None

    best_loss = math.inf
    for epoch in range(1, args.n_epochs):
        accum_loss = 0
        start_time = time()
        for x0 in train_dl: # "$x_{0} \sim q(x_{0})$"
            # break
            x0 = x0.to(DEVICE)
            # image_to_grid(x0, n_cols=4).show()

            t = sample_timestep(
                n_timesteps=CONFIG["N_TIMESTEPS"], batch_size=args.batch_size, device=DEVICE,
            ) # "$t \sim Uniform({1, \ldots, T})$"
            eps = get_noise(
                batch_size=args.batch_size,
                n_channels=CONFIG["N_CHANNELS"],
                img_size=CONFIG["IMG_SIZE"],
                device=DEVICE,
            ) # "$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
            ):
                noisy_image = ddpm(x0, t=t, eps=eps)
                # image_to_grid(noisy_image, n_cols=4).show()
                pred_eps = ddpm.estimate_noise(noisy_image, t=t) # (b, 1, `img_size`, `img_size`)
                # image_to_grid(pred_eps, n_cols=4).show()
                loss = crit(pred_eps, eps)

            optim.zero_grad()
            if DEVICE.type == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            accum_loss += loss.item()

        accum_loss /= len(train_dl)
        accum_loss /= args.batch_size

        msg = f"[ {epoch}/{args.n_epochs} ]"
        msg += f"[ {get_elapsed_time(start_time)} ]"
        msg += f"[ Loss: {loss:.4f} ]"
        print(msg)

        if accum_loss < best_loss:
            save_checkpoint(ddpm=ddpm, save_path=Path(__file__).resolve().parent/"checkpoints/ddpm.pth")
            best_loss = accum_loss
