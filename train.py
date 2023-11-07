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
    get_noise,
    sample_timestep,
    get_elapsed_time,
    modify_state_dict,
)
from celeba import CelebADataset
from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=32)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--resume_from", type=str, required=False)
    parser.add_argument("--n_timesteps", type=int, required=False, default=1000)
    parser.add_argument("--torch_compile", action="store_true", required=False)

    args = parser.parse_args()
    return args


def save_checkpoint(epoch, ddpm, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "epoch": epoch,
        "n_timesteps": ddpm.n_timesteps,
        "initial_beta": ddpm.init_beta,
        "final_beta": ddpm.fin_beta,
        "model": modify_state_dict(ddpm.state_dict()),
    }
    torch.save(state_dict, str(save_path))


def get_ddpm_from_checkpoint(ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    ddpm = DDPM(
        n_timesteps=state_dict["n_timesteps"],
        init_beta=state_dict["initial_beta"],
        fin_beta=state_dict["final_beta"],
    ).to(device)
    ddpm.load_state_dict(state_dict["model"])

    epoch = state_dict["epoch"]
    return ddpm, epoch


if __name__ == "__main__":
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    args = get_args()

    DEVICE = get_device()

    if args.resume_from is not None:
        ddpm, init_epoch = get_ddpm_from_checkpoint(ckpt_path=args.resume_from, device=DEVICE)
    else:
        ddpm = DDPM(
            n_timesteps=args.n_timesteps,
            init_beta=CONFIG["INIT_BETA"],
            fin_beta=CONFIG["FIN_BETA"],
        ).to(DEVICE)
        init_epoch = 0
    if args.torch_compile:
        ddpm = torch.compile(ddpm)

    crit = nn.MSELoss()

    train_ds = CelebADataset(
        data_dir=args.data_dir, img_size=args.img_size, mean=CONFIG["MEAN"], std=CONFIG["STD"],
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpus,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Number of train data samples: {len(train_ds):,}")

    optim = Adam(ddpm.parameters(), lr=CONFIG["LR"])

    scaler = GradScaler() if DEVICE.type == "cuda" else None

    best_loss = math.inf
    n_cols = int(args.batch_size ** 0.5)
    for epoch in range(init_epoch + 1, args.n_epochs + 1):
        accum_loss = 0
        start_time = time()
        for x0 in train_dl: # "$x_{0} \sim q(x_{0})$"
            # break
            x0 = x0.to(DEVICE)
            # image_to_grid(x0, n_cols=n_cols).show()

            t = sample_timestep(
                n_timesteps=ddpm.n_timesteps, batch_size=args.batch_size, device=DEVICE,
            ) # "$t \sim Uniform({1, \ldots, T})$"
            eps = get_noise(
                batch_size=args.batch_size,
                n_channels=CONFIG["N_CHANNELS"],
                # img_size=CONFIG["IMG_SIZE"],
                img_size=args.img_size,
                device=DEVICE,
            ) # "$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
            ):
                noisy_image = ddpm(x0, t=t, eps=eps)
                # image_to_grid(noisy_image, n_cols=n_cols).show()
                pred_eps = ddpm.estimate_noise(x=noisy_image, t=t)
                # image_to_grid(pred_eps, n_cols=n_cols).show()
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
        msg += f"[ Loss: {accum_loss:.7f} ]"

        if accum_loss < best_loss:
            save_checkpoint(
                epoch=epoch,
                ddpm=ddpm,
                save_path=Path(__file__).resolve().parent/f"checkpoints/ddpm_epoch_{epoch}.pth",
            )
            msg += f" (Saved checkpoint.)"
            best_loss = accum_loss
        print(msg)
