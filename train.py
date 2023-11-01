# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pathlib import Path

from utils import (
    load_config,
    get_device,
    image_to_grid,
    get_random_noise_like,
    sample_timestep,
)
from data import get_mnist_dataset
from model import UNetForDDPM
from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--run_id", type=str, required=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/DDPM/config.yaml")
    CONFIG = load_config(Path(__file__).parent/"config.yaml")

    DEVICE = get_device()

    args = get_args()

    model = UNetForDDPM(n_timesteps=CONFIG["N_TIMESTEPS"])
    ddpm = DDPM(
        model=model,
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
        n_timesteps=CONFIG["N_TIMESTEPS"],
        device=DEVICE,
    )
    crit = nn.MSELoss()

    ds = get_mnist_dataset("/Users/jongbeomkim/Documents/datasets")
    train_dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpus,
        pin_memory=True,
        drop_last=True,
    )

    optim = Adam(ddpm.parameters(), lr=0.0001)

    n_epochs = 20
    for epoch in range(1, n_epochs):
        accum_loss = 0
        for x0, _ in tqdm(train_dl):
            # break
            x0 = x0.to(DEVICE)
            # image_to_grid(x0, n_cols=4).show()

            # Pick some noise for each of the images in the batch, a timestep and the respective alpha_bars
            t = sample_timestep(
                n_timesteps=CONFIG["N_TIMESTEPS"], batch_size=args.batch_size, device=DEVICE,
            )
            eps = get_random_noise_like(x0)

            noisy_image = ddpm(x0, t=t, eps=eps)
            # image_to_grid(noisy_image, n_cols=4).show()

            # Getting model estimation of noise based on the images and the time-step
            pred_eps = ddpm.estimate_noise(noisy_image, t=t) # (b, 1, `img_size`, `img_size`)
            image_to_grid(pred_eps, n_cols=4).show()
            loss = crit(pred_eps, eps)

            optim.zero_grad()
            loss.backward()
            optim.step()

            accum_loss += loss.item()

        accum_loss /= len(train_dl)
        accum_loss /= args.batch_size
        msg = f"[ {epoch}/{n_epochs} ]"
        msg += f"[ Loss: {loss: .5f} ]"
        print(msg)
