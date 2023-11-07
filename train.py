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
import wandb

from utils import (
    set_seed,
    load_config,
    get_device,
    get_noise,
    sample_timestep,
    get_elapsed_time,
    modify_state_dict,
)
from celeba import CelebADataset
from ddpm import DDPM


def _get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=32)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--run_id", type=str, required=False)
    parser.add_argument("--torch_compile", action="store_true", required=False)

    args = parser.parse_args()
    return args


def get_config():
    config = load_config(Path(__file__).parent/"config.yaml")
    args = _get_args()
    config["RUN_ID"] = args.run_id
    config["DATA_DIR"] = args.data_dir
    config["N_EPOCHS"] = args.n_epochs
    config["IMG_SIZE"] = args.img_size
    config["BATCH_SIZE"] = args.batch_size
    config["LR"] = args.lr
    config["N_CPUS"] = args.n_cpus
    config["TORCH_COMPILE"] = args.torch_compile

    config["PARENT_DIR"] = Path(__file__).resolve().parent
    config["CKPTS_DIR"] = config["PARENT_DIR"]/"checkpoints"
    config["CKPT_PATH"] = config["CKPTS_DIR"]/"checkpoint.tar"

    config["DEVICE"] = get_device()
    return config


def init_wandb(run_id, img_size):
    wandb.init(project="DDPM", resume=run_id)
    wandb.config.update(
        {
            "IMG_SIZE": img_size,
        },
        allow_val_change=True,
    )
    print(wandb.config)


def train_single_step(x0, ddpm, optim, scaler, crit, config):
    x0 = x0.to(config["DEVICE"])

    t = sample_timestep(
        n_timesteps=config["N_TIMESTEPS"], batch_size=config["BATCH_SIZE"], device=config["DEVICE"],
    ) # "$t \sim Uniform({1, \ldots, T})$"
    eps = get_noise(
        batch_size=config["BATCH_SIZE"],
        n_channels=config["N_CHANNELS"],
        img_size=config["IMG_SIZE"],
        device=config["DEVICE"],
    ) # "$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"

    with torch.autocast(
        device_type=config["DEVICE"].type,
        dtype=torch.float16 if config["DEVICE"].type == "cuda" else torch.bfloat16,
    ):
        noisy_image = ddpm(x0, t=t, eps=eps)
        pred_eps = ddpm.predict_noise(x=noisy_image, t=t)
        loss = crit(pred_eps, eps)

    optim.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()
    return loss


def save_checkpoint(epoch, ddpm, scaler, optim, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "epoch": epoch,
        "ddpm": modify_state_dict(ddpm.state_dict()),
        "scaler": scaler.state_dict(),
        "optimizer": optim.state_dict(),
    }
    torch.save(state_dict, str(save_path))
    wandb.save(str(save_path), base_path=Path(save_path).parent)


def get_tain_dl(config):
    train_ds = CelebADataset(
        data_dir=config["DATA_DIR"], img_size=config["IMG_SIZE"], mean=config["MEAN"], std=config["STD"],
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["N_CPUS"],
        pin_memory=True,
        drop_last=True,
    )
    print(f"Number of train data samples: {len(train_ds):,}")
    return train_dl


if __name__ == "__main__":
    CONFIG = get_config()

    set_seed(CONFIG["SEED"])

    init_wandb(run_id=CONFIG["RUN_ID"], img_size=CONFIG["IMG_SIZE"])

    ddpm = DDPM(
        n_timesteps=CONFIG["N_TIMESTEPS"],
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
    ).to(CONFIG["DEVICE"])
    if CONFIG["TORCH_COMPILE"]:
        ddpm = torch.compile(ddpm)
    optim = Adam(ddpm.parameters(), lr=CONFIG["LR"])
    scaler = GradScaler() if CONFIG["DEVICE"].type == "cuda" else None
    crit = nn.MSELoss()

    if wandb.run.resumed:
        state_dict = torch.load(str(CONFIG["CKPT_PATH"]), map_location=CONFIG["DEVICE"])
        ddpm.load_state_dict(state_dict["ddpm"])
        optim.load_state_dict(state_dict["optimizer"])
        scaler.load_state_dict(state_dict["scaler"])
        init_epoch = state_dict["epoch"]
        print(f"Resuming from epoch {init_epoch + 1}...")
    else:
        init_epoch = 0

    train_dl = get_tain_dl(CONFIG)

    best_loss = math.inf
    n_cols = int(CONFIG["BATCH_SIZE"] ** 0.5)
    for epoch in range(init_epoch + 1, CONFIG["N_EPOCHS"] + 1):
        accum_loss = 0
        start_time = time()
        for x0 in train_dl: # "$x_{0} \sim q(x_{0})$"
            loss = train_single_step(
                x0=x0, ddpm=ddpm, optim=optim, scaler=scaler, crit=crit, config=CONFIG,
            )
            accum_loss += loss.item()
        accum_loss /= len(train_dl)
        accum_loss /= CONFIG["BATCH_SIZE"]

        msg = f"""[ {epoch}/{CONFIG["N_EPOCHS"]} ]"""
        msg += f"[ {get_elapsed_time(start_time)} ]"
        msg += f"[ Loss: {accum_loss:.7f} ]"

        wandb.log(
            {
                "Loss": accum_loss,
            },
            step=epoch,
        )

        if accum_loss < best_loss:
            filename = f"""{CONFIG["IMG_SIZE"]}×{CONFIG["IMG_SIZE"]}_epoch_{epoch}.pth"""
            save_checkpoint(
                epoch=epoch,
                ddpm=ddpm,
                scaler=scaler,
                optim=optim,
                save_path=CONFIG["CKPTS_DIR"]/filename,
            )
            msg += f" (Saved checkpoint.)"
            best_loss = accum_loss
        print(msg)
