# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import argparse
from pathlib import Path
import math
from time import time
from tqdm import tqdm
# import wandb

from utils import set_seed, get_elapsed_time, modify_state_dict, get_device
from celeba import CelebADataset
from model import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--run_id", type=str, required=False)

    parser.add_argument("--seed", type=int, default=223, required=False)
    parser.add_argument("--img_size", type=int, default=64, required=False)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def init_wandb(run_id, img_size):
    if run_id is None:
        run_id = wandb.util.generate_id()
    wandb.init(project="DDPM", resume="allow", id=run_id)
    wandb.config.update({"IMG_SIZE": img_size})
    # print(wandb.config)


def train_single_step(ori_image, model, optim, scaler, device):
    ori_image = ori_image.to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
    ):
        loss = model.get_loss(ori_image)

    optim.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()
    return loss


def save_wandb_checkpoint(epoch, model, scaler, optim, loss, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "epoch": epoch,
        "model": modify_state_dict(model.state_dict()),
        "scaler": scaler.state_dict(),
        "optimizer": optim.state_dict(),
        "loss": loss,
    }
    torch.save(state_dict, str(save_path))
    # wandb.save(str(save_path), base_path=Path(save_path).parent)


def save_model(model, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(modify_state_dict(model.state_dict()), str(save_path))


def get_tain_dl(data_dir, img_size, batch_size, n_cpus):
    train_ds = CelebADataset(data_dir=data_dir, img_size=img_size)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    print(f"Number of train data samples: {len(train_ds):,}")
    return train_dl


def main():
    DEVICE = get_device()
    args = get_args()
    set_seed(args.SEED)
    # init_wandb(run_id=args.RUN_ID, img_size=args.IMG_SIZE)
    # WANDB_CKPT_PATH = Path(args.SAVE_DIR)/"checkpoint.h5"

    train_dl = get_tain_dl(
        data_dir=args.DATA_DIR,
        img_size=args.IMG_SIZE,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
    )

    model = DDPM(device=DEVICE)
    model = torch.compile(model)
    optim = Adam(model.parameters(), lr=args.LR)
    scaler = GradScaler() if DEVICE.type == "cuda" else None

    # if wandb.run.resumed:
    #     state_dict = torch.load(
    #         str(wandb.restore(WANDB_CKPT_PATH)), map_location=DEVICE,
    #     )
    #     model.load_state_dict(state_dict["model"])
    #     optim.load_state_dict(state_dict["optimizer"])
    #     scaler.load_state_dict(state_dict["scaler"])
    #     init_epoch = state_dict["epoch"]
    #     min_loss = state_dict["loss"]
    #     print(f"Resuming from epoch {init_epoch + 1}...")
    # else:
    init_epoch = 0
    min_loss = math.inf

    for epoch in range(init_epoch + 1, args.N_EPOCHS + 1):
        cum_loss = 0
        start_time = time()
        for ori_image in tqdm(train_dl, leave=False): # "$x_{0} \sim q(x_{0})$"
            loss = train_single_step(
                ori_image=ori_image, model=model, optim=optim, scaler=scaler, device=DEVICE,
            )
            cum_loss += loss.item()
        cur_loss = cum_loss / len(train_dl)

        if cur_loss < min_loss:
            min_loss = cur_loss

        log = f"[ {get_elapsed_time(start_time)} ]"
        log += f"[ {epoch}/{args.N_EPOCHS} ]"
        log += f"[ Min loss: {min_loss:.5f} ]"
        log += f"[ Loss: {cur_loss:.5f} ]"
        print(log)

        # wandb.log({"Min loss": min_loss, "Loss": cur_loss}, step=epoch)

        filename = f"DDPM_{args.IMG_SIZE}×{args.IMG_SIZE}_epoch_{epoch}.pth"
        save_model(model=model, save_path=args.SAVE_DIR/filename)

        # save_wandb_checkpoint(
        #     epoch=epoch,
        #     model=model,
        #     scaler=scaler,
        #     optim=optim,
        #     loss=cur_loss,
        #     save_path=WANDB_CKPT_PATH,
        # )


if __name__ == "__main__":
    main()
