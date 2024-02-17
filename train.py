# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://docs.wandb.ai/guides/runs/resuming

import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler
import argparse
from pathlib import Path
import math
from time import time
from tqdm import tqdm
import wandb

from utils import (
    set_seed,
    get_elapsed_time,
    modify_state_dict,
    get_device,
    print_n_prams,
    image_to_grid,
    save_image,
)
from celeba import get_dls
from model2 import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--channels", type=int, default=32, required=False)
    parser.add_argument("--n_blocks", type=int, default=2, required=False)

    parser.add_argument("--run_id", type=str, required=False)
    parser.add_argument("--seed", type=int, default=223, required=False)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


class Trainer(object):
    def __init__(self, run_id, train_dl, val_dl, save_dir, device):
        self.run_id = run_id
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.save_dir = Path(save_dir)
        self.device = device
        
        self.ckpt_path = self.save_dir/"checkpoint.tar"

        self.init_wandb()

    def init_wandb(self):
        if self.run_id is not None:
            self.run = wandb.init(project="DDPM", resume="must", id=self.run_id)
        else:
            # run_id = wandb.util.generate_id()
            self.run = wandb.init(project="DDPM")
        # wandb.config.update({"IMG_SIZE": img_size})
        # print(wandb.config)

    def train(self, n_epochs, model, optim, scaler):
        if self.run.resumed:
            ckpt = torch.load(
                str(wandb.restore(self.ckpt_path)), map_location=self.device,
            )
            model.load_state_dict(modify_state_dict(ckpt["model"]))
            optim.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            init_epoch = ckpt["epoch"]
            min_val_loss = ckpt["min_val_loss"]
            print(f"Resuming from epoch {init_epoch + 1}...")
        else:
            init_epoch = 0
            min_val_loss = math.inf
        model = torch.compile(model)

        for epoch in range(init_epoch + 1, n_epochs + 1):
            cum_train_loss = 0
            start_time = time()
            for ori_image in tqdm(self.train_dl, leave=False): # "$x_{0} \sim q(x_{0})$"
                loss = self.train_single_step(
                    ori_image=ori_image, model=model, optim=optim, scaler=scaler,
                )
                cum_train_loss += loss.item()
            train_loss = cum_train_loss / len(self.train_dl)

            log = f"[ {get_elapsed_time(start_time)} ]"
            log += f"[ {epoch}/{n_epochs} ]"
            log += f"[ Train loss: {train_loss:.4f} ]"
            log += f"[ Val loss: {val_loss:.4f} | Best: {min_val_loss:.4f} ]"
            print(log)
            wandb.log({"Val loss": val_loss, "Min val loss": min_val_loss}, step=epoch)

            self.save_ckpt(
                epoch=epoch,
                model=model,
                optim=optim,
                scaler=scaler,
                min_val_loss=min_val_loss,
            )

            val_loss = self.validate(model)
            if val_loss < min_val_loss:
                model_params_path = self.save_dir/f"epoch={epoch}-val_loss={val_loss:.4f}.pth"
                self.save_model_params(model=model, save_path=model_params_path)
                min_val_loss = val_loss

            gen_image = model.sample(batch_size=16)
            gen_grid = image_to_grid(gen_image, n_cols=4)
            sample_path = self.save_dir/f"sample-epoch={epoch}.jpg"
            save_image(gen_grid, save_path=sample_path)
            wandb.log({"Samples": wandb.Image(sample_path)}, step=epoch)

    def train_single_step(self, ori_image, model, optim, scaler):
        ori_image = ori_image.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
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

    @torch.no_grad()
    def validate(self, model):
        model.eval()

        cum_val_loss = 0
        for ori_image in self.val_dl:
            ori_image = ori_image.to(self.device)
            loss = model.get_loss(ori_image)
            cum_val_loss += loss.item()
        val_loss = cum_val_loss / len(self.val_dl)

        model.train()
        return val_loss

    @staticmethod
    def save_model_params(model, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(modify_state_dict(model.state_dict()), str(save_path))

    def save_ckpt(self, epoch, model, optim, scaler, min_val_loss):
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model": modify_state_dict(model.state_dict()),
            "scaler": scaler.state_dict(),
            "optimizer": optim.state_dict(),
            "min_val_loss": min_val_loss,
        }
        torch.save(ckpt, str(self.ckpt_path))
        wandb.save(str(self.ckpt_path), base_path=Path(self.ckpt_path).parent)


def main():
    DEVICE = get_device()
    args = get_args()
    set_seed(args.SEED)

    train_dl, val_dl, _ = get_dls(
        data_dir=args.DATA_DIR,
        img_size=args.IMG_SIZE,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
        seed=args.SEED,
    )
    trainer = Trainer(
        run_id=args.RUN_ID,
        train_dl=train_dl,
        val_dl=val_dl,
        save_dir=args.SAVE_DIR,
        device=DEVICE,
    )

    model = DDPM(
        device=DEVICE,
        channels=args.CHANNELS,
        channel_mults=[2, 2, 2, 2],
        attns=[True, True, True, True],
        n_blocks=args.N_BLOCKS,
    )
    print_n_prams(model)
    optim = Adam(model.parameters(), lr=args.LR)
    scaler = GradScaler() if DEVICE.type == "cuda" else None

    trainer.train(n_epochs=args.N_EPOCHS, model=model, optim=optim, scaler=scaler)


if __name__ == "__main__":
    main()
