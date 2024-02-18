# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import gc
import argparse
from pathlib import Path
import math
from time import time
from tqdm import tqdm
import contextlib

from utils import (
    set_seed,
    get_device,
    get_grad_scaler,
    get_elapsed_time,
    modify_state_dict,
    print_n_prams,
    image_to_grid,
    save_image,
)
from data import get_dls
from model3 import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
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

    def train_single_step(self, ori_image, model, optim, scaler):
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16,
        ) if self.device.type == "cuda" else contextlib.nullcontext():
            # loss = model.get_loss(ori_image)
            diffusion_step = torch.randint(
                0, 1000, size=(self.train_dl.batch_size,), device=self.device,
            )
            random_noise = torch.randn(
                size=(self.train_dl.batch_size, 3, 32, 32),
                device=self.device,
            )
            noisy_image = model.get_noisy_image(
                ori_image=ori_image, diffusion_step=diffusion_step, random_noise=random_noise,
            )
            pred_noise = self.net(noisy_image=noisy_image, diffusion_step=diffusion_step)
            loss =  F.mse_loss(pred_noise, random_noise, reduction="mean")

        optim.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        return loss

    @torch.inference_mode()
    def validate(self, model):
        val_loss = 0
        for ori_image in self.val_dl:
            ori_image = ori_image.to(self.device)
            loss = model.get_loss(ori_image)
            val_loss += (loss.item() / len(self.val_dl))
        return val_loss

    @staticmethod
    def save_model_params(model, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(modify_state_dict(model.state_dict()), str(save_path))

    def save_ckpt(self, epoch, model, optim, min_val_loss, scaler):
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model": modify_state_dict(model.state_dict()),
            "optimizer": optim.state_dict(),
            "min_val_loss": min_val_loss,
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, str(self.ckpt_path))

    def test_sampling(self, epoch, model, batch_size):
        gen_image = model.sample(batch_size=batch_size)
        gen_grid = image_to_grid(gen_image, n_cols=int(batch_size ** 0.5))
        sample_path = self.save_dir/f"sample-epoch={epoch}.jpg"
        save_image(gen_grid, save_path=sample_path)

    def train(self, n_epochs, model, optim, scaler):
        init_epoch = 0
        min_val_loss = math.inf
        model = torch.compile(model)

        for epoch in range(init_epoch + 1, n_epochs + 1):
            train_loss = 0
            start_time = time()
            for ori_image in tqdm(self.train_dl, leave=False): # "$x_{0} \sim q(x_{0})$"
                ori_image = ori_image.to(self.device)
                loss = self.train_single_step(
                    ori_image=ori_image, model=model, optim=optim, scaler=scaler,
                )
                train_loss += (loss.item() / len(self.train_dl))

            # val_loss = self.validate(model)
            val_loss = 0
            if val_loss < min_val_loss:
                model_params_path = str(self.save_dir/f"epoch={epoch}-val_loss={val_loss:.4f}.pth")
                self.save_model_params(model=model, save_path=model_params_path)
                min_val_loss = val_loss

            log = f"[ {get_elapsed_time(start_time)} ]"
            log += f"[ {epoch}/{n_epochs} ]"
            log += f"[ Train loss: {train_loss:.4f} ]"
            log += f"[ Val loss: {val_loss:.4f} | Best: {min_val_loss:.4f} ]"
            print(log)

            self.save_ckpt(
                epoch=epoch,
                model=model,
                optim=optim,
                min_val_loss=min_val_loss,
                scaler=scaler,
            )

            self.test_sampling(epoch=epoch, model=model, batch_size=4)


def main():
    DEVICE = get_device()
    args = get_args()
    set_seed(args.SEED)
    print(f"[ DEVICE: {DEVICE} ]")

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_dl, val_dl, _ = get_dls(
        data_dir=args.DATA_DIR,
        img_size=args.IMG_SIZE,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
    )
    trainer = Trainer(
        run_id=args.RUN_ID,
        train_dl=train_dl,
        val_dl=val_dl,
        save_dir=args.SAVE_DIR,
        device=DEVICE,
    )

    model = DDPM(
        img_size=args.IMG_SIZE,
        device=DEVICE,
        channels=(32, 64, 128, 256, 512),
        attns=(False, True, False, False),
        n_blocks=args.N_BLOCKS,
    )
    print_n_prams(model)
    optim = AdamW(model.net.parameters(), lr=args.LR)
    scaler = get_grad_scaler(device=DEVICE)

    trainer.train(n_epochs=args.N_EPOCHS, model=model, optim=optim, scaler=scaler)


if __name__ == "__main__":
    main()
