# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=e3eb5811-c10b-4dae-a58d-9583c42e7f57

import torch
from torch.optim import AdamW
import gc
import argparse
from pathlib import Path
import math
from time import time
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from diffusers import UNet2DModel


from utils import (
    set_seed,
    get_device,
    get_grad_scaler,
    get_elapsed_time,
    modify_state_dict,
    print_n_params,
    image_to_grid,
    save_image,
)
from data import get_train_and_val_dls
from unet import UNet
from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--n_warmup_steps", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=True)

    parser.add_argument("--seed", type=int, default=223, required=False)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


class Trainer(object):
    def __init__(self, train_dl, val_dl, save_dir, device):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.save_dir = Path(save_dir)
        self.device = device

        self.ckpt_path = self.save_dir/"ckpt.pth"

    def train_for_one_epoch(self, epoch, model, optim, scaler):
        train_loss = 0
        pbar = tqdm(self.train_dl, leave=False)
        for step_idx, ori_image in enumerate(pbar): # "$x_{0} \sim q(x_{0})$"
            pbar.set_description("Training...")

            ori_image = ori_image.to(self.device)
            loss = model.get_loss(ori_image)
            train_loss += (loss.item() / len(self.train_dl))

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            self.scheduler.step((epoch - 1) * len(self.train_dl) + step_idx)
        return train_loss

    @torch.inference_mode()
    def validate(self, model):
        val_loss = 0
        pbar = tqdm(self.val_dl, leave=False)
        for ori_image in pbar:
            pbar.set_description("Validating...")

            ori_image = ori_image.to(self.device)
            loss = model.get_loss(ori_image.detach())
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

    @torch.inference_mode()
    def test_sampling(self, epoch, model, batch_size):
        gen_image = model.sample(batch_size=batch_size)
        gen_grid = image_to_grid(gen_image, n_cols=int(batch_size ** 0.5))
        sample_path = self.save_dir/f"sample-epoch={epoch}.jpg"
        save_image(gen_grid, save_path=sample_path)

    def train(self, n_epochs, model, optim, scaler, n_warmup_steps):
        for param in model.parameters():
            param.register_hook(lambda grad: torch.clip(grad, -1, 1))

        model = torch.compile(model)

        self.scheduler = CosineLRScheduler(
            optimizer=optim,
            t_initial=n_epochs * len(self.train_dl),
            warmup_t=n_warmup_steps,
            warmup_lr_init=optim.param_groups[0]["lr"] * 0.1,
            warmup_prefix=True,
            t_in_epochs=False,
        )

        init_epoch = 0
        min_val_loss = math.inf
        for epoch in range(init_epoch + 1, n_epochs + 1):
            start_time = time()
            train_loss = self.train_for_one_epoch(
                epoch=epoch, model=model, optim=optim, scaler=scaler,
            )
            val_loss = self.validate(model)
            if val_loss < min_val_loss:
                model_params_path = str(self.save_dir/f"epoch={epoch}-val_loss={val_loss:.4f}.pth")
                self.save_model_params(model=model, save_path=model_params_path)
                min_val_loss = val_loss

            self.save_ckpt(
                epoch=epoch,
                model=model,
                optim=optim,
                min_val_loss=min_val_loss,
                scaler=scaler,
            )

            self.test_sampling(epoch=epoch, model=model, batch_size=4)

            log = f"[ {get_elapsed_time(start_time)} ]"
            log += f"[ {epoch}/{n_epochs} ]"
            log += f"[ Train loss: {train_loss:.4f} ]"
            log += f"[ Val loss: {val_loss:.4f} | Best: {min_val_loss:.4f} ]"
            print(log)


def main():
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()
    args = get_args()
    set_seed(args.SEED)
    print(f"[ DEVICE: {DEVICE} ]")

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_dl, val_dl = get_train_and_val_dls(
        data_dir=args.DATA_DIR,
        img_size=args.IMG_SIZE,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
    )
    trainer = Trainer(
        train_dl=train_dl,
        val_dl=val_dl,
        save_dir=args.SAVE_DIR,
        device=DEVICE,
    )

    net = UNet()
    model = DDPM(img_size=args.IMG_SIZE, model=net, device=DEVICE)
    print_n_params(model)
    optim = AdamW(model.parameters(), lr=args.LR)
    scaler = get_grad_scaler(device=DEVICE)

    trainer.train(
        n_epochs=args.N_EPOCHS,
        model=model,
        optim=optim,
        scaler=scaler,
        n_warmup_steps=args.N_WARMUP_STEPS,
    )


if __name__ == "__main__":
    main()
