# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=e3eb5811-c10b-4dae-a58d-9583c42e7f57
    # https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py

import torch
from torch.optim import AdamW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import argparse
from pathlib import Path
import math
from time import time
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
import wandb

from utils import (
    set_seed,
    get_grad_scaler,
    get_elapsed_time,
    modify_state_dict,
    print_n_params,
    image_to_grid,
    save_image,
)
from data import get_train_and_val_dls_ddp
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
    parser.add_argument("--rank", type=int, default=0, required=False)
    parser.add_argument("--port", type=int, default=12345, required=False)

    parser.add_argument("--seed", type=int, default=223, required=False)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


class Trainer(object):
    def __init__(self, run, train_dl, val_dl, save_dir, device, rank):
        self.run = run
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.save_dir = Path(save_dir)
        self.device = device
        self.rank = rank

        self.ckpt_path = self.save_dir/"ckpt.pth"

    def train_for_one_epoch(self, epoch, model, optim, scaler):
        train_loss = 0
        if self.rank == 0:
            pbar = tqdm(self.train_dl, leave=False)
        else:
            pbar = self.train_dl
        for step_idx, ori_image in enumerate(pbar): # "$x_{0} \sim q(x_{0})$"
            if self.rank == 0:
                pbar.set_description("Training...")

            ori_image = ori_image.to(self.device)
            loss = model.module.get_loss(ori_image)
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
        if self.rank == 0:
            pbar = tqdm(self.val_dl, leave=False)
        else:
            pbar = self.val_dl
        for ori_image in pbar:
            if self.rank == 0:
                pbar.set_description("Validating...")

            ori_image = ori_image.to(self.device)
            loss = model.module.get_loss(ori_image.detach())
            val_loss += (loss.item() / len(self.val_dl))
        return val_loss

    @staticmethod
    def save_model_params(model, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(modify_state_dict(model.module.state_dict()), str(save_path))
        print(f"Saved model params as '{str(save_path)}'.")

    def save_ckpt(self, epoch, model, optim, min_val_loss, scaler):
        if self.rank == 0:
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "epoch": epoch,
                "model": modify_state_dict(model.module.state_dict()),
                "optimizer": optim.state_dict(),
                "min_val_loss": min_val_loss,
            }
            if scaler is not None:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, str(self.ckpt_path))

    @torch.inference_mode()
    def test_sampling(self, epoch, model, batch_size):
        if self.rank == 0:
            gen_image = model.module.sample(batch_size=batch_size)
            gen_grid = image_to_grid(gen_image, n_cols=int(batch_size ** 0.5))
            sample_path = str(self.save_dir/f"sample-epoch={epoch}.jpg")
            save_image(gen_grid, save_path=sample_path)
            wandb.log({"Samples": wandb.Image(sample_path)}, step=epoch)

    def train(self, n_epochs, model, optim, scaler, n_warmup_steps):
        for param in model.parameters():
            try:
                param.register_hook(lambda grad: torch.clip(grad, -1, 1))
            except Exception:
                continue

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
            if val_loss < min_val_loss and self.rank == 0:
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

            self.test_sampling(epoch=epoch, model=model, batch_size=16)

            if self.rank == 0:
                log = f"[ {get_elapsed_time(start_time)} ]"
                log += f"[ {epoch}/{n_epochs} ]"
                log += f"[ Train loss: {train_loss:.4f} ]"
                log += f"[ Val loss: {val_loss:.4f} | Best: {min_val_loss:.4f} ]"
                print(log)
                wandb.log(
                    {"Train loss": train_loss, "Val loss": val_loss, "Min val loss": min_val_loss},
                    step=epoch,
                )


class DistDataParallel(object):
    def __init__(self, args):
        self.args = args

    def setup(self, rank, world_size, port):
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )

    def cleanup(self):
        dist.destroy_process_group()

    def main_worker(self, rank, workld_size, run):
        self.setup(rank=rank, world_size=workld_size, port=self.args.PORT)
        DEVICE = torch.device(f"cuda:{rank}")
        set_seed(self.args.SEED + rank)
        print(f"[ DEVICE: {DEVICE} ][ RANK: {rank} ]")

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        train_dl, val_dl = get_train_and_val_dls_ddp(
            data_dir=self.args.DATA_DIR,
            img_size=self.args.IMG_SIZE,
            batch_size=self.args.BATCH_SIZE,
            n_cpus=self.args.N_CPUS,
            rank=rank,
            world_size=workld_size,
        )
        trainer = Trainer(
            run=run,
            train_dl=train_dl,
            val_dl=val_dl,
            save_dir=self.args.SAVE_DIR,
            device=DEVICE,
            rank=rank,
        )

        net = UNet()
        model = DDPM(model=net, img_size=self.args.IMG_SIZE, device=DEVICE)
        model = DDP(model, device_ids=[rank])
        print_n_params(model)
        # "We set the batch size to 128 for CIFAR10 and 64 for larger images."
        optim = AdamW(model.parameters(), lr=self.args.LR)
        scaler = get_grad_scaler(device=DEVICE)

        trainer.train(
            n_epochs=self.args.N_EPOCHS,
            model=model,
            optim=optim,
            scaler=scaler,
            n_warmup_steps=self.args.N_WARMUP_STEPS,
        )

        self.cleanup()

    def run(self, run):
        world_size = torch.cuda.device_count()
        mp.spawn(
            self.main_worker,
            args=(world_size, run),
            nprocs=world_size,
            join=True,
        )


def main():
    args = get_args()
    ddp = DistDataParallel(args)
    run = wandb.init(project="DDPM")
    ddp.run(run)


if __name__ == "__main__":
    main()
