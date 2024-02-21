import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/DDPM/")

import torch
import gc
from torch.optim import AdamW
from torch.nn import functional as F
from pathlib import Path
from time import time
from tqdm import tqdm

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
# from model import DDPM
# from model_labml import labmlUNet
from old_model import DDPM


DEVICE = get_device()

batch_size = 1
train_dl, val_dl, _ = get_dls(
    data_dir="/Users/jongbeomkim/Documents/datasets/",
    img_size=32,
    batch_size=batch_size,
    n_cpus=2,
)
train_di = iter(train_dl)

ori_image = next(train_di).to(DEVICE)
# image_to_grid(ori_image, 1).show()


batch_size = 64
ori_image = ori_image.repeat(batch_size, 1, 1, 1)


model = DDPM(
    img_size=32,
    init_channels=64,
    channels=(64, 128, 256, 1024),
    attns=(False, True, False, False),
    n_blocks=2,
    device=DEVICE,
)
optim = AdamW(model.parameters(), lr=0.0003)

# for i in tqdm(range(100)):
for i in range(200):
    loss = model.get_loss(ori_image)
    # noisy_image, pred_noise, diffusion_step, random_noise = model.get_loss(ori_image)
    # loss = F.mse_loss(pred_noise, random_noise, reduction="mean")
    print(f"{loss.item():.4f}")

    optim.zero_grad()
    loss.backward()
    optim.step()


    # recon_image = model.reconstruct(noisy_image=noisy_image, noise=pred_noise, diffusion_step=diffusion_step)
    # image_to_grid(recon_image, n_cols=int(recon_image.size(0) ** 0.5)).show()

    # diffusion_step = model.sample_diffusion_step(batch_size)
    # diffusion_step
    # noisy_image = model.perform_diffusion_process(ori_image, diffusion_step)
    # model.perform_denoising_process(noisy_image)
    # image_to_grid(noisy_image, 8).show()

    # pred_noise = model(noisy_image, diffusion_step)
    # image = (noisy_image - ((1 - alpha_bar_t) ** 0.5) * random_noise) / (alpha_bar_t ** 0.5)

gen_image = model.sample(16)
image_to_grid(gen_image, 4).show()



# for cur_diffusion_step in tqdm(range(999, -1, -1)):
for cur_diffusion_step in tqdm(range(1000)):
    # cur_diffusion_step = 300
    diffusion_step = torch.full(
        size=(64,),
        fill_value=cur_diffusion_step,
        dtype=torch.long,
        device=DEVICE,
    )
    random_noise = model.sample_noise(batch_size)
    alpha_bar_t = model.index(model.alpha_bar, diffusion_step=diffusion_step)
    mean = (alpha_bar_t ** 0.5) * ori_image
    var = 1 - alpha_bar_t
    noisy_image = mean + (var ** 0.5) * random_noise
    # image_to_grid(noisy_image, 1).show()

    pred_noise = model.net(noisy_image=noisy_image, diffusion_step=diffusion_step)
    loss = F.mse_loss(pred_noise, random_noise, reduction="mean")
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    if cur_diffusion_step % 50 == 0:
        print(f"{loss.item():.4f}")

        recon_image = (noisy_image - ((1 - alpha_bar_t) ** 0.5) * pred_noise) / (alpha_bar_t ** 0.5)
        # recon_image.mean(), recon_image.std()
        image = (noisy_image - ((1 - alpha_bar_t) ** 0.5) * random_noise) / (alpha_bar_t ** 0.5)
        image_to_grid(torch.cat([image, recon_image]), 2).show()

        # x = noisy_image
        # for step in range(cur_diffusion_step, -1, -1):
        #     x = model.take_single_denoising_step(x, step)
        # image_to_grid(torch.cat([ori_image, noisy_image, x]), 3).show()


gen_image = model.sample(1)
image_to_grid(gen_image, 1).show()



x = model.sample_noise(batch_size)
for step in range(999, -1, -1):
    x = model.take_single_denoising_step(x, step)
image_to_grid(x, 1).show()



# print_n_prams(model)
# labml_model = labmlUNet().to(DEVICE)
# print_n_prams(labml_model)
# ori_image = torch.randn(1, 3, 32, 32).to(DEVICE)
# diffusion_step = model.sample_diffusion_step(batch_size=ori_image.size(0)).to(DEVICE)
# pred_noise = model.net(ori_image, diffusion_step)
# pred_noise = labml_model(ori_image, diffusion_step)


# model.sample(16)


# ori_image.mean().item(), ori_image.std().item()
# diffusion_step = model.batchify_diffusion_steps(999, 4)
# _, noisy_image = model.get_noise_and_noisy_image(ori_image, diffusion_step)
# image_to_grid(noisy_image, 2).show()
# noisy_image.mean().item(), noisy_image.std().item()

# random_noise = model.sample_noise(4)
# random_noise.mean().item(), random_noise.std().item()




# model.beta
# model.alpha[0], model.alpha[-1]



# for _ in range(30):
#     loss = model.get_loss(ori_image)
#     diffusion_step = model.sample_diffusion_step(batch_size=ori_image.size(0))
#     random_noise = model.sample_noise(batch_size=ori_image.size(0))
#     noisy_image = model(
#         ori_image=ori_image, diffusion_step=diffusion_step, random_noise=random_noise,
#     )
#     # pred_noise = net(noisy_image=noisy_image, diffusion_step=diffusion_step)
#     # loss = F.mse_loss(pred_noise, random_noise, reduction="mean")
#     print(f"{loss.item():.3f}")

#     optim.zero_grad()
#     loss.backward()
#     optim.step()

#     for name, model_param in model.named_parameters():
#         if torch.any(torch.isnan(model_param.grad)):
#             print(name)


# time_channels = 128
# d_model = time_channels // 4
# n_diffusion_steps = 1000

# emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
# emb = torch.exp(-emb)
# pos = torch.arange(n_diffusion_steps).float()
# emb = pos[:, None] * emb[None, :]
# emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
# emb = emb.view(n_diffusion_steps, d_model)
# emb.shape, pe_mat.shape
# emb[: 5, : 5] == pe_mat[: 5, : 5]
# torch.abs(emb - pe_mat).max()
# emb[-1, -5:] == pe_mat[-1, -5:]
# emb[-1, -5].item(), pe_mat[-1, -5].item()


# pos = torch.arange(n_diffusion_steps).unsqueeze(1)
# i = torch.arange(d_model // 2).unsqueeze(0)
# angle = pos / (10_000 ** (2 * i / d_model))

# pe_mat = torch.zeros(size=(n_diffusion_steps, d_model))
# pe_mat[:, 0:: 2] = torch.sin(angle)
# pe_mat[:, 1:: 2] = torch.cos(angle)

from diffusers import DDPMScheduler

