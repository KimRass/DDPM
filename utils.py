# "We assume that image data consists of integers in $\{0, 1, \ldots, 255\}$ scaled linearly
# to $[-1, 1]$. This ensures that the neural network reverse process operates
# on consistently scaled inputs starting from the standard normal prior $p(x_{T})$."

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from datetime import timedelta
from time import time
import yaml

from ddpm import DDPMForCelebA


def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def denorm(tensor):
    tensor /= 2
    tensor += 0.5
    return tensor


def image_to_grid(image, n_cols):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def show_forward_process(ddpm, dl, device):
    for batch in dl:
        image = batch[0]
        image = image.to(device)

        grid = image_to_grid(image, n_cols=4)
        grid.show()

        for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            grid = image_to_grid(
                ddpm(image, t=[int(percent * ddpm.n_timesteps) - 1] * len(image)),
                n_cols=4,
            )
            grid.show()
        break


def gather(x, t):
    return x[t].view(-1, 1, 1, 1)


def get_noise(batch_size, n_channels, img_size, device):
    return torch.randn(batch_size, n_channels, img_size, img_size, device=device)


def sample_timestep(n_timesteps, batch_size, device):
    return torch.randint(low=0, high=n_timesteps, size=(batch_size, 1), device=device)


def get_ddpm_from_checkpoint(ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    ddpm = DDPMForCelebA(
        n_timesteps=state_dict["n_timesteps"],
        time_dim=state_dict["time_dimension"],
        init_beta=state_dict["initial_beta"],
        fin_beta=state_dict["final_beta"],
    ).to(device)
    ddpm.load_state_dict(state_dict["model"])
    return ddpm


# n_timesteps = 300
# betas = linear_beta_schedule(n_timesteps)
# alphas = 1 - betas # $\alpha_{t} = 1 - \beta_{t}$
# alpha_bars = torch.cumprod(alphas, dim=0) # $\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$
# alpha_bars_prev = F.pad(alpha_bars[:-1], pad=(1, 0), value=1.0)
# # sqrt_recip_alphas = torch.sqrt(1 / alphas)
# sqrt_recip_alphas = (1 / alphas) ** 0.5

# # sqrt_one_minus_alpha_bars = (1 - alpha_bars) ** 0.5

# posterior_variance = betas * (1. - alpha_bars_prev) / (1. - alpha_bars)
# posterior_variance

# image = Image.open("/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages/2007_001709.jpg")


# IMG_SIZE = 128
# transformer = T.Compose([
#     T.Resize(IMG_SIZE),
#     T.CenterCrop(IMG_SIZE),
#     T.ToTensor(),
#     T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])
# reverse_transformer = T.Compose([
#      T.Lambda(lambda t: (t + 1) / 2),
#      T.ToPILImage(),
# ])

# init_x = transformer(image).unsqueeze(0)
# init_x


# def extract(a, t, shape):
#     b = t.shape[0]
#     out = torch.gather(a, dim=-1, index=t)
#     out = out.view(b, *((1,) * (len(shape) - 1)))
#     return out


# if noise is None:
#     noise = torch.randn_like(init_x) # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

# t = torch.tensor([40])
# alpha_bar = extract(alpha_bars, t=t, shape=init_x.shape)
# var = 1 - alpha_bars # $(1 - \bar{\alpha_{t}})\mathbf{I}$
# (var ** 0.5) * noise

# sqrt_alpha_bars = alpha_bars ** 0.5
