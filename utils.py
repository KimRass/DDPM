# "We assume that image data consists of integers in $\{0, 1, \ldots, 255\}$ scaled linearly
# to $[-1, 1]$. This ensures that the neural network reverse process operates
# on consistently scaled inputs starting from the standard normal prior $p(x_{T})$."

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from datetime import timedelta
from time import time
import yaml


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
