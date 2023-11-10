# "We assume that image data consists of integers in $\{0, 1, \ldots, 255\}$ scaled linearly
# to $[-1, 1]$. This ensures that the neural network reverse process operates
# on consistently scaled inputs starting from the standard normal prior $p(x_{T})$."

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from datetime import timedelta
from time import time
from PIL import Image
from pathlib import Path
import yaml
from collections import OrderedDict
import random
import numpy as np
import os
from copy import deepcopy


def _args_to_config(args, config):
    copied = deepcopy(config)
    for k, v in vars(args).items():
        copied[k.upper()] = v
    return copied


def get_config(args=None):
    config = load_config(Path(__file__).parent/"config.yaml")
    if args is not None:
        config = _args_to_config(args=args, config=config)

    config["PARENT_DIR"] = Path(__file__).resolve().parent
    config["CKPTS_DIR"] = config["PARENT_DIR"]/"checkpoints"
    config["CKPT_TAR_PATH"] = config["CKPTS_DIR"]/"checkpoint.tar"

    config["DEVICE"] = get_device()
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


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


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(image).save(str(path), quality=100)


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


def index(x, t):
    return x[t].view(-1, 1, 1, 1)


def sample_noise(batch_size, n_channels, img_size, device):
    return torch.randn(batch_size, n_channels, img_size, img_size, device=device)


def sample_t(n_timesteps, batch_size, device):
    return torch.randint(low=0, high=n_timesteps, size=(batch_size,), device=device)


def modify_state_dict(state_dict, keyword="_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key in list(state_dict.keys()):
        if old_key and old_key.startswith(keyword):
            new_key = old_key[len(keyword):]
        else:
            new_key = old_key
        new_state_dict[new_key] = state_dict[old_key]
    return new_state_dict
