import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config, get_device, image_to_grid, show_forward_process
from data import get_mnist_dataset
from model import UNetForDDPM
from ddpm import DDPM

CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/DDPM/config.yaml")
# CONFIG = load_config(Path(__file__).parent/"config.yaml")

DEVICE = get_device()

if __name__ == "__main__":
    model = UNetForDDPM(n_timesteps=CONFIG["N_TIMESTEPS"])
    ddpm = DDPM(
        model=model,
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
        n_timesteps=CONFIG["N_TIMESTEPS"],
        device=DEVICE,
    )
    crit = nn.MSELoss()

    batch_size = 16
    ds = get_mnist_dataset("/Users/jongbeomkim/Documents/datasets")
    dl = DataLoader(ds, batch_size, shuffle=True)

    for x0, _ in dl:
        x0 = x0.to(DEVICE)
        # image_to_grid(x0, n_cols=4).show()

        # Pick some noise for each of the images in the batch, a timestep and the respective alpha_bars
        t = torch.randint(low=0, high=CONFIG["N_TIMESTEPS"], size=(batch_size,), device=DEVICE)
        t
        eta = torch.randn_like(x0, device=DEVICE)

        noisy_image = ddpm(x0, t=t, eta=eta)
        image_to_grid(noisy_image, n_cols=4).show()