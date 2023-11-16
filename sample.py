# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

import torch
import argparse

from utils import get_config, save_image
from ddpm import DDPM
from ddim import DDIM
from celeba import CelebADataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["normal", "progression", "interpolation", "coarse_to_fine"],
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, required=False) # For `"normal"`, `"progression"`

    parser.add_argument("--data_dir", type=str, required=False) # For `"interpolation"`, `"coarse_to_fine"`
    parser.add_argument("--timestep", type=int, required=False) # For `"interpolation"`, `"coarse_to_fine"`
    parser.add_argument("--idx1", type=int, required=False) # For `"interpolation"`, `"coarse_to_fine"`
    parser.add_argument("--idx2", type=int, required=False) # For `"interpolation"`, `"coarse_to_fine"`

    args = parser.parse_args()
    return args


def get_ddpm_from_checkpoint(ckpt_path, n_timesteps, init_beta, fin_beta, device):
    # ddpm = DDPM(
    ddpm = DDIM(
        n_timesteps=n_timesteps,
        init_beta=init_beta,
        fin_beta=fin_beta,
    ).to(device)
    state_dict = torch.load(str(ckpt_path), map_location=device)
    ddpm.load_state_dict(state_dict)
    return ddpm


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(args)

    ddpm = get_ddpm_from_checkpoint(
        ckpt_path=CONFIG["CKPT_PATH"],
        n_timesteps=CONFIG["N_TIMESTEPS"],
        init_beta=CONFIG["INIT_BETA"],
        fin_beta=CONFIG["FIN_BETA"],
        device=CONFIG["DEVICE"],
    )

    if CONFIG["MODE"] == "progression":
        ddpm.progressively_sample(
            batch_size=CONFIG["BATCH_SIZE"],
            n_channels=CONFIG["N_CHANNELS"],
            img_size=CONFIG["IMG_SIZE"],
            device=CONFIG["DEVICE"],
            save_path=CONFIG["SAVE_PATH"],
        )
    else:
        if CONFIG["MODE"] == "normal":
            gen_image = ddpm.sample(
                batch_size=CONFIG["BATCH_SIZE"],
                n_channels=CONFIG["N_CHANNELS"],
                img_size=CONFIG["IMG_SIZE"],
                device=CONFIG["DEVICE"],
            )
        else:
            ds = CelebADataset(data_dir=CONFIG["DATA_DIR"], img_size=CONFIG["IMG_SIZE"])
            image1 = ds[CONFIG["IDX1"]][None, :]
            image2 = ds[CONFIG["IDX2"]][None, :]
            if CONFIG["MODE"]  == "interpolation":
                gen_image = ddpm.interpolate(image1, image2, timestep=CONFIG["TIMESTEP"])
            else:
                gen_image = ddpm.coarse_to_fine_interpolate(image1, image2)
        save_image(gen_image, path=CONFIG["SAVE_PATH"])
