# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from utils import get_config, save_image
from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def get_ddpm_from_checkpoint(ckpt_path, n_timesteps, init_beta, fin_beta, device):
    ddpm = DDPM(
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

    for idx in tqdm(range(1, CONFIG["N_EVAL_IMAGES"] + 1)):
        gen_image = ddpm.sample(
            batch_size=1,
            n_channels=CONFIG["N_CHANNELS"],
            img_size=CONFIG["IMG_SIZE"],
            device=CONFIG["DEVICE"],
        )
        save_image(
            gen_image,
            path=Path(CONFIG["SAVE_DIR"])/f"""{str(idx).zfill(len(str(CONFIG["N_EVAL_IMAGES"])))}.jpg""",
        )
