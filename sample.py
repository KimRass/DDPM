# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import argparse

from utils import get_device, image_to_grid, save_image
from model import UNet, DDPM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["normal", "denoising_process", "interpolation", "coarse_to_fine"],
    )
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    # For `"normal"`, `"denoising_process"`
    parser.add_argument("--batch_size", type=int, required=False)

    # For `"interpolation"`, `"coarse_to_fine"`
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--image_idx1", type=int, required=False)
    parser.add_argument("--image_idx2", type=int, required=False)

    # Architecture
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--init_channels", type=int, required=True)
    parser.add_argument("--channels", type=str, required=True)
    parser.add_argument("--attns", type=str, required=True)
    parser.add_argument("--n_blocks", type=int, default=2, required=False)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


if __name__ == "__main__":
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()
    args = get_args()
    
    net = UNet(
        init_channels=args.INIT_CHANNELS,
        channels=eval(args.CHANNELS),
        attns=eval(args.ATTNS),
        n_blocks=args.N_BLOCKS,
    )
    model = DDPM(img_size=args.IMG_SIZE, net=net, device=DEVICE)
    state_dict = torch.load(str(args.MODEL_PARAMS), map_location=DEVICE)
    model.load_state_dict(state_dict)

    if args.MODE == "denoising_process":
        model.vis_denoising_process(
            batch_size=args.BATCH_SIZE, save_path=args.SAVE_PATH,
        )
    else:
        if args.MODE == "normal":
            gen_image = model.sample(args.BATCH_SIZE)
            gen_grid = image_to_grid(gen_image, n_cols=int(args.BATCH_SIZE ** 0.5))
            save_image(gen_grid, save_path=args.SAVE_PATH)
        else:
            if args.MODE  == "interpolation":
                gen_image = model.interpolate(
                    data_dir=args.DATA_DIR,
                    image_idx1=args.IMAGE_IDX1,
                    image_idx2=args.IMAGE_IDX2,
                )
                gen_grid = image_to_grid(gen_image, n_cols=12)
                save_image(gen_grid, save_path=args.SAVE_PATH)
            else:
                gen_image = model.coarse_to_fine_interpolate(
                    data_dir=args.DATA_DIR,
                    image_idx1=args.IMAGE_IDX1,
                    image_idx2=args.IMAGE_IDX2,
                )
                gen_grid = image_to_grid(gen_image, n_cols=12)
                save_image(gen_grid, save_path=args.SAVE_PATH)
