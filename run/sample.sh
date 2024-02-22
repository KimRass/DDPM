#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/64-128-128_128_256_256_512-FFTFF/epoch=14-val_loss=0.0227.pth"
save_dir="/Users/jongbeomkim/Desktop/workspace/DDPM/samples/new"
img_size=64
init_channels=128
channels="(128, 128, 256, 256, 512)"
attns="(False, False, True, False, False)"
n_blocks=2

# python3 ../sample.py\
#     --mode="normal"\
#     --model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/64-128_128_256_256_512-FFTFF/epoch=14-val_loss=0.0227.pth"\
#     --save_path="$save_dir/normal/0.jpg"\
#     --batch_size=1\
#     --img_size=$img_size\
#     --init_channels=$init_channels\
#     --channels="$channels"\
#     --attns="$attns"\
#     --n_blocks=$n_blocks\

# python3 ../sample.py\
#     --mode="denoising_process"\
#     --model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/64-128_128_256_256_512-FFTFF/epoch=14-val_loss=0.0227.pth"\
#     --save_path="$save_dir/denoising_process/0.gif"\
#     --batch_size=1\
#     --img_size=$img_size\
#     --init_channels=$init_channels\
#     --channels="$channels"\
#     --attns="$attns"\
#     --n_blocks=$n_blocks\

# python3 ../sample.py\
#     --mode="interpolation"\
#     --model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/64-128-128_128_256_256_512-FFTFF/epoch=14-val_loss=0.0227.pth"\
#     --save_path="$save_dir/interpolation/0.jpg"\
#     --data_dir="/Users/jongbeomkim/Documents/datasets/"\
#     --image_idx1=50\
#     --image_idx2=100\
#     --img_size=$img_size\
#     --init_channels=$init_channels\
#     --channels="$channels"\
#     --attns="$attns"\
#     --n_blocks=$n_blocks\

python3 ../sample.py\
    --mode="coarse_to_fine"\
    --model_params="$model_params"\
    --save_path="$save_dir/coarse_to_fine_interpolation/0.jpg"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --image_idx1=50\
    --image_idx2=100\
    --img_size=$img_size\
    --init_channels=$init_channels\
    --channels="$channels"\
    --attns="$attns"\
    --n_blocks=$n_blocks\
