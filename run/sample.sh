#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/ddpm_celeba_64×64.pth"
save_dir="/Users/jongbeomkim/Desktop/workspace/DDPM/samples/"
img_size=64

python3 ../sample.py\
    --mode="normal"\
    --model_params="$model_params"\
    --save_path="$save_dir/normal/0.jpg"\
    --batch_size=1\
    --img_size=$img_size\

# python3 ../sample.py\
#     --mode="denoising_process"\
#     --model_params="$model_params"\
#     --save_path="$save_dir/denoising_process/0.gif"\
#     --img_size=$img_size\
#     --batch_size=1\

# python3 ../sample.py\
#     --mode="interpolation"\
#     --model_params="$model_params"\
#     --save_path="$save_dir/interpolation/0.jpg"\
#     --img_size=$img_size\
#     --data_dir="/Users/jongbeomkim/Documents/datasets/"\
#     --image_idx1=50\
#     --image_idx2=100\

# python3 ../sample.py\
#     --mode="coarse_to_fine"\
#     --model_params="$model_params"\
#     --save_path="$save_dir/coarse_to_fine_interpolation/0.jpg"\
#     --img_size=$img_size\
#     --data_dir="/Users/jongbeomkim/Documents/datasets/"\
#     --image_idx1=50\
#     --image_idx2=100\
