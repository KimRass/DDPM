#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train.py\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/ddpm"\
    --img_size=128\
    --n_epochs=50\
    --batch_size=16\
    --lr=0.0003\
    --n_cpus=2\
    --n_blocks=2\
