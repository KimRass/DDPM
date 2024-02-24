#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train.py\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/ddpm"\
    --n_epochs=50\
    --batch_size=8\
    --lr=0.0003\
    --n_cpus=2\
    --img_size=64\
    --channels=128\
    --channel_mults="(1, 2, 2, 2)"\
    --attns="(False, True, True, True)"\
    --n_res_blocks=2\
