#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train.py\
    --data_dir="/Users/jongbeomkim/Documents/datasets/celeba"\
    --save_dir="/Users/jongbeomkim/Documents/ddpm"\
    --img_size=64\
    --n_epochs=500\
    --batch_size=8\
    --lr=0.0003\
    --n_cpus=7\
    --n_blocks=2\
    # --run_id="gorgeous-candles-21"\