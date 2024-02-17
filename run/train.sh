#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train.py\
    --data_dir="/Users/jongbeomkim/Documents/datasets/celeba"\
    --save_dir="/Users/jongbeomkim/Documents/ddpm"\
    --img_size=64\
    --n_epochs=50\
    --batch_size=8\
    --lr=0.0003\
    --n_cpus=2\
    --channels=32\
    --n_blocks=1\
    # --run_id="gorgeous-candles-21"\
