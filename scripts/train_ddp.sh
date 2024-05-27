#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train_ddp.py\
    --data_dir="/home/jongbeomkim/Documents/datasets/"\
    --save_dir="/home/jongbeomkim/Documents/ddpm"\
    --n_epochs=50\
    --batch_size=16\
    --lr=0.0003\
    --num_workers=8\
    --img_size=64\
    --n_warmup_steps=1000\
