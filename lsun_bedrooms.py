# Source: https://huggingface.co/datasets/pcuenq/lsun-bedrooms

import os
import shutil

from miniai.imports import *
from miniai.diffusion import *

from datasets import load_dataset

path_data = Path('data')
path_data.mkdir(exist_ok=True)
path = path_data/'bedroom'

url = 'https://s3.amazonaws.com/fast-ai-imageclas/bedroom.tgz'
if not path.exists():
    path_zip = fc.urlsave(url, path_data)
    shutil.unpack_archive('data/bedroom.tgz', 'data')

dataset = load_dataset("imagefolder", data_dir="data/bedroom")
dataset = dataset.remove_columns('label')
dataset = dataset['train'].train_test_split(test_size=0.05)
dataset.push_to_hub("pcuenq/lsun-bedrooms")
