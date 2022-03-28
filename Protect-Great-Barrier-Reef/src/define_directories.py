import os
import sys
import shutil

import numpy as np
import pandas as pd

# from data.data_processing import train_df, valid_df
# from Ipython.display import display

BASE_DIR = '/'    
print(BASE_DIR)   


ROOT_DIR = f'{BASE_DIR}/Data/raw'
IMAGE_DIR = f'{BASE_DIR}/Data/interim/images'
LABEL_DIR = f'{BASE_DIR}/Data/interim/labels'




##############
# Define path for train/ validation file
import yaml

main_dir = f'{BASE_DIR}/Data/processed/'

with open(os.path.join(main_dir, 'train.txt'),'w') as f:
    for path in train_df.image_path.tolist():
        f.write(path+'\n')

with open(os.path.join(main_dir,'val.txt'),'w') as f:
    for path in valid_df.image_path.tolist():
        f.write(path+'\n')                                            # @ERS: val.txt file error during model training ---> resolved

data = dict(
    path = f'{BASE_DIR}/Data/processed/',
    train = os.path.join(main_dir, 'train.txt'),
    val = os.path.join(main_dir, 'val.txt'),
    nc = 1,
    names= ['cots'],
)

with open(os.path.join(main_dir, 'gbr.yaml'), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(os.path.join(main_dir, 'gbr.yaml'), 'r')
print('\nyaml')
print(f.read())

