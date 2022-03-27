import os
from define_directories import BASE_DIR
from models.model_parameters import EPOCHS,MODEL, PROJECT, BATCH, NAME, DIM, OPTIMIZER


# define hyper parameter file
hyp_file =  f'{BASE_DIR}/Data/processed/hyp.yaml'

os.system("python3 {BASE_DIR}/Data/processed/yolov5/train.py --img {DIM} --batch {BATCH} --epochs {EPOCHS} --optimizer {OPTIMIZER} --data {BASE_DIR}/Data/processed/gbr.yaml --hyp {BASE_DIR}/Data/processed/hyp.yaml --weights {MODEL}.pt --project {PROJECT} --name {NAME} --exist-ok")

OUTPUT_DIR = '{}/{}'.format(PROJECT, NAME)
os.system("ls {OUTPUT_DIR}")