# Parameters
FOLD = 1
DIM = 3000
MODEL = 'yolov5s6'
BATCH = 4
EPOCHS = 7
OPTIMIZER = 'Adam'
PROJECT = 'great-barrier-reef'

import time
TIME = time.time_ns()
SIGN = 'VD'
NAME = f'{MODEL}-dim{DIM}-fold{FOLD}-{TIME}--{SIGN}'              # time is dynamic incase if we forget to change
