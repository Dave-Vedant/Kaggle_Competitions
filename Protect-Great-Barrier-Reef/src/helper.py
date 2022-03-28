import shutil
import numpy as np

import tqdm
from tqdm import trange

import imagesize

# use bbox library created on https://github.com/awsaf49/bbox
from bbox.utils import coco2yolo, coco2voc, voc2yolo
from bbox.utils import draw_bboxes, load_image
from bbox.utils import clip_bbox, str2annot, annot2str

# copy files
def make_copy(row):
    shutil.copyfile(row.old_image_path, row.image_path)
    return

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row

np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for idx in range(1)]



