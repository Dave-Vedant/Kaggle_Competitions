import os
import sys
import shutil

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
tqdm.pandas()

from joblib import Parallel, delayed
from IPython.display import display

# importing supportive functions
from define_directories import ROOT_DIR, IMAGE_DIR, LABEL_DIR
from helper import make_copy, get_bbox

from bbox.utils import coco2yolo, coco2voc, voc2yolo
from bbox.utils import draw_bboxes, load_image
from bbox.utils import clip_bbox, str2annot, annot2str

# cerate directory and path
df = pd.read_csv(f'{ROOT_DIR}/train.csv')
df['old_image_path'] = f'{ROOT_DIR}/train_images/video_'+df.video_id.astype(str)+'/'+df.video_frame.astype(str)+'.jpg'
df['image_path'] = f'{IMAGE_DIR}/'+ df.image_id + '.jpg'
df['label_path'] = f'{LABEL_DIR}/'+df.image_id+'.txt'
df['annotations'] = df['annotations'].progress_apply(eval)
display(df.head(5))


# Number of BBoxes
df['num_bbox'] = df['annotations'].progress_apply(lambda x: len(x))
data = (df.num_bbox>0).value_counts(normalize=True)*100
print(f"No BBox: {data[0]:0.2f}% | with BBox: {data[1]:0.2f}%")

# Data Cleaning
if True:
    df =df.query('num_bbox>0')

# image path
image_paths = df.old_image_path.tolist() 
_ = Parallel(n_jobs=-1, backend='threading') (delayed(make_copy)(row) for _ , row in tqdm(df.iterrows(), total=len(df)))

df['bboxes'] = df.annotations.progress_apply(get_bbox)
df.head()

df['width'] = 1280
df['height'] = 720
display(df.head())

count = 0
all_bboxes = []
bboxes_info = []

for row_index in tqdm(range(df.shape[0])):
    row = df.iloc[row_index]
    image_height = row.height
    image_width = row.width
    bboxes_coco = np.array(row.bboxes).astype(np.float32).copy()
    num_bbox = len(bboxes_coco)
    names = ['cots']* num_bbox
    labels = np.array([0]*num_bbox)[..., None].astype(str)

    with open(row.label_path, 'w') as f:
        if num_bbox < 1:
            annot = ''
            f.write(annot)
            count+=1
            continue
        bboxes_voc = coco2voc(bboxes_coco, image_height, image_width)
        bboxes_voc = clip_bbox(bboxes_voc, image_height, image_width)
        bboxes_yolo = voc2yolo(bboxes_voc, image_height, image_width).astype(str)
        all_bboxes.extend(bboxes_yolo.astype(float))
        bboxes_info.extend([[row.image_id, row.video_id, row.sequence]]* len(bboxes_yolo))
        annots = np.concatenate([labels, bboxes_yolo], axis=1)
        string = annot2str(annots)
        f.write(string)
    print('Missing', count)


# get the dimensions of each box in images to get an idea where the reef are in image as training data
bbox_df = pd.DataFrame(np.concatenate([bboxes_info, all_bboxes], axis=1),
columns= ['image_id','video_id','sequence','xmid','ymid','w','h'])

bbox_df[['xmid', 'ymid', 'w', 'h']] = bbox_df[['xmid', 'ymid', 'w', 'h']].astype(float)
bbox_df['area'] = bbox_df.w * bbox_df.h * 1280 * 720
bbox_df = bbox_df.merge(df[['image_id', 'fold']], on='image_id', how='left')
bbox_df.head(5)


# DataSet
# final dataset size for training
train_files = []
val_files = []
train_df = df.query("fold!=@FOLD")
valid_df = df.query("fold==@FOLD")
train_files += list(train_df.image_path.unique())
val_files += list(valid_df.image_path.unique())
len(train_files), len(val_files)



