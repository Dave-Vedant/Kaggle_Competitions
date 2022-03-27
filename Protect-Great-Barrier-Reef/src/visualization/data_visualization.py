import numpy as np
import matplotlib.pyplot as plt
from bbox.utils import draw_bboxes, load_image, str2annot

from scipy.stats import gaussian_kde
from data.data_processing import all_bboxes, df, bbox_df
from helper import colors


all_bboxes = np.array(all_bboxes)

x_val = all_bboxes[...,0]
y_val = all_bboxes[...,1]

xy = np.vstack([x_val,y_val])
z = gaussian_kde(xy)(xy)

fig,ax = plt.subplots(figsize= (10,10))
ax.scatter(x_val, y_val, c=z, s=100, cmap='viridis')
print('x_center VS y_center')
plt.show()


x_val = all_bboxes[...,2]
y_val = all_bboxes[...,3]

xy = np.vstack([x_val,y_val])
z= gaussian_kde(xy)(xy)

fig,ax = plt.subplots(figsize=(10,10))

ax.scatter(x_val, y_val, c=z, s=100, cmap='viridis')
print('width VS height')
plt.show()



import matplotlib as mpl
import seaborn as sns

f, ax = plt.subplots(figsize=(12, 6))
sns.despine(f)

sns.histplot(
    bbox_df,
    x="area", hue="fold",
    multiple="stack",
    palette="viridis",
    edgecolor=".3",
    linewidth=.5,
    log_scale=True,
)
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([500, 1000, 2000, 5000, 10000]);
print('Count over Area')


df2 = df[(df.num_bbox>0)].sample(100) # takes samples with bbox
y = 3; x = 2
plt.figure(figsize=(12.8*x, 7.2*y))
for idx in range(x*y):
    row = df2.iloc[idx]
    img           = load_image(row.image_path)
    image_height  = row.height
    image_width   = row.width
    with open(row.label_path) as f:
        annot = str2annot(f.read())
    bboxes_yolo = annot[...,1:]
    labels      = annot[..., 0].astype(int).tolist()
    names         = ['cots']*len(bboxes_yolo)
    plt.subplot(y, x, idx+1)
    plt.imshow(draw_bboxes(img = img,
                           bboxes = bboxes_yolo, 
                           classes = names,
                           class_ids = labels,
                           class_name = True, 
                           colors = colors, 
                           bbox_format = 'yolo',
                           line_thickness = 2))
    plt.axis('OFF')
plt.tight_layout()
plt.show()


# visualize final output...
