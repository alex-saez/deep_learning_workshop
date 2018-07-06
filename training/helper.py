from matplotlib import pyplot as plt

import  fastai
from  fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

def display_images_from_class(labels_df, label, num_images = 10):

        
    imgs = labels_df[labels_df["class"]==label].head(num_images).to_dict(orient='records')

    plt.figure(figsize=(50,30))
    plt.suptitle(f'{num_images} images of {label}', fontsize=16)

    columns = 5
    for i, record in enumerate(imgs):

        image_path = record['file']
        image = PIL.Image.open(image_path)
        plt.subplot(len(imgs) / columns + 1, columns, i + 1)
        plt.imshow(image)
        #plt.title(image_id)
        
def get_top_n_predictions(preds,labels,n=5):
    if len(preds.shape) >1:
        preds = preds.flatten()
    top_idxs = np.argpartition(preds, -n)[-n:]
    
    res =[]
    for idx in top_idxs[::-1]:
        res.append((labels[idx],preds[idx] ))
    return res