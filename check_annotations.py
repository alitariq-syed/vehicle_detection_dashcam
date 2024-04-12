# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 23:25:25 2022

@author: Ali
"""
import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

gt_path = "train_gt_320_02.json"

with open(gt_path) as f:
    train_gt = json.load(f)
    
img_ids = train_gt["annotations"]

df = pd.DataFrame(img_ids)
count = df['image_id'].value_counts()

unique_ids = df['image_id'].unique()

images_list =  pd.DataFrame(train_gt["images"])


delete_list=[]
delete_ids=[]
for i in range(len(images_list)):
    if images_list['id'][i] in unique_ids:
        #keep
        continue
    else:
        delete_ids.append(images_list['id'][i])
        delete_list.append(images_list['file_name'][i])
        
#delete training images with no bbox annotation
removed=0
for file in tqdm(delete_list):
    # print(file)
    try:
        os.remove(path="H:/sliced/train_gt_images_320_02/"+file)
        removed+=1
    except:
        continue
print("file removed: ",removed)