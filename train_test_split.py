# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:54:10 2022

@author: Ali
"""


import json
import numpy as np

gt_path = "mmdetection-master/data/train/train_gt.json"

with open(gt_path) as f:
    train_gt = json.load(f)
    
#try removing images that have no annotations (image id 38732)
train_images = []
train_annotations = []
for item in train_gt["images"]:
    if item["id"] < 38732:
        # print(item)
        train_images.append(item)
for item in train_gt["annotations"]:
    if item["image_id"] < 38732:
        # print(item)
        train_annotations.append(item)        
train_gt = dict({"annotations":train_annotations,"categories":train_gt["categories"],"images":train_images})

    
"""
image ids range from 0 - 39997

randomly select say 5% ids as validation set

extract thoses ids from "images" and corresponding "annotations" from train_gt and save as valid_gt
"""

from sklearn.model_selection import train_test_split


np.random.seed(111)
val_split = 0.05
valid_ids = np.random.randint(0,len(train_gt["images"])-1,int(val_split*len(train_gt["images"])))


valid_images = []
valid_annotations = []
for item in train_gt["images"]:
    if item["id"] in valid_ids:
        print(item)
        valid_images.append(item)

for item in train_gt["annotations"]:
    if item["image_id"] in valid_ids:
        item.update({"iscrowd":0})
        item.update({"area":100})
        print(item)
        
        valid_annotations.append(item)

# train_gt["categories"] = [{'id': 1, 'name': 'Car', 'supercategory': 'none'},
#   {'id': 2, 'name': 'Truck', 'supercategory': 'none'},
#   {'id': 3, 'name': 'StopSign', 'supercategory': 'none'},
#   {'id': 4, 'name': 'traffic_lights', 'supercategory': 'none'},
#   {'id': 5, 'name': 'background', 'supercategory': 'none'}]        
        
valid_gt = dict({"annotations":valid_annotations,"categories":train_gt["categories"],"images":valid_images})

with open("mmdetection-master/data/train/valid_gt.json",'w') as f:
    json.dump(valid_gt, f)
    
    

    
# with open("mmdetection-master/data/train/train_gt.json",'w') as f:
#     json.dump(train_gt, f)    