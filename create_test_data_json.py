# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:54:10 2022

@author: Ali
"""


import json
import numpy as np
import os
import cv2

test_path = "mmdetection-master/data/public_test/test2_images"


file_list = os.listdir(test_path)

test_gt = []
id_count=0
for img in file_list:
    print(id_count)
    img_load = cv2.imread(test_path+'/'+img)
    height, width, channels = img_load.shape
    
    test_gt.append({"file_name":img,
                    "id":id_count,
                    "height ":height,
                    "width":width,})
    id_count+=1
    
test_gt_dict = dict({"images":test_gt})

with open("mmdetection-master/data/public_test/test_gt.json",'w') as f:
    json.dump(test_gt_dict, f)