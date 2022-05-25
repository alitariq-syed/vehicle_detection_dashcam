# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:54:10 2022

@author: Ali
"""


import json
import numpy as np

results_path = "mmdetection-master/tutorial_swin/my_results.bbox.json"
with open(results_path) as f:
    results_coco = json.load(f)
    
"""
rename "score" to "confidence"
add "id":0 ; that is sequential id for each detection box
"""

id_count=0
results_submission = []
for item in results_coco:
    results_submission.append({"image_id":item["image_id"],
                               "bbox":item["bbox"],
                               "category_id":item["category_id"],
                               "id":id_count,
                               "confidence":item["score"]})
    id_count+=1

        
        
results_submission_format = dict({"annotations":results_submission})

with open("mmdetection-master/tutorial_swin/my_results_for_submission.json",'w') as f:
    json.dump(results_submission_format, f)