# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:54:10 2022

@author: Ali
"""


import json
import numpy as np

# results_path = "mmdetection-master/tutorial_swin/my_results.bbox.json"
results_path = "SAHI_result_mscrop_60.json"

with open(results_path) as f:
    results_coco = json.load(f)

images_path = "mmdetection-master/data/public_test/test_gt.json"
with open(images_path) as f:
    images_gt = json.load(f)

"""
rename "score" to "confidence"
add "id":0 ; that is sequential id for each detection box
"""

id_count=0
results_submission = []
for item in results_coco:
    item["bbox"] = [int(val) for val in item["bbox"]]
    if item["score"]>=0.9:
        results_submission.append({"image_id":item["image_id"],
                                   "bbox":item["bbox"],
                                   "category_id":item["category_id"],
                                   "id":id_count,
                                   "confidence":item["score"]})
        id_count+=1

        
        
results_submission_format = dict({"annotations":results_submission,
                                  "images":images_gt["images"],
                                  "categories":[{"id": 1, "name": "Car", "supercategory": "none"}, {"id": 2, "name": "Truck", "supercategory": "none"}, {"id": 3, "name": "StopSign", "supercategory": "none"}, {"id": 4, "name": "traffic_lights", "supercategory": "none"}]})

with open("mmdetection-master/tutorial_swin/SAHI_result_mscrop_60_thr_0.9_for_submission.json",'w') as f:
    json.dump(results_submission_format, f)