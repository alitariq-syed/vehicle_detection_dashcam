# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:36:17 2022

@author: Ali
"""
import os
os.listdir()

# Check Pytorch installation
import torch, torchvision
print('Torch Version: ',torch.__version__)
print('Is Torch CUDA available? ', torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print('MMDetection Version: ', mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('CUDA Compiler Version: ', get_compiling_cuda_version())
print('Compiler Version: ',get_compiler_version())


from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
config = 'configs/swin/faster_rcnn_swin-t-p4-w7_fpn_1x_motive_challenge.py'
#config = 'configs/swin/faster_rcnn_swin-t-p4-w7_fpn_1x_motive_challenge_fp16.py'
#config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_fp16_1x_motive_challenge.py'
#config = 'configs/vfnet/vfnet_r50_fpn_1x_motive_challenge.py'

# Setup a checkpoint file to load
# checkpoint = '/content/mmdetection/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth'
# checkpoint = 'checkpoints/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth'
# initialize the detector
model = init_detector(config, device='cuda:0')


from mmdet.apis import set_random_seed


from mmcv import Config
cfg = Config.fromfile(config)
print(f'Config:\n{cfg.pretty_text}')