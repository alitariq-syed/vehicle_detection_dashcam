# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/TBX11K/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
	
	 dict(
         type='MinIoURandomCrop',
         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
         min_crop_size=0.3),
    
	dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
	
	#dict(type='RandomAffine'),
	
    dict(type='RandomFlip', flip_ratio=0.5),				  
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/json/all_train.json',
        img_prefix=data_root + 'imgs/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/json/all_val.json',
        img_prefix=data_root + 'imgs/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/json/all_val.json',
        img_prefix=data_root + 'imgs/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')