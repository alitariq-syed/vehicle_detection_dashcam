# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
	
	dict(
        type='AutoAugment',
        policies=[[
			dict(
				type='RandomCrop',
				crop_type='absolute',
				crop_size=(180,320),
				allow_negative_crop=False),
				],
				[
                dict(
				type='RandomCrop',
				crop_type='absolute',
				crop_size=(180,320),
				allow_negative_crop=True),
                  ],
				  
				  
				  
				  ]),		
				  
				
	dict(type='Resize', img_scale=(180, 320), keep_ratio=True),

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
        img_scale=(180, 320),
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
        ann_file=data_root + 'train/train_gt.json',
        img_prefix=data_root + 'train/train_images/',
        #ann_file=data_root + 'train/valid_gt.json',
        #img_prefix=data_root + 'train/train_images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train/valid_gt.json',
        img_prefix=data_root + 'train/train_images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'public_test/test_gt.json',
        img_prefix=data_root + 'public_test/test2_images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
