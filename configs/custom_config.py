log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[16, 30],
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=128,
        max_width=128,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(
        type='NormalizeOCR',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=128,
        max_width=128,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(
        type='NormalizeOCR',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'valid_ratio',
            'resize_shape', 'img_norm_cfg', 'ori_filename'
        ])
]
num_chars = 229
max_seq_len = 26
label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT_VN_187',
    with_unknown=False,
    with_padding=False,
    lower=False,
    max_seq_len=26)
model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
        type='ABIVisionModel',
        encoder=dict(
            type='TransformerEncoder',
            n_layers=4,
            n_head=8,
            d_model=512,
            d_inner=2048,
            dropout=0.2,
            max_len=256),
        decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=8,
            attn_width=32,
            attn_mode='nearest',
            use_result='feature',
            num_chars=229,
            max_seq_len=26,
            init_cfg=dict(type='Xavier', layer='Conv2d'))),
    decoder=dict(
        type='ABILanguageDecoder',
        d_model=512,
        n_head=8,
        d_inner=2048,
        n_layers=5,
        dropout=0.2,
        detach_tokens=True,
        use_self_attn=False,
        pad_idx=228,
        num_chars=229,
        max_seq_len=26,
        init_cfg=None),
    fuser=dict(
        type='ABIFuser',
        d_model=512,
        num_chars=229,
        init_cfg=None,
        max_seq_len=26),
    loss=dict(
        type='ABILoss',
        enc_weight=1.0,
        dec_weight=1.0,
        fusion_weight=1.0,
        num_classes=229),
    label_convertor=dict(
        type='ABIConvertor',
        dict_type='DICT_VN_187',
        with_unknown=False,
        with_padding=False,
        lower=False,
        max_seq_len=26),
    max_seq_len=26,
    iter_size=3)
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/mnt/disk1/mbbank/OCR/DATA/data_quangnd/new_train',
                ann_file=
                '/mnt/disk1/mbbank/OCR/DATA/team/train.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=128,
                max_width=128,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(
                type='RandomWrapper',
                p=0.5,
                transforms=[
                    dict(
                        type='OneOfWrapper',
                        transforms=[
                            dict(type='RandomRotateTextDet', max_angle=15),
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomAffine',
                                degrees=8,
                                translate=(0.2, 0.2),
                                scale=(0.5, 1.0),
                                shear=(-25, 25)),
                            dict(
                                type='TorchVisionWrapper',
                                op='RandomPerspective',
                                distortion_scale=0.5,
                                p=1)
                        ])
                ]),
            dict(
                type='RandomWrapper',
                p=0.25,
                transforms=[
                    dict(type='PyramidRescale'),
                    dict(
                        type='Albu',
                        transforms=[
                            dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                            dict(type='GaussianBlur', blur_limit=3, p=0.5)
                        ])
                ]),
            dict(
                type='RandomWrapper',
                p=0.25,
                transforms=[
                    dict(
                        type='TorchVisionWrapper',
                        op='ColorJitter',
                        brightness=0.5,
                        saturation=0.5,
                        contrast=0.5,
                        hue=0.1)
                ]),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'text',
                    'valid_ratio', 'resize_shape'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/mnt/disk1/mbbank/OCR/DATA/data_quangnd/new_train',
                ann_file=
                '/mnt/disk1/mbbank/OCR/DATA/team/test.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=128,
                max_width=128,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape', 'img_norm_cfg', 'ori_filename'
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/mnt/disk1/naver/hieunguyen/ocr-task/Handwritten OCR/new_train/',
                ann_file=
                '/mnt/disk1/naver/hieunguyen/ocr-task/Handwritten OCR/val_annotation.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=128,
                max_width=128,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape', 'img_norm_cfg', 'ori_filename'
                ])
        ]))
evaluation = dict(interval=1, metric='acc', save_best='0_char_precision')
seed = 0
gpu_ids = [2]
