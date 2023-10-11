naver_textrecog_data_root = '/mnt/disk1/mbbank/OCR/DATA/data_quangnd/new_train'
naver_textrecog_test = dict(
    ann_file='/mnt/disk1/mbbank/OCR/DATA/team/val.json',
    data_root='/mnt/disk1/mbbank/OCR/DATA/data_quangnd/new_train',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
naver_textrecog_train = dict(
    ann_file='/mnt/disk1/mbbank/OCR/DATA/team/train.json',
    data_root='/mnt/disk1/mbbank/OCR/DATA/data_quangnd/new_train',
    pipeline=None,
    test_mode=False,
    type='OCRDataset')

default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best='auto'),
    # checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
load_from = 'https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth'
resume = False
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[dict(type='WordMetric', mode=['exact']),
            dict(type='CharMetric')],
    dataset_prefixes=['NAVER'])
test_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[dict(type='WordMetric', mode=['exact']),
            dict(type='CharMetric')],
    dataset_prefixes=['NAVER'])
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextRecogLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')])
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', end=2, start_factor=0.001,
        convert_to_iter_based=True),
    dict(type='MultiStepLR', milestones=[16, 26, 36], end=40)
]
dictionary = dict(
    type='Dictionary',
    dict_file=
    'configs/textrecog/abinet/../../../dicts/vietnamese.txt',
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=False,
    with_unknown=False)
model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
        type='ABIEncoder',
        n_layers=3,
        n_head=8,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        max_len=256),
    decoder=dict(
        type='ABIFuser',
        vision_decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=8,
            attn_width=32,
            attn_mode='nearest',
            init_cfg=dict(type='Xavier', layer='Conv2d')),
        module_loss=dict(type='ABIModuleLoss', letter_case='unchanged'),
        postprocessor=dict(type='AttentionPostprocessor'),
        dictionary=dict(
            type='Dictionary',
            dict_file=
            'configs/textrecog/abinet/../../../dicts/vietnamese.txt',
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=False,
            with_unknown=False),
        max_seq_len=26,
        d_model=512,
        num_iters=3,
        language_decoder=dict(
            type='ABILanguageDecoder',
            d_model=512,
            n_head=8,
            d_inner=2048,
            n_layers=4,
            dropout=0.1,
            detach_tokens=True,
            use_self_attn=False)),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        ignore_empty=True,
        min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(128, 32)),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(type='RandomRotate', max_angle=15),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAffine',
                        degrees=15,
                        translate=(0.3, 0.3),
                        scale=(0.5, 2.0),
                        shear=(-45, 45)),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomPerspective',
                        distortion_scale=0.5,
                        p=1)
                ])
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                    dict(type='MotionBlur', blur_limit=5, p=0.5)
                ])
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1)
        ]),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(128, 32)),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
train_list = [
    dict(
        type='OCRDataset',
        data_root='/mnt/disk1/mbbank/OCR/DATA/data_quangnd/new_train',
        # data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
        ann_file='/mnt/disk1/mbbank/OCR/DATA/team/train.json',
        test_mode=False,
        pipeline=None)
]
test_list = [
    dict(
        type='OCRDataset',
        data_root='/mnt/disk1/mbbank/OCR/DATA/data_quangnd/new_train',
        ann_file='/mnt/disk1/mbbank/OCR/DATA/team/val.json',
        test_mode=True,
        pipeline=None)
]
train_dataset = dict(
    type='ConcatDataset',
    datasets=train_list,
    pipeline=train_pipeline)
test_dataset = dict(
    type='ConcatDataset',
    datasets=test_list,
    pipeline=test_pipeline)
train_dataloader = dict(
    batch_size=80,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

launcher = 'slurm'
work_dir = './work_dirs/abinet_case_unchanged'