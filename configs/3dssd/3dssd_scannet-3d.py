_base_ = [
    '../_base_/datasets/scannet-3d-18class.py', '../_base_/default_runtime.py'
]

model = dict(
    type='SSD3DNet',
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 512, (256, 256)),
        radii=((0.2, ), (0.4, ), (0.8, )),
        num_samples=((64, ), (32, ), (32, )),
        sa_channels=(((64, 64, 128), ), ((128, 128, 256), ), ((128, 128,
                                                               256), )),
        aggregation_channels=(128, 256, 256),
        fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=((-1), (-1), (512, -1)),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    bbox_head=dict(
        type='SSD3DHead',
        in_channels=256,
        num_classes=18,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=False),
        vote_module_cfg=dict(
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(1.0, 1.0, 0.5)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=256,
            radii=(1.2, ),
            sample_nums=(32, ),
            mlp_channels=((256, 256, 512, 1024), ),
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
            use_xyz=True,
            normalize_xyz=False,
            bias=True),
        pred_layer_cfg=dict(
            in_channels=1024,
            shared_conv_channels=(512, 256),
            cls_conv_channels=(256, ),
            reg_conv_channels=(256, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=0),
        dir_res_loss=dict(type='SmoothL1Loss', reduction='sum', loss_weight=0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(type='SmoothL1Loss', reduction='sum', loss_weight=1.0)))

# model training and testing settings
train_cfg = dict(
    sample_mod='spec', pos_distance_thr=5.0, expand_dims_length=0.05)
test_cfg = dict(
    nms_cfg=dict(type='nms', iou_thr=0.1),
    sample_mod='spec',
    score_thr=0.0,
    per_class_proposal=True,
    max_output_num=100)

data = dict(samples_per_gpu=4, workers_per_gpu=4)

evaluation = dict(interval=1)

# optimizer
lr = 0.002  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[16, 24])
# runtime settings
total_epochs = 36

# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
