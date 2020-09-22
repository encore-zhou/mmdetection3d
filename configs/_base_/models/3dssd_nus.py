model = dict(
    type='SSD3DNet',
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        out_indices=(3, ),
        num_points=((4096, 12288), (1024, 2048), (512, 1024), (1024)),
        radii=((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)),
        num_samples=((32, 64), (32, 64), (32, 32), (32, 32)),
        sa_channels=(((64, 64, 128), (64, 96, 128)), ((128, 128, 256),
                                                      (128, 128, 256)),
                     ((128, 128, 256), (128, 128, 256)), ((128, 128, 256),
                                                          (128, 128, 256))),
        aggregation_channels=(64, 128, 256, 256),
        fps_mods=(('D-FPS', 'D-FPS'), ('FS', 'D-FPS'), ('FS', 'FS'), ('FS')),
        fps_sample_range_lists=((16384, -1), (4096, -1), (1024, -1), (-1)),
        dilated_group=(True, True, True, True),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    bbox_head=dict(
        type='SSD3DHead',
        in_channels=256,
        vote_module_cfg=dict(
            in_channels=256,
            num_points=1024,
            gt_per_seed=1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=1024,
            radii=(8, 10),
            sample_nums=(16, 32),
            mlp_channels=((256, 256, 256, 512), (256, 256, 512, 1024)),
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
            use_xyz=True,
            normalize_xyz=False,
            bias=True),
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, 128),
            cls_conv_channels=(128, ),
            reg_conv_channels=(128, ),
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
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        velocity_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0)))

# model training and testing settings
train_cfg = dict(
    sample_mod='spec',
    pos_distance_thr=10.0,
    expand_dims_length=0.05,
    use_voxel_sample=True,
    voxel_sampler_cfg=[
        dict(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-50, -50, -4, 50, 50, 2],
            max_num_points=1,
            max_voxels=16384),
        dict(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-50, -50, -4, 50, 50, 2],
            max_num_points=1,
            max_voxels=49152)
    ])
test_cfg = dict(
    nms_cfg=dict(type='nms', iou_thr=0.1),
    sample_mod='spec',
    score_thr=0.0,
    per_class_proposal=True,
    max_output_num=100,
    use_voxel_sample=True,
    voxel_sampler_cfg=[
        dict(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-50, -50, -4, 50, 50, 2],
            max_num_points=1,
            max_voxels=16384),
        dict(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-50, -50, -4, 50, 50, 2],
            max_num_points=1,
            max_voxels=49152)
    ])

# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 0.002  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 12])
# runtime settings
total_epochs = 15
