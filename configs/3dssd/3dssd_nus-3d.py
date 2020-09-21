_base_ = [
    '../_base_/models/3dssd_nus.py', '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,
        pad_empty_sweeps=True,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
                   'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'rect', 'Trv2c', 'P2', 'pcd_trans',
                   'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                   'pts_filename', 'cur_points_num'))
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['points'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'pad_shape', 'scale_factor', 'flip',
                           'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'rect', 'Trv2c', 'P2', 'pcd_trans', 'sample_idx',
                           'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                           'cur_points_num'))
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

evaluation = dict(interval=2)

# model settings
model = dict(
    bbox_head=dict(
        num_classes=10,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)))

# optimizer
lr = 0.002  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[80, 120])
# runtime settings
total_epochs = 150

# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
