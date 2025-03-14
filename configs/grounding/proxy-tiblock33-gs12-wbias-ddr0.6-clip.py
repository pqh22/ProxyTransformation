_base_ = ['../default_runtime.py']
n_points = 100000

backend_args = None
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/',
#         'data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/'
#     }))

metainfo = dict(classes='all')

model = dict(
    type='SparseFeatureFusion3DGrounderPreshape',
    num_queries=256,
    voxel_size=0.01,
    gaussian_loss = False, # gaussian loss
    t_type ='clip-vit-large-patch14-336',
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        base_channels=16,  # to make it consistent with mink resnet
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    preshape = dict(type='ProxyTransformationNormReverse', n_points=n_points, grid_size=12, text_blocks=3, img_blocks=3, dynamic_drop_radio=0.6, num_sub=30),
    backbone_3d=dict(type='MinkResNet', in_channels=3, depth=34),
    use_xyz_feat=True,
    # # change due to no img feature fusion
    # neck=dict(
    #     type='CrossViewFeatureInteractor',
    #     input_dim=512, 
    #     embed_dim=256, 
    #     num_heads=8, 
    #     l_dim=256,
    #     use_new_text=True,),
    neck_3d=dict(type='MinkNeck',
                 num_classes=1,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=256,
                 voxel_size=0.01,
                 pts_prune_threshold=1000),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256,
                         feedforward_channels=2048,
                         ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(type='GroundingHead',
                   num_classes=256,
                   sync_cls_avg_factor=True,
                   decouple_bbox_loss=True,
                   decouple_groups=4,
                   share_pred_layer=True,
                   decouple_weights=[0.2, 0.2, 0.2, 0.4],
                   contrastive_cfg=dict(max_text_len=256,
                                        log_scale='auto',
                                        bias=True),
                   loss_cls=dict(type='mmdet.FocalLoss',
                                 use_sigmoid=True,
                                 gamma=2.0,
                                 alpha=0.25,
                                 loss_weight=1.0),
                   loss_bbox=dict(type='BBoxCDLoss',
                                  mode='l1',
                                  loss_weight=1.0,
                                  group='g8')),
    coord_type='DEPTH',
    # training and testing settings
    train_cfg=dict(assigner=dict(type='HungarianAssigner3D',
                                 match_costs=[
                                     dict(type='BinaryFocalLossCost',
                                          weight=1.0),
                                     dict(type='BBox3DL1Cost', weight=2.0),
                                     dict(type='IoU3DCost', weight=2.0)
                                 ]), ),
    test_cfg=None)

dataset_type = 'MultiView3DGroundingDataset'
data_root = '/cluster/nvme9a/embodied_data/'

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='MultiViewPipeline',
         n_images=20,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(480, 480), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH'),
    dict(type='PointSample', num_points=n_points),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.087266, 0.087266],
         scale_ratio_range=[.9, 1.1],
         translation_std=[.1, .1, .1],
         shift_height=False),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='MultiViewPipeline',
         n_images=50,
         ordered=True,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(480, 480), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH'),
    dict(type='PointSample', num_points=n_points),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# TODO: to determine a reasonable batch size
train_dataloader = dict(
    batch_size=6,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='RepeatDataset',
                 times=1,
                 dataset=dict(type=dataset_type,
                              data_root=data_root,
                              ann_file='embodiedscan_infos_train.pkl',
                              vg_file='embodiedscan_train_mini_vg_gene_gpt35v2_10k_wcomplex_fixed_tp.json',
                              metainfo=metainfo, 
                              pipeline=train_pipeline,
                              test_mode=False,
                              filter_empty_gt=True,
                              box_type_3d='Euler-Depth',
                            #   share_serialize_data=True,
                            #   tokens_positive_rebuild=True,
                            #   load_data_shared_memory=False,
                              )))

val_dataloader = dict(batch_size=1,
                      num_workers=1,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='embodiedscan_infos_val.pkl',
                                   vg_file='embodiedscan_val_mini_vg.json',
                                   metainfo=metainfo,
                                   pipeline=test_pipeline,
                                   test_mode=True,
                                   filter_empty_gt=True,
                                   box_type_3d='Euler-Depth'))

test_dataloader = dict(batch_size=1,
                       num_workers=1,
                       persistent_workers=True,
                       drop_last=False,
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       dataset=dict(type=dataset_type,
                                    data_root=data_root,
                                    ann_file='embodiedscan_infos_test.pkl',
                                    vg_file='embodiedscan_test_vg.json',
                                    metainfo=metainfo,
                                    pipeline=test_pipeline,
                                    test_mode=True,
                                    filter_empty_gt=True,
                                    box_type_3d='Euler-Depth'))

val_evaluator = dict(type='GroundingMetric')
test_evaluator = dict(type='GroundingMetric', format_only=True)

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
lr = 5e-4
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0005),
                     paramwise_cfg=dict(
                         custom_keys={
                             'text_encoder': dict(lr_mult=0.0),
                             'decoder': dict(lr_mult=0.1, decay_mult=1.0)
                         }),
                     clip_grad=dict(max_norm=10, norm_type=2))

# learning rate
param_scheduler = dict(type='MultiStepLR',
                       begin=0,
                       end=12,
                       by_epoch=True,
                       milestones=[8, 11],
                       gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

# vis_backends = [
#     dict(type='TensorboardVisBackend'),
#     dict(type='LocalVisBackend')
# ]
# visualizer = dict(
#     type='Det3DLocalVisualizer',
#     vis_backends=vis_backends, name='visualizer')


visualizer = dict(
    _scope_='mmdet3d',
    type='Det3DLocalVisualizer',
    vis_backends = [
        dict(type='LocalVisBackend')
    ]
)

find_unused_parameters = True
load_from = 'ckpt/multiview-detection-resnet50-8gpu-bs4-baselr-cbgs/epoch_12.pth'  # noqa
