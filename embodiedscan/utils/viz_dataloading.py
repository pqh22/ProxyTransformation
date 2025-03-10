from embodiedscan.registry import DATASETS
from mmengine import DefaultScope
from mmcv.transforms import BaseTransform, Compose
import numpy as np
import pickle
import os.path as osp

default_scope = DefaultScope.get_instance('embodiedscan', scope_name='embodiedscan')
dataset_type = 'MultiView3DGroundingDataset'
data_root = '/cluster/nvme4b/embodied_data/'
n_points = 100000
backend_args = None

visualization_pipeline = [
    dict(type='LoadAnnotations3D', _scope_='embodiedscan'),
    dict(type='MultiViewPipeline',
         n_images=20,
         ordered=False,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
             dict(type='PointSample', num_points=n_points // 10),
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH'),
]

metainfo = dict(classes='all')
dataset_config=dict(type=dataset_type,
            data_root=data_root,
            ann_file='embodiedscan_infos_train_val.pkl',
            vg_file='embodiedscan_train_mini_vg_4samples_fordebug.json',
            metainfo=metainfo,
            pipeline=visualization_pipeline,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Euler-Depth')
            
dataset = DATASETS.build(dataset_config)
from tqdm import tqdm
for idx in tqdm(range(len(dataset))):
    data = dataset[idx]
