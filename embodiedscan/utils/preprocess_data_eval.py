from embodiedscan.registry import DATASETS
from mmengine import DefaultScope
from mmcv.transforms import BaseTransform, Compose
import numpy as np
import pickle
import os.path as osp

default_scope = DefaultScope.get_instance('embodiedscan', scope_name='embodiedscan')
# data_root = '/cluster/nvme4b/embodied_data/'
data_root = '/cluster/home2/zjh/EmbodiedScan/data'
n_points = 100000
backend_args = None

save_pipeline = [
    dict(type='LoadAnnotations3D'),
]

metainfo = dict(classes='all_from_info')
dataset_type = 'MultiView3DGroundingDataset'
dataset_config=dict(type=dataset_type,
            data_root=data_root,
            ann_file='embodiedscan_infos_test_referit_scanrefer.pkl',
            vg_file='sr3d_test_vg_esstyle_orig.json',
            metainfo=metainfo,
            pipeline=save_pipeline,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Euler-Depth')
            
dataset = DATASETS.build(dataset_config)

from tqdm import tqdm
from glob import glob
import mmengine
import os
import copy

eval_anno = []
for idx in tqdm(range(len(dataset))):
    data = dataset[idx]
    data_copy = copy.deepcopy(data)
    data_copy.pop('img_path')
    data_copy.pop('depth_img_path')
    data_copy['gt_bboxes_3d'] = data_copy['gt_bboxes_3d'].numpy()
    data_copy['ann_info']['gt_bboxes_3d'] = data_copy['ann_info']['gt_bboxes_3d'].numpy()
    eval_anno.append(data_copy)
mmengine.dump(eval_anno, os.path.join(data_root, 'sr3d_test_vg_esstyle_orig_gt.json'))
