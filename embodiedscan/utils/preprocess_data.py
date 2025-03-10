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

class MultiViewPipelinePreprocess(BaseTransform):
    """Multiview data processing pipeline.

    The transform steps are as follows:

        1. Select frames.
        2. Re-ororganize the selected data structure.
        3. Apply transforms for each selected frame.
        4. Concatenate data to form a batch.

    Args:
        transforms (list[dict | callable]):
            The transforms to be applied to each select frame.
        n_images (int): Number of frames selected per scene.
        ordered (bool): Whether to put these frames in order.
            Defaults to False.
    """

    def __init__(self, transforms, n_images=None, ordered=True, path=None):
        super().__init__()
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.ordered = ordered
        self.path = path

    def transform(self, results: dict):
        """Transform function.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """


        if self.path is not None:
            scan_id = results['scan_id'].split('/')
            scan_id = '_'.join(scan_id)
            save_file = osp.join(self.path, f'{scan_id}.pkl')
            if osp.exists(save_file):
                return None

        imgs = []
        img_paths = []
        points = []
        intrinsics = []
        extrinsics = []
        ids = np.arange(len(results['img_path']))
        print(len(ids))
        

        # results['depth_img'] = depth_img
        # results['img'] = img
        # results['img_shape'] = img.shape[:2]
        # results['ori_shape'] = img.shape[:2]

        replace = True if self.n_images > len(ids) else False
        step = (len(ids) - 1) // (self.n_images - 1
                                    )  # TODO: BUG, fix from branch fbocc
        if step > 0:
            ids = ids[::step]
            # sometimes can not get the accurate n_images in this way
            # then take the first n_images one
            ids = ids[:self.n_images]
        else:
            ids = ids[:self.n_images]
    
        # print('modified keys in results')
        new_dict = dict()
        for i in ids.tolist():
            _results = dict()
            _results['img_path'] = results['img_path'][i]
            if 'depth_img_path' in results:
                _results['depth_img_path'] = results['depth_img_path'][i]
                if isinstance(results['depth_cam2img'], list):
                    _results['depth_cam2img'] = results['depth_cam2img'][i]
                    _results['cam2img'] = results['depth2img']['intrinsic'][i]
                else:
                    _results['depth_cam2img'] = results['depth_cam2img']
                    _results['cam2img'] = results['cam2img']
                _results['depth_shift'] = results['depth_shift']
            _results = self.transforms(_results)
            
            if 'depth_shift' in _results:
                _results.pop('depth_shift')
            if 'img' in _results:
                imgs.append(_results['img'])
                img_paths.append(_results['img_path'])
            if 'points' in _results:
                points.append(_results['points'])
            if isinstance(results['depth2img']['intrinsic'], list):
                intrinsics.append(results['depth2img']['intrinsic'][i])
            else:
                intrinsics.append(results['depth2img']['intrinsic'])
            extrinsics.append(results['depth2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'points', 'img_path']:
                # print(key)
                new_dict[key] = _results[key]
                results[key] = _results[key]
        if len(imgs):
            # print('img')
            # print('img_path')
            results['img'] = imgs
            results['img_path'] = img_paths
            new_dict['img'] = imgs
            new_dict['img_path'] = img_paths
        if len(points):
            # print('points')
            results['points'] = points
            new_dict['points'] = points
        if 'visible_instance_masks' in results:
            # print('visible_instance_masks')
            results['visible_instance_masks'] = [
                results['visible_instance_masks'][i] for i in ids
            ]
            new_dict['visible_instance_masks'] = [
                results['visible_instance_masks'][i] for i in ids
            ]
        if 'visible_occupancy_masks' in results:
            # print('visible_occupancy_masks')
            results['visible_occupancy_masks'] = [
                results['visible_occupancy_masks'][i] for i in ids
            ]
            new_dict['visible_occupancy_masks'] = [
                results['visible_occupancy_masks'][i] for i in ids
            ]
        # print('depth2img-intrinsic')
        # print('depth2img-extrinsic')
        results['depth2img']['intrinsic'] = intrinsics
        results['depth2img']['extrinsic'] = extrinsics
        new_dict['depth2img'] = results['depth2img']
        new_dict['scan_id'] = results['scan_id']
        return new_dict

save_pipeline = [
    dict(type='LoadAnnotations3D'),
    MultiViewPipelinePreprocess(
         n_images=200,
         ordered=True,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(480, 480), keep_ratio=False)
         ]),
]
        

metainfo = dict(classes='all_from_info')
dataset_type = 'MultiView3DGroundingDataset'
dataset_config=dict(type=dataset_type,
            data_root=data_root,
            ann_file='embodiedscan_infos_train_val_referit_scanrefer.pkl',
            vg_file='embodiedscan_referit_scanrefer_train_val_for_preprocessing.json',
            metainfo=metainfo,
            pipeline=save_pipeline,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Euler-Depth')
            
dataset = DATASETS.build(dataset_config)

from tqdm import tqdm
import pdb; pdb.set_trace()
save_path = '/cluster/nvme4b/zjh/preprocessed_data_referit'
import time
start = time.time()

from glob import glob
import mmengine
import os
files_in_path = glob(save_path+'/*.pkl')
files_in_path = [os.path.basename(file)[:-4] for file in files_in_path]
scans2process = mmengine.load('/cluster/home2/zjh/EmbodiedScan/data/embodiedscan_referit_scanrefer_train_val_for_preprocessing.json')
scans_ids = ['_'.join(scan['scan_id'].split('/')) for scan in scans2process]
indices = [i for i, scan in enumerate(scans_ids) if scan not in files_in_path]

for idx in tqdm(indices):
    data = dataset[idx]
    # print(time.time()-start, f'img_count:{data}')
    # start = time.time()
    if data is None:
        continue
    scan_id = data['scan_id'].split('/')
    scan_id = '_'.join(scan_id)
    save_file = osp.join(save_path, f'{scan_id}.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
