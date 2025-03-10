import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose
from typing import Optional, Tuple, Union
import os
import mmengine

from embodiedscan.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PointsToGPU(BaseTransform):
    """Point sample.

    Sampling data to a certain number.

    Required Keys:

    - points
    - pts_instance_mask (optional)
    - pts_semantic_mask (optional)

    Modified Keys:

    - points
    - pts_instance_mask (optional)
    - pts_semantic_mask (optional)

    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool): Whether the sampling is with or without replacement.
            Defaults to False.
    """

    def __init__(self) -> None:
        pass

    def transform(self, input_dict: dict) -> dict:
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        input_dict['points'].tensor = input_dict['points'].tensor.cuda()
        return input_dict
        

@TRANSFORMS.register_module()
class SavingPreprocessData(BaseTransform):
    def __init__(self, save_dir='data') -> None:
        self.save_dir = save_dir

    def transform(self, input_dict: dict) -> dict:
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        return 0
        # dataset = input_dict['scan_id'].split('/')[0]
        os.makedirs(self.save_dir, exist_ok=True)
        filename = '_'.join(input_dict['scan_id'].split('/')[1:]) + '.pkl'
        save_path = os.path.join(self.save_dir, filename)
        # mmengine.dump(input_dict, save_path)
        if not os.path.exists(save_path):
            mmengine.dump(input_dict, save_path)
            print(f"Saved preprocessed data to {save_path}")
            return 1

        print(f"File already exists at {save_path}")
        return save_path


    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(saves point clouds)'
        return repr_str