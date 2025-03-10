# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from ...structures.bbox_3d import xywhr2xyxyr
from mmdet3d.utils import ConfigType
from mmdet3d.structures import bbox3d_mapping_back
from mmdet3d.models.layers import nms_bev, nms_normal_bev


def merge_aug_bboxes_3d(aug_results: List[dict],
                        aug_batch_input_metas: List[dict],
                        test_cfg: ConfigType = dict()) -> dict:
    """Merge augmented detection 3D bboxes and scores.

    Args:
        aug_results (List[dict]): The dict of detection results.
            The dict contains the following keys

            - bbox_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (Tensor): Detection scores.
            - labels_3d (Tensor): Predicted box labels.
        aug_batch_input_metas (List[dict]): Meta information of each sample.
        test_cfg (dict or :obj:`ConfigDict`): Test config.

    Returns:
        dict: Bounding boxes results in cpu mode, containing merged results.

            - bbox_3d (:obj:`BaseInstance3DBoxes`): Merged detection bbox.
            - scores_3d (torch.Tensor): Merged detection scores.
            - labels_3d (torch.Tensor): Merged predicted box labels.
    """

    assert len(aug_results) == len(aug_batch_input_metas), \
        '"aug_results" should have the same length as ' \
        f'"aug_batch_input_metas", got len(aug_results)={len(aug_results)} ' \
        f'and len(aug_batch_input_metas)={len(aug_batch_input_metas)}'
    recovered_bboxes = []
    recovered_scores = []

    for bboxes, input_info in zip(aug_results, aug_batch_input_metas):
        scale_factor = input_info['pcd_scale_factor']
        pcd_horizontal_flip = input_info['pcd_horizontal_flip']
        pcd_vertical_flip = input_info['pcd_vertical_flip']
        recovered_scores.append(bboxes['scores_3d'])
        tmp_bboxes = bbox3d_mapping_back(bboxes['bboxes_3d'], scale_factor,
                                     pcd_horizontal_flip, pcd_vertical_flip)
        recovered_bboxes.append(tmp_bboxes)

    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
    aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
    aug_scores = torch.cat(recovered_scores, dim=0)

    # TODO: use a more elegent way to deal with nms
    if test_cfg.get('use_rotate_nms', False):
        nms_func = nms_bev
    else:
        nms_func = nms_normal_bev

    nms_thr = test_cfg.get('nms_thr', 0.5)
    selected = nms_func(aug_bboxes_for_nms, aug_scores, nms_thr)

    merged_bboxes = aug_bboxes[selected, :]
    merged_scores = aug_scores[selected]

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.get('max_num', 500), len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]

    return dict(bboxes_3d=merged_bboxes, scores_3d=merged_scores)