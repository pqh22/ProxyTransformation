# Copyright (c) OpenMMLab and OpenRobotLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/dense_heads/fcaf3d_neck_with_head.py # noqa
from typing import List, Optional, Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow get_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

import torch
from mmcv.cnn import Scale
from mmcv.ops import nms3d, nms3d_normal
from mmengine.model import BaseModel, bias_init_with_prob
from mmengine.structures import InstanceData
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from torch import Tensor, nn

from embodiedscan.models.losses import BBoxCDLoss, RotatedIoU3DLoss
from embodiedscan.registry import MODELS
from embodiedscan.structures import (BaseInstance3DBoxes, rotation_3d_in_axis,
                                     rotation_3d_in_euler)
from embodiedscan.utils.dist_utils import reduce_mean
from embodiedscan.utils.typing_config import InstanceList, SampleList


@MODELS.register_module()
class FCAF3DHead(BaseModel):
    r"""Bbox head of `FCAF3D <https://arxiv.org/abs/2112.00322>`_.

    Actually here we store both the sparse 3D FPN and a head. The neck and
    the head can not be simply separated as pruning score on the i-th level
    of FPN requires classification scores from i+1-th level of the head.

    Args:
        num_classes (int): Number of classes.
        in_channels (tuple(int)): Number of channels in input tensors.
        out_channels (int): Number of channels in the neck output tensors.
        num_reg_outs (int): Number of regression layer channels.
        voxel_size (float): Voxel size in meters.
        pts_prune_threshold (int): Pruning threshold on each feature level.
        pts_assign_threshold (int): Box to location assigner parameter.
            Assigner selects the maximum feature level with more locations
            inside the box than pts_assign_threshold.
        pts_center_threshold (int): Box to location assigner parameter.
            After feature level for the box is determined, assigner selects
            pts_center_threshold locations closest to the box center.
        center_loss (dict): Config of centerness loss. Defaults to
            dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True).
        bbox_loss (dict): Config of bbox loss. Defaults to
            dict(type='AxisAlignedIoULoss').
        cls_loss (dict): Config of classification loss. Defaults to
            dict = dict(type='mmdet.FocalLoss').
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Tuple[int],
                 out_channels: int,
                 num_reg_outs: int,
                 voxel_size: float,
                 pts_prune_threshold: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 center_loss: dict = dict(type='mmdet.CrossEntropyLoss',
                                          use_sigmoid=True),
                 bbox_loss: dict = dict(type='AxisAlignedIoULoss'),
                 cls_loss: dict = dict(type='mmdet.FocalLoss'),
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(FCAF3DHead, self).__init__(init_cfg)
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.`')
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.center_loss = MODELS.build(center_loss)
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, out_channels, num_reg_outs, num_classes)

    @staticmethod
    def _make_block(in_channels: int, out_channels: int) -> nn.Module:
        """Construct Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU())

    @staticmethod
    def _make_up_block(in_channels: int, out_channels: int) -> nn.Module:
        """Construct DeConv-Norm-Act-Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels,
                                                       out_channels,
                                                       kernel_size=2,
                                                       stride=2,
                                                       dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU())

    def _init_layers(self, in_channels: Tuple[int], out_channels: int,
                     num_reg_outs: int, num_classes: int):
        """Initialize layers.

        Args:
            in_channels (tuple[int]): Number of channels in input tensors.
            out_channels (int): Number of channels in the neck output tensors.
            num_reg_outs (int): Number of regression layer channels.
            num_classes (int): Number of classes.
        """
        # neck layers
        self.pruning = ME.MinkowskiPruning()
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self._make_up_block(in_channels[i], in_channels[i - 1]))
            self.__setattr__(f'out_block_{i}',
                             self._make_block(in_channels[i], out_channels))

        # head layers
        self.conv_center = ME.MinkowskiConvolution(out_channels,
                                                   1,
                                                   kernel_size=1,
                                                   dimension=3)
        self.conv_reg = ME.MinkowskiConvolution(out_channels,
                                                num_reg_outs,
                                                kernel_size=1,
                                                dimension=3)
        self.conv_cls = ME.MinkowskiConvolution(out_channels,
                                                num_classes,
                                                kernel_size=1,
                                                bias=True,
                                                dimension=3)
        self.scales = nn.ModuleList(
            [Scale(1.) for _ in range(len(in_channels))])

    def init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.conv_center.kernel, std=.01)
        nn.init.normal_(self.conv_reg.kernel, std=.01)
        nn.init.normal_(self.conv_cls.kernel, std=.01)
        nn.init.constant_(self.conv_cls.bias, bias_init_with_prob(.01))

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Args:
            x (list[Tensor]): Features from the backbone.

        Returns:
            Tuple[List[Tensor], ...]: Predictions of the head.
        """
        center_preds, bbox_preds, cls_preds, points = [], [], [], []
        inputs = x
        x = inputs[-1]
        prune_score = None
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self._prune(x, prune_score)

            out = self.__getattr__(f'out_block_{i}')(x)
            center_pred, bbox_pred, cls_pred, point, prune_score = \
                self._forward_single(out, self.scales[i])
            center_preds.append(center_pred)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return center_preds[::-1], bbox_preds[::-1], cls_preds[::-1], \
            points[::-1]

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (batch_gt_instances_3d, batch_input_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)
        predictions = self.predict_by_feat(*outs,
                                           batch_input_metas=batch_input_metas,
                                           rescale=rescale)
        return predictions

    def _prune(self, x: SparseTensor, scores: SparseTensor) -> SparseTensor:
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    def _forward_single(self, x: SparseTensor,
                        scale: Scale) -> Tuple[Tensor, ...]:
        """Forward pass per level.

        Args:
            x (SparseTensor): Per level neck output tensor.
            scale (mmcv.cnn.Scale): Per level multiplication weight.

        Returns:
            tuple[Tensor]: Per level head predictions.
        """
        center_pred = self.conv_center(x).features
        scores = self.conv_cls(x)
        cls_pred = scores.features
        prune_scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        reg_final = self.conv_reg(x).features
        reg_distance = torch.exp(scale(reg_final[:, :6])).clamp(min=1e-3)
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        center_preds, bbox_preds, cls_preds, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            center_preds.append(center_pred[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        return center_preds, bbox_preds, cls_preds, points, prune_scores

    def _loss_by_feat_single(self, center_preds: List[Tensor],
                             bbox_preds: List[Tensor], cls_preds: List[Tensor],
                             points: List[Tensor],
                             gt_bboxes: BaseInstance3DBoxes, gt_labels: Tensor,
                             input_meta: dict) -> Tuple[Tensor, ...]:
        """Loss function of single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor, ...]: Centerness, bbox, and classification loss
            values.
        """
        center_targets, bbox_targets, cls_targets = self.get_targets(
            points, gt_bboxes, gt_labels)

        center_preds = torch.cat(center_preds)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        pos_inds = torch.nonzero(cls_targets >= 0).squeeze(1)
        n_pos = points.new_tensor(len(pos_inds))
        n_pos = max(reduce_mean(n_pos), 1.)

        # to avoid some corner case in 3RScan
        if len(cls_preds) > 0:
            cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=n_pos)
        else:
            cls_loss = cls_preds.sum()

        # bbox and centerness losses
        pos_center_preds = center_preds[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_center_targets = center_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # reduce_mean is outside if / else block to prevent deadlock
        center_denorm = max(reduce_mean(pos_center_targets.sum().detach()),
                            1e-6)
        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            center_loss = self.center_loss(pos_center_preds,
                                           pos_center_targets,
                                           avg_factor=n_pos)

            decode_bbox_preds = self._bbox_pred_to_bbox(
                pos_points, pos_bbox_preds)
            if decode_bbox_preds.shape[-1] > 7 and isinstance(
                    self.bbox_loss, RotatedIoU3DLoss):
                decode_bbox_preds = decode_bbox_preds[..., :7]
                pos_bbox_targets = pos_bbox_targets[..., :7]

            bbox_loss = self.bbox_loss(decode_bbox_preds,
                                       pos_bbox_targets,
                                       weight=pos_center_targets,
                                       avg_factor=center_denorm)
        else:
            center_loss = pos_center_preds.sum()
            bbox_loss = pos_bbox_preds.sum()
        return center_loss, bbox_loss, cls_loss

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     bbox_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     points: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_gt_instances_ignore: Optional[InstanceList] = None,
                     **kwargs) -> dict:
        """Loss function about feature.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
                The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each input,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: Centerness, bbox, and classification losses.
        """
        center_losses, bbox_losses, cls_losses = [], [], []
        for i in range(len(batch_input_metas)):
            center_loss, bbox_loss, cls_loss = self._loss_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=batch_input_metas[i],
                gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                gt_labels=batch_gt_instances_3d[i].labels_3d)
            center_losses.append(center_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(loss_center=torch.mean(torch.stack(center_losses)),
                    loss_bbox=torch.mean(torch.stack(bbox_losses)),
                    loss_cls=torch.mean(torch.stack(cls_losses)))

    def _predict_by_feat_single(self, center_preds: List[Tensor],
                                bbox_preds: List[Tensor],
                                cls_preds: List[Tensor], points: List[Tensor],
                                input_meta: dict) -> InstanceData:
        """Generate boxes for single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            input_meta (dict): Scene meta info.

        Returns:
            InstanceData: Predicted bounding boxes, scores and labels.
        """
        mlvl_bboxes, mlvl_scores = [], []
        for center_pred, bbox_pred, cls_pred, point in zip(
                center_preds, bbox_preds, cls_preds, points):
            scores = cls_pred.sigmoid() * center_pred.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, input_meta)

        bboxes = input_meta['box_type_3d'](bboxes,
                                           box_dim=bboxes.shape[1],
                                           with_yaw=bboxes.shape[1] == 7,
                                           origin=(.5, .5, .5))

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        return results

    def predict_by_feat(self, center_preds: List[List[Tensor]],
                        bbox_preds: List[List[Tensor]], cls_preds,
                        points: List[List[Tensor]],
                        batch_input_metas: List[dict],
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[InstanceData]: Predicted bboxes, scores, and labels for
            all scenes.
        """
        results = []
        for i in range(len(batch_input_metas)):
            result = self._predict_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=batch_input_metas[i])
            results.append(result)
        return results

    @staticmethod
    def _bbox_to_loss(bbox: Tensor) -> Tensor:
        """Transform box to the axis-aligned or rotated iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points: Tensor, bbox_pred: Tensor) -> Tensor:
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
            y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
            z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

            # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max-> x, y, z, w, l, h
            base_bbox = torch.stack([
                x_center,
                y_center,
                z_center,
                bbox_pred[:, 0] + bbox_pred[:, 1],
                bbox_pred[:, 2] + bbox_pred[:, 3],
                bbox_pred[:, 4] + bbox_pred[:, 5],
            ], -1)
            return base_bbox
        """ original implementation for rotated boxes
        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
            bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)
        """
        # for rotated boxes (7-DoF or 9-DoF)
        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, alpha ->
        # x_center, y_center, z_center, w, l, h, alpha
        shift = torch.stack(((bbox_pred[:, 1] - bbox_pred[:, 0]) / 2,
                             (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2,
                             (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2),
                            dim=-1).view(-1, 1, 3)  # (N, 1, 3)
        if bbox_pred.shape[-1] == 7:
            shift = rotation_3d_in_axis(shift, bbox_pred[:, 6], axis=2)[:,
                                                                        0, :]
        else:
            shift = rotation_3d_in_euler(shift, bbox_pred[:, 6:])[:, 0, :]
        center = points + shift
        size = torch.stack(
            (bbox_pred[:, 0] + bbox_pred[:, 1], bbox_pred[:, 2] +
             bbox_pred[:, 3], bbox_pred[:, 4] + bbox_pred[:, 5]),
            dim=-1)
        return torch.cat((center, size, bbox_pred[:, 6:]), dim=-1)

    @staticmethod
    def _get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
            (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        shift = torch.stack(
            (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
             points[..., 2] - boxes[..., 2]),
            dim=-1).permute(1, 0, 2)
        if boxes.shape[-1] == 7:
            shift = rotation_3d_in_axis(shift, -boxes[0, :, 6],
                                        axis=2).permute(1, 0, 2)
        else:
            shift = rotation_3d_in_euler(shift,
                                         -boxes[0, :, 6:]).permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)

    @staticmethod
    def _get_centerness(face_distances: Tensor) -> Tensor:
        """Compute point centerness w.r.t containing box.

        Args:
            face_distances (Tensor): Face distances of shape (B, N, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

        Returns:
            Tensor: Centerness of shape (B, N).
        """
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
            y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
            z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)

    @torch.no_grad()
    def get_targets(self, points: Tensor, gt_bboxes: BaseInstance3DBoxes,
                    gt_labels: Tensor) -> Tuple[Tensor, ...]:
        """Compute targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.

        Returns:
            tuple[Tensor, ...]: Centerness, bbox and classification
            targets for all locations.
        """
        float_max = points[0].new_tensor(1e8)
        n_levels = len(points)
        levels = torch.cat([
            points[i].new_tensor(i).expand(len(points[i]))
            for i in range(len(points))
        ])
        points = torch.cat(points)
        gt_bboxes = gt_bboxes.to(points.device)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if n_boxes == 0:
            # return pseudo center_targets, bbox_targets and cls_targets
            return gt_bboxes.tensor.new_zeros((n_points,)), \
                gt_bboxes.tensor.new_zeros((n_points, gt_bboxes.shape[-1])), \
                gt_labels.new_full((n_points,), -1)

        n_dims = gt_bboxes.tensor.shape[-1]
        volumes = gt_bboxes.volume.unsqueeze(0).expand(n_points, n_boxes)

        # condition 1: point inside box
        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                          dim=1)
        boxes = boxes.expand(n_points, n_boxes, n_dims)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        face_distances = self._get_face_distances(points, boxes)
        inside_box_condition = face_distances.min(dim=-1).values > 0

        # condition 2: positive points per level >= limit
        # calculate positive points per scale
        n_pos_points_per_level = []
        for i in range(n_levels):
            n_pos_points_per_level.append(
                torch.sum(inside_box_condition[levels == i], dim=0))
        # find best level
        n_pos_points_per_level = torch.stack(n_pos_points_per_level, dim=0)
        lower_limit_mask = n_pos_points_per_level < self.pts_assign_threshold
        lower_index = torch.argmax(lower_limit_mask.int(), dim=0) - 1
        lower_index = torch.where(lower_index < 0, 0, lower_index)
        all_upper_limit_mask = torch.all(torch.logical_not(lower_limit_mask),
                                         dim=0)
        best_level = torch.where(all_upper_limit_mask, n_levels - 1,
                                 lower_index)
        # keep only points with best level
        best_level = best_level.expand(n_points, n_boxes)
        levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = best_level == levels

        # condition 3: limit topk points per box by centerness
        centerness = self._get_centerness(face_distances)
        centerness = torch.where(inside_box_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        centerness = torch.where(level_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness,
                                    min(self.pts_center_threshold + 1,
                                        len(centerness)),
                                    dim=0).values[-1]
        topk_condition = centerness > top_centerness.unsqueeze(0)

        # condition 4: min volume box per point
        volumes = torch.where(inside_box_condition, volumes, float_max)
        volumes = torch.where(level_condition, volumes, float_max)
        volumes = torch.where(topk_condition, volumes, float_max)
        min_volumes, min_inds = volumes.min(dim=1)

        center_targets = centerness[torch.arange(n_points), min_inds]
        bbox_targets = boxes[torch.arange(n_points), min_inds]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = gt_labels[min_inds]
        cls_targets = torch.where(min_volumes == float_max, -1, cls_targets)
        return center_targets, bbox_targets, cls_targets

    def _single_scene_multiclass_nms(self, bboxes: Tensor, scores: Tensor,
                                     input_meta: dict) -> Tuple[Tensor, ...]:
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor, ...]: Predicted bboxes, scores and labels.
        """
        num_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] >= 7
        if bboxes.shape[-1] == 9:
            bboxes = bboxes[..., :7]  # only consider yaw during nms
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(num_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if with_yaw:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)

            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(class_scores[nms_ids].shape,
                                i,
                                dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if bboxes.shape[-1] < 9:
            if with_yaw:
                box_dim = 7
            else:
                box_dim = 6
                nms_bboxes = nms_bboxes[:, :box_dim]

        return nms_bboxes, nms_scores, nms_labels


@MODELS.register_module()
class FCAF3DHeadRotMat(BaseModel):
    r"""FCAF3D head with a 6D representation for rotation of boxes.

    Actually here we store both the sparse 3D FPN and a head. The neck and
    the head can not be simply separated as pruning score on the i-th level
    of FPN requires classification scores from i+1-th level of the head.

    Args:
        num_classes (int): Number of classes.
        in_channels (tuple(int)): Number of channels in input tensors.
        out_channels (int): Number of channels in the neck output tensors.
        num_reg_outs (int): Number of regression layer channels.
        voxel_size (float): Voxel size in meters.
        pts_prune_threshold (int): Pruning threshold on each feature level.
        pts_assign_threshold (int): Box to location assigner parameter.
            Assigner selects the maximum feature level with more locations
            inside the box than pts_assign_threshold.
        pts_center_threshold (int): Box to location assigner parameter.
            After feature level for the box is determined, assigner selects
            pts_center_threshold locations closest to the box center.
        center_loss (dict): Config of centerness loss. Defaults to
            dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True).
        bbox_loss (dict): Config of bbox loss. Defaults to
            dict(type='BBoxCDLoss', mode='l1', loss_weight=1.0, group='g8').
        cls_loss (dict): Config of classification loss. Defaults to
            dict = dict(type='mmdet.FocalLoss').
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Tuple[int],
                 out_channels: int,
                 num_reg_outs: int,
                 voxel_size: float,
                 pts_prune_threshold: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 center_loss: dict = dict(type='mmdet.CrossEntropyLoss',
                                          use_sigmoid=True),
                 bbox_loss: dict = dict(type='BBoxCDLoss',
                                        mode='l1',
                                        loss_weight=1.0,
                                        group='g8'),
                 cls_loss: dict = dict(type='mmdet.FocalLoss'),
                 decouple_bbox_loss: bool = False,
                 decouple_groups: int = 3,
                 decouple_weights: Optional[list] = None,
                 norm_decouple_loss: bool = False,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(FCAF3DHeadRotMat, self).__init__(init_cfg)
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.`')
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.center_loss = MODELS.build(center_loss)
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)
        self.decouple_bbox_loss = decouple_bbox_loss
        self.decouple_groups = decouple_groups
        self.norm_decouple_loss = norm_decouple_loss
        if decouple_weights is None:
            self.decouple_weights = [
                1.0 / self.decouple_groups for _ in range(self.decouple_groups)
            ]
        else:
            self.decouple_weights = decouple_weights
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, out_channels, num_reg_outs, num_classes)

    @staticmethod
    def _make_block(in_channels: int, out_channels: int) -> nn.Module:
        """Construct Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU())

    @staticmethod
    def _make_up_block(in_channels: int, out_channels: int) -> nn.Module:
        """Construct DeConv-Norm-Act-Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels,
                                                       out_channels,
                                                       kernel_size=2,
                                                       stride=2,
                                                       dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU())

    def _init_layers(self, in_channels: Tuple[int], out_channels: int,
                     num_reg_outs: int, num_classes: int):
        """Initialize layers.

        Args:
            in_channels (tuple[int]): Number of channels in input tensors.
            out_channels (int): Number of channels in the neck output tensors.
            num_reg_outs (int): Number of regression layer channels.
            num_classes (int): Number of classes.
        """
        # neck layers
        self.pruning = ME.MinkowskiPruning()
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self._make_up_block(in_channels[i], in_channels[i - 1]))
            self.__setattr__(f'out_block_{i}',
                             self._make_block(in_channels[i], out_channels))

        # head layers
        self.conv_center = ME.MinkowskiConvolution(out_channels,
                                                   1,
                                                   kernel_size=1,
                                                   dimension=3)
        self.conv_reg = ME.MinkowskiConvolution(out_channels,
                                                num_reg_outs,
                                                kernel_size=1,
                                                dimension=3)
        self.conv_cls = ME.MinkowskiConvolution(out_channels,
                                                num_classes,
                                                kernel_size=1,
                                                bias=True,
                                                dimension=3)
        self.scales = nn.ModuleList(
            [Scale(1.) for _ in range(len(in_channels))])

    def init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.conv_center.kernel, std=.01)
        nn.init.normal_(self.conv_reg.kernel, std=.01)
        nn.init.normal_(self.conv_cls.kernel, std=.01)
        nn.init.constant_(self.conv_cls.bias, bias_init_with_prob(.01))

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Args:
            x (list[Tensor]): Features from the backbone.

        Returns:
            Tuple[List[Tensor], ...]: Predictions of the head.
        """
        center_preds, bbox_preds, cls_preds, points = [], [], [], []
        inputs = x
        x = inputs[-1]
        prune_score = None
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self._prune(x, prune_score)

            out = self.__getattr__(f'out_block_{i}')(x)
            center_pred, bbox_pred, cls_pred, point, prune_score = \
                self._forward_single(out, self.scales[i])
            center_preds.append(center_pred)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return center_preds[::-1], bbox_preds[::-1], cls_preds[::-1], \
            points[::-1]

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (batch_gt_instances_3d, batch_input_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)
        predictions = self.predict_by_feat(*outs,
                                           batch_input_metas=batch_input_metas,
                                           rescale=rescale)
        return predictions

    def _prune(self, x: SparseTensor, scores: SparseTensor) -> SparseTensor:
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    def _forward_single(self, x: SparseTensor,
                        scale: Scale) -> Tuple[Tensor, ...]:
        """Forward pass per level.

        Args:
            x (SparseTensor): Per level neck output tensor.
            scale (mmcv.cnn.Scale): Per level multiplication weight.

        Returns:
            tuple[Tensor]: Per level head predictions.
        """
        center_pred = self.conv_center(x).features
        scores = self.conv_cls(x)
        cls_pred = scores.features
        prune_scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        reg_final = self.conv_reg(x).features
        reg_distance = torch.exp(scale(reg_final[:, :6])).clamp(min=1e-3)
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        center_preds, bbox_preds, cls_preds, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            center_preds.append(center_pred[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        return center_preds, bbox_preds, cls_preds, points, prune_scores

    def _loss_by_feat_single(self, center_preds: List[Tensor],
                             bbox_preds: List[Tensor], cls_preds: List[Tensor],
                             points: List[Tensor],
                             gt_bboxes: BaseInstance3DBoxes, gt_labels: Tensor,
                             input_meta: dict) -> Tuple[Tensor, ...]:
        """Loss function of single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor, ...]: Centerness, bbox, and classification loss
            values.
        """
        center_targets, bbox_targets, cls_targets = self.get_targets(
            points, gt_bboxes, gt_labels)

        center_preds = torch.cat(center_preds)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        pos_inds = torch.nonzero(cls_targets >= 0).squeeze(1)
        n_pos = points.new_tensor(len(pos_inds))
        n_pos = max(reduce_mean(n_pos), 1.)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=n_pos)

        # angle loss
        x_raw, y_raw = bbox_preds[pos_inds, 6:9], bbox_preds[pos_inds, 9:12]
        rot_mat_preds = ortho_6d_2_Mat(x_raw, y_raw)
        euler_targets = bbox_targets[pos_inds, 6:]
        rot_mat_targets = euler_angles_to_matrix(euler_targets, 'ZXY')
        if isinstance(self.bbox_loss, BBoxCDLoss):
            # there is a weird coupling here
            # when we are using CD for bbox,
            #   we only care about the rotated vector of (1, 0, 0)
            # that is, the direction of the x-axis
            #   (defining the "left/right" direction of the box)")
            rot_mat_preds = rot_mat_preds[:, :3, :1]
            rot_mat_targets = rot_mat_targets[:, :3, :1]

        # bbox and centerness losses
        pos_center_preds = center_preds[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_center_targets = center_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]

        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            center_loss = self.center_loss(pos_center_preds,
                                           pos_center_targets,
                                           avg_factor=n_pos)

            decode_bbox_preds = self._bbox_pred_to_bbox(
                pos_points, pos_bbox_preds)

            if self.decouple_bbox_loss:
                bbox_targ_center = pos_bbox_targets[:, :3]
                bbox_targ_size = pos_bbox_targets[:, 3:6]
                bbox_targ_euler = pos_bbox_targets[:, 6:]
                bbox_pred_center = decode_bbox_preds[:, :3]
                bbox_pred_size = decode_bbox_preds[:, 3:6]
                bbox_pred_euler = decode_bbox_preds[:, 6:]

            corner_bbox_loss = 0
            if isinstance(self.bbox_loss, BBoxCDLoss):
                if self.decouple_bbox_loss:
                    assert self.decouple_groups in (
                        3, 4
                    ), 'Only support groups=3 or 4 with stable performance.'
                    if self.norm_decouple_loss:
                        corner_bbox_loss += self.decouple_weights[
                            0] * self.bbox_loss(torch.concat(
                                (bbox_pred_center, bbox_targ_size,
                                 bbox_targ_euler),
                                dim=-1),
                                                pos_bbox_targets,
                                                reduction_override='none')
                        corner_bbox_loss += self.decouple_weights[
                            1] * self.bbox_loss(torch.concat(
                                (bbox_targ_center, bbox_pred_size,
                                 bbox_targ_euler),
                                dim=-1),
                                                pos_bbox_targets,
                                                reduction_override='none')
                        corner_bbox_loss += self.decouple_weights[
                            2] * self.bbox_loss(torch.concat(
                                (bbox_targ_center, bbox_targ_size,
                                 bbox_pred_euler),
                                dim=-1),
                                                pos_bbox_targets,
                                                reduction_override='none')
                        bbox_sizes = bbox_targ_size.norm(
                            dim=-1)[:, None].clamp(min=0.1)
                        corner_bbox_loss = (corner_bbox_loss /
                                            bbox_sizes).mean()
                    else:
                        corner_bbox_loss += self.decouple_weights[
                            0] * self.bbox_loss(
                                torch.concat((bbox_pred_center, bbox_targ_size,
                                              bbox_targ_euler),
                                             dim=-1), pos_bbox_targets)
                        corner_bbox_loss += self.decouple_weights[
                            1] * self.bbox_loss(
                                torch.concat((bbox_targ_center, bbox_pred_size,
                                              bbox_targ_euler),
                                             dim=-1), pos_bbox_targets)
                        corner_bbox_loss += self.decouple_weights[
                            2] * self.bbox_loss(
                                torch.concat((bbox_targ_center, bbox_targ_size,
                                              bbox_pred_euler),
                                             dim=-1), pos_bbox_targets)

                    if self.decouple_groups == 4:
                        corner_bbox_loss += self.decouple_weights[
                            3] * self.bbox_loss(decode_bbox_preds,
                                                pos_bbox_targets)

                else:
                    corner_bbox_loss += self.bbox_loss(decode_bbox_preds,
                                                       pos_bbox_targets)

                bbox_loss = corner_bbox_loss

        else:
            center_loss = pos_center_preds.sum()
            bbox_loss = pos_bbox_preds.sum()
        center_nan_mask = torch.isnan(center_loss)
        bbox_nan_mask = torch.isnan(bbox_loss)
        if center_nan_mask.any():
            torch.nan_to_num(center_loss)
            # print("center loss nan, filled with 0")
        if bbox_nan_mask.any():
            torch.nan_to_num(bbox_loss)
            # print("bbox loss nan, filled with 0")
        return center_loss, bbox_loss, cls_loss

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     bbox_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     points: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_gt_instances_ignore: Optional[InstanceList] = None,
                     **kwargs) -> dict:
        """Loss function about feature.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
                The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each input,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: Centerness, bbox, and classification losses.
        """
        center_losses, bbox_losses, cls_losses = [], [], []
        for i in range(len(batch_input_metas)):
            center_loss, bbox_loss, cls_loss = \
                self._loss_by_feat_single(
                    center_preds=[x[i] for x in center_preds],
                    bbox_preds=[x[i] for x in bbox_preds],
                    cls_preds=[x[i] for x in cls_preds],
                    points=[x[i] for x in points],
                    input_meta=batch_input_metas[i],
                    gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                    gt_labels=batch_gt_instances_3d[i].labels_3d
                )
            center_losses.append(center_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(loss_center=torch.mean(torch.stack(center_losses)),
                    loss_bbox=torch.mean(torch.stack(bbox_losses)),
                    loss_cls=torch.mean(torch.stack(cls_losses)))

    def _predict_by_feat_single(self, center_preds: List[Tensor],
                                bbox_preds: List[Tensor],
                                cls_preds: List[Tensor], points: List[Tensor],
                                input_meta: dict) -> InstanceData:
        """Generate boxes for single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            input_meta (dict): Scene meta info.

        Returns:
            InstanceData: Predicted bounding boxes, scores and labels.
        """
        mlvl_bboxes, mlvl_scores = [], []
        for center_pred, bbox_pred, cls_pred, point in zip(
                center_preds, bbox_preds, cls_preds, points):
            scores = cls_pred.sigmoid() * center_pred.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, input_meta)

        bboxes = input_meta['box_type_3d'](bboxes,
                                           box_dim=bboxes.shape[1],
                                           with_yaw=bboxes.shape[1] == 7,
                                           origin=(.5, .5, .5))

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        return results

    def predict_by_feat(self, center_preds: List[List[Tensor]],
                        bbox_preds: List[List[Tensor]], cls_preds,
                        points: List[List[Tensor]],
                        batch_input_metas: List[dict],
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[InstanceData]: Predicted bboxes, scores, and labels for
            all scenes.
        """
        results = []
        for i in range(len(batch_input_metas)):
            result = self._predict_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=batch_input_metas[i])
            results.append(result)
        return results

    @staticmethod
    def _bbox_to_loss(bbox: Tensor) -> Tensor:
        """Transform box to the axis-aligned or rotated iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points: Tensor, bbox_pred: Tensor) -> Tensor:
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
            y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
            z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

            # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max ->x, y, z, w, l, h
            base_bbox = torch.stack([
                x_center,
                y_center,
                z_center,
                bbox_pred[:, 0] + bbox_pred[:, 1],
                bbox_pred[:, 2] + bbox_pred[:, 3],
                bbox_pred[:, 4] + bbox_pred[:, 5],
            ], -1)
            return base_bbox
        """ original implementation for rotated boxes
        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
            bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)
        """
        # for rotated boxes (7-DoF or 9-DoF)
        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, alpha ->
        # x_center, y_center, z_center, w, l, h, alpha
        shift = torch.stack(((bbox_pred[:, 1] - bbox_pred[:, 0]) / 2,
                             (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2,
                             (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2),
                            dim=-1).view(-1, 1, 3)  # (N, 1, 3)
        if bbox_pred.shape[-1] == 7:
            shift = rotation_3d_in_axis(shift, bbox_pred[:, 6], axis=2)[:,
                                                                        0, :]
        elif bbox_pred.shape[-1] == 9:
            shift = rotation_3d_in_euler(shift, bbox_pred[:, 6:])[:, 0, :]
        else:
            assert bbox_pred.shape[-1] == 12, \
                'bbox_pred must be 2D tensor of shape ' + \
                '(N, 6) or (N, 7) or (N, 9) or (N, 12)'
            x_raw, y_raw = bbox_pred[:, 6:9], bbox_pred[:, 9:]
            rot_mat = ortho_6d_2_Mat(x_raw, y_raw)
            euler = matrix_to_euler_angles(rot_mat, 'ZXY')
            shift = rotation_3d_in_euler(shift, euler)[:, 0, :]
        center = points + shift
        size = torch.stack(
            (bbox_pred[:, 0] + bbox_pred[:, 1], bbox_pred[:, 2] +
             bbox_pred[:, 3], bbox_pred[:, 4] + bbox_pred[:, 5]),
            dim=-1)
        if bbox_pred.shape[-1] <= 9:
            return torch.cat((center, size, bbox_pred[:, 6:]), dim=-1)
        return torch.cat((center, size, euler), dim=-1)

    @staticmethod
    def _get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
            (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        shift = torch.stack(
            (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
             points[..., 2] - boxes[..., 2]),
            dim=-1).permute(1, 0, 2)
        if boxes.shape[-1] == 7:
            shift = rotation_3d_in_axis(shift, -boxes[0, :, 6],
                                        axis=2).permute(1, 0, 2)
        else:
            shift = rotation_3d_in_euler(shift,
                                         -boxes[0, :, 6:]).permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)

    @staticmethod
    def _get_centerness(face_distances: Tensor) -> Tensor:
        """Compute point centerness w.r.t containing box.

        Args:
            face_distances (Tensor): Face distances of shape (B, N, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

        Returns:
            Tensor: Centerness of shape (B, N).
        """
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
            y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
            z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)

    @torch.no_grad()
    def get_targets(self, points: Tensor, gt_bboxes: BaseInstance3DBoxes,
                    gt_labels: Tensor) -> Tuple[Tensor, ...]:
        """Compute targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.

        Returns:
            tuple[Tensor, ...]: Centerness, bbox and classification
            targets for all locations.
        """
        float_max = points[0].new_tensor(1e8)
        n_levels = len(points)
        levels = torch.cat([
            points[i].new_tensor(i).expand(len(points[i]))
            for i in range(len(points))
        ])
        points = torch.cat(points)
        gt_bboxes = gt_bboxes.to(points.device)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if n_boxes == 0:
            # return pseudo center_targets, bbox_targets and cls_targets
            return gt_bboxes.tensor.new_zeros((n_points,)), \
                gt_bboxes.tensor.new_zeros((n_points, gt_bboxes.shape[-1])), \
                gt_labels.new_full((n_points,), -1)

        n_dims = gt_bboxes.tensor.shape[-1]
        volumes = gt_bboxes.volume.unsqueeze(0).expand(n_points, n_boxes)

        # condition 1: point inside box
        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                          dim=1)
        boxes = boxes.expand(n_points, n_boxes, n_dims)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        face_distances = self._get_face_distances(points, boxes)
        inside_box_condition = face_distances.min(dim=-1).values > 0

        # condition 2: positive points per level >= limit
        # calculate positive points per scale
        n_pos_points_per_level = []
        for i in range(n_levels):
            n_pos_points_per_level.append(
                torch.sum(inside_box_condition[levels == i], dim=0))
        # find best level
        n_pos_points_per_level = torch.stack(n_pos_points_per_level, dim=0)
        lower_limit_mask = n_pos_points_per_level < self.pts_assign_threshold
        lower_index = torch.argmax(lower_limit_mask.int(), dim=0) - 1
        lower_index = torch.where(lower_index < 0, 0, lower_index)
        all_upper_limit_mask = torch.all(torch.logical_not(lower_limit_mask),
                                         dim=0)
        best_level = torch.where(all_upper_limit_mask, n_levels - 1,
                                 lower_index)
        # keep only points with best level
        best_level = best_level.expand(n_points, n_boxes)
        levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = best_level == levels

        # condition 3: limit topk points per box by centerness
        centerness = self._get_centerness(face_distances)
        centerness = torch.where(inside_box_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        centerness = torch.where(level_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness,
                                    min(self.pts_center_threshold + 1,
                                        len(centerness)),
                                    dim=0).values[-1]
        topk_condition = centerness > top_centerness.unsqueeze(0)

        # condition 4: min volume box per point
        volumes = torch.where(inside_box_condition, volumes, float_max)
        volumes = torch.where(level_condition, volumes, float_max)
        volumes = torch.where(topk_condition, volumes, float_max)
        min_volumes, min_inds = volumes.min(dim=1)

        center_targets = centerness[torch.arange(n_points), min_inds]
        bbox_targets = boxes[torch.arange(n_points), min_inds]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = gt_labels[min_inds]
        cls_targets = torch.where(min_volumes == float_max, -1, cls_targets)
        return center_targets, bbox_targets, cls_targets

    def _single_scene_multiclass_nms(self, bboxes: Tensor, scores: Tensor,
                                     input_meta: dict) -> Tuple[Tensor, ...]:
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor, ...]: Predicted bboxes, scores and labels.
        """
        num_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] >= 7
        if bboxes.shape[-1] == 9:
            bboxes = bboxes[..., :7]  # only consider yaw during nms
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(num_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if with_yaw:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)

            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(class_scores[nms_ids].shape,
                                i,
                                dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if bboxes.shape[-1] < 9:
            if with_yaw:
                box_dim = 7
            else:
                box_dim = 6
                nms_bboxes = nms_bboxes[:, :box_dim]

        return nms_bboxes, nms_scores, nms_labels


def normalize_vector(vector):
    norm = torch.norm(vector, dim=1, keepdim=True) + 1e-8
    normalized_vector = vector / norm
    return normalized_vector


def cross_product(a, b):
    cross_product = torch.cross(a, b, dim=1)
    return cross_product


def ortho_6d_2_Mat(x_raw, y_raw):
    """x_raw, y_raw: both tensors batch*3."""
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)  # batch*3
    x = cross_product(y, z)  # batch*3

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix
