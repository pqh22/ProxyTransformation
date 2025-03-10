import torch
from torch import nn as nn


from embodiedscan.registry import MODELS

class GaussionKernelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, gt_bbox, cluster_centers, clusters, alpha=1.0, epsilon=1e-8):
        # 提取 GT 中心坐标
        c_gt = gt_bbox[:, :3]  # Shape: (1, 3)

        # 计算 sigma
        bbox_dims = gt_bbox[:, 3:6]  # Shape: (1, 3)
        sigma = torch.mean(bbox_dims) * alpha
        sigma_squared = 2 * sigma ** 2

        # 聚类中心的损失
        center_distances = cluster_centers - c_gt  # Shape: (M, 3)
        center_squared_distances = torch.sum(center_distances ** 2, dim=1)  # Shape: (M,)
        center_gaussian_similarities = torch.exp(-center_squared_distances / sigma_squared)
        center_loss = -torch.sum(torch.log(center_gaussian_similarities + epsilon)) / cluster_centers.shape[0]

        # 聚类点的损失
        distances = clusters - c_gt.unsqueeze(1).unsqueeze(1)  # Shape: (M, K, 3)
        squared_distances = torch.sum(distances ** 2, dim=2)  # Shape: (M, K)
        gaussian_similarities = torch.exp(-squared_distances / sigma_squared)
        point_loss = -torch.sum(torch.log(gaussian_similarities + epsilon)) / (clusters.shape[0] * clusters.shape[1])

        # 总损失
        total_loss = 0.5 * center_loss + 0.5 * point_loss

        return total_loss