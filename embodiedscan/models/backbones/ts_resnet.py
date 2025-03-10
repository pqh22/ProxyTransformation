from typing import List, Union
import os
import torch.nn as nn
from mmengine.model import BaseModule
# from torchsparse import SparseTensor
import torchsparse.nn as spnn
from mmdet3d.models.layers.torchsparse_block import (TorchSparseBasicBlock,
                                                     TorchSparseBottleneck,
                                                     TorchSparseConvModule)

from embodiedscan.registry import MODELS

@MODELS.register_module()
class MinkResNet(BaseModule):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input channels, 3 for RGB.
        num_stages (int): Resnet stages. Defaults to 4.
        pool (bool): Whether to add max pooling after first conv.
            Defaults to True.
    """
    arch_settings = {
        18: (TorchSparseBasicBlock, (2, 2, 2, 2)),
        34: (TorchSparseBasicBlock, (3, 4, 6, 3)),
        50: (TorchSparseBottleneck, (3, 4, 6, 3)),
        101: (TorchSparseBottleneck, (3, 4, 23, 3)),
        152: (TorchSparseBottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 in_channels: int,
                 num_stages: int = 4,
                 pool: bool = True):
        super(MinkResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.num_stages = num_stages
        self.pool = pool

        self.inplanes = 64
        # self.conv1 = SparseConv3d(in_channels,
        #                           self.inplanes,
        #                           kernel_size=3,
        #                           stride=2)
        self.conv1 = nn.Sequential(
            TorchSparseConvModule(
                in_channels,
                self.inplanes,
                kernel_size=3,
                padding=1,
                indice_key='subm0'),
            TorchSparseConvModule(
                self.inplanes,
                self.inplanes,
                kernel_size=3,
                padding=1,
                indice_key='subm0'))
        # May be BatchNorm is better, but we follow original implementation.
        if self.pool:
            self.maxpool = MaxPool3d(kernel_size=2,
                                     stride=2)

        for i in range(len(stage_blocks)):
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, 64 * 2**i, stage_blocks[i], stride=2))

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, spnn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

            if isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Union[TorchSparseBasicBlock, TorchSparseBottleneck], 
                    planes: int, blocks: int, stride: int) -> nn.Module:
        """Make single level of residual blocks.

        Args:
            block (BasicBlock | Bottleneck): Residual block class.
            planes (int): Number of convolution filters.
            blocks (int): Number of blocks in the layers.
            stride (int): Stride of the first convolutional layer.

        Returns:
            nn.Module: With residual blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SparseConv3d(self.inplanes,
                             planes * block.expansion,
                             kernel_size=1,
                             stride=stride),
                BatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride=stride,
                  downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: SparseTensor) -> List[SparseTensor]:
        """Forward pass of ResNet.

        Args:
            x (SparseTensor): Input sparse tensor.

        Returns:
            list[SparseTensor]: Output sparse tensors.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs