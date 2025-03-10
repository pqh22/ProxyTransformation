# Copyright (c) OpenRobotLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/resnet.py # noqa
# and mmcv.cnn.ResNet
from typing import List, Union
import os
try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    # blocks are used in the static part of MinkResNet
    ME = BasicBlock = Bottleneck = SparseTensor = None

import torch.nn as nn

from mmengine.model import BaseModule
from timm.models.layers import DropPath

from embodiedscan.registry import MODELS


@MODELS.register_module()
class GlobalAlignResNet(BaseModule):
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
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 in_channels: int,
                 num_stages: int = 4,
                 pool: bool = True,
                 align_dim: int = [64, 128, 256, 512],
                 score_dim = 256,
                 drop_path_rate = 0.5):
        super(GlobalAlignResNet, self).__init__()
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.`')
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.num_stages = num_stages
        self.pool = pool

        self.align_dim = align_dim
        self.inplanes = 64
        self.conv1 = ME.MinkowskiConvolution(in_channels,
                                             self.inplanes,
                                             kernel_size=3,
                                             stride=2,
                                             dimension=3)
        # May be BatchNorm is better, but we follow original implementation.
        if os.getenv('BATCHNORM', '0') == '1':
            self.norm1 = ME.MinkowskiBatchNorm(self.inplanes)
        else:
            self.norm1 = ME.MinkowskiInstanceNorm(self.inplanes)
            
        self.relu = ME.MinkowskiReLU(inplace=True)
        if self.pool:
            self.maxpool = ME.MinkowskiMaxPooling(kernel_size=2,
                                                  stride=2,
                                                  dimension=3)
            
        if align_dim is not None:
            self.channel_mapper = nn.ModuleList([nn.Linear(score_dim, dim) for dim in align_dim])
            self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        for i in range(len(stage_blocks)):
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, 64 * 2**i, stage_blocks[i], stride=2))

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel,
                                         mode='fan_out',
                                         nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Union[BasicBlock, Bottleneck], planes: int,
                    blocks: int, stride: int) -> nn.Module:
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
                ME.MinkowskiConvolution(self.inplanes,
                                        planes * block.expansion,
                                        kernel_size=1,
                                        stride=stride,
                                        dimension=3),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride=stride,
                  downsample=downsample,
                  dimension=3))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dimension=3))
        return nn.Sequential(*layers)

    def align_or_not(self, x, global_scores, i):
        if x.F.shape[-1] == self.align_dim[i]:
            global_scores = self.channel_mapper[i](global_scores) # (B,AC)
            feat = x.F
            coord = x.C
            assert global_scores.shape[-1] == feat.shape[-1] # ensure AC
            batch_indices = coord[:, 0].long()
            pointwise_weight = global_scores[batch_indices]
            aligned_feat = feat * pointwise_weight
            aligned_feat = self.drop_path(aligned_feat)
            x = ME.SparseTensor(features=aligned_feat, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
            return x
        else: 
            assert False, "Important error: aligned_dim is not right"
    
    def forward(self, x: SparseTensor, global_scores) -> List[SparseTensor]:
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        import os
        if os.getenv('PQH_PDB_DEBUG', '0') == '1':
            import pdb; pdb.set_trace()

        # x = self.align_or_not(x, global_scores)
        x = self.conv1(x) # (504714, 3) -> (490054,64) 其实是这一步有问题
        x = self.norm1(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x) # [427928, 64] 点云数量少了一半 前面的local transformation 有问题
        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x) # [259237, 64] [90748, 128] [22401, 256] [5021, 512]
            align_feat =  self.align_or_not(x, global_scores, i)
            x = x + align_feat
            outs.append(x)
        return outs

