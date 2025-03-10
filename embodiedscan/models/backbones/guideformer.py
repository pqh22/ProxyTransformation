# Copyright (c) OpenRobotLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/resnet.py # noqa
# and mmcv.cnn.ResNet
from typing import List, Union
import torch
try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    # blocks are used in the static part of MinkResNet
    ME = BasicBlock = Bottleneck = SparseTensor = None

import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from embodiedscan.utils import ConfigType, OptConfigType
from torch import Tensor
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn import build_norm_layer
from torch.nn import functional as F

from embodiedscan.registry import MODELS


@MODELS.register_module()
class MinkGuideFormer(BaseModule):
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
                 embed_dims: int = 256,
                 pool: bool = True,
                 use_text_guide: bool = False,
                 use_img_guide: bool = False):
        super(MinkGuideFormer, self).__init__()
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
        self.embed_dims = embed_dims

        self.inplanes = 64
        self.conv1 = ME.MinkowskiConvolution(in_channels,
                                             self.inplanes,
                                             kernel_size=3,
                                             stride=2,
                                             dimension=3)
        # May be BatchNorm is better, but we follow original implementation.
        self.norm1 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if self.pool:
            self.maxpool = ME.MinkowskiMaxPooling(kernel_size=2,
                                                  stride=2,
                                                  dimension=3)

        for i in range(len(stage_blocks)):
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, 64 * 2**i, stage_blocks[i], stride=2))
        
        self.guide_former = GuideFormer(num_heads=8, grid_size=16, use_text_guide=use_text_guide, use_img_guide=use_img_guide)

        self.scores1 = nn.Linear(embed_dims, 3)
        self.scores2 = nn.Linear(embed_dims, 64)
        self.scores3 = nn.Linear(embed_dims, 128)
        self.scores4 = nn.Linear(embed_dims, 512)

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

        for m in [self.scores1, self.scores2, self.scores3, self.scores4]:
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

    def forward(self, x: SparseTensor, guide, use_text_guide=False, use_img_guide=False) -> List[SparseTensor]:
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        if use_text_guide == True:
            density_scores = self.guide_former(x.C, guide, use_text_guide=True, use_img_guide=False)
        if use_img_guide == True:
            density_scores = self.guide_former(x.C, guide, use_text_guide=False, use_img_guide=True)
        if use_text_guide == False and use_img_guide == False:
            density_scores = self.guide_former(x.C, guide, use_text_guide=False, use_img_guide=False)

        if x.F.shape[1] == 3:
            density_scores = self.scores1(density_scores)
        elif x.F.shape[1] == 64:
            density_scores = self.scores2(density_scores)
        elif x.F.shape[1] == 128:
            density_scores = self.scores3(density_scores)
        elif x.F.shape[1] == 512:
            density_scores = self.scores4(density_scores)
        
        B, C = density_scores.shape
        batch_indices = x.C[:, 0].long()

        new_feat = []
        for i in range(B):
            mask = (batch_indices == i)
            scores_ex = density_scores[i].unsqueeze(0).expand(mask.sum(), C)
            new_feat.append((x.F[mask] * scores_ex).to('cuda'))
        
        new_feat_cat = torch.cat(new_feat, dim=0)
        
        x = ME.SparseTensor(coordinates=x.C, features=new_feat_cat)

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

class GuideFormer(nn.Module):
    def __init__(self, num_heads=8, grid_size=16, use_text_guide=False, use_img_guide=False, input_dim=512, spacial_dim=15):
        super(GuideFormer, self).__init__()
        self.embed_dim = grid_size**2
        self.grid_size = grid_size
        self.proj_views = self.embed_dim
        self.num_heads = num_heads

        self.use_text_guide = use_text_guide
        self.use_img_guide = use_img_guide

        self.position_embedding = self.get_position_embedding(self.grid_size, self.embed_dim)
        self.view_change = nn.Linear(3, self.proj_views)
        self.inter_view_attention = nn.MultiheadAttention(self.proj_views, self.num_heads, dropout=0.0, batch_first=True)

        if self.use_img_guide:
            # self.channel_mapper = nn.Linear(14400, 4)
            self.channel_mapper = nn.Conv2d(input_dim, self.embed_dim, kernel_size=1)
            self.attn_pool2d = AttentionPool2d(spacial_dim=spacial_dim, embed_dim=self.embed_dim, num_heads=self.num_heads, output_dim=self.embed_dim)
            self.norm_img = nn.LayerNorm(self.embed_dim)

        self.guide_attention = GuideAttention(self.embed_dim, self.num_heads) # 补充输入参数

        self.conv1 = nn.Conv1d(in_channels=self.embed_dim, out_channels=1, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)


    def get_position_embedding(self, grid_size, embed_dim): # 之后尝试替换为可学习的位置编码
        position_embedding = torch.zeros((grid_size * grid_size, embed_dim))
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        position = torch.arange(grid_size).float().unsqueeze(1)
        pos_emb = torch.zeros((grid_size, grid_size, embed_dim))
        pos_emb[:, :, 0::2] = torch.sin(position * div_term).unsqueeze(0)
        pos_emb[:, :, 1::2] = torch.cos(position * div_term).unsqueeze(1)
        
        position_embedding = pos_emb.view(-1, embed_dim)
        position_embedding = position_embedding.to('cuda')
        return position_embedding
    
    def forward(self, point_coordinates, guide, use_text_guide=False, use_img_guide=False):
        self.use_text_guide = use_text_guide
        self.use_img_guide = use_img_guide
        batch_indices = point_coordinates[:, 0].long()
        unique_batches = torch.unique(batch_indices)
        B = len(unique_batches)
        V = self.proj_views
        C = self.embed_dim
        
        density_grid = self.project_point_clouds(point_coordinates, self.grid_size)
        assert density_grid.shape == (B, 3, self.grid_size, self.grid_size)
        # print(density_grid.device)


        density_grid = density_grid.view(B, 3, -1) # (6,3,256) (bs,view,embed)
        multiview_density = self.view_change(density_grid.permute(0, 2, 1)).permute(0,2,1) + self.position_embedding
        assert multiview_density.shape == (B, V, C) # (6,256,256)
        # 看看这里要不要归一化一下 后边再加上与文本特征的融合
        # 就可以说是借助文本特征确定观察密度的视角 借助密度为点云赋予特定的重要性分数
        multiview_density = self.norm1(multiview_density)
        multiview_density, _ = self.inter_view_attention(query=multiview_density, key=multiview_density, value=multiview_density)
        # 这里会返回两个值，第一个是输出，第二个是注意力权重

        if self.use_text_guide:
            text_feats = guide['text_feats'] # (6,14,256)
            text_token_mask = ~guide['text_token_mask'] # (6,14) ~是取反操作
            multiview_density = self.guide_attention(query=multiview_density, guide=text_feats, text_attention_mask=text_token_mask) # (6,256,256)

        if self.use_img_guide:
            img_feat = guide[-1] 
            B, V, C, H, W = img_feat.shape # (6,50,512,15,15)
            img_feat = img_feat.view(B*V, C, H, W)
            img_feat = self.channel_mapper(img_feat)
            img_feat = self.attn_pool2d(img_feat)
            img_feat = self.norm_img(img_feat)
            img_feat = img_feat.view(B, V, self.embed_dim) # (6,50,256)
            multiview_density = self.guide_attention(query=multiview_density, guide=img_feat, text_attention_mask=None) # (6,256,256)

        # if self.use_img_guide: # 该模块还能再优化
        #     img_feat = guide[0] 
        #     B, V, C, H, W = img_feat.shape # (6,50,64,120,120)
        #     img_feat = img_feat.view(B, V, C, H*W)
        #     img_feat = self.channel_mapper(img_feat)
        #     img_feat = img_feat.view(B, V, -1) # (6,50,256)
        #     assert img_feat.shape == (B, V, self.embed_dim)
        #     multiview_density = self.guide_attention(query=multiview_density, guide=img_feat, text_attention_mask=None) # (6,256,256)


        density_scores = self.conv1(multiview_density) # 1*1 conv 降维
        density_scores = self.norm2(density_scores)
        density_scores.squeeze_(1)

        # 这里尝试将维度对齐，形成重要性分数
        # 回到上边按照视角的重要性分数对点云进行加权
        return density_scores # (B, C) (6,256)

    def project_point_clouds(self,point_clouds, grid_size=16):
        batch_indices = point_clouds[:, 0].long()
        unique_batches = torch.unique(batch_indices)
        
        front_grids = torch.zeros((len(unique_batches), grid_size, grid_size), device='cuda')
        left_grids = torch.zeros((len(unique_batches), grid_size, grid_size), device='cuda')
        bottom_grids = torch.zeros((len(unique_batches), grid_size, grid_size), device='cuda')
        
        for i, batch in enumerate(unique_batches):
            batch_points = point_clouds[batch_indices == batch][:, 1:]

            # 投影到正平面 (x, y)
            front_projection = batch_points[:, [0, 1]]
            front_min = front_projection.min(dim=0).values
            front_max = front_projection.max(dim=0).values
            front_normalized = (front_projection - front_min) / (front_max - front_min + 1e-8)
            front_indices = torch.clamp((front_normalized * grid_size).long(), 0, grid_size - 1)

            # 投影到左平面 (z, y)
            left_projection = batch_points[:, [2, 1]]
            left_min = left_projection.min(dim=0).values
            left_max = left_projection.max(dim=0).values
            left_normalized = (left_projection - left_min) / (left_max - left_min + 1e-8)
            left_indices = torch.clamp((left_normalized * grid_size).long(), 0, grid_size - 1)

            # 投影到下平面 (x, z)
            bottom_projection = batch_points[:, [0, 2]]
            bottom_min = bottom_projection.min(dim=0).values
            bottom_max = bottom_projection.max(dim=0).values
            bottom_normalized = (bottom_projection - bottom_min) / (bottom_max - bottom_min + 1e-8)
            bottom_indices = torch.clamp((bottom_normalized * grid_size).long(), 0, grid_size - 1)

            def count_points_in_grid(indices, grid):
                idx = indices[:, 0] * grid_size + indices[:, 1]
                unique_idx, counts = torch.unique(idx, return_counts=True)
                grid_indices = torch.stack([unique_idx // grid_size, unique_idx % grid_size], dim=1)
                grid[grid_indices[:, 0], grid_indices[:, 1]] += counts.float()

            count_points_in_grid(front_indices, front_grids[i])
            count_points_in_grid(left_indices, left_grids[i])
            count_points_in_grid(bottom_indices, bottom_grids[i])

        combined_grids = torch.stack([front_grids, left_grids, bottom_grids], dim=1)
        return combined_grids
    
class GuideAttention(BaseModule):
    def __init__(self, embed_dims: int = 256, num_heads: int = 8,
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        self.self_attn = MultiheadAttention(embed_dims=256,num_heads=8,dropout=0.0,batch_first=True)
        self.cross_attn = MultiheadAttention(embed_dims=256,num_heads=8,dropout=0.0,batch_first=True)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor, # multiview density
                guide: Tensor = None,
                text_attention_mask: Tensor = None) -> Tensor:
        
        # self attention  multiview density

        import os
        if os.getenv('IPDB_DEBUG', '0') == '1':
            import ipdb; ipdb.set_trace() 
        # cross attention between density and text
        query = self.cross_attn(query=query,key=guide,value=guide,
                                     key_padding_mask=text_attention_mask)
        query = self.norms[0](query)

        query = self.self_attn(query=query,key=query,value=query,)
        query = self.norms[1](query)

        return query 

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__() # spacial_dim=15 embed_dim=256 num_heads=8 output_dim=256
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5) # 这里其实只是设置了形状，是可学习的位置嵌入
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim) # 线性层作为投影矩阵使用 好 其中包含了权重和偏置 weight bias
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads # 8

    def forward(self, x): # (300,256,15,15) -> (300,256)
        #! 理解为什么要reshape 因为注意力操作要求输入squence 所以将HW展开为sequence
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC (225,300,256)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC = (1,300,256)+(225,300,256) 为什么加上均值，与输出有一定关系
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC 楽 这种可学习的位置编码感觉就纯是起个名字
        x, _ = F.multi_head_attention_forward( # 好奇这种函数都是在哪里找到的 感觉这个函数不是很好找 大致思路是torch -> nn -> search
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1], # 256
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight, # 还是要仔细阅读api函数的 不仅要明确输入输出 还要理解其中逻辑
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training, # false  evaling
            need_weights=False
        )# 注意力魔改池化 -> (226,300,256)
        # 注意力池化层常用于图像与文本的特征融合，此处将图像特征转化为与文本特征形状相同的特征向量，以便后续的特征融合操作。
        return x[0] # (300,256) 为什么呢 只是单纯的转换形状吗 因为第一个元素在经过注意力模块时是后边元素的均值，所以还是有一定代表性的
        # 思路源自于vit中的class token (cls) idea很重要 解释性很看故事