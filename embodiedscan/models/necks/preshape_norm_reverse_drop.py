# import math
# import logging
# from functools import partial
# from collections import OrderedDict
# from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import ball_query, sample_farthest_points

from typing import List, Optional, Tuple, Union
from random import randint

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_

from embodiedscan.registry import MODELS


class DeformablePointCluster(nn.Module):
    def __init__(self, grid_size, num_sub, radius=3, embed_dim=256, margin=4, mlp_radio=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_cluster = grid_size**3
        self.num_sub = num_sub
        self.radius = radius
        self.embed_dim = embed_dim
        self.margin = margin
        self.get_offsets = OffsetNetwork(in_features=6, hidden_features=embed_dim)

    def init_uniform_cluster_center(self, points, gs):
        B = points.shape[0]
        
        # 1. find min&max
        min_coords = points.min(dim=1, keepdim=True)[0] # (B, 1, 3)
        max_coords = points.max(dim=1, keepdim=True)[0] # (B, 1, 3)
        
        # 2. linear [0,1]
        linspace = torch.linspace(0, 1, gs, device=points.device)
        
        # 3. generate 3d mesh
        grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).unsqueeze(0)  # (1, M, 3)

        # 4. transform to origin space
        uniform_points = min_coords + self.margin + grid * (max_coords - min_coords - 2 * self.margin) 
        assert uniform_points.shape == (B, gs**3, 3)

        return uniform_points, min_coords, max_coords
    
    def forward(self, points):
        # uniformly get grid center point
        cluster_center, min_coords, max_coords = self.init_uniform_cluster_center(points, self.grid_size) # (B,M,3)
        _, _, temp_cluster = ball_query(p1=cluster_center, p2=points, K=self.num_sub, radius=self.radius) # (B,M,K) (B,M,K,3)

        offsets = self.get_offsets(cluster_center, temp_cluster)
        offsets = offsets.tanh() * self.margin

        new_cluster_center = cluster_center + offsets
        clamped_centers = torch.max(torch.min(new_cluster_center, max_coords), min_coords)

        # get final_cluster
        _, idx, final_cluster = ball_query(p1=clamped_centers, p2=points, K=self.num_sub, radius=self.radius) # (B,M,K,3)

        return clamped_centers, final_cluster, idx

class OffsetNetwork(nn.Module):
    def __init__(self, in_features=3+3, hidden_features=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features,hidden_features,1),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU()
        )
        self.channel_mapper = nn.Conv1d(in_channels=hidden_features, out_channels=3, kernel_size=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, center, cluster):
        '''
        center : (b,m,3)
        cluster: (b,m,k,3)
        '''
        b, m, _ = center.shape
        relative = cluster - center.unsqueeze(2) # (b,m,k,3)
        padding_mask = (cluster == 0).all(dim=-1)
        relative[padding_mask] = 0
        # here cat relative and cluster
        # relative -> intra local  -> cluster feature
        # cluster  -> inter global -> position information
        x = torch.cat([relative, cluster], dim=-1) # (b,m,k,3+3)
        x = self.mlp(x.permute(0,3,1,2)) # (b,c,m,k)
        # x = x.permute(0,2,3,1) # (b,m,k,c)
        x = torch.mean(x, dim=-1) # (b,c,m)
        x = self.channel_mapper(x) # (b,3,m)
        x = x.transpose(-2,-1)
        assert x.shape == (b,m,3)

        return x
    
class SimplifiedPointNet(nn.Module):
    def __init__(self, in_features=3+3, out_features=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features,out_features,1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
        self.reset_parameters()

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, center, cluster):
        '''
        center : (b,m,3)
        cluster: (b,m,k,3)
        '''
        relative = cluster - center.unsqueeze(2) # (b,m,k,3)
        padding_mask = (cluster == 0).all(dim=-1)
        relative[padding_mask] = 0
        # here cat relative and cluster
        # relative -> intra local  -> cluster feature
        # cluster  -> inter global -> position information
        x = torch.cat([relative, cluster], dim=-1) # (b,m,k,3+3)
        x = self.mlp(x.permute(0,3,1,2))
        x = x.permute(0,2,3,1) # (b,m,k,c)
        x = torch.max(x, dim=2)[0]

        return x

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__() # spacial_dim=15 embed_dim=256 num_heads=8 output_dim=256
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads # 8

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]
    
class ProxyAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 num_cluster=12**3, dynamic_drop_radio=0.8,**kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proxy_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.num_cluster = num_cluster
        self.dynamic_drop_radio = dynamic_drop_radio
        self.real_cluster_num = int(num_cluster * (1 - dynamic_drop_radio))
        self.spatial_dim = int(dim ** 0.5)

        # block column row
        self.pb_bias = nn.Parameter(torch.zeros(1, self.real_cluster_num, 4, 4))
        self.pc_bias = nn.Parameter(torch.zeros(1, self.real_cluster_num, self.spatial_dim, 1))
        self.pr_bias = nn.Parameter(torch.zeros(1, self.real_cluster_num, 1, self.spatial_dim))
        trunc_normal_(self.pb_bias, std=.02)
        trunc_normal_(self.pc_bias, std=.02)
        trunc_normal_(self.pr_bias, std=.02)

    def forward(self, x, proxy, mask=None):
        b, n, c = x.shape

        assert n == self.real_cluster_num
        s = self.spatial_dim
        # get bias
        bias1 = F.interpolate(self.pb_bias, size=(s,s), mode='bilinear') # (1,n,s,s)
        bias1 = bias1.reshape(1, n, -1).repeat(b, 1, 1) # (b,n,c)
        bias2 = (self.pc_bias + self.pr_bias).reshape(1, n, -1).repeat(b, 1, 1)
        bias = bias1 + bias2

        x = x + bias
        _, l, _ = proxy.shape
        num_heads = self.num_heads
        head_dim = c // num_heads        
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2] # (b,n,c)
        proxy_tokens = self.proxy_proj(proxy) # (b,l,c)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        proxy_tokens = proxy_tokens.reshape(b, l, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # proxy as query
        # proxy_tokens (b,l,c) q k v (b,n,c)
        proxy_attn = (proxy_tokens * self.scale) @ k.transpose(-2,-1)# (b,head,l,n)
        # if mask is not None:
        #     # mask = mask.repeat(1,num_heads,1,1)
        #     proxy_attn = proxy_attn.masked_fill(mask, float('-inf')) # (b,head,l,n)
        proxy_attn = self.softmax(proxy_attn)
        proxy_attn = self.attn_drop(proxy_attn)
        proxy_v = proxy_attn @ v # (b,head,l,c')

        # proxy as key
        query_attn = (q * self.scale) @ proxy_tokens.transpose(-2, -1)
        if mask is not None:
            mask = ~mask
            mask = mask.unsqueeze(1).unsqueeze(-1)
            mask = mask.expand(-1,num_heads,-1,n)
            mask = mask.transpose(-2, -1)
            query_attn = query_attn.masked_fill(mask, -1e9) # float('-inf')
        query_attn = self.softmax(query_attn)
        query_attn = self.attn_drop(query_attn)
        x = query_attn @ proxy_v

        x = x.transpose(1, 2).reshape(b, n, c)
        #TODO: more bias?

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ProxyBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_radio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_cluster=12**3, dynamic_drop_radio=0.8 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ProxyAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                    num_cluster=num_cluster, dynamic_drop_radio=dynamic_drop_radio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_radio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, proxy, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), proxy, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
#TODO: get submanifold transform matrix with several proxy block, img4intra-grid text4inter-grid
@MODELS.register_module()
class ProxyTransformationNormReverse(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, n_points=100000, grid_size=4 , text_blocks=1, img_blocks=1, dynamic_drop_radio=0.8,
                 mlp_radio=4, qkv_bias=False, drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_sub=30,
                 drop_radio=0.2, input_dim=512, img_spacial_dim=15):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.num_cluster = grid_size**3
        self.num_sub = num_sub or n_points // self.num_cluster
        self.input_dim = input_dim
        self.img_spacial_dim = img_spacial_dim
        self.drop_radio = drop_radio
        self.text_blocks = text_blocks
        self.img_blocks = img_blocks
        self.dynamic_drop_radio = dynamic_drop_radio
        text_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, text_blocks)]  # stochastic depth decay rule
        img_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, img_blocks)]  # stochastic depth decay rule

        self.get_deformable_cluster = DeformablePointCluster(grid_size=grid_size, num_sub=self.num_sub)
        self.simple_encoder = SimplifiedPointNet()
        
        self.channel_mapper = nn.Conv2d(input_dim, self.embed_dim, kernel_size=1)
        self.attn_pool2d = AttentionPool2d(spacial_dim=self.img_spacial_dim, embed_dim=self.embed_dim, num_heads=self.num_heads, output_dim=self.embed_dim)
        self.norm_img = nn.LayerNorm(self.embed_dim)

        self.textformer = nn.ModuleList([
            ProxyBlock(
                dim=embed_dim, num_heads=num_heads, mlp_radio=mlp_radio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=text_dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                num_cluster=self.num_cluster, dynamic_drop_radio=dynamic_drop_radio
            ) for i in range(text_blocks)
        ])
        self.text_norm = nn.ModuleList([norm_layer(embed_dim) for _ in range(text_blocks)])
        
        self.imgformer = nn.ModuleList([
            ProxyBlock(
                dim=embed_dim, num_heads=num_heads, mlp_radio=mlp_radio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=img_dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                num_cluster=self.num_cluster, dynamic_drop_radio=dynamic_drop_radio
            ) for i in range(img_blocks)
        ])
        self.img_norm = nn.ModuleList([norm_layer(embed_dim) for _ in range(img_blocks)])
        
        self.text_trans = nn.Linear(embed_dim, 3)
        self.img_trans = nn.Linear(embed_dim, 9)

        self.text_trans_norm = nn.BatchNorm1d(3)
        self.img_trans_norm = nn.BatchNorm1d(9)

    def get_text_proxy(self, text_dict):
        return text_dict.values()
    
    def get_img_proxy(self, img_feat):
        B, V, C, H, W = img_feat.shape # (6,50,512,15,15)
        img_feat = img_feat.view(B*V, C, H, W)
        img_feat = self.channel_mapper(img_feat)
        img_feat = self.attn_pool2d(img_feat)
        img_feat = self.norm_img(img_feat)
        img_feat = img_feat.view(B, V, self.embed_dim)
        return img_feat
    
    def get_point_proxy(self, center, cluster):
        point_proxy = self.simple_encoder(center, cluster)
        return point_proxy
    
    def get_point_cluster(self, points):
        center, cluster, idx = self.get_deformable_cluster(points) # (b,m,k,3)
        return center, cluster, idx
    
    def dynamic_cluster_dropout(self, cluster, center, idx, empty_drop=0.3):
        """
        Dynamically drops clusters based on their loss values.

        Args:
            cluster (torch.Tensor): Tensor of shape (B, M, K, D), the clustered points.
            center (torch.Tensor): Tensor of shape (B, M, D), the cluster centers.
            idx (torch.Tensor): Tensor of shape (B, M, K), the indices from ball query.
            drop_ratio (float): The proportion of clusters to drop per batch (0 < drop_ratio < 1).

        Returns:
            clusters_updated (torch.Tensor): Updated clusters tensor of shape (B, M', K, D).
            centers_updated (torch.Tensor): Updated centers tensor of shape (B, M', D).
            idx_updated (torch.Tensor): Updated idx tensor of shape (B, M', K).
        """
        # import pdb;pdb.set_trace()
        
        B, M, K, D = cluster.shape

        # Calculate the number of padding points in each cluster
        padding_counts = (idx == -1).sum(dim=2)  # Shape: (B, M)
        # Determine the number of clusters to drop based on padding counts
        num_clusters_to_drop = int(M * empty_drop)

        temp_keep_num = M - num_clusters_to_drop
        # Get the indices of clusters sorted by padding counts
        sorted_indices = torch.argsort(padding_counts, dim=1)
        indices_to_keep1 = sorted_indices[:, :temp_keep_num]  # Shape: (B, temp_keep_num)
        
        batch_indices = torch.arange(B, device=cluster.device).unsqueeze(-1).expand(B, temp_keep_num)

        updated_center = center[batch_indices,indices_to_keep1]
        updated_cluster = cluster[batch_indices,indices_to_keep1]
        updated_idx = idx[batch_indices,indices_to_keep1]


        temp = updated_center.shape[1]
        num_to_keep = int(M * (1 - self.dynamic_drop_radio))
        num_to_drop = temp - num_to_keep


        _, fps_indices_to_drop = sample_farthest_points(points=updated_center, lengths=None, K=num_to_drop) # (B,num_drop)

        all_indices = torch.arange(temp, device=updated_center.device) # (temp)

        keep_indices_list = []

        for i in range(B):
            mask = ~torch.isin(all_indices, fps_indices_to_drop[i]) # (temp)
            temp_indices_to_keep = torch.masked_select(all_indices, mask)
            keep_indices_list.append(temp_indices_to_keep)


        for i, item in enumerate(keep_indices_list):
            keep_indices_list[i] = item[:num_to_keep]
        # align its size, here i don't understand why they are different
        indices_to_keep = torch.stack(keep_indices_list, dim=0) # (B,num_to_keep)

        # Use indices to index cluster and idx
        batch_indices_keep = torch.arange(B, device=updated_cluster.device).unsqueeze(-1).expand(-1, num_to_keep)
        batch_indices_drop = torch.arange(B, device=updated_cluster.device).unsqueeze(-1).expand(-1, num_to_drop)

        new_center = updated_center[batch_indices_keep, indices_to_keep]
        new_cluster = updated_cluster[batch_indices_keep, indices_to_keep]
        new_idx = updated_idx[batch_indices_keep, indices_to_keep]
        drop_idx = updated_idx[batch_indices_drop, fps_indices_to_drop] # (b,num_to_drop,num_sub)
        drop_idx = drop_idx.reshape(B, -1) # ok!

        return new_cluster, new_center, new_idx, drop_idx



    def forward(self, points, text_dict, img_feat): # self-attn on global points is expensive, but with proxy it's linear 

        points = [p.unsqueeze(0) for p in points]
        points = torch.cat(points, dim=0) # (b,n,3)

        # get cluster
        point_center, point_cluster, idx = self.get_point_cluster(points) # (b,m,3) (b,m,k,3) (b,m,k)

        # dynamic drop
        point_cluster, point_center, idx, drop_idx = self.dynamic_cluster_dropout(point_cluster, point_center, idx)
        b, m, k, _ = point_cluster.shape

        # get proxy
        point_proxy = self.get_point_proxy(point_center, point_cluster) # (b,m,c)
        
        # text proxy guide
        text_proxy, text_mask = self.get_text_proxy(text_dict)
        for block, norm in zip(self.textformer, self.text_norm):
            text_guide_point = block(point_proxy,text_proxy,text_mask)
            text_guide_point = norm(text_guide_point)
            
        translate = self.text_trans(text_guide_point) # (b,m,3)
        translate = self.text_trans_norm(translate.transpose(-2, -1)).transpose(-2, -1)

        # img proxy guide
        img_proxy = self.get_img_proxy(img_feat)
        for block, norm in zip(self.imgformer, self.img_norm):
            img_guide_point = block(point_proxy,img_proxy,None)
            img_guide_point = norm(img_guide_point)

        transform = self.img_trans(img_guide_point) # (b,m,9)
        transform = self.img_trans_norm(transform.transpose(-2, -1)).transpose(-2, -1) # normalize to 1
        # translate cannot be too big, maybe need a constraint

        # submanifold reshape
        transform = transform.reshape(b, m, 3, 3) # (b,m,3,3)
        translate = translate.unsqueeze(-2) # (b,m,1,3)
        temp_center = point_center.unsqueeze(-2)
        new_clusters = (transform @ (point_cluster - temp_center).transpose(-2,-1)).transpose(-2,-1) + temp_center + translate # (b,m,k,3)

        # only replace at valid points
        new_points = pt_replace(points, idx, new_clusters) # (b,n,3)

        new_points_list = remove_points_by_index(new_points, drop_idx)
        
        return new_points_list


def pt_replace(p2, idx, cluster):
    # p2: Tensor of shape (B, N, 3)
    # idx: LongTensor of shape (B, M, K)
    # transformed_clusters: Tensor of shape (B, M, K, 3)

    # Create a mask of valid indices (where idx != -1)
    valid_mask = idx != -1  # BoolTensor of shape (B, M, K)

    # Flatten idx and transformed_clusters to work with them easily
    idx_flat = idx.reshape(-1)  # Shape: (B*M*K,)
    cluster_flat = cluster.reshape(-1, 3)  # Shape: (B*M*K, 3)
    valid_mask_flat = valid_mask.reshape(-1)  # Shape: (B*M*K,)

    # Generate batch indices corresponding to each element in idx
    B, M, K = idx.shape
    batch_indices = torch.arange(B, device=idx.device).reshape(B, 1, 1).expand(B, M, K).contiguous().reshape(-1)  # Shape: (B*M*K,)

    # Select only the valid indices and their corresponding batch indices
    valid_idx_flat = idx_flat[valid_mask_flat]  # Shape: (num_valid,)
    valid_batch_indices = batch_indices[valid_mask_flat]  # Shape: (num_valid,)
    cluster_valid = cluster_flat[valid_mask_flat]  # Shape: (num_valid, 3)

    # Replace the points in p2 with the transformed clusters at the specified indices
    p2[valid_batch_indices, valid_idx_flat, :] = cluster_valid

    # Now p2_updated contains the updated point cloud with the transformed points
    return p2 # (B,N,3)


def remove_points_by_index(new_points, drop_idx):
    """
    Removes points in `new_points` at the indices specified by `drop_idx`.
    
    Args:
        new_points (torch.Tensor): Tensor of shape (B, N, 3) containing the points.
        drop_idx (torch.Tensor): Tensor of shape (B, M) containing indices of points to remove for each batch.
        
    Returns:
        torch.Tensor: Tensor of shape (B, N - M, 3) with specified points removed.
    """

    filtered_points = []
    B, N, _ = new_points.shape
    M = drop_idx.shape[-1]
    all_indices = torch.arange(N, device=new_points.device)
    for i in range(B):
        idx = drop_idx[i]
        new_idx = torch.unique(idx)
        mask = ~torch.isin(all_indices,new_idx)
        keep = torch.masked_select(all_indices,mask)
        point = new_points[i][keep]
        filtered_points.append(point)
    
    return filtered_points

def sample_farthest_points_naive(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same Args/Returns as sample_farthest_points
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("Invalid lengths.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_K = torch.max(K)

    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(N):
        # Initialize an array for the sampled indices, shape: (max_K,)
        sample_idx_batch = torch.full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (max_K,),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        closest_dists = points.new_full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (lengths[n],),
            float("inf"),
            dtype=torch.float32,
        )

        # Select a random point index and save it as the starting point
        # pyre-fixme[6]: For 2nd argument expected `int` but got `Tensor`.
        selected_idx = randint(0, lengths[n] - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        k_n = min(lengths[n], K[n])

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # it's distance will be 0.0 so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            selected_idx = torch.argmax(closest_dists)
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

    # Gather the points
    all_sampled_points = masked_gather(points, all_sampled_indices)

    # Return (N, max_K, D) subsampled points and indices
    return all_sampled_points, all_sampled_indices

def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding

    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """

    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points