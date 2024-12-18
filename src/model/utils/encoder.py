import numpy as np
import torch
import torch.nn as nn
import copy
import warnings
from einops import rearrange

from ..encoder.backbone.unimatch.geometry import coords_grid
from .ffn import FFN
from .attention import UVSelfAttention, UVCrossAttention, UVCoarseAttention
import pdb

class UVTransformerEncoder(nn.Module):
    def __init__(self, num_layers=1, return_intermediate=False, embed_dims=256, mode=None):

        super(UVTransformerEncoder, self).__init__()
        self.return_intermediate = return_intermediate
        self.embed_dims = embed_dims

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(UVTransformerEncoderLayer(embed_dims, mode=mode))
        
    def forward(self,
                bev_query,
                key,
                value,
                bev_u=None,
                bev_v=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                intrinsics=None,
                extrinsics=None,
                depth_sup=None,
                grid=None,
                depth=None):
        output = bev_query
        intermediate = []
        
        bev_query = bev_query.permute(1, 0, 2) # b vhw c
        bs = bev_query.shape[0]
        bev_query = bev_query.reshape(bs*2,-1, self.embed_dims) # [bv, num_q, 256]

        if bev_pos is not None:
            bev_pos = bev_pos.permute(1, 0, 2)
            bev_pos = bev_pos.reshape(bs*2,-1, self.embed_dims) # [bv, num_q, 256]

        if depth_sup is not None:
            depth_sup = depth_sup.permute(1, 0, 2)
            depth_sup = depth_sup.reshape(bs*2,-1, 1) # [bv, num_q, 256]
        
        num_depth = 128
        eps = 1e-3

        ref_3d = grid.reshape(2,bs,num_depth, bev_u*bev_v, 2).permute(1,0,3,2,4) # b v
        ref_3d = ref_3d.reshape(bs*2, bev_u*bev_v, num_depth, 2)
        ref_3d = ref_3d / 2 + 0.5 # from -1 1 to 0 1
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, bev_v - 0.5, bev_v, dtype=bev_query.dtype, device=bev_query.device),
            torch.linspace(
                0.5, bev_u - 0.5, bev_u, dtype=bev_query.dtype, device=bev_query.device)
        )
        ref_y = ref_y.reshape(-1)[None] / bev_v
        ref_x = ref_x.reshape(-1)[None] / bev_u
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, bev_u*bev_v, 1, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                bev_u=bev_u,
                bev_v=bev_v,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                depth_sup=depth_sup)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class UVTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_haed=8,
                 dropout=0.1,
                 feedforward_channels=256,
                 mode=None,
                 with_cp=True):
        super(UVTransformerEncoderLayer, self).__init__()
        self.use_checkpoint = with_cp
        self.embed_dims = embed_dims
        self.mode = mode
        
        self.batch_first = False
        self.num_attn = 1
        if mode == 'coarse':
            self.attentions = nn.ModuleList()
            self.attentions.append(UVCoarseAttention(embed_dims=embed_dims))
        elif mode == 'fine':
            self.attentions = nn.ModuleList()
            # self attn
            self.attentions.append(UVSelfAttention(embed_dims=embed_dims))
            # cross attn
            self.attentions.append(UVCrossAttention(embed_dims=embed_dims))

            self.ffns = nn.ModuleList()
            self.ffns.append(FFN(embed_dims, feedforward_channels))

            self.norms = nn.ModuleList()
            for _ in range(3):
                self.norms.append(nn.LayerNorm(embed_dims))
        else:
            raise NotImplementedError

    def forward(self,
                query,
                key,
                value,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_u=None,
                bev_v=None,
                spatial_shapes=None,
                level_start_index=None,
                depth_sup=None):
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        if self.mode == 'coarse':
            query = self.attentions[0](
                query,
                key,
                value,
                None,
                query_pos=query_pos,
                ref_3d=ref_3d,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                depth_sup=depth_sup)
        elif self.mode == 'fine':
            # self attn
            temp_key = temp_value = query
            query = self.attentions[0](
                query,
                temp_key,
                temp_value,
                None,
                query_pos=bev_pos,
                key_padding_mask=query_key_padding_mask,
                ref_2d=ref_2d,
                spatial_shapes=torch.tensor(
                    [[bev_v, bev_u]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                depth_sup=depth_sup)
            query = self.norms[0](query)
                        
            # # cross attn
            query = self.attentions[1](
                query,
                key,
                value,
                None,
                query_pos=query_pos,
                ref_3d=ref_3d,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                depth_sup=depth_sup)
            query = self.norms[1](query)

            query = self.ffns[0](query, None)
            query = self.norms[2](query)
        else:
            raise NotImplementedError

        return query
