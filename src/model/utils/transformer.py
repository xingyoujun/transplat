import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

import copy
import warnings

from .encoder import UVTransformerEncoder
from .attention import MAttention, MultiheadAttention
from .ffn import FFN


class Transformer(nn.Module):
    def __init__(self, embed_dims):
        super(Transformer, self).__init__()
        self.decoder = TransformerDecoder(embed_dims=embed_dims)
        self.embed_dims = embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_normal_(m)

    def forward(self, x, mask, query, query_embed, pos_embed):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)  # [bs, n, c, h, w] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]

        # out_dec: [num_query, bs, dim]
        out_dec = self.decoder(
            query=query,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            )

        out_dec = out_dec.transpose(0, 1) #[num_query, bs, dim] -> [bs, num_query, dim]
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory

class DecoderLayer(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_haed=8,
                 dropout=0.1,
                 feedforward_channels= 2048,
                 with_cp=True):
        super(DecoderLayer, self).__init__()
        self.use_checkpoint = with_cp
        self.embed_dims = embed_dims
        
        self.batch_first = False
        
        self.num_attn = 2
        self.attentions = nn.ModuleList()
        # self attn
        self.attentions.append(MultiheadAttention(embed_dims, num_haed, dropout))
        # cross attn
        self.attentions.append(MAttention(embed_dims, num_haed, dropout))

        self.ffns = nn.ModuleList()
        self.ffns.append(FFN(embed_dims, feedforward_channels))

        self.norms = nn.ModuleList()
        for _ in range(3):
            self.norms.append(nn.LayerNorm(self.embed_dims))
            
    def _forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
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

        # self attn
        temp_key = temp_value = query
        query = self.attentions[0](
            query,
            temp_key,
            temp_value,
            None,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=attn_masks[0],
            key_padding_mask=query_key_padding_mask)
        query = self.norms[0](query)

        # cross attn
        query = self.attentions[1](
            query,
            key,
            value,
            None,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_masks[1],
            key_padding_mask=key_padding_mask)
        query = self.norms[1](query)

        query = self.ffns[0](query, None)
        query = self.norms[2](query)

        return query

    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        else:
            x = self._forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask
            )
        return x



class TransformerDecoder(nn.Module):
    def __init__(self, embed_dims=256, num_layers=4, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DecoderLayer(embed_dims))
        self.embed_dims = embed_dims

        self.post_norm = nn.LayerNorm(embed_dims)
        self.return_intermediate = return_intermediate

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None):
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        if not self.return_intermediate:
            if self.post_norm is not None:
                return self.post_norm(query)
            else:
                return query
        else:
            return torch.stack(intermediate)
        
#####################################################################################

class UVTransformer(nn.Module):
    """Implements the VoxelFormer transformer.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_layers=1,
                 mode='coarse',):
        super(UVTransformer, self).__init__()
        self.encoder = UVTransformerEncoder(embed_dims=embed_dims, mode=mode, num_layers=num_layers)
        self.embed_dims = embed_dims

    def forward(
            self,
            mlvl_feats,
            bev_queries,
            bev_u,
            bev_v,
            bev_pos=None,
            intrinsics=None,
            extrinsics=None,
            depth_sup=None,
            grid=None,
            depth=None):

        bs = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam bs hw embed_dims
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_u=bev_u,
            bev_v=bev_v,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_sup=depth_sup,
            grid=grid,
            depth=depth,
        )
        bev_embed = bev_embed.reshape(bs, -1, self.embed_dims)
        bev_embed = bev_embed.transpose(0, 1)

        return bev_embed