import warnings
import math

import torch
import torch.nn as nn

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.1,
                 batch_first=False):
        super(MultiheadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)
        self.proj_drop = nn.Dropout(0.0)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

class MAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False):
        super(MAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)
        self.proj_drop = nn.Dropout(0.0)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

class UVSelfAttention(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_heads=1,
                 num_levels=1,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data, gain=1.0)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                ref_2d=None,
                spatial_shapes=None,
                level_start_index=None,
                depth_sup=None):
        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        bsv,  num_query, embed_dims = query.shape
        if depth_sup is not None:
            query = torch.cat((query,depth_sup),-1)

        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bsv, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bsv, num_query, self.num_heads,  self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bsv, num_query,  self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bsv, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if ref_2d.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = ref_2d[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 but get {ref_2d.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
    
class UVCrossAttention(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_cams=1,
                 dropout=0.1,
                 bev_u=64,
                 bev_v=64,
                 batch_first=False):
        super(UVCrossAttention, self).__init__()

        self.bev_u = bev_u
        self.bev_v = bev_v

        self.num_levels = 1
        self.num_heads = 1
        self.num_points = 4
        self.num_depth = 128
        self.im2col_step = 64

        self.dropout = nn.Dropout(dropout)
        self.embed_dims = embed_dims
        self.num_cams = num_cams

        self.sampling_offsets = nn.Linear(embed_dims, self.num_cams * self.num_heads * self.num_levels * self.num_depth * self.num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, self.num_cams * self.num_heads * self.num_levels * self.num_depth * self.num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(1, self.num_heads, 1, 1, 1, 2).repeat(self.num_cams, 1, self.num_levels, self.num_depth, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, :, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data, gain=1.0)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)
    
    def forward(self,
                query,
                key,
                value,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                ref_3d=None,
                spatial_shapes=None,
                level_start_index=None,
                depth_sup=None):
        if identity is None:
            identity = query

        if key is None:
            key = query
        if value is None:
            value = key
        bsv, num_query, _ = query.shape

        if query_pos is not None:
            query = query + query_pos

        if depth_sup is not None:
            query = torch.cat((query,depth_sup),-1)

        num_cams, num_value, bs, embed_dims = key.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        
        key = key.permute(2, 0, 1, 3).reshape(bsv, num_value, self.embed_dims)
        value = torch.flip(value, dims=[0])
        value = value.permute(2, 0, 1, 3).reshape(bsv, num_value, self.embed_dims)

        value = self.value_proj(value)
        value = value.reshape(bsv*self.num_cams, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(bsv, num_query, self.num_cams, self.num_depth, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bsv, num_query, self.num_cams, self.num_depth, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bsv,
                                                   num_query,
                                                   self.num_cams,
                                                   self.num_depth,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 2, 1, 3, 4, 5, 6)\
            .reshape(bsv*self.num_cams, num_query*self.num_depth, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 2, 1, 3, 4, 5, 6, 7)\
            .reshape(bsv*self.num_cams, num_query*self.num_depth, self.num_heads, self.num_levels, self.num_points, 2)
        
        ref_3d = ref_3d.reshape(bsv*self.num_cams, num_query*self.num_depth, 2)

        if ref_3d.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = ref_3d[:, :, None, None, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 but get {ref_3d.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = output.reshape(bsv, self.num_cams, num_query, self.num_depth, self.embed_dims)
        output = output.mean(1)
        key = key.unsqueeze(2)
        output = output * key
        output = output.mean(-1)
        # output = output.sum(-1) / (self.embed_dims ** 0.5)

        output = self.output_proj(output)
        
        # return output + identity
        return self.dropout(output) + identity

class UVCoarseAttention(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_cams=1,
                 dropout=0.1,
                 bev_u=64,
                 bev_v=64,
                 batch_first=False):
        super(UVCoarseAttention, self).__init__()

        self.bev_u = bev_u
        self.bev_v = bev_v

        self.num_levels = 1
        self.num_heads = 1
        self.num_points = 1
        self.num_depth = 128
        self.im2col_step = 64

        # self.dropout = nn.Dropout(dropout)
        self.embed_dims = embed_dims
        self.num_cams = num_cams

        # self.sampling_offsets = nn.Linear(embed_dims, self.num_cams * self.num_heads * self.num_levels * self.num_depth * self.num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, self.num_cams * self.num_heads * self.num_levels * self.num_depth * self.num_points)
        # self.value_proj = nn.Linear(embed_dims, embed_dims)
        # self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        # nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        # thetas = torch.arange(
        #     self.num_heads,
        #     dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(1, self.num_heads, 1, 1, 1, 2).repeat(self.num_cams, 1, self.num_levels, self.num_depth, self.num_points, 1)
        # for i in range(self.num_points):
        #     grid_init[:, :, :, :, i, :] *= i + 1

        # self.sampling_offsets.bias.data = grid_init.view(-1)
        # nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        # nn.init.xavier_uniform_(self.value_proj.weight.data, gain=1.0)
        # nn.init.constant_(self.value_proj.bias.data, 0.)
        # nn.init.xavier_uniform_(self.output_proj.weight.data)
        # nn.init.constant_(self.output_proj.bias.data, 0.)
    
    def forward(self,
                query,
                key,
                value,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                ref_3d=None,
                spatial_shapes=None,
                level_start_index=None,
                depth_sup=None):
        if identity is None:
            identity = query

        if key is None:
            key = query
        if value is None:
            value = key
        bsv, num_query, _ = query.shape

        if depth_sup is not None:
            query = torch.cat((query,depth_sup),-1)

        num_cams, num_value, bs, embed_dims = key.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        
        key = key.permute(2, 0, 1, 3).reshape(bsv, num_value, self.embed_dims)
        value = torch.flip(value, dims=[0])
        value = value.permute(2, 0, 1, 3).reshape(bsv, num_value, self.embed_dims)

        # value = self.value_proj(value)
        value = value.reshape(bsv*self.num_cams, num_value, self.num_heads, -1)

        # sampling_offsets = self.sampling_offsets(query).view(bsv, num_query, self.num_cams, self.num_depth, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bsv, num_query, self.num_cams, self.num_depth, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bsv,
                                                   num_query,
                                                   self.num_cams,
                                                   self.num_depth,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 2, 1, 3, 4, 5, 6)\
            .reshape(bsv*self.num_cams, num_query*self.num_depth, self.num_heads, self.num_levels, self.num_points).contiguous()
        # sampling_offsets = sampling_offsets.permute(0, 2, 1, 3, 4, 5, 6, 7)\
        #     .reshape(bsv*self.num_cams, num_query*self.num_depth, self.num_heads, self.num_levels, self.num_points, 2)
        
        ref_3d = ref_3d.reshape(bsv*self.num_cams, num_query*self.num_depth, 2)

        if ref_3d.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = ref_3d[:, :, None, None, None, :] # + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 but get {ref_3d.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = output.reshape(bsv, self.num_cams, num_query, self.num_depth, self.embed_dims)
        output = output.mean(1)
        key = key.unsqueeze(2)
        output = output * key
        # output = output.mean(-1)
        output = output.sum(-1) / (self.embed_dims ** 0.5)

        # output = self.output_proj(output)
        
        return output + identity
