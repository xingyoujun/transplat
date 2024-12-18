import torch
import torch.nn as nn 
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import numpy as np

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class cam_param_encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, embed_dims):
        super(cam_param_encoder, self).__init__()
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.context_ch = self.embed_dims
        self.cam_param_len = 16 # The transformation matrix from camera pixel to the augmented target coordinate system.

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.context_conv = nn.Conv2d(mid_channels,
            self.context_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn = nn.BatchNorm1d(self.cam_param_len)

        self.context_mlp = Mlp(self.cam_param_len, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)

    def forward(self, feat, cam_params):
        '''
        Input:
            feat: shape (B, N, C, H, W)
        Output:
            context: (B, N, C, H, W)
        '''
        vb, C, H, W = feat.shape
        cam_params = cam_params.view(vb,-1)   # Left shape: (B*N, 16)

        mlp_input = self.bn(cam_params) # mlp_input shape: (B * N, 16)
        feat = self.reduce_conv(feat)   # feat shape: (B * N, mid_ch, H, W)
        
        context_se = self.context_mlp(mlp_input)[..., None, None]   # context_se shape: (B * N, mid_ch, 1, 1)
        context = self.context_se(feat, context_se)
        context = self.context_conv(context)    # context_se shape: (B * N, context_ch, H, W)

        context = context.view(vb, context.shape[-3], context.shape[-2], context.shape[-1])

        return context