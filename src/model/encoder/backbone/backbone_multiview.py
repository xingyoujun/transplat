import torch
from einops import rearrange

from .unimatch.backbone import CNNEncoder
from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine

from ....geometry.epipolar_lines import get_depth
from ...encoder.matching.conversions import depth_to_relative_disparity

from ...utils import cam_param_encoder

def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list

class BackboneMultiview(torch.nn.Module):
    """docstring for BackboneMultiview."""
    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
        downscale_factor=8,
    ):
        super(BackboneMultiview, self).__init__()
        self.feature_channels = feature_channels

        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=1 if downscale_factor == 8 else 0,
        )

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )
        
        self.cam_param_encoder = cam_param_encoder(in_channels=128, mid_channels=128, embed_dims=128)

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(v)]
        for feature in features:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            for idx in range(v):
                features_list[idx].append(feature[:, idx])

        return features_list

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
        img2world=None,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of features

        cur_features_list = [x[0] for x in features_list]

        if return_cnn_features:
            cnn_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        # add cam param to features
        feature_list_with_cam = [] 
        for v_id, cur_features in enumerate(cur_features_list):
            feature_list_with_cam.append(self.cam_param_encoder(cur_features, img2world[:,v_id]))
        cur_features_list = feature_list_with_cam

        # add position to features
        cur_features_list = feature_add_position_list(
            cur_features_list, attn_splits, self.feature_channels)

        # Transformer
        cur_features_list = self.transformer(
            cur_features_list, attn_num_splits=attn_splits)

        features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if return_cnn_features:
            out_lists = [features, cnn_features]
        else:
            out_lists = [features, None]

        return out_lists
