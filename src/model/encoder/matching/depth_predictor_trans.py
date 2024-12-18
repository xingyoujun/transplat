import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel

from ...utils import UVTransformer, cam_param_encoder

def calculate_grid(
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3 # vb 3 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    return grid, depth

def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().detach().inverse()
                    @ extrinsics[:, v0].clone().detach()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)
    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0].clone().detach()
        pose_tgt = extrinsics[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
        * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr

class DepthPredictorTrans(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        DA_size=128,
        **kwargs,
    ):
        super(DepthPredictorTrans, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor

        # Cost volume refinement: 2D U-Net
        input_channels = num_depth_candidates + feature_channels
        channels = self.regressor_feat_dim
        modules = [
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=costvolume_unet_attn_res,
                channel_mult=costvolume_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
        ]
        self.corr_refine_net = nn.Sequential(*modules)
        # cost volume u-net skip connection
        self.regressor_residual = nn.Conv2d(input_channels, num_depth_candidates, 1, 1, 0)

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # CNN-based feature upsampler
        proj_in_channels = feature_channels + feature_channels
        upsample_out_channels = feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + 1 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        self.refine_unet = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1, 
                attention_resolutions=depth_unet_attn_res,
                channel_mult=depth_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
        )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        channels = depth_unet_feat_dim
        disps_models = [
            nn.Conv2d(channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
        ]
        self.to_disparity = nn.Sequential(*disps_models)

        self.embed_dims = 128

        self.coarse_transformer = UVTransformer(embed_dims=self.embed_dims, mode='coarse', num_layers=1)
        self.fine_transformer = UVTransformer(embed_dims=self.embed_dims, mode='fine', num_layers=2)

        self.cam_param_encoder = cam_param_encoder(in_channels=DA_size, mid_channels=128, embed_dims=128)
        
    def match_two(self, intr_curr, pose_curr, extrinsics, disp_candi_curr, dino_feature, features):
        b, v, c, h, w = features.shape
        dtype = features.dtype
        grid, depth = calculate_grid(
            intr_curr,
            pose_curr,
            1.0 / disp_candi_curr.repeat([1, 1, *features.shape[-2:]]),
        )  # [B, C, D, H, W]

        raw_correlation_in = torch.zeros((v*64*64,b, self.embed_dims), device=features.device).to(dtype)

        camk = torch.eye(4).view(1,4,4).repeat(intr_curr.shape[0], 1, 1).to(intr_curr.device).float()
        camk[:,:3,:3] = intr_curr
        c2w = extrinsics.clone().detach()
        c2w = rearrange(c2w, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]
        camk = torch.inverse(camk)
        img2world = torch.matmul(c2w, camk)
        img2world = img2world.reshape(-1,16) # vb 16
        pos_feature = self.cam_param_encoder(dino_feature, img2world)

        bev_pos = pos_feature.reshape(v, b, c, h, w)
        bev_pos = bev_pos.permute(0,3,4,1,2)
        bev_pos = bev_pos.reshape(-1,b,c) # vhw b c

        # coarse matching
        raw_correlation_in = self.coarse_transformer(
                [features],
                raw_correlation_in,
                64,
                64,
                bev_pos=None,
                intrinsics=None,
                extrinsics=None,
                depth_sup=None,
                grid=grid,
            )

        # coarse-to-fine matching
        raw_correlation_in = self.fine_transformer(
                [features],
                raw_correlation_in,
                64,
                64,
                bev_pos=bev_pos,
                intrinsics=None,
                extrinsics=None,
                depth_sup=None,
                grid=grid,
            )

        raw_correlation_in = raw_correlation_in.reshape(v, h, w, b, c)
        raw_correlation_in = raw_correlation_in.permute(3,0,4,1,2)
        raw_correlation_in = rearrange(raw_correlation_in, "b v ... -> (v b) ...")
        
        return raw_correlation_in
        
    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
        da_depth=None,
        dino_feature=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""
        # format the input
        b, v, c, h, w = features.shape
        dtype = features.dtype
        
        if da_depth is not None:
            da_depth = rearrange(da_depth, "b v ... -> (v b) ...")
        if cnn_features is not None:
            cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")
        if dino_feature is not None:
            dino_feature = rearrange(dino_feature, "b v ... -> (v b) ...")
            dino_feature = F.interpolate(
                dino_feature,
                size=(64,64),
                mode="bilinear",
                align_corners=True,
            )
        
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )
        # cost volume constructions
        feat01 = feat_comb_lists[0]
        
        if v == 2:
            raw_correlation_in = self.match_two(intr_curr, pose_curr_lists[0], extrinsics, disp_candi_curr, dino_feature, features)
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
        elif v == 3:
            raw_correlation_in_part_list = []
            for i in range(3):
                ind_1 = i
                ind_2 = (i + 1) % 3
                # pose_curr_lists [0-1,1-2,2-0] [0-2,1-0,2-1]
                pose_cur = torch.stack((pose_curr_lists[0][i],pose_curr_lists[1][ind_2]), dim=0)
                intr_cur = torch.stack((intr_curr[ind_1],intr_curr[ind_2]), dim=0)
                extrinsics_cur = torch.cat((extrinsics[:,ind_1:ind_1+1], extrinsics[:,ind_2:ind_2+1]),dim=1)
                disp_candi_cur = torch.stack((disp_candi_curr[ind_1],disp_candi_curr[ind_2]), dim=0)
                dino_feature_cur = torch.stack((dino_feature[ind_1],dino_feature[ind_2]), dim=0)
                feature_cur = torch.cat((features[:,ind_1:ind_1+1], features[:,ind_2:ind_2+1]),dim=1)
                raw_correlation_in_part = self.match_two(intr_cur, pose_cur, extrinsics_cur, disp_candi_cur, dino_feature_cur, feature_cur)
                raw_correlation_in_part_list.append(raw_correlation_in_part)
            raw_correlation_in_list = []
            for i in range(3):
                ind_1 = i
                ind_2 = (i + 2) % 3
                raw_correlation_in_i = torch.mean(torch.stack((raw_correlation_in_part_list[ind_1][0],raw_correlation_in_part_list[ind_2][1]), dim=0), dim=0)
                raw_correlation_in_list.append(raw_correlation_in_i)
            raw_correlation_in = torch.stack(raw_correlation_in_list, dim=0)
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
        elif v == 4:
            raw_correlation_in_part_list = []
            for i in range(4):
                ind_1 = i
                ind_2 = (i + 1) % 4
                # pose_curr_lists [0-1,1-2,2-3,3-0] [0-2,1-3,2-0,3-1] [0-3,1-0,2-1,3-2]
                pose_cur = torch.stack((pose_curr_lists[0][ind_1],pose_curr_lists[2][ind_2]), dim=0)
                intr_cur = torch.stack((intr_curr[ind_1],intr_curr[ind_2]), dim=0)
                extrinsics_cur = torch.cat((extrinsics[:,ind_1:ind_1+1], extrinsics[:,ind_2:ind_2+1]),dim=1)
                disp_candi_cur = torch.stack((disp_candi_curr[ind_1],disp_candi_curr[ind_2]), dim=0)
                dino_feature_cur = torch.stack((dino_feature[ind_1],dino_feature[ind_2]), dim=0)
                feature_cur = torch.cat((features[:,ind_1:ind_1+1], features[:,ind_2:ind_2+1]),dim=1)
                raw_correlation_in_part = self.match_two(intr_cur, pose_cur, extrinsics_cur, disp_candi_cur, dino_feature_cur, feature_cur)
                raw_correlation_in_part_list.append(raw_correlation_in_part)
            for i in range(2):
                ind_1 = i
                ind_2 = (i + 2) % 4
                # pose_curr_lists [0-1,1-2,2-3,3-0] [0-2,1-3,2-0,3-1] [0-3,1-0,2-1,3-2]
                pose_cur = torch.stack((pose_curr_lists[1][ind_1],pose_curr_lists[1][ind_2]), dim=0)
                intr_cur = torch.stack((intr_curr[ind_1],intr_curr[ind_2]), dim=0)
                extrinsics_cur = torch.cat((extrinsics[:,ind_1:ind_1+1], extrinsics[:,ind_2:ind_2+1]),dim=1)
                disp_candi_cur = torch.stack((disp_candi_curr[ind_1],disp_candi_curr[ind_2]), dim=0)
                dino_feature_cur = torch.stack((dino_feature[ind_1],dino_feature[ind_2]), dim=0)
                feature_cur = torch.cat((features[:,ind_1:ind_1+1], features[:,ind_2:ind_2+1]),dim=1)
                raw_correlation_in_part = self.match_two(intr_cur, pose_cur, extrinsics_cur, disp_candi_cur, dino_feature_cur, feature_cur)
                raw_correlation_in_part_list.append(raw_correlation_in_part)
            raw_correlation_in_list = []
            for i in range(2):
                ind_1 = i
                ind_2 = (i + 4) % 6
                ind_3 = (i + 3) % 4
                raw_correlation_in_i = torch.mean(torch.stack((raw_correlation_in_part_list[ind_1][0],raw_correlation_in_part_list[ind_2][0], raw_correlation_in_part_list[ind_3][1]), dim=0), dim=0)
                raw_correlation_in_list.append(raw_correlation_in_i)
            for i in range(2):
                ind_1 = i + 2
                ind_2 = i + 1
                ind_3 = i + 4
                raw_correlation_in_i = torch.mean(torch.stack((raw_correlation_in_part_list[ind_1][0],raw_correlation_in_part_list[ind_2][1], raw_correlation_in_part_list[ind_3][1]), dim=0), dim=0)
                raw_correlation_in_list.append(raw_correlation_in_i)
            raw_correlation_in = torch.stack(raw_correlation_in_list, dim=0)
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)

        # refine cost volume via 2D u-net
        raw_correlation = self.corr_refine_net(raw_correlation_in)  # (vb d h w)
        # apply skip connection
        raw_correlation = raw_correlation + self.regressor_residual(raw_correlation_in)
        
        # softmax to get coarse depth and density
        pdf = F.softmax(
            self.depth_head_lowres(raw_correlation), dim=1
        )  # [2xB, D, H, W]
        coarse_disps = (disp_candi_curr * pdf).sum(
            dim=1, keepdim=True
        )  # (vb, 1, h, w)
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
        pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)
        fullres_disps = F.interpolate(
            coarse_disps,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=True,
        )

        # depth refinement
        proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1))
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        refine_out = self.refine_unet(torch.cat((extra_info["images"], da_depth, proj_feature, fullres_disps, pdf_max), dim=1))

        # gaussians head
        raw_gaussians_in = [refine_out, extra_info["images"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(
            raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        )

        # delta fine depth and density
        delta_disps_density = self.to_disparity(refine_out)
        delta_disps, raw_densities = delta_disps_density.split(gaussians_per_pixel, dim=1)

        # combine coarse and fine info and match shape
        densities = repeat(
            F.sigmoid(raw_densities),
            "(v b) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        )

        fine_disps = (fullres_disps + delta_disps).clamp(
            1.0 / rearrange(far, "b v -> (v b) () () ()"),
            1.0 / rearrange(near, "b v -> (v b) () () ()"),
        )
        depths = 1.0 / fine_disps
        depths = repeat(
            depths,
            "(v b) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        )

        return depths, densities, raw_gaussians
