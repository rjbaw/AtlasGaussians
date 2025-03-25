import numpy as np
from einops import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization_extended import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from util.graphics_utils import getWorld2View2, getProjectionMatrix

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


class GaussianModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg  # config.model.gs
        self.pos_act = lambda x: torch.clamp(x, min=-cfg.max_range, max=cfg.max_range)
        self.rgb_act = lambda x: torch.sigmoid(x)

        if cfg.isotropic:
            self.rot_act = lambda x: torch.cat((torch.ones_like(x[..., 0:1]), torch.zeros_like(x[..., 1:4])), dim=-1)
        else:
            self.rot_act = lambda x: F.normalize(x, dim=-1)

        if cfg.const_opacity:
            self.opacity_act = lambda x: torch.ones_like(x)
        else:
            self.opacity_act = lambda x: torch.sigmoid(x)

        # IMPORTANT NOTE: We have lots of 3DGS whose scales need to be initialized properly.
        # If using a larger scale weight, e.g. 0.1, rendering will consume very large GPU memory.
        if cfg.const_scale:
            self.scale_act = lambda x: torch.ones_like(x) * 0.005
        else:
            self.scale_act = lambda x: 0.01 * F.softplus(x)

        # IMPORTANT NOTE: the saved values from origianl 3DGS are SH coefficients instead of colors. GaussianRasterizer uses RGB for shs.
        # self.scale_act_ori_3dgs = lambda x: torch.exp(x)
        # self.rgb_act_ori_3dgs = lambda x: SH2RGB(x)

    def forward(self, feats):
        '''
        Args:
            feats: [..., 14]
        Returns:
            gaussians: [..., 14], the activated values
        '''
        assert(feats.shape[-1] == 14)

        pos = self.pos_act(feats[..., :3])  # [..., 3]
        opacity = self.opacity_act(feats[..., 3:4])
        scale = self.scale_act(feats[..., 4:7])
        # IMPORTANT TODO: check whether we have to use uniform scaling
        if self.cfg.isotropic:
            scale = repeat(torch.mean(scale, dim=-1, keepdim=True), '... 1 -> ... d', d=3)  # [..., 3]
        rotation = self.rot_act(feats[..., 7:11])
        rgbs = self.rgb_act(feats[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [..., 14]

        return gaussians


def gs_render(gaussians, R, T, fov_rad, output_size, bg_color=None, scale_modifier=1):
    '''
    Args:
        gaussians: [B, N, 14]
        R: [B, V, 3, 3], extrinsics
        T: [B, V, 3], extrinsics
    Returns:
        images: [B, V, 3, H, W]
        depths: [B, V, 1, H, W]
        alphas: [B, V, 1, H, W]
    '''
    device = gaussians.device
    B, V = R.shape[:2]

    if bg_color is None:
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # loop of loop...
    images = []
    depths = []
    alphas = []
    for b in range(B):

        # pos, opacity, scale, rotation, shs
        means3D = gaussians[b, :, 0:3].contiguous().float()
        opacity = gaussians[b, :, 3:4].contiguous().float()
        scales = gaussians[b, :, 4:7].contiguous().float()
        rotations = gaussians[b, :, 7:11].contiguous().float()
        rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

        for v in range(V):
            
            # render novel views
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=10.0, fovX=fov_rad, fovY=fov_rad).transpose(0,1).cuda()
            tanfovx = np.tan(fov_rad * 0.5)
            tanfovy = np.tan(fov_rad * 0.5)

            # IMPORTANT NOTE: we need to transpose here!!!
            # world_view_transform = torch.tensor(getWorld2View2(R[b, v].T, T[b, v], np.array([0, 0, 0]), 1.0)).transpose(0, 1).cuda()
            # NOTE: the following should have the same effects, verified by visualization.
            world_view_transform = torch.eye(4, device=device)
            world_view_transform[:3, :3] = R[b, v]
            world_view_transform[:3, 3] = T[b, v]
            world_view_transform = world_view_transform.transpose(0, 1)  # GaussianRasterizer format
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            raster_settings = GaussianRasterizationSettings(
                image_height=output_size,
                image_width=output_size,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scale_modifier,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=0,
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D,
                means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                shs=None,
                colors_precomp=rgbs,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )

            rendered_image = rendered_image.clamp(0, 1)

            images.append(rendered_image)
            depths.append(rendered_depth)
            alphas.append(rendered_alpha)

    images = torch.stack(images, dim=0).view(B, V, 3, output_size, output_size)
    depths = torch.stack(depths, dim=0).view(B, V, 1, output_size, output_size)
    alphas = torch.stack(alphas, dim=0).view(B, V, 1, output_size, output_size)

    return {
        "images": images, # [B, V, 3, H, W]
        "depths": depths, # [B, V, 1, H, W]
        "alphas": alphas, # [B, V, 1, H, W]
    }


