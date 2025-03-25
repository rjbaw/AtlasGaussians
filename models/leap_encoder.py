import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
import math
import random
from models.backbone import build_backbone, BackboneOutBlock
from models.encoder import CrossViewEncoder


class LEAP_Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # input and output size
        self.input_size = config.dataset.img_size_input
        
        # build backbone
        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)
        self.backbone_name = config.model.backbone_name
        self.backbone_out_dim = config.model.backbone_out_dim
        self.backbone_out = BackboneOutBlock(in_dim=self.backbone_dim, out_dim=self.backbone_out_dim)
        self.feat_res = int(self.input_size // self.down_rate)

        if config.model.use_cross_view_refinement:
            # build cross-view feature encoder
            assert config.model.encoder_layers > 0
            self.encoder = CrossViewEncoder(config, in_dim=self.backbone_out_dim, in_res=self.feat_res)
        else:
            assert config.model.encoder_layers == 0

    def extract_feature(self, x, return_h_w=False):
        if self.backbone_name == 'dinov2':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]  # [B, 16*16, 768]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])  # patch_size = 14
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)  # [B, 768, 16, 16]
        else:
            raise NotImplementedError('unknown image backbone')
        return out

    def forward(self, imgs):
        '''
        Args:
            imgs: [B, V, C, H, W]
        Returns:
            features: [B, V, C, self.feat_res, self.feat_res]
        '''
        assert (len(imgs.shape) == 5)
        B = imgs.shape[0]
        
        # 2D per-view feature extraction
        imgs = rearrange(imgs, 'b v c h w -> (b v) c h w')
        if self.config.model.backbone_fix:
            with torch.no_grad():
                features = self.extract_feature(imgs)                     # [b*v, c=768, h, w]
        else:
            features = self.extract_feature(imgs)
        features = self.backbone_out(features)
        features = rearrange(features, '(b v) c h w -> b v c h w', b=B)    # [b, v, c, h, w]

        if self.config.model.use_cross_view_refinement:
            # cross-view feature refinement
            features = self.encoder(features)                                  # [b, v, c, h, w]

        return features


