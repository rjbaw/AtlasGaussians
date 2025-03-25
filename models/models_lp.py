from functools import wraps
import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch import nn
import pytorch3d.ops
from torch_cluster import fps

from util.lpips import LPIPS
from models.base_module.TransformerDecoder import TransformerDecoderLayerAda
from models.leap_encoder import LEAP_Encoder


class PointFourierEncoder(nn.Module):
    def __init__(self, point_dim=3, num_freq=8, dim=128):
        ''' PE for point of shape [..., point_dim]
        '''
        super().__init__()
        hidden_dim = point_dim * num_freq * 2  # sin + cos, so *2
        scales = torch.pow(2, torch.arange(num_freq)).float() * np.pi  # [8,]
        self.register_buffer('scales', scales)
        self.mlp = nn.Linear(hidden_dim + point_dim, dim)

    def forward(self, points):
        '''
        Args:
            points: [..., N, point_dim]
        Returns:
            x: [..., N, dim]
        '''
        x = points[..., None] * self.scales  # [..., N, point_dim, 8]
        x = rearrange(x, '... d l -> ... (d l)')  # [..., N, point_dim * 8]
        x = torch.cat([x.sin(), x.cos()], dim = -1)  # [..., N, point_dim * 8 * 2]
        x = torch.cat((x, points), dim = -1)  # [..., N, hidden_dim + point_dim]
        x = self.mlp(x)  # [..., N, dim]
        return x


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class LPSNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        depth,
        use_uv_pe=True,
        norm_first=None,
        bias=True,
    ):
        super().__init__()

        self.depth = depth
        self.use_uv_pe = use_uv_pe

        self.up_fc = nn.Linear(in_dim, 4 * out_dim)
        self.global_fc = nn.Linear(in_dim, out_dim)
        self.anchor_fc = nn.Linear(in_dim, out_dim)

        # NOTE: don't use bias=False for encoder layer: https://github.com/pytorch/pytorch/issues/116385
        self.geom_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=int(out_dim * 4),
                                       dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            for _ in range(depth)
        ])
        self.attr_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=int(out_dim * 4),
                                       dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            for _ in range(depth)
        ])
        self.global_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=int(out_dim * 4),
                                       dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            for _ in range(depth)
        ])
        assert len(self.geom_layers) == len(self.attr_layers) == len(self.global_layers)

        self.anchor_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=int(out_dim * 4),
                                       dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            for _ in range(4)
        ])

        self.to_anchor = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 3),
        )

    def forward(self, x, uv_pe=None):
        '''
        Args:
            x: [B, N=2048, in_dim]
            uv_pe: [4, out_dim]
        Returns:
            anchors: [B, N, 3]
            anchor_feats_geom: [B, N, p=4, out_dim]
            anchor_feats_attr: [B, N, p=4, out_dim]
        '''
        B, N = x.shape[:2]

        # gen anchors
        x_anchor = F.gelu(self.anchor_fc(x))  # [B, N, out_dim]
        x_anchor = self.anchor_layers(x_anchor)
        anchors = self.to_anchor(x_anchor)  # [B, N, 3]

        # gen up features
        x_up = F.gelu(self.up_fc(x))  # [B, N, 4*out_dim]
        x_up = rearrange(x_up, 'b n (p d) -> b n p d', p=4)  # [B, N, 4, out_dim]
        if self.use_uv_pe:
            assert (uv_pe is not None)
            assert (uv_pe.shape[-1] == x_up.shape[-1])
            uvs_pe = repeat(uv_pe, 'p d -> b n p d', b=x_up.shape[0], n=x_up.shape[1])  # [B, N, 4, out_dim]
            x_up += uvs_pe

        x_geom, x_attr = x_up, x_up  # [B, N, 4, out_dim]
        x_global = F.gelu(self.global_fc(x))  # [B, N, out_dim]

        for i in range(self.depth):
            x_global = self.global_layers[i](x_global)  # [B, N, out_dim]

            x_geom += repeat(x_global, 'b n d -> b n p d', p=4)
            x_geom = self.geom_layers[i](rearrange(x_geom, 'b n p d -> (b n) p d'))  # [B*N, 4, out_dim]
            x_geom = rearrange(x_geom, '(b n) p d -> b n p d', b=B)  # [B, N, 4, out_dim]

            x_attr += repeat(x_global, 'b n d -> b n p d', p=4)
            x_attr = self.attr_layers[i](rearrange(x_attr, 'b n p d -> (b n) p d'))  # [B*N, 4, out_dim]
            x_attr = rearrange(x_attr, '(b n) p d -> b n p d', b=B)  # [B, N, 4, out_dim]

        anchor_feats_geom = x_geom
        anchor_feats_attr = x_attr

        return anchors, anchor_feats_geom, anchor_feats_attr


class GS_Decoder(nn.Module):
    def __init__(
        self,
        dim,
        norm_first,
        bias,
    ):
        super().__init__()

        self.norm_first = norm_first

        self.decoder_cross_attn = nn.MultiheadAttention(dim, num_heads=4, dropout=0.0, batch_first=True, bias=bias)
        self.norm = nn.LayerNorm(dim)

        self.to_outputs = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.to_pos = nn.Linear(dim, 3)
        self.to_opacity = nn.Linear(dim, 1)
        self.to_scale = nn.Linear(dim, 3)
        self.to_rotation = nn.Linear(dim, 4)
        self.to_rgb = nn.Linear(dim, 3)

        self.init_layers()

    def init_layers(self):
        inverse_sigmoid = lambda x: np.log(x / (1 - x))
        # init opacity
        nn.init.constant_(self.to_opacity.weight, 0.0)
        nn.init.constant_(self.to_opacity.bias, inverse_sigmoid(0.1))
        # init scale
        nn.init.constant_(self.to_scale.weight, 0.0)
        nn.init.constant_(self.to_scale.bias, -1.0)  # 0.01 * softplus = 0.0031
        # init rotation
        nn.init.constant_(self.to_rotation.weight, 0.0)
        nn.init.constant_(self.to_rotation.bias, 0)
        nn.init.constant_(self.to_rotation.bias[0], 1.0)

    def forward(self, x, uvs):
        '''
        Args:
            x: [B, N, p, d], output features from LPSNet.
            uvs: [B, N, S, d], PE(samples) in uv.
        Returns:
            gs: [B, N, S, 14]
        '''
        B, N = x.shape[:2]
        assert (x.shape[-1] == uvs.shape[-1])

        query = rearrange(uvs, 'b n s d -> (b n) s d')
        key = value = rearrange(x, 'b n p d -> (b n) p d')

        if self.norm_first:
            feat = self.decoder_cross_attn(self.norm(query), key, value)[0]  # [B*N, S, d]
        else:
            feat = self.norm(self.decoder_cross_attn(query, key, value)[0])  # [B*N, S, d]
        feat = self.to_outputs(feat)  # [B*N, S, d]

        pos = self.to_pos(feat)
        opacity = self.to_opacity(feat)
        scale = self.to_scale(feat)
        rotation = self.to_rotation(feat)
        rgb = self.to_rgb(feat)

        gs = torch.cat([pos, opacity, scale, rotation, rgb], dim=-1)  # [B*N, S, 14]
        gs = rearrange(gs, '(b n) s d -> b n s d', b=B)  # [B, N, S, 14]

        return gs


class LatentNet(nn.Module):
    def __init__(
        self,
        num_lp = 512,
        lp_dim = 512,
        out_dim = 256,
        latent_net_depth = 8,
        norm_first = None,
        bias = True,
    ):
        super().__init__()

        self.latent_patch = nn.Parameter(torch.randn(num_lp, lp_dim) * 0.02)

        self.ca_layer = nn.TransformerDecoderLayer(d_model=lp_dim, nhead=8, dim_feedforward=int(lp_dim * 4),
                                                   dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first, bias=bias)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=lp_dim, nhead=8, dim_feedforward=int(lp_dim * 4),
                                       dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            for _ in range(latent_net_depth)
        ])

        self.up_fc = nn.Linear(lp_dim, 4 * out_dim)

        self.up_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=out_dim, nhead=8, dim_feedforward=int(out_dim * 4),
                                       dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            for _ in range(4)
        ])

    def forward(self, z):
        '''
        Args:
            z: [B, num_lp, lp_dim], the latent set features from bottleneck
        Returns:
            x: [B, num_lp*4, out_dim]
        '''
        B = z.shape[0]
        x = repeat(self.latent_patch, 'n d -> b n d', b=B)  # [B, num_lp, lp_dim]
        assert x.shape == z.shape

        x = self.ca_layer(x, z)  # [B, num_lp, lp_dim]

        for layer in self.layers:
            x = layer(x)  # [B, num_lp, lp_dim]

        x_up = F.gelu(self.up_fc(x))  # [B, num_lp, 4*out_dim]
        x_up = rearrange(x_up, 'b n (p d) -> b (n p) d', p=4)  #[B, num_lp*4, out_dim]

        for layer in self.up_layers:
            x_up = layer(x_up)  # [B, num_lp*4, out_dim]

        return x_up


class LPNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_uv_pe = None,
        norm_first = None,
        bias = True,
    ):
        super().__init__()
        assert (use_uv_pe is not None and norm_first is not None)

        self.point_fourier_encoder = PointFourierEncoder(point_dim=2, num_freq=8, dim=out_dim)
        uv = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)  # [4, 2]
        self.register_buffer('uv', uv)

        self.lps_net = LPSNet(in_dim, out_dim, depth=4, use_uv_pe=use_uv_pe, norm_first=norm_first, bias=bias)

        self.gs_decoder_geom = GS_Decoder(out_dim, norm_first=norm_first, bias=bias)
        self.gs_decoder_attr = GS_Decoder(out_dim, norm_first=norm_first, bias=bias)

    def query_decode(self, queries, anchors, anchor_feats_geom, anchor_feats_attr):
        '''
        Args:
            queries: [B, N, S, 2], query uvs
            anchors: [B, N, 3]
            anchor_feats_geom: [B, N, 4, out_dim]
            anchor_feats_attr: [B, N, 4, out_dim]
        Returns:
            gs: [B, N, S, 14]
        '''
        assert (queries.shape[:2] == anchors.shape[:2] == anchor_feats_geom.shape[:2] == anchor_feats_attr.shape[:2])
        assert len(queries.shape) == 4 and len(anchors.shape) == 3 and len(anchor_feats_geom.shape) == 4 and len(anchor_feats_attr.shape) == 4

        query_uv_pe = self.point_fourier_encoder(queries)  # [B, N, S, out_dim]
        gs_geom = self.gs_decoder_geom(anchor_feats_geom, query_uv_pe)  # [B, N, S, 14]
        gs_geom[:, :, :, :3] += repeat(anchors, 'b n d -> b n s d', s=gs_geom.shape[-2])

        gs_attr = self.gs_decoder_attr(anchor_feats_attr, query_uv_pe)  # [B, N, S, 14]
        gs = torch.cat((gs_geom[:, :, :, :3], gs_attr[:, :, :, 3:]), dim=-1)

        return gs

    def forward(self, x, queries):
        '''
        Args:
            x: [B, N=2048, in_dim], features
            queries: [B, N, S, 2], query uvs
        Returns:
            gs: [B, N, S, 14]
        '''
        assert (len(queries.shape) == 4)
        assert (queries.shape[-1] == 2)
        assert (queries.shape[:2] == x.shape[:2])

        anchor_uv_pe = self.point_fourier_encoder(self.uv)  # [4, out_dim]
        anchors, anchor_feats_geom, anchor_feats_attr = self.lps_net(x, anchor_uv_pe)  # [B, N, 3], [B, N, 4, out_dim], [B, N, 4, out_dim]

        gs = self.query_decode(queries, anchors, anchor_feats_geom, anchor_feats_attr)  # [B, num_lp*4, S, 14]

        # IMPORTANT NOTE: use the below for line memory profile
        # with LineProfiler(self.hier_occ_decoder.forward) as prof:
        #     occ = self.hier_occ_decoder(feats3, gs3, queries, gs1[:, :, :3])  # [B, Q]
        # prof.display()
        # from IPython import embed; embed()

        return gs, anchors, anchor_feats_geom, anchor_feats_attr


class KLAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.img_encoder = LEAP_Encoder(config)

        self.point_embed = PointFourierEncoder(point_dim=3, num_freq=8, dim=config.model.lp_dim)
        self.pc_encoder = nn.TransformerDecoderLayer(
            d_model=config.model.lp_dim, nhead=8, dim_feedforward=int(config.model.lp_dim * 4),
            dropout=0.0, activation='gelu', batch_first=True, norm_first=config.model.norm_first, bias=config.model.bias
        )

        self.embed_layers = nn.ModuleList([
            TransformerDecoderLayerAda(
                d_model=config.model.lp_dim, nhead=8, kv_dim=config.model.backbone_out_dim,
                dim_feedforward=int(config.model.lp_dim * 4), dropout=0.0, activation='gelu',
                batch_first=True, norm_first=config.model.norm_first, bias=config.model.bias,
            ) for _ in range(config.model.embed_layers_depth)
        ])

        self.enc_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=config.model.lp_dim, nhead=8, dim_feedforward=int(config.model.lp_dim * 4),
                                       dropout=0.0, activation='gelu', batch_first=True, norm_first=config.model.norm_first)
            for _ in range(3)
        ])

        if not self.config.model.deterministic:
            self.mean_fc = nn.Linear(config.model.lp_dim, config.model.latent_dim)
            self.logvar_fc = nn.Linear(config.model.lp_dim, config.model.latent_dim)
            self.proj = nn.Linear(config.model.latent_dim, config.model.lp_dim)

        dim1 = config.model.lp_dim // 2
        dim2 = dim1 // 2

        self.latent_net = LatentNet(config.model.num_lp, config.model.lp_dim, dim1, config.model.latent_net_depth, config.model.norm_first, config.model.bias)

        self.lp_net = LPNet(dim1, dim2, config.model.use_uv_pe, config.model.norm_first, config.model.bias)

        self.lpips_loss = LPIPS(net='vgg')
        self.lpips_loss.requires_grad_(False)

    def encode_pc(self, pc):
        '''
        Args:
            pc: [B, N, 3]
        Returns:
            x: [B, num_lp, lp_dim]
        '''
        B, N = pc.shape[:2]
        # assert N == 2048
        
        ###### fps
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = rearrange(pc, 'b n d -> (b n) d')
        ratio = 1.0 * self.config.model.num_lp / N
        idx = fps(pos, batch, ratio=ratio)

        ######
        pc_embeddings = self.point_embed(pc)  # [B, N, lp_dim]
        sampled_pc_embeddings = rearrange(pc_embeddings, 'b n d -> (b n) d')
        sampled_pc_embeddings = sampled_pc_embeddings[idx]
        sampled_pc_embeddings = rearrange(sampled_pc_embeddings, '(b n) d -> b n d', b=B)

        x = self.pc_encoder(sampled_pc_embeddings, pc_embeddings)
        assert (x.shape[1] == self.config.model.num_lp and x.shape[2] == self.config.model.lp_dim)

        return x

    def encode(self, imgs, pc, return_all=False):
        '''
        Args:
            imgs: [B, V, C, H, W]
            pc: [B, N, 3]
        Returns:
        '''
        B, V = imgs.shape[:2]
        # encode images
        feats = self.img_encoder(imgs)
        feats = rearrange(feats, 'b v c h w -> b (v h w) c')  # [B, V*h*w, C]
        # encode points
        x = self.encode_pc(pc)  # [B, num_lp, lp_dim]
        # multi-modality fusion
        for layer in self.embed_layers:
            x = layer(x, feats)  # [B, num_lp, lp_dim]

        x = self.enc_layers(x)  # [B, num_lp, lp_dim]

        if not self.config.model.deterministic:
            mean = self.mean_fc(x)
            logvar = self.logvar_fc(x)
            posterior = DiagonalGaussianDistribution(mean, logvar)
            x = posterior.sample()
            kl = posterior.kl()
        else:
            kl = None

        if return_all:
            return kl, x, mean, logvar
        else:
            return kl, x

    def decode(self, x, queries):

        if not self.config.model.deterministic:
            x = self.proj(x)  # [B, num_lp, lp_dim]

        features = self.latent_net(x)  # [B, num_lp*4, dim1]

        gs, anchors, anchor_feats_geom, anchor_feats_attr = self.lp_net(features, queries)

        return {'features': features, 'gs': gs, 'anchors': anchors, 'anchor_feats_geom': anchor_feats_geom, 'anchor_feats_attr': anchor_feats_attr}

    def forward(self, imgs, pc, queries):
        kl, x = self.encode(imgs, pc)

        out_dict = self.decode(x, queries)

        out_dict['latents'] = x
        out_dict['kl'] = kl

        return out_dict





