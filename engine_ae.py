# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import os
import sys
from typing import Iterable
from tqdm import tqdm
import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
import pytorch3d.loss

import util.misc as misc
import util.lr_sched as lr_sched
from util.image_utils import compute_psnr
from util.loss_utils import chamfer_distance
from util.geom_utils import build_grid2D
from util.lpips import LPIPS
from util.emd.emd_module import earth_mover_distance as EMD
from gs import GaussianModel, gs_render

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def compute_loss(data_dict, outputs_dict, gaussians_lp, gaussians_render, model, config, epoch, device):
    # gaussians_lp are used to compute chamfer distance, gaussians_render are used to compute rendering loss

    # kl loss
    loss_kl = torch.zeros(1, dtype=torch.float32, device=device).mean()
    if 'kl' in outputs_dict and outputs_dict['kl'] is not None:
        loss_kl = outputs_dict['kl']
        loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
    outputs_dict['loss_kl'] = loss_kl

    ######################## chamfer loss ########################
    assert data_dict['points'].shape[1] == (2048 + (config.loss.sample_cd_iter + 1) * 8192)  # extra 8192 for xyz_render
    # anchor
    loss_lp0, _ = pytorch3d.loss.chamfer_distance(outputs_dict['anchors'], data_dict['points'][:, :2048, :])
    outputs_dict['loss_lp0'] = loss_lp0

    # lp
    assert (gaussians_lp.shape[1] == 2048)
    if config.loss.cd_sample_type == 'rand':
        assert config.loss.num_samples_cd == 4  # ensure 8192 gaussians_lp in total
        xyz_lp = rearrange(gaussians_lp[..., :3], 'b n (m s) d -> (b m) n s d', m=config.loss.sample_cd_iter, s=config.loss.num_samples_cd)  # [B*m, N=2048, S, 3]
        xyz_gt = rearrange(data_dict['points'][:, 2048:2048+config.loss.sample_cd_iter*8192, :], 'b (m n s) d -> (b m) n s d', m=config.loss.sample_cd_iter, s=config.loss.num_samples_cd)  # [B*m, N=2048, S=4, 3]
    else:
        raise NotImplementedError
    assert (xyz_lp.shape[-2] == 4) and (xyz_lp.shape == xyz_gt.shape)

    loss_lp1, _ = pytorch3d.loss.chamfer_distance(rearrange(xyz_lp[:, :, :1, :], 'b n s d -> b (n s) d'),
                                                  rearrange(xyz_gt[:, :, :1, :], 'b n s d -> b (n s) d'))
    loss_lp2, _ = pytorch3d.loss.chamfer_distance(rearrange(xyz_lp[:, :, :4, :], 'b n s d -> b (n s) d'),
                                                  rearrange(xyz_gt[:, :, :4, :], 'b n s d -> b (n s) d'))
    outputs_dict['loss_lp1'] = loss_lp1
    outputs_dict['loss_lp2'] = loss_lp2

    # gaussians_render
    xyz_render = rearrange(gaussians_render[:, :, :, :3], 'b n s d -> b (n s) d')  # [B, N*S, 3]
    rand_idx = np.random.default_rng().choice(xyz_render.shape[1], 8192, replace=False)
    loss_lp_render, _ = pytorch3d.loss.chamfer_distance(xyz_render[:, rand_idx, :], data_dict['points'][:, -8192:, :])
    outputs_dict['loss_lp_render'] = loss_lp_render

    ######################## EMD loss ########################
    if config.loss.emd_weight != 0.0:
        loss_lp_emd0 = EMD(outputs_dict['anchors'], data_dict['points'][:, :2048, :],
                           origin=config.loss.emd_origin, scale=config.loss.emd_scale, check_range=(epoch==0)).mean()
        loss_lp_emd2 = EMD(rearrange(xyz_lp[:, :, :4, :], 'b n s d -> b (n s) d'), rearrange(xyz_gt[:, :, :4, :], 'b n s d -> b (n s) d'),
                           origin=config.loss.emd_origin, scale=config.loss.emd_scale, check_range=(epoch==0)).mean()
        loss_lp_emd_render = EMD(xyz_render[:, rand_idx, :], data_dict['points'][:, -8192:, :],
                                 origin=config.loss.emd_origin, scale=config.loss.emd_scale, check_range=(epoch==0)).mean()
    else:
        loss_lp_emd0 = torch.zeros(1, dtype=torch.float32, device=device).mean()
        loss_lp_emd2 = torch.zeros(1, dtype=torch.float32, device=device).mean()
        loss_lp_emd_render = torch.zeros(1, dtype=torch.float32, device=device).mean()
    outputs_dict['loss_lp_emd0'] = loss_lp_emd0
    outputs_dict['loss_lp_emd2'] = loss_lp_emd2
    outputs_dict['loss_lp_emd_render'] = loss_lp_emd_render

    ######################## rendering loss ########################
    if config.loss.imagenet_background:
        bg_color = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32, device=device)
    else:
        bg_color = torch.ones(3, dtype=torch.float32, device=device)
    # rgb_gt: [B, V, H, W, 4]
    masks_gt = rearrange(data_dict['mask_gt'], 'b v h w d -> b v d h w')
    depths_gt = rearrange(data_dict['depth_gt'], 'b v h w d -> b v d h w')
    images_gt = rearrange(data_dict['rgb_gt'], 'b v h w d -> b v d h w')
    images_gt = images_gt * masks_gt + bg_color.view(1, 1, 3, 1, 1) * (1 - masks_gt)  # [B, V, 3, H, W]
    outputs_dict['masks_gt'] = masks_gt
    outputs_dict['depths_gt'] = depths_gt
    outputs_dict['images_gt'] = images_gt

    render_dict = gs_render(
        gaussians=rearrange(gaussians_render,  'b n s d -> b (n s) d'),
        R=data_dict['world2cams'][:, :, :3, :3],
        T=data_dict['world2cams'][:, :, :3, 3],
        fov_rad=data_dict['fov_rad'][0].item(),
        output_size=data_dict['img_size_render'][0].item(),
        bg_color=bg_color,
    )

    # mse loss
    loss_render_rgb = F.mse_loss(render_dict['images'], images_gt)
    loss_render_depth = F.mse_loss(render_dict['depths'], depths_gt)
    loss_render_mask = F.mse_loss(render_dict['alphas'], masks_gt)
    outputs_dict['loss_render_rgb'] = loss_render_rgb
    outputs_dict['loss_render_depth'] = loss_render_depth
    outputs_dict['loss_render_mask'] = loss_render_mask

    loss = loss_kl * config.loss.kl_weight + (loss_lp0 + loss_lp1 + loss_lp2 + loss_lp_render) * config.loss.cd_weight
    loss += (loss_lp_emd0 + loss_lp_emd2 + loss_lp_emd_render) * config.loss.emd_weight
    # Hack to receive gradients
    loss += model.module.img_encoder.backbone.mask_token.sum() * 0.0

    ######################## regularization loss ########################
    # NOTE: the loss is actually not used. I am too lazy to remove it.
    loss_scale_std = torch.std(gaussians_render[..., 4:7].mean(dim=-1))
    outputs_dict['loss_scale_std'] = loss_scale_std
    loss += loss_scale_std * config.loss.scale_std_weight

    # NOTE: the loss has little impact on (quantitative) training results. I am too lazy to remove it.
    xyz = rearrange(gaussians_lp[..., :3], 'b n s d -> (b n) s d')  # [B*N, S, 3]
    dmat = torch.linalg.norm(xyz[:, :, None, :] - xyz[:, None, :, :], dim=-1)  # [B*N, S, S]
    dmat_avg = torch.sum(dmat, dim=(1, 2), keepdim=True) / (dmat.shape[-1] ** 2 - dmat.shape[-1])  # [B*N, 1, 1]
    expand_mask = dmat > 2.5 * dmat_avg
    loss_expand = dmat[expand_mask].sum() / (expand_mask.sum() + 1e-6)
    outputs_dict['loss_expand'] = loss_expand
    loss += loss_expand * config.loss.expand_weight

    ######################## total loss ########################
    if epoch > config.loss.render_epochs:
        loss += loss_render_rgb * config.loss.rgb_weight + loss_render_mask * config.loss.mask_weight + loss_render_depth * config.loss.depth_weight

    assert (config.loss.render_epochs <= config.loss.lpips_epochs)
    if epoch > config.loss.lpips_epochs:
        # lpips loss: downsampled to at most 256 to reduce memory cost
        loss_lpips = model.module.lpips_loss(
            F.interpolate(rearrange(images_gt, 'b v d h w -> (b v) d h w') * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            F.interpolate(rearrange(render_dict['images'], 'b v d h w -> (b v) d h w') * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
        ).mean()
        outputs_dict['loss_lpips'] = loss_lpips
        loss += loss_lpips * config.loss.lpips_weight

    return loss, render_dict


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, config=None):
    assert (config is not None)
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config.train.accum_iter
    N = config.model.num_lp * 4
    num_samples = config.loss.num_samples

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    gaussian_model = GaussianModel(config.model.gs)

    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        data_dict = misc.to_device(data_dict, device)

        B = data_dict['rgb_inputs'].shape[0]

        with torch.cuda.amp.autocast(enabled=config.train.use_fp16):
            if config.loss.render_gs_type == 'grid':
                assert (config.loss.render_gs_type == 'grid')
                queries_grid = build_grid2D(vmin=0., vmax=1., res=int(np.sqrt(num_samples)), device=device).reshape(-1, 2)
                queries_grid = repeat(queries_grid, 's d -> b n s d', b=B, n=N)
                outputs_dict = model(data_dict['rgb_inputs'], data_dict['points_input'], queries_grid)  # [B, N, S, 14]
                gaussians_render = gaussian_model(outputs_dict['gs'])  # [B, N, S, 14], the activated values
            else:  # grid samples
                raise NotImplementedError()

            # random samples for geometry
            if config.loss.cd_sample_type == 'rand':
                num_samples_cd_total = config.loss.num_samples_cd * config.loss.sample_cd_iter
                queries_lp = torch.rand(B, N, num_samples_cd_total, 2, dtype=torch.float32, device=device)
                gs_lp = model.module.lp_net.query_decode(queries_lp, outputs_dict['anchors'], outputs_dict['anchor_feats_geom'], outputs_dict['anchor_feats_attr'])
                gaussians_lp = gaussian_model(gs_lp)  # [B, N, S, 14], the activated values
            else:
                raise NotImplementedError()

        if config.train.use_fp16:
            outputs_dict = {k: v.float() for k, v in outputs_dict.items()}
        loss, render_dict = compute_loss(data_dict, outputs_dict, gaussians_lp.float(), gaussians_render.float(), model, config, epoch, device)
        outputs_dict['loss'] = loss

        psnr = compute_psnr(render_dict['images'], outputs_dict['images_gt']).mean()  # [B, V] -> float

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in outputs_dict.items() if 'loss' in k})
        metric_logger.update(psnr=psnr.item())

        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_reduce_dict = {}
        for k, v in outputs_dict.items():
            if 'loss' in k:
                loss_reduce_dict[k+'_reduce'] = misc.all_reduce_mean(v.item())

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:  # Only True for main process
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            log_writer.add_scalar('psnr', psnr.item(), epoch_1000x)
            for k, v in loss_reduce_dict.items():
                log_writer.add_scalar(k.split('_reduce')[0], v, epoch_1000x)

            if (epoch % config.train.vis_interval == 0 or epoch == config.loss.render_epochs + 1) and (data_iter_step + 1) == accum_iter:
                log_writer.add_images('GT_images', outputs_dict['images_gt'][:, 0, :, :, :].detach().cpu(), epoch_1000x)
                log_writer.add_images('GT_depths', outputs_dict['depths_gt'][:, 0, :, :, :].detach().cpu() / 2.0, epoch_1000x)  # normalize depth into [0, 1]
                log_writer.add_images('GT_alphas', outputs_dict['masks_gt'][:,  0, :, :, :].detach().cpu(), epoch_1000x)
                log_writer.add_images('render_images', render_dict['images'][:, 0, :, :, :].detach().cpu(), epoch_1000x)
                log_writer.add_images('render_depths', render_dict['depths'][:, 0, :, :, :].detach().cpu() / 2.0, epoch_1000x)  # normalize depth into [0, 1]
                log_writer.add_images('render_alphas', render_dict['alphas'][:, 0, :, :, :].detach().cpu(), epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, config, epoch):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    N = config.model.num_lp * 4
    num_samples = config.loss.num_samples

    # switch to evaluation mode
    model.eval()

    log_img_dict = {}
    log_img_dict['GT'] = []
    log_img_dict['render_images'] = []
    log_img_dict['render_alpha'] = []

    gaussian_model = GaussianModel(config.model.gs)

    for data_dict in metric_logger.log_every(data_loader, 50, header):

        data_dict = misc.to_device(data_dict, device)

        B = data_dict['rgb_inputs'].shape[0]

        with torch.cuda.amp.autocast(enabled=config.train.use_fp16):
            if config.loss.render_gs_type == 'grid':
                assert (config.loss.render_gs_type == 'grid')
                queries_grid = build_grid2D(vmin=0., vmax=1., res=int(np.sqrt(num_samples)), device=device).reshape(-1, 2)
                queries_grid = repeat(queries_grid, 's d -> b n s d', b=B, n=N)
                outputs_dict = model(data_dict['rgb_inputs'], data_dict['points_input'], queries_grid)  # [B, N, S, 14]
                gaussians_render = gaussian_model(outputs_dict['gs'])  # [B, N, S, 14], the activated values
            else:  # grid samples
                raise NotImplementedError()

            # random samples for geometry
            if config.loss.cd_sample_type == 'rand':
                num_samples_cd_total = config.loss.num_samples_cd * config.loss.sample_cd_iter
                queries_lp = torch.rand(B, N, num_samples_cd_total, 2, dtype=torch.float32, device=device)
                gs_lp = model.module.lp_net.query_decode(queries_lp, outputs_dict['anchors'], outputs_dict['anchor_feats_geom'], outputs_dict['anchor_feats_attr'])
                gaussians_lp = gaussian_model(gs_lp)  # [B, N, S, 14], the activated values
            else:
                raise NotImplementedError()

        if config.train.use_fp16:
            outputs_dict = {k: v.float() for k, v in outputs_dict.items()}
        loss, render_dict = compute_loss(data_dict, outputs_dict, gaussians_lp.float(), gaussians_render.float(), model, config, epoch, device)
        outputs_dict['loss'] = loss

        if len(log_img_dict['GT']) < 16:
            log_img_dict['GT'].append(outputs_dict['images_gt'][0, 0, :, :, :].detach().cpu().numpy())
            log_img_dict['render_images'].append(render_dict['images'][0, 0, :, :, :].detach().cpu().numpy())
            log_img_dict['render_alpha'].append(render_dict['alphas'][0, 0, :, :, :].detach().cpu().numpy())

        psnr = compute_psnr(render_dict['images'], outputs_dict['images_gt']).mean()  # [B, V] -> float

        metric_logger.update(**{k: v.item() for k, v in outputs_dict.items() if 'loss' in k})
        metric_logger.meters['psnr'].update(psnr.mean().item(), n=B)  # use distribution mean not rigorous mean

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_img_dict = {k: np.stack(v, axis=0) for k, v in log_img_dict.items()}

    return test_stats, log_img_dict


@torch.no_grad()
def inference(data_loader, model, device, config, args):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Infer:'

    # switch to evaluation mode
    model.eval()

    epoch = args.start_epoch
    dump_root = f"{args.log_dir}/latents/epoch_{epoch}"

    for data_dict in metric_logger.log_every(data_loader, 50, header):

        data_dict = misc.to_device(data_dict, device)

        B = data_dict['rgb_inputs'].shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            _, _, latent_mean, latent_logvar = model.module.encode(data_dict['rgb_inputs'], data_dict['points_input'], return_all=True)  # [B, num_lp, latent_dim]

        pbar = tqdm(enumerate(data_dict['idx']))
        for b, idx in pbar:

            fidx = data_loader.dataset.models[idx]

            pbar.set_postfix({'b': b, 'fidx': fidx}, refresh=True)

            dump_dir = f"{dump_root}/{fidx}"
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
            dump_dict = {'mean': latent_mean[b].detach().cpu(), 'logvar': latent_logvar[b].detach().cpu()}
            torch.save(dump_dict, f"{dump_dir}/latent.pt")



