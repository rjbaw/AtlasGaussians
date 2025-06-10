# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched

from models.clip import clip
from models.models_lp import DiagonalGaussianDistribution

scale_factor = 1.0 / 1.33  # NOTE: this scale is specific to Objaverse, NOT ShapeNet

def train_one_epoch(model: torch.nn.Module, ae: torch.nn.Module, criterion: torch.nn.Module, clip_model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    null_token = clip.tokenize([''], truncate=True).to(device)  # [1, 77]
    with torch.no_grad():
        null_features = clip_model.encode_text(null_token).float().unsqueeze(1)  # [1, 1, 512]
    model_kwargs = {'uncond': null_features}

    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate2(optimizer, data_iter_step / len(data_loader) + epoch, args)

        data_dict = misc.to_device(data_dict, device)

        with torch.amp.autocast(device_type="cuda", enabled=False):

            posterior = DiagonalGaussianDistribution(data_dict['mean'], data_dict['logvar'])
            x = posterior.sample()
            x = x * scale_factor

            # batch_text = [data_loader.dataset.texts[data_loader.dataset.models[idx]] for idx in data_dict['idx']]
            # text_token = clip.tokenize(batch_text, truncate=True).to(device)  # [B, 77]

            # with torch.no_grad():
            #     text_features = clip_model.encode_text(text_token).float().unsqueeze(1)  # [B, 1, 512]

            if hasattr(data_loader.dataset, 'texts'):
                batch_text = [
                    data_loader.dataset.texts[data_loader.dataset.models[idx]]
                    for idx in data_dict['idx']
                ]
                text_token = clip.tokenize(batch_text, truncate=True).to(device)  # [B, 77]
                with torch.no_grad():
                    text_features = (
                        clip_model.encode_text(text_token)
                        .float()
                        .unsqueeze(1)
                    )  # [B, 1, 512]
            else:
                B = data_dict['mean'].shape[0]
                text_features = null_features.expand(B, -1, -1)  # [B, 1, 512]

            loss = criterion(model, x, text_features, **model_kwargs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, ae, clip_model, criterion, device):


    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    null_token = clip.tokenize([''], truncate=True).to(device)  # [1, 77]
    with torch.no_grad():
        null_features = clip_model.encode_text(null_token).float().unsqueeze(1)  # [1, 1, 512]
    model_kwargs = {'uncond': null_features}

    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, 50, header)):

        data_dict = misc.to_device(data_dict, device)

        # compute output
        with torch.amp.autocast(device_type="cuda", enabled=False):

            posterior = DiagonalGaussianDistribution(data_dict['mean'], data_dict['logvar'])
            x = posterior.sample()
            x = x * scale_factor

            # batch_text = [data_loader.dataset.texts[data_loader.dataset.models[idx]] for idx in data_dict['idx']]
            # text_token = clip.tokenize(batch_text, truncate=True).to(device)

            # with torch.no_grad():
            #     text_features = clip_model.encode_text(text_token).float().unsqueeze(1)

            if hasattr(data_loader.dataset, 'texts'):
                batch_text = [
                    data_loader.dataset.texts[data_loader.dataset.models[idx]]
                    for idx in data_dict['idx']
                ]
                text_token = clip.tokenize(batch_text, truncate=True).to(device)  # [B, 77]
                with torch.no_grad():
                    text_features = (
                        clip_model.encode_text(text_token)
                        .float()
                        .unsqueeze(1)
                    )  # [B, 1, 512]
            else:
                B = data_dict['mean'].shape[0]
                text_features = null_features.expand(B, -1, -1)  # [B, 1, 512]

            loss = criterion(model, x, text_features, **model_kwargs)
            
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
