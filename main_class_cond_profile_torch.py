import argparse
import datetime
import json
import numpy as np
import os
import time
import shutil
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_shape_surface_occupancy_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.models_lp import KLAutoEncoder
import models_class_cond, models_ae
from models.clip import clip

from engine_class_cond import train_one_epoch, evaluate

from torch.profiler import profile, record_function, ProfilerActivity

def get_args_parser():
    parser = argparse.ArgumentParser('Latent Diffusion', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--replica', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='kl_d512_m512_l8_edm', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument("--config_ae", required=True, help='config file path fo ae')

    parser.add_argument('--ae_pth', required=True, help='Autoencoder checkpoint')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--latent_dir', default=None, help='path where to latent')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=60, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args, config_ae):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if config_ae.dataset.name == "objaverse":
        build_dataset_ldm = misc.load_module(f"datasets.objaverse_ldm", 'build_dataset_ldm')
        dataset_train = build_dataset_ldm('train', cfg=config_ae.dataset, args=args)
        dataset_val = build_dataset_ldm('test', cfg=config_ae.dataset, args=args)
    else:
        build_dataset_ldm = misc.load_module(f"datasets.shapenet_ldm", 'build_dataset_ldm')
        dataset_train = build_dataset_ldm('train', cfg=config_ae.dataset, args=args)
        dataset_val = build_dataset_ldm('test', cfg=config_ae.dataset, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        logger.add(f"{args.log_dir}/log.txt", level="DEBUG")
        git_env, run_command = misc.get_run_env()
        logger.info(git_env)
        logger.info(run_command)
        shutil.copy2(args.config_ae, args.log_dir)
        misc.backup_modified_files(f"{args.log_dir}/code/")
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        # prefetch_factor=2,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=args.batch_size,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    ae = KLAutoEncoder(config_ae)
    ae.eval()
    print("Loading autoencoder %s" % args.ae_pth)
    ae.load_state_dict(torch.load(args.ae_pth, map_location='cpu', weights_only=False)['model'])
    ae.to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    model = models_class_cond.__dict__[args.model]()
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    assert args.lr is not None
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = models_class_cond.__dict__['EDMLoss']()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/ldm'),
    ) as prof:
        for epoch in range(args.start_epoch, args.start_epoch+2):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            with record_function(f"train_epoch_{epoch}"):
                train_stats = train_one_epoch(
                    model, ae, criterion, clip_model, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    args.clip_grad,
                    log_writer=log_writer,
                    args=args,
                )
            if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

            if epoch % 20 == 0 or epoch + 1 == args.epochs:
                with record_function(f"eval_epoch_{epoch}"):
                    test_stats = evaluate(data_loader_val, model, ae, clip_model, criterion, device)
                print(f"loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}")

                if log_writer is not None:
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,
                                'n_parameters': n_parameters}

                # if args.output_dir and misc.is_main_process():
                #     if log_writer is not None:
                #         log_writer.flush()
                #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                #         f.write(json.dumps(log_stats) + "\n")

            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'epoch': epoch,
                                'n_parameters': n_parameters}

            if args.log_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                logger.info(json.dumps(log_stats) + "\n")

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.output_dir = f"{args.log_dir}/ckpt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config_ae = OmegaConf.load(args.config_ae)
    OmegaConf.resolve(config_ae)

    main(args, config_ae)
