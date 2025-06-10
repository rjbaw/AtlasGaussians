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

torch.set_num_threads(8)

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.models_lp import KLAutoEncoder
from engine_ae import train_one_epoch, evaluate, inference

import torch.cuda.nvtx as nvtx

def collate_fn_filter_none(batch):
    from torch.utils.data._utils.collate import default_collate

    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return {}
    return default_collate(batch)


def get_args_parser():
    parser = argparse.ArgumentParser("Autoencoder", add_help=False)
    parser.add_argument(
        "--config", dest="config_path", required=True, help="config file path"
    )
    # log parameters
    parser.add_argument(
        "--log_dir", default="./output/", help="path where to tensorboard log"
    )

    # optimization parameters
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--infer", action="store_true", help="infer latents only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )

    # distributed training parameters
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--distributed", action="store_true", help="enable DDP")
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def get_optimizer(model, config):
    backbone_params = [
        params
        for name, params in model.named_parameters()
        if "encoder.backbone" in name
    ]
    other_params = [
        params
        for name, params in model.named_parameters()
        if "encoder.backbone" not in name
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.train.lr_backbone},
            {"params": other_params, "lr": config.train.lr},
        ],
        weight_decay=config.train.weight_decay,
    )
    return optimizer


def main(args, config):
    if args.distributed:
        misc.init_distributed_mode(args)
        global_rank = misc.get_rank()
    else:
        args.gpu = 0
        global_rank = 0

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = config.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # cudnn.enabled = False  # NOTE: if multiprocessing error, disable cudnn, but it might slow down the training.

    build_dataset = misc.load_module(f"datasets.{config.dataset.name}", "build_dataset")
    dataset_train = build_dataset("train", cfg=config.dataset)
    dataset_val = build_dataset("test", cfg=config.dataset)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            raise NotImplementedError("reduce of statistics is required.")
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        if (not args.eval) and (not args.infer):
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            args.log_dir = f"{args.log_dir}/inference"
            os.makedirs(args.log_dir, exist_ok=True)
        logger.add(f"{args.log_dir}/log.txt", level="DEBUG")
        git_env, run_command = misc.get_run_env()
        logger.info(git_env)
        logger.info(run_command)
        shutil.copy2(args.config_path, args.log_dir)
        misc.backup_modified_files(f"{args.log_dir}/code/")
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        drop_last=True,
        prefetch_factor=2,
        collate_fn=collate_fn_filter_none,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.test.batch_size,
        # batch_size=1,
        num_workers=config.num_workers,
        # num_workers=1,
        pin_memory=config.pin_mem,
        drop_last=False,
        collate_fn=collate_fn_filter_none,
    )

    model = KLAutoEncoder(config)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = (
        config.train.batch_size * config.train.accum_iter * misc.get_world_size()
    )

    print("actual lr: %.2e" % config.train.lr)

    print("accumulate grad iterations: %d" % config.train.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    optimizer = get_optimizer(model_without_ddp, config)

    loss_scaler = NativeScaler()

    criterion = torch.nn.BCEWithLogitsLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        print(f"evaluating epoch: {args.start_epoch}")
        test_stats, _ = evaluate(
            data_loader_val, model, device, config, args.start_epoch
        )
        exit(0)

    if args.infer:
        nvtx.range_push("inference")
        logger.info(f"infer epoch: {args.start_epoch}")
        for dset in [dataset_train, dataset_val]:
            data_loader_infer = torch.utils.data.DataLoader(
                dset,
                sampler=torch.utils.data.SequentialSampler(dset),
                batch_size=100,
                num_workers=10,
                pin_memory=False,
                drop_last=False,
                collate_fn=collate_fn_filter_none,
            )
            inference(data_loader_infer, model, device, config, args)
        nvtx.range_pop(0)
        exit(0)

    print(f"Start training for {config.train.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.start_epoch+2):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        nvtx.range_push(f"train_epoch_{epoch}")
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            config.train.clip_grad,
            log_writer=log_writer,
            config=config,
        )
        nvtx.range_pop()
        if args.output_dir and (
            epoch % config.train.save_ckpt_interval == 0
            or epoch + 1 == config.train.epochs
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        if epoch % config.train.eval_interval == 0 or epoch + 1 == config.train.epochs:
            nvtx.range_push(f"eval_epoch_{epoch}")
            test_stats, log_img_dict = evaluate(
                data_loader_val, model, device, config, epoch
            )
            nvtx.range_pop()

            if log_writer is not None:
                for k, v in test_stats.items():
                    if ("loss" in k) or ("psnr" in k):
                        log_writer.add_scalar(f"perf/{k}", v, epoch)
                # for k, v in log_img_dict.items():
                #     log_writer.add_images(f"perf/{k}", v, epoch)
                for k, v in log_img_dict.items():
                    if isinstance(v, np.ndarray) and v.ndim == 4 and v.shape[0] > 0:
                        log_writer.add_images(f"perf/{k}", v, epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.log_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            logger.info(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    args.output_dir = f"{args.log_dir}/ckpt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config_path)
    OmegaConf.resolve(config)

    main(args, config)
