import argparse
import math
import os
import cv2
import json
from tqdm import tqdm

import numpy as np
import imageio.v3 as iio

import torch

import models_class_cond_profile_nvtx as models_class_cond
from models.models_lp import KLAutoEncoder
import util.misc as misc
from models.clip import clip

from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from einops import repeat, rearrange

from util.geom_utils import build_grid2D, get_W2C_uniform, fusion
from gs import GaussianModel, gs_render
from engine_class_cond import scale_factor

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import torch.cuda.nvtx as nvtx


if __name__ == "__main__":

    parser = argparse.ArgumentParser("", add_help=False)
    parser.add_argument("--config_ae", required=True, help="config file path fo ae")
    parser.add_argument("--ae_pth", type=str, required=True)
    parser.add_argument("--dm", type=str, required=True)
    parser.add_argument("--dm_pth", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset",   type=str, default="objaverse",
                        choices=["objaverse", "shapenet"],
                        help="Output directory layout")
    parser.add_argument("--class_id",  type=str, default="02958343",
                        help="ShapeNet class id (used only if --dataset shapenet)")
    parser.add_argument("--shape_id",  type=str, default="auto",
                        help="ShapeNet shape folder. "
                            "`auto` keeps one folder per prompt index; any other "
                            "string puts every view into that one folder.")
    args = parser.parse_args()
    print(args)

    dm_epoch = args.dm_pth.split("/")[-1].split(".pth")[0].split("-")[-1]
    dump_root = os.path.dirname(args.dm_pth).split("ckpt")[0] + f"/results_E{dm_epoch}/"
    print(dump_root)
    if not os.path.exists(dump_root):
        os.makedirs(dump_root)

    logger.add(f"{dump_root}/log_sample.txt", level="DEBUG")
    git_env, run_command = misc.get_run_env()
    logger.info(git_env)
    logger.info(run_command)

    config_ae = OmegaConf.load(args.config_ae)
    OmegaConf.resolve(config_ae)

    device = torch.device("cuda:0")

    ae = KLAutoEncoder(config_ae)
    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth, weights_only=False)["model"])
    ae.to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    model = models_class_cond.__dict__[args.dm]()
    model.eval()

    model.load_state_dict(torch.load(args.dm_pth, weights_only=False)["model"])
    model.to(device)

    N = config_ae.model.num_lp * 4
    num_samples = config_ae.loss.num_samples
    queries_grid = build_grid2D(
        vmin=0.0, vmax=1.0, res=int(np.sqrt(num_samples)), device=device
    ).reshape(-1, 2)
    queries_grid = repeat(queries_grid, "s d -> b n s d", b=1, n=N)  # [1, N, S, 2]

    gaussian_model = GaussianModel(config_ae.model.gs)

    # render settings
    all_eval_pose = torch.load("./objv_eval_pose.pt", weights_only=False)[
        :27
    ]  # [40, 25]
    # intrinsic
    fov_rad = 0.691150367
    # extrinsic
    cam2worlds = all_eval_pose[:, :16].reshape(1, -1, 4, 4).cuda()  # [1, V, 4, 4]
    world2cams = torch.linalg.inv(cam2worlds)

    W2C_uniform = get_W2C_uniform(
        n_views=100, radius=1.75, device=device
    )  # [100, 4, 4]. gobjaverse radius seems in the range [1.5, 2.0]

    # load input text prompts

    # text_fpath = f"{config_ae.dataset.data_root}/gobjaverse_cap3d/text_captions_cap3d.json"
    # split_fpath = f"./datasets/splits/objaverse/eval250.txt"
    # with open(split_fpath, 'r') as f:
    #     id_test = f.read().splitlines()
    # texts = json.load(open(text_fpath, 'r'))
    # texts = [texts[v] for v in id_test]

    with open("./evaluations/baseline_prompts.txt", "r") as f:
        texts = f.read().splitlines()

    total = len(texts)
    iters = 1  # num of texts for each iteration

    with torch.no_grad():

        null_token = clip.tokenize([""] * iters, truncate=True).to(
            device
        )  # [B==iters, 77]
        with torch.no_grad():
            null_features = (
                clip_model.encode_text(null_token).float().unsqueeze(1)
            )  # [B==iters, 1, 512]

        for cfg_scale in [3.5]:
            # dump_dir = f"{dump_root}/cfg_{cfg_scale}"
            base_cfg_dir = f"cfg_{cfg_scale}"
            
            for seed in [args.seed]:

                for i in tqdm(range(3)):
                    nvtx.range_push("tokenize+encode")

                    text_token = clip.tokenize(
                        texts[i * iters : (i + 1) * iters], truncate=True
                    ).to(
                        device
                    )  # [B==iters, 77]
                    with torch.no_grad():
                        text_features = (
                            clip_model.encode_text(text_token).float().unsqueeze(1)
                        )  # [B==iters, 1, 512]

                    cond = torch.cat(
                        [text_features, null_features], dim=0
                    )  # [2B, 1, 512]
                    batch_seeds = (
                        torch.arange(i * iters, (i + 1) * iters).to(device) * 0 + seed
                    )
                    batch_seeds = torch.cat([batch_seeds, batch_seeds], dim=0)  # [2B,]

                    nvtx.range_pop()

                    nvtx.range_push("model.sample")

                    sampled_array = model.sample(
                        cond=cond, batch_seeds=batch_seeds, cfg_scale=cfg_scale
                    ).float()  # [2B=2*iters, num_lp, latent_dim]
                    sampled_array, _ = sampled_array.chunk(2, dim=0)
                    sampled_array = sampled_array / scale_factor

                    nvtx.range_pop()

                    # print("Iteration: ", i)
                    # print("    Shape: ", sampled_array.shape)
                    # print("    Max: ", sampled_array.max())
                    # print("    Min: ", sampled_array.min())
                    # print("    Mean: ", sampled_array.mean())
                    # print("    Std: ", sampled_array.std())

                    for j in range(sampled_array.shape[0]):

                        if args.dataset == "shapenet":
                            shape_folder = (
                                args.shape_id
                                if args.shape_id != "auto"
                                else f"{i:05d}"
                            )
                            dump_dir = os.path.join(
                                dump_root,
                                args.class_id,
                                shape_folder,
                                "rgb",
                                base_cfg_dir,
                            )
                        else:
                            dump_dir = os.path.join(dump_root, base_cfg_dir, "images")

                        nvtx.range_push("ae decode")
                        outputs_dict = ae.decode(sampled_array[j : j + 1], queries_grid)
                        nvtx.range_pop()

                        nvtx.range_push("gs model")
                        gaussians_render = gaussian_model(
                            outputs_dict["gs"]
                        )  # [B, N, S, 14], the activated values
                        nvtx.range_pop()

                        # render
                        bg_color = torch.ones(3, dtype=torch.float32, device=device)

                        nvtx.range_push("gs render")
                        render_dict = gs_render(
                            gaussians=rearrange(
                                gaussians_render, "b n s d -> b (n s) d"
                            ),
                            R=world2cams[:, :, :3, :3],
                            T=world2cams[:, :, :3, 3],
                            fov_rad=fov_rad,
                            output_size=512,
                            bg_color=bg_color,
                        )
                        nvtx.range_pop()

                        text_prefix = "_".join(
                            texts[i * iters : (i + 1) * iters][j].split()
                        ).replace("/", "=")

                        # if not os.path.exists(f"{dump_dir}/images/"):
                        #     os.makedirs(f"{dump_dir}/images/")
                        os.makedirs(dump_dir, exist_ok=True)
                        assert render_dict["images"].shape[0] == 1

                        nvtx.range_push("IO")
                        for v, img in enumerate(render_dict["images"][0]):
                            img = (
                                img.permute(1, 2, 0).detach().cpu().numpy()
                            )  # [H, W, 3]
                            img = np.clip(img, 0, 1)
                            img_uint8 = (img * 255).astype(np.uint8)
                            # iio.imwrite(
                            #     f"{dump_dir}/images/seed{seed}_{i*iters+j:05d}-{text_prefix}-{v}.png",
                            #     img_uint8,
                            # )
                            if args.dataset == "shapenet":
                                out_fname = f"{v:05d}.png"
                            else:
                                out_fname = f"seed{seed}_{i*iters+j:05d}-{text_prefix}-{v}.png"
                            iio.imwrite(os.path.join(dump_dir, out_fname), img_uint8)

                        nvtx.range_pop()
