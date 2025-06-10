import os
import json
import random
import pickle
import numpy as np
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import imageio.v3 as iio

import torch
from torch.utils import data
import torchvision.transforms.functional as TF

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# along z axis counter-clockwise
rotation_matrices_z_ccw = {
    0: np.eye(3),
    1: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    2: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    3: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
}


def rotate_extrinsic_matrix(extrinsic_matrix, view_index):
    assert view_index in {0, 1, 2, 3}, "View index must be in {0, 1, 2, 3}"

    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]

    R_rotated = rotation_matrices_z_ccw[view_index] @ R

    extrinsic_matrix_rotated = np.eye(4)
    extrinsic_matrix_rotated[:3, :3] = R_rotated
    extrinsic_matrix_rotated[:3, 3] = t

    return extrinsic_matrix_rotated.astype(np.float32)


def read_gobjaverse_depth(normald_path, cond_pos):
    # Reference: https://github.com/modelscope/richdreamer/blob/aa5a0266380f3e5c64bc65ff00491b639169ac7b/dataset/gobjaverse/depth_warp_example.py#L52
    # cond_pos is the camera position
    cond_cam_dis = np.linalg.norm(cond_pos, 2)

    near = 0.867  # slightly larger than sqrt(3) * 0.5, the object is scaled inside [-0.5, 0.5], so the max length from the origin is sqrt(3) * 0.5
    near_distance = cond_cam_dis - near

    normald = cv2.imread(normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = normald[..., 3:]

    # IMPORTANT NOTE: we allos contour smoothing for consistency since gs renderer also smooth contour
    # i.e. np.all((mask>0) == (depth>0)) == True
    # depth[depth<near_distance] = 0  # deal with the contour smoothing

    return depth


class Objaverse(data.Dataset):
    def __init__(
        self,
        split,
        data_root,
        num_views=None,
        img_size_input=224,
        img_size_render=512,
        replica=16,
    ):

        self.split = split
        self.data_root = data_root
        self.img_root = f"{data_root}/gobjaverse"
        self.pc_root = f"{data_root}/gobjaverse_pc/gobjaverse_280k_pc_subset2_pt"
        self.text_fpath = f"{data_root}/gobjaverse_cap3d/text_captions_cap3d.json"
        self.num_views = num_views
        self.img_size_input = img_size_input
        self.img_size_render = img_size_render
        self.replica = replica

        cur_dir = os.path.dirname(__file__)
        split_fpath = f"{cur_dir}/splits/objaverse/{split}.txt"
        assert os.path.exists(split_fpath)

        with open(split_fpath, "r") as f:
            self.models = f.read().splitlines()
        self.models.sort()

        label_view_fpath = f"{cur_dir}/splits/objaverse/label_view.pkl"
        self.label_view = pickle.load(open(label_view_fpath, "rb"))

        self.texts = json.load(open(self.text_fpath, "r"))

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        model = self.models[idx]

        # Load rendering.
        # NOTE: in the 76 view dataset, use fixed view 1, 5, 9, 13 as input.
        idx_dataset = np.array(list(range(24)) + [25, 26])
        idx_input = (np.array([0, 8, 16]) - 6 * self.label_view[model]) % 24
        idx_input = np.append(idx_input, [26])  # always input 4 views
        mask_idx = np.isin(idx_dataset, idx_input)
        idx_rest = idx_dataset[~mask_idx]
        assert idx_input.shape[0] < self.num_views
        ids = np.random.default_rng().choice(
            idx_rest, self.num_views - idx_input.shape[0], replace=False
        )
        ids = np.append(idx_input, ids)

        rgb_gt = []
        depth_gt = []
        mask_gt = []
        rgb_inputs = []
        world2cams = []
        for ind in ids:
            img_path = f"{self.img_root}/{model}/{ind:05d}/{ind:05d}.png"
            json_path = f"{self.img_root}/{model}/{ind:05d}/{ind:05d}.json"
            exr_path = f"{self.img_root}/{model}/{ind:05d}/{ind:05d}_nd.exr"

            try:
                rgba = iio.imread(img_path) / 255.0  # [H, W, 4]
            except Exception as e:
                print(f"{img_path} is corrupted")
                # exit(1)
                return None

            #### Read raw RGB
            rgb = rgba[:, :, :3]  # [H, W, 3]

            #### Read mask
            mask = rgba[:, :, 3:4]  # [H, W, 1]

            ##### Read intrinsic and extrinsic
            width, height = rgb.shape[:2]
            assert width == height, "Ensure fov_x == fov_y"

            try:
                jdata = json.load(open(json_path))
            except Exception as e:
                print(f"{json_path} is corrupted")
                # exit(1)
                return None

            assert jdata["x_fov"] == jdata["y_fov"]
            fov_rad = jdata["x_fov"]

            cam2world = np.eye(4)
            cam2world[:3, 0] = np.array(jdata["x"])
            cam2world[:3, 1] = np.array(jdata["y"])
            cam2world[:3, 2] = np.array(jdata["z"])
            cam2world[:3, 3] = np.array(jdata["origin"])
            world2cam = np.linalg.inv(cam2world)
            # NOTE: extrinsic is also rotated to align with the rotated point clouds.
            world2cam[:3, :3] = (
                world2cam[:3, :3] @ rotation_matrices_z_ccw[self.label_view[model]].T
            )

            ##### Read depth

            try:
                depth = read_gobjaverse_depth(exr_path, cam2world[:3, 3:])  # [H, W, 1]
            except Exception as e:
                print(f"{exr_path} is corrupted")
                # exit(1)
                return None

            depth = np.nan_to_num(depth, posinf=0, neginf=0)

            if ind == 26:
                rgb = np.rot90(rgb, -self.label_view[model])  # clockwise
                mask = np.rot90(mask, -self.label_view[model])
                depth = np.rot90(depth, -self.label_view[model])
                world2cam = rotate_extrinsic_matrix(world2cam, self.label_view[model])

            world2cams.append(world2cam)

            ##### Prepare render images for GT
            if self.img_size_render != width:
                raise NotImplementedError()
            rgb_gt.append(rgb)
            depth_gt.append(depth)
            mask_gt.append(mask)

            ##### Prepare input images
            # NOTE: use IMAGENET_DEFAULT_MEAN as background so that the normalized inputs have background 0.
            bg_color = np.array(IMAGENET_DEFAULT_MEAN).reshape(1, 1, 3)
            rgb_input = rgb * mask + bg_color * (1 - mask)  # [H, W, 3]
            if self.img_size_input != width:
                rgb_input = cv2.resize(
                    rgb_input,
                    (self.img_size_input, self.img_size_input),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            rgb_input = torch.from_numpy(rgb_input).float()  # [H, W, 3]

            rgb_input = TF.normalize(
                rgb_input.permute(2, 0, 1), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            )  # [3, H, W]
            rgb_inputs.append(rgb_input)

        if len(rgb_inputs) == 0:
            return None

        rgb_inputs = torch.stack(
            rgb_inputs
        ).float()  # [V, 3, img_size_input, img_size_input]
        rgb_gt = torch.from_numpy(
            np.stack(rgb_gt)
        ).float()  # [V, img_size_render, img_size_render, 3]
        depth_gt = torch.from_numpy(
            np.stack(depth_gt)
        ).float()  # [V, img_size_render, img_size_render, 1]
        mask_gt = torch.from_numpy(
            np.stack(mask_gt)
        ).float()  # [V, img_size_render, img_size_render, 1]
        world2cams = torch.from_numpy(np.stack(world2cams)).float()  # [V, 4, 4]

        # TODO: only preprocess necessary data for better efficiency
        # use the fixed views as the input views
        rgb_inputs = rgb_inputs[
            : idx_input.shape[0]
        ]  # [4, 3, img_size_input, img_size_input]
        # For training efficiency, randomly select 2 input views and all the rest random views as GT views
        assert idx_input.shape[0] >= 2
        select_i = np.random.default_rng().choice(
            np.arange(idx_input.shape[0]), 2, replace=False
        )
        select_i = select_i.tolist() + list(range(idx_input.shape[0], self.num_views))
        rgb_gt = rgb_gt[select_i]
        depth_gt = depth_gt[select_i]
        mask_gt = mask_gt[select_i]
        world2cams = world2cams[select_i]

        # Load surface points
        pc_path = f"{self.pc_root}/{model}/points.pt"
        points_all = torch.load(pc_path)["points"]
        # The same as rotating point clouds using Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        points_all = torch.stack(
            (points_all[:, 0], -points_all[:, 2], points_all[:, 1]), dim=-1
        )
        # Rotate point clouds
        points_all = points_all @ rotation_matrices_z_ccw[self.label_view[model]].T
        points_all = points_all.float()
        rand_idx = np.random.default_rng().choice(
            points_all.shape[0], 2048 + 8192 * 5, replace=False
        )
        points = points_all[rand_idx]  # for GT
        rand_idx = np.random.default_rng().choice(
            points_all.shape[0], 2048, replace=False
        )
        points_input = points_all[rand_idx]  # for input

        data_dict = {
            "idx": idx,
            "rgb_inputs": rgb_inputs,
            "rgb_gt": rgb_gt,
            "depth_gt": depth_gt,
            "mask_gt": mask_gt,
            "world2cams": world2cams,
            "fov_rad": fov_rad,
            "img_size_render": self.img_size_render,
            "points": points,
            "points_input": points_input,
        }

        return data_dict

    def __len__(self):
        if self.split != "train":
            return len(self.models)
        else:
            return len(self.models) * self.replica


def build_dataset(split, cfg):
    dataset = Objaverse(
        split,
        cfg.data_root,
        num_views=cfg.num_views,
        img_size_input=cfg.img_size_input,
        img_size_render=cfg.img_size_render,
        replica=cfg.replica,
    )
    return dataset
