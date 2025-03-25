import os
import glob
import h5py
import random
import yaml 
import numpy as np
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
import imageio.v3 as iio

import torch
from torch.utils import data
import torchvision.transforms.functional as TF

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

category_ids = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9, 
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54,
}

class ShapeNet(data.Dataset):
    def __init__(self, split, data_root, categories=['03001627'], num_views=None, img_size_input=224, img_size_render=512, replica=16):
        
        self.split = split
        self.data_root = data_root
        self.num_views = num_views
        self.img_size_input = img_size_input
        self.img_size_render = img_size_render
        self.replica = replica

        self.img_size_raw = None  # the raw img size from the dataset
        self.fov_rad = None # NOTE: we assume all data share the same intrinsics

        if categories is None:
            categories = os.listdir(self.data_root)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()

        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            cur_dir = os.path.dirname(__file__)
            split_fpath = f"{cur_dir}/splits/shapenet/{c}_{split}.txt"
            assert os.path.exists(split_fpath)

            with open(split_fpath, 'r') as f:
                models_c = f.read().splitlines()
            models_c.sort()
            
            self.models += [ {'category': c, 'model': m} for m in models_c ]


    def __getitem__(self, idx):
        idx = idx % len(self.models)

        category = self.models[idx]['category']
        model = self.models[idx]['model']
        
        # Load intrinsic. All intrinsics are the same.
        if (self.img_size_raw is None) or (self.fov_rad is None):
            with open(f"{self.data_root}/{category}/{model}/intrinsics.txt", 'r') as f:
                lines = f.readlines()
            line0 = [float(v) for v in lines[0].strip().split()]
            line3 = [int(v) for v in lines[3].strip().split()]

            width=line3[0]
            height=line3[1]
            assert (width == height), "Ensure fov_x == fov_y" 
            self.img_size_raw = width
            focal_length = line0[0]
            self.fov_rad = 2 * np.arctan(width / (2 * focal_length))

        # Load rendering.
        # NOTE: in the 76 view dataset, use fixed view 1, 5, 9, 13 as input.
        idx_dataset = np.array(list(range(26)) + list(range(100, 150)))
        idx_input = np.array([1, 5, 9, 13])  # always input 4 views
        mask_idx = np.isin(idx_dataset, idx_input)
        idx_rest = idx_dataset[~mask_idx]
        assert (idx_input.shape[0] < self.num_views)
        ids = np.random.default_rng().choice(idx_rest, self.num_views - idx_input.shape[0], replace=False)
        ids = np.append(idx_input, ids)

        rgb_gt = []
        depth_gt = []
        mask_gt = []
        rgb_inputs = []
        world2cams = []
        for ind in ids:
            ##### Read depth
            depth = iio.imread(f"{self.data_root}/{category}/{model}/images/{ind:06d}_depth0001.exr")
            depth = np.nan_to_num(depth, posinf=0, neginf=0)
            depth = depth[:, :, 0:1]  # [H, W, 1]

            #### Read raw RGB
            rgba = iio.imread(f"{self.data_root}/{category}/{model}/images/{ind:06d}.png") / 255.0  # [H, W, 4]
            rgb = rgba[:, :, :3]  # [H, W, 3]

            #### Read mask. Use depth in shapenet since no alpha available
            mask = (depth > 0).astype(np.float32)  # [H, W, 1]

            ##### Prepare render images for GT
            if self.img_size_render != self.img_size_raw:
                raise NotImplementedError()
            rgb_gt.append(rgb)
            depth_gt.append(depth)
            mask_gt.append(mask)

            ##### Prepare input images
            if self.img_size_input != self.img_size_raw:
                rgb_input  = cv2.resize(rgb,  (self.img_size_input, self.img_size_input), interpolation=cv2.INTER_LANCZOS4)
                mask_input = cv2.resize(mask, (self.img_size_input, self.img_size_input), interpolation=cv2.INTER_LANCZOS4)
                mask_input = mask_input[:, :, None]  # cv2.resize remove the channel when channel dim is 1
            else:
                rgb_input  = rgb
                mask_input = mask

            rgb_input  = torch.from_numpy(rgb_input).float()   # [H, W, 3]
            mask_input = torch.from_numpy(mask_input).float()  # [H, W, 1]
            # NOTE: use IMAGENET_DEFAULT_MEAN as background so that the normalized inputs have background 0.
            bg_color = torch.FloatTensor(IMAGENET_DEFAULT_MEAN).reshape(1, 1, 3)
            rgb_input = rgb_input * mask_input + bg_color * (1 - mask_input)  # [H, W, 3]

            rgb_input = TF.normalize(rgb_input.permute(2, 0, 1), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # [3, H, W]
            rgb_inputs.append(rgb_input)

            ##### Prepare camera poses
            with open(f"{self.data_root}/{category}/{model}/pose/{ind:06d}.txt") as f:
                lines = f.readlines()
            cam2world = np.array([float(v) for v in lines[0].strip().split()]).reshape(4, 4)
            world2cam = np.linalg.inv(cam2world)
            world2cams.append(world2cam)

        rgb_inputs = torch.stack(rgb_inputs).float()  # [V, 3, img_size_input, img_size_input]
        rgb_gt = torch.from_numpy(np.stack(rgb_gt)).float()  # [V, img_size_render, img_size_render, 3]
        depth_gt = torch.from_numpy(np.stack(depth_gt)).float()  # [V, img_size_render, img_size_render, 1]
        mask_gt = torch.from_numpy(np.stack(mask_gt)).float()  # [V, img_size_render, img_size_render, 1]
        world2cams = torch.from_numpy(np.stack(world2cams)).float()  # [V, 4, 4]

        # TODO: only preprocess necessary data for better efficiency
        # use the fixed views as the input views
        rgb_inputs = rgb_inputs[:idx_input.shape[0]]  # [4, 3, img_size_input, img_size_input]
        # For training efficiency, randomly select 2 input views and all the rest random views as GT views
        assert (idx_input.shape[0] >= 2)
        select_i = np.random.default_rng().choice(np.arange(idx_input.shape[0]), 2, replace=False)
        select_i = select_i.tolist() + list(range(idx_input.shape[0], self.num_views))
        rgb_gt = rgb_gt[select_i]
        depth_gt = depth_gt[select_i]
        mask_gt = mask_gt[select_i]
        world2cams = world2cams[select_i]

        # Load surface points
        pc_path = f"{self.data_root}/{category}/{model}/surface_sample.npz"
        with np.load(pc_path) as data:
            points_all = data['points_sample'].astype(np.float32)
        rand_idx = np.random.default_rng().choice(points_all.shape[0], 2048+8192*5, replace=False)
        points = points_all[rand_idx]  # for GT
        rand_idx = np.random.default_rng().choice(points_all.shape[0], 2048, replace=False)
        points_input = points_all[rand_idx]  # for input

        data_dict = {'category_idx': category_ids[category], 'idx': idx, 'rgb_inputs': rgb_inputs,
                     'rgb_gt': rgb_gt, 'depth_gt': depth_gt, 'mask_gt': mask_gt,
                     'world2cams': world2cams, 'fov_rad': self.fov_rad, 'img_size_render': self.img_size_render, 
                     'points': points, 'points_input': points_input}

        return data_dict

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica



def build_dataset(split, cfg):
    dataset = ShapeNet(split, cfg.data_root, categories=cfg.categories, num_views=cfg.num_views,
                       img_size_input=cfg.img_size_input, img_size_render=cfg.img_size_render, replica=cfg.replica)
    return dataset


