import os
import json
import random
import numpy as np
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
import imageio.v3 as iio

import torch
from torch.utils import data
import torchvision.transforms.functional as TF


class ShapenetLDM(data.Dataset):
    def __init__(self, split, data_root, category, latent_dir, replica=16):
        
        self.split = split
        self.data_root = data_root
        self.latent_dir = latent_dir
        self.replica = replica

        cur_dir = os.path.dirname(__file__)
        split_fpath = f"{cur_dir}/splits/shapenet/{category}_{split}.txt"
        assert os.path.exists(split_fpath)

        with open(split_fpath, 'r') as f:
            self.models = f.read().splitlines()
        self.models.sort()
        
    def __getitem__(self, idx):
        idx = idx % len(self.models)

        model = self.models[idx]

        latent_dict = torch.load(f"{self.latent_dir}/{model}/latent.pt")

        data_dict = {'idx': idx, 'mean': latent_dict['mean'], 'logvar': latent_dict['logvar']}

        return data_dict

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica



def build_dataset_ldm(split, cfg, args):
    dataset = ShapenetLDM(split, cfg.data_root, cfg.categories[0], args.latent_dir, replica=args.replica)
    return dataset


