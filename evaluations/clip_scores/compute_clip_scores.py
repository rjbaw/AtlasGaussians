#!/usr/bin/env python
# coding=utf-8
import os
import json
import argparse
import torch
from PIL import Image
import numpy as np
import sys
sys.path.append('../../models/')
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) # and L/14


def compute_clip_score_per_method(results_dir, seed, texts):
    clip_score_dict = {}
    for v in range(24):
        for idx, raw_text in enumerate(texts):
            text_prefix = '_'.join(raw_text.split()).replace('/', '=')
            fpath = f"{results_dir}/seed{seed}_{idx:05d}-{text_prefix}-{v}.png"
            image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)
            text = clip.tokenize([raw_text]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                logits_per_image, logits_per_text = model(image, text)
                print(raw_text, logits_per_image)
                clip_score_dict[f"{text_prefix}-{v}"] = logits_per_image.detach().cpu().numpy()
    print('mean clip score = ', np.mean([v for k, v in clip_score_dict.items()]))
    print('-'*30)
    return clip_score_dict


text_fpath = f"/scratch/cluster/yanght/Projects/AliGeoReg/Dataset/Objaverse/gobjaverse_cap3d/text_captions_cap3d.json"
split_fpath = f"../../datasets/splits/objaverse/eval250.txt"
with open(split_fpath, 'r') as f:
    id_test = f.read().splitlines()
texts = json.load(open(text_fpath, 'r'))
texts = [texts[v] for v in id_test]


results_ours_dir = '../../output/ldm/objaverse/ns49_AE800_kl1e-4_d512_m512_l16_d24_edm_cfg_sf133_8B27/results_E850/cfg_3.5/images/'  # 5


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='method')
    args = parser.parse_args()
    
    if args.method == 'ours':
        clip_score_dict = compute_clip_score_per_method(results_ours_dir, 5, texts)

