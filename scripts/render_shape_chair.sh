#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python sample_class_cond_cfg.py \
	--config_ae config/shapenet/train_chair_full.yaml \
	--ae_pth ./output/vae/shapenet/vae_chair_full/ckpt/checkpoint-999.pth \
	--dm kl_d512_m512_l16_d24_edm \
	--dm_pth output/ldm/shapenet/chair/ckpt/checkpoint-999.pth \
	--seed 5
