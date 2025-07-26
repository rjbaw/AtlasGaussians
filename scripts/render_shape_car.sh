#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python sample_class_cond_cfg.py \
	--config_ae config/shapenet/train_car_full.yaml \
	--ae_pth ./output/vae/shapenet/vae_car_full/ckpt/checkpoint-999.pth \
	--dm kl_d512_m512_l16_d24_edm \
	--dm_pth output/ldm/shapenet/car/ckpt/checkpoint-999.pth \
	--seed 5
