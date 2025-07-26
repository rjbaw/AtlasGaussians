#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python sample_class_cond_cfg.py --config_ae config/objaverse/train_18k_full.yaml --ae_pth ./output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4_full_7B7/ckpt/checkpoint-799.pth --dm kl_d512_m512_l16_d24_edm --dm_pth output/ldm/objaverse/ns49_AE800_kl1e-4_d512_m512_l16_d24_edm_cfg_sf133_8B27/ckpt/checkpoint-850.pth --seed 5
