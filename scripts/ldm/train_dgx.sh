#!/bin/bash
#log_dir=output/ldm/objaverse/ldm176h/ns49_AE800_kl1e-4_d512_m512_l16_d24_edm_cfg_sf133_8B27 # scale_factor = 1.0 / 1.33, full176h/vae_L16_ep001_7B7_kl1e-4_full_7B7, replica=8
log_dir=output/ldm/vae/shapenet/vae_car_full/

# torchrun \
#     --nproc_per_node=8 main_class_cond.py \
#     --accum_iter 1 \
#     --config_ae ./config/objaverse/train_18k_full.yaml \
#     --model kl_d512_m512_l16_d24_edm \
#     --ae_pth ./output/vae/objaverse/full176h/vae_L16_ep001_7B7_kl1e-4_full_7B7/ckpt/checkpoint-799.pth \
#     --log_dir ${log_dir} \
#     --num_workers 10 \
#     --batch_size 27 \
#     --epochs 1000 \
#     --replica 8 \
#     --warmup_epochs 50 \
#     --lr 1e-4 \
#     --latent_dir ./output/vae/objaverse/full176h/vae_L16_ep001_7B7_kl1e-4_full_7B7/inference/latents/epoch_800 \

python3 \
	main_class_cond.py \
	--accum_iter 1 \
	--config_ae ./config/shapenet/train_car_full.yaml \
	--model kl_d512_m512_l16_d24_edm \
	--ae_pth ./output/vae/shapenet/vae_car_full/ckpt/checkpoint-900.pth \
	--log_dir ${log_dir} \
	--num_workers 10 \
	--batch_size 27 \
	--epochs 1000 \
	--replica 8 \
	--warmup_epochs 50 \
	--lr 1e-4 \
	--latent_dir ./output/vae/shapenet/vae_car_full/inference/latents/epoch_800
