#!/bin/bash
set -e

log_dir="output/vae/shapenet/vae_car"
config_path="config/shapenet/train_car_base.yaml"
full_config="config/shapenet/train_car_full.yaml"
dm_log_dir="output/ldm/shapenet/car"
full_log_dir="output/vae/shapenet/vae_car_full"
model_only_path="output/vae/shapenet/vae_car/ckpt/checkpoint-999.pth"
resume_path="${full_log_dir}/ckpt/checkpoint-199_model_only.pth"

ncu -f \
	--nvtx \
	--set full \
	-c 20 \
	--kernel-name regex:"Bid.*" \
	--print-summary per-nvtx \
	--target-processes all \
	-o profiler/bid-ncu \
	python main_ae_profile_nvtx.py \
	--config "${full_config}" \
	--log_dir "${full_log_dir}" \
	--resume "${resume_path}"
