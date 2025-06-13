#!/bin/sh

log_dir="output/vae/shapenet/vae_car"
config_path="config/shapenet/train_car_base.yaml"
full_config="config/shapenet/train_car_full.yaml"
dm_log_dir="output/ldm/shapenet/car"
full_log_dir="output/vae/shapenet/vae_car_full"
model_only_path="output/vae/shapenet/vae_car/ckpt/checkpoint-999.pth"
resume_path="${full_log_dir}/ckpt/checkpoint-199_model_only.pth"

nsys profile -f true -o profiler/vae --trace=cuda,nvtx,cublas,cusparse \
    python main_ae_profile_nvtx.py \
  --distributed \
  --config "${full_config}" \
  --log_dir "${full_log_dir}" \
  --resume "${resume_path}" > profiler/profile_nvtx_vae_log.txt

nsys profile -f true -o profiler/ldm --trace=cuda,nvtx,cublas,cusparse \
    python main_class_cond_profile_nvtx.py \
  --accum_iter 1 \
  --config_ae "${full_config}" \
  --model kl_d512_m512_l16_d24_edm \
  --ae_pth "${full_log_dir}/ckpt/$(basename "${model_only_path}")" \
  --log_dir "${dm_log_dir}" \
  --num_workers 10 \
  --batch_size 27 \
  --epochs 1000 \
  --replica 8 \
  --warmup_epochs 50 \
  --lr 1e-4 \
  --latent_dir "${full_log_dir}/inference/latents/epoch_1000" > profiler/profile_nvtx_ldm_log.txt

CUDA_VISIBLE_DEVICES=0 \
        nsys profile -f true -o profiler/inference --trace=cuda,nvtx,cublas,cusparse \
        python sample_class_cond_cfg_profile_nvtx.py \
        --config_ae config/shapenet/train_car_full.yaml \
        --ae_pth ./output/vae/shapenet/vae_car_full/ckpt/checkpoint-999.pth \
        --dm kl_d512_m512_l16_d24_edm \
        --dm_pth output/ldm/shapenet/car/ckpt/checkpoint-999.pth \
        --seed 5 > profiler/profile_nvtx_infer_log.txt
