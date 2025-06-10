#!/bin/bash

set -euo pipefail

##########################
# Stage 1: Base VAE Training
##########################

log_dir="output/vae/shapenet/vae_chair"
config_path="config/shapenet/train_chair_base.yaml"
full_config="config/shapenet/train_chair_full.yaml"
dm_log_dir="output/ldm/shapenet/chair"
full_log_dir="output/vae/shapenet/vae_chair_full"

echo ">>> Stage 0: generate data splits"
python split.py --config "${config_path}" --ratio 0.8

echo ">>> Stage 1: training with config=${config_path} → log_dir=${log_dir}"
python main_ae.py --config "${config_path}" --log_dir "${log_dir}"


##########################
# Extract “model-only” from the latest checkpoint
##########################

ckpt_dir="${log_dir}/ckpt"
latest_ckpt=$(
  ls -1v "${ckpt_dir}"/checkpoint-[0-9]*.pth | grep -v '_model_only' | sort -V | tail -n1
)
echo ">>> Found latest checkpoint: ${latest_ckpt}"

model_only_filename="$(basename "${latest_ckpt%.*}")_model_only.pth"
model_only_path="${ckpt_dir}/${model_only_filename}"

echo ">>> Will write model-only to: ${model_only_path}"
python - <<EOF
import torch
ckpt = torch.load("${latest_ckpt}", map_location="cpu", weights_only=False)
torch.save({"model": ckpt["model"]}, "${model_only_path}")
EOF

echo ">>> Saved model-only checkpoint."


##########################
# Stage 2: Resume Training from model-only
##########################

full_ckpt_dir="${full_log_dir}/ckpt"

# Ensure the “full” directory exists, then copy model-only in
mkdir -p "${full_ckpt_dir}"
cp "${model_only_path}" "${full_ckpt_dir}/"

resume_path="${full_ckpt_dir}/${model_only_filename}"

echo ">>> Stage 2: training with config=${full_config}, resume=${resume_path} → log_dir=${full_log_dir}"
python main_ae.py \
  --distributed \
  --config "${full_config}" \
  --log_dir "${full_log_dir}" \
  --resume "${resume_path}"

##########################
# Stage 2.5: Precompute latents (inference)
##########################
epoch=1000

echo ">>> Stage 2.5: precomputing latents at epoch ${epoch}"
python main_ae.py \
  --config "${full_config}" \
  --log_dir "${full_log_dir}" \
  --resume "${full_log_dir}/ckpt/$(basename "${model_only_path}")" \
  --start_epoch ${epoch} \
  --infer

##########################
# Stage 3: Diffusion (Class‐Conditional) Training
##########################

mkdir -p "${dm_log_dir}"

echo ">>> Stage 3: training diffusion model → log_dir=${dm_log_dir}"

python main_class_cond.py \
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
  --latent_dir "${full_log_dir}/inference/latents/epoch_${epoch}"

echo ">>> All done!"
