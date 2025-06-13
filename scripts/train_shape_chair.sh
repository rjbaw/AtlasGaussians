#!/bin/bash
set -euo pipefail

format_time() {
	local T=$1
	printf '%02dh:%02dm:%02ds' $((T / 3600)) $(((T % 3600) / 60)) $((T % 60))
}

##########################
# Stage 1: Base VAE Training
##########################

# record start of all VAE work
vae_start_ts=$(date +%s)

log_dir="output/vae/shapenet/vae_chair"
config_path="config/shapenet/train_chair_base.yaml"
full_config="config/shapenet/train_chair_full.yaml"
dm_log_dir="output/ldm/shapenet/chair"
full_log_dir="output/vae/shapenet/vae_chair_full"

echo ">>> Stage 0: generate data splits"
python split.py --config "${config_path}" --ratio 0.8

echo ">>> Stage 1: training with config=${config_path} → log_dir=${log_dir}"
python main_ae.py --config "${config_path}" --log_dir "${log_dir}"

echo ">>> Extract “model-only” from the latest checkpoint"
ckpt_dir="${log_dir}/ckpt"
latest_ckpt=$(
	ls -1v "${ckpt_dir}"/checkpoint-[0-9]*.pth |
		grep -v '_model_only' |
		sort -V |
		tail -n1
)

model_only_filename="$(basename "${latest_ckpt%.*}")_model_only.pth"
model_only_path="${ckpt_dir}/${model_only_filename}"
python - <<EOF
import torch
ckpt = torch.load("${latest_ckpt}", map_location="cpu", weights_only=False)
torch.save({"model": ckpt["model"]}, "${model_only_path}")
EOF

echo ">>> Stage 2: Resume training from model-only"
full_ckpt_dir="${full_log_dir}/ckpt"
mkdir -p "${full_ckpt_dir}"
cp "${model_only_path}" "${full_ckpt_dir}/"
resume_path="${full_ckpt_dir}/${model_only_filename}"
python main_ae.py \
	--distributed \
	--config "${full_config}" \
	--log_dir "${full_log_dir}" \
	--resume "${resume_path}"

echo ">>> Stage 2.5: Precompute latents (inference)"
epoch=1000
python main_ae.py \
	--config "${full_config}" \
	--log_dir "${full_log_dir}" \
	--resume "${full_log_dir}/ckpt/$(basename "${model_only_path}")" \
	--start_epoch ${epoch} \
	--infer

vae_end_ts=$(date +%s)
vae_elapsed=$((vae_end_ts - vae_start_ts))

##########################
# Stage 3: Diffusion (Class‐Conditional) Training
##########################

ldm_start_ts=$(date +%s)

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

ldm_end_ts=$(date +%s)
ldm_elapsed=$((ldm_end_ts - ldm_start_ts))

echo ">>> All done!"
echo
echo "==== Timing Summary ===="
echo "VAE (training + resume + inference): $(format_time ${vae_elapsed})"
echo "LDM (diffusion training):         $(format_time ${ldm_elapsed})"
