#!/bin/bash

########################## VAE training Stage 1 ##########################
log_dir=output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4-test
config_path=config/objaverse/train_18k_base.yaml
#torchrun --nproc_per_node=7 main_ae.py --distributed --config ${config_path} --log_dir ${log_dir}
python main_ae.py --config ${config_path} --log_dir ${log_dir}


########################## VAE training preparation for Stage 2 ##########################
# Step 1:
# In the checkpoint folder `output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4/ckpt`, run the following code in python to obtain the model-only checkpoint:
# ```
# import torch
# ckpt = torch.load('./checkpoint-199.pth', map_location='cpu')
# ckpt_new = {}
# ckpt_new['model'] = ckpt['model']
# torch.save(ckpt_new, './checkpoint-199_model_only.pth')
# ```
#
# Step 2:
# ```
# mkdir -p output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4_full_7B7/
# cp output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4/ckpt/checkpoint-199_model_only.pth vae_L16_ep001_7B7_kl1e-4_full_7B7/
# ```


########################## VAE training Stage 2 ##########################
# log_dir=output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4_full_7B7  # base on vae_L16_ep001_7B7_kl1e-4
# config_path=config/objaverse/train_18k_full.yaml
# torchrun --rdzv-endpoint=localhost:27001 --nproc_per_node=7 main_ae.py --distributed --config ${config_path} --log_dir ${log_dir} \
#     --resume ${log_dir}/ckpt/checkpoint-199_model_only.pth


########################## VAE inference for LDM training ##########################
# log_dir=output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4_full_7B7  # base on vae_L16_ep001_7B7_kl1e-4
# config_path=config/objaverse/train_18k_full.yaml
# torchrun --rdzv-endpoint=localhost:27002 --nproc_per_node=1 main_ae.py --distributed --config ${config_path} --log_dir ${log_dir} --infer \
#     --resume ${log_dir}/ckpt/checkpoint-799.pth

