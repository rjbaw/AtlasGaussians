# AtlasGaussians

Code for ICLR 2025 paper: [Atlas Gaussians Diffusion for 3D Generation](https://openreview.net/pdf?id=H2Gxil855b) 


## Environment

```
conda create -n atlas python=3.11.7
pip install numpy==1.26.3
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0%2Bcu118.html
```

### Install diff-gaussian-rasterization
```
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
mv diff-gaussian-rasterization diff-gaussian-rasterization_extended  # to avoid conflicts between original gaussian-rasterization
cd diff-gaussian-rasterization_extended
mv diff_gaussian_rasterization diff_gaussian_rasterization_extended
```

update `setup.py` with the following changes:
```
name="diff_gaussian_rasterization_extended",
packages=['diff_gaussian_rasterization_extended'],
    ...
    name="diff_gaussian_rasterization_extended._C",
```

Then:
```
cd ..
pip install ./diff-gaussian-rasterization_extended
```

### Install EMD
The EMD module is from [https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd)

```
cd util/emd
python setup.py install
```

### Install pytorch3d
```
pip install ninja
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
```

### Install other dependencel
```
pip install -r requirements.txt
```


## Generate samples (eval)

To reproduce the results of Figure 6 in the paper:

```
CUDA_VISIBLE_DEVICES=0 python sample_class_cond_cfg.py --config_ae config/objaverse/train_18k_full.yaml --ae_pth ./output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4_full_7B7/ckpt/checkpoint-799.pth --dm kl_d512_m512_l16_d24_edm --dm_pth output/ldm/objaverse/ns49_AE800_kl1e-4_d512_m512_l16_d24_edm_cfg_sf133_8B27/ckpt/checkpoint-850.pth --seed 5
```





