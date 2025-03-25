# AtlasGaussians

Code for ICLR 2025 paper: [Atlas Gaussians Diffusion for 3D Generation](https://yanghtr.github.io/projects/atlas_gaussians/) 


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

Download the checkpoint from [this link](https://huggingface.co/yanghtr/AtlasGaussians/tree/main/output).

To reproduce the results of Figure 6 in the paper:

```
CUDA_VISIBLE_DEVICES=0 python sample_class_cond_cfg.py --config_ae config/objaverse/train_18k_full.yaml --ae_pth ./output/vae/objaverse/vae_L16_ep001_7B7_kl1e-4_full_7B7/ckpt/checkpoint-799.pth --dm kl_d512_m512_l16_d24_edm --dm_pth output/ldm/objaverse/ns49_AE800_kl1e-4_d512_m512_l16_d24_edm_cfg_sf133_8B27/ckpt/checkpoint-850.pth --seed 5
```


## Training

### Dataset

Download the [G-buffer Objaverse dataset](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse). Change the `data_root` specified in the [config](./config/objaverse/train_18k_base.yaml).
We only use part of the dataset and the data split is in [this file](./datasets/splits/objaverse/train.txt).
The model training requires point clouds as input, the point clouds are in [this link](https://huggingface.co/yanghtr/AtlasGaussians/tree/main/Dataset/Objaverse/gobjaverse_pc). Unzip the file and put it under the `data_root`.
For more details of the dataset, please refer to Section A.1 in the appendix of the paper. Please also cite [G-buffer Objaverse dataset](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) if you use the point clouds.

### VAE training
The VAE training consists of two stages. For the detailed commands, see [VAE Training Script](./scripts/vae/train_dgx.sh). Note that after training, we pre-compute the latents for latent diffusion.

```
bash scripts/vae/train_dgx.sh
```

### LDM training

```
bash scripts/ldm/train_dgx.sh
```

## Acknowledgements

The repo is built based on:

- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) (for codebase)
- [LGM](https://github.com/3DTopia/LGM) (for 3DGS)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (for 3DGS)
- [G-buffer Objaverse](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) (for Objaverse dataset)
- [DiffTF](https://github.com/ziangcao0312/DiffTF/) (for ShapeNet dataset)


## Others

- We also provide `evaluations/readme.md` for evaluation. The evaluation scripts are built from [LN3Diff](https://github.com/NIRVANALAN/LN3Diff).

- If you encounter errors related to CUDA, consider setting `cudnn.enabled = False` (line 80 in `main_ae.py`).

- We find that cleaning the dataset is quite important for training. For more details, see Section A.1 in the appendix of the paper.

- Our models are trained on three general categories only (Transportation, Furniture, and Animals) and can not generalize to other general categories.


## Citation

```bibtex
@inproceedings{
      yang2025atlas,
      title={Atlas Gaussians Diffusion for 3D Generation},
      author={Haitao Yang and Yuan Dong and Hanwen Jiang and Dejia Xu and Georgios Pavlakos and Qixing Huang},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=H2Gxil855b}
}
```

## Contact

If you have any questions, you can contact Haitao Yang (yanghtr [AT] outlook [DOT] com).



