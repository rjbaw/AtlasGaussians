# AtlasGaussians

Code for ICLR 2025 paper: [Atlas Gaussians Diffusion for 3D Generation](https://yanghtr.github.io/projects/atlas_gaussians/) 


## Environment

```
bash scripts/env_setup.sh

```

## Reproducing Results

Download the checkpoint from [this link](https://huggingface.co/yanghtr/AtlasGaussians/tree/main/output). 
Place them according to original path.
To reproduce the results of Figure 6 in the paper:

```
bash scripts/render_paper.sh
```


## Training

### Dataset

#### Objaverse
Download the [G-buffer Objaverse dataset](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse).

```
bash util/download_objaverse.sh
```

We only use part of the dataset and the data split is in [this file](./datasets/splits/objaverse/paper_train.txt).
The model training requires point clouds as input, the point clouds are in [this link](https://huggingface.co/yanghtr/AtlasGaussians/tree/main/Dataset/Objaverse/gobjaverse_pc). 

Unzip the file and put it under the `data_root` (default at `data/objaverse/`).

For more details of the dataset, please refer to Section A.1 in the appendix of the paper. Please also cite [G-buffer Objaverse dataset](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) if you use the point clouds.


#### ShapeNetCorev2
Download [ShapenetCorev2](https://shapenet.org). Place them under `util/shapenet_renderer/data/` or any other path.

```
cd util/shapenet_renderer/
bash render_car.sh
bash render_plane.sh
bash render_chair.sh
```
Move the renders accordingly to paths specified inside the configuration files inside the `config/` directory.


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


- If you encounter errors related to CUDA, consider setting `cudnn.enabled = False` (line 80 in `main_ae.py`).

- We find that cleaning the dataset is quite important for training. For more details, see Section A.1 in the appendix of the paper.

- Our models are trained on three general c
