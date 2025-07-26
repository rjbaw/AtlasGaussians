# AtlasGaussians

I have tried reproducing the results on ShapeNetCorev2. I have made improvements to the code and did some GPU profiling.    

Code for ICLR 2025 paper: [Atlas Gaussians Diffusion for 3D Generation](https://yanghtr.github.io/projects/atlas_gaussians/).

## Profiling Results

https://www.dropbox.com/scl/fo/tkknepp6bv852ginismki/AMQN1_VBzDt4t6SAfsEAlt4?rlkey=m0w4ycteypv28mwna3bsnk5ay&st=yefhx7b8&dl=0


## Environment

```
conda env create -f env.yml
conda activate atlas
```
## Dataset

### ShapeNetCorev2

#### Preparation
Download [ShapenetCorev2](https://shapenet.org)(*.obj). 
Place them under `util/shapenet_renderer/data/`or any other path.

```
cd util/shapenet_renderer/
./render_car.sh car
./render_plane.sh plane
./render_chair.sh
```
Move the renders accordingly to `data_root` paths (default = `./data/render_view200_r1.2/${class_id}`) as specified inside the config files inside `config/` directory.  
  
#### Training
```
./scripts/train_shape_car.sh
./scripts/train_shape_chair.sh
./scripts/train_shape_plane.sh
```

#### Inference

```
./scripts/render_shape_car.sh
./scripts/render_shape_chair.sh
./scripts/render_shape_plane.sh
```

#### Evaluation
`${class_name}` can be replaced by car, chair, or plane.

```
cd evaluations/fid_scores/
./kid_${class_name}.sh
./fid_${class_name}.sh
```

### Objaverse
Download the [G-buffer Objaverse dataset](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse).

```
bash scripts/download_objaverse.sh
```

We only use part of the dataset and the data split is in [this file](./datasets/splits/objaverse/paper_train.txt).
The model training requires point clouds as input, the point clouds are in [this link](https://huggingface.co/yanghtr/AtlasGaussians/tree/main/Dataset/Objaverse/gobjaverse_pc). 

Unzip the file and put it under the `data_root` (default at `data/objaverse/`).

For more details of the dataset, please refer to Section A.1 in the appendix of the paper. Please also cite [G-buffer Objaverse dataset](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) if you use the point clouds.


##### VAE training
The VAE training consists of two stages. For the detailed commands, see [VAE Training Script](./scripts/vae/train_dgx.sh). Note that after training, we pre-compute the latents for latent diffusion.

```
./scripts/vae/train_dgx.sh
```

##### LDM training

```
./scripts/ldm/train_dgx.sh
```




## Acknowledgements

The repo is built based on:

- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) (for codebase)
- [LGM](https://github.com/3DTopia/LGM) (for 3DGS)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (for 3DGS)
- [G-buffer Objaverse](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) (for Objaverse dataset)
- [DiffTF](https://github.com/ziangcao0312/DiffTF/) (for ShapeNet dataset)


## Reproducing figures from the paper

Download the checkpoint from [this link](https://huggingface.co/yanghtr/AtlasGaussians/tree/main/output). 
Place them according to original path.
To reproduce the results of Figure 6 in the paper:

```
./scripts/render_paper.sh
```
