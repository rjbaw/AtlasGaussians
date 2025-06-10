#!/bin/sh
conda env create -f env.yml
conda activate atlas
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
conda install -y cuda -c nvidia
pip install -r requirements
