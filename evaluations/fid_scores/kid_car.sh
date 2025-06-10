#!/bin/bash

dataset_path=../../data/render_view200_r1.2/02958343/
ours_path=../../output/ldm/shapenet/car/results_E999/cfg_3.5/images/
dump_ours_path="./shapenet-results/pred/"
python kid_score.py --true $dataset_path --fake $ours_path --dataset shapenet --reso 128 --basepath $dump_ours_path


