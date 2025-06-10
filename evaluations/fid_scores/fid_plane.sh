#!/bin/bash

dataset_path=../../data/render_view200_r1.2/02691156/
ours_path=../../output/ldm/shapenet/plane/results_E999/cfg_3.5/images/
dump_ours_path="./shapenet-results/pred/ours/fid128"
python fid_score.py $dataset_path $ours_path --dataset shapenet --num-workers 8 --reso 128 --basepath $dump_ours_path


