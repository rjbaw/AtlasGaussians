#!/bin/bash

dataset_path=../../data/objaverse/gobjaverse/
ours_path=../../output/ldm/objaverse/ns49_AE800_kl1e-4_d512_m512_l16_d24_edm_cfg_sf133_8B27/results_E850/cfg_3.5/images/
dump_ours_path="./objaverse-results/pred/ours/fid128"
python fid_score.py $dataset_path $ours_path --dataset objaverse --num-workers 8 --reso 128 --basepath $dump_ours_path


