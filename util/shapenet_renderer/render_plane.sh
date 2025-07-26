#!/bin/bash
set -x

# input parameter
obj_id=$1
split=$2

# rendering parameters
dataset_path=./data/ShapeNetCore.v2/02691156 #mesh path
blender_path=./blender-2.79-linux-glibc219-x86_64
output_path=./output/renders
view=200
radius=1.2
mode=test

output_dir=${output_path}/${obj_id}_view${view}_r${radius}
mkdir -p ${output_dir}

find "${dataset_path}"* -name '*.obj' -print0 \
  | xargs -0 -P1 -I {} "${blender_path}/blender" \
      --background \
      --python shapenet_spherical_renderer.py -- \
        --output_dir "$output_dir" \
        --mesh_fpath {} \
        --num_observations ${view} \
        --sphere_radius ${radius} \
        --mode=${mode}
