#!/bin/bash
YAML="outputs/neus-facto-dtu65/neus-facto/2023-07-07_131531/config.yml"
PLY="meshes/sheep_out_track.ply"
ns-extract-mesh --load-config $YAML --resolution 1024 --create_visibility_mask True --output-path $PLY 
python scripts/texture.py --load-config $YAML --input-mesh-filename $PLY --output-dir ./textures_sheep_out_track --target_num_faces 600000