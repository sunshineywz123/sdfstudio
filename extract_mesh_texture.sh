#!/bin/bash
YAML="outputs/data-custom-pikaqiu_dazhanbi/neus-facto/2023-07-06_202056/config.yml"
PLY="meshes/pikaqiu_dazhanbi.ply"
ns-extract-mesh --load-config $YAML --output-path $PLY
python scripts/texture.py --load-config $YAML --input-mesh-filename $PLY --output-dir ./textures_pikaqiu --target_num_faces 50000