#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch
import tyro
from rich.console import Console

from nerfstudio.model_components.ray_samplers import save_points
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import (
    get_surface_occupancy,
    get_surface_sliding,
    get_surface_sliding_with_contraction,
)
import ipdb
import sys
from nerfstudio.exporter.exporter_utils import (generate_point_cloud,get_mesh_from_filename)
from nerfstudio.exporter import texture_utils, tsdf_utils
CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


@dataclass
class ExtractMesh:
    """Load a checkpoint, run marching cubes, extract mesh, and save it to a ply file."""

    # Path to config YAML file.
    load_config: Path
    # Marching cube resolution.
    resolution: int = 1024
    # Name of the output file.
    output_path: Path = Path("output.ply")
    # Whether to simplify the mesh.
    simplify_mesh: bool = False
    # extract the mesh using occupancy field (unisurf) or SDF, default sdf
    is_occupancy: bool = False
    """Minimum of the bounding box."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Maximum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """marching cube threshold"""
    marching_cube_threshold: float = 0.0
    """create visibility mask"""
    create_visibility_mask: bool = False
    """save visibility grid"""
    save_visibility_grid: bool = False
    """visibility grid resolution"""
    visibility_grid_resolution: int = 512
    """threshold for considering a points is valid when splat to visibility grid"""
    valid_points_thres: float = 0.005
    """sub samples factor of images when creating visibility grid"""
    sub_sample_factor: int = 8
    """torch precision"""
    torch_precision: Literal["highest", "high"] = "high"

    def main(self) -> None:
        """Main function."""
        torch.set_float32_matmul_precision(self.torch_precision)
        assert str(self.output_path)[-4:] == ".ply"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        _, pipeline, _ = eval_setup(self.load_config)

        CONSOLE.print("Extract mesh with marching cubes and may take a while")
        #import ipdb;ipdb.set_trace()
        if self.create_visibility_mask:
            assert self.resolution % 512 == 0

            coarse_mask = pipeline.get_visibility_mask(
                self.visibility_grid_resolution, self.valid_points_thres, self.sub_sample_factor
            )

            def inv_contract(x):
                mag = torch.linalg.norm(x, ord=pipeline.model.scene_contraction.order, dim=-1)
                mask = mag >= 1
                x_new = x.clone()
                x_new[mask] = (1 / (2 - mag[mask][..., None])) * (x[mask] / mag[mask][..., None])
                return x_new

            if self.save_visibility_grid:
                offset = torch.linspace(-2.0, 2.0, 512)
                x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
                offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(coarse_mask.device)
                points = offset_cube[coarse_mask.reshape(-1) > 0]
                points = inv_contract(points)
                save_points("mask.ply", points.cpu().numpy())
                torch.save(coarse_mask, "coarse_mask.pt")

            get_surface_sliding_with_contraction(
                sdf=lambda x: (
                    pipeline.model.field.forward_geonetwork(x)[:, 0] - self.marching_cube_threshold
                ).contiguous(),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                coarse_mask=coarse_mask,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
                inv_contraction=inv_contract,
            )
            # import ipdb;ipdb.set_trace()
            
            # texture_utils.export_textured_mesh(
            #     mesh,
            #     pipeline,
            #     self.output_dir,
            #     px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            #     unwrap_method=self.unwrap_method,
            #     num_pixels_per_side=self.num_pixels_per_side,
            # )
            return

        if self.is_occupancy:
            # for unisurf
            get_surface_occupancy(
                occupancy_fn=lambda x: torch.sigmoid(
                    10 * pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous()
                ),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                level=0.5,
                device=pipeline.model.device,
                output_path=self.output_path,
            )
        else:
            assert self.resolution % 512 == 0
            # for sdf we can multi-scale extraction.
            get_surface_sliding(
                sdf=lambda x: pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous(),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                coarse_mask=pipeline.model.scene_box.coarse_binary_gird,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
            )
            if 0:
                mesh = get_mesh_from_filename(
				    str(self.output_path)
	            )
                CONSOLE.print("Texturing mesh with NeRF")
                ipdb.set_trace()
                texture_utils.export_textured_mesh(
				mesh,
				pipeline,
				self.output_path,
				num_pixels_per_side=2048
     	        )

def entrypoint():
    try:
        """Entrypoint for use with pyproject scripts."""
        tyro.extras.set_accent_color("bright_yellow")
        tyro.cli(tyro.conf.FlagConversionOff[ExtractMesh]).main()
    except:
      type, value, traceback = sys.exc_info()
      ipdb.post_mortem(traceback)

if __name__ == "__main__":

   try:
       tyro.cli(tyro.conf.FlagConversionOff[ExtractMesh]).main()
   except:
      type, value, traceback = sys.exc_info()
      ipdb.post_mortem(traceback)


# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ExtractMesh)  # noqa
