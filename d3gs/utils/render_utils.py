import os
import sys
import time
import torch
import mediapy
import imageio
import trimesh
import warp as wp
import numpy as np
from PIL import Image
from pathlib import Path
from natsort import natsorted
from typing import Optional, List, Tuple
from typing_extensions import Literal
from d3gs.scene.gaussian_model import GaussianModel
from d3gs.gaussian_renderer import get_rasterizer
from d3gs.utils.simulation_utils import (
    torch2warp_mat33,
    deform_cov_by_F
)

def diff_rasterization(
    x: torch.Tensor,
    deform_grad: Optional[torch.Tensor],
    gaussians: Optional[GaussianModel],
    view_cam,
    background_color: torch.Tensor,
    gaussians_active_sh: Optional[int] = None,
    guassians_cov: Optional[torch.Tensor] = None,
    gaussians_opa: Optional[torch.Tensor] = None,
    gaussians_shs: Optional[torch.Tensor] = None,
    scaling_modifier: Optional[float] = 1.,
    force_mask_data: Optional[bool] = False
) -> torch.Tensor:  
    device = x.device
    means3D = x

    if gaussians is not None:
        cov3D_precomp = gaussians.get_covariance(scaling_modifier=scaling_modifier)
        opacity = gaussians.get_opacity
        shs = gaussians.get_features
        sh_degree = gaussians.active_sh_degree
    else:
        cov3D_precomp = guassians_cov
        opacity = gaussians_opa
        shs = gaussians_shs
        sh_degree = gaussians_active_sh

    assert means3D.shape[0] == cov3D_precomp.shape[0], \
        f"Shape mismatch: means3D {means3D.shape[0]} cov3D {cov3D_precomp.shape[0]}"

    
    
    if deform_grad is not None:
        tensor_F = torch.reshape(deform_grad, (-1, 3, 3))
        wp_F = torch2warp_mat33(tensor_F, dvc=device.type)

        assert cov3D_precomp.shape[0] == tensor_F.shape[0], \
            f"Shape mismatch: cov3D {cov3D_precomp.shape[0]} F {tensor_F.shape[0]}"

        wp_cov3D_precomp = wp.from_torch(
            cov3D_precomp.reshape(-1),
            dtype=wp.float32
        )
        wp_cov3D_deformed = wp.zeros_like(wp_cov3D_precomp)
        wp.launch(
            deform_cov_by_F,
            dim=tensor_F.shape[0],
            inputs=[wp_cov3D_precomp, wp_F, wp_cov3D_deformed],
            device=device.type
        )
        wp.synchronize()

        cov3D_deformed = wp.to_torch(wp_cov3D_deformed).reshape(-1, 6)
    else:
        cov3D_deformed = cov3D_precomp

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points
    
    rasterizer = get_rasterizer(
        view_cam, sh_degree,
        debug=False, bg_color=background_color,
    )

    if force_mask_data:  
        # Rasterize visible Gaussians to image.
        rendered_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(means3D.shape[0], 3, device=device),
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D_deformed
            # scales=scales,
            # rotations=rotations,
            # cov3D_precomp=None
        )
    else:
        # Rasterize visible Gaussians to image.
        rendered_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D_deformed
            # scales=scales,
            # rotations=rotations,
            # cov3D_precomp=None
        )

    return rendered_image