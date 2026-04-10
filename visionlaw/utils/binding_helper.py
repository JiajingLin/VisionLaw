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

def denormalize_points_helper_func(points, size, center):
    if isinstance(size, np.ndarray):
        size = torch.from_numpy(size).to(points)
    if isinstance(center, np.ndarray):
        center = torch.from_numpy(center).to(points)

    denorm_points = (points.clone() - center) / size

    return denorm_points

def compute_bindings_xyz(
    p_curr: torch.Tensor,
    p_prev: torch.Tensor,
    k_prev: torch.Tensor,
    bindings: torch.Tensor,
):
    """Compute updated location of gaussian kernels.

    Args:
        p_curr: Current particles xyz.
        p_prev: Previous particles xyz.
        k_prev: Previous kernels xyz.
        bindings: Binding matrix.
    
    Returns:
        k_curr: Updated kernels xyz.
    """
    delta_x = p_curr - p_prev.detach()

    # calculate means3D
    delta_means3D = torch.sparse.mm(bindings, delta_x)
    delta_means3D = delta_means3D.to_dense()
    k_curr = k_prev.detach() + delta_means3D

    return k_curr


def compute_bindings_F(
    deform_grad: torch.Tensor,
    bindings: torch.Tensor,
):
    """Compute updated deformation gradiant for each gaussian kernel.

    Args:
        deform_grad: Deformation gradient of each particle.
        bindings: Binding matrix.
    
    Returns:
        tensor_F: Deformation gradiant for each gaussian kernel.
    """

    # calculate deformation gradient
    tensor_F = torch.reshape(deform_grad, (-1, 9))
    tensor_F = torch.sparse.mm(bindings, tensor_F)
    tensor_F = tensor_F.to_dense()

    # reshape to (kernels, 3, 3)
    tensor_F = torch.reshape(tensor_F, (-1, 3, 3))
    return tensor_F
