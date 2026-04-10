import sys

import dataclasses
import random
from pathlib import Path
import argparse
import yaml

import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import warp as wp

import visionlaw
from visionlaw.dataset import video_guidance
from visionlaw.sim import MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer, MPMInitData, MPMForwardSim, VolumeElasticity, SigmaElasticity

from omegaconf import DictConfig, OmegaConf

from visionlaw.utils.config_helper import parse_yaml_config, parse_unknown_args
from d3gs.scene.gaussian_model import GaussianModel

import os

from visionlaw.utils.binding_helper import *
from d3gs.utils.render_utils import diff_rasterization

from visionlaw.utils.visualization_helper import save_video, save_render_video, save_render_gt_video

import copy

from visionlaw.dataset.video_guidance import VideoGuidance
from visionlaw.config import VisionConfig
from d3gs.utils.loss_utils import l1_loss, l2_loss
from torchvision.utils import save_image
from datetime import datetime

root: Path = visionlaw.utils.get_root(__file__)

@torch.no_grad()
def forward(path: str, config: str, model_dir: str, scene_name: str, unknown_args):

    unknown_args = unknown_args + parse_yaml_config(config)
    cfg = visionlaw.config.VisionConfig(path=path)
    cfg.update(unknown_args)
    

    
    wp.config.quiet = True
    wp.init()
    wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': True})

    torch_device = torch.device(f'cuda:{cfg.gpu}')
    torch.backends.cudnn.benchmark = True
    visionlaw.warp.replace_torch_svd()
    visionlaw.warp.replace_torch_polar()
    visionlaw.warp.replace_torch_trace()
    visionlaw.warp.replace_torch_cbrt()
    torch.set_default_device(torch_device)
    
    
    exp_root = Path(cfg.path)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    export_dir = exp_root / f"{scene_name}_{current_time}"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    sim_images_root = export_dir / 'images'
    sim_pts_root =  export_dir / 'pts'
    sim_images_root.mkdir(parents=True, exist_ok=True)
    sim_pts_root.mkdir(parents=True, exist_ok=True)
     
    model_dir = Path(model_dir)
    physics_py_path = model_dir / 'physics.py'
    assert physics_py_path.exists(), f"Physics python file {physics_py_path} does not exist"
    


    plasticity: nn.Module = visionlaw.utils.get_class_from_path(physics_py_path, 'PlasticityModel')()
    plasticity.to(torch_device)
    elasticity: nn.Module = visionlaw.utils.get_class_from_path(physics_py_path, 'ElasticityModel')()
    elasticity.to(torch_device)

    plasticity_ckpt_path = Path(model_dir/ 'ckpt' / 'final_plasticity.pt')
    elasticity_ckpt_path = Path(model_dir / 'ckpt' / 'final_elasticity.pt')
    plasticity_ckpt = torch.load(plasticity_ckpt_path, map_location='cpu')
    elasticity_ckpt = torch.load(elasticity_ckpt_path, map_location='cpu')
    plasticity.load_state_dict(plasticity_ckpt)
    elasticity.load_state_dict(elasticity_ckpt)
    plasticity.requires_grad_(False)
    plasticity.eval()
    elasticity.requires_grad_(False)
    elasticity.eval()
    
    cfg.dataset.data.skip_frames = 1
    cfg.dataset.data.used_views = args.debug_views
    guidance = VideoGuidance(cfg.dataset)
    print("\n===================================")  
    print(f'Loading initial velocity from checkpoint ...\n')
    init_x_and_v = torch.load(cfg.dataset.data.init_path, map_location="cpu")
    guidance.set_init_x_and_v(init_x=init_x_and_v['init_x'], init_v=init_x_and_v['init_v'])

    print(f'\nInitial velocity obtained: {guidance.get_init_v.mean(0)}.')
    print("===================================")
    

    force_mask_data = cfg.dataset.data.read_mask_only
    if force_mask_data:

        cfg.dataset.data.white_background = False
        print(f"[Warning] Force to use black background when loading mask data")

    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if cfg.dataset.data.white_background
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )

    gaussians = GaussianModel(cfg.gaussians.sh_degree)
    gaussians.load_ply(cfg.gaussians.kernels_path, requires_grad=False)


    bind_data = torch.load(os.path.join(os.path.dirname(cfg.gaussians.kernels_path),'bindings.pt'))

    bindings: torch.Tensor = torch.sparse_coo_tensor(
        bind_data['bindings_ind'], bind_data['bindings_val'], bind_data['bindings_size']
    ).to(torch_device).float()

    model = MPMModelBuilder().parse_cfg(cfg.physics.sim).finalize(wp_device)
    state_initializer = MPMStateInitializer(model)
    statics_initializer = MPMStaticsInitializer(model)

    
    init_data = MPMInitData.get(cfg.particles)

    init_data.set_ind_vel(guidance.get_init_v.cpu().numpy())


    state_initializer.add_group(init_data)
    statics_initializer.add_group(init_data)

    state, _ = state_initializer.finalize()
    statics = statics_initializer.finalize()
    sim = MPMForwardSim(model, statics)    
    
    

    x, v, C, F, _ = guidance.get_init_material_data()
        
    de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

    de_x_prev = de_x.clone().detach()
    g_prev = gaussians.get_xyz.clone().detach()
    
    
    render_dict = {}
    gt_dict = {}
    
    renders = []
    gts = []
    cfg.physics.sim.num_steps = cfg.dataset.data.nframes * cfg.physics.sim.substeps
    for it in trange(0, cfg.physics.sim.num_steps, desc=f'[eval] {scene_name}', file=sys.stdout, leave=None):
        F_corrected = plasticity(F)
        stress = elasticity(F_corrected)
        state.from_torch(F=F_corrected, stress=stress)
        x, v, C, F = sim(state)
        
        if (it + 1) % (cfg.physics.sim.substeps * cfg.dataset.data.skip_frames) == 0:

            cur_frame = (it + 1) // (cfg.physics.sim.substeps * cfg.dataset.data.skip_frames)

            de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)
            means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
            deform_grad = compute_bindings_F(F, bindings)

            t = trimesh.PointCloud(vertices=de_x.clone().detach().cpu())
            t.export(sim_pts_root / f'{cur_frame * cfg.dataset.data.skip_frames:03d}.ply')
            
            for view in guidance.views:
                cam = guidance.getCameras(view, cur_frame)

                render = diff_rasterization(
                    means3D, deform_grad, gaussians,
                    cam, background,
                    scaling_modifier=cfg.gaussians.scaling_modifier,
                    force_mask_data=force_mask_data
                )
                gt = cam.original_image.to(x.device)
                

                save_image(render, sim_images_root / f"{view}_{cur_frame * cfg.dataset.data.skip_frames:03d}.png")

                
                render = render.to('cpu')
                gt = gt.to('cpu')
                

                
                if view in render_dict:
                    render_dict[view].update({
                        cur_frame: render.clone().detach()
                    })
                else:
                    render_dict.update({
                        view: {cur_frame: render.clone().detach()}
                    }) 
                
                if view in gt_dict:
                    gt_dict[view].update({
                        cur_frame: gt.clone().detach()
                    })
                else:
                    gt_dict.update({
                        view: {cur_frame: gt.clone().detach()}
                    })
                
                renders.append(render)
                gts.append(gt)
                

            de_x_prev = de_x.clone().detach()
            g_prev = means3D.clone().detach()
        
        
    save_render_gt_video(render_dict, gt_dict, export_dir, fps=200)
    
    renders = torch.stack(renders).to('cuda')
    gts = torch.stack(gts).to('cuda')
    

    metrics = guidance.evaluate(renders, gts)


    return metrics
    
    
        
if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./results')
    parser.add_argument('--config', type=str, default='experiment/configs/finetune-bb.yaml')
    parser.add_argument('--model_dir', type=str, default='xxx')
    args, unknown_args = parser.parse_known_args()
    
    scene_name = args.config.split('.')[0].split('-')[-1]
    args.debug_views = ['d_0']
    metrics = forward(args.path, args.config, args.model_dir, scene_name, unknown_args)
    print(metrics)
