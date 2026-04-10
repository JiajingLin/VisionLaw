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

root: Path = visionlaw.utils.get_root(__file__)

@torch.no_grad()
def eval_one_epoch(cfg: VisionConfig, epoch: int, exp_root: Path, guidance: VideoGuidance, plasticity, elasticity, gaussians, bindings, init_data: MPMInitData):
    assert guidance is not None

    wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    torch_device = torch.device(f'cuda:{cfg.gpu}')
    
    
    eval_exp_root = exp_root / 'eval'
    export_dir = eval_exp_root / f'epoch_{epoch}'
    export_dir.mkdir(parents=True, exist_ok=True)
        

    plasticity.eval()
    elasticity.eval()
    

    force_mask_data = cfg.dataset.data.read_mask_only
    if force_mask_data:

        cfg.dataset.data.white_background = False
        print(f"[Warning] Force to use black background when loading mask data")

    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if cfg.dataset.data.white_background
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )

    model = MPMModelBuilder().parse_cfg(cfg.physics.sim).finalize(wp_device)
    state_initializer = MPMStateInitializer(model)
    statics_initializer = MPMStaticsInitializer(model)

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
    
    num_frames = guidance.valid_num_frames
    steps_per_frame = cfg.optim.lower.steps_per_frame
    
    render_dict = {}
    gt_dict = {}
    
    renders = []
    gts = []
    
    for frame in range(num_frames):
        cur_frame = frame + 1
        
        for _ in range(steps_per_frame):
            F_corrected = plasticity(F)
            stress = elasticity(F_corrected)
            state.from_torch(F=F_corrected, stress=stress)
            x, v, C, F = sim(state)
        
        de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)
        means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
        deform_grad = compute_bindings_F(F, bindings)
        
        for view in range(len(guidance.views)):
            cam = guidance.getCameras(guidance.views[view], cur_frame)

            render = diff_rasterization(
                means3D, deform_grad, gaussians,
                cam, background,
                scaling_modifier=cfg.gaussians.scaling_modifier,
                force_mask_data=force_mask_data
            )
            
            gt = cam.original_image.to(x.device)
            

            
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
        
    save_render_gt_video(render_dict, gt_dict, export_dir, fps=30)
    
    metrics = guidance.evaluate(renders, gts)


    return metrics
    
    

@torch.no_grad()
def eval(path : str, config : str, guidance : VideoGuidance, **unknown_args):
    print("grad mode:", torch.is_grad_enabled())
    unknown_args = parse_unknown_args(unknown_args) + parse_yaml_config(config)
    cfg = visionlaw.config.VisionConfig(path=path)
    cfg.update(unknown_args)

    tpos = cfg.tpos

    wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    torch_device = torch.device(f'cuda:{cfg.gpu}')


    log_root = root / 'log'
    if Path(cfg.path).is_absolute():
        exp_root = Path(cfg.path)
    else:
        exp_root = log_root / cfg.path
    if exp_root.is_relative_to(log_root):
        exp_name = str(exp_root.relative_to(log_root))
    else:
        exp_name = str(exp_root)
    img_root = exp_root / 'img'
    vid_root = exp_root / 'vid'
    state_root = exp_root / 'state'
    visionlaw.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    img_root.mkdir(parents=True, exist_ok=True)
    vid_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    with (exp_root / 'config.yaml').open('w') as f:
        yaml.safe_dump(cfg_dict, f)



    full_py = Path(cfg.physics.env.physics.path).read_text('utf-8')
    full_py = full_py.format(**cfg.physics.env.physics.__dict__)
    physics_py_path = exp_root / 'physics.py'
    physics_py_path.write_text(full_py, 'utf-8')

    physics: nn.Module = visionlaw.utils.get_class_from_path(physics_py_path, 'Physics')()

    if cfg.ckpt_path is not None:
        ckpt_path = Path(cfg.ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        physics.load_state_dict(ckpt)

    physics.to(torch_device)
    physics.requires_grad_(False)
    physics.eval()

    used_views = cfg.dataset.data.used_views if len(cfg.dataset.data.used_views) > 0 else guidance.views 
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

    x, v, C, F, stress = state.to_torch()
    state_recorder = visionlaw.utils.StateRecorder()
    state_recorder.add_hyper(key_indices=cfg.physics.env.shape.key_indices)
    state_recorder.add(x=x, v=v)
    
    
        
    de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

    de_x_prev = de_x.clone().detach()
    g_prev = gaussians.get_xyz.clone().detach()

    render_dict = {}
    gt_dict = {}
    
    cfg.physics.sim.num_steps = cfg.physics.sim.nframes * cfg.physics.sim.substeps
    for it in trange(0, cfg.physics.sim.num_steps, desc=f'[eval] {exp_name}', file=sys.stdout, position=tpos, leave=None):


        
        stress = physics(F)
        state.from_torch(F=F, stress=stress)

        x, v, C, F = sim(state)
        
        de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)
        means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
        deform_grad = compute_bindings_F(F, bindings)
        

        if (it + 1) % cfg.physics.sim.substeps == 0:
            cur_step = (it + 1) // cfg.physics.sim.substeps
            cur_frame = guidance.steps[cur_step]

            if cur_frame in cfg.dataset.data.exclude_steps:
                continue
            for view in used_views:
                render = diff_rasterization(
                    means3D, deform_grad, gaussians,
                    guidance.getCameras(view, cur_frame), background,
                    scaling_modifier=cfg.gaussians.scaling_modifier,
                    force_mask_data=cfg.dataset.data.force_mask_data
                )
                if view in render_dict:
                    render_dict[view].update({
                        cur_frame: render
                    })
                else:
                    render_dict.update({
                        view: {cur_frame: render}
                    })
                    
                gt = guidance.getCameras(view, cur_frame).original_image.to(x.device)
                
                if view in gt_dict:
                    gt_dict[view].update({
                        cur_frame: gt
                    })
                else:
                    gt_dict.update({
                        view: {cur_frame: gt}
                    })

        de_x_prev = de_x.clone().detach()
        g_prev = means3D.clone().detach()

        state_recorder.add(x=x, v=v)

    save_render_gt_video(render_dict, gt_dict, vid_root, fps=90)
    state_recorder.save(state_root / 'ckpt.pt')
    

