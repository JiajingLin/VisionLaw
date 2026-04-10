import sys
import dataclasses
import random
from pathlib import Path
import argparse
import math

import yaml
from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import warp as wp

import visionlaw
from visionlaw.sim import MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer, MPMInitData, MPMCacheDiffSim

from visionlaw.utils.config_helper import parse_yaml_config, parse_unknown_args
from d3gs.scene.gaussian_model import GaussianModel

import os

from visionlaw.utils.binding_helper import *
from d3gs.utils.render_utils import diff_rasterization

from visionlaw.utils.visualization_helper import save_render_gt_video
from d3gs.utils.loss_utils import l1_loss, l2_loss
from visionlaw.utils.eval_helper import compute_eval_loss

from experiment.entry.vision.elastoplasticity.eval import eval_one_epoch
from visionlaw.dataset.video_guidance import VideoGuidance
import copy
import shutil

root: Path = visionlaw.utils.get_root(__file__)

PIXEL_LOSSES = {
    "l1": l1_loss,
    "l2": l2_loss
}

def train_one_epoch():
    pass
    


@torch.enable_grad()
def train(path: str, config: str, unknown_args):
    print("grad mode:", torch.is_grad_enabled())

    unknown_args = unknown_args + parse_yaml_config(config)
    cfg = visionlaw.config.VisionConfig(path=path)
    cfg.update(unknown_args)
    
    
    tpos = cfg.tpos


    num_steps = cfg.physics.sim.num_steps = cfg.physics.sim.nframes * cfg.physics.sim.substeps

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

    log_root = root / 'log'
    if Path(cfg.path).is_absolute():
        exp_root = Path(cfg.path)
    else:
        exp_root = log_root / cfg.path
    if exp_root.is_relative_to(log_root):
        exp_name = str(exp_root.relative_to(log_root))
    else:
        exp_name = str(exp_root)
    state_root = exp_root / 'state'
    ckpt_root = exp_root / 'ckpt'
    visionlaw.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    state_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    with (exp_root / 'config.yaml').open('w') as f:
        yaml.safe_dump(cfg_dict, f)

    writer = SummaryWriter(exp_root, purge_step=0)

    guidance = VideoGuidance(cfg.dataset, use_ssim=cfg.loss.use_ssim, use_flow=cfg.loss.use_flow)
    
    

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


    pixel_loss = PIXEL_LOSSES[cfg.loss.pixel_loss]

    model = MPMModelBuilder().parse_cfg(cfg.physics.sim).finalize(wp_device, requires_grad=True)
    statics_initializer = MPMStaticsInitializer(model)

    init_data = MPMInitData.get(cfg.particles)
    init_data.set_lin_vel(cfg.physics.env.vel.lin_vel)
    init_data.set_ang_vel(cfg.physics.env.vel.ang_vel)

    statics_initializer.add_group(init_data)
    statics = statics_initializer.finalize()


    diff_sim = MPMCacheDiffSim(model, statics, num_steps=num_steps)
    
    print("\n===================================")  
    print(f'Loading initial velocity from checkpoint ...\n')
    init_x_and_v = torch.load(cfg.dataset.data.init_path, map_location="cpu")
    guidance.set_init_x_and_v(init_x=init_x_and_v['init_x'], init_v=init_x_and_v['init_v'])
    print(f'\nInitial velocity obtained: {guidance.get_init_v.mean(0)}.')
    print("===================================")


    full_py = Path(cfg.physics.env.physics.path).read_text('utf-8')

    physics_py_path = exp_root / 'physics.py'
    physics_py_path.write_text(full_py, 'utf-8')

    plasticity: nn.Module = visionlaw.utils.get_class_from_path(physics_py_path, 'PlasticityModel')()
    plasticity.to(torch_device)
    elasticity: nn.Module = visionlaw.utils.get_class_from_path(physics_py_path, 'ElasticityModel')()
    elasticity.to(torch_device)

    plasticity_parametric = len(list(plasticity.parameters())) > 0
    elasticity_parametric = len(list(elasticity.parameters())) > 0

    if plasticity_parametric:
        if cfg.optim.lower.optimizer == 'adam':
            plasticity_optimizer = torch.optim.Adam(plasticity.parameters(), lr=cfg.optim.lower.lr)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optim.lower.optimizer}')

        if cfg.optim.lower.scheduler == 'cosine':
            plasticity_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(plasticity_optimizer, T_max=cfg.optim.lower.num_epochs)
        elif cfg.optim.lower.scheduler == 'none':
            plasticity_scheduler = None
        else:
            raise ValueError(f'Unknown scheduler: {cfg.optim.lower.scheduler}')
        
    if elasticity_parametric:
        if cfg.optim.lower.optimizer == 'adam':
            elasticity_optimizer = torch.optim.Adam(elasticity.parameters(), lr=cfg.optim.lower.lr)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optim.lower.optimizer}')   

        if cfg.optim.lower.scheduler == 'cosine':
            elasticity_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(elasticity_optimizer, T_max=cfg.optim.lower.num_epochs)
        elif cfg.optim.lower.scheduler == 'none':
            elasticity_scheduler = None
        else:
            raise ValueError(f'Unknown scheduler: {cfg.optim.lower.scheduler}')    

    num_epochs = cfg.optim.lower.num_epochs 
    num_frames = guidance.valid_num_frames
    frames_per_stage = cfg.optim.lower.frames_per_stage
    steps_per_frame = cfg.optim.lower.steps_per_frame

    assert num_frames % frames_per_stage == 0, "guidance.num_frames must be divisible by cfg.optim.lower.frames_per_stage"
    stages_per_epoch = num_frames // frames_per_stage

    t = trange(num_epochs + 1, desc=f'[train] {os.path.basename(exp_name)}', file=sys.stdout, position=tpos, leave=None)
    for epoch in t:
        
        renders_dict = {}
        gt_dict = {}

        if plasticity_parametric:
            for name, param in plasticity.named_parameters():
                writer.add_scalar(f'param/plasticity/{name}', param.item(), epoch)
        if elasticity_parametric:
            for name, param in elasticity.named_parameters():
                writer.add_scalar(f'param/elasticity/{name}', param.item(), epoch)

        x, v, C, F, _ = guidance.get_init_material_data()        
        de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

        de_x_prev = de_x.clone().detach()
        g_prev = gaussians.get_xyz.clone().detach()
        
        plasticity.train()
        elasticity.train()
        for stage in range(stages_per_epoch):

            if plasticity_parametric:
                plasticity_optimizer.zero_grad()
            if elasticity_parametric:
                elasticity_optimizer.zero_grad()
            renders = []
            gts = []

            for frame in range(frames_per_stage):

                for step in range(steps_per_frame):
                    cur_step = stage * frames_per_stage * steps_per_frame + frame * steps_per_frame + step
                    
                    F = plasticity(F)
                    stress = elasticity(F)
                    x, v, C, F = diff_sim(cur_step, x, v, C, F, stress)

                cur_frame = stage * frames_per_stage + frame + 1
                
                de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)
                means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
                deform_grad = compute_bindings_F(F, bindings)
                
                for view in range(guidance.valid_num_views):
                    cam = guidance.getCameras(guidance.views[view], cur_frame)
                    
                    render = diff_rasterization(
                    means3D, deform_grad, gaussians,
                    cam, background,
                    scaling_modifier=cfg.gaussians.scaling_modifier,
                    force_mask_data=force_mask_data
                    )
                    gt = cam.original_image.to(x.device)
                    
                    renders.append(render)
                    gts.append(gt)

                    
                    if view in renders_dict:
                        renders_dict[view].update({
                            cur_frame: render.clone().detach()
                        })
                    else:
                        renders_dict.update({
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
                    
                
                de_x_prev = de_x.clone().detach() 
                g_prev = means3D.clone().detach()
            
            renders = torch.stack(renders, dim=0)
            gts = torch.stack(gts, dim=0)
            loss_dict = guidance(renders, gts)
            loss_rgb = loss_dict['loss_rgb']
            loss_ssim = loss_dict['loss_ssim']

            loss = loss_rgb
            
            loss.backward()
        
            try:
                if elasticity_parametric:
                    for param in elasticity.parameters():
                        if param is not None and param.grad is not None:
                            torch.nan_to_num_(param.grad, 0.0, 0.0, 0.0)
                    clip_grad_norm_(elasticity.parameters(), 1.0, error_if_nonfinite=True)
                    elasticity_optimizer.step()
                    if elasticity_scheduler is not None:
                        elasticity_scheduler.step()
                if plasticity_parametric:
                    for param in plasticity.parameters():
                        if param is not None and param.grad is not None:
                            torch.nan_to_num_(param.grad, 0.0, 0.0, 0.0)
                    clip_grad_norm_(plasticity.parameters(), 1.0, error_if_nonfinite=True)
                    plasticity_optimizer.step()
                    if plasticity_scheduler is not None:
                        plasticity_scheduler.step()
                    
            except RuntimeError as e:
                tqdm.write(str(e))
                break
        
            x, v, C, F = (
                x.detach(),
                v.detach(),
                C.detach(),
                F.detach()
            )
        
        save_render_gt_video(renders_dict, gt_dict, os.path.join(exp_root, f'epoch_{epoch}'), fps=30)

        with torch.no_grad():

            for name, param in plasticity.named_parameters():
                print(f"plasticity {name}: {param.item()}")
            for name, param in elasticity.named_parameters():
                print(f"elasticity {name}: {param.item()}")
                
            metrics = eval_one_epoch(cfg, epoch, exp_root, guidance, plasticity, elasticity, gaussians, bindings, copy.deepcopy(init_data))
            
            print(f"metrics: {metrics['rgb_loss'] * 10000:.6f}, {metrics['psnr']:.4f}, {metrics['ssim']:.4f}")
            
            loss_eval = compute_eval_loss(renders_dict, gt_dict, pixel_loss)
            loss_eval_item = loss_eval.item() * 10000
            t.set_postfix(l_eval=f'{loss_eval_item:.6f}')
            

            writer.add_scalar('loss/rgb', metrics['rgb_loss'] * 10000, epoch)
            writer.add_scalar('metric/psnr', metrics['psnr'], epoch)
            writer.add_scalar('metric/ssim', metrics['ssim'], epoch)
            
        torch.save(plasticity.state_dict(), ckpt_root / f'plasticity_{epoch:04d}.pt')
        torch.save(elasticity.state_dict(), ckpt_root / f'elasticity_{epoch:04d}.pt')
            
        if epoch == num_epochs:
            t.refresh()
            break

        if not plasticity_parametric and not elasticity_parametric:
            break
        
    t.close()
    writer.close()
    
    losses, params, metrics = visionlaw.utils.parse_tensorboard(path)
    loss_rgb_curve = losses['rgb']

    best_epoch = np.argmin(loss_rgb_curve)

    shutil.copy(ckpt_root / f'plasticity_{best_epoch:04d}.pt', ckpt_root / 'final_plasticity.pt')
    shutil.copy(ckpt_root / f'elasticity_{best_epoch:04d}.pt', ckpt_root / 'final_elasticity.pt')
    
    eval_key = 'final'
    ind_root = Path(path)

    eval_args = {
        'tpos': tpos,
        'physics.env.physics.path': cfg.physics.env.physics.path,
        'ckpt_path': ind_root / 'ckpt' / f'{eval_key}.pt'
    }

def main(path: str, config: str, **unknown_args):

    args = parse_unknown_args(unknown_args) + parse_yaml_config(config)
    cfg = visionlaw.config.VisionConfig(path=path)
    cfg.update(args)

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    

    
    train(path, config, **unknown_args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--config', type=str)
    args, unknown_args = parser.parse_known_args()
    
    train(args.path, args.config, unknown_args)

