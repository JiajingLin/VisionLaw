from pathlib import Path
import dataclasses
import sys
import os
import argparse
import random
import math
from pprint import pprint
import time
import yaml
from tqdm import trange, tqdm
import numpy as np
import torch

import visionlaw
from visionlaw.agent import ConstitutivePhysicist, Population, BiphaselConstitutivePhysicist, BiphasePopulation

from visionlaw.utils.config_helper import parse_yaml_config
from experiment.entry.vision.elastoplasticity.train import train
from experiment.entry.vision.elastoplasticity.eval import eval


root = visionlaw.utils.get_root(__file__)


def get_perf_feedback(losses: dict[str, list[float]], params: dict[str, list[float]]) -> list[str]:
    feedbacks = []

    parametric = len(params) > 0

    if parametric:
        best_idx = min(range(len(losses['rgb'])), key=lambda i: losses['rgb'][i] if not math.isnan(
            losses['rgb'][i]) else float('inf'))  

        feedbacks.append('#### Physical parameter training curves (versus iteration)')
        feedbacks.append('')
        for tag, traj in sorted(params.items()):  
            msg = ', '.join([f'{loss:.2f}' for loss in traj])  
            msg = f'- {tag}: [{msg}] (Best: {traj[best_idx]:.2f})'
            feedbacks.append(msg)

        feedbacks.append('')

        feedbacks.append('#### Loss training curves (versus iteration)')
        feedbacks.append('')
        for tag, traj in sorted(losses.items()):  
            msg = ', '.join([f'{loss:.4f}' for loss in traj])  
            if tag == 'rgb':
                tag = f'{tag} (Key loss)'
            msg = f'- {tag}: [{msg}] (Best: {traj[best_idx]:.4f})'
            feedbacks.append(msg)
    else:
        feedbacks.append('#### Evaluation loss (since it is a non-parametric model)')
        feedbacks.append('')
        for tag, traj in sorted(losses.items()):
            msg = f'{traj[-1]:.4f}'
            if tag == 'rgb':
                tag = f'{tag} (Key loss)'
            msg = f'- {tag}: [{msg}]'
            feedbacks.append(msg)

    return feedbacks

def get_state_feedback(states: dict[str, torch.Tensor | tuple], state_size: int) -> list[str]:
    feedbacks = []

    key_indices: list[int] = list(states['key_indices'])
    key_frames: list[int] = np.linspace(0, states['x'].size(0) - 1, state_size, dtype=int).tolist()

    particle_frame_pos: np.ndarray = states['x'].permute(1, 0, 2).detach().cpu().numpy().copy()
    particle_frame_vel: np.ndarray = states['v'].permute(1, 0, 2).detach().cpu().numpy().copy()

    feedbacks.append('#### Representative particle trajectories (versus time)')
    feedbacks.append('')
    for i_particle, particle in enumerate(key_indices):
        
        
        feedbacks.append(f'- Particle {i_particle}')
        pos_msgs = []
        vel_msgs = []
        
        for frame in key_frames: 
            pos_msg = ', '.join([f'{pos:.2f}' for pos in particle_frame_pos[particle, frame]])
            pos_msg = f'({pos_msg})'
            pos_msgs.append(pos_msg)
            vel_msg = ', '.join([f'{vel:.2f}' for vel in particle_frame_vel[particle, frame]])
            vel_msg = f'({vel_msg})'
            vel_msgs.append(vel_msg)
        pos_msg = ', '.join(pos_msgs)
        pos_msg = f'    - positions: [{pos_msg}]'
        feedbacks.append(pos_msg)
        vel_msg = ', '.join(vel_msgs)
        vel_msg = f'    - velocities: [{vel_msg}]'
        feedbacks.append(vel_msg)

    return feedbacks

@torch.no_grad()
def main():
    python_path = Path(sys.executable).resolve() 
    my_env = os.environ.copy()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--config', type=str)
    args, unknown_args = parser.parse_known_args()
    print(args, unknown_args, sep='\n')
    unknown_args = unknown_args + parse_yaml_config(args.config)
    cfg = visionlaw.config.VisionConfig(path=args.path)
    cfg.update(unknown_args) 
    pprint(cfg) 

    train_py_path = root / 'entry' / 'vision' / 'elastoplasticity' / 'train.py' 
    eval_py_path = root / 'entry' / 'vision' / 'elastoplasticity' / 'eval.py' 
    
    
    train_worker = visionlaw.utils.get_function_from_path(train_py_path, 'train')
    eval_worker = visionlaw.utils.get_function_from_path(eval_py_path, 'eval')
    
    base_train_cmds = [python_path, train_py_path] 
    base_eval_cmds = [python_path, eval_py_path] 

    tpos = cfg.tpos
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_root = root / 'log'  
    if Path(cfg.path).is_absolute():    
        exp_root = Path(cfg.path)
    else:
        exp_root = log_root / cfg.path
    if exp_root.is_relative_to(log_root):
        exp_name = str(exp_root.relative_to(log_root))
    else:
        exp_name = str(exp_root)

    primitive_root = exp_root / 'primitive'
    offspring_root = exp_root / 'offspring'
    iteration_root = exp_root / 'iteration'
    visionlaw.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume, verbose=True)
    primitive_root.mkdir(parents=True, exist_ok=True)
    offspring_root.mkdir(parents=True, exist_ok=True)
    iteration_root.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    yaml.safe_dump(cfg_dict, (exp_root / 'config.yaml').open('w')) 


    if cfg.llm.name.startswith('openai-gpt-4'):
        dataset_feedback = ''
    physicist = BiphaselConstitutivePhysicist(cfg.llm, seed=seed, env_info=dataset_feedback)
    population = BiphasePopulation(cfg.llm)
    
    
    for i_ind, ind_physics in enumerate(tqdm(cfg.llm.primitives, desc=f'[primitive] {exp_name}', file=sys.stdout, position=tpos)):
        ind_root = primitive_root / f'{i_ind:04d}'  
        ind_root.mkdir(parents=True, exist_ok=True)


        train_args = {
            'tpos': tpos + 1,
            'path': str(ind_root),
            'config': args.config,
            'physics.env.physics': ind_physics
        }
        
        print("grad mode:", torch.is_grad_enabled())

        error = visionlaw.utils.run_exp(base_train_cmds, train_args, unknown_args, my_env) 
        print("grad mode:", torch.is_grad_enabled())

        losses, params, metrics = visionlaw.utils.parse_tensorboard(ind_root) 
        states = None
        
        fitness = min(losses['rgb'], key=lambda x: x if not math.isnan(x) else float('inf')) 
        
        feedbacks = []
            
        feedbacks += get_perf_feedback(losses, params)
        feedback = '\n'.join(feedbacks)
        code_path = ind_root / 'physics.py'

        
        population.add_primitive(code_path, feedback, fitness, losses, params, states, metrics, ind_root)
    
    total_epochs = cfg.optim.upper.alternate_epochs * 2 + cfg.optim.upper.joint_epochs
    for i_iter in trange(total_epochs + 1, desc=f'[iteration] {exp_name}', file=sys.stdout, position=tpos):
        
        if i_iter < cfg.optim.upper.alternate_epochs * 2:
            if i_iter % 2 == 0:
                entry = 'elasticity'
                print(f"Alternate Optimization epoch {i_iter} : {entry}")
            else:
                entry = 'plasticity'
                print(f"Alternate Optimization epoch {i_iter} : {entry}")
        
        else:
            entry = 'elastoplasticity'
            print(f"Joint Optimization epoch {i_iter} : {entry}")
            
        iter_ind_root = offspring_root / f'{i_iter:04d}'  
        iter_ind_root.mkdir(parents=True, exist_ok=True)

        iter_root = iteration_root / f'{i_iter:04d}'  
        iter_root.mkdir(parents=True, exist_ok=True)

        indices = population.sample(iter_root, entry) 
        
        msgs = physicist.get_msgs(population, indices, iter_root / 'messages', entry) 
        
        if i_iter == total_epochs:
            break
        
        if entry == 'elasticity' or entry == 'plasticity':
            generation_times = getattr(cfg.llm, entry).batch_size / len(indices)
            print(f"generation_times: {generation_times}")
        else:
            generation_times = 1
        
        response = physicist.generate(msgs, iter_root / 'choices', iter_root, entry, generation_times) 
        
         
        
        
        for i_ind, ind_choice in enumerate(tqdm(response.choices, desc=f'[offspring] {exp_name}', file=sys.stdout, position=tpos + 1)):
            
            try:
                ind_root = iter_ind_root / f'{i_ind:04d}'
                ind_root.mkdir(parents=True, exist_ok=True)
 
                code_path = ind_choice.dump_root / 'code.py'    

                if len(ind_choice.code) == 0:
                    raise RuntimeError('No code generated or generated solution violated format requirements.')


                train_args = {
                    'tpos': tpos + 2,
                    'path': str(ind_root),
                    'config': args.config,
                    'physics.env.physics.path': code_path  
                }

                error = visionlaw.utils.run_exp(base_train_cmds, train_args, unknown_args, my_env)
                if len(error) > 0:
                    raise RuntimeError(error.rsplit('\n', maxsplit=1)[-1])
                
                losses, params, metrics = visionlaw.utils.parse_tensorboard(ind_root) 
                states = None
                
                fitness = min(losses['rgb'], key=lambda x: x if not math.isnan(x) else float('inf'))
                
                feedbacks = []

                feedbacks += get_perf_feedback(losses, params)
                feedback = '\n'.join(feedbacks)
                
                population.add_offspring(ind_choice, feedback, fitness, losses, params, states, metrics, ind_root)
            except Exception as e:
                feedback = str(e)           
                fitness = float('inf')      
                losses = None
                params = None
                states = None
                metrics = None
                population.add_offspring(ind_choice, feedback, fitness, losses, params, states, metrics, ind_root)
        
        indices = population.sample(None, entry)
        for i, idx in enumerate(indices):
            save_path = iter_root / f'best_{i:04d}'
            save_path.mkdir(parents=True, exist_ok=True)
            population.offsprings[idx].dump_best(save_path)
    
if __name__ == '__main__':
    main()
