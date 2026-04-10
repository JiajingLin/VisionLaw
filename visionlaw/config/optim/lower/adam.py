from dataclasses import dataclass
from .base import LowerBaseOptimConfig

@dataclass(kw_only=True)    
class AdamOptimConfig(LowerBaseOptimConfig, name='adam'):
    num_epochs: int = 30
    optimizer: str = 'adam'
    lr: float = 3e-2  # learning rate 0.03
    scheduler: str = 'none'
    # num_teacher_steps: int = 100
    num_teacher_steps: int = 1

    alpha_position: float = 1e4
    alpha_velocity: float = 1e1
