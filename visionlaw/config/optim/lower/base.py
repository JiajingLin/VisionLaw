from dataclasses import dataclass
from visionlaw.utils import Config

@dataclass(kw_only=True)
class LowerBaseOptimConfig(Config):
    num_epochs: int
    optimizer: str

    alpha_position: float = 1e4
    alpha_velocity: float = 1e1
    
    frames_per_stage: int = 10
    steps_per_frame: int = 5
    
    
