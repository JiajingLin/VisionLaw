from dataclasses import dataclass, field
from .base import BaseSimConfig

# This file defines the low simulation configuration for the MPM model in the visionlaw framework.
@dataclass(kw_only=True)
class VisionSimConfig(BaseSimConfig, name='vision'):
    nframes: int = 400
    substeps: int = 1
    
    gravity: list[float] = field(default_factory=lambda: [0.0, -9.8, 0.0])
    bc: str = 'freeslip'
    num_grids: int = 20
    dt: float = 5e-4
    bound: int = 3
    eps: float = 1e-7
    skip_frames: int = 1
