from dataclasses import dataclass, field
from visionlaw.utils import Config
from .shape import PCDShapeConfig

@dataclass(kw_only=True)
class ParticlesConfig(Config, name='particles'):
    shape: PCDShapeConfig = field(default_factory=PCDShapeConfig)
    rho: float = 1e3
    clip_bound: float = 0.1
    mesh_path: str = None    # Replace this with the path to the reconstructed mesh
    mesh_sample_mode: str = 'volumetric'    
    mesh_sample_resolution: int = 32      # For sampling from the mesh, larger values will result in more particles
    