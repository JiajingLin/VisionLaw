from dataclasses import dataclass, field
from visionlaw.utils import Config

@dataclass(kw_only=True)
class VideoDataConfig(Config, name='videodata'):
    path: str = '/path/to/BouncyBall'
    transformsfile: str = 'data_dynamic.json'
    white_background: bool = True
    exclude_steps: list[int] = field(default_factory=lambda: [])
    used_views: list[str] = field(default_factory=lambda: [])
    force_mask_data: bool = False
    read_mask_only: bool = False  
    
    init_path: str = '/path/to/init.pt'
    nframes: int = 400
    skip_frames: int = 1  
    
    



