from dataclasses import dataclass
from visionlaw.utils import Config
from typing import Callable

@dataclass(kw_only=True)
class BaseLossConfig(Config, name='loss'):
    # function
    pixel_loss: str = 'l2'
    decay_steps: int = 80
    decay_init: float = 0.5
    decay_final: float = 1.0
    lambda_max_decay: float = 0.5
    
    use_ssim: bool = False
    use_flow: bool = False