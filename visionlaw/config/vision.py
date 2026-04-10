from dataclasses import dataclass, field
from .base import BaseConfig
from .optim import AdamOptimConfig
from .optim import BilevelBaseOptimConfig
from .llm import GPT4LLMConfig
from .physics import VisionPhysicsConfig
from .gs import GaussiansConfig
from .particles import ParticlesConfig
from .dataset import VideoDatasetConfig
from .loss import BaseLossConfig

@dataclass(kw_only=True)
class VisionConfig(BaseConfig, name='vision'):
    optim: BilevelBaseOptimConfig = field(default_factory=BilevelBaseOptimConfig)
    llm: GPT4LLMConfig = field(default_factory=GPT4LLMConfig)
    physics: VisionPhysicsConfig = field(default_factory=VisionPhysicsConfig)
    gaussians: GaussiansConfig = field(default_factory=GaussiansConfig)
    particles: ParticlesConfig = field(default_factory=ParticlesConfig)
    dataset: VideoDatasetConfig = field(default_factory=VideoDatasetConfig)
    loss: BaseLossConfig = field(default_factory=BaseLossConfig)

    dataset_path: str = None
    ckpt_path: str = None
    is_dataset: str = True
