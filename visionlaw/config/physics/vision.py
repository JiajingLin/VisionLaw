from dataclasses import dataclass, field
from .base import BasePhysicsConfig
from .env import JellyEnvConfig
from .render import PyVistaRenderConfig
from .sim import VisionSimConfig

@dataclass(kw_only=True)
class VisionPhysicsConfig(BasePhysicsConfig, name='vision'):
    env: JellyEnvConfig = field(default_factory=JellyEnvConfig)
    render: PyVistaRenderConfig = field(default_factory=PyVistaRenderConfig)
    sim: VisionSimConfig = field(default_factory=VisionSimConfig)
