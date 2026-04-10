from dataclasses import dataclass
from visionlaw.utils import Config

@dataclass(kw_only=True)
class BaseVelConfig(Config):
    random: bool
