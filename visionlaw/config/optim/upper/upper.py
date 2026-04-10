from dataclasses import dataclass
from visionlaw.utils import Config

@dataclass(kw_only=True)
class UpperBaseOptimConfig(Config, name='upperoptim'):
    alternate_epochs: int = 10
    joint_epochs: int = 10


