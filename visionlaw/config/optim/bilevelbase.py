from dataclasses import dataclass, field
from visionlaw.utils import Config
from .upper import UpperBaseOptimConfig
from .lower import AdamOptimConfig

@dataclass(kw_only=True)
class BilevelBaseOptimConfig(Config, name='bileveloptim'):
    # upper level
    upper: UpperBaseOptimConfig = field(default_factory=UpperBaseOptimConfig)
    # lower level
    lower: AdamOptimConfig = field(default_factory=AdamOptimConfig)
