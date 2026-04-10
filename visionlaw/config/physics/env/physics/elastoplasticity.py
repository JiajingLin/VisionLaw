from dataclasses import dataclass
from pathlib import Path
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class ElastoplasticityPhysicsConfig(BasePhysicsConfig, name='elastoplasticity'): 
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'elastoplasticity.py')
