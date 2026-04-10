from dataclasses import dataclass
from .base import BaseLLMConfig

@dataclass(kw_only=True)
class PlasticityLLMConfig(BaseLLMConfig):
    entry: str = 'plasticity'
    name: str = 'plasticityllm_name'