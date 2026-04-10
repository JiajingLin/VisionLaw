from dataclasses import dataclass
from .base import BaseLLMConfig    

@dataclass(kw_only=True)
class ElasticityLLMConfig(BaseLLMConfig):
    entry: str = 'elasticity'
    name: str = 'elasticityllm_name'


