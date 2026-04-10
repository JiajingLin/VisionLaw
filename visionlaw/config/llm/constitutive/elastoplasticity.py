from dataclasses import dataclass
from .base import BaseLLMConfig

@dataclass(kw_only=True)
class ElastoplasticityLLMConfig(BaseLLMConfig):
    entry: str = 'elastoplasticity'
    name: str = 'elastoplasticityllm_name'