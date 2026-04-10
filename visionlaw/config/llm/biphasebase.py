from dataclasses import dataclass, field
from visionlaw.utils import Config
from .constitutive import ElasticityLLMConfig, PlasticityLLMConfig, ElastoplasticityLLMConfig

@dataclass(kw_only=True)
class BiphaselBaseLLMConfig(Config):
    api_key: str
    model: str
    primitives: tuple[str] = ('elastoplasticity',)  
    
    elasticity: ElasticityLLMConfig = field(default_factory=ElasticityLLMConfig)
    plasticity: PlasticityLLMConfig = field(default_factory=PlasticityLLMConfig)
    elastoplasticity: ElastoplasticityLLMConfig = field(default_factory=ElastoplasticityLLMConfig)
    
    def __post_init__(self):
        print("Running BiphaselBaseLLMConfig.__post_init__")
        self.elasticity = ElasticityLLMConfig(api_key=self.api_key, model=self.model, name=self.name)
        self.plasticity = PlasticityLLMConfig(api_key=self.api_key, model=self.model, name=self.name)
        self.elastoplasticity = ElastoplasticityLLMConfig(api_key=self.api_key, model=self.model, name=self.name)
