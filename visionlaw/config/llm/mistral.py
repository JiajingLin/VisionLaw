from dataclasses import dataclass
from visionlaw.utils import Config

@dataclass(kw_only=True)
class Mixtral8x7BLLMConfig(Config, name='mistral-open-mixtral-8x7b'):
    api_key: str | None = None
    model: str = 'open-mixtral-8x7b'
