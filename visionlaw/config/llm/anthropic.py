from dataclasses import dataclass
from visionlaw.utils import Config

@dataclass(kw_only=True)
class Claude3SonnetLLMConfig(Config, name='anthropic-claude-3-sonnet-20240229'):
    api_key: str | None = None
    model: str = 'claude-3-sonnet-20240229'
