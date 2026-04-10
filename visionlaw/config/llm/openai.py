from dataclasses import dataclass
from visionlaw.utils import Config
from .biphasebase import BiphaselBaseLLMConfig


@dataclass(kw_only=True)
class GPT4LLMConfig(BiphaselBaseLLMConfig, name='openai-gpt-4.1-mini'):
    api_key: str | None = None
    model: str = 'gpt-4.1-mini'
