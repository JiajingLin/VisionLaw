from dataclasses import dataclass
from visionlaw.utils import Config

@dataclass(kw_only=True)
class VideoCameraConfig(Config, name='videocamera'):
    resolution: int = 1
    data_device: str = 'gpu'