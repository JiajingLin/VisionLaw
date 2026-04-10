from dataclasses import dataclass, field
from visionlaw.utils import Config
from .data import VideoDataConfig
from .camera import VideoCameraConfig

@dataclass(kw_only=True)
class VideoDatasetConfig(Config, name='videodataset'):
    eval: bool = False
    camera_type: str = 'NeuMASynthetic'
    device: str = 'gpu'
    data: VideoDataConfig = field(default_factory=VideoDataConfig)
    camera: VideoCameraConfig = field(default_factory=VideoCameraConfig)