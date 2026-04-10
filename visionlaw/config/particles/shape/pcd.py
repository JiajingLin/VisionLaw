from dataclasses import dataclass, field
from visionlaw.utils import Config


@dataclass(kw_only=True)
class PCDShapeConfig(Config, name='pcd'):
    pcd_name : str = None
    asset_root: str = None
    sort: str = None
    ori_bounds: list[list[float]] = field(default_factory=lambda: [[-1, -1, -1], [1, 1, 1]])
    sim_bounds: list[list[float]] = field(default_factory=lambda: [[-1, -1, -1], [1, 1, 1]])
    