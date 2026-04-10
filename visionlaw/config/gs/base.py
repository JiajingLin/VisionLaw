from dataclasses import dataclass, field
from visionlaw.utils import Config


@dataclass(kw_only=True)
class GaussiansConfig(Config, name='gs'):
    sh_degree: int = 3
    opacity_thres: float = 0.02
    kernels_path: str = '/path/to/point_cloud.ply'    # Replace this with the path to the reconstructed Gaussian kernels
    scaling_modifier: float = 1.0