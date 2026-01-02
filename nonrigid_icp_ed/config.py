from omegaconf import OmegaConf
from dataclasses import dataclass

@dataclass
class KNNConfig:
    k: int = 1

@dataclass
class MinimizationConfig:
    learning_rate: float = 1.0
    max_iters: int = 20
    w_chamfer: float = 1.0
    w_landmark: float = 1.0
    w_arap: float = 1.0
    trunc_th: float = -1.0
    eps: float = 1e-7
    fix_anchors: bool = False


@dataclass
class NonrigidICPEDConfig:
    num_iterations: int = 20
    knn_conf: KNNConfig = KNNConfig()
    minimization_conf: MinimizationConfig = MinimizationConfig()