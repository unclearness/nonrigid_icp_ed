from dataclasses import dataclass, field
from typing import Literal
from omegaconf import OmegaConf


@dataclass
class GraphConfig:
    edges_k: int = 6
    num_nodes_per_point: int = 4
    weight_type: Literal["inv", "gaussian"] = "inv"
    sigma: float | None = None
    eps: float = 1e-8


@dataclass
class CorrespondenceConfig:
    k: int = 1


@dataclass
class MinimizationConfig:
    learning_rate: float = 0.001
    max_iters: int = 20
    w_chamfer: float = 1.0
    w_landmark: float = 1.0
    w_arap: float = 400.0
    trunc_th: float = -1.0
    eps: float = 1e-7
    fix_anchors: bool = True
    report_interval: int = 10


@dataclass
class NonrigidICPEDConfig:
    num_iterations: int = 20
    graph_conf: GraphConfig = field(default_factory=GraphConfig)
    correspondence_conf: CorrespondenceConfig = field(default_factory=CorrespondenceConfig)
    minimization_conf: MinimizationConfig = field(default_factory=MinimizationConfig)
