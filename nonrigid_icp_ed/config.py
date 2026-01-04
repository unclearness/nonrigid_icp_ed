from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from omegaconf import OmegaConf


# ----------------------------
# Configs
# ----------------------------
class WeightType(str, Enum):
    inv = "inv"
    gaussian = "gaussian"


@dataclass
class GraphConfig:
    edges_k: int = 6
    num_nodes_per_point: int = 4
    weight_type: WeightType = WeightType.inv
    sigma: float | None = None
    eps: float = 1e-8

    def __post_init__(self):
        if self.weight_type == WeightType.gaussian and self.sigma is None:
            raise ValueError("sigma must be set when weight_type == 'gaussian'")


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
    w_edge_length_uniform: float = 0.0
    w_normal_consistency: float = 0.0
    trunc_th: float = -1.0
    eps: float = 1e-7
    fix_anchors: bool = True
    report_interval: int = 10


@dataclass
class NonrigidIcpEdConfig:
    num_iterations: int = 20
    keep_history_on_memory: bool = True
    write_history_dir: str | Path | None = None
    global_deform : bool = True
    graph_conf: GraphConfig = field(default_factory=GraphConfig)
    correspondence_conf: CorrespondenceConfig = field(
        default_factory=CorrespondenceConfig
    )
    minimization_conf: MinimizationConfig = field(default_factory=MinimizationConfig)

    # --------
    # helpers
    # --------
    def resolve_paths(
        self, base_dir: str | Path | None = None
    ) -> "NonrigidIcpEdConfig":
        """Normalize path-like fields (str -> Path) and optionally resolve relative paths."""
        if self.write_history_dir is not None:
            p = Path(self.write_history_dir)
            if base_dir is not None and not p.is_absolute():
                p = Path(base_dir) / p
            self.write_history_dir = p
        return self

    # --------
    # file I/O
    # --------
    def save_yaml(self, path: str | Path) -> Path:
        """Save config as YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # OmegaConf converts dataclass -> DictConfig (nested supported)
        cfg = OmegaConf.structured(self)

        # Make YAML nice & stable
        OmegaConf.save(cfg, str(path), resolve=True)
        return path

    @classmethod
    def load_yaml(cls, path: str | Path) -> "NonrigidIcpEdConfig":
        """Load YAML into dataclass with schema validation."""
        path = Path(path)
        cfg = OmegaConf.load(str(path))

        # Merge loaded values onto schema (ensures unknown keys error if you want strict)
        schema = OmegaConf.structured(cls)
        merged = OmegaConf.merge(schema, cfg)

        # Convert to dataclass instance
        obj: NonrigidIcpEdConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

        # Normalize path-like fields relative to config file location
        obj.resolve_paths(base_dir=path.parent)
        return obj

    @classmethod
    def load_yaml_strict(cls, path: str | Path) -> "NonrigidIcpEdConfig":
        """
        Strict loader: errors on unknown keys / type mismatch.
        (OmegaConf is mostly strict when you use structured + set_struct)
        """
        path = Path(path)
        cfg = OmegaConf.load(str(path))

        schema = OmegaConf.structured(cls)
        OmegaConf.set_struct(schema, True)  # disallow new keys
        merged = OmegaConf.merge(schema, cfg)  # will raise if unknown keys exist

        obj: NonrigidIcpEdConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
        obj.resolve_paths(base_dir=path.parent)
        return obj


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    conf = NonrigidIcpEdConfig(write_history_dir="runs/exp01")
    conf.save_yaml("config.yaml")

    loaded = NonrigidIcpEdConfig.load_yaml("config.yaml")
    print(loaded)
    print(type(loaded.write_history_dir), loaded.write_history_dir)
