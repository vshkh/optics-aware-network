# src/opticsnet/utils/config.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, List, MutableMapping, Optional
import tomllib


@dataclass
class NoiseCfg:
    enabled: bool = True
    sigma: float = 0.01
    sigma_range: List[float] = field(default_factory=list)


@dataclass
class QuantCfg:
    enabled: bool = False
    bits: int = 8
    per_channel: bool = False
    bits_range: List[int] = field(default_factory=list)
    per_channel_options: List[bool] = field(default_factory=list)


@dataclass
class DriftCfg:
    enabled: bool = False
    sigma: float = 1e-4
    theta: float = 0.0
    sigma_range: List[float] = field(default_factory=list)
    theta_range: List[float] = field(default_factory=list)


@dataclass
class ComplexCfg:
    enabled: bool = False


@dataclass
class OpticsCfg:
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    quant: QuantCfg = field(default_factory=QuantCfg)
    drift: DriftCfg = field(default_factory=DriftCfg)
    complex: ComplexCfg = field(default_factory=ComplexCfg)


@dataclass
class DatasetCfg:
    name: str = "mnist"
    root: str = "./data"
    download: bool = True
    num_classes: Optional[int] = None


@dataclass
class TrainerCfg:
    epochs: int = 1
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 2
    learning_rate: float = 1e-3
    seed: Optional[int] = 42
    deterministic: bool = False


@dataclass
class LoggingCfg:
    output_dir: str = "./runs"
    save_curves: bool = True
    save_confusion: bool = True
    save_metrics: bool = True


def config_to_dict(cfg: ExperimentCfg) -> dict:
    return asdict(cfg)


@dataclass
class ExperimentCfg:
    name: str = "demo"
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    optics: OpticsCfg = field(default_factory=OpticsCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> "ExperimentCfg":
        cfg = cls()
        _merge_dataclass(cfg, data)
        return cfg


def _merge_dataclass(instance: Any, values: MutableMapping[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current):
            if isinstance(value, MutableMapping):
                _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_experiment_cfg(path: str | Path) -> ExperimentCfg:
    with Path(path).open("rb") as fh:
        data = tomllib.load(fh)
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Config at {path} must be a TOML table.")
    return ExperimentCfg.from_dict(data)


__all__ = [
    "NoiseCfg",
    "QuantCfg",
    "DriftCfg",
    "ComplexCfg",
    "OpticsCfg",
    "DatasetCfg",
    "TrainerCfg",
    "LoggingCfg",
    "ExperimentCfg",
    "config_to_dict",
    "load_experiment_cfg",
]
