# src/opticsnet/utils/config.py
from dataclasses import dataclass

@dataclass
class NoiseCfg:  enabled: bool = True;  sigma: float = 0.01

@dataclass
class QuantCfg:  enabled: bool = False; bits: int = 8; per_channel: bool = False

@dataclass
class DriftCfg:  enabled: bool = False; rate: float = 1e-4

@dataclass
class ComplexCfg: enabled: bool = False

@dataclass
class OpticsCfg:
    noise: NoiseCfg = NoiseCfg()
    quant: QuantCfg = QuantCfg()
    drift: DriftCfg = DriftCfg()
    complex: ComplexCfg = ComplexCfg()
