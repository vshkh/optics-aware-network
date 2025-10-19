# src/opticsnet/constraints/quantization.py
import torch
import torch.nn as nn

def quantize_param_inplace(param: torch.Tensor, bits: int) -> None:
    from .quantization import uniform_quantize  # or local import if in same file
    with torch.no_grad():
        param.copy_(uniform_quantize(param, bits))

def add_weight_quant_hooks(module: nn.Module, bits: int):
    """
    Quantizes weights *before each forward* to simulate low-bit DAC.
    Attach to Conv/Linear modules only.
    """
    def _hook(mod: nn.Module, _inp):
        if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
            quantize_param_inplace(mod.weight, bits)

    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.register_forward_pre_hook(_hook)

def uniform_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    if bits >= 16: return x
    qlevels = 2 ** bits
    x_clamped = x.clamp(-1, 1) # assume pre-normalized
    step = 2 / (qlevels - 1)
    return torch.round((x_clamped + 1) / step) * step - 1

class ActQuant:
    def __init__(self, bits: int = 8): self.bits = bits
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return uniform_quantize(x, self.bits)

class WeightQuant:
    def __init__(self, bits: int = 8): self.bits = bits
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return uniform_quantize(w, self.bits)
