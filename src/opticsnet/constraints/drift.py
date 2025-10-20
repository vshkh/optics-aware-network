# src/opticsnet/constraints/drift.py
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

@torch.no_grad()
def apply_weight_drift_once(module: nn.Module, sigma: float = 1e-4, theta: float = 0.0):
    """
    Apply additive or OU-style drift to parameters of `module` in place, once.
    theta=0 -> pure random walk; theta>0 -> mean reversion (OU).
    """
    for p in module.parameters(recurse=False):
        if not p.requires_grad:
            continue
        if theta != 0.0:
            p.mul_(1.0 - theta)
        p.add_(torch.randn_like(p) * sigma)

def add_weight_drift_hooks(
    root: nn.Module,
    sigma: float = 1e-4,
    theta: float = 0.0,
    layer_filter: Optional[Callable[[nn.Module], bool]] = None,
) -> list[RemovableHandle]:
    """
    Register forward_pre_hooks on selected layers to drift weights before each forward.
    By default, applies to Conv2d/Linear modules.
    """
    def default_filter(m: nn.Module) -> bool:
        return isinstance(m, (nn.Conv2d, nn.Linear))

    lf = layer_filter or default_filter

    def _hook(mod: nn.Module, _inp):
        apply_weight_drift_once(mod, sigma=sigma, theta=theta)

    handles: list[RemovableHandle] = []
    for m in root.modules():
        if lf(m):
            handles.append(m.register_forward_pre_hook(_hook))
    return handles
