# src/opticsnet/constraints/quantization.py
from typing import Optional, Tuple
import torch
import torch.nn as nn


def _compute_dynamic_range(
    x: torch.Tensor,
    *,
    per_channel: bool,
    channel_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not per_channel:
        return x.amin(), x.amax()

    if channel_dim < 0:
        channel_dim = x.ndim + channel_dim
    if channel_dim >= x.ndim:
        raise ValueError(f"channel_dim={channel_dim} out of bounds for tensor with {x.ndim} dims")
    if x.ndim == 1:
        return x.amin(), x.amax()
    reduce_dims = tuple(d for d in range(x.ndim) if d != channel_dim)
    return x.amin(dim=reduce_dims, keepdim=True), x.amax(dim=reduce_dims, keepdim=True)


def uniform_quantize(
    x: torch.Tensor,
    bits: int = 8,
    *,
    min_val: Optional[torch.Tensor] = None,
    max_val: Optional[torch.Tensor] = None,
    per_channel: bool = False,
    channel_dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Uniform quantization with optional dynamic range estimation and per-channel support.
    Args:
        x: tensor to quantize.
        bits: number of quantization bits. Must be >= 1.
        min_val/max_val: optional precomputed range. If omitted, inferred from `x`.
        per_channel: if True, infer range per `channel_dim`.
        channel_dim: dimension treated as the channel axis when `per_channel` is True.
        eps: numerical stability epsilon for scale computation.
    """
    if bits < 1:
        raise ValueError(f"bits must be >=1, got {bits}")
    if bits >= 16:
        return x

    if min_val is None or max_val is None:
        min_val, max_val = _compute_dynamic_range(x, per_channel=per_channel, channel_dim=channel_dim)

    qlevels = 2 ** bits
    range_tensor = (max_val - min_val).clamp_min(eps)
    scale = range_tensor / (qlevels - 1)
    zero_point = torch.round(-min_val / scale)

    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, 0, qlevels - 1)
    return (q - zero_point) * scale


class WeightQuant:
    def __init__(self, bits: int = 8, *, per_channel: bool = False, channel_dim: int = 0):
        self.bits = bits
        self.per_channel = per_channel
        self.channel_dim = channel_dim

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return uniform_quantize(
            w,
            self.bits,
            per_channel=self.per_channel,
            channel_dim=self.channel_dim,
        )
