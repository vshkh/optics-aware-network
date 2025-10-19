from typing import Optional
import torch

@torch.no_grad()
def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.01, inplace: bool = False, seed: Optional[int] = None) -> torch.Tensor:
    """
    Add zero-mean Gaussian noise to a tensor.
    Args:
        x: input tensor (on CPU or GPU).
        sigma: stddev of noise (relative magnitude).
        inplace: if True, modifies x in place.
        seed: optional seed for reproducible noise.
    """
    if seed is not None:
        g = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.normal(mean=0.0, std=sigma, size=x.shape, generator=g, device=x.device, dtype=x.dtype)
    else:
        noise = torch.randn_like(x) * sigma
    if inplace:
        x.add_(noise)
        return x
    return x + noise
