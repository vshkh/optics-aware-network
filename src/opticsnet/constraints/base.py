# src/opticsnet/constraints/base.py

"""
Purpose: create an interface to keep the forward pass relatively simple; be able to add constraints with ease:
"""

from typing import Callable, Iterable, List
import torch

ConstraintFn = Callable[[torch.Tensor], torch.Tensor]

class ConstraintPipeline:
    def __init__(self, steps: Iterable[ConstraintFn] = ()):
        self.steps: List[ConstraintFn] = list(steps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for fn in self.steps:
            x = fn(x)
        return x

    def add(self, fn: ConstraintFn) -> None:
        self.steps.append(fn)
