"""
Handling injections after each activation (at the moment)
"""

from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F
from .base_model import SimpleConvNet
from ..constraints.base import ConstraintPipeline
from ..constraints.quantization import ActQuant
from ..constraints.noise import add_gaussian_noise   # you added earlier
from ..utils.config import OpticsCfg

class OpticsAwareNet(nn.Module):
    def __init__(self, cfg: Optional[OpticsCfg] = None):
        super().__init__()
        self.cfg = cfg or OpticsCfg()
        self.backbone = SimpleConvNet()

        steps = []
        if self.cfg.noise.enabled:
            steps.append(lambda x: add_gaussian_noise(x, self.cfg.noise.sigma))
        if self.cfg.quant.enabled:
            steps.append(ActQuant(self.cfg.quant.bits))
        self.post_act = ConstraintPipeline(steps)

    def _apply_post_act(self, x: torch.Tensor) -> torch.Tensor:
        # no-ops if pipeline is empty
        return self.post_act(x) if self.post_act.steps else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.backbone.conv1(x))
        x = self._apply_post_act(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.backbone.conv2(x))
        x = self._apply_post_act(x)
        x = F.max_pool2d(x, 2)

        x = x.flatten(1)
        return self.backbone.head(x)
