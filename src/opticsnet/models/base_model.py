from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    """
    A tiny CNN with clean hooks for constraints (noise, quant, drift later).
    Shape: (N, 1, 28, 28) -> logits (N, 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.head  = nn.Linear(32 * 7 * 7, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))   # (N,16,28,28)
        x = F.max_pool2d(x, 2)      # (N,16,14,14)
        x = F.relu(self.conv2(x))   # (N,32,14,14)
        x = F.max_pool2d(x, 2)      # (N,32,7,7)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        flat  = feats.flatten(1)
        return self.head(flat)
