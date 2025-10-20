import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    """
    Compact CNN that works for grayscale (MNIST/FashionMNIST) and RGB (CIFAR10) inputs.
    Uses global average pooling so feature size stays fixed regardless of image resolution.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.head = nn.Linear(64, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        pooled = feats.mean(dim=(-2, -1))
        return self.head(pooled)
