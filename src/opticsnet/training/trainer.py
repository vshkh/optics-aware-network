from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import json
from datetime import datetime

def log_metrics(epoch: int, train_stats: dict, test_stats: dict, cfg: object, path: str = None):
    """
    Prints and optionally saves epoch metrics and constraint config.
    Args:
        epoch: current epoch number
        train_stats: {"loss": float, "acc": float}
        test_stats: {"loss": float, "acc": float}
        cfg: an OpticsCfg or similar dataclass
        path: optional file to append JSON logs
    """
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "epoch": epoch,
        "train": train_stats,
        "test": test_stats,
        "noise": vars(cfg.noise),
        "quant": vars(cfg.quant),
        "drift": vars(cfg.drift),
        "complex": vars(cfg.complex),
    }

    print(json.dumps(record, indent=2))
    if path:
        with open(path, "a") as f:
            json.dump(record, f)
            f.write("\n")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    
    return {"loss": running_loss / total, "acc": correct / total}

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return {"loss": running_loss / total, "acc": correct / total}
