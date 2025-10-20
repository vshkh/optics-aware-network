from typing import Dict, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import json
from datetime import datetime

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
    device: torch.device,
    *,
    return_outputs: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]:
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    preds_list = []
    targets_list = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        if return_outputs:
            preds_list.append(logits.argmax(1).detach().cpu())
            targets_list.append(y.detach().cpu())

    metrics = {"loss": running_loss / total, "acc": correct / total}
    if not return_outputs:
        return metrics
    preds_tensor = torch.cat(preds_list) if preds_list else torch.empty(0, dtype=torch.long)
    targets_tensor = torch.cat(targets_list) if targets_list else torch.empty(0, dtype=torch.long)
    return metrics, preds_tensor, targets_tensor
