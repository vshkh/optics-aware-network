from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    preds = torch.as_tensor(preds).view(-1).cpu()
    targets = torch.as_tensor(targets).view(-1).cpu()
    if num_classes is None:
        if preds.numel() == 0 and targets.numel() == 0:
            raise ValueError("Cannot infer num_classes from empty predictions and targets.")
        max_pred = preds.max() if preds.numel() > 0 else torch.tensor(-1)
        max_target = targets.max() if targets.numel() > 0 else torch.tensor(-1)
        num_classes = int(torch.max(torch.stack([max_pred, max_target]))) + 1
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, t in zip(preds, targets):
        cm[t.item(), p.item()] += 1
    return cm.numpy()


def _ensure_list(value: Optional[Sequence[Any]], fallback: int) -> List[Any]:
    if value is None:
        return list(range(fallback))
    return list(value)


class ExperimentLogger:
    def __init__(
        self,
        output_dir: str | Path,
        experiment_name: str,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.metrics: List[Dict[str, Any]] = []
        self.sweeps: Dict[str, List[Dict[str, Any]]] = {}
        if config is not None:
            self._write_json("config.json", config)

    def log_epoch(
        self,
        epoch_idx: int,
        train_stats: Mapping[str, Any],
        eval_stats: Mapping[str, Any],
    ) -> None:
        self.metrics.append(
            {
                "epoch": epoch_idx,
                "train": dict(train_stats),
                "eval": dict(eval_stats),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def log_sweep_result(
        self,
        constraint_name: str,
        setting: Any,
        metrics: Mapping[str, Any],
    ) -> None:
        records = self.sweeps.setdefault(constraint_name, [])
        records.append(
            {
                "value": setting,
                **dict(metrics),
            }
        )

    def save_metrics(self, filename: str = "metrics.json") -> None:
        self._write_json(filename, {"experiment": self.experiment_name, "epochs": self.metrics})

    def plot_metrics(self, filename: str = "metrics.png") -> None:
        if not self.metrics:
            return
        epochs = [m["epoch"] for m in self.metrics]
        train_loss = [m["train"]["loss"] for m in self.metrics if "loss" in m["train"]]
        eval_loss = [m["eval"]["loss"] for m in self.metrics if "loss" in m["eval"]]
        train_acc = [m["train"].get("acc") for m in self.metrics]
        eval_acc = [m["eval"].get("acc") for m in self.metrics]
        train_acc = [np.nan if v is None else v for v in train_acc]
        eval_acc = [np.nan if v is None else v for v in eval_acc]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(epochs, train_loss, marker="o", label="train")
        axes[0].plot(epochs, eval_loss, marker="o", label="eval")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs, train_acc, marker="o", label="train")
        axes[1].plot(epochs, eval_acc, marker="o", label="eval")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        fig.tight_layout()
        self._save_figure(fig, filename)

    def save_confusion_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        *,
        num_classes: Optional[int] = None,
        class_labels: Optional[Sequence[str]] = None,
        filename: str = "confusion_matrix.png",
        normalize: bool = True,
    ) -> None:
        cm = compute_confusion_matrix(preds, targets, num_classes=num_classes)
        labels = _ensure_list(class_labels, cm.shape[0])
        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
            display = cm_norm
        else:
            display = cm

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(display, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = ".2f" if normalize else "d"
        thresh = display.max() / 2.0 if display.size else 0.0
        for i in range(display.shape[0]):
            for j in range(display.shape[1]):
                ax.text(
                    j,
                    i,
                    format(display[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if display[i, j] > thresh else "black",
                )

        fig.tight_layout()
        self._save_figure(fig, filename)
        self._write_json(
            filename.replace(".png", ".json"),
            {
                "matrix": cm.tolist(),
                "labels": labels,
                "normalized": normalize,
            },
        )

    def save_sweeps(self, filename: str = "sweeps.json") -> None:
        if not self.sweeps:
            return
        self._write_json(filename, self.sweeps)

    def plot_sweep_metric(
        self,
        constraint_name: str,
        metric_key: str,
        filename: Optional[str] = None,
    ) -> None:
        records = self.sweeps.get(constraint_name, [])
        if not records:
            return
        filename = filename or f"sweep_{constraint_name}_{metric_key}.png"
        x = [entry["value"] for entry in records]
        y = [entry.get(metric_key) for entry in records]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, marker="o")
        ax.set_title(f"{constraint_name} sweep: {metric_key}")
        ax.set_xlabel(constraint_name)
        ax.set_ylabel(metric_key)
        fig.tight_layout()
        self._save_figure(fig, filename)

    def _write_json(self, filename: str, data: Mapping[str, Any]) -> None:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


__all__ = ["ExperimentLogger", "compute_confusion_matrix"]
