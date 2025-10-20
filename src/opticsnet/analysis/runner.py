from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from functools import partial
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from src.opticsnet.analysis.artifacts import ExperimentLogger
from src.opticsnet.utils.config import ExperimentCfg, config_to_dict
from src.opticsnet.utils.device import get_device
from src.opticsnet.utils.seed import set_seed
from src.opticsnet.models.base_model import SimpleConvNet
from src.opticsnet.training.trainer import train_one_epoch, evaluate
from src.opticsnet.constraints.base import ConstraintPipeline
from src.opticsnet.constraints.noise import add_gaussian_noise
from src.opticsnet.constraints.quantization import uniform_quantize
from src.opticsnet.constraints.drift import add_weight_drift_hooks


def collate_apply_constraints(
    batch,
    *,
    pipeline: ConstraintPipeline | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys)
    if pipeline is not None and pipeline.steps:
        x = pipeline(x)
    return x, y


def build_input_pipeline(
    cfg: ExperimentCfg,
) -> ConstraintPipeline:
    steps = []
    if cfg.optics.noise.enabled:
        steps.append(partial(add_gaussian_noise, sigma=cfg.optics.noise.sigma))
    if cfg.optics.quant.enabled:
        steps.append(
            partial(
                uniform_quantize,
                bits=cfg.optics.quant.bits,
                per_channel=cfg.optics.quant.per_channel,
                channel_dim=1,
            )
        )
    return ConstraintPipeline(steps=steps)


def prepare_data_loaders(
    cfg: ExperimentCfg,
    *,
    collate_fn,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor()])
    ds_name = cfg.dataset.name.lower()
    if ds_name != "mnist":
        raise ValueError(f"Unsupported dataset '{cfg.dataset.name}'. Only 'mnist' is available for the demo runner.")

    train_ds = datasets.MNIST(
        root=cfg.dataset.root,
        train=True,
        download=cfg.dataset.download,
        transform=tfm,
    )
    test_ds = datasets.MNIST(
        root=cfg.dataset.root,
        train=False,
        download=cfg.dataset.download,
        transform=tfm,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.trainer.eval_batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader


def run_experiment(
    cfg: ExperimentCfg,
    *,
    output_root: Optional[Path] = None,
    run_name: Optional[str] = None,
    collect_confusion: Optional[bool] = None,
) -> dict:
    cfg_local = copy.deepcopy(cfg)
    if cfg_local.trainer.seed is not None:
        set_seed(cfg_local.trainer.seed, deterministic=cfg_local.trainer.deterministic)
    else:
        set_seed(deterministic=cfg_local.trainer.deterministic)

    device = get_device()
    pin_memory = device.type == "cuda"

    output_root = output_root or Path(cfg_local.logging.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = run_name or cfg_local.name
    run_dir = output_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(run_dir, run_name, config=config_to_dict(cfg_local))

    print(
        f"[{run_name}] Device: {device} | CUDA name: "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
    )
    print(f"[{run_name}] Artifacts -> {run_dir}")

    input_pipeline = build_input_pipeline(cfg_local)
    collate = partial(collate_apply_constraints, pipeline=input_pipeline)
    train_loader, test_loader = prepare_data_loaders(cfg_local, collate_fn=collate, pin_memory=pin_memory)

    model = SimpleConvNet().to(device)
    drift_handles = []
    if cfg_local.optics.drift.enabled:
        drift_handles = add_weight_drift_hooks(
            model,
            sigma=cfg_local.optics.drift.sigma,
            theta=cfg_local.optics.drift.theta,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg_local.trainer.learning_rate)

    collect_outputs = collect_confusion if collect_confusion is not None else cfg_local.logging.save_confusion
    last_preds = None
    last_targets = None
    last_train_stats = None
    last_eval_stats = None

    for epoch in range(cfg_local.trainer.epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        is_last = (epoch == cfg_local.trainer.epochs - 1)
        eval_kwargs = {"return_outputs": collect_outputs and is_last}
        eval_result = evaluate(model, test_loader, criterion, device, **eval_kwargs)
        if eval_kwargs["return_outputs"]:
            eval_stats, preds, targets = eval_result
            last_preds, last_targets = preds, targets
        else:
            eval_stats = eval_result  # type: ignore[assignment]

        last_train_stats = train_stats
        last_eval_stats = eval_stats
        logger.log_epoch(epoch + 1, train_stats, eval_stats)
        print(
            f"[{run_name}] epoch {epoch + 1}/{cfg_local.trainer.epochs} "
            f"train={train_stats} test={eval_stats}"
        )

    for handle in drift_handles:
        handle.remove()

    if cfg_local.logging.save_metrics:
        logger.save_metrics()
    if cfg_local.logging.save_curves:
        logger.plot_metrics()
    if cfg_local.logging.save_confusion and last_preds is not None and last_targets is not None:
        logger.save_confusion_matrix(last_preds, last_targets)

    logger.save_sweeps()

    return {
        "run_dir": run_dir,
        "train_stats": last_train_stats or {},
        "eval_stats": last_eval_stats or {},
    }


__all__ = [
    "run_experiment",
    "build_input_pipeline",
    "collate_apply_constraints",
    "prepare_data_loaders",
]
