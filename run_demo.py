import argparse
from datetime import datetime
from pathlib import Path

import torch
from functools import partial
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from src.opticsnet.analysis.artifacts import ExperimentLogger
from src.opticsnet.utils.config import config_to_dict, load_experiment_cfg
from src.opticsnet.utils.seed import set_seed
from src.opticsnet.utils.device import get_device
from src.opticsnet.models.base_model import SimpleConvNet
from src.opticsnet.training.trainer import train_one_epoch, evaluate
from src.opticsnet.constraints.base import ConstraintPipeline
from src.opticsnet.constraints.noise import add_gaussian_noise
from src.opticsnet.constraints.quantization import uniform_quantize
from src.opticsnet.constraints.drift import add_weight_drift_hooks
from src.opticsnet.utils.config import ExperimentCfg




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
    *,
    use_noise: bool,
    noise_sigma: float,
    use_quant: bool,
    quant_bits: int,
    quant_per_channel: bool,
) -> ConstraintPipeline:
    steps = []
    if use_noise:
        steps.append(partial(add_gaussian_noise, sigma=noise_sigma))
    if use_quant:
        steps.append(
            partial(
                uniform_quantize,
                bits=quant_bits,
                per_channel=quant_per_channel,
                channel_dim=1,
            )
        )
    return ConstraintPipeline(steps=steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optics-aware network demo runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment TOML config.",
    )
    return parser.parse_args()


def load_cfg(path: Path | None) -> ExperimentCfg:
    if path is None:
        return ExperimentCfg()
    return load_experiment_cfg(path)


def prepare_data_loaders(
    cfg: ExperimentCfg,
    *,
    collate_fn,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor()])

    if cfg.dataset.name.lower() != "mnist":
        raise ValueError(f"Unsupported dataset '{cfg.dataset.name}'. Only 'mnist' is available in run_demo.")

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


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    if cfg.trainer.seed is not None:
        set_seed(cfg.trainer.seed, deterministic=cfg.trainer.deterministic)
    else:
        set_seed(deterministic=cfg.trainer.deterministic)

    device = get_device()
    pin_memory = device.type == "cuda"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.logging.output_dir) / f"{cfg.name}_{timestamp}"
    logger = ExperimentLogger(run_dir, cfg.name, config=config_to_dict(cfg))

    print(
        "Device:",
        device,
        "| CUDA name:",
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    )
    print(f"Artifacts will be saved to: {run_dir}")

    input_pipeline = build_input_pipeline(
        use_noise=cfg.optics.noise.enabled,
        noise_sigma=cfg.optics.noise.sigma,
        use_quant=cfg.optics.quant.enabled,
        quant_bits=cfg.optics.quant.bits,
        quant_per_channel=cfg.optics.quant.per_channel,
    )
    collate = partial(collate_apply_constraints, pipeline=input_pipeline)

    train_loader, test_loader = prepare_data_loaders(
        cfg,
        collate_fn=collate,
        pin_memory=pin_memory,
    )

    model = SimpleConvNet().to(device)
    drift_handles = []
    if cfg.optics.drift.enabled:
        drift_handles = add_weight_drift_hooks(
            model,
            sigma=cfg.optics.drift.sigma,
            theta=cfg.optics.drift.theta,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.learning_rate)

    collect_outputs = cfg.logging.save_confusion
    last_preds = None
    last_targets = None

    for epoch in range(cfg.trainer.epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        eval_result = evaluate(
            model,
            test_loader,
            criterion,
            device,
            return_outputs=collect_outputs,
        )
        if collect_outputs:
            eval_stats, preds, targets = eval_result
            last_preds, last_targets = preds, targets
        else:
            eval_stats = eval_result  # type: ignore[assignment]

        logger.log_epoch(epoch + 1, train_stats, eval_stats)
        print(f"[epoch {epoch + 1}/{cfg.trainer.epochs}] train={train_stats} test={eval_stats}")

    for handle in drift_handles:
        handle.remove()

    if cfg.logging.save_metrics:
        logger.save_metrics()
    if cfg.logging.save_curves:
        logger.plot_metrics()
    if cfg.logging.save_confusion and last_preds is not None and last_targets is not None:
        logger.save_confusion_matrix(last_preds, last_targets)
    logger.save_sweeps()


if __name__ == "__main__":
    main()
