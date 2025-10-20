import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from src.opticsnet.analysis.runner import run_experiment
from src.opticsnet.analysis.artifacts import ExperimentLogger
from src.opticsnet.utils.config import ExperimentCfg, config_to_dict, load_experiment_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated constraint sweeps.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mnist.toml"),
        help="Path to base experiment TOML config.",
    )
    return parser.parse_args()


def ensure_ranges(values: List) -> List:
    return list(values) if values else []


def disable_all_constraints(cfg: ExperimentCfg) -> None:
    cfg.optics.noise.enabled = False
    cfg.optics.quant.enabled = False
    cfg.optics.drift.enabled = False


def run_noise_sweep(base_cfg: ExperimentCfg, root_dir: Path, aggregator: ExperimentLogger) -> None:
    sigma_values = ensure_ranges(base_cfg.optics.noise.sigma_range)
    if not sigma_values:
        return

    for sigma in sigma_values:
        cfg = copy.deepcopy(base_cfg)
        disable_all_constraints(cfg)
        cfg.optics.noise.enabled = True
        cfg.optics.noise.sigma = sigma
        result = run_experiment(
            cfg,
            output_root=root_dir,
            run_name=f"noise_sigma_{sigma}",
        )
        aggregator.log_sweep_result(
            "noise_sigma",
            sigma,
            {
                "eval_loss": result["eval_stats"].get("loss"),
                "eval_acc": result["eval_stats"].get("acc"),
                "train_loss": result["train_stats"].get("loss"),
                "train_acc": result["train_stats"].get("acc"),
            },
        )


def run_quant_bits_sweep(base_cfg: ExperimentCfg, root_dir: Path, aggregator: ExperimentLogger) -> None:
    bit_values = ensure_ranges(base_cfg.optics.quant.bits_range)
    if not bit_values:
        return

    per_channel = base_cfg.optics.quant.per_channel
    for bits in bit_values:
        cfg = copy.deepcopy(base_cfg)
        disable_all_constraints(cfg)
        cfg.optics.quant.enabled = True
        cfg.optics.quant.bits = bits
        cfg.optics.quant.per_channel = per_channel
        result = run_experiment(
            cfg,
            output_root=root_dir,
            run_name=f"quant_bits_{bits}",
        )
        aggregator.log_sweep_result(
            "quant_bits",
            bits,
            {
                "eval_loss": result["eval_stats"].get("loss"),
                "eval_acc": result["eval_stats"].get("acc"),
                "train_loss": result["train_stats"].get("loss"),
                "train_acc": result["train_stats"].get("acc"),
            },
        )


def run_quant_per_channel_sweep(base_cfg: ExperimentCfg, root_dir: Path, aggregator: ExperimentLogger) -> None:
    options = ensure_ranges(base_cfg.optics.quant.per_channel_options)
    if not options:
        return

    bits = base_cfg.optics.quant.bits
    for flag in options:
        cfg = copy.deepcopy(base_cfg)
        disable_all_constraints(cfg)
        cfg.optics.quant.enabled = True
        cfg.optics.quant.per_channel = bool(flag)
        cfg.optics.quant.bits = bits
        suffix = "pc_on" if cfg.optics.quant.per_channel else "pc_off"
        result = run_experiment(
            cfg,
            output_root=root_dir,
            run_name=f"quant_{suffix}",
        )
        aggregator.log_sweep_result(
            "quant_per_channel",
            bool(flag),
            {
                "eval_loss": result["eval_stats"].get("loss"),
                "eval_acc": result["eval_stats"].get("acc"),
                "train_loss": result["train_stats"].get("loss"),
                "train_acc": result["train_stats"].get("acc"),
            },
        )


def run_drift_sigma_sweep(base_cfg: ExperimentCfg, root_dir: Path, aggregator: ExperimentLogger) -> None:
    sigma_values = ensure_ranges(base_cfg.optics.drift.sigma_range)
    if not sigma_values:
        return

    theta = base_cfg.optics.drift.theta
    for sigma in sigma_values:
        cfg = copy.deepcopy(base_cfg)
        disable_all_constraints(cfg)
        cfg.optics.drift.enabled = True
        cfg.optics.drift.sigma = sigma
        cfg.optics.drift.theta = theta
        result = run_experiment(
            cfg,
            output_root=root_dir,
            run_name=f"drift_sigma_{sigma}",
        )
        aggregator.log_sweep_result(
            "drift_sigma",
            sigma,
            {
                "eval_loss": result["eval_stats"].get("loss"),
                "eval_acc": result["eval_stats"].get("acc"),
                "train_loss": result["train_stats"].get("loss"),
                "train_acc": result["train_stats"].get("acc"),
            },
        )


def run_drift_theta_sweep(base_cfg: ExperimentCfg, root_dir: Path, aggregator: ExperimentLogger) -> None:
    theta_values = ensure_ranges(base_cfg.optics.drift.theta_range)
    if not theta_values:
        return

    sigma = base_cfg.optics.drift.sigma
    for theta in theta_values:
        cfg = copy.deepcopy(base_cfg)
        disable_all_constraints(cfg)
        cfg.optics.drift.enabled = True
        cfg.optics.drift.theta = theta
        cfg.optics.drift.sigma = sigma
        result = run_experiment(
            cfg,
            output_root=root_dir,
            run_name=f"drift_theta_{theta}",
        )
        aggregator.log_sweep_result(
            "drift_theta",
            theta,
            {
                "eval_loss": result["eval_stats"].get("loss"),
                "eval_acc": result["eval_stats"].get("acc"),
                "train_loss": result["train_stats"].get("loss"),
                "train_acc": result["train_stats"].get("acc"),
            },
        )


def run_sweeps(cfg: ExperimentCfg) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path(cfg.logging.output_dir) / "sweeps" / f"{cfg.name}_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)
    aggregator = ExperimentLogger(base_dir, f"{cfg.name}_sweeps", config=config_to_dict(cfg))

    run_noise_sweep(cfg, base_dir, aggregator)
    run_quant_bits_sweep(cfg, base_dir, aggregator)
    run_quant_per_channel_sweep(cfg, base_dir, aggregator)
    run_drift_sigma_sweep(cfg, base_dir, aggregator)
    run_drift_theta_sweep(cfg, base_dir, aggregator)

    aggregator.save_sweeps()
    for constraint in list(aggregator.sweeps.keys()):
        aggregator.plot_sweep_metric(constraint, "eval_loss")
        aggregator.plot_sweep_metric(constraint, "eval_acc")
        aggregator.plot_sweep_metric(constraint, "train_loss")
        aggregator.plot_sweep_metric(constraint, "train_acc")


def main() -> None:
    args = parse_args()
    cfg = load_experiment_cfg(args.config)
    run_sweeps(cfg)


if __name__ == "__main__":
    main()
