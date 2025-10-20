import argparse
from pathlib import Path

from src.opticsnet.analysis.runner import run_experiment
from src.opticsnet.utils.config import ExperimentCfg, load_experiment_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optics-aware network demo runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment TOML config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for the run directory suffix.",
    )
    return parser.parse_args()


def load_cfg(path: Path | None) -> ExperimentCfg:
    if path is None:
        return ExperimentCfg()
    return load_experiment_cfg(path)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    run_experiment(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
