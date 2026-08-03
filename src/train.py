"""Single entrypoint for every mia-train job. Launch with torchrun, never directly.

    torchrun --standalone --nproc_per_node=<gpus> src/train.py --config configs/<run>.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.run import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model described by a .toml config.")
    parser.add_argument(
        "--config", type=Path, required=True, help="path to the run's .toml configuration"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "directory under which this run's artifact directory is created; defaults to "
            "[environment].checkpoint_dir from configs/cluster/active.toml"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run(args.config, args.output_root)
    print(f"run artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
