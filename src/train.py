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
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_DIR",
        help=(
            "continue a previous run instead of starting fresh. Bare --resume takes the newest "
            "run of this experiment (and starts fresh if there is none), so the same submission "
            "script works for the first launch and every resubmission; pass a directory to "
            "continue that exact run"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Imported here rather than at module scope: it pulls in miao (~2s, via tensorstore and
    # zarr), which neither --help nor the argument-parsing tests should have to pay for.
    import components  # noqa: F401  (populates the registries; see its docstring)

    output_dir = run(args.config, args.output_root, args.resume)
    print(f"run artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
