"""Run a trained checkpoint over a whole OME-NGFF volume and write the prediction as an artifact.

    python src/predict.py <run_dir> --data-config <miao.yaml> --volume <name> --out <dir> [--step N]

Writes two artifacts per volume, on one shared grid:

    <dir>/<volume>.zarr        the prediction, kind from the algorithm
    <dir>/<volume>.gt.zarr     the co-registered ground truth, kind="instances"

**Nothing about the run is assumed.** Patch size, channel count, what the channels *mean*, how they
are squashed for storage, and the spatial rank all come from the run's own resolved config and from
the algorithm object rebuilt from it. A model trained at patch 128, an algorithm emitting 64 class
scores instead of 6 affinities, a non-cubic patch -- none of those need a change here. The one thing
that is checked rather than adopted is the patch size: the data config and the model must agree,
because RoPE normalises coordinates by the *runtime* grid extent, so predicting at a patch size the
encoder was not trained at silently changes every position it sees.

**The geometry lives in `prediction/`.** `prediction.grid.VolumeGrid` builds the aligned tile
lattice for a volume and reads its tiles and ground truth through miao -- the rationale for the
lattice is there. `prediction.dense` is the blended default for fixed-channel outputs, and
`prediction.types` the contract a strategy's own predictor satisfies. What is left here is the
part that knows about *runs*: rebuilding the algorithm from `resolved_config.json` and its
checkpoint, patching it with `--override`, choosing the patch size, and writing both artifacts with
the provenance needed to reproduce them.

**Why the ground truth is written here too.** It is read by the same `VolumeGrid` the prediction
was made on, so the two artifacts share a lattice by construction; a scorer that resampled labels
onto a prediction's grid itself would be a second transform chain with a second chance to be wrong.
Every input to the chain is recorded in the artifact's attrs, so the result stays auditable.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from prediction.dense import select_predictor
from prediction.grid import VolumeGrid, storage_axes_of
from prediction.types import VolumePredictor


def assert_checkpoint_is_fully_consumed(algorithm: Any, checkpoint_dir: Path) -> None:
    """Fail if the checkpoint holds model tensors this rebuild has nowhere to put.

    **DCP loads *into* a state dict, and skips whatever the template does not ask for, silently.**
    That makes an incomplete rebuild the worst kind of bug here: a model reconstructed without its
    LoRA adapter loads every base weight, ignores every `lora_a`/`lora_b`, and predicts with the
    *un-adapted* encoder -- which scores near the released-checkpoint baseline it started from. A
    plausible number, attributed to the wrong model, and nothing anywhere says so.

    Compared against `get_model_state_dict`, the same function `CheckpointManager` saves through,
    rather than against `named_parameters()`: an algorithm may hold one module under two names
    (`affinity_seg` exposes its encoder as both `model` and `encoder`), `named_parameters()`
    deduplicates by tensor identity and would report only one of them, and the other would then look
    orphaned. Only the `model.` half of the checkpoint is checked -- `optim.` and `train_state.` are
    not rebuilt here and are not meant to be.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    stored = set(FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata)
    have = {f"model.{key}" for key in get_model_state_dict(algorithm)}
    orphaned = sorted(key for key in stored if key.startswith("model.") and key not in have)
    if not orphaned:
        return

    hint = ""
    if any(".lora_" in key for key in orphaned):
        hint = (
            "\nThese are LoRA adapter tensors. The run trained an adapted encoder, so its "
            "resolved_config.json must carry a [lora] section for this rebuild to reproduce it. If "
            "the section is present and this still fires, the model and the checkpoint disagree "
            "about which projections were adapted."
        )
    raise SystemExit(
        f"{len(orphaned)} tensor(s) in {checkpoint_dir} have no slot in the rebuilt model, so DCP "
        f"would load the rest and ignore these without a word: {orphaned[:6]}"
        f"{' ...' if len(orphaned) > 6 else ''}{hint}"
    )


OVERRIDABLE_SECTIONS = ("algorithm", "model")


def _accepted_keys(section: str, name: str) -> set[str]:
    """Constructor keywords the registered `section` implementation named `name` takes.

    Checked against the class rather than only against the run's recorded kwargs, because a knob
    added to a strategy *after* a run was trained is exactly the case `--override` exists for --
    mask-generation thresholds on a checkpoint that predates them -- and the recorded kwargs cannot
    know about it. The class signature can. A typo still fails, against the real menu.
    """
    import inspect

    import components  # noqa: F401  (populates the registries)
    from algorithms.registry import AlgorithmRegistry
    from models.registry import ModelRegistry

    registry = AlgorithmRegistry if section == "algorithm" else ModelRegistry
    parameters = inspect.signature(registry.get(name).__init__).parameters
    return {k for k in parameters if k not in ("self", "model", "dataset", "args", "kwargs")}


def apply_overrides(resolved: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """`section.key=value` settings patched into a run's resolved record -> the patched record.

    For knobs that are legitimately decided *after* training. The clearest case is a promptable
    model's mask-generation thresholds -- how confident a mask must be to keep, how much two must
    overlap to be one -- which have to be tuned against a score on a held-out volume and would
    otherwise need a retrain per value. Restricted to `[algorithm]` and `[model]`: `[data]` is
    supplied by `--data-config`, and nothing under `[trainer]` bears on inference.

    Values are parsed as TOML, so `0.7`, `true`, `"whole"` and `[1, 2]` all mean what they would in
    the config file that set them originally. A key the run never used is refused rather than
    silently added, since a typo would otherwise read as a setting that took effect.
    """
    patched = json.loads(json.dumps(resolved))  # a copy; the caller's record stays as written
    for override in overrides:
        target, separator, literal = override.partition("=")
        section, dot, key = target.partition(".")
        if not separator or not dot or not key:
            raise SystemExit(
                f"--override {override!r}: expected section.key=value, e.g. "
                "algorithm.pred_iou_thresh=0.7"
            )
        if section not in OVERRIDABLE_SECTIONS:
            raise SystemExit(
                f"--override {override!r}: only {OVERRIDABLE_SECTIONS} may be overridden at "
                "prediction time"
            )
        kwargs = patched[section]["kwargs"]
        accepted = _accepted_keys(section, patched[section]["name"])
        if key not in kwargs and key not in accepted:
            raise SystemExit(
                f"--override {override!r}: {patched[section]['name']!r} accepts no {key!r}; the "
                f"run set {sorted(kwargs)} and the class also takes "
                f"{sorted(accepted - set(kwargs))}"
            )
        try:
            kwargs[key] = tomllib.loads(f"value = {literal}")["value"]
        except tomllib.TOMLDecodeError as error:
            raise SystemExit(
                f"--override {override!r}: {literal!r} is not a TOML value ({error}); quote "
                'strings, e.g. algorithm.prefer="whole"'
            ) from None
    return patched


def load_algorithm(
    run_dir: Path,
    device: torch.device,
    step: int | None = None,
    overrides: list[str] | None = None,
) -> tuple[Any, int, dict[str, Any]]:
    """Rebuild the trained algorithm from a run directory. Returns (algorithm, step, resolved).

    `resolved_config.json` records each section as {"name": ..., "kwargs": {...}}, which is exactly
    what the registries take -- so the model is rebuilt from what the run really used rather than
    from a config file that may since have moved on. The resolved settings are returned too, because
    a caller needs them to know what the run's patch size was. `overrides` patches that record
    first; see `apply_overrides`.
    """
    import components  # noqa: F401  (populates the registries)
    from algorithms.registry import AlgorithmRegistry
    from engine.checkpoint import CheckpointManager
    from engine.config import LoRAConfig
    from engine.lora import apply_lora
    from models.registry import ModelRegistry

    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    if overrides:
        resolved = apply_overrides(resolved, overrides)
    model_cfg, algo_cfg, data_cfg = resolved["model"], resolved["algorithm"], resolved["data"]
    model = ModelRegistry.build(model_cfg["name"], **model_cfg["kwargs"])

    # Low-rank adaptation, in the same position `engine.run.build_trainer` applies it: on the bare
    # model, before the algorithm wraps it. `[lora]` is a top-level section rather than part of
    # `[model]`, so rebuilding from `model_cfg` alone produces a *plain* encoder -- see
    # `assert_checkpoint_is_fully_consumed` for what that costs. Absent from every run predating the
    # feature, hence the default: `LoRAConfig()` has rank 0 and is disabled.
    lora_cfg = LoRAConfig(**resolved.get("lora", {}))
    if lora_cfg.enabled():
        print(f"[lora] {apply_lora(model, lora_cfg).summary()}", flush=True)

    algorithm = AlgorithmRegistry.build(
        algo_cfg["name"], model, None,
        input_axes=data_cfg["kwargs"]["output_axes"],
        **algo_cfg["kwargs"],
    )
    algorithm.to(device).eval()

    optimizer = torch.optim.AdamW(algorithm.parameters(), lr=1e-4)
    manager = CheckpointManager(algorithm, optimizer, run_dir / "checkpoints")

    # Checked before loading, so a mismatch costs a second rather than a long job whose output is
    # quietly wrong. Only when the directory is actually there: a missing checkpoint is reported
    # below, and `load_step` names the steps that *do* exist, which is the more useful message.
    path = manager.latest_checkpoint() if step is None else run_dir / "checkpoints" / f"step_{step}"
    if path is not None and path.is_dir():
        assert_checkpoint_is_fully_consumed(algorithm, path)

    loaded = manager.load_latest() if step is None else manager.load_step(step)
    if loaded == 0:
        raise SystemExit(f"no checkpoint found under {run_dir / 'checkpoints'}")
    print(f"loaded step {loaded} from {run_dir.name}", flush=True)
    return algorithm, loaded, resolved


def resolve_patch(
    config: Any, resolved: dict[str, Any], volume_name: str, override: int | None
) -> list[int]:
    """The patch size to predict at, per axis in storage order.

    Taken from the data config, which states it per axis, and cross-checked against the model's own
    `img_size`. They must agree: RoPE normalises coordinates by the *runtime* grid extent, so an
    encoder fed a patch size other than the one it trained at sees every position shifted, silently
    and without any error. `--patch` overrides both, for deliberately probing that.
    """
    output_axes = "".join(axis for axis in config.output_axes if axis in "xyz")
    if len(config.patch_size) != len(output_axes):
        raise SystemExit(
            f"the data config's patch_size {list(config.patch_size)} has "
            f"{len(config.patch_size)} entries but its output_axes imply "
            f"{len(output_axes)} spatial axes ({output_axes})"
        )
    storage_axes = storage_axes_of(config, volume_name)
    by_axis = dict(zip(output_axes, (int(p) for p in config.patch_size), strict=True))
    patch = [by_axis[axis] for axis in storage_axes]

    if override is not None:
        print(f"[patch] overriding {patch} with {override} on every axis", flush=True)
        return [override] * len(storage_axes)

    img_size = resolved.get("model", {}).get("kwargs", {}).get("img_size")
    if img_size is not None:
        trained = (
            [int(img_size)] * len(storage_axes)
            if isinstance(img_size, (int, float))
            else [int(v) for v in img_size]
        )
        if sorted(trained) != sorted(patch):
            raise SystemExit(
                f"the data config asks for patch {patch} but the model was built with img_size "
                f"{img_size}. Predicting at a different patch size than the encoder trained at "
                "silently moves every position it sees, because RoPE normalises coordinates by the "
                "runtime grid extent. Fix the config, or pass --patch to override deliberately."
            )
    return patch


def shared_attrs(
    grid: VolumeGrid, run_dir: Path, step: int, data_config: Path
) -> dict[str, Any]:
    """Everything needed to reproduce this grid, on both artifacts so neither can drift."""
    return {
        "origin": [0] * grid.rank,
        "axes": grid.axes,
        "source_path": str(grid.volume.path),
        "source_image_key": grid.volume.image_key,
        "source_label_key": grid.volume.label_key,
        "image_level": grid.image_level,
        "label_level": grid.label_level,
        "native_box": grid.native_box(),
        "annotated_box": [
            [low, low + extent]
            for low, extent in zip(grid.box_low, grid.box_extent, strict=True)
        ],
        "read_shape": list(grid.read),
        "scale": [p / r for p, r in zip(grid.patch, grid.read, strict=True)],
        "effective_voxel_nm": [round(v, 6) for v in grid.effective_voxel],
        "covers_full_box": grid.covers_full_box,
        "box_coverage": round(grid.box_coverage, 6),
        "patch": list(grid.patch),
        "stride": [p // 2 for p in grid.patch],
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "step": step,
        "data_config": str(data_config),
        "volume": grid.volume.name,
    }


def write_array(path: Path, array: np.ndarray, **attrs: Any) -> Path:
    store = zarr.open(
        str(path), mode="w", shape=array.shape, dtype=array.dtype,
        chunks=tuple(min(256, s) for s in array.shape),
    )
    store[:] = array
    store.attrs.update(**attrs)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="a mia-train run directory")
    parser.add_argument("--data-config", type=Path, required=True,
                        help="the miao YAML describing the volume (e.g. an eval split's config)")
    parser.add_argument("--volume", type=str, required=True, help="which volume in it to predict")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory; artifacts are named after the volume")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step to load; default is the newest")
    parser.add_argument("--patch", type=int, default=None,
                        help="override the patch size on every axis. Defaults to the data config's "
                             "own patch_size, cross-checked against the model's img_size; an "
                             "override changes what the encoder's positions mean, so it is for "
                             "probing that deliberately, not for tuning")
    parser.add_argument("--truth-only", action="store_true",
                        help="write only the ground-truth artifact; needs no GPU and no checkpoint")
    parser.add_argument("--override", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="patch one [algorithm] or [model] setting from the run's resolved "
                             "config before rebuilding it; repeatable. For knobs decided after "
                             "training, e.g. algorithm.pred_iou_thresh=0.7. Values are TOML.")
    args = parser.parse_args()

    from miao.config import load_config

    config = load_config(args.data_config)
    if config.resolutions is None:
        raise SystemExit(
            f"{args.data_config} sets no `resolutions`, so there is no single target resolution to "
            "predict at. Resolution sampling is a training-time device."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algorithm: Any = None
    predictor: VolumePredictor | None = None
    step = -1
    resolved: dict[str, Any] = {}
    if not args.truth_only:
        algorithm, step, resolved = load_algorithm(
            args.run_dir, device, args.step, overrides=args.override
        )
        # Resolved before any store is opened, so a strategy that cannot be run over a volume fails
        # here in a second rather than after the reads.
        predictor = select_predictor(algorithm)

    grid = VolumeGrid(
        config, args.volume, resolve_patch(config, resolved, args.volume, args.patch)
    )

    if predictor is not None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prediction = predictor.run(grid, device)
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(0)
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB of "
                  f"{properties.total_memory / 2**30:.0f} GiB ({properties.name})", flush=True)

        path = write_array(
            args.out / f"{args.volume}.zarr", prediction.array,
            kind=prediction.kind,
            **prediction.attrs,
            **shared_attrs(grid, args.run_dir, step, args.data_config),
        )
        print(f"wrote {path}  {prediction.array.shape} {prediction.array.dtype}", flush=True)

    truth = grid.read_ground_truth()
    instances = int((np.unique(truth) != 0).sum())
    path = write_array(
        args.out / f"{args.volume}.gt.zarr", truth,
        kind="instances",
        # 0 is background in every label store in this corpus, and -1 never occurs in one, so
        # nothing here is unannotated. Stated rather than assumed: for a thresholded-components
        # prediction 0 means "no edge survived", which is a different claim entirely.
        background_id=0,
        instances=instances,
        **shared_attrs(grid, args.run_dir, step, args.data_config),
    )
    print(f"wrote {path}  {truth.shape} int64, {instances} instances, "
          f"{100.0 * float((truth != 0).mean()):.1f}% annotated", flush=True)


if __name__ == "__main__":
    main()
