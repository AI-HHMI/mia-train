"""Whole-volume prediction as a library: the lattice, the reads, the contract, the dense default.

What every strategy's whole-volume prediction has in common, kept apart from `predict.py` -- the
CLI that loads a run and writes artifacts -- so that strategies can build on it. Three modules:

  * `types` -- `VolumePrediction` (what a prediction *is*) and `VolumePredictor` (how one is run),
    the contract behind `BaseAlgorithm.volume_predictor`.
  * `grid` -- `VolumeGrid`: one volume's aligned tile lattice and the miao reads that fill it, with
    the resampling and normalisation that make those reads the reads training saw.
  * `dense` -- `predict_volume` and `DensePredictor`: the weighted-average tiling every
    fixed-channel output gets for free, and `select_predictor`, the one-line dispatch.

**This package is a leaf.** It imports numpy, torch and miao and nothing from `algorithms/`,
`models/`, `engine/`, `data/` or the entrypoints, and `tests/unit/test_package_layout.py` keeps it
that way. That is the reason it exists: `algorithms/` imports it at runtime to type
`volume_predictor` and to build predictors of its own, and `predict.py` imports it to drive them.
This code used to live in `predict.py`, where a strategy could reach it only through a
`TYPE_CHECKING` import -- a runtime import of the entrypoint either fails as a circular import
(`predict.py` imports `algorithms.base` before it defines the grid) or, when `predict.py` is the
running script, executes the file a second time under the name `predict` and hands the strategy a
different `VolumeGrid` class from the one `__main__` holds.
"""

from .dense import DENSE_PROTOCOL, DensePredictor, blend_weight, predict_volume, select_predictor
from .grid import (
    VolumeGrid,
    aligned_tiling,
    normalize,
    resample_image,
    resample_labels,
    storage_axes_of,
)
from .types import VolumePrediction, VolumePredictor

__all__ = [
    "DENSE_PROTOCOL",
    "DensePredictor",
    "VolumeGrid",
    "VolumePrediction",
    "VolumePredictor",
    "aligned_tiling",
    "blend_weight",
    "normalize",
    "predict_volume",
    "resample_image",
    "resample_labels",
    "select_predictor",
    "storage_axes_of",
]
