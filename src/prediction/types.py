"""What a whole-volume prediction is, and how one is run: the contract behind `volume_predictor`."""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import numpy as np
import torch

from .grid import VolumeGrid


class VolumePrediction(NamedTuple):
    """What a whole-volume prediction is: the array, what it means, and how it was made.

    `kind` is the artifact's declared meaning (`"affinity"`, `"class_scores"`, `"instances"`) and
    is what `mia-evals` dispatches on. `attrs` is whatever the predictor needs recorded beside the
    data for it to be interpretable -- a squash convention, a channel count, the prompt grid that
    produced it. `predict.py` adds the geometry and provenance that are the same for every kind.
    """

    array: np.ndarray
    kind: str
    attrs: dict[str, Any]


class VolumePredictor(Protocol):
    """How one strategy is run over a whole volume, tile by tile, and its tiles reconciled.

    Everything a whole-volume prediction has in common is owned elsewhere: `prediction.grid`
    resolves the lattice and reads tiles through miao, `predict.py` loads the run and writes the
    artifact and the ground truth beside it. What differs between strategies is exactly two things:
    what runs on a tile, and how tiles combine. A dense prediction (affinities, class scores) is
    blended by weighted average across overlaps; a set of instance masks is reconciled by identity,
    and averaging two labellings is not a labelling. So those two things belong to the algorithm,
    behind this one method, and the entrypoint drives it blindly -- the same division
    `BaseAlgorithm.training_step` makes for the trainer.

    `grid` supplies the tiles and their reads in storage axis order; the returned array must be on
    `grid.output_shape` (with any leading channel axis) so the ground truth written beside it is
    co-registered.
    """

    def run(self, grid: VolumeGrid, device: torch.device) -> VolumePrediction: ...
