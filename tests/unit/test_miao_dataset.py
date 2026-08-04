from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from data.miao_dataset import MiaoVolumeDataset
from data.registry import DataRegistry


def _config(volume: Path, **overrides: object) -> dict:
    config = {
        "volumes": [{"name": "test", "path": str(volume), "image_key": "raw"}],
        # The 64^3 fixture's level voxel sizes are [1,1,1], [2,2,2], [4,4,4], so these
        # resolutions map exactly onto pyramid levels 0 and 1.
        "resolutions": [[1, 1, 1], [2, 2, 2]],
        "output_axes": "lzyx",
        "patch_size": [8, 8, 8],
        "samples_per_epoch": 4,
    }
    config.update(overrides)
    return config


@pytest.mark.unit
def test_registered_under_its_config_name():
    assert DataRegistry.get("miao_volumes") is MiaoVolumeDataset


@pytest.mark.unit
def test_rejects_output_axes_without_scale_level_dim(ome_zarr_volume):
    # miao requires "l" in output_axes; the wrapper must surface that at construction.
    with pytest.raises(ValidationError, match="output_axes must contain 'l'"):
        MiaoVolumeDataset(**_config(ome_zarr_volume, output_axes="zyx"))


@pytest.mark.unit
def test_rejects_unknown_config_key(ome_zarr_volume):
    # MiaoConfig keeps pydantic's extra="ignore" default, so the wrapper — not pydantic — is
    # what stops a typo'd [data] key from silently selecting a miao default.
    with pytest.raises(ValueError, match=r"unknown \[data\] key\(s\).*not_a_miao_field"):
        MiaoVolumeDataset(**_config(ome_zarr_volume, not_a_miao_field=1))


@pytest.mark.unit
def test_sample_shape_is_scale_levels_then_spatial(ome_zarr_volume):
    dataset = MiaoVolumeDataset(**_config(ome_zarr_volume))
    with contextlib.redirect_stdout(io.StringIO()):  # miao prints a summary on construction
        sample = dataset.dataset[0]

    assert sample["img"].shape == (2, 8, 8, 8)  # (L, Z, Y, X)
    assert sample["img"].dtype == torch.float32
    assert sample["meta"]["volume"] == "test"


@pytest.mark.unit
def test_underlying_dataset_is_built_once(ome_zarr_volume):
    dataset = MiaoVolumeDataset(**_config(ome_zarr_volume))
    with contextlib.redirect_stdout(io.StringIO()):
        first = dataset.dataset
        second = dataset.dataset
    assert first is second


@pytest.mark.unit
def test_dataloader_collates_into_a_batched_encoder_input(ome_zarr_volume):
    dataset = MiaoVolumeDataset(**_config(ome_zarr_volume))
    with contextlib.redirect_stdout(io.StringIO()):
        loader = dataset.build_dataloader(
            batch_size=2, rank=0, world_size=1, shuffle=False, drop_last=True
        )
        batch = next(iter(loader))

    # Collation keeps the scale levels on their own axis: (B, L, Z, Y, X). Turning that into an
    # encoder input is the model's job (ViT3D.prepare_input requires L == 1).
    assert batch["img"].shape == (2, 2, 8, 8, 8)
    assert batch["bbox"].shape == (2, 2, 2, 3)
    assert batch["pixel_size"].shape == (2, 2, 3)
