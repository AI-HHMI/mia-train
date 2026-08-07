from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import torch
from miao.config import MiaoConfig
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


def _write_yaml(tmp_path: Path, volume: Path, **overrides: object) -> Path:
    import yaml

    body = _config(volume)
    body.update(overrides)
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


@pytest.mark.unit
def test_dataset_can_be_described_by_a_miao_yaml(ome_zarr_volume, tmp_path):
    # miao's own config format, referenced instead of restated. The round trip is the part worth
    # checking: the file is parsed by miao, dumped back to plain settings, and revalidated, so a
    # field that did not survive that would silently fall back to a miao default.
    path = _write_yaml(tmp_path, ome_zarr_volume)
    dataset = MiaoVolumeDataset(config_path=str(path))

    assert dataset.sample_axes == "lzyx"
    assert list(dataset.config.patch_size) == [8, 8, 8]
    assert dataset.config.samples_per_epoch == 4
    assert [v.name for v in dataset.config.volumes] == ["test"]


@pytest.mark.unit
def test_a_yaml_dataset_produces_the_same_config_as_inlining_it(ome_zarr_volume, tmp_path):
    # The two ways of describing a dataset must be exactly equivalent, or "move this into a YAML"
    # would quietly change what a run trains on.
    path = _write_yaml(tmp_path, ome_zarr_volume)
    assert (
        MiaoVolumeDataset(config_path=str(path)).config
        == MiaoVolumeDataset(**_config(ome_zarr_volume)).config
    )


@pytest.mark.unit
def test_inline_keys_override_the_yaml(ome_zarr_volume, tmp_path):
    # What makes a shared dataset definition usable: adjust it per run without copying it.
    path = _write_yaml(tmp_path, ome_zarr_volume, samples_per_epoch=4)
    dataset = MiaoVolumeDataset(config_path=str(path), samples_per_epoch=2)
    assert dataset.config.samples_per_epoch == 2
    assert list(dataset.config.patch_size) == [8, 8, 8], "unrelated file settings must survive"


@pytest.mark.unit
def test_a_relative_config_path_resolves_against_the_repository(ome_zarr_volume):
    # Not the working directory: a submitted job starts wherever it was launched from, so a
    # cwd-relative path would resolve differently under bsub than it does interactively.
    dataset = MiaoVolumeDataset(config_path="configs/data/lmd_33nm.yaml")
    assert dataset.sample_axes == "lcxyz"
    assert len(dataset.config.volumes) == 2


@pytest.mark.unit
def test_reports_a_missing_config_path_with_the_place_it_looked(ome_zarr_volume):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MiaoVolumeDataset(config_path="configs/data/no_such_file.yaml")


@pytest.mark.unit
def test_config_path_is_not_mistaken_for_a_miao_field(ome_zarr_volume, tmp_path):
    # It is handled here, not forwarded, so the unknown-key guard must not reject it -- while
    # still rejecting an actual typo alongside it.
    path = _write_yaml(tmp_path, ome_zarr_volume)
    with pytest.raises(ValueError, match=r"unknown \[data\] key\(s\).*not_a_miao_field"):
        MiaoVolumeDataset(config_path=str(path), not_a_miao_field=1)


@pytest.mark.unit
def test_resolve_settings_expands_a_referenced_file(ome_zarr_volume, tmp_path):
    # The point of expanding rather than recording the path: the run record must not depend on a
    # file that can change afterwards. Asking the class, not an instance, is what lets the engine
    # write a complete record before touching any data.
    path = _write_yaml(tmp_path, ome_zarr_volume)
    settings = MiaoVolumeDataset.resolve_settings(config_path=str(path))

    assert "config_path" not in settings, "the reference is replaced by what it referred to"
    assert settings["patch_size"] == [8, 8, 8]
    assert [v["name"] for v in settings["volumes"]] == ["test"]
    # Defaults miao filled in are captured too, so the record survives a miao that changes one.
    assert "image_dtype" in settings


@pytest.mark.unit
def test_editing_the_referenced_file_cannot_rewrite_an_existing_record(ome_zarr_volume, tmp_path):
    path = _write_yaml(tmp_path, ome_zarr_volume, samples_per_epoch=4)
    recorded = MiaoVolumeDataset.resolve_settings(config_path=str(path))

    path.write_text(path.read_text().replace("samples_per_epoch: 4", "samples_per_epoch: 999"))
    assert recorded["samples_per_epoch"] == 4
    assert MiaoVolumeDataset.resolve_settings(config_path=str(path))["samples_per_epoch"] == 999


@pytest.mark.unit
def test_the_record_is_what_the_dataset_actually_uses(ome_zarr_volume, tmp_path):
    # `__init__` and the run record go through the same classmethod, so they cannot disagree --
    # a record that drifted from the dataset would be worse than no record.
    path = _write_yaml(tmp_path, ome_zarr_volume)
    dataset = MiaoVolumeDataset(config_path=str(path), samples_per_epoch=2)
    recorded = MiaoVolumeDataset.resolve_settings(config_path=str(path), samples_per_epoch=2)
    assert dataset.config == MiaoConfig(**recorded)


@pytest.mark.unit
def test_inline_settings_are_expanded_too(ome_zarr_volume):
    # Not only referenced files: an inline section is expanded to include miao's own defaults, so
    # every run records the settings it actually ran with rather than the subset it spelled out.
    settings = MiaoVolumeDataset.resolve_settings(**_config(ome_zarr_volume))
    assert set(settings) > set(_config(ome_zarr_volume))
    assert settings["patch_size"] == [8, 8, 8]


@pytest.mark.unit
def test_resolved_settings_are_json_serialisable(ome_zarr_volume, tmp_path):
    # They go straight into resolved_config.json, so a stray Path or numpy scalar would break the
    # artifact write at the very end of a long run's startup.
    import json

    path = _write_yaml(tmp_path, ome_zarr_volume)
    json.dumps(MiaoVolumeDataset.resolve_settings(config_path=str(path)))


@pytest.mark.unit
def test_referenced_files_names_the_yaml_for_copying(ome_zarr_volume, tmp_path):
    path = _write_yaml(tmp_path, ome_zarr_volume)
    assert MiaoVolumeDataset.referenced_files(config_path=str(path)) == (path,)
    assert MiaoVolumeDataset.referenced_files(**_config(ome_zarr_volume)) == ()


@pytest.mark.unit
def test_miao_supplies_the_bounding_box_contract_muvit_positions_tokens_from(ome_zarr_volume):
    """Pin the miao behaviour `MuViT3D` depends on, since nothing downstream can detect its loss.

    MuViT places every token at a world coordinate derived from these boxes, and wrong coordinates
    are still perfectly well-shaped: the model trains, the loss falls, and the cross-scale geometry
    is quietly meaningless. `pyproject.toml` therefore floors miao-io rather than capping it, and
    relies on this test to fail loudly if a future release changes the contract.
    """
    dataset = MiaoVolumeDataset(**_config(ome_zarr_volume))
    sample = dataset.dataset[0]

    # 1. A box per level, as [low, high] corners over the spatial axes.
    assert "bbox" in sample
    assert tuple(sample["bbox"].shape) == (2, 2, 3), "expected (levels, [low, high], spatial)"

    low, high = sample["bbox"][:, 0, :], sample["bbox"][:, 1, :]
    extents = high - low

    # 2. Physical units, not voxel counts: the fixture's levels have voxel sizes 1 and 2, and the
    #    patch is 8 voxels, so a level's extent is its voxel size times the patch.
    assert torch.allclose(extents[0], torch.full((3,), 8.0))
    assert torch.allclose(extents[1], torch.full((3,), 16.0))

    # 3. The ratio between levels tracks the resolution ratio. This is what makes a coarse level
    #    cover proportionally more scene, which is the whole premise of multi-resolution encoding.
    assert torch.allclose(extents[1] / extents[0], torch.full((3,), 2.0))

    # 4. The levels overlap rather than sitting in unrelated places.
    assert ((low[1] <= low[0] + 1e-6) & (high[1] >= high[0] - 1e-6)).all(), (
        "the coarse level should contain the fine one"
    )


@pytest.mark.unit
def test_a_miao_bbox_drops_straight_into_muvit(ome_zarr_volume):
    # The integration the contract exists for: miao's box, collated, is exactly what the encoder's
    # coordinate machinery accepts -- no reshaping, reordering or unit conversion in between.
    from models.muvit import MuViT3D

    dataset = MiaoVolumeDataset(**_config(ome_zarr_volume))
    loader = dataset.build_dataloader(batch_size=2, rank=0, world_size=1, shuffle=False)
    batch = next(iter(loader))

    model = MuViT3D(
        levels=(1, 2), img_size=(8, 8, 8), patch_size=(4, 4, 4), in_channels=1,
        embed_dim=32, depth=1, num_heads=2, attention_backend="sdpa",
    )
    coords = model.world_coords(batch["bbox"])
    assert coords.shape == (2, model.num_patches, 3)

    # Coarse-level patches spread over twice the span of fine-level ones, carried through from
    # miao's extents rather than from the model's own concentric fallback.
    per_level = coords.reshape(2, 2, model.patches_per_level, 3)
    fine_span = per_level[0, 0].max(0).values - per_level[0, 0].min(0).values
    coarse_span = per_level[0, 1].max(0).values - per_level[0, 1].min(0).values
    assert torch.allclose(coarse_span / fine_span, torch.full((3,), 2.0), atol=1e-4)
