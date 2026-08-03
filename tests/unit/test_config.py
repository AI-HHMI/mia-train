from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import as_plain_dict, load_run_config

_MINIMAL = """
experiment_name = "demo"

[model]
name = "my_model"
width = 32

[algorithm]
name = "my_algorithm"
mask_ratio = 0.75

[data]
name = "my_dataset"

[trainer]
max_steps = 100
batch_size = 4
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
def test_loads_components_and_splits_name_from_kwargs(tmp_path):
    config = load_run_config(_write(tmp_path, _MINIMAL))

    assert config.experiment_name == "demo"
    assert config.model.name == "my_model"
    assert config.model.kwargs == {"width": 32}
    assert config.algorithm.kwargs == {"mask_ratio": 0.75}
    assert config.data.kwargs == {}


@pytest.mark.unit
def test_trainer_section_becomes_trainer_config(tmp_path):
    config = load_run_config(_write(tmp_path, _MINIMAL))
    assert config.trainer.max_steps == 100
    assert config.trainer.batch_size == 4
    assert config.trainer.precision == "fp32"


@pytest.mark.unit
def test_parallelism_defaults_to_single_rank_when_absent(tmp_path):
    config = load_run_config(_write(tmp_path, _MINIMAL))
    assert config.parallelism.world_size == 1


@pytest.mark.unit
def test_parallelism_section_is_parsed(tmp_path):
    body = _MINIMAL + "\n[parallelism]\ndp_replicate = 2\ndp_shard = 2\ntp = 2\n"
    config = load_run_config(_write(tmp_path, body))
    assert config.parallelism.world_size == 8
    assert config.parallelism.hsdp_enabled
    assert config.parallelism.tp_enabled


@pytest.mark.unit
def test_val_data_is_optional(tmp_path):
    assert load_run_config(_write(tmp_path, _MINIMAL)).val_data is None

    body = _MINIMAL + '\n[val_data]\nname = "my_val_dataset"\n'
    loaded = load_run_config(_write(tmp_path, body))
    assert loaded.val_data is not None
    assert loaded.val_data.name == "my_val_dataset"


@pytest.mark.unit
def test_missing_experiment_name_is_rejected(tmp_path):
    body = _MINIMAL.replace('experiment_name = "demo"', "")
    with pytest.raises(ValueError, match="experiment_name"):
        load_run_config(_write(tmp_path, body))


@pytest.mark.unit
@pytest.mark.parametrize("section", ["model", "algorithm", "data", "trainer"])
def test_missing_required_section_is_named_in_the_error(tmp_path, section):
    # Rename the section header so the section itself is absent while the file stays valid TOML.
    body = _MINIMAL.replace(f"[{section}]", "[some_other_section]")
    with pytest.raises(ValueError, match=section):
        load_run_config(_write(tmp_path, body))


@pytest.mark.unit
def test_component_section_without_name_is_rejected(tmp_path):
    # Delete the 'name' line rather than overwriting it: reusing a key already in the section
    # would make the file invalid TOML, and tomllib's decode error is itself a ValueError, so
    # the missing-'name' branch would never be reached.
    body = _MINIMAL.replace('name = "my_model"\n', "")
    with pytest.raises(ValueError, match="model"):
        load_run_config(_write(tmp_path, body))


@pytest.mark.unit
def test_unknown_trainer_key_is_named_in_the_error(tmp_path):
    body = _MINIMAL + "\nlearning_rate = 0.1\n"
    with pytest.raises(ValueError, match="learning_rate"):
        load_run_config(_write(tmp_path, body))


@pytest.mark.unit
def test_invalid_trainer_value_still_validated_by_trainer_config(tmp_path):
    body = _MINIMAL.replace("max_steps = 100", "max_steps = 0")
    with pytest.raises(ValueError, match="max_steps"):
        load_run_config(_write(tmp_path, body))


@pytest.mark.unit
def test_as_plain_dict_includes_unspecified_defaults(tmp_path):
    resolved = as_plain_dict(load_run_config(_write(tmp_path, _MINIMAL)))
    assert resolved["trainer"]["seed"] == 0
    assert resolved["parallelism"]["tp"] == 1
    assert resolved["model"]["kwargs"] == {"width": 32}
