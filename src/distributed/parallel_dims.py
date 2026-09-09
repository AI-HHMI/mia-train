from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


@dataclass(frozen=True)
class ParallelDims:
    """Ranks laid out across replicate / shard / tensor-parallel mesh dimensions."""

    dp_replicate: int = 1
    dp_shard: int = 1
    tp: int = 1
    # Gathered-parameter budget for one FSDP unit, in GiB, or "auto" to derive it from the
    # device. `distributed.parallelize` packs a model's declared `fsdp_units` into groups that
    # each stay under this, which is the knob between two costs that pull opposite ways: one
    # unit per block bounds resident weights to a block but issues a collective per block, and
    # a single unit for the whole model issues one collective but materializes every parameter
    # for the whole forward. Measured on the production SimMIM arm, 8xH200, ViT-L: per-block
    # units cost 7.7% throughput (114.3 +/- 2.1 over 4 runs against 123.8 +/- 0.3) to save
    # ~1.2 GB of resident weights, which on a 141 GB device buys nothing. At 7B the same saving
    # is ~28 GB and decides whether the run fits at all -- hence a budget, not a boolean.
    fsdp_unit_budget_gb: float | str = "auto"

    def __post_init__(self) -> None:
        if self.dp_replicate < 1 or self.dp_shard < 1 or self.tp < 1:
            raise ValueError(f"dp_replicate, dp_shard, and tp must all be >= 1, got {self}")
        budget = self.fsdp_unit_budget_gb
        # `bool` is excluded explicitly because it is a subclass of `int`, so `True > 0` would
        # otherwise sail through as a 1 GiB budget.
        numeric = isinstance(budget, int | float) and not isinstance(budget, bool)
        if not (budget == "auto" or (numeric and budget > 0)):
            raise ValueError(
                f'fsdp_unit_budget_gb must be a positive number of GiB or "auto", '
                f"got {budget!r}"
            )

    @property
    def world_size(self) -> int:
        return self.dp_replicate * self.dp_shard * self.tp

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def hsdp_enabled(self) -> bool:
        return self.dp_replicate > 1 and self.dp_shard > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def dp_world_size(self) -> int:
        """Number of distinct data shards: tensor-parallel peers all read the same batch."""
        return self.dp_replicate * self.dp_shard

    def dp_rank(self, mesh: DeviceMesh) -> int:
        """This rank's index within the data-parallel plane (0 <= dp_rank < dp_world_size)."""
        replicate_rank = mesh.get_local_rank("dp_replicate") if self.dp_replicate > 1 else 0
        shard_rank = mesh.get_local_rank("dp_shard") if self.dp_shard > 1 else 0
        return replicate_rank * self.dp_shard + shard_rank

    def build_mesh(self, device_type: str) -> DeviceMesh:
        actual_world_size = dist.get_world_size()
        if actual_world_size != self.world_size:
            raise ValueError(
                f"process group world_size={actual_world_size} does not match "
                f"dp_replicate({self.dp_replicate}) * dp_shard({self.dp_shard}) * "
                f"tp({self.tp}) = {self.world_size}"
            )
        return init_device_mesh(
            device_type,
            (self.dp_replicate, self.dp_shard, self.tp),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )
