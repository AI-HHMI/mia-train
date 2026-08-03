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

    def __post_init__(self) -> None:
        if self.dp_replicate < 1 or self.dp_shard < 1 or self.tp < 1:
            raise ValueError(f"dp_replicate, dp_shard, and tp must all be >= 1, got {self}")

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
