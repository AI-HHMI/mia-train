"""Imports every concrete implementation so the registries are populated.

A registry decorator only runs when its module is imported, so a config naming
"miao_volumes" resolves only once this module has been imported. `src/train.py` imports it
before building a run. Adding a component means adding one line here — the engine and the
registries themselves never need to change.
"""

from __future__ import annotations

import algorithms.affinity_seg  # noqa: F401  (imported for its registration side effect)
import algorithms.dinov3_ssl  # noqa: F401  (imported for its registration side effect)
import algorithms.mae  # noqa: F401  (imported for its registration side effect)
import algorithms.muvit_mae  # noqa: F401  (imported for its registration side effect)
import algorithms.semantic_seg  # noqa: F401  (imported for its registration side effect)
import algorithms.simmim  # noqa: F401  (imported for its registration side effect)
import data.huggingface  # noqa: F401  (imported for its registration side effect)
import data.miao_dataset  # noqa: F401  (imported for its registration side effect)
import evals.semantic_seg  # noqa: F401  (imported for its registration side effect)
import models.dinov3_vit  # noqa: F401  (imported for its registration side effect)
import models.dinov3_vit3d  # noqa: F401  (imported for its registration side effect)
import models.muvit  # noqa: F401  (imported for its registration side effect)
import models.vit  # noqa: F401  (imported for its registration side effect)
