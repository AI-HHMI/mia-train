"""Imports every concrete implementation so the registries are populated.

A registry decorator only runs when its module is imported, so a config naming
"miao_volumes" resolves only once this module has been imported. `src/train.py` imports it
before building a run. Adding a component means adding one line here — the engine and the
registries themselves never need to change.
"""

from __future__ import annotations

import algorithms.mae  # noqa: F401  (imported for its registration side effect)
import data.miao_dataset  # noqa: F401  (imported for its registration side effect)
import models.vit  # noqa: F401  (imported for its registration side effect)
