# SPDX-License-Identifier: Apache-2.0
"""WOOF-ArWen (CUDA WRF-ARW) and hex (CUDA MPAS-A) for NVIDIA Earth2Studio.

- ``WoofHistory``: a woof / woof hex run's history as an Earth2Studio
  ``DataSource`` (read through woof's Earth-2 exporter).
- ``WOOF-ArWen``: WOOF-ArWen as an Earth2Studio ``PrognosticModel`` (one woof run per
  step, boundaries from any Earth2Studio data source).

Earth2Studio has no plugin registry: its extension points are the
``DataSource`` and ``PrognosticModel`` protocols, and any object that
satisfies them goes straight into ``earth2studio.run.deterministic``,
``diagnostic`` and ``ensemble``. Both classes here satisfy them
(``isinstance(x, DataSource)`` / ``isinstance(m, PrognosticModel)``).
"""
from .data import WoofHistory

__all__ = ["WoofHistory", "WOOF-ArWen"]
__version__ = "0.1.0.dev0"


def __getattr__(name):
    if name == "WOOF-ArWen":
        from .px import ArWen
        return ArWen
    raise AttributeError(name)
