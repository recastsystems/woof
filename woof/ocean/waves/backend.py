"""Geographic backend ABI. Implementations enqueue on the supplied CUDA stream.

See CONTRACT.md for units, indexing, ownership, boundary and restart semantics.
This module intentionally imports no GPU runtime; it is safe for lane tooling.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GridBackend(Protocol):
    name: str
    ncell: int
    full_shape: tuple[int, ...]
    wet_to_full: Any
    full_to_wet: Any
    lon: Any
    lat: Any
    depth: Any
    area: Any
    depth_gradient_east: Any
    depth_gradient_north: Any
    boundary_cells: Any

    def max_dt(self, cg_max: float) -> float:
        """Return a conservative bound using precomputed host geometry scalars."""
        ...

    def advect(self, state: Any, group_velocity: Any, dt: float,
               stream: Any, incident_boundary: Any | None) -> Any:
        """Enqueue fluxes and return distinct persistent scratch, no host sync."""
        ...

    def set_depth(self, depth: Any, stream: Any) -> None:
        """Update compact geometry on device, preserving topology and CFL bound."""
        ...
