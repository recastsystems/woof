"""Validated logical MPAS prognostic fields without a GPU-layout decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


@dataclass(slots=True)
class PrognosticState:
    """Fortran-logical state: level first, then cell/edge entity.

    Each field is a separate named array for readability.  No contiguity is
    required, so this class does not adjudicate the eventual GPU layout.
    """

    rho: FloatArray
    rho_theta: FloatArray
    rho_u: FloatArray
    rho_w: FloatArray
    scalars: FloatArray
    time_seconds: float = 0.0

    def validate(self, *, n_cells: int, n_edges: int, n_vert_levels: int) -> None:
        expected = {
            "rho": (n_vert_levels, n_cells),
            "rho_theta": (n_vert_levels, n_cells),
            "rho_u": (n_vert_levels, n_edges),
            "rho_w": (n_vert_levels + 1, n_cells),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if value.dtype.kind != "f":
                raise TypeError(f"{name} must be floating, got {value.dtype}")
            # The same boolean, checked in blocks on host threads (conus3km start-up, 2026-10-29).
            from .host_parallel import all_finite

            if not all_finite(value):
                raise FloatingPointError(f"{name} contains non-finite values")
        if self.scalars.ndim != 3 or self.scalars.shape[1:] != (n_vert_levels, n_cells):
            raise ValueError(
                f"scalars shape {self.scalars.shape} must be (nScalars, {n_vert_levels}, {n_cells})"
            )
        if np.any(self.rho <= 0.0):
            raise ValueError("rho must be strictly positive")

    def copy(self) -> "PrognosticState":
        return PrognosticState(
            rho=self.rho.copy(),
            rho_theta=self.rho_theta.copy(),
            rho_u=self.rho_u.copy(),
            rho_w=self.rho_w.copy(),
            scalars=self.scalars.copy(),
            time_seconds=self.time_seconds,
        )

    def bounds_receipt(self) -> dict[str, dict[str, float | int]]:
        receipt: dict[str, dict[str, float | int]] = {}
        for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
            value = np.asarray(getattr(self, name))
            receipt[name] = {
                "count": int(value.size),
                "finite": int(np.count_nonzero(np.isfinite(value))),
                "min": float(np.nanmin(value)),
                "max": float(np.nanmax(value)),
            }
        return receipt

