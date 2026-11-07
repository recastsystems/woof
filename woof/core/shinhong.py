"""Shin-Hong scale-aware PBL column kernel (bl_pbl_physics=11).

This module is the device launcher for the WRF v4.6.1 Shin-Hong
transcription in ``kernels/shinhong.cu``: arm A (``ctopo = ctopo2 = 1``,
the Registry ``topo_wind=0`` default) and arm B (WRF's ``topo_wind`` = 1 or
2, ``topo=``).  The scheme consumes mass-point wind, potential temperature,
moisture, pressure/Exner, layer depths, the scheme's own prognostic SGS TKE,
and the surface-layer coupling fields (UST, HFX/QFX, PSIM/PSIH, plus the
Coriolis parameter for the QNSE mixing length).  It returns tendencies
rather than updating the state; the physics driver owns their application
(Phase D wires it).

Array layout is woof's standard ``(nz, ny, nx)`` with x fastest.  Every
CUDA thread owns one complete column and performs the implicit vertical
solves in-thread, matching the acoustic Thomas-solve pattern.  The float32
authority is :func:`woof.verify.shinhong_ref.np_shinhong_column`, itself
bitwise against the WRF v4.6.1 oracle; the kernel's measured distance from
WRF is pinned in ``tests/test_shinhong_wrf461_parity.py``.

``dx``/``dy``/``dt``/``tke_diag`` are launch-wide scalars: the whole point
of Shin-Hong is that the answer moves with ``sqrt(dx*dy)``, so each nest
launches with its own spacing.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import cupy as cp

from woof.core.kernels import get_kernel
from woof.core.state import DTYPE
from woof.core.ysu import _launch_output_validation
from woof.physics_vertical_contract import (
    outside_vertical_bounds, refuse_vertical_levels)

from woof.core.physics_inventory import (
    SHWS_SLOTS, SHINHONG_BLOCK, SHINHONG_TILE_BLOCKS_PER_SM,
    shinhong_workspace_floats,
)

_TPB = SHINHONG_BLOCK
_VALIDATE_TPB = 256
_KMAX = 128

#: The launcher's first-call vertical bound, restated for front doors by
#: ``woof.physics_vertical_contract.SHINHONG_VERTICAL_LEVEL_BOUNDS`` and
#: bound to it by ``tests/test_pbl_vertical_bounds.py``.
VERTICAL_LEVEL_BOUNDS = (4, _KMAX)

_SHINHONG_3D_FLOAT_OUTPUTS = (
    "du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "tke", "el",
)
_SHINHONG_2D_OUTPUTS = ("hpbl", "kpbl", "wstar", "delta")
_SHINHONG_OUTPUTS = _SHINHONG_3D_FLOAT_OUTPUTS + _SHINHONG_2D_OUTPUTS
_SHINHONG_2D_FLOAT_OUTPUTS = tuple(
    name for name in _SHINHONG_2D_OUTPUTS if name != "kpbl")

#: The scheme's PASSENGER diagnostics: outputs no tendency ever reads.
#: The SGS TKE chain (``mixlen``/``prodq2``/``vdifq``, tke_diag = 1) and
#: its mixing length exist so ``state.e_sgs`` carries a gray-zone
#: instrument field; WRF's own default never computes them, and WRF's
#: arithmetic inside them divides by quantities that legitimately reach
#: zero (prodq2's ``q2**1.5/disel`` at q2 == 0, ``mf/xkzh``, prfac2's
#: cubes).  A non-finite CONFINED to this pair is therefore a documented
#: hazard of the transcribed chain, not corrupted physics -- the
#: tendencies above it are computed independently and stay finite
#: (proven through the float32 authority in
#: tests/test_shinhong_composed_tke_guard.py, task #206).
SHINHONG_PASSENGER_OUTPUTS = ("tke", "el")

#: WRF's shinhonginit cold-start TKE, epsq2l/2 -- the chain's own floor
#: (the authority's EPSQ2L lives in the verification tree this module
#: must not depend on; the pair is gated equal in the task #206 suite).
SHINHONG_TKE_FLOOR = 0.005


def shinhong_tile_columns(fn, ncol: int) -> int:
    """Bound workspace storage to columns that can be resident."""
    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    per_sm = SHINHONG_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        from woof.backend import active_blocks_per_multiprocessor
        resident = active_blocks_per_multiprocessor(
            fn.kernel.ptr, SHINHONG_BLOCK, 0)
        per_sm = min(per_sm, int(resident))
    except Exception:  # noqa: BLE001
        pass
    per_sm = max(1, int(per_sm))
    return int(max(SHINHONG_BLOCK,
                   min(int(ncol), sms * per_sm * SHINHONG_BLOCK)))


def launch_shinhong(u, v, theta, qv, qc, qi, p, p_interface, exner, dz,
                    tke, *,
                    psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland,
                    u10, v10, corf,
                    dt: float, dx: float, dy: float, tke_diag: int = 1,
                    topo: tuple[cp.ndarray, cp.ndarray] | None = None,
                    u10_out: cp.ndarray | None = None,
                    v10_out: cp.ndarray | None = None):
    """Launch Shin-Hong for a batch of device columns; return device outputs.

    Three-dimensional inputs have shape ``(nz, ny, nx)`` except
    ``p_interface`` which is ``(nz+1, ny, nx)``; ``tke`` is the scheme's own
    prognostic-diagnostic SGS TKE.  Surface inputs have shape ``(ny, nx)``.
    All are C-contiguous float32 CuPy arrays.  The returned dict contains
    3-D ``du/dv/dtheta/dqv/dqc/dqi/exch_h/tke/el`` float32 arrays (``dtheta``
    is the wrapper's ``rthblten``, already divided by the Exner function),
    2-D ``hpbl/wstar/delta`` float32 arrays, and 2-D one-based ``kpbl``
    int32.

    ``topo`` (WRF ``topo_wind`` = 1 or 2) is ``(ctopo, ctopo2)``, the two
    (ny, nx) coefficients of
    :func:`woof.core.terrain_drag.topo_wind_coefficients`: the kernel then
    takes WRF's arm B (``shinhong_column_topo``): CTOPO scales the surface
    drag (module_bl_shinhong.F:1387) and the blended 10 m wind of
    :1472-1473 is written into ``u10_out``/``v10_out``, which the caller
    supplies; ``u10``/``v10`` are read, never written.
    """
    columns = {"u": u, "v": v, "theta": theta, "qv": qv, "qc": qc,
               "qi": qi, "p": p, "exner": exner, "dz": dz, "tke": tke}
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"theta must have shape (nz, ny, nx), got {shape}")
    nz, ny, nx = shape
    if outside_vertical_bounds(nz, VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "Shin-Hong PBL", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "one CUDA thread owns a whole column and holds it in "
                f"a global column workspace bounded by SHINHONG_KMAX={_KMAX} "
                "(kernels/shinhong.cu), and the scheme's YSU-inherited "
                "counter-gradient and entrainment indexing needs four levels "
                "to exist."))
    for name, arr in columns.items():
        if not isinstance(arr, cp.ndarray) or arr.shape != shape:
            raise ValueError(f"{name} must be a CuPy array with shape {shape}")
        if arr.dtype != DTYPE or not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32")
    if (not isinstance(p_interface, cp.ndarray)
            or p_interface.shape != (nz + 1, ny, nx)
            or p_interface.dtype != DTYPE
            or not p_interface.flags.c_contiguous):
        raise ValueError("p_interface must be C-contiguous float32 with shape "
                         f"{(nz + 1, ny, nx)}")
    surfaces = {"psfc": psfc, "znt": znt, "ust": ust, "hfx": hfx,
                "qfx": qfx, "wspd": wspd, "br": br, "psim": psim,
                "psih": psih, "xland": xland, "u10": u10, "v10": v10,
                "corf": corf}
    for name, arr in surfaces.items():
        if (not isinstance(arr, cp.ndarray) or arr.shape != (ny, nx)
                or arr.dtype != DTYPE or not arr.flags.c_contiguous):
            raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                             f"array with shape {(ny, nx)}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Shin-Hong requires a positive finite dt, got {dt}")
    for label, value in (("dx", dx), ("dy", dy)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Shin-Hong requires a positive finite {label}, got {value}")
    if tke_diag not in (0, 1, False, True):
        raise ValueError("tke_diag must be 0 or 1")
    topo_args = ()
    if topo is not None:
        ctopo, ctopo2 = topo
        for name, arr in (("ctopo", ctopo), ("ctopo2", ctopo2),
                          ("u10_out", u10_out), ("v10_out", v10_out)):
            if (not isinstance(arr, cp.ndarray) or arr.shape != (ny, nx)
                    or arr.dtype != DTYPE or not arr.flags.c_contiguous):
                raise ValueError(f"topo_wind needs {name}, a C-contiguous "
                                 f"float32 CuPy array with shape {(ny, nx)}")
        topo_args = (ctopo, ctopo2, u10_out, v10_out)
    elif u10_out is not None or v10_out is not None:
        raise ValueError("u10_out/v10_out are only written with topo")

    out = {name: cp.empty_like(theta) for name in _SHINHONG_3D_FLOAT_OUTPUTS}
    out.update(hpbl=cp.empty((ny, nx), dtype=DTYPE),
               kpbl=cp.empty((ny, nx), dtype=cp.int32),
               wstar=cp.empty((ny, nx), dtype=DTYPE),
               delta=cp.empty((ny, nx), dtype=DTYPE))
    ncol = ny * nx
    kernel = get_kernel("shinhong", "shinhong_column_topo" if topo_args
                        else "shinhong_column")
    tile = shinhong_tile_columns(kernel, ncol)
    ws = cp.empty(shinhong_workspace_floats(nz, tile), dtype=DTYPE)
    for lo in range(0, ncol, tile):
        blocks = (min(tile, ncol - lo) + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,),
               (u, v, theta, qv, qc, qi, p, p_interface, exner, dz, tke,
                psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, u10, v10,
                corf,
                out["du"], out["dv"], out["dtheta"], out["dqv"], out["dqc"],
                out["dqi"], out["exch_h"], out["tke"], out["el"],
                out["hpbl"], out["kpbl"], out["wstar"], out["delta"],
                DTYPE(dt), DTYPE(dx), DTYPE(dy), np.int32(int(tke_diag)),
                np.int32(nz), np.int32(ny), np.int32(nx),
                ws, np.int32(nz + 2), np.int32(lo)) + topo_args)
    return out


def validate_shinhong_outputs(
        out: Mapping[str, cp.ndarray], status: cp.ndarray) -> str | None:
    """Return the first non-finite native Shin-Hong output name, or ``None``.

    The first-invalid readback in launcher order, kept for the callers
    that want the historical scalar answer; the driver's disposition
    policy reads :func:`invalid_shinhong_outputs` instead, because
    "which output" decides fatality (task #206).
    """
    names = invalid_shinhong_outputs(out, status)
    return names[0] if names else None


def invalid_shinhong_outputs(
        out: Mapping[str, cp.ndarray], status: cp.ndarray) -> tuple[str, ...]:
    """Return EVERY non-finite native Shin-Hong output name, launcher order.

    ``launch_shinhong`` owns this exact output layout.  One validation kernel
    records a bit per floating-point output and one scalar readback preserves
    the driver's historical first-invalid error ordering.  The integer
    ``kpbl`` is necessarily finite and needs no device work.  Note that a
    NaN here is not always a port defect: WRF's own ``prfac2`` 0/0
    (module_bl_shinhong.F:1010) writes NaN heat columns on purpose-built
    inputs, and the kernel reproduces it; this validator exists so the
    driver can say so instead of silently advancing corrupted state.
    """
    _launch_output_validation(
        out, status, scheme="Shin-Hong", names=_SHINHONG_OUTPUTS,
        level_names=_SHINHONG_3D_FLOAT_OUTPUTS,
        surface_names=_SHINHONG_2D_FLOAT_OUTPUTS,
        tpb=_VALIDATE_TPB, load_kernel=get_kernel)
    invalid = int(status[0].item())
    return tuple(name for bit, name in enumerate(_SHINHONG_OUTPUTS)
                 if invalid & (1 << bit))


def repair_shinhong_passenger_outputs(
        out: dict[str, cp.ndarray]) -> dict[str, int]:
    """Repair non-finite passenger diagnostics, rebinding ``out``'s entries.

    The caller has already established (via
    :func:`invalid_shinhong_outputs`) that every CONSUMED output is
    finite; this replaces each non-finite ``tke`` value with the chain's
    own cold-start floor (:data:`SHINHONG_TKE_FLOOR`, WRF's
    ``shinhonginit`` epsq2l/2) and each non-finite ``el`` value with the
    scheme's :673 zero, so ``state.e_sgs`` never inherits a NaN that
    would poison every later ``mixlen`` call.  DOCUMENTED DIVERGENCE
    from WRF: with tke_diag on, stock WRF leaves the NaN in ``TKE_PBL``
    forever (and by default never computes the chain at all); carrying a
    poisoned instrument field silently is the WRF-undefined behaviour
    this repair replaces with a defined one.  Returns ``{name: repaired
    count}`` for the driver's advisory; array-module dispatch keeps the
    function testable on NumPy arrays without a device.
    """
    counts: dict[str, int] = {}
    for name, floor in (("tke", SHINHONG_TKE_FLOOR), ("el", 0.0)):
        value = out[name]
        xp = cp.get_array_module(value)
        bad = ~xp.isfinite(value)
        repaired = int(bad.sum())
        if repaired:
            out[name] = xp.where(bad, value.dtype.type(floor), value)
            counts[name] = repaired
    return counts
