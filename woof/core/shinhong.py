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


# ---------------------------------------------------------------------------
# WRF v4.8.0 generation (physics_mmm bl_shinhong.F90), kernels/shinhong_480.cu.
# ---------------------------------------------------------------------------

#: Generation keys, ``RunConfig.shinhong_version``.  ``wrf_461`` is the
#: 4.6.1/4.7.1 scheme (module_bl_shinhong.F sha256 99f44dbe, the same file in
#: both releases) on the unchanged ``shinhong`` unit; ``wrf_480`` is WRF
#: 4.8.0's rewrite on ``shinhong_480``.
from woof.core.shinhong_generations import (  # noqa: E402
    SHINHONG_DEFAULT_GENERATION, SHINHONG_GENERATIONS)

_SH480_3D_FLOAT_OUTPUTS = (
    "du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "exch_m", "tke", "el",
)
_SH480_2D_FLOAT_OUTPUTS = ("hpbl", "wstar", "delta", "u10", "v10")
_SH480_OUTPUTS = _SH480_3D_FLOAT_OUTPUTS + ("kpbl",) + _SH480_2D_FLOAT_OUTPUTS

#: kernels/shinhong_480.cu option bits.
SH480_NONLOCAL_FLUX = 1
SH480_SCU_MIXING = 2
SH480_KE_DISSIPATION = 4


def shinhong480_flags(*, nonlocal_flux: bool = True, scu_mixing: bool = False,
                      ke_dissipation: bool = False) -> int:
    """The launch's option word from WRF's three namelist logicals."""
    return ((SH480_NONLOCAL_FLUX if nonlocal_flux else 0)
            | (SH480_SCU_MIXING if scu_mixing else 0)
            | (SH480_KE_DISSIPATION if ke_dissipation else 0))


def launch_shinhong_480(u, v, theta, qv, qc, qi, p, p_interface, exner, dz,
                        tke, *, rthraten: cp.ndarray | None = None,
                        psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland,
                        u10, v10, corf, dt: float, dx: float,
                        nonlocal_flux: bool = True, scu_mixing: bool = False,
                        ke_dissipation: bool = False,
                        uoce: cp.ndarray | None = None,
                        voce: cp.ndarray | None = None,
                        topo: tuple[cp.ndarray, cp.ndarray] | None = None):
    """Launch WRF 4.8.0 Shin-Hong on device columns; return device outputs.

    The inputs are :func:`launch_shinhong`'s, plus ``rthraten`` (the
    radiation theta tendency, read only by the stratocumulus option) and the
    optional sea-surface current ``uoce``/``voce`` (zero when absent, as
    WRF's driver passes it uncoupled).  ``dx`` is WRF's ``dx2d``, which is
    the domain ``dx`` (module_physics_init.F compute_2d_dx_area).  The TKE
    chain always runs in 4.8.0.  Returned: ``du/dv/dtheta/dqv/dqc/dqi/
    exch_h/exch_m/tke/el`` (nz, ny, nx), ``hpbl/wstar/delta`` and the
    scheme's ``u10``/``v10`` (ny, nx) float32, ``kpbl`` (ny, nx) int32.
    """
    columns = {"u": u, "v": v, "theta": theta, "qv": qv, "qc": qc,
               "qi": qi, "p": p, "exner": exner, "dz": dz, "tke": tke}
    if rthraten is not None:
        columns["rthraten"] = rthraten
    elif scu_mixing:
        raise ValueError(
            "shinhong_scu_mixing reads the radiation theta tendency "
            "(bl_shinhong.F90 radsum, :783-787); pass rthraten")
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
    for name, arr in (("uoce", uoce), ("voce", voce)):
        if arr is not None:
            surfaces[name] = arr
    if topo is not None:
        surfaces["ctopo"], surfaces["ctopo2"] = topo
    for name, arr in surfaces.items():
        if (not isinstance(arr, cp.ndarray) or arr.shape != (ny, nx)
                or arr.dtype != DTYPE or not arr.flags.c_contiguous):
            raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                             f"array with shape {(ny, nx)}")
    if (uoce is None) != (voce is None):
        raise ValueError("uoce and voce are passed together or not at all")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Shin-Hong requires a positive finite dt, got {dt}")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"Shin-Hong requires a positive finite dx, got {dx}")

    out = {name: cp.empty_like(theta) for name in _SH480_3D_FLOAT_OUTPUTS}
    for name in _SH480_2D_FLOAT_OUTPUTS:
        out[name] = cp.empty((ny, nx), dtype=DTYPE)
    out["kpbl"] = cp.empty((ny, nx), dtype=cp.int32)
    flags = shinhong480_flags(nonlocal_flux=nonlocal_flux,
                              scu_mixing=scu_mixing,
                              ke_dissipation=ke_dissipation)
    ncol = ny * nx
    kernel = get_kernel("shinhong_480", "shinhong480_column_topo" if topo
                        is not None else "shinhong480_column")
    tile = shinhong_tile_columns(kernel, ncol)
    ws = cp.empty(shinhong_workspace_floats(nz, tile), dtype=DTYPE)
    null = np.uintp(0)
    topo_args = tuple(topo) if topo is not None else ()
    for lo in range(0, ncol, tile):
        blocks = (min(tile, ncol - lo) + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,),
               (u, v, theta, qv, qc, qi, p, p_interface, exner, dz, tke,
                rthraten if rthraten is not None else null, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih,
                xland, u10, v10, corf,
                uoce if uoce is not None else null,
                voce if voce is not None else null,
                out["du"], out["dv"], out["dtheta"], out["dqv"], out["dqc"],
                out["dqi"], out["exch_h"], out["exch_m"], out["tke"],
                out["el"], out["hpbl"], out["kpbl"], out["wstar"],
                out["delta"], out["u10"], out["v10"],
                DTYPE(dt), DTYPE(dx), np.int32(flags),
                np.int32(nz), np.int32(ny), np.int32(nx),
                ws, np.int32(nz + 2), np.int32(lo)) + topo_args)
    return out


_SH480_VALIDATED = _SH480_3D_FLOAT_OUTPUTS + _SH480_2D_FLOAT_OUTPUTS


def invalid_shinhong480_outputs(
        out: Mapping[str, cp.ndarray], status: cp.ndarray) -> tuple[str, ...]:
    """Every non-finite :func:`launch_shinhong_480` float output, in order.

    One device census over the bundle and one scalar readback, the same
    cost as the 4.6.1 unit's :func:`invalid_shinhong_outputs`.
    """
    if (not isinstance(status, cp.ndarray) or status.shape != (1,)
            or status.dtype != cp.uint32 or not status.flags.c_contiguous):
        raise ValueError("Shin-Hong validation status must be a C-contiguous "
                         "uint32 device array with shape (1,)")
    status.fill(cp.uint32(0))
    count_3d = out["du"].size
    count_2d = out["hpbl"].size
    blocks = (max(count_3d, count_2d) + _VALIDATE_TPB - 1) // _VALIDATE_TPB
    kernel = get_kernel("shinhong_480", "shinhong480_validate_outputs")
    kernel((blocks,), (_VALIDATE_TPB,),
           tuple(out[name] for name in _SH480_VALIDATED)
           + (status, np.int64(count_3d), np.int64(count_2d)))
    invalid = int(status[0].item())
    return tuple(name for bit, name in enumerate(_SH480_VALIDATED)
                 if invalid & (1 << bit))
