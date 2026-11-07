"""YSU planetary-boundary-layer column kernel.

This module selects the corrected WRF 4.8 generation by default, or the
exported engine's explicit WRF 4.6.1 reproduction path. The scheme consumes mass-point wind, potential
temperature, moisture, pressure/Exner, layer depths, and the surface-layer
coupling fields (UST, HFX/QFX, PSIM/PSIH).  It returns tendencies rather than
updating the state; Phase 3 Task 12's physics driver owns their application.

Array layout is woof's standard ``(nz, ny, nx)`` with x fastest.  Every CUDA
thread owns one complete column and performs the implicit vertical solves
in-thread, matching the acoustic Thomas-solve pattern. The independent
referee is the pinned public Fortran scheme driven by
``tools/ysu_generations_oracle``; the NumPy mirror is a secondary check.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import cupy as cp

from woof.core.kernels import get_kernel, get_kernel_int_defines
# The workspace constants and the pricing function live in the
# runtime-free inventory so the VRAM estimator can read them without
# cupy resolving; this module is their launcher-side re-export.  The
# measured history behind them -- the 9,232 B local frame that took
# 844.0 MiB of per-context backing store at first launch -- is recorded
# beside the constants there.
from woof.core.physics_inventory import (YSUWS_SLOTS, YSU_BLOCK,
                                          YSU_TILE_BLOCKS_PER_SM,
                                          ysu_workspace_floats)
from woof.core.state import DTYPE
from woof.ysu_generations import YSU_DEFAULT_GENERATION, ysu_generation, ysu_kernel_selection
from woof.physics_vertical_contract import (
    outside_vertical_bounds, refuse_vertical_levels)

_VALIDATE_TPB = 256
_KMAX = 128


def ysu_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The whole point of the workspace is that it is charged per thread IN
    FLIGHT where the local-memory backing store was charged per thread the
    card could ever hold.  Over-sizing the tile hands that ratio back.
    """
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = YSU_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        # A RawKernel wraps its CUDA function (``.kernel``); the shared
        # no-FTZ loader the WRF 4.8 generation compiles through hands back
        # the function itself.  Ask whichever it is, so the default
        # generation's tile is bounded by the card's occupancy too.
        from woof.backend import active_blocks_per_multiprocessor
        resident = active_blocks_per_multiprocessor(
            getattr(fn, "kernel", fn).ptr, YSU_BLOCK, 0)
        # Never launch more blocks than the card can hold resident: those
        # columns would wait while their workspace slots stayed allocated.
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    # Floored at one block.  The tile is the STEP of the launcher's loop, so
    # a zero here would be a zero-step ``range`` -- a ValueError instead of
    # the empty launch a zero-column domain should produce.
    return int(max(YSU_BLOCK, min(int(ncol), sms * per_sm * YSU_BLOCK)))

#: The launcher's first-call vertical bound, restated for front doors by
#: ``woof.physics_vertical_contract.YSU_VERTICAL_LEVEL_BOUNDS`` and bound to
#: it by ``tests/test_pbl_vertical_bounds.py``.
VERTICAL_LEVEL_BOUNDS = (4, _KMAX)

_YSU_3D_FLOAT_OUTPUTS = (
    "du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "exch_m",
)
_YSU_2D_OUTPUTS = (
    "hpbl", "kpbl", "wstar", "delta", "topdown_radsum", "wstar3_2",
    "cloudflg",
)
_YSU_OUTPUTS = _YSU_3D_FLOAT_OUTPUTS + _YSU_2D_OUTPUTS
_YSU_2D_FLOAT_OUTPUTS = tuple(
    name for name in _YSU_2D_OUTPUTS if name not in ("kpbl", "cloudflg"))

#: The BEP/BEP+BEM source terms WRF's flag_bep arm reads (bl_ysu.F90:450-497),
#: in ``ysu_column_bep`` argument order.  ``sf_bep`` may carry nz+1 levels
#: (the UrbanState array does); only the first nz are read, as
#: module_bl_ysu.F:360-375 copies kts:kte.  a_e/b_e/dlg/dl_u are part of the
#: handoff but YSU never reads them.
YSU_BEP_TERMS = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "b_u_bep",
                 "b_v_bep", "b_t_bep", "b_q_bep", "sf_bep", "vl_bep")


def launch_ysu(u, v, theta, qv, qc, qi, p, p_interface, exner, dz,
               rthraten=None, *,
               psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland,
               u10, v10, dt: float, ysu_topdown_pblmix: int = 1,
               ysu_version: str = YSU_DEFAULT_GENERATION,
               bep: Mapping[str, cp.ndarray] | None = None,
               frc_urb2d: cp.ndarray | None = None,
               topo: tuple[cp.ndarray, cp.ndarray] | None = None,
               u10_out: cp.ndarray | None = None,
               v10_out: cp.ndarray | None = None):
    """Launch YSU for a batch of device columns and return device outputs.

    Three-dimensional inputs have shape ``(nz, ny, nx)`` except
    ``p_interface`` which is ``(nz+1, ny, nx)``.  Surface inputs have shape
    ``(ny, nx)``.  All are C-contiguous float32 CuPy arrays.  The returned
    dict contains 3-D ``du/dv/dtheta/dqv/dqc/dqi/exch_h/exch_m``, 2-D
    ``hpbl/wstar/delta`` float32 arrays, and 2-D one-based ``kpbl`` int32.

    ``bep`` (sf_urban_physics 2/3) is ``UrbanState.pbl_terms``: WRF's
    flag_bep arm then runs (``ysu_column_bep``).  ``frc_urb2d`` is still
    required and shape-checked, but the kernel no longer reads it: WRF
    removes only the urban fraction of YSU's own surface drag while the BEP
    couple already carries the rural drag in ``a_u_bep``, so woof removes
    all of it (a declared divergence, ``kernels/ysu.cu`` and
    ``tests/test_ysu_bep_rural_drag.py``).  ``bep=None`` launches
    ``ysu_column`` exactly as before.

    ``topo`` (WRF ``topo_wind`` = 1 or 2) is ``(ctopo, ctopo2)``, the two
    (ny, nx) coefficients of :func:`woof.core.terrain_drag.topo_wind_coefficients`:
    the kernel then takes WRF's ctopo-present arm (``ysu_column_topo``,
    bl_ysu.F90:1254-1314) and writes the blended 10 m wind of :1402-1408
    into ``u10_out``/``v10_out``, which the caller supplies; ``u10``/``v10``
    are read, never written, so the returned dict keeps its layout.  It
    does not combine with ``bep``.
    """
    generation = ysu_generation(ysu_version)
    columns = {"u": u, "v": v, "theta": theta, "qv": qv, "qc": qc,
               "qi": qi, "p": p, "exner": exner, "dz": dz}
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"theta must have shape (nz, ny, nx), got {shape}")
    nz, ny, nx = shape
    if outside_vertical_bounds(nz, VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "YSU PBL", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "one CUDA thread owns a whole column and holds it in "
                f"per-thread local memory at YSU_KMAX={_KMAX} "
                "(kernels/ysu.cu), and the counter-gradient and entrainment "
                "layers the scheme indexes need four levels to exist."))
    for name, arr in columns.items():
        if not isinstance(arr, cp.ndarray) or arr.shape != shape:
            raise ValueError(f"{name} must be a CuPy array with shape {shape}")
        if arr.dtype != DTYPE or not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32")
    if rthraten is None:
        rthraten = cp.zeros_like(theta)
    elif (not isinstance(rthraten, cp.ndarray) or rthraten.shape != shape
          or rthraten.dtype != DTYPE or not rthraten.flags.c_contiguous):
        raise ValueError("rthraten must be C-contiguous float32 with shape "
                         f"{shape}")
    if (not isinstance(p_interface, cp.ndarray)
            or p_interface.shape != (nz + 1, ny, nx)
            or p_interface.dtype != DTYPE
            or not p_interface.flags.c_contiguous):
        raise ValueError("p_interface must be C-contiguous float32 with shape "
                         f"{(nz + 1, ny, nx)}")
    surfaces = {"psfc": psfc, "znt": znt, "ust": ust, "hfx": hfx,
                "qfx": qfx, "wspd": wspd, "br": br, "psim": psim,
                "psih": psih, "xland": xland, "u10": u10, "v10": v10}
    for name, arr in surfaces.items():
        if (not isinstance(arr, cp.ndarray) or arr.shape != (ny, nx)
                or arr.dtype != DTYPE or not arr.flags.c_contiguous):
            raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                             f"array with shape {(ny, nx)}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"YSU requires a positive finite dt, got {dt}")
    if ysu_topdown_pblmix not in (0, 1, False, True):
        raise ValueError("ysu_topdown_pblmix must be 0 or 1")

    bep_args = ()
    if bep is not None:
        missing = [name for name in YSU_BEP_TERMS if name not in bep]
        if missing:
            raise ValueError(f"bep is missing {missing}")
        for name in YSU_BEP_TERMS:
            arr = bep[name]
            want = (nz + 1, ny, nx) if name == "sf_bep" else shape
            if (not isinstance(arr, cp.ndarray) or arr.dtype != DTYPE
                    or not arr.flags.c_contiguous
                    or arr.shape not in (shape, want)):
                raise ValueError(
                    f"bep[{name!r}] must be C-contiguous float32 with shape "
                    f"{want}")
        if (not isinstance(frc_urb2d, cp.ndarray)
                or frc_urb2d.shape != (ny, nx) or frc_urb2d.dtype != DTYPE
                or not frc_urb2d.flags.c_contiguous):
            raise ValueError(
                "bep needs frc_urb2d, C-contiguous float32 with shape "
                f"{(ny, nx)}: bl_ysu.F90:1313 removes the urban fraction of "
                "the surface drag, and without it that drag is counted twice")
        bep_args = tuple(bep[name] for name in YSU_BEP_TERMS) + (frc_urb2d,)
    elif frc_urb2d is not None:
        raise ValueError("frc_urb2d is only read with bep")
    topo_args = ()
    if topo is not None:
        if bep_args:
            raise ValueError(
                "topo_wind does not combine with the BEP arm: ysu_column_bep "
                "removes YSU's whole own surface drag (the declared rural-drag "
                "divergence), the drag ctopo scales")
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

    out = {name: cp.empty_like(theta) for name in _YSU_3D_FLOAT_OUTPUTS}
    out.update(hpbl=cp.empty((ny, nx), dtype=DTYPE),
               kpbl=cp.empty((ny, nx), dtype=cp.int32),
               wstar=cp.empty((ny, nx), dtype=DTYPE),
               delta=cp.empty((ny, nx), dtype=DTYPE),
               topdown_radsum=cp.empty((ny, nx), dtype=DTYPE),
               wstar3_2=cp.empty((ny, nx), dtype=DTYPE),
               cloudflg=cp.empty((ny, nx), dtype=cp.int32))
    ncol = ny * nx
    from woof.wrf_exact import ENABLED
    module, defines = ysu_kernel_selection(ysu_version, strict=ENABLED)
    entry = "ysu_column_bep" if bep_args else "ysu_column_topo" if topo_args else "ysu_column"
    kernel = (get_kernel_int_defines(module, entry, defines) if defines
              else get_kernel(module, entry))
    # The column arrays live in a global workspace sized to the threads in
    # flight, not in the per-thread local frame (which CUDA prices at the
    # card's whole resident-thread capacity).  Columns therefore go in
    # tiles of `tile`, each launched with the SAME one-thread-one-column
    # mapping the kernel has always had -- the tile is passed as a column
    # OFFSET rather than as a slice, because the (nz, ny, nx) fields make a
    # column tile non-contiguous, so the kernel indexes them exactly as
    # before.
    tile = ysu_tile_columns(kernel, ncol)
    wskp = nz + 1
    ws = cp.empty(ysu_workspace_floats(nz, tile), dtype=DTYPE)
    try:
        for lo in range(0, ncol, tile):
            count = min(tile, ncol - lo)
            blocks = (count + YSU_BLOCK - 1) // YSU_BLOCK
            kernel((blocks,), (YSU_BLOCK,),
                   (u, v, theta, qv, qc, qi, p, p_interface, exner, dz,
                    rthraten, psfc, znt, ust, hfx, qfx, wspd, br, psim,
                    psih, xland, u10, v10,
                    out["du"], out["dv"], out["dtheta"], out["dqv"],
                    out["dqc"], out["dqi"], out["hpbl"], out["kpbl"],
                    out["exch_h"], out["exch_m"], out["wstar"],
                    out["delta"], DTYPE(dt),
                    out["topdown_radsum"], out["wstar3_2"],
                    out["cloudflg"],
                    np.int32(int(ysu_topdown_pblmix)),
                    np.int32(nz), np.int32(ny), np.int32(nx),
                    ws, np.int32(wskp), np.int32(lo)) + bep_args
                   + topo_args)
    finally:
        del ws
    return out


def _launch_output_validation(out, status, *, scheme, names, level_names,
                              surface_names, tpb, load_kernel):
    """Check the common PBL output layout and launch its validation kernel."""
    if tuple(out) != names:
        raise ValueError(
            f"native {scheme} validation requires outputs in launcher order "
            f"{names}, got {tuple(out)}")
    if (not isinstance(status, cp.ndarray) or status.shape != (1,)
            or status.dtype != cp.uint32 or not status.flags.c_contiguous):
        raise ValueError(
            f"{scheme} validation status must be a C-contiguous uint32 device "
            "array with shape (1,)")

    shape_3d = out["du"].shape
    shape_2d = out["hpbl"].shape
    if len(shape_3d) != 3 or len(shape_2d) != 2:
        raise ValueError(f"native {scheme} outputs must retain 3-D/2-D shapes")
    for group, shape in ((level_names, shape_3d), (surface_names, shape_2d)):
        for name in group:
            value = out[name]
            if (not isinstance(value, cp.ndarray) or value.shape != shape
                    or value.dtype != DTYPE or not value.flags.c_contiguous):
                raise ValueError(
                    f"native {scheme} output {name} must be C-contiguous float32 "
                    f"with shape {shape}")

    status.fill(cp.uint32(0))
    count_3d = out["du"].size
    count_2d = out["hpbl"].size
    blocks = (max(count_3d, count_2d) + tpb - 1) // tpb
    key = "ysu" if scheme == "YSU" else "shinhong"
    kernel = load_kernel(key + "_validation", key + "_validate_outputs")
    kernel(
        (blocks,), (tpb,),
        tuple(out[name] for name in level_names)
        + tuple(out[name] for name in surface_names)
        + (status, np.int64(count_3d), np.int64(count_2d)))


def validate_ysu_outputs(
        out: Mapping[str, cp.ndarray], status: cp.ndarray,
        *, refuse=None) -> str | None:
    """Return the first non-finite native YSU output name, or ``None``.

    ``launch_ysu`` owns this exact output layout.  One validation kernel
    records a bit per floating-point output and one scalar readback preserves
    the driver's historical first-invalid error ordering.  The two integer
    outputs are necessarily finite and need no device work.

    ``refuse`` opts this site into :mod:`woof.core.health_ledger`.  It is
    the caller's own refusal -- the closure that builds the forensic message
    -- and with a ledger active it is called at the drain with the first
    flagged name, preserving the historical first-invalid ordering because
    the bit order is the launcher order.  Without it the word is read here.
    """
    _launch_output_validation(
        out, status, scheme="YSU", names=_YSU_OUTPUTS,
        level_names=_YSU_3D_FLOAT_OUTPUTS, surface_names=_YSU_2D_FLOAT_OUTPUTS,
        tpb=_VALIDATE_TPB, load_kernel=get_kernel)
    from woof.core import health_ledger

    def _first_name(flags: int) -> str | None:
        for bit, name in enumerate(_YSU_OUTPUTS):
            if flags & (1 << bit):
                return name
        return None

    invalid = health_ledger.read_status(
        status, site="ysu",
        describe=None if refuse is None
        else lambda flags: refuse(_first_name(flags)))
    return _first_name(invalid)
