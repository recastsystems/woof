"""NCEP shallow convection, NSCV (``shcu_physics = 4``): launch and slot.

The CUDA column source was recovered from the October port. WRF 4.7.1
ships the same native module bytes as WRF 4.6.1. The real-input
4.7.1 replay is tools/shallow_wrf471_replay.py; its capture and
native diagnostic witness recipes are documented in
docs/shallow-wrf471.md.

What the slot hands the scheme is what WRF's shallow-cumulus driver hands
it (``module_first_rk_step_part1.F:1569-1662`` into
``module_shallowcu_driver.F:572-596``): the phy_prep hydrostatic pressures
``p_hyd``/``p_hyd_w``, ``pi_phy``, ``t_phy``, moist ``rho``, the A-grid
winds, ``dz8w``, the current ``w``, the PBL's HPBL/HFX/QFX, XLAND, this
step's RAINCV from the deep scheme, ``mp_physics`` (which fixes ncloud) and
``sas_pgcon``.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float32

#: Must equal kernels/shcu_nscv.cu's NSCV_SLOTS; the parity test re-reads
#: the .cu source and fails if either side moves alone.
NSCV_SLOTS = 22
NSCV_LANES = 32
_TPB = 128
#: Blocks kept in flight per SM: the workspace is charged per column in
#: flight (kf.py's argument), so the tile fills the card and no more.
NSCV_TILE_BLOCKS_PER_SM = 2

#: kernel status bits
STATUS_TOP_TOO_LOW = 1
STATUS_NONFINITE = 2

#: The scheme's level floor.  nscv2d reads levels up to kmax+1 and needs
#: kbm < kte, which a column cannot satisfy with fewer levels than this
#: under any model top; below it every column is refused (status bit 1).
VERTICAL_LEVEL_BOUNDS = (8, None)


def nscv_ncloud(mp_physics: int) -> int:
    """shcu_nscv's microphysics -> ncloud rule (module_shcu_nscv.F:130-136).

    0 detrains no condensate, 1 detrains it all as cloud water, 2 splits it
    between cloud water and cloud ice at the 263.16 K / 233.16 K ramp.
    """
    mp = int(mp_physics)
    if mp == 0:
        return 0
    if mp in (1, 3):
        return 1
    return 2


def nscv_constants() -> dict[str, np.float32]:
    """The constants WRF's shallow-cumulus driver hands the scheme.

    ``share/module_model_constants.F`` folds them at compile time, one
    correctly rounded float32 operation at a time; this repeats those
    operations in that order.  The oracle records the words gfortran folded
    and the parity test compares them with these before any output.
    """
    f = DTYPE
    r_d = f(287.0)
    r_v = f(461.6)
    return {
        "cp": f(f(f(7.0) * r_d) / f(2.0)),
        "cliq": f(4190.0),
        "cpv": f(f(4.0) * r_v),
        "g": f(9.81),
        "xlv": f(2.5e6),
        "r_d": r_d,
        "r_v": r_v,
        "ep_1": f(f(r_v / r_d) - f(1.0)),
        "ep_2": f(r_d / r_v),
        "cice": f(2106.0),
        "xls": f(2.85e6),
        "psat": f(610.78),
    }


def nscv_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more."""
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = NSCV_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            fn.kernel.ptr, _TPB, 0)
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass
    per_sm = max(1, int(per_sm))
    return int(min(int(ncol), sms * per_sm * _TPB))


def nscv_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``."""
    blocks = (int(columns) + _TPB - 1) // _TPB
    return blocks * NSCV_SLOTS * int(nz) * _TPB


def _describe_status(flags: int) -> None:
    if flags & STATUS_TOP_TOO_LOW:
        raise ValueError(
            "shcu_physics=4 (NSCV): a column's pressure one level below the "
            "model top is still above 0.7 of its surface pressure (or the "
            "top level's above 0.6), so nscv2d's cloud-top search would "
            "read level kte+1, past WRF's own column arrays "
            "(module_shcu_nscv.F:512-513, :1097).  Lower p_top for this "
            "domain.")
    if flags & STATUS_NONFINITE:
        raise FloatingPointError(
            "shcu_physics=4 (NSCV) produced a non-finite tendency, PRATESH, "
            "HBOT or HTOP")


def launch_shcu_nscv(*, t, qv, qc, qi, p, p8w, pi, rho, u, v, dz8w, w,
                     raincv, xland, hpbl, hfx, qfx, hbot, htop,
                     dt: float, mp_physics: int, pgcon: float,
                     f_qc: bool, f_qi: bool, out=None,
                     check_status: bool = True):
    """Run NSCV over FP32 device columns; return its WRF-named outputs.

    Level arrays are ``(nz, ny, nx)``; ``p8w`` and ``w`` are
    ``(nz + 1, ny, nx)``; the surface inputs ``(ny, nx)``.  ``hbot`` and
    ``htop`` are updated IN PLACE for the columns that enter the scheme and
    left as they arrived elsewhere (the kernel header's divergence 1).
    ``out`` may supply the six tendency arrays and ``pratesh`` to write into.
    """
    import cupy as cp

    shape = t.shape
    if len(shape) != 3:
        raise ValueError(f"t must have shape (nz, ny, nx), got {shape}")
    nz, ny, nx = shape
    if nz < VERTICAL_LEVEL_BOUNDS[0]:
        raise ValueError(
            f"shcu_physics=4 (NSCV) needs at least {VERTICAL_LEVEL_BOUNDS[0]} "
            f"levels, got {nz}")
    level = {"t": t, "qv": qv, "qc": qc, "qi": qi, "p": p, "pi": pi,
             "rho": rho, "u": u, "v": v, "dz8w": dz8w}
    stag = {"p8w": p8w, "w": w}
    surface = {"raincv": raincv, "xland": xland, "hpbl": hpbl, "hfx": hfx,
               "qfx": qfx, "hbot": hbot, "htop": htop}
    for group, want in ((level, shape), (stag, (nz + 1, ny, nx)),
                        (surface, (ny, nx))):
        for name, array in group.items():
            if (not isinstance(array, cp.ndarray) or array.shape != want
                    or array.dtype != DTYPE
                    or not array.flags.c_contiguous):
                raise ValueError(f"NSCV {name} must be a C-contiguous float32 "
                                 f"CuPy array of shape {want}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"NSCV dt must be positive and finite, got {dt}")
    if out is None:
        out = {}
    for name in ("rthshten", "rqvshten", "rqcshten", "rqishten", "rushten",
                 "rvshten"):
        if name not in out:
            out[name] = cp.empty(shape, dtype=DTYPE)
    if "pratesh" not in out:
        out["pratesh"] = cp.empty((ny, nx), dtype=DTYPE)
    for name in ("rthshten", "rqvshten", "rqcshten", "rqishten", "rushten", "rvshten", "pratesh"):
        want = (ny, nx) if name == "pratesh" else shape
        array = out[name]
        if (not isinstance(array, cp.ndarray) or array.shape != want
                or array.dtype != DTYPE or not array.flags.c_contiguous):
            raise ValueError(f"NSCV output {name} must be C-contiguous float32 of shape {want}")
    if not np.isfinite(pgcon):
        raise ValueError("NSCV pgcon must be finite")
    status = cp.zeros((1,), dtype=cp.uint32)

    from woof.core.kernels import get_kernel

    kernel = get_kernel("shcu_nscv", "shcu_nscv_column")
    k = nscv_constants()
    ncol = ny * nx
    tile = nscv_tile_columns(kernel, ncol)
    ws = cp.empty(nscv_workspace_floats(nz, tile), dtype=DTYPE)
    for col0 in range(0, ncol, tile):
        span = min(tile, ncol - col0)
        blocks = (span + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,), (
            t, qv, qc, qi, p, p8w, pi, rho, u, v, dz8w, w,
            raincv, xland, hpbl, hfx, qfx,
            out["rthshten"], out["rqvshten"], out["rqcshten"],
            out["rqishten"], out["rushten"], out["rvshten"],
            out["pratesh"], hbot, htop, status, ws,
            DTYPE(dt), DTYPE(pgcon), np.int32(nscv_ncloud(mp_physics)),
            np.int32(bool(f_qc)), np.int32(bool(f_qi)),
            k["cp"], k["cliq"], k["cpv"], k["g"], k["xlv"], k["r_d"],
            k["r_v"], k["ep_1"], k["cice"], k["xls"], k["psat"],
            np.int32(nz), np.int32(ncol), np.int32(col0)))
    del ws
    out["status"] = status
    if check_status:
        from woof.core import health_ledger

        flags = health_ledger.read_status(status, site="shcu_nscv",
                                          describe=_describe_status)
        _describe_status(flags)
    return out


__all__ = ["NSCV_SLOTS", "STATUS_NONFINITE", "STATUS_TOP_TOO_LOW",
           "VERTICAL_LEVEL_BOUNDS", "launch_shcu_nscv", "nscv_constants",
           "nscv_ncloud", "nscv_tile_columns", "nscv_workspace_floats"]


def nscvinit(out, *, restart=False, f_qc=True, f_qi=True):
    """WRF nscvinit: fresh tendencies zero; restart and absent species untouched."""
    if restart:
        return
    for name in ("rthshten", "rqvshten", "rushten", "rvshten"):
        out[name][...] = DTYPE(0.0)
    for enabled, name in ((f_qc, "rqcshten"), (f_qi, "rqishten")):
        if enabled:
            out[name][...] = DTYPE(0.0)
