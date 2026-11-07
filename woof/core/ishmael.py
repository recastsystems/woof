"""ISHMAEL adaptive-habit ice microphysics (WRF ``mp_physics = 55``).

Jensen et al. 2017 (JAS), 2018 (MWR), 2018 (JAS): three ice species --
planar-nucleated, columnar-nucleated and aggregates -- each carrying mass,
number and two volume moments (a^2 c and c^2 a), so particle habit and
density evolve under vapour growth, riming and melting.  Rain mass and
number and cloud mass are predicted; cloud droplet number is a fixed
200 cm-3.

Transcription authority: WRF v4.7.1 ``phys/module_mp_jensen_ishmael.F``
(public domain), ported to ``woof/core/kernels/ishmael.cu``. Parallel cell
physics is followed by ordered column sedimentation, preserving the native
arithmetic order. A strict replay of a real native
forecast call matches all 213,156 output words (2026-11-01). The module and
tables have identical bytes in v4.6.1 and v4.7.1.

WHAT RUNS WHERE
---------------
Once per process: the three WRF lookup tables (``ishmael-gamma-tab.bin``,
``ishmael-qi-qc.bin``, ``ishmael-qi-qr.bin``) are read through the table
path in ``rw-tempo`` -- size, SHA-256 and every Fortran record marker
checked in Rust against :data:`ISHMAEL_TABLE_ASSETS` --
and uploaded; the aggregation collection tables WRF builds at init
(``mkcoltb``) are built on the card by ``ishk_coltab``. Per step,
``ishk_levels`` advances independent cells, then ``ishk_column`` carries out
ordered sedimentation and its final checks. No prognostic field leaves the card.

COMPILE
-------
Default numerics use CUDA elementary functions and allow fused arithmetic.
Both arms retain subnormal volume moments with ``--ftz=false`` through the
shared direct NVRTC loader. ``GPUWM_WRF_EXACT=1`` additionally disables
contraction and selects the strict elementary functions for native replay.

DECLARED DIVERGENCE D1 (default arm only):
WRF bounds ice-rain riming by the ice-cloud riming sum (module line 1105);
The default bounds it by the ice-rain sum the guard was written for.
Strict replay retains the pristine native limiter. The default arm completed
the paired three-hour real forecast; see the pinned oracle's qualification.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.core.thompson_contract import (
    TableAsset,
)

#: The three WRF v4.6.1 ``run/`` tables, pinned (git tag v4.6.1,
#: d66e442f).  Big-endian Fortran sequential records with 4-byte markers,
#: the layout WRF's BYTESWAPIO build reads.
ISHMAEL_TABLE_ASSETS = (
    TableAsset("ishmael-gamma-tab.bin", 6_060_012,
               "d85e4960053024a73034fb962ff52bd62aeae1b975dd5b4f62244ec82d622daa"),
    TableAsset("ishmael-qi-qc.bin", 23_346_576,
               "b8e4cea138b78668f027934bb8e39c6ea2cb62c1d277c17249c72003c8f4646c"),
    TableAsset("ishmael-qi-qr.bin", 46_693_152,
               "69dd41d3d285318c95cac0d2dde10f8a3b25f802bf52ac781f29c87adb964cc4"),
)
#: Records per file and values per record (jensen_ishmael_init :116-167).
_TABLE_LAYOUT = {
    "ishmael-gamma-tab.bin": (505_001, 1),
    "ishmael-qi-qc.bin": (51 * 51 * 51 * 11, 2),
    "ishmael-qi-qr.bin": (51 * 51 * 51 * 11, 6),
}
#: Where ``woof fetch-tables --ishmael`` downloads them: WRF's own
#: repository at the pinned tag.  The bytes are checked against the pins
#: above before they are installed, so the URL is a location, not trust.
TABLE_URL_BASE = "https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/run"
TABLE_ROOT_ENV = "WOOF_ISHMAEL_TABLE_ROOT"
USER_TABLE_ROOT_PARTS = (".woof", "tables", "ishmael")


def user_table_root() -> Path:
    """``~/.woof/tables/ishmael``, where ``woof fetch-tables --ishmael``
    stages the set (beside the Thompson set, for the same reason)."""
    return Path.home().joinpath(*USER_TABLE_ROOT_PARTS)


def table_root() -> Path:
    override = os.environ.get(TABLE_ROOT_ENV)
    return Path(override) if override else user_table_root()


def missing_tables(root: Path | None = None) -> list[str]:
    """The pinned files absent or mis-sized under ``root`` (no hashing)."""
    root = table_root() if root is None else Path(root)
    out = []
    for asset in ISHMAEL_TABLE_ASSETS:
        path = root / asset.filename
        if not path.is_file() or path.stat().st_size != asset.bytes:
            out.append(asset.filename)
    return out


def require_tables(root: Path | None = None) -> Path:
    """The table root, or a refusal naming the files and the staging command.

    The breakage this prevents: without the three tables the scheme has no
    riming, ice-rain or gamma-function lookups at all, and a run would die
    inside its first microphysics call after paying for preparation.
    """
    root = table_root() if root is None else Path(root)
    missing = missing_tables(root)
    if missing:
        raise FileNotFoundError(
            "mp_physics=55 (ISHMAEL) needs WRF v4.6.1's lookup tables "
            f"{', '.join(missing)} in {root}; stage them with "
            "`woof fetch-tables --ishmael` (or set "
            f"{TABLE_ROOT_ENV} to a WRF run/ directory holding the "
            "byte-identical files)")
    return root


@lru_cache(maxsize=4)
def load_tables(root: str | None = None) -> dict[str, np.ndarray]:
    """Host float32 tables in file order, validated against the pins."""
    from concurrent.futures import ThreadPoolExecutor
    from woof.core.wrf_table_rust import read_f32_records

    root_path = require_tables(None if root is None else Path(root))

    def read(asset):
        count, values = _TABLE_LAYOUT[asset.filename]
        return asset.filename, read_f32_records(
            root_path / asset.filename, sha256=asset.sha256,
            count=count, values=values)

    with ThreadPoolExecutor(max_workers=3) as workers:
        out = dict(workers.map(read, ISHMAEL_TABLE_ASSETS))
    return {"gtab": out["ishmael-gamma-tab.bin"].reshape(-1),
            "itab": out["ishmael-qi-qc.bin"].reshape(-1),
            "itabr": out["ishmael-qi-qr.bin"].reshape(-1)}


# ---------------------------------------------------------------------------
# Device side.
# ---------------------------------------------------------------------------
#: Prognostic and input fields, in the kernel's IF_* order.
FIELD_SLOTS = ("p", "dz", "th", "qv", "qc", "qr", "nr",
               "qi1", "qi2", "qi3", "ni1", "ni2", "ni3",
               "ai1", "ai2", "ai3", "ci1", "ci2", "ci3")
#: Diagnostics, in the kernel's ID_* order (WRF's RE_CLOUD, RE_ICE,
#: REFL_10CM, then VMI3D/DI3D/RHOPO3D/PHII3D/ITYPE for the three species).
DIAG_SLOTS = ("effc", "effi", "dbz",
              "vmi1", "vmi2", "vmi3", "di1", "di2", "di3",
              "rhopo1", "rhopo2", "rhopo3", "phii1", "phii2", "phii3",
              "itype1", "itype2", "itype3")
#: Level values carried from the level loop into sedimentation (IW_*).
SCRATCH_SLOTS = ("rho", "vtrm", "vtrn", "vtrmi1", "vtrmi2", "vtrmi3",
                 "vtrni1", "vtrni2", "vtrni3",
                 "level_sedi", "rhobar1", "rhobar2", "rhobar3")
SURF_SLOTS = ("rainnc", "rainncv", "snownc", "snowncv")

#: The compile site's options (see COMPILE in the module docstring).
def compile_options():
    from woof.core.kernels import module_options
    return module_options("ishmael")
_TPB = 64
_NDN = 60


from woof.core.device_cache import cuda_cache


@cuda_cache(maxsize=None)
def module(source: str | None = None):
    from woof.core.kernels import load_module, compile_ishmael_no_ftz
    if source is None:
        return load_module("ishmael")
    return compile_ishmael_no_ftz("ishmael-mutated", source, compile_options())


_TABLE_CACHE: dict = {}


def device_tables(root: str | None = None, *, source: str | None = None):
    """The lookup tables and the built collection tables, resident per card."""
    import cupy as cp
    from woof.core.device_cache import cached_ready

    key = (int(cp.cuda.Device().id), root, source)

    def upload():
        host = load_tables(root)
        tabs = {name: cp.asarray(np.ascontiguousarray(arr, np.float32))
                for name, arr in host.items()}
        tabs["coltab"] = cp.zeros(7 * _NDN * _NDN, dtype=cp.float32)
        tabs["coltabn"] = cp.zeros(7 * _NDN * _NDN, dtype=cp.float32)
        n = 7 * _NDN * _NDN
        module(source).get_function("ishk_coltab")(
            ((n + _TPB - 1) // _TPB,), (_TPB,), (tabs["coltab"], tabs["coltabn"]))
        return tabs

    return cached_ready(cp, _TABLE_CACHE, key, upload)


def run_column_step(fields: dict, diag: dict, surf: dict, scratch: dict, *,
                    dt: float, tables=None, source: str | None = None) -> None:
    """One MP_JENSEN_ISHMAEL call, in place, on device arrays.

    ``fields``/``diag``/``scratch`` map the slot names above to contiguous
    float32 cupy arrays of ``nz * ncol`` elements, level-major (woof's
    native ``(nz, ny, nx)`` storage is that with ncol = ny * nx); ``surf``
    to ``ncol`` elements.  ``p`` and ``dz`` are read only; every other field
    is updated as WRF updates it.
    """
    import cupy as cp

    nzcol = int(fields["p"].size)
    ncol = int(surf["rainnc"].size)
    if ncol < 1:
        raise ValueError("ISHMAEL needs at least one column")
    if nzcol % ncol:
        raise ValueError(f"ISHMAEL fields hold {nzcol} values, not a multiple "
                         f"of the {ncol} columns the surface fields describe")
    nz = nzcol // ncol
    if nz < 1:
        raise ValueError("ISHMAEL needs at least one vertical level")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    def check(group, names, size):
        for name in names:
            arr = group.get(name)
            if arr is None:
                raise ValueError(f"ISHMAEL launch is missing {name!r}")
            if (arr.dtype != cp.float32 or int(arr.size) != size
                    or not arr.flags.c_contiguous):
                raise ValueError(f"ISHMAEL array {name!r} must be contiguous "
                                 f"float32 with {size} elements")
    check(fields, FIELD_SLOTS, nzcol)
    check(diag, DIAG_SLOTS, nzcol)
    check(scratch, SCRATCH_SLOTS, nzcol)
    check(surf, SURF_SLOTS, ncol)
    if tables is None:
        tables = device_tables(source=source)
    kernels = module(source)
    args = (*[fields[n] for n in FIELD_SLOTS],
        *[diag[n] for n in DIAG_SLOTS],
        *[scratch[n] for n in SCRATCH_SLOTS],
        *[surf[n] for n in SURF_SLOTS],
        tables["itab"], tables["itabr"], tables["gtab"],
        tables["coltab"], tables["coltabn"],
        np.int32(ncol), np.int32(nz), np.float32(dt))
    kernels.get_function("ishk_levels")(
        ((nzcol + _TPB - 1) // _TPB,), (_TPB,), args)
    kernels.get_function("ishk_column")(
        ((ncol + _TPB - 1) // _TPB,), (_TPB,), args)


def reflectivity(state):
    """Native three-habit PSD radar diagnostic without advancing state."""
    import cupy as cp
    shape = state.p.shape
    th = state.scratch(shape, "mp_th")
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    if getattr(state, "use_theta_m", 0):
        th[...] = state.total_theta()
    else:
        cp.add(thb, state.thp, out=th)
    result = state.scratch(shape, "refl_10cm")
    names = ("qr", "nr", "qi", "qi2", "qi3", "ni", "ni2", "ni3",
             "qvoli", "qvoli2", "qvoli3", "qaoli", "qaoli2", "qaoli3")
    module().get_function("ishk_reflectivity")(
        ((result.size + 255) // 256,), (256,),
        (state.p, th, *[getattr(state, n) for n in names], device_tables()["gtab"],
         result, np.uint64(result.size)))
    return result
