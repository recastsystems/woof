"""The ACM2 (Pleim) PBL, ``bl_pbl_physics = 7``: compile site and launcher.

WRF v4.6.1 ``phys/module_bl_acm.F`` (ACMPBL -> ACM2D -> EDDYX, ACM, ACMM,
MATRIX, TRI) runs on the card as ``woof/core/kernels/acm2pbl.cu``, one
thread per mass column, graded word for word against the gfortran build of
the byte-pinned module that the combo-sweep referee ``wrf.exe`` links
(tools/acm2_wrf461_oracle, tests/test_acm2_wrf461_parity.py).

The compile goes straight through NVRTC with ``--ftz=false`` and, under
strict numerics, ``--fmad=false``. Fast numerics permits contraction in
the column solver. ``cupy.RawModule`` appends ``-ftz=true`` after the
caller's options and NVRTC honours the last one, which would flush the
subnormal tendencies a quiet column legitimately produces.

The host half is WRF's binding of the call (module_pbl_driver.F:1585-1611):
``XTIME`` is the one-based ``itimestep``, ``DTPBL`` the PBL interval,
``RR3D`` phy_prep's density and ``PBLH`` the carried PBL height the scheme
reads for the convective velocity scale before it writes the new one.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

#: Strict options. The shared loader applies the process numerics mode.
MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.acm2pbl:acm2pbl"
_SOURCES = ("glibc_flt32.cuh", "acm2pbl.cu")
#: Kernel status bit: WRF's wrf_error_fatal at module_bl_acm.F:464-470.
STATUS_RIB_NEVER_EXCEEDS_RIC = 1
_FINITE_FIELDS = ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "pblh",
                  "rmol", "regime", "exch_h", "exch_m")

#: Packaged scalar table, pinned through the existing immutable asset reader.
from woof.core.thompson_contract import TableAsset, read_validated_table_asset

CONSTANT_ASSET = TableAsset("acm2_constants.json", 1354, "58539e772ff5cd866c46f57744443053fa25cc118d47347b379db0203715c92f")


def constants_table(root=None):
    """Read the pinned constants once per call, refusing substituted bytes."""
    import json
    path = Path(__file__).parents[1] if root is None else Path(root)
    data = read_validated_table_asset(path, CONSTANT_ASSET,
        label="ACM2 constant asset",
        remedy="reinstall woof to restore its packaged ACM2 constants")
    return json.loads(data)


def _driver_constant(name):
    word = int(constants_table()["driver_words"][name], 16)
    return np.asarray([word], dtype=np.uint32).view(np.float32)[0]


G, R_D, R_V, CP, EP_1 = (_driver_constant(n)
                         for n in ("G", "R_D", "R_V", "CP", "EP_1"))


def module_source(*, kernel_text: str | None = None) -> str:
    """The exact source string NVRTC compiles.

    ``kernel_text`` replaces acm2pbl.cu's text (the planted-constant check
    in tests/test_acm2_wrf461_parity.py compiles an edited copy through
    this same assembly); ``None`` is the shipped kernel.
    """
    from woof.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    header = (root / _SOURCES[0]).read_text(encoding="utf-8")
    kernel = (kernel_text if kernel_text is not None
              else (root / _SOURCES[1]).read_text(encoding="utf-8"))
    from woof.core.physics_inventory import ACM2_BLOCK, ACM2WS_SLOTS

    defines = (f"#define ACM2_BLOCK {ACM2_BLOCK}\n"
               f"#define ACM2WS_SLOTS {ACM2WS_SLOTS}\n")
    table = constants_table()
    constants = "\n".join(
        f"#define {row['macro']:<11} {row['literal']}" for row in table["kernel"])
    # The pin is part of the compiler's source manifest even for a default run.
    from woof.numerics import is_strict
    arithmetic = ""
    if not is_strict():
        # CUDA's float32 power and contracted expressions are the default
        # forecast path. The native libm helper is needed only for proof.
        arithmetic = """
#define ACM2_FAST 1
#undef FADD
#undef FSUB
#undef FMUL
#undef FDIV
#undef FSQRT
#define FADD(a,b) ((a)+(b))
#define FSUB(a,b) ((a)-(b))
#define FMUL(a,b) ((a)*(b))
#define FDIV(a,b) ((a)/(b))
#define FSQRT(a) sqrtf(a)
#define gfk_pow(a,b) powf((a),(b))
"""
    return (_preamble() + header + arithmetic + defines
            + f"// ACM2 constant asset SHA-256 {CONSTANT_ASSET.sha256}\n"
            + constants + "\n" + kernel)


def compile_module(source: str, device: int):
    """Compile through the shared rounded-column loader."""
    from woof.core.kernels import compile_no_ftz

    return compile_no_ftz(source, module_key=MODULE_KEY,
                          filename="acm2pbl.cu", device=device)


@lru_cache(maxsize=None)
def _module(device: int):
    return compile_module(module_source(), device)


class AcmFatal(FloatingPointError):
    """WRF's own stop: the bulk Richardson number never reaches RIC."""


def _describe_status(word, plane, *, first=None, deferred=False):
    """Decode device flags without retaining a column's atmospheric arrays."""
    from woof.core.health_ledger import deferred_note
    note = deferred_note() if deferred else ""
    if word & STATUS_RIB_NEVER_EXCEEDS_RIC:
        where = ""
        if first is not None:
            idx = np.unravel_index(first, plane) if plane else ()
            where = f" in column {tuple(int(i) for i in idx)} (flat {first})"
        raise AcmFatal(
            "ACM2 PBL: the bulk Richardson number never exceeds RIC = 0.25"
            f"{where}; WRF stops the model here (module_bl_acm.F:464-470, "
            "wrf_error_fatal 'RIBX never exceeds RIC'): no level satisfies "
            f"the scheme's critical Richardson-number test{note}")
    for bit, name in enumerate(_FINITE_FIELDS, 1):
        if word & (1 << bit):
            raise FloatingPointError(
                f"the ACM2 PBL returned non-finite {name}{note}")


def acm2_pbl_step(*, u, v, theta, t, qv, qc, qi, rho, dz8w, ust, hfx, qfx,
                  wspd, pblh, regime, rmol, exch_h, exch_m, dtpbl: float,
                  xtime: int, ep1=EP_1, g=G, rd=R_D, cpd=CP, module=None):
    """One ACMPBL call over every column.

    3-D inputs are ``(nz, ...)`` float32 on the device, bottom level first,
    any trailing shape (the columns are its flattening); 2-D inputs carry
    that trailing shape.  ``pblh``, ``regime``, ``rmol``, ``exch_h`` and
    ``exch_m`` are updated IN PLACE, as WRF's INOUT arguments are
    (``exch_h``/``exch_m`` on levels 1..nz-1 only; the top level keeps its
    value, as WRF's kte does).  Returns the six tendencies (``du``, ``dv``,
    ``dtheta``, ``dqv``, ``dqc``, ``dqi``, m/s2, K/s, kg/kg/s, uncoupled)
    and ``kpbl`` (int32, one-based like WRF's KPBL2D).

    Raises :class:`AcmFatal` where WRF would stop. Standalone calls name
    the first column; model calls report at the next health-ledger drain.
    """
    import cupy as cp
    from woof.core.physics_inventory import (
        ACM2_BLOCK, ACM2_TILE_BLOCKS_PER_SM, acm2_workspace_floats)

    nz = int(theta.shape[0])
    if nz < 4:
        raise ValueError(
            f"the ACM2 PBL needs at least 4 mass levels, got {nz}: ACM reads "
            "AI(2) and DZFI(kte-1) on every column and its free-convection "
            "test needs THETAV(2) under a PBL top above level 3")
    plane = theta.shape[1:]
    ncol = int(np.prod(plane))

    def col3(a, name):
        a = cp.ascontiguousarray(a, dtype=cp.float32)
        if a.shape != theta.shape:
            raise ValueError(f"{name} has shape {a.shape}, "
                             f"expected {theta.shape}")
        return a

    def col2(a, name):
        if a.shape != plane or a.dtype != cp.float32 or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous float32 {plane}")
        return a

    fields3 = [col3(a, n) for a, n in ((u, "u"), (v, "v"), (theta, "theta"),
                                       (t, "t"), (qv, "qv"), (qc, "qc"),
                                       (qi, "qi"), (rho, "rho"),
                                       (dz8w, "dz8w"))]
    fields2 = [cp.ascontiguousarray(a, dtype=cp.float32).reshape(plane)
               for a in (ust, hfx, qfx, wspd)]
    for a, n in ((pblh, "pblh"), (regime, "regime"), (rmol, "rmol")):
        col2(a, n)
    for a, n in ((exch_h, "exch_h"), (exch_m, "exch_m")):
        if (a.shape[1:] != plane or a.shape[0] < nz - 1
                or a.dtype != cp.float32 or not a.flags.c_contiguous):
            raise ValueError(f"{n} must be contiguous float32 with at least "
                             f"{nz - 1} levels over {plane}")
    out = {name: cp.empty(theta.shape, dtype=cp.float32)
           for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi")}
    kpbl = cp.empty(plane, dtype=cp.int32)
    status = cp.zeros(2, dtype=cp.int32)
    status[1] = np.iinfo(np.int32).max
    device = cp.cuda.Device()
    if module is None:
        module = _module(device.id)
    kernel = module.get_function("acm2_pbl_column")
    # One tile of columns in flight at a time, so the workspace is the
    # tile's (woof.core.preflight.acm2pbl_column_workspace_bytes prices
    # exactly this), not the domain's.
    tile = (int(device.attributes["MultiProcessorCount"])
            * ACM2_TILE_BLOCKS_PER_SM * ACM2_BLOCK)
    tile = min(tile, ncol)
    ws = cp.empty(acm2_workspace_floats(nz, tile), dtype=cp.float32)
    for col0 in range(0, ncol, tile):
        blocks = (min(tile, ncol - col0) + ACM2_BLOCK - 1) // ACM2_BLOCK
        kernel((blocks,), (ACM2_BLOCK,), (
            *fields3, *fields2, pblh, regime, rmol, kpbl, exch_h, exch_m,
            out["du"], out["dv"], out["dtheta"], out["dqv"], out["dqc"],
            out["dqi"], status[0:1], status[1:2], ws, np.int32(col0),
            np.int32(ncol), np.int32(nz), np.float32(dtpbl),
            np.int32(xtime), np.float32(ep1), np.float32(g), np.float32(rd),
            np.float32(cpd)))
    from woof.core.health_ledger import read_status
    word = read_status(status, site="acm2:column",
                       describe=lambda flags: _describe_status(
                           flags, plane, deferred=True))
    if word:
        # Only a standalone failing call reads the column index. Production
        # steps accumulate flags on the GPU and report at the model drain.
        first = (int(status[1].item())
                 if word & STATUS_RIB_NEVER_EXCEEDS_RIC else None)
        _describe_status(word, plane, first=first)
    out["kpbl"] = kpbl
    return out
