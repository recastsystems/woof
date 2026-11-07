"""The operational HRRR fork's Thompson microphysics as one column kernel.

``RunConfig.thompson_version = "wrf_39_noaa"`` (mp_physics = 28) runs
woof/core/kernels/thompson_fork.cu: NOAA-EMC/HRRR v4.1.21's
module_mp_thompson.F (mp_gt_driver, mp_thompson, calc_refl10cm,
calc_effectRad) transcribed statement for statement, one thread per column,
in the Fortran's own tendency form and arithmetic.  Graded word for word
against the unmodified fork by tools/thompson_fork_oracle/
gpu_column_oracle.py and tests/test_thompson_fork_column_gpu.py.

The launcher takes the state's arrays in WOOF's (nz, ny, nx) layout, the
fork's table owner (woof.core.thompson_runtime.load_classic_device_tables
with version "wrf_39_noaa") and the CCN activation table, and writes every
output the fork's driver writes, with the effective radii in WOOF's micron
convention.

The fork's column arrays live in a device workspace (the kernel's column
arena), sized to the tile of threads the launcher keeps in flight rather
than held in the thread's local frame, which the driver reserves for the
card's whole resident-thread capacity.  ``workspace_bytes`` is the one
formula the launcher allocates from and woof.core.preflight prices.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MODULE = "thompson_fork"
KERNEL = "thompson_fork_column"
BLOCK = 64

#: The fork tables the kernel reads, in its argument order (names are the
#: Fortran's; woof.core.thompson_contract.FORK_TABLE_ASSETS holds them).
TABLE_ORDER = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    "tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz", "tpi_qcfz", "tni_qcfz",
    "tps_iaus", "tni_iaus", "tpi_ide", "t_Efrw", "t_Efsw", "tnc_wev",
)

#: The column arena's slots per thread, by element type: the kernel's
#: TF_WS_FLOATS / TF_WS_DOUBLES / TF_WS_BOOLS (19 driver arrays plus
#: mp_thompson's 60, which calc_refl10cm's 11 reuse after it returns; 7
#: double; 5 logical).  tests/test_thompson_fork_column.py holds these to
#: the source.  Each slot is nz + 1 levels.
WS_FLOATS = 79
WS_DOUBLES = 7
WS_BOOLS = 5

#: Blocks per multiprocessor in the launcher's tile.  At the unit's 244
#: registers a 64-thread block fits four times on an SM (sm_120, NVRTC
#: 12.9.86), so the tile is every block that can be resident at once; the
#: launcher takes the smaller of this and the driver's occupancy answer.
TILE_BLOCKS_PER_SM = 4


def _check_levels(nz: int) -> int:
    if int(nz) < 2:
        raise ValueError(f"thompson_fork needs at least 2 levels, got {nz}")
    return int(nz)


def tile_threads(multiprocessors: int, ncol: int,
                 blocks_per_sm: int = TILE_BLOCKS_PER_SM) -> int:
    """Threads the launcher keeps in flight: whole blocks, no more than the
    columns need and no more than the tile."""
    blocks = min((int(ncol) + BLOCK - 1) // BLOCK,
                 int(multiprocessors) * max(1, int(blocks_per_sm)))
    return max(1, blocks) * BLOCK


def workspace_layout(nz: int, threads: int) -> tuple[int, int, int]:
    """Byte sizes of the double, float and bool regions, in that order (the
    double region first keeps every region aligned)."""
    cells = (_check_levels(nz) + 1) * int(threads)
    return (WS_DOUBLES * cells * 8, WS_FLOATS * cells * 4, WS_BOOLS * cells)


def workspace_bytes(nz: int, threads: int) -> int:
    """Device bytes of the column arena for ``threads`` threads in flight."""
    return sum(workspace_layout(nz, threads))


#: The unit's compile options.  ``--fmad=false`` because gfortran -O2 on
#: x86-64 contracts no multiply-add, and ``--ftz=false`` because the fork's
#: REAL arithmetic keeps subnormal words, and CuPy's RawModule appends
#: ``-ftz=true`` after the caller's options.  Measured on the 126 oracle
#: columns with -ftz=true: 435 cells off the fork's words, up to 573 ULP.
#: The unit therefore compiles through NVRTC directly, as the legacy
#: RRTMG, RUC SPP and plume-rise units do.
MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")


@lru_cache(maxsize=None)
def _module(device: int):
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.core.kernels import MODULE_KEY_ROOT, module_source
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source(MODULE)
    key = f"{MODULE_KEY_ROOT}:{MODULE}"
    with cp.cuda.Device(device), observe_module_compile(key):
        binary, _ = compiler.compile_using_nvrtc(
            source, ("-std=c++17", "--fmad=false", "--ftz=false"), None,
            "thompson_fork.cu")
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    record_module(key, source=source, options=MODULE_OPTIONS, module=None)
    return module


def _kernel():
    import cupy as cp
    return _module(cp.cuda.runtime.getDevice()).get_function(KERNEL)


def _tile(kernel, ncol: int) -> int:
    import cupy as cp
    from cupy_backends.cuda.api import driver

    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    per_sm = TILE_BLOCKS_PER_SM
    try:
        from woof.backend import active_blocks_per_multiprocessor
        resident = active_blocks_per_multiprocessor(
            kernel.ptr, BLOCK, 0)
        per_sm = min(per_sm, int(resident))
    except Exception:  # noqa: BLE001  -- the tile bound alone still holds
        pass
    return tile_threads(sms, ncol, per_sm)


def launch_thompson_fork_column(
        th, pii, p, w, dz, qv, qc, qr, qi, qs, qg, ni, nr, nc, nwfa, nifa,
        nwfa2d, th_out, rainnc, rainncv, snownc, snowncv, graupelnc,
        graupelncv, sr, refl, effc, effi, effs, table_owner, ccn_table,
        dt: float, *, first_time_step: bool, do_refl: bool,
        singular_fall: bool) -> None:
    """One fork microphysics call over every column, in place.

    ``th``/``pii``/``p``/``dz`` and the species are (nz, ny, nx) float32;
    ``w`` has nz + 1 levels (the Fortran's w1d(k) is level k's lower face);
    the surface fields are (ny, nx).  ``th_out`` may alias ``th``.  ``refl``
    is written only when ``do_refl``.  ``singular_fall`` selects the fork's
    live melting-snow fall speed (thompson_fork_snow_fall = "wrf_39_noaa");
    false selects WRF v4.6.1's rain-share blend ("blend").
    """
    import cupy as cp

    nz, ny, nx = qv.shape
    ncol = ny * nx
    vol = (th, pii, p, dz, qv, qc, qr, qi, qs, qg, ni, nr, nc, nwfa, nifa,
           th_out, effc, effi, effs) + ((refl,) if do_refl else ())
    for a in vol:
        if a.shape != (nz, ny, nx) or a.dtype != np.float32 \
                or not a.flags.c_contiguous:
            raise ValueError("thompson_fork expects C-contiguous float32 "
                             f"(nz, ny, nx) arrays; got {a.shape} {a.dtype}")
    if w.shape != (nz + 1, ny, nx) or w.dtype != np.float32 \
            or not w.flags.c_contiguous:
        raise ValueError(f"thompson_fork expects w on nz + 1 faces; "
                         f"got {w.shape}")
    for a in (nwfa2d, rainnc, rainncv, snownc, snowncv, graupelnc,
              graupelncv, sr):
        if a.size != ncol or a.dtype != np.float32 \
                or not a.flags.c_contiguous:
            raise ValueError("thompson_fork expects (ny, nx) float32 "
                             "surface fields")
    tables = []
    for name in TABLE_ORDER:
        t = table_owner.arrays[name]
        if t.dtype != np.float64:
            raise ValueError(f"fork table {name} must be float64")
        tables.append(t)
    if ccn_table.dtype != np.float64 or ccn_table.size != 7 * 9 * 7 * 5 * 4:
        raise ValueError("the CCN activation table must be the float64 "
                         "(7, 9, 7, 5, 4) Fortran-order array")
    tables.append(ccn_table)
    refl_arg = refl if do_refl else effc
    kernel = _kernel()
    threads = _tile(kernel, ncol)
    d_bytes, f_bytes, b_bytes = workspace_layout(nz, threads)
    arena = cp.empty(d_bytes + f_bytes + b_bytes, dtype=cp.uint8)
    ws_d = arena[:d_bytes].view(cp.float64)
    ws_f = arena[d_bytes:d_bytes + f_bytes].view(cp.float32)
    ws_b = arena[d_bytes + f_bytes:].view(cp.bool_)
    kernel((threads // BLOCK,), (BLOCK,), (
        th, pii, p, w, dz, qv, qc, qr, qi, qs, qg, ni, nr, nc, nwfa, nifa,
        nwfa2d, th_out, rainnc, rainncv, snownc, snowncv, graupelnc,
        graupelncv, sr, refl_arg, effc, effi, effs, *tables,
        ws_f, ws_d, ws_b,
        np.int32(nz), np.int32(ncol), np.float32(dt),
        np.int32(1 if first_time_step else 0), np.int32(1 if do_refl else 0),
        np.int32(1 if singular_fall else 0)))


__all__ = ["BLOCK", "KERNEL", "MODULE", "TABLE_ORDER", "TILE_BLOCKS_PER_SM",
           "WS_BOOLS", "WS_DOUBLES", "WS_FLOATS", "launch_thompson_fork_column",
           "tile_threads", "workspace_bytes", "workspace_layout"]
