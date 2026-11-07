"""CUDA wrappers for the pinned WRF v4.6.1 Noah-MP *thermal* leaf group.

One thread per column/case.  The device layout is the flat slot layout the
oracle harness packs, so ``woof/data/noahmp/oracle/noahmp-thermal.csv`` is
replayed slot for slot with no repacking on either side.

These wrappers are used by both the leaf parity tests and the vectorised
Noah-MP runtime, including the inherited thermoprop export. Scheme-4 memory
admission requires a measurement of this exact composition on the target.

``noahmp_thermal.cu`` is compiled **after** ``noahmp_leaves.cu``.  PHASECHANGE
and FRH2O need glibc 2.39's ``powf`` and ``logf`` on the device, and there must
be exactly one transcription of those: two copies could drift and only one of
them would be audited against ``glibc-libm-fp32.csv``.  Composing the two
sources rather than duplicating the tables is what keeps that single copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from woof.core.noahmp_driver_gpu import _current_device
from woof.core.noahmp_kernel_sources import (
    DEFAULT_OPTIONS, compile_runtime_unit, runtime_unit)


@dataclass(frozen=True)
class ThermalLayout:
    """Flat slot counts for one thermal leaf, matching run_thermal.F90."""

    n_int: int
    n_in: int
    n_out: int


NSNOW = 3
NSOIL = 4
NLAY = NSNOW + NSOIL

THERMAL_LAYOUTS: dict[str, ThermalLayout] = {
    "hrt": ThermalLayout(n_int=1, n_in=5 * NLAY + 4, n_out=4 * NLAY + 1),
    "hstep": ThermalLayout(n_int=1, n_in=5 * NLAY + 1, n_out=5 * NLAY),
    "tsnosoi": ThermalLayout(n_int=5, n_in=5 * NLAY + 7, n_out=NLAY + 1),
    "phasechange": ThermalLayout(
        n_int=4,
        n_in=4 * NLAY + 2 * NSNOW + 2 * NSOIL + 3 + 3 * NSOIL,
        n_out=NLAY + 2 * NSNOW + 2 * NSOIL + 4 + NLAY),
    "frh2o": ThermalLayout(n_int=0, n_in=6, n_out=1),
}

_THREADS = 64


def thermal_source() -> str:
    """The exact translation unit the thermal kernels are compiled from."""
    return runtime_unit("noahmp_thermal").source


@lru_cache(maxsize=None)
def _module(device: int, options: tuple[str, ...], log10_profile="glibc_2_39"):
    return compile_runtime_unit(
        "noahmp_thermal", module_key="woof.core.noahmp_thermal_gpu:thermal",
        options=options,log10_profile=log10_profile)


def thermal_module(options: tuple[str, ...] = DEFAULT_OPTIONS, *, log10_profile="glibc_2_39"):
    """Compile (once per option set) the composed thermal translation unit."""
    return _module(_current_device(), tuple(options),log10_profile)


def evaluate_thermal(leaf: str, x, ix=None):
    """Run one Noah-MP thermal leaf kernel over a batch of columns.

    ``x`` is ``(ncase, n_in)`` FP32 and ``ix`` is ``(ncase, n_int)`` int32.
    Returns a device array of shape ``(ncase, n_out)``, FP32.
    """
    import cupy as cp

    try:
        layout = THERMAL_LAYOUTS[leaf]
    except KeyError:
        raise KeyError(
            f"unknown Noah-MP thermal leaf {leaf!r}; ported leaves are "
            f"{sorted(THERMAL_LAYOUTS)}") from None

    host_x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if host_x.ndim != 2 or host_x.shape[1] != layout.n_in:
        raise ValueError(
            f"{leaf}: expected x with shape (ncase, {layout.n_in}), "
            f"got {host_x.shape}")
    ncase = host_x.shape[0]

    if layout.n_int == 0:
        host_ix = np.zeros((ncase, 1), dtype=np.int32)
    else:
        if ix is None:
            raise ValueError(f"{leaf}: needs an integer topology vector")
        host_ix = np.ascontiguousarray(np.asarray(ix, dtype=np.int32))
        if host_ix.shape != (ncase, layout.n_int):
            raise ValueError(
                f"{leaf}: expected ix with shape ({ncase}, {layout.n_int}), "
                f"got {host_ix.shape}")

    device_x = cp.asarray(host_x)
    device_ix = cp.asarray(host_ix)
    device_y = cp.zeros((ncase, layout.n_out), dtype=cp.float32)

    kernel = thermal_module().get_function(f"noahmp_thermal_{leaf}")
    blocks = (ncase + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (device_x, device_ix, device_y, np.int32(ncase)))
    return device_y


# ---------------------------------------------------------------------------
# Runtime batches
# ---------------------------------------------------------------------------
# Everything below turns a *physical* paused call from
# ``woof.core.noahmp_energy.energy_steps`` into the flat row the kernels above
# already take, and nothing else.  No arithmetic is added, no kernel is
# recompiled with a different option set, and the flat layouts are the ones
# ``tools/noahmp_wrf461_oracle/validate_thermal_oracle.py`` pins -- so the
# max_ulp 0 claim for these three leaves is the one their oracle tests already
# make, extended only by the packing, which the whole-column fixture gates.
#
# THERMOPROP lives in noahmp_leaves.cu and its entry point is spelled
# ``noahmp_leaf_thermoprop`` rather than ``noahmp_thermal_*``; it is reached
# through the same composed translation unit, so there is still exactly one
# device glibc libm on the path.

def _thermoprop_kernel(log10_profile="glibc_2_39"):
    return thermal_module(log10_profile=log10_profile).get_function("noahmp_leaf_thermoprop")


THERMOPROP_N_IN = 44
THERMOPROP_N_OUT = 30


def pack_thermoprop_calls(calls):
    """``(float32[n, 44], int32[n, 4])`` for captured physical THERMOPROP calls.

    The five slots WRF declares and never reads -- TG, UR, LAT, Z0M, ZLVL --
    are written as zero rather than left uninitialised.  The kernel does not
    read them (see ``noahmp_leaf_thermoprop``'s header), and a deterministic
    zero is what makes the packed row reproducible between two runs.
    """
    count = len(calls)
    x = np.zeros((count, THERMOPROP_N_IN), dtype=np.float32)
    ix = np.empty((count, 4), dtype=np.int32)
    for row, (args, kw) in enumerate(calls):
        if args:
            raise TypeError(
                f"THERMOPROP takes no positional arguments, got {len(args)}")
        if int(kw.get("nsnow", NSNOW)) != NSNOW or \
                int(kw.get("nsoil", NSOIL)) != NSOIL:
            raise ValueError(
                "the runtime CUDA THERMOPROP is pinned to three snow and four "
                f"soil layers, got {kw.get('nsnow')} and {kw.get('nsoil')}")
        line = x[row]
        line[0:NLAY] = kw["dzsnso"]
        line[NLAY:NLAY + NSNOW] = kw["snice"]
        line[NLAY + NSNOW:NLAY + 2 * NSNOW] = kw["snliq"]
        base = NLAY + 2 * NSNOW
        line[base:base + NSOIL] = kw["smc"]
        line[base + NSOIL:base + 2 * NSOIL] = kw["sh2o"]
        base += 2 * NSOIL
        line[base:base + NLAY] = kw["stc"]
        base += NLAY
        line[base] = kw["snowh"]
        line[base + 1] = kw["dt"]
        base += 7                     # snowh, dt, then the five inert slots
        line[base:base + NSOIL] = kw["smcmax"]
        line[base + NSOIL] = kw["csoil"]
        line[base + NSOIL + 1:base + 2 * NSOIL + 1] = kw["quartz"]
        ix[row] = (int(kw["isnow"]), int(kw["ist"]), 0,
                   1 if kw["urban_flag"] else 0)
    return x, ix


def evaluate_thermoprop_calls(calls):
    """One launch for every column's THERMOPROP; returns WRF's six arrays."""
    if not calls:
        return []

    import cupy as cp

    x, ix = pack_thermoprop_calls(calls)
    count = x.shape[0]
    device_y = cp.zeros((count, THERMOPROP_N_OUT), dtype=cp.float32)
    kernel = _thermoprop_kernel()
    blocks = (count + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (cp.asarray(x), cp.asarray(ix), device_y, np.int32(count)))
    y = cp.asnumpy(device_y)
    out = []
    for row in y:
        out.append((
            np.ascontiguousarray(row[0:NLAY]),                       # df
            np.ascontiguousarray(row[NLAY:2 * NLAY]),                # hcpct
            np.ascontiguousarray(row[2 * NLAY:2 * NLAY + NSNOW]),    # snicev
            np.ascontiguousarray(
                row[2 * NLAY + NSNOW:2 * NLAY + 2 * NSNOW]),         # snliqv
            np.ascontiguousarray(
                row[2 * NLAY + 2 * NSNOW:2 * NLAY + 3 * NSNOW]),     # epore
            np.ascontiguousarray(row[2 * NLAY + 3 * NSNOW:]),        # fact
        ))
    return out


TSNOSOI_N_IN = 5 * NLAY + 7
TSNOSOI_N_OUT = NLAY + 1


def pack_tsnosoi_calls(calls):
    """``(float32[n, 42], int32[n, 5])`` for captured physical TSNOSOI calls.

    Slots 28..34 (DZSNSO), 37 (SAG) and 40 (TG) are the ones the routine reads
    only after its own RETURN at :5346.  DZSNSO is packed because the caller
    has it; SAG and TG are not passed to this leaf by ENERGY at all and are
    written as zero.
    """
    count = len(calls)
    x = np.zeros((count, TSNOSOI_N_IN), dtype=np.float32)
    ix = np.zeros((count, 5), dtype=np.int32)
    for row, (args, kw) in enumerate(calls):
        if args:
            raise TypeError(
                f"TSNOSOI takes no positional arguments, got {len(args)}")
        line = x[row]
        line[0:NLAY] = kw["zsnso"]
        line[NLAY:2 * NLAY] = kw["stc"]
        line[2 * NLAY:3 * NLAY] = kw["df"]
        line[3 * NLAY:4 * NLAY] = kw["hcpct"]
        base = 5 * NLAY
        line[base] = kw["tbot"]
        line[base + 1] = kw["ssoil"]
        line[base + 3] = kw["dt"]
        line[base + 4] = kw["snowh"]
        line[base + 6] = kw["zbot"]
        ix[row, 0] = int(kw["isnow"])
    return x, ix


def evaluate_tsnosoi_calls(calls):
    """One launch for every column's TSNOSOI; returns ``(stc, eflxb)`` rows."""
    if not calls:
        return []

    import cupy as cp

    x, ix = pack_tsnosoi_calls(calls)
    count = x.shape[0]
    device_y = cp.zeros((count, TSNOSOI_N_OUT), dtype=cp.float32)
    kernel = thermal_module().get_function("noahmp_thermal_tsnosoi")
    blocks = (count + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (cp.asarray(x), cp.asarray(ix), device_y, np.int32(count)))
    y = cp.asnumpy(device_y)
    return [(np.ascontiguousarray(row[0:NLAY]), np.float32(row[NLAY]))
            for row in y]


PHASECHANGE_N_IN = 4 * NLAY + 2 * NSNOW + 2 * NSOIL + 3 + 3 * NSOIL
PHASECHANGE_N_OUT = NLAY + 2 * NSNOW + 2 * NSOIL + 4 + NLAY


def pack_phasechange_calls(calls):
    """``(float32[n, 57], int32[n, 4])`` for captured PHASECHANGE calls.

    HCPCT occupies slots 14..20 and the routine never references it (:5617);
    it is written as zero rather than threaded through, so the packed row does
    not depend on a value no arithmetic can observe.
    """
    count = len(calls)
    x = np.zeros((count, PHASECHANGE_N_IN), dtype=np.float32)
    ix = np.zeros((count, 4), dtype=np.int32)
    for row, (args, kw) in enumerate(calls):
        if args:
            raise TypeError(
                f"PHASECHANGE takes no positional arguments, got {len(args)}")
        line = x[row]
        line[0:NLAY] = kw["fact"]
        line[NLAY:2 * NLAY] = kw["dzsnso"]
        line[3 * NLAY:4 * NLAY] = kw["stc"]
        base = 4 * NLAY
        line[base:base + NSNOW] = kw["snice"]
        line[base + NSNOW:base + 2 * NSNOW] = kw["snliq"]
        base += 2 * NSNOW
        line[base:base + NSOIL] = kw["smc"]
        line[base + NSOIL:base + 2 * NSOIL] = kw["sh2o"]
        base += 2 * NSOIL
        line[base] = kw["sneqv"]
        line[base + 1] = kw["snowh"]
        line[base + 2] = kw["dt"]
        base += 3
        line[base:base + NSOIL] = kw["smcmax"]
        line[base + NSOIL:base + 2 * NSOIL] = kw["psisat"]
        line[base + 2 * NSOIL:base + 3 * NSOIL] = kw["bexp"]
        ix[row, 0] = int(kw["isnow"])
        ix[row, 1] = int(kw["ist"])
    return x, ix


def evaluate_phasechange_calls(calls):
    """One launch for every column's PHASECHANGE; WRF's ten return values."""
    if not calls:
        return []

    import cupy as cp

    x, ix = pack_phasechange_calls(calls)
    count = x.shape[0]
    device_y = cp.zeros((count, PHASECHANGE_N_OUT), dtype=cp.float32)
    kernel = thermal_module().get_function("noahmp_thermal_phasechange")
    blocks = (count + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (cp.asarray(x), cp.asarray(ix), device_y, np.int32(count)))
    y = cp.asnumpy(device_y)
    out = []
    for row in y:
        base = NLAY
        snice = np.ascontiguousarray(row[base:base + NSNOW])
        snliq = np.ascontiguousarray(row[base + NSNOW:base + 2 * NSNOW])
        base += 2 * NSNOW
        smc = np.ascontiguousarray(row[base:base + NSOIL])
        sh2o = np.ascontiguousarray(row[base + NSOIL:base + 2 * NSOIL])
        base += 2 * NSOIL
        out.append((
            np.ascontiguousarray(row[0:NLAY]),          # stc
            snice, snliq,
            np.float32(row[base]),                      # sneqv
            np.float32(row[base + 1]),                  # snowh
            smc, sh2o,
            np.float32(row[base + 2]),                  # qmelt
            np.float32(row[base + 3]),                  # ponding
            np.ascontiguousarray(row[base + 4:].astype(np.int32)),   # imelt
        ))
    return out
