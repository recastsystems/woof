"""Batched CUDA execution of Noah-MP's BARE_FLUX subtree.

The arithmetic lives in ``woof/core/kernels/noahmp_bareflux.cu``, whose
``d_bare_flux`` body is already held at max_ulp 0 against the unmodified WRF
v4.6.1 leaf oracle (``noahmp-bareflux.csv``), with its glibc ``logf``/``atanf``
and its ESAT coefficient table separately probed on the device.  This module
supplies the runtime-facing part the oracle harness did not: the physical
Python call is packed in column order, all columns are evaluated by one CUDA
launch, and the rows are unpacked as :class:`BareFluxOut` records.

Two things this deliberately does not do:

* It does not compile with any translation-unit-wide flag.  The kernel pins
  every FP32 operation with a rounding intrinsic and keeps every constant in
  ``__constant__`` memory, so ``-fmad`` is not essential and a global flag
  would silently move unrelated arithmetic if the source is composed later.
* It does not touch ``glibc_powf``.  ``d_bare_flux`` never calls it -- only
  ``powi3``/``powi4`` (integer powers, written as multiplies) and
  ``glibc_logf`` are on the live path -- so this seam neither depends on nor
  widens that leaf's narrow subnormal/negative-base domain guard.

Unlike ``noahmp_vegeflux_gpu``, no coefficient table is uploaded from the host:
``noahmp_bareflux.cu`` carries every constant it needs as a ``__constant__``
bit-pattern array of its own, which is what
``tests/test_noahmp_bareflux_cuda.py`` probes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from woof.core.noahmp_bareflux import BareFluxOut

NSNOW = 3
NSOIL = 4
NLAYER = NSNOW + NSOIL
N_INPUT = 53
N_INT = 5
N_OUTPUT = 13
THREADS = 64

#: Scalar slots 0..31 of the packed row, in the order ``noahmp_bareflux.cu``
#: documents and ``tools/noahmp_wrf461_oracle/gen_bareflux_cases.py`` writes.
#: ``dt``, ``thair``, ``q2``, ``dx``, ``dz8w``, ``qc``, ``sfcprs``, ``fsno``
#: and the incoming ``cm``/``ch`` are packed and never read by the kernel:
#: under opt_sfc=1 and opt_stc=1 no live statement references them.  They are
#: carried anyway so the runtime row and the fixture row are the same object.
SCALAR_NAMES = (
    "dt", "sag", "lwdn", "ur", "uu", "vv", "sfctmp", "thair", "qair", "eair",
    "rhoair", "snowh", "zlvl", "zpd", "z0m", "fsno", "emg", "rsurf", "lathea",
    "gamma", "rhsur", "q2", "pahb", "dx", "dz8w", "qc", "psfc", "sfcprs",
    "tgb", "cm", "ch", "qsfc",
)

ARRAY_NAMES = ("dzsnso", "stc", "df")

#: ``ii`` slots.  ``ivgtyp``, ``iloc`` and ``jloc`` are likewise inert.
INT_NAMES = ("isnow", "ivgtyp", "iloc", "jloc", "iurban")

OUTPUT_NAMES = (
    "tgb", "cm", "ch", "qsfc", "tauxb", "tauyb", "irb", "shb", "evb", "ghb",
    "t2mb", "q2b", "ehb2",
)

assert len(SCALAR_NAMES) + len(ARRAY_NAMES) * NLAYER == N_INPUT
assert len(INT_NAMES) == N_INT
assert len(OUTPUT_NAMES) == N_OUTPUT


def _bind_call(args, kwargs) -> dict:
    """BARE_FLUX is keyword-only in Python; refuse anything else."""
    if args:
        raise TypeError(
            f"BARE_FLUX takes no positional arguments, got {len(args)}")
    bound = dict(kwargs)
    if int(bound.get("nsnow", NSNOW)) != NSNOW or \
            int(bound.get("nsoil", NSOIL)) != NSOIL:
        raise ValueError(
            "the runtime CUDA BARE_FLUX is pinned to three snow and four "
            f"soil layers, got {bound.get('nsnow')} and {bound.get('nsoil')}")
    for option, admitted in (("opt_sfc", 1), ("opt_stc", 1)):
        if int(bound.get(option, admitted)) != admitted:
            raise NotImplementedError(
                f"device BARE_FLUX admits {option}={admitted} only")
    missing = [name for name in SCALAR_NAMES if name not in bound]
    missing += [name for name in ARRAY_NAMES if name not in bound]
    missing += [name for name in ("isnow", "ivgtyp", "iloc", "jloc",
                                  "urban_flag") if name not in bound]
    if missing:
        raise TypeError(f"BARE_FLUX batch call is missing {missing}")
    return bound


def pack_bare_flux_calls(calls: Sequence[tuple[tuple, dict]]):
    """Return the ``(float32[n, 53], int32[n, 5])`` device input for ``calls``."""
    bound = [_bind_call(args, kwargs) for args, kwargs in calls]
    count = len(bound)
    inputs = np.empty((count, N_INPUT), dtype=np.float32)
    ints = np.empty((count, N_INT), dtype=np.int32)
    for row, call in enumerate(bound):
        inputs[row, :len(SCALAR_NAMES)] = [call[name] for name in SCALAR_NAMES]
        base = len(SCALAR_NAMES)
        for name in ARRAY_NAMES:
            values = call[name]
            if len(values) != NLAYER:
                raise ValueError(
                    f"BARE_FLUX {name} must span -{NSNOW - 1}..{NSOIL} as "
                    f"{NLAYER} elements, got {len(values)}")
            inputs[row, base:base + NLAYER] = np.asarray(values,
                                                         dtype=np.float32)
            base += NLAYER
        ints[row] = (int(call["isnow"]), int(call["ivgtyp"]),
                     int(call["iloc"]), int(call["jloc"]),
                     1 if call["urban_flag"] else 0)
    return inputs, ints


def evaluate_bare_flux_calls(
        calls: Sequence[tuple[tuple, dict]]) -> list[BareFluxOut]:
    """Evaluate captured physical BARE_FLUX calls in one device batch."""
    if not calls:
        return []

    import cupy as cp

    from woof.core.kernels import get_kernel

    inputs, ints = pack_bare_flux_calls(calls)
    count = inputs.shape[0]
    device_out = cp.empty(count * N_OUTPUT, dtype=cp.float32)
    from woof.core.noahmp_kernel_sources import STAGED_MELT_RESET, melt_reset_kernel
    kernel = (melt_reset_kernel("noahmp_bareflux", "noahmp_bare_flux")
              if STAGED_MELT_RESET.get() else
              get_kernel("noahmp_bareflux", "noahmp_bare_flux"))
    blocks = (count + THREADS - 1) // THREADS
    kernel(
        (blocks,), (THREADS,),
        (cp.asarray(inputs.ravel()), cp.asarray(ints.ravel()), device_out,
         np.int32(count)),
    )
    host_out = cp.asnumpy(device_out).reshape(count, N_OUTPUT)
    return [BareFluxOut(**dict(zip(OUTPUT_NAMES, row))) for row in host_out]


__all__ = [
    "ARRAY_NAMES",
    "INT_NAMES",
    "OUTPUT_NAMES",
    "SCALAR_NAMES",
    "evaluate_bare_flux_calls",
    "pack_bare_flux_calls",
]
