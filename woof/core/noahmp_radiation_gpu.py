"""Batched CUDA execution of Noah-MP's RADIATION subtree.

The arithmetic lives in ``woof/core/kernels/noahmp_radiation.cu``.  Its
``d_albedo`` and ``d_surrad`` bodies -- and, underneath them, ``d_snow_age``,
``d_snowalb_class``, ``d_groundalb`` and ``d_twostream`` -- are already held at
max_ulp 0 against the unmodified WRF v4.6.1 leaf oracles by
``tests/test_noahmp_radiation_cuda.py``.  This module supplies the
runtime-facing part: one physical ``RADIATION`` call per land column, packed in
column order, evaluated by **one** launch, and unpacked into the same dict
:func:`woof.core.noahmp_energy._radiation` returns.

Three things this deliberately does not do.

* It does not compile with ``--fmad=false``.  That flag is a diagnostic in this
  project, never a shipped fix.  ``tests/test_noahmp_radiation_cuda.py`` pins
  the leaf oracles with *and* without it, so the claim that every arithmetic
  site already carries its own rounding intrinsic is a measurement rather than
  a comment.
* It does not split RADIATION into an ALBEDO launch and a SURRAD launch with
  :2787-2790 on the host.  Five host statements per land column is exactly the
  per-column CPython this seam exists to remove.
* It does not widen ``glibc_powf``'s domain guard.  See
  :data:`POWF_DOMAIN_NOTE`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

NSOIL = 4
N_INPUT = 56
N_OUTPUT = 19
THREADS = 64

#: Slots 0..23: the parameter block, constant for a parameter identity.
PARAMETER_SLOTS = (
    "tau0", "grain_growth", "extra_growth", "dirt_soot", "swemx",
    "albsat", "albdry", "alblak", "rhol", "rhos", "taul", "taus",
    "xl", "omegas", "betads", "betais",
)

# Native SNOWALB_BATS parameter order. These are supplied by the actual
# transferred MPTABLE; CLASS parameters cannot stand in for them.
BATS_PARAMETER_NAMES = (
    "bats_cosz", "bats_vis_new", "bats_nir_new", "bats_vis_age",
    "bats_nir_age", "bats_vis_dir", "bats_nir_dir",
)

#: Slots 24..36, then SMC(1..4) at 37..40, then the seven carried entry
#: values at 41..51, then SOLAD(2)/SOLAI(2) at 52..55.  The order is
#: ``_ALB_IN`` in ``tests/test_noahmp_radiation_cuda.py`` extended by the two
#: forcing vectors SURRAD needs, and ``noahmp_radiation.cu`` documents the same
#: layout at ``d_radiation``.
STATE_SLOTS = (
    "dt", "cosz", "elai", "esai", "tg", "tv", "snowh", "fsno", "fwet",
    "sneqvo", "sneqv", "qsnow", "fveg",
)

#: ``FAGE`` (:2758) and ``FREVD``/``FREVI``/``FREGD``/``FREGI`` are
#: uninitialised locals in WRF that ALBEDO returns untouched at ``COSZ <= 0``.
#: The CPU transcription passes the defined 0.0 for all five and documents why;
#: the device packing must pass the same bytes or the two are not the same
#: function at night.
_UNDEFINED_ENTRY = 0.0

OUTPUT_NAMES = (
    "fsun", "laisun", "laisha", "parsun", "parsha", "sav", "sag", "fsa",
    "fsr", "fsrv", "fsrg",
)

#: What the five narrow ``glibc_powf`` copies do outside their domain, and why
#: RADIATION is the first conversion that had to answer it.
POWF_DOMAIN_NOTE = (
    "noahmp_radiation.cu's glibc_powf returns NaN for a subnormal or negative "
    "base rather than taking glibc's sign/zero/inf path.  RADIATION reaches "
    "powf at exactly two live sites -- SNOW_AGE's (1-FAGE)**10 style term and "
    "TWOSTREAM's EXP-free algebra -- and "
    "tests/test_noahmp_radiation_cuda.py::test_the_powf_domain_guard_fires "
    "drives both with the whole subnormal/signed-zero/negative neighbourhood "
    "of the measured live range and records which inputs actually reach it."
)


def _parameter_row(p) -> np.ndarray:
    """Slots 0..23 for one ``EnergyParameters``, memoised on identity.

    ``NoahmpRuntimeParameters.handle`` already memoises one ``SflxParameters``
    per distinct land-use/soil pair, so a 360,000-column nest presents a
    handful of distinct objects here and this cache is bounded by that count,
    not by the column count.  The entry keeps a strong reference to ``p`` so
    the ``id`` cannot be reused by a later object.
    """
    key = id(p)
    hit = _PARAMETER_CACHE.get(key)
    if hit is not None:
        return hit[1]
    row = np.empty(24, dtype=np.float32)
    row[0] = p.tau0
    row[1] = p.grain_growth
    row[2] = p.extra_growth
    row[3] = p.dirt_soot
    row[4] = p.swemx
    row[5:7] = p.albsat
    row[7:9] = p.albdry
    row[9:11] = p.alblak
    row[11:13] = p.rhol
    row[13:15] = p.rhos
    row[15:17] = p.taul
    row[17:19] = p.taus
    row[19] = p.xl
    row[20:22] = p.omegas
    row[22] = p.betads
    row[23] = p.betais
    row.flags.writeable = False
    _PARAMETER_CACHE[key] = (p, row)
    return row


_PARAMETER_CACHE: dict[int, tuple[object, np.ndarray]] = {}


def _bind_call(args, kwargs) -> tuple:
    """RADIATION is called as ``_radiation(p, options, **kw)``; refuse others."""
    if len(args) != 2:
        raise TypeError(
            "device RADIATION expects (p, options) positionally, got "
            f"{len(args)} positional arguments")
    p, options = args
    if options.opt_alb != 2:
        raise NotImplementedError("device RADIATION admits opt_alb=2 only")
    if int(kwargs.get("nsoil", NSOIL)) != NSOIL:
        raise ValueError(
            "the runtime CUDA RADIATION is pinned to four soil layers, got "
            f"{kwargs.get('nsoil')}")
    if int(kwargs.get("ice", 0)) != 0:
        raise NotImplementedError(
            "device RADIATION admits ice=0 only; ice=1 is the glacier column")
    return p, kwargs


def pack_radiation_calls(calls: Sequence[tuple[tuple, dict]]):
    """Return the ``(float32[n, 56], int32[n])`` device input for ``calls``."""
    bound = [_bind_call(args, kwargs) for args, kwargs in calls]
    count = len(bound)
    inputs = np.empty((count, N_INPUT), dtype=np.float32)
    ist = np.empty(count, dtype=np.int32)
    for row, (p, kw) in enumerate(bound):
        line = inputs[row]
        line[0:24] = _parameter_row(p)
        line[24:37] = [kw[name] for name in STATE_SLOTS]
        line[37:41] = np.asarray(kw["smc"], dtype=np.float32)[:NSOIL]
        line[41] = kw["albold"]
        line[42] = kw["tauss"]
        line[43:52] = _UNDEFINED_ENTRY          # FAGE, FREVD/I, FREGD/I
        line[52:54] = np.asarray(kw["solad"], dtype=np.float32)
        line[54:56] = np.asarray(kw["solai"], dtype=np.float32)
        ist[row] = int(kw["ist"])
    return inputs, ist


def unpack_radiation_row(row) -> dict:
    """One device row as the dict ``_radiation`` returns."""
    out = {name: np.float32(row[k]) for k, name in enumerate(OUTPUT_NAMES)}
    out["albsnd"] = (np.float32(row[11]), np.float32(row[12]))
    out["albsni"] = (np.float32(row[13]), np.float32(row[14]))
    out["bgap"] = np.float32(row[15])
    out["wgap"] = np.float32(row[16])
    out["albold"] = np.float32(row[17])
    out["tauss"] = np.float32(row[18])
    return out


def evaluate_radiation_calls(
        calls: Sequence[tuple[tuple, dict]]) -> list[dict]:
    """Evaluate captured physical RADIATION calls in one device batch."""
    if not calls:
        return []

    import cupy as cp

    from woof.core.kernels import get_kernel

    from woof.noahmp_generation import radiation_kernel
    generations = {args[1].noahmp_snow_age for args, _kw in calls}
    if len(generations) > 1:
        result = [None] * len(calls)
        for generation in sorted(generations):
            indices = [i for i, (args, _kw) in enumerate(calls)
                       if args[1].noahmp_snow_age == generation]
            values = evaluate_radiation_calls([calls[i] for i in indices])
            for i, value in zip(indices, values):
                result[i] = value
        return result
    inputs, ist = pack_radiation_calls(calls)
    count = inputs.shape[0]
    device_out = cp.empty(count * N_OUTPUT, dtype=cp.float32)
    kernel = get_kernel("noahmp_radiation", radiation_kernel(next(iter(generations))))
    blocks = (count + THREADS - 1) // THREADS
    kernel((blocks,), (THREADS,),
           (cp.asarray(inputs.ravel()), cp.asarray(ist), device_out,
            np.int32(count)))
    host_out = cp.asnumpy(device_out).reshape(count, N_OUTPUT)
    return [unpack_radiation_row(row) for row in host_out]


__all__ = [
    "N_INPUT",
    "N_OUTPUT",
    "OUTPUT_NAMES",
    "POWF_DOMAIN_NOTE",
    "evaluate_radiation_calls",
    "pack_radiation_calls",
    "unpack_radiation_row",
]
