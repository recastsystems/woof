"""Kernel numerics: ``fast`` (the default) or ``strict`` (the opt-in for bitwise identity to native MPAS-A).

Owner ruling 2026-10-16: strict numerics is opt-in, never the default.

``strict``
    Every kernel compiles the way the bitwise proofs measured it: no FMA contraction (``--fmad=false``),
    IEEE subnormals where a unit asks for them (``--ftz=false``, or the guarded FP32 fallback of
    ``cuda_fp32.CUDA_FTZ_HELPERS`` on the central route), IEEE division and square root.  This is the
    route that is byte-identical to native MPAS-A v8.4.2 built with the pinned strict recipe; every
    identity proof, oracle replay and the landing gate names it (``--numerics strict``).
``fast``
    FMA contraction on (``--fmad=true``), FP32 subnormals flushed (``--ftz=true``; the guarded fallback is
    compiled out with ``-DMPAS_FTZ_FALLBACK_ENABLED=0``), IEEE division and square root kept
    (``--prec-div=true --prec-sqrt=true``: physics table lookups floor quotients, so an approximate
    quotient can land one index low), ptxas optimisation on.  ``--use_fast_math`` is never used.
    Land-surface units keep ``--ftz=false`` (``options(..., keep_ftz=True)``): flushing loses dry-soil
    drainage, flips snow branches and loses QSFC (the engine's ``ieee_module`` refusal names the breakage).

One process runs one mode.  The mode is read once (``--numerics``, the namelist's ``config_numerics``, or
``WOOF_HEX_NUMERICS``); after the first kernel compile it cannot change, so fast and strict kernels never
mix in one run.  Every kernel cache key carries the effective options, so they never mix on disk either.
"""

from __future__ import annotations

import os
from typing import Iterable

ENV = "WOOF_HEX_NUMERICS"
MODES = ("fast", "strict")
DEFAULT = "fast"

#: The numeric flags fast mode compiles with (replacing whatever numeric flags a unit's strict tuple names).
FAST_FLAGS: tuple[str, ...] = ("--fmad=true", "--ftz=true", "--prec-div=true", "--prec-sqrt=true")
#: Source switches fast mode defines: the guarded FP32 subnormal fallback compiles to plain arithmetic,
#: and units with a fast source path take it.
FAST_DEFINES: tuple[str, ...] = ("-DMPAS_FTZ_FALLBACK_ENABLED=0", "-DHEX_NUMERICS_FAST=1")

#: Base options of the central ``KernelCache`` route (CuPy's RawModule appends ``-ftz=true`` itself).
STRICT_KERNEL_BASE: tuple[str, ...] = ("--std=c++17", "--fmad=false")
FAST_KERNEL_BASE: tuple[str, ...] = ("--std=c++17", "--fmad=true", *FAST_DEFINES)

_NUMERIC_FLAG_NAMES = ("fmad", "ftz", "prec-div", "prec-sqrt", "use_fast_math")
#: Strict-only ptxas switches: fast mode drops them if a unit names one.  (bl_mynnedmf_c942418 compiled its
#: strict image at -O0 until rrfs/mynn-speed, 2026-10-28; its optimised strict image replays the same words.)
_STRICT_ONLY = ("--ptxas-options=-O0", "-Xptxas=-O0")

_mode: str | None = None
_bound = False


class NumericsError(ValueError):
    """A numerics mode that is not fast or strict, or a change after kernels were compiled."""


def normalise(value: object, source: str = ENV) -> str:
    text = str(value).strip().lower()
    if text not in MODES:
        raise NumericsError(f"{source} must be one of {', '.join(MODES)}, not {value!r}")
    return text


def mode() -> str:
    """The process's numerics mode (``WOOF_HEX_NUMERICS``, default fast) unless :func:`set_mode` chose one."""
    global _mode
    if _mode is None:
        _mode = normalise(os.environ.get(ENV) or DEFAULT, ENV)
    return _mode


def is_strict() -> bool:
    return mode() == "strict"


def set_mode(value: object, *, source: str = "--numerics") -> str:
    """Choose the mode for this process and the processes it starts (ranks, segments, subprocesses).

    Refused after a kernel was compiled under the other mode: THE BREAKAGE THIS PREVENTS is a run whose
    kernels mix fast and strict images while its receipt names one mode."""
    global _mode
    chosen = normalise(value, source)
    if _bound and chosen != mode():
        raise NumericsError(
            f"{source} {chosen}: kernels were already compiled with {mode()} numerics in this process; "
            "one process runs one numerics mode")
    _mode = chosen
    os.environ[ENV] = chosen
    # the engine (ArWen physics, woof) reads its own knob; one run, one mode on both sides
    os.environ["GPUWM_NUMERICS"] = chosen
    return chosen


def _bind() -> str:
    global _bound
    current = mode()
    _bound = True
    return current


def _is_numeric_flag(option: str) -> bool:
    if option in _STRICT_ONLY:
        return True
    name = option.lstrip("-").split("=", 1)[0]
    return name in _NUMERIC_FLAG_NAMES


def options(strict: Iterable[str], *, keep_ftz: bool = False, numerics: str | None = None) -> tuple[str, ...]:
    """The options a unit compiles with in this mode.

    ``strict`` is the unit's strict tuple; strict mode returns it unchanged (the bytes every identity proof
    measured).  Fast mode drops its numeric flags and appends :data:`FAST_FLAGS` and :data:`FAST_DEFINES`;
    ``keep_ftz`` keeps ``--ftz=false`` for land-surface units."""
    chosen = _bind() if numerics is None else normalise(numerics)
    strict = tuple(str(option) for option in strict)
    if chosen == "strict":
        return strict
    kept = tuple(option for option in strict if not _is_numeric_flag(option) and option not in FAST_DEFINES)
    flags = tuple("--ftz=false" if keep_ftz and flag.startswith("--ftz") else flag for flag in FAST_FLAGS)
    return (*kept, *flags, *FAST_DEFINES)


def kernel_base(numerics: str | None = None) -> tuple[str, ...]:
    """Base options of the central ``KernelCache`` route in this mode."""
    chosen = _bind() if numerics is None else normalise(numerics)
    return STRICT_KERNEL_BASE if chosen == "strict" else FAST_KERNEL_BASE


def receipt_block() -> dict[str, object]:
    """What a receipt records about the numerics a run compiled with."""
    current = mode()
    return {
        "mode": current,
        "kernel_base_options": list(kernel_base(current)),
        "flags": list(STRICT_KERNEL_BASE[1:]) + ["--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
        if current == "strict" else list(FAST_FLAGS + FAST_DEFINES),
        "land_surface_ftz": "false",
        "bitwise_to_native": False,
        "bitwise_proof_requested": current == "strict",
    }


def resolve(cli_value: str | None, namelist_text: str | None = None) -> tuple[str, str]:
    """The mode a door runs and where it came from: ``--numerics``, else the namelist's
    ``config_numerics`` (any group), else ``WOOF_HEX_NUMERICS``, else fast.  A namelist and a flag that
    disagree are refused (the receipt would name one and the deck the other)."""
    named = None
    if namelist_text:
        import re
        found = re.findall(r"(?im)^\s*config_numerics\s*=\s*['\"]?([A-Za-z]+)['\"]?", namelist_text)
        if len(set(v.lower() for v in found)) > 1:
            raise NumericsError("the namelist names config_numerics more than once with different values")
        if found:
            named = normalise(found[0], "config_numerics")
    if cli_value is not None:
        chosen = normalise(cli_value, "--numerics")
        if named is not None and named != chosen:
            raise NumericsError(f"conflicting --numerics {chosen} and namelist config_numerics {named}")
        return chosen, "--numerics"
    if named is not None:
        return named, "namelist config_numerics"
    if os.environ.get(ENV):
        return normalise(os.environ[ENV], ENV), ENV
    return DEFAULT, "default"


def _reset_for_tests() -> None:
    global _mode, _bound
    _mode = None
    _bound = False


__all__ = ["DEFAULT", "ENV", "FAST_DEFINES", "FAST_FLAGS", "FAST_KERNEL_BASE", "MODES", "NumericsError",
           "STRICT_KERNEL_BASE", "is_strict", "kernel_base", "mode", "normalise", "options", "receipt_block",
           "resolve", "set_mode"]
