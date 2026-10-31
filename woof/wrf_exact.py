"""Opt-in strict arithmetic for WRF verification, selected before import.

This strict installer does not import CuPy in a default process. Urban
modules have their own source-scoped subnormal policy. CuPy adds FTZ
after caller options, so verification normalizes options at the final
NVRTC program boundary as well as before forming the compiled-image cache key.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import os
from pathlib import Path
import threading

ENABLED = os.environ.get("GPUWM_WRF_EXACT", "0") == "1"

#: The four strict controls.  Under ``GPUWM_WRF_EXACT=1`` each one is ON
#: unless its variable is set to ``0``: strict mode is WRF's arithmetic,
#: and with any control off the very first operation of a forecast already
#: leaves WRF (the combo sweep of 2026-10-07 ran ``GPUWM_WRF_EXACT=1``
#: alone, and the stage-1 pressure diagnosis differed in ~90 % of words
#: before any tendency; round2/LOCALIZE.md).  ``=0`` keeps a control off
#: for an experiment that measures one control at a time.
CONTROL_VARIABLES = ("WOOF_WRF_EXACT_DIAGNOSTICS", "WOOF_WRF_EXACT_ADVECTION",
                     "GPUWM_WRF_EXACT_DIFFUSION", "GPUWM_WRF_EXACT_BIGSTEP")


def _control(name: str) -> bool:
    return ENABLED and os.environ.get(name, "1") != "0"


DIAGNOSTICS_ENABLED = _control("WOOF_WRF_EXACT_DIAGNOSTICS")
ADVECTION_ENABLED = _control("WOOF_WRF_EXACT_ADVECTION")
DIFFUSION_ENABLED = not ENABLED or _control("GPUWM_WRF_EXACT_DIFFUSION")
BIGSTEP_ENABLED = _control("GPUWM_WRF_EXACT_BIGSTEP")
STRICT_OPTIONS = ("--fmad=false", "--ftz=false", "--prec-div=true",
                  "--prec-sqrt=true", "-DGPUWM_WRF_EXACT=1")
for _selected, _macro in (
    (DIAGNOSTICS_ENABLED, "GPUWM_WRF_EXACT_D_DIAGNOSTICS"),
    (ADVECTION_ENABLED, "GPUWM_WRF_EXACT_C_ADVECTION"),
    (DIFFUSION_ENABLED, "GPUWM_WRF_EXACT_C_DIFFUSION"),
    (BIGSTEP_ENABLED, "GPUWM_WRF_EXACT_C_BIGSTEP"),
):
    if _selected:
        STRICT_OPTIONS += (f"-D{_macro}=1",)
_OVERRIDDEN = {"fmad", "ftz", "prec-div", "prec-sqrt", "use_fast_math",
               "DGPUWM_WRF_EXACT", "DGPUWM_WRF_EXACT_D_DIAGNOSTICS",
               "DGPUWM_WRF_EXACT_C_ADVECTION", "DGPUWM_WRF_EXACT_C_DIFFUSION",
               "DGPUWM_WRF_EXACT_C_BIGSTEP"}
_RECORDS: list[dict] = []
_LOCK = threading.Lock()


#: The C library of the WRF build a run is matched against.  WRF's
#: initialization tables call the host libm, and glibc 2.41 replaced several
#: binary32 routines with CORE-MATH's correctly rounded ones (atanf among
#: them), so the same WRF 4.6.1 source writes different table words on a
#: glibc 2.39 host and on a 2.41+ host.  The default is 2.39, the platform
#: of the published strict record; ``GPUWM_WRF_GLIBC=2.43`` (any 2.41+)
#: names a referee built on a newer glibc, such as the pinned node builds.
#: The breakage it prevents: MYJSFCINIT's PSIM table built with the other
#: generation's atanf differs from the referee's in 816 of 10001 words, and
#: every column that reads one of them leaves WRF.
GLIBC_VARIABLE = "GPUWM_WRF_GLIBC"


def referee_glibc() -> tuple[int, int]:
    """``(major, minor)`` of the referee WRF's glibc, from GPUWM_WRF_GLIBC."""
    raw = (os.environ.get(GLIBC_VARIABLE) or "2.39").strip().lower()
    text = raw[len("glibc"):].lstrip("-_ ") if raw.startswith("glibc") else raw
    try:
        major, minor = (int(part) for part in text.split(".")[:2])
    except ValueError:
        raise ValueError(f"{GLIBC_VARIABLE} must name a glibc version such "
                         f"as 2.39 or 2.43, not {raw!r}") from None
    return major, minor


def referee_correctly_rounded_trig() -> bool:
    """True when the referee's binary32 atanf is CORE-MATH's (glibc 2.41+)."""
    return referee_glibc() >= (2, 41)


def effective_controls() -> dict[str, bool]:
    """Which strict controls this process runs (receipts record these, not
    only the environment, because an unset control is on)."""
    return {"diagnostics": DIAGNOSTICS_ENABLED, "advection": ADVECTION_ENABLED,
            "diffusion": DIFFUSION_ENABLED, "bigstep": BIGSTEP_ENABLED}


def effective_options(options) -> tuple[str, ...]:
    """Replace conflicting arithmetic switches, preserving every other option."""
    kept = []
    for option in options:
        key = option.lstrip("-").split("=", 1)[0]
        if key not in _OVERRIDDEN:
            kept.append(option)
    return tuple(kept) + STRICT_OPTIONS


def _record(kind, source, options):
    with _LOCK:
        _RECORDS.append({"kind": kind,
                         "source_sha256": hashlib.sha256(
                             source.encode("utf-8")).hexdigest(),
                         "options": list(options)})


def compile_receipt() -> dict:
    """Requests include cache hits; NVRTC entries identify actual compilations."""
    with _LOCK:
        return {"enabled": ENABLED, "controls": effective_controls(),
                "strict_options": list(STRICT_OPTIONS),
                "records": list(_RECORDS)}


def _write_receipt():
    target = os.environ.get("WOOF_WRF_EXACT_COMPILE_RECEIPT")
    receipt = compile_receipt()
    if target and receipt["records"]:
        # A CLI parent can exit after its forecast subprocess. It compiled
        # nothing and must not overwrite that subprocess's actual receipt.
        Path(target).write_text(json.dumps(receipt, indent=2) + "\n",
                                encoding="utf-8")


def install(compiler=None):
    """Install once, only in a process explicitly opting into verification.

    Normalized cache options prevent loading a default compiled image. The
    program hook also covers direct NVRTC loads, preprocessing, RawKernel,
    RawModule and CuPy-generated elementwise/reduction kernels.
    """
    if not ENABLED:
        return
    if compiler is None:
        from cupy.cuda import compiler
    if getattr(compiler, "_gpuwm_wrf_exact", False):
        return
    original_cache = compiler._compile_with_cache_cuda
    original_program = compiler._NVRTCProgram.compile

    @functools.wraps(original_cache)
    def cached(source, options, *args, **kwargs):
        options = effective_options(options)
        _record("cache_request", source, options)
        return original_cache(source, options, *args, **kwargs)

    @functools.wraps(original_program)
    def program(self, options=(), log_stream=None):
        options = effective_options(options)
        _record("nvrtc_compile", self.src, options)
        return original_program(self, options, log_stream)

    compiler._compile_with_cache_cuda = cached
    compiler._NVRTCProgram.compile = program
    compiler._gpuwm_wrf_exact = True
    atexit.register(_write_receipt)
