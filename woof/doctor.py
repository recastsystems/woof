"""``woof doctor``: verify the runtime estate, print exact remedies.

The pip package deliberately splits its runtime across estates the
installer cannot see from ``pip install`` alone: the GPU runtime
(CuPy), the optional pip extras (``render``, ``obs``, ``dealias`` and
whatever else the installed distribution declares), the compiled Rust
artifacts (never shipped in the wheel; ``woof fetch-bridges`` stages a
release's prebuilt bundle where one exists for the platform), the
packaged physics tables, and the data roots
(``WPS_GEOG``/``GPUWM_CASE_DATA_ROOT``).  Doctor checks each one for
real, not by presence: it imports every declared package in short-lived
subprocesses, probe-executes every bridge and front-door executable,
loads the CPU preprocessing library through ctypes and reads its ABI
version, sha256-validates the packaged Thompson tables with the same
routine the model uses at launch, parses the Noah/landuse tables with
the model's own parsers, re-hashes every artifact a sealed manifest
declares, resolves each data route's decoder and byte transport through
the resolvers the route itself calls, and requires each WPS_GEOG
dataset's ``index`` file.  No cargo builds, no network; the device work
is three deliberate short-lived subprocess probes (the CuPy-wheel/box
cuBLAS pairing, a cold-cache NVRTC compile, and the radar-DA
eigensolver), isolated so a wedged runtime cannot poison this process.
Every gap prints a remedy whose every line is either a command that
runs as printed in this platform's own shell or a ``#`` comment --
never prose fused onto a command -- instead of letting the user meet a
raw traceback three commands later.

**The extras are read from the INSTALLED distribution's metadata**, not
from a checkout's ``pyproject.toml``.  What a box can do is decided by
what pip resolved onto it, and a checkout sitting beside a wheel is
exactly the configuration where a transcription and the truth part
company.

Statuses distinguish what was proven: ``verified`` means the deep check
ran and passed; ``present`` is for the few items where nothing deeper
than existence can accurately be checked, plus the one deep check that
can half-pass and says which half (the radar-DA eigensolver);
``untested`` is for a question this module deliberately does not answer
-- it never runs cargo, so it cannot say whether a build would succeed
-- and its detail always opens with "not tested"; ``missing`` is a gap
with a remedy; ``info`` is context.  ``verified`` is never printed over
a question nobody asked: two binaries once carried ``ok`` on a box
where the command that needed them could not find them, and that is
what ``untested`` exists to stop.

The report is layered (:mod:`woof.explain`).  By default every finding
is one line -- status, subject, and THE command that closes it -- and
adjacent findings that share a remedy fold into one, because a fresh pip
install gaps every Rust artifact at once and six identical remedies read
as six problems.  ``--explain`` prints what this module always printed:
the evidence behind each check and the whole pasteable remedy block,
verbatim.  Nothing was shortened; the long form moved one flag away.

Exit status: 1 when any finding is ``broken`` or ``unreachable``, 0
otherwise.  A ``degraded`` or ``opt-in`` finding still prints MISSING
and still prints its command; it does not fail the process.  The rule
that assigns those four words is stated once, in :func:`blocking_gaps`.
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from dataclasses import dataclass
import importlib.metadata
from importlib.util import find_spec

from woof.local_gpu import no_local_gpu
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from woof import bridges
from woof import rustwx
from woof import rustwx_fetch
from woof import science_core
from woof.explain import explain_enabled
from woof.file_digest import sha256_file as _sha256

# The pip extras exactly as the README installs them.
#
# Each hint is a command first and explanation second, on separate
# lines, because a remedy is read under stress and pasted whole: the
# parenthetical that used to trail the command on the same line --
# `pip install 'recast-woof[gpu]'   (or: pip install cupy-cuda12x)` -- is a
# shell error the moment anyone does the obvious thing with it.
#
# The CuPy hint is the one remedy in this module that must NOT lead with
# a single command, and it is the exception that proves the rule: CuPy
# ships one wheel per CUDA major, pip cannot detect the major, and this
# text is only reached when the box's own major could not be read
# either.  A hint that led with `pip install 'recast-woof[gpu]'` -- as it did
# through 1.8.0 -- handed a CUDA-13 box the cu12 wheel and called it the
# default.  So when the major is unknown, BOTH are printed and neither
# is the default; when it is known, `_gpu_extra_hint` below replaces
# this text with the one line that matches the box.
GPU_EXTRA_HINT = ("nvidia-smi\n"
                  "  # read the CUDA version out of that header.  CuPy\n"
                  "  # ships one wheel per CUDA major and pip cannot\n"
                  "  # detect which one this box serves, so the extra\n"
                  "  # has to name it.  Then, for a CUDA 12.x box:\n"
                  "pip install 'recast-woof[gpu-cu12]'\n"
                  "  # or, for a box whose CUDA is 13-only:\n"
                  "pip install 'recast-woof[gpu-cu13]'")

#: The brief report has room for exactly one command.  It carries the
#: cu12 wheel with the cu13 alternative fused on as a shell comment --
#: which runs as printed in sh and in PowerShell alike -- because the
#: one thing this line may never do is imply cu12 is simply correct.
GPU_EXTRA_UNKNOWN_ACTION = ("pip install 'recast-woof[gpu-cu12]'  "
                            "# CUDA 12.x; cu13 box: recast-woof[gpu-cu13]")
#: NOT "wrf-rust + matplotlib".  matplotlib is a BASE dependency, on
#: every install already; the extra's second package is pyshp, the
#: shapefile reader every basemap needs, and naming matplotlib here left
#: a reader no way to learn that.  Measured wrong by the 2026-08-14
#: reachability audit against the shipped METADATA.
RENDER_EXTRA_HINT = ("pip install 'recast-woof[render]'\n"
                     "  # installs wrf-rust (the derived-quantity core\n"
                     "  # woof enprod and --engine matplotlib import) and\n"
                     "  # pyshp (the shapefile reader the DA nowcast's\n"
                     "  # basemaps read).  matplotlib is NOT in this extra:\n"
                     "  # it is a base dependency, installed already.")
GEOG_STACK_HINT = (
    "pip install --upgrade 'recast-woof[geog]'\n"
    "  # rasterio and pyproj are the PARITY FALLBACK for the\n"
    "  # high-resolution warp, not what a default run reads: the\n"
    "  # default engine is the Rust static-fields library (the\n"
    "  # `static builder` row).  Install these only to run\n"
    "  # WOOF_STATIC_PYTHON=1.  To add just the two libraries:\n"
    "  #   pip install --upgrade rasterio pyproj")


def _geog_hint() -> str:
    """The fetch-geog remedy, with sizes read off the pin table.

    A literal here was wrong the day a pin was added: the text kept
    saying 16 GB while the default set had grown past 29, in the one
    place a reader decides whether to start the download.
    """

    from woof.geog_assets import (
        GEOG_CONSUMER_WRF, datasets_required_by, size_phrase)

    wrf = datasets_required_by(GEOG_CONSUMER_WRF)
    return ("woof fetch-geog\n"
            "  # downloads every pinned WPS_GEOG dataset\n"
            f"  # ({size_phrase()}) into\n"
            "  # $GPUWM_CASE_DATA_ROOT/WPS_GEOG, or --root DIR.\n"
            "  # Resumable, SHA-256-verified, re-run safe.\n"
            f"  # `--datasets wrf` stages the {len(wrf)} pins the WRF static\n"
            f"  # builder opens ({size_phrase(wrf)}) and\n"
            "  # skips the pins only `woof mesh` or a named\n"
            "  # geog_data_res option reads.")


GEOG_HINT = _geog_hint()

REINSTALL_HINT = (
    "pip install -e .\n"
    "  # the installed package is incomplete; reinstall from a clone\n"
    "  # (or from a rebuilt wheel)")

#: Bridge executables the real-data routes launch, with the consumer
#: that fails without each one.
#:
#: The last two entries are worded the way they are because the obvious
#: wording was measured false.  They used to read "20CRv3/mapped GRIB2
#: routes", which a reader takes to mean the staged binary this report
#: probes is what those routes use.  It is not: the mapped route and
#: ``woof adapt`` build their own copies with cargo and never consult
#: the resolver ladder, and the 20CRv3 direct route makes the paths
#: ``required=True``.  :func:`_mapped_grib2_route_check` reports that
#: separately; these lines now claim only what they prove, which is that
#: the staged file exists and runs.
_BRIDGE_CONSUMERS = {
    "grib1_bridge": "ERA5 route (woof check/run, rw-wps --source era5)",
    "gfs_grib2_bridge": "GFS front door (rw-wps --source gfs, and every "
                        "stage of the woof go chain)",
    "hrrr_grib2_bridge": "HRRR front door (rw-wps --source hrrr)",
    "grib2_inventory": "the 20CRv3, mapped and packaged-profile GRIB2 "
                       "prep routes (woof prep/rw-wps), which resolve it "
                       "from here (--grib2-inventory overrides) -- see the "
                       "mapped/20CRv3 route line",
    "grib2_dump": "the 20CRv3, mapped and packaged-profile GRIB2 prep "
                  "routes (woof prep/rw-wps), which resolve it from here "
                  "(--grib2-dump overrides) -- see the mapped/20CRv3 "
                  "route line",
    "gdt101_remap": "every source whose native grid is a GDT-101 "
                    "unstructured mesh (woof fetch/prep/go --source "
                    "icon-global or icon-d2), whose input-normalization "
                    "stage writes its regional intermediates with this "
                    "binary and "
                    "refuses rather than falling back without it",
    "met_intermediate": "public initial-condition preparation that writes "
                        "WPS intermediates from a regular-grid source "
                        "(the installed woof hex RRFS preset and ICON init "
                        "rows), which refuses by name without it",
}

#: The doors a box without CuPy cannot open, and the half it keeps.
#: Declared once because two lines report it -- the deep ``cupy (GPU
#: runtime)`` check and the ``[gpu-cu12]``/``[gpu-cu13]`` extras lines --
#: and a reader who found them disagreeing would be right to distrust
#: both.  Every entry was traced to a real refusal or a real import.
#: `woof domain` and `woof check` are split, not listed whole: only
#: their measure-this-card forms need the device, and the declared-
#: capacity forms WITHOUT-A-GPU.md walks through run on a box with no
#: CuPy at all -- a doctor that said otherwise sent that reader away
#: from the one sizing route the page exists to show them.
_GPU_DOORS = ("woof run", "woof go", "woof resume", "woof verify",
              "woof stream", "woof multi-run", "woof downscale",
              "woof ingest", "woof-prepared-forecast",
              "woof-prepared-tree-forecast", "the DA nowcast",
              "woof domain and woof check sized by measuring this "
              "machine's card")
_GPU_STILL_WORKS = ("the whole preprocessing half -- woof fetch, "
                    "import-namelist, adapt, render, report and rw-wps -- "
                    "and sizing for a declared card: woof domain with "
                    "--card or --vram-gib, woof check with --free-gib or "
                    "--budget-gib and --vram-gib")

_PROBE_TIMEOUT_S = 30


#: The four severities, and the two of them that move the exit code.
#: ``blocking_gaps`` states the rule; these are its vocabulary.
#:
#: They exist because a two-valued model could not tell a reader the
#: difference between "the ~16 GB terrain download you deliberately
#: skipped" and "this box cannot run a forecast", and doctor printed
#: the same ``0 of them blocking (exit 0)`` over both.
SEVERITY_BROKEN = "broken"
SEVERITY_UNREACHABLE = "unreachable"
SEVERITY_DEGRADED = "degraded"
SEVERITY_OPT_IN = "opt-in"

#: There is deliberately NO ``BLOCKING_SEVERITIES`` constant.  ``broken``
#: always blocks; ``unreachable`` blocks for a README/FIRST-LIGHT door and
#: not for a CLI-reference-only one, so a constant equating severity with
#: the exit code would be a false claim in the one module that exists to
#: stop those.  ``Check.blocking`` is the authority and
#: :func:`blocking_gaps` states the rule.

def _companion_distribution_name() -> str:
    """The companion row's name, so cross-references cannot drift."""

    from woof import data_assets

    return data_assets.COMPANION_DISTRIBUTION


@dataclass(frozen=True)
class Check:
    """One doctor line: verified/present/MISSING/info plus the remedy.

    Four fields carry the report; three more carry its *layering*.

    ``detail`` and ``remedy`` are the full text -- the evidence and the
    whole pasteable remedy block -- and ``--explain`` prints them
    unchanged.  ``brief`` and ``action`` are the same finding at one
    line: the short evidence token, and THE single command to run.
    They are declared here rather than sliced out of ``detail`` and
    ``remedy`` by a parser, because most remedy blocks open with a ``#``
    comment and several have no command in them at all (an unset
    ``GPUWM_CASE_DATA_ROOT`` needs a path only the reader knows).  A
    parser would have had to guess, and a guessed next command is the
    one thing this report cannot afford to get wrong.

    ``group`` lets the terse report fold repeats: a pip install gaps
    every bridge at once and prints ``woof fetch-bridges`` once per
    bridge, which reads as that many problems.  Grouping is a
    *presentation* of the same checks -- ``--explain`` and ``--json``
    still carry each one by name.

    ``status`` has five values and one of them is newer than the rest.
    ``untested`` exists because a 2026-08-14 audit caught this report
    printing ``ok`` for two binaries on a box where the command that
    needs them could not find them: doctor had proven the FILES were
    runnable and said nothing about the RESOLUTION, and ``ok`` is not a
    word for a question nobody asked.  Where a check cannot exercise the
    thing it reports on, it says ``untested`` and its detail opens with
    "not tested".  Never ``verified``, and never a quiet omission.
    """

    name: str
    status: str  # "verified" | "present" | "untested" | "missing" | "info"
    detail: str
    remedy: str | None = None
    #: The single next command, or a short imperative when there is no
    #: command to print.  ``None`` for a check with nothing to do.
    action: str | None = None
    #: Short evidence for the one-line form; ``None`` prints the name
    #: alone, which is the right answer for a check whose only news is
    #: that it passed.
    brief: str | None = None
    #: Fold key for the terse report; ``None`` never folds.
    group: str | None = None
    #: Does this gap justify a nonzero exit?  The rule is stated once,
    #: in :func:`blocking_gaps`, and ``severity`` below is the word for
    #: WHY.  The report text is identical either way -- MISSING stays
    #: MISSING and the remedy still prints; only the exit code and the
    #: summary line read these two.
    blocking: bool = True
    #: Which of the four severities this finding carries:
    #: ``"broken"`` (something present and wrong), ``"unreachable"``
    #: (a documented command cannot run at all), ``"degraded"`` (an
    #: optional mode is gone, a documented default still works), or
    #: ``"opt-in"`` (an explicit choice the user has not made).  The
    #: first two block; the last two do not.  ``None`` on a finding
    #: that is not a gap.
    #:
    #: Declared as a field rather than inferred from ``blocking``
    #: because "why does this fail my install" and "does this fail my
    #: install" are different questions, and a boolean answering both
    #: is how ``0 of them blocking`` came to sit over a box that could
    #: not run a forecast.
    #:
    #: A gap that does not declare one gets the word its ``blocking``
    #: flag has always meant (see ``__post_init__``), so no finding is
    #: ever unclassified and no call site has to repeat itself.
    severity: str | None = None

    def __post_init__(self) -> None:
        # The severities are new; ``blocking`` is not, and every legacy
        # call site already carried the distinction in it.  Its
        # docstring through 2.3.3 read: True for "anything broken or
        # integrity-suspect", False for "an absent optional piece with
        # a documented fallback or a documented opt-in".  That IS the
        # broken/opt-in pair, so the default is a translation rather
        # than a guess -- and the two words the boolean cannot express,
        # `unreachable` and `degraded`, are passed explicitly by the
        # checks that mean them.
        if self.status == "missing" and self.severity is None:
            object.__setattr__(
                self, "severity",
                SEVERITY_BROKEN if self.blocking else SEVERITY_OPT_IN)


#: Terse-report fold keys.  Both name a crate, because that is what
#: makes the members share one remedy: everything under
#: ``tools/grib1_bridge`` is staged (or built) together, and so is
#: everything under ``tools/rustwx``.
_GROUP_BRIDGES = "bridges"
_GROUP_ENGINES = "rust engines"

#: The two ``woof setup`` steps, by the command each one runs.  Doctor
#: names ``woof setup`` in its summary exactly when it would otherwise
#: print more than one of these, which is the definition of a fresh
#: install: the wrapper is a shorter true answer than the list.
SETUP_ACTIONS = ("woof fetch-bridges", "woof fetch-tables")


def _short(text: str, limit: int = 64) -> str:
    """One line of at most ``limit`` characters, ellipsis when cut.

    The terse report shows evidence, not the whole finding: a failed
    import's last traceback line can be any length, and a line that
    wraps three times is the wall this report exists to stop being.  The
    untruncated text is one flag away and the ellipsis says so.
    """

    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3].rstrip() + "..."


def _build_action(crate_relative: str = bridges.CRATE_RELATIVE) -> str:
    """THE single next command for a missing Rust artifact, here.

    Three installs, three true one-liners.  In a checkout it is the
    cargo build.  On a wheel with a bundle published for this platform
    it is the download.  On a wheel without one the accurate answer is
    that there is no single command -- it is a clone and a build -- so
    the action is the flag that prints those steps rather than a
    fabricated one-liner naming a directory this install does not have.
    """

    if bridges.sources_present(crate_relative):
        return bridges.cargo_build_one_liner(crate_relative)
    if bridges.prebuilt_bundle_offer() is not None:
        return "woof fetch-bridges"
    return "woof doctor --explain"


# ---------------------------------------------------------------------------
# Python packages: actual imports, in a subprocess
# ---------------------------------------------------------------------------

def _import_probe(module: str,
                  distribution: str | None = None) -> tuple[bool, str]:
    """Actually import ``module`` in a subprocess; (ok, evidence).

    ``find_spec`` alone lies green for a package whose install is broken
    (ABI mismatch, missing native dependency, half-removed dist-info).
    The subprocess keeps a failing import from poisoning this process
    and allocates nothing beyond the import itself.

    The version reported is the *installed distribution's*, read from
    package metadata, falling back to the module's ``__version__``
    attribute.  A module attribute is whatever the author last edited by
    hand and can lag the release it shipped in -- a field report had
    doctor announcing wrf-rust 0.2.34 on a machine with 0.2.35
    installed, which is the wrong number to hand someone debugging a
    version-sensitive problem.  ``distribution`` is the pip name when it
    differs from the import name (``wrf`` is installed as ``wrf-rust``).
    """

    if find_spec(module) is None:
        return False, "not installed"
    code = (
        "import sys, importlib.metadata as md\n"
        f"import {module} as m\n"
        "try:\n"
        f"    version = md.version({(distribution or module)!r})\n"
        "except Exception:\n"
        "    version = str(getattr(m, '__version__', 'imported'))\n"
        "sys.stdout.write(version)\n")
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            errors="replace", timeout=_PROBE_TIMEOUT_S * 10)
    except subprocess.TimeoutExpired as error:
        return False, (f"installed; import probe timed out after {error.timeout:g} s "
                       "without an answer (box load can delay a probe); retry doctor "
                       "when the load falls. No import failure was established.")
    except OSError as error:
        return False, f"installed but the import probe failed to run: {error}"
    if probe.returncode == 0:
        return True, (probe.stdout or "").strip() or "imported"
    tail = [line for line in (probe.stderr or "").strip().splitlines()
            if line.strip()]
    reason = tail[-1] if tail else f"exit {probe.returncode}"
    if probe.returncode == -2 or reason == "KeyboardInterrupt":
        return False, ("installed; import probe interrupted before an answer "
                       "(SIGINT/KeyboardInterrupt, including an external probe timeout); "
                       "retry doctor when the load falls. No import failure was established.")
    return False, f"installed but failed to import: {reason}"


def _import_probe_incomplete(evidence: str) -> bool:
    return ("import probe timed out" in evidence
            or "import probe interrupted" in evidence)


#: CuPy ships one wheel per CUDA major (``cupy-cuda12x``, ``cupy-cuda13x``)
#: and the wrong one is invisible to an import probe: the wheel hard-codes
#: its CUDA major's library names (``libcublas.so.12``) and loads cuBLAS
#: lazily, so on a CUDA-13-only box ``import cupy`` succeeds, kernels
#: compile, and the first matmul of a real run is what dies.  Proven on a
#: rented CUDA 13.2 node (2026-08-05): import probes, certification, and a
#: 57-test GPU slice all passed accurately before the first cuBLAS load
#: killed the campaign.  So this probe performs that first load on
#: purpose, in a fresh subprocess, and reports the CUDA majors of both
#: sides so the check can name the extra that matches the box.
#:
#: The judgment is the LOAD, not the majors: a newer driver serves older
#: wheels wherever the older runtime libraries exist (the reference box
#: runs a 13.3 driver over a working cupy-cuda12x install), so refusing on
#: a bare major mismatch would refuse machines that work.  The majors are
#: read for the diagnosis and the remedy, after the load has failed.
_CUBLAS_PAIRING_PROBE = """
import json, sys
out = {}
try:
    import cupy
except Exception as error:
    sys.stdout.write(json.dumps({"cupy": repr(error)}))
    raise SystemExit(0)
for key, call in (("wheel_runtime", "runtimeGetVersion"),
                  ("driver", "driverGetVersion")):
    try:
        out[key] = int(getattr(cupy.cuda.runtime, call)())
    except Exception:
        out[key] = 0
try:
    out["devices"] = int(cupy.cuda.runtime.getDeviceCount())
except Exception as error:
    out["devices"] = 0
    out["device_error"] = f"{type(error).__name__}: {error}"
if out["devices"] > 0:
    try:
        a = cupy.arange(4.0, dtype=cupy.float32).reshape(2, 2)
        b = a @ a
        float(b[0, 0])
        out["cublas"] = "ok"
    except Exception as error:
        out["cublas"] = f"{type(error).__name__}: {error}"
sys.stdout.write(json.dumps(out))
"""

_CUPY_WHEEL_NAME = re.compile(r"^cupy-cuda(\d+)x$")

#: CUDA major -> the pip extra that installs its wheel.  A major outside
#: this table gets the bare wheel name instead of an extra.
#:
#: 12 names ``gpu-cu12``, not the ``gpu`` alias: a remedy that says which
#: major it chose and why is the whole point, and ``gpu`` says neither.
#: The alias still resolves for anyone who already types it.
_GPU_EXTRA_BY_MAJOR = {12: "gpu-cu12", 13: "gpu-cu13"}


def _driver_library_names() -> tuple[str, ...]:
    """The CUDA driver library this platform would load, by name."""

    if sys.platform == "win32":
        return ("nvcuda.dll",)
    if sys.platform == "darwin":
        return ()
    return ("libcuda.so.1", "libcuda.so")


def _driver_cuda_major() -> int | None:
    """The CUDA major this box's DRIVER serves, or ``None`` if unknown.

    Read with ``ctypes`` straight off the driver library rather than
    through CuPy, because the case that needs the answer most is the box
    that has NO CuPy yet: that is where the remedy has to name an extra,
    and where every CuPy-based probe is by definition unavailable.  Until
    1.8.1 that case got a static hint whose only command was the cu12
    extra, so a CUDA-13 box asked doctor what to install and was told to
    install the wheel that cannot load cuBLAS on it.

    ``cuDriverGetVersion`` is the one entry point that answers this
    without ``cuInit``: no context, no device open, nothing that could
    disturb a card another process is using.  It is still driver contact,
    so ``GPUWM_NO_LOCAL_GPU`` suppresses it and the caller falls back to
    naming both extras.  Every failure is a ``None`` -- a box with no
    NVIDIA driver is the ordinary case here, not an error.
    """

    if no_local_gpu():
        return None
    for name in _driver_library_names():
        try:
            library = ctypes.CDLL(name)
            version = ctypes.c_int(0)
            if library.cuDriverGetVersion(ctypes.byref(version)) != 0:
                continue
        except (OSError, AttributeError, ValueError):
            continue
        if version.value > 0:
            return version.value // 1000
    return None


def _gpu_extra_hint(box_major: int | None) -> tuple[str, str]:
    """``(remedy, action)`` for a box with no working CuPy.

    With the box's CUDA major in hand the remedy leads with the ONE
    extra that matches it and says which major it read.  Without it,
    both extras are printed and neither is presented as the default,
    because a silent default is exactly how a CUDA-13 box ends up
    running cupy-cuda12x.
    """

    extra = _GPU_EXTRA_BY_MAJOR.get(box_major) if box_major else None
    if extra is None:
        return GPU_EXTRA_HINT, GPU_EXTRA_UNKNOWN_ACTION
    install = f"pip install 'recast-woof[{extra}]'"
    remedy = (f"# this box's driver serves CUDA {box_major}, and CuPy ships\n"
              f"# one wheel per CUDA major; this extra is the matching one\n"
              f"{install}")
    return remedy, install


def _installed_cupy_wheels() -> list[tuple[str, int | None]]:
    """Every installed CuPy distribution as ``(name, CUDA major)``.

    The major comes from the distribution NAME, because that is the
    thing pip resolved and the thing an uninstall must name; a
    source-built ``cupy`` has no major in its name and reports ``None``.

    A ``~upy_cuda12x-*.dist-info`` folder is what pip leaves when an
    uninstall hits a locked file.  Its METADATA still says cupy-cuda12x,
    but pip ignores it ("Ignoring invalid distribution") and cannot
    uninstall it by that name, so counting it read as two builds that
    no remedy could ever bring down to one.
    """

    found = []
    for dist in importlib.metadata.distributions():
        folder = getattr(dist, "_path", None)
        if folder is not None and Path(str(folder)).name.startswith("~"):
            continue
        name = (dist.metadata["Name"] or "").strip().lower()
        match = _CUPY_WHEEL_NAME.match(name)
        if match:
            found.append((name, int(match.group(1))))
        elif name == "cupy":
            found.append((name, None))
    return sorted(set(found))


def _cublas_pairing_probe() -> dict:
    """Run :data:`_CUBLAS_PAIRING_PROBE` out of process; ``{}`` if not."""
    return _json_probe(_CUBLAS_PAIRING_PROBE)


def _json_probe(script: str) -> dict:
    """Run an isolated JSON probe with the shared timeout and failure evidence."""
    try:
        probe = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, errors="replace",
            timeout=_PROBE_TIMEOUT_S * 4)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"probe": f"did not run: {error}"}
    try:
        return json.loads(probe.stdout or "{}")
    except ValueError:
        tail = [line for line in (probe.stderr or "").strip().splitlines()
                if line.strip()]
        return {"probe": tail[-1] if tail else f"exit {probe.returncode}"}


def _wrong_wheel_remedy(wheels: list[tuple[str, int | None]],
                        box_major: int) -> tuple[str, str]:
    """(remedy, action) that moves the install onto ``box_major``.

    Every line is a command as printed or a ``#`` comment, identical in
    both shells: ``pip`` spells the same everywhere.
    """

    lines = [f"# this box serves CUDA {box_major}; the installed CuPy "
             "cannot load cuBLAS on it"]
    for name, _ in wheels:
        lines.append(f"pip uninstall -y {name}")
    extra = _GPU_EXTRA_BY_MAJOR.get(box_major)
    if extra is not None:
        install = f"pip install 'recast-woof[{extra}]'"
    else:
        install = f"pip install cupy-cuda{box_major}x"
    lines.append(install)
    return "\n".join(lines), install


def _overlapping_cupy_check(wheels: list[tuple[str, int | None]]) -> Check:
    """Two CuPy distributions in one environment: always broken.

    ``cupy-cuda12x`` and ``cupy-cuda13x`` install the same ``cupy``
    package files, and pip keeps both registered over them.  Whichever
    wrote last answers ``import cupy``, reinstalling the other says
    "already satisfied", and uninstalling either removes files the other
    needs.  No import probe can judge that, so the pair itself is the
    finding, and the remedy removes every one before installing the
    build this box's driver serves.
    """

    names = [name for name, _ in wheels]
    box_major = _driver_cuda_major()
    lines = ["# more than one CuPy build is installed over the same files;",
             "# remove every one, then install the build for this box"]
    uninstall = "pip uninstall -y " + " ".join(names)
    lines.append(uninstall)
    extra = _GPU_EXTRA_BY_MAJOR.get(box_major) if box_major else None
    if extra is not None:
        lines.append(f"pip install 'recast-woof[{extra}]'")
    else:
        lines.append("# then one of these, for this box's CUDA major:")
        lines.extend(f"pip install 'recast-woof[{name}]'" for name in _GPU_EXTRA_BY_MAJOR.values())
    served = "" if box_major is None else f"; this box serves CUDA {box_major}"
    return Check(
        "cupy (GPU runtime)", "missing",
        f"{len(names)} CuPy builds are installed at once ({', '.join(names)}); they share one set of "
        f"files, so which one answers is whichever wrote last{served}",
        "\n".join(lines), action=uninstall, brief=f"{len(names)} CuPy builds installed",
        severity=SEVERITY_BROKEN)


def _host_threads_check() -> Check:
    """The interpreter and its lock: whether ``--devices N>1`` will run here.

    THE BREAKAGE THIS NAMES: every slab of a multi-card forecast steps from
    its own Python thread, and on a GIL build those threads take turns on
    one interpreter lock -- measured 2 cards at 0.78x of one card on 3.12,
    against 33/20/17 s per forecast hour on 1/2/4 cards on free-threaded
    3.14t.  So every multi-card door refuses on such an interpreter
    (:func:`woof.free_threading.refuse_multi_card_under_gil`), and this
    row says so before a run does.  Never blocking: one-card forecasts do
    not care, and a GIL build still runs every one-card forecast.
    """
    from woof.free_threading import (MULTI_CARD_GIL_REFUSAL, free_threaded_build,
                                      multi_card_lock_reason)

    name = "multi-card host threads"
    version = ".".join(str(v) for v in sys.version_info[:3])
    gil_env = os.environ.get("PYTHON_GIL")
    interpreter = (f"{sys.executable} (Python {version}, "
                   f"{'free-threaded' if free_threaded_build() else 'GIL'} build, "
                   f"PYTHON_GIL={gil_env if gil_env is not None else 'unset'})")
    reason = multi_card_lock_reason()
    if reason is None:
        return Check(name, "verified",
                     f"{interpreter}: the interpreter lock is off in this "
                     "process, so --devices N>1 slab threads run at once "
                     "(command lines re-run themselves with PYTHON_GIL=0)",
                     brief=f"free-threaded {version}, GIL off")
    # install.sh and install.ps1 make .venv on free-threaded 3.14t by
    # default (2.8.8), so the remedy is to let them make it again; a
    # pip-only environment is remade on a 3.14t uv provides.
    #
    # Every line is a comment, deliberately.  The closing line of the
    # report says every remedy line is a command "to run as printed, in
    # the order printed", so a pasted report would have deleted .venv --
    # the environment this report runs in, and on a pip install whatever
    # .venv the reader's own directory holds -- part-way through the
    # other gaps' commands, then called an installer a pip install does
    # not have.  Remaking an environment is a choice made on purpose, not
    # a line in a select-all paste; it is spelled out here for the shell
    # the rest of the report is spelled for (bridges.WINDOWS_SHELL, which
    # tests force both ways), not the host's os.name.
    if bridges.WINDOWS_SHELL:
        rebuild = "Remove-Item -Recurse -Force .venv; .\\install.ps1"
        installer = ".\\install.ps1"
    else:
        rebuild = "rm -rf .venv && bash install.sh"
        installer = "install.sh"
    remedy = (f"# {MULTI_CARD_GIL_REFUSAL}\n"
              "# for multi-card forecasts, remake the environment on free-threaded Python 3.14t\n"
              "# (not pasted as a command: it deletes the environment this report runs in);\n"
              f"# in a source checkout {installer} provisions 3.14t by default, so from its root:\n"
              f"#   {rebuild}\n"
              "# a pip install: uv venv --seed --python 3.14t, then in it\n"
              "#   pip install --prefer-binary woof (cftime 1.6.6 has no 3.14t wheel)\n"
              "#   and export PYTHON_GIL=0 in the venv's activate")
    return Check(
        name, "info",
        f"{interpreter}: {reason}, so woof go/preflight/sim/ensemble refuse "
        "--devices N>1 here (multi-card slab threads would take turns on the "
        "lock: 2 cards measured 0.78x of 1); one-card forecasts are unaffected",
        remedy,
        action="remake the environment on free-threaded python3.14t with PYTHON_GIL=0 for multi-card forecasts",
        brief="GIL on: --devices N>1 refused", blocking=False)


def _cupy_check() -> Check:
    wheels = _installed_cupy_wheels()
    if len(wheels) > 1:
        return _overlapping_cupy_check(wheels)
    ok, evidence = _import_probe("cupy")
    if ok:
        wheels = _installed_cupy_wheels()
        named = ", ".join(name for name, _ in wheels) or "cupy"
        if no_local_gpu():
            # The documented never-open-the-local-device switch (the
            # rented-GPU workflow leaves it set on the user's own
            # machine).  The pairing probe is real device contact, so
            # under this flag it does not run, and the report says which
            # question went unanswered rather than pretending it passed.
            return Check(
                "cupy (GPU runtime)", "present",
                f"imported in a subprocess (cupy {evidence}, {named}); "
                "GPUWM_NO_LOCAL_GPU is set, so the CuPy-wheel/box CUDA "
                "pairing was not judged (no device contact)",
                brief=f"cupy {evidence}, device not touched",
                blocking=False)
        result = _cublas_pairing_probe()
        cublas = result.get("cublas")
        driver = int(result.get("driver") or 0)
        wheel_runtime = int(result.get("wheel_runtime") or 0)
        wheel_majors = sorted({major for _, major in wheels
                               if major is not None})
        if not wheel_majors and wheel_runtime:
            wheel_majors = [wheel_runtime // 1000]
        if cublas == "ok":
            return Check(
                "cupy (GPU runtime)", "verified",
                f"imported in a subprocess (cupy {evidence}, {named}); "
                f"cuBLAS loaded on this box (wheel CUDA "
                f"{wheel_runtime // 1000}.{wheel_runtime % 1000 // 10}, "
                f"driver CUDA {driver // 1000}.{driver % 1000 // 10})",
                brief=f"cupy {evidence}, cuBLAS ok")
        if result.get("devices", 0) == 0:
            # No device to load cuBLAS against, so the pairing cannot be
            # judged -- and a box without a CUDA device must not fail
            # doctor for it.  ``present``, accurately: import proven,
            # pairing not.
            reason = result.get("device_error") or result.get(
                "cupy") or result.get("probe") or "no CUDA device visible"
            return Check(
                "cupy (GPU runtime)", "present",
                f"imported in a subprocess (cupy {evidence}, {named}), "
                f"but the CuPy-wheel/box CUDA pairing was not judged: "
                f"{_short(str(reason), 96)}",
                brief=f"cupy {evidence}, no device", blocking=False)
        if _names_full_device(cublas):
            return _full_device_check("cupy (GPU runtime)", cublas,
                                      status="present")
        # A device exists and the first cuBLAS load failed.  This is the
        # CUDA-major trap when the majors disagree, and a broken install
        # either way; both refuse, because the next real run dies at its
        # first matmul.
        box_major = driver // 1000
        detail = (f"cupy {evidence} ({named}) imports, but cuBLAS failed "
                  f"its first load on this box: {_short(str(cublas), 120)}.")
        if (_names_missing_cuda_headers(cublas)
                and (not wheel_majors or box_major in wheel_majors
                     or not box_major)):
            # THE CIRCULAR REMEDY, closed.  CuPy 14 raises its
            # headers error from the first device call of any kind, so
            # the right wheel on the right box reaches HERE -- and this
            # branch used to answer "the installed CuPy cannot load
            # cuBLAS on it: uninstall cupy-cuda13x, install
            # recast-woof[gpu-cu13]", which removes the correct wheel and
            # installs the same correct wheel, forever.  The wheel is
            # fine; what is missing is a CUDA toolkit.  The
            # majors-disagree case is deliberately NOT folded in: there
            # the wheel really is wrong, and its own remedy replaces it
            # with one whose woof extra carries the toolkit too.
            detail += ("  The wheel matches this box; what is absent is the "
                       "CUDA toolkit HEADER tree, which no cupy wheel ships "
                       "and no wheel reinstall supplies")
            remedy, action = _cuda_headers_remedy(box_major or None)
            return Check("cupy (GPU runtime)", "missing", detail, remedy,
                         action=action, brief=_short(str(cublas)),
                         severity=SEVERITY_BROKEN)
        if box_major and wheel_majors and box_major not in wheel_majors:
            targets = ", ".join(f"CUDA {major}" for major in wheel_majors)
            detail += (f"  The wheel targets {targets}; the box serves "
                       f"CUDA {driver // 1000}.{driver % 1000 // 10}.  "
                       "An import probe cannot see this: the wheel "
                       "hard-codes its own major's library names and "
                       "loads cuBLAS lazily, so everything short of a "
                       "real matmul passes")
        if box_major:
            remedy, action = _wrong_wheel_remedy(wheels, box_major)
        else:
            remedy, action = _gpu_extra_hint(_driver_cuda_major())
        return Check("cupy (GPU runtime)", "missing", detail, remedy,
                     action=action, brief=_short(str(cublas)),
                     severity=SEVERITY_BROKEN)
    # No usable CuPy.  The extra to name is a property of the BOX, so it
    # is read from the driver here rather than defaulted: this is the
    # branch a fresh CUDA-13 machine lands on, and the branch that used
    # to hand it the cu12 wheel.
    box_major = _driver_cuda_major()
    remedy, action = _gpu_extra_hint(box_major)
    served = "" if box_major is None else f"; this box serves CUDA {box_major}"
    if evidence == "not installed":
        # BLOCKING, and it was not until 2026-08-14.  This line used to
        # be an "absent optional" beside `0 of them blocking (exit 0)`,
        # on a box where `woof run` -- the product's headline command,
        # and the whole point of installing it -- dies in a raw
        # ModuleNotFoundError after fetching gigabytes and preparing
        # them.  A green light over that is worse than no light: an
        # installer script that trusts the exit code ships the box, and
        # the user meets the traceback instead of this remedy.
        return Check(
            "cupy (GPU runtime)", "missing",
            "not installed -- without it this box cannot run "
            + ", ".join(_GPU_DOORS)
            + f".  It can still run {_GPU_STILL_WORKS}{served}", remedy,
            action=action, brief="not installed",
            severity=SEVERITY_UNREACHABLE)
    if _import_probe_incomplete(evidence):
        return Check("cupy (GPU runtime)", "untested", evidence,
                     brief=_short(evidence), blocking=False)
    # CuPy IS installed and would not import.  The extra cannot be the
    # answer to every version of that -- see _cupy_import_failure_remedy,
    # which reads the import's own message and routes accordingly.
    remedy, action = _cupy_import_failure_remedy(evidence, box_major)
    return Check("cupy (GPU runtime)", "missing", evidence + served, remedy,
                 action=action, brief=_short(evidence),
                 severity=SEVERITY_BROKEN)


#: Compiles and executes two kernels, keeping execution failures separate
#: from the named compiler-library and toolkit-header failures:
#:
#: 1. a SELF-CONTAINED kernel with no ``#include`` at all -- how every woof
#:    kernel is written.  It exercises NVRTC itself, which ships INSIDE the
#:    CuPy wheel. Compile failures are recorded separately from allocation,
#:    launch, synchronization and readback failures.
#: 2. a CuPy reduction, which compiles through CuPy's own cub/jitify
#:    preamble.  That preamble ``#include``s the CUDA runtime headers, and
#:    those come from a TOOLKIT, found through ``CUDA_PATH``. A failure naming
#:    missing headers selects that remedy; runtime failures and wrong answers
#:    remain execution failures even when (1) passed.
#:
#: THE COLD CACHE IS ESSENTIAL.  ``CUPY_CACHE_DIR`` is redirected to an
#: empty directory because a warm kernel cache is exactly what hides this
#: fault: the box that produced this check compiled its reductions once under
#: CUDA 12, moved to a CUDA-13 toolkit, and kept serving the cached cubins for
#: weeks while every UNCACHED reduction failed.  A probe that reuses the
#: cache reports green on a box where the next new kernel dies.
_NVRTC_HEADER_PROBE = """
import json, sys
out = {}
try:
    import cupy
except Exception as error:
    sys.stdout.write(json.dumps({"cupy": repr(error)}))
    raise SystemExit(0)
try:
    if cupy.cuda.runtime.getDeviceCount() < 1:
        sys.stdout.write(json.dumps({"devices": 0}))
        raise SystemExit(0)
except Exception as error:
    sys.stdout.write(json.dumps({"devices": 0, "device_error": repr(error)}))
    raise SystemExit(0)
stage = "compile"
out["self_contained_compiled"] = False
try:
    module = cupy.RawModule(code=(
        'extern "C" __global__ '
        'void gpuwm_probe(float* out) { out[0] = 1.0f; }'))
    kernel = module.get_function("gpuwm_probe")
    out["self_contained_compiled"] = True
    stage = "allocation"
    result = cupy.empty(1, dtype=cupy.float32)
    stage = "launch"
    kernel((1,), (1,), (result,))
    stage = "synchronize"
    cupy.cuda.get_current_stream().synchronize()
    stage = "readback"
    value = float(cupy.asnumpy(result)[0])
    out["self_contained"] = "ok" if value == 1.0 else f"wrong value {value}"
    if value != 1.0:
        out["self_contained_stage"] = "validation"
except Exception as error:
    out["self_contained"] = f"{type(error).__name__}: {error}"
    out["self_contained_stage"] = stage
try:
    total = int(cupy.arange(64, dtype=cupy.int64).sum())
    out["toolkit_headers"] = "ok" if total == 2016 else f"wrong sum {total}"
    if total != 2016:
        out["toolkit_headers_stage"] = "validation"
except Exception as error:
    out["toolkit_headers"] = f"{type(error).__name__}: {error}"
    out["toolkit_headers_stage"] = "reduction"
import os
out["cuda_path"] = os.environ.get("CUDA_PATH") or ""
sys.stdout.write(json.dumps(out))
"""


def _nvrtc_header_probe(*, reuse_bound=False) -> dict:
    """Run :data:`_NVRTC_HEADER_PROBE` cold, out of process; ``{}`` if not."""

    if find_spec("cupy") is None:
        return {"cupy": "not installed"}
    with tempfile.TemporaryDirectory(prefix="gpuwm-nvrtc-probe-") as cache:
        environment = dict(os.environ, CUPY_CACHE_DIR=cache)
        try:
            probe = subprocess.run(
                [sys.executable, "-c", (
                    "from woof.forecast_readiness import main; main()"
                    if reuse_bound else _NVRTC_HEADER_PROBE)],
                capture_output=True, text=True, errors="replace",
                env=environment, timeout=_PROBE_TIMEOUT_S * 6)
        except subprocess.TimeoutExpired:
            # NOT a verdict.  A cold NVRTC compile on a slow or busy box
            # can outrun this budget while being perfectly healthy, and
            # reporting that as a missing header tree would send a reader
            # to install a toolkit they already have.
            return {"slow": f"no answer within {_PROBE_TIMEOUT_S * 6} s"}
        except OSError as error:
            return {"probe": f"did not run: {error}"}
    try:
        result = json.loads(probe.stdout or "{}")
        if not isinstance(result, dict) or not result or probe.returncode != 0:
            raise ValueError("the probe produced no successful structured result")
        if reuse_bound:
            # Cache rejection and unsupported-layout warnings originate in
            # the child. Keep them visible even when the cold fallback passes.
            if probe.stderr:
                sys.stderr.write(probe.stderr)
            if result.get("readiness_cache") == "bound-cold":
                print("note: forecast readiness used a cold probe for the "
                      "current compiler/header binding", file=sys.stderr)
        return result
    except ValueError:
        tail = [line for line in (probe.stderr or "").strip().splitlines()
                if line.strip()]
        return {"probe": tail[-1] if tail else f"exit {probe.returncode}"}


#: The CuPy wheel extra that installs a matching CUDA toolkit, as wheels.
#:
#: CuPy 14.0 added it; 13.x has none, which is why the gpu extras floor at
#: 14.0 (pip only WARNS on an extra the resolved version does not provide,
#: so a lower floor would let the toolkit go silently missing).
_CUPY_TOOLKIT_EXTRA = "ctk"


def _cupy_ctk_install(major: int) -> str:
    """The one command that gives a driver-only box a working CuPy."""

    return f"pip install 'cupy-cuda{major}x[{_CUPY_TOOLKIT_EXTRA}]'"


def _cuda_headers_remedy(box_major: int | None) -> tuple[str, str]:
    """``(remedy, action)`` for a box whose CuPy cannot find CUDA headers.

    NOT A WHEEL PROBLEM, SO NOT A WHEEL REMEDY.  This is the correction
    that matters most on a fresh box: the gap used to be reported -- when
    it was reported at all -- with ``pip install 'recast-woof[gpu-cu13]'``,
    which reinstalls the CuPy wheel that is already present and supplies
    no headers, because no CuPy wheel has ever carried a header tree.  A
    buyer following it watches pip succeed and the fault survive.

    WHAT LEADS NOW, and why it changed.  MEASURED on test host A
    (Ubuntu, CUDA 13 driver, no ``nvcc``, no ``/usr/local/cuda``, no
    conda), 2026-08-17: a bare ``pip install cupy-cuda13x`` gives a 274
    MiB environment in which cuBLAS, a self-contained ``RawModule`` and a
    ``cupy.arange().sum()`` reduction ALL fail with "Failed to find CUDA
    headers", and ``pip install 'cupy-cuda13x[ctk]'`` gives an 1867 MiB
    environment in which all three pass.  CuPy publishes the toolkit as
    an extra of its own wheel, pinned to that wheel's CUDA major, so the
    fix is one line, needs no conda, no administrator and no
    ``CUDA_PATH``.  The old lead -- a conda toolkit -- assumed an estate
    the failing box did not have.

    CuPy's own message hardcodes ``cupy-cuda12x[ctk]`` at every major.
    Pasted on a CUDA-13 box that installs the wheel that cannot load
    cuBLAS there, which is the trap this whole module was rewritten
    around, so the major printed here is the one read off the driver.

    The external-toolkit routes survive as commented alternatives, for an
    estate that wants a real system toolkit or holds a pinned CuPy older
    than 14.0 (which has no ``[ctk]`` extra to ask for).  Commented, not
    live: a remedy is pasted whole, and a paste that installs a toolkit
    two different ways is a worse install than the one it repaired.
    """

    major = box_major if box_major in _GPU_EXTRA_BY_MAJOR else None
    pin = f"={major}" if major else ""
    if bridges.WINDOWS_SHELL:
        point_at_conda = "#   $env:CUDA_PATH = $env:CONDA_PREFIX"
    else:
        point_at_conda = '#   export CUDA_PATH="$CONDA_PREFIX"'
    lines = [
        "# the CuPy wheel ships NVRTC -- the COMPILER -- and no headers.",
        "# CuPy's own kernels include the CUDA runtime headers, so on a",
        "# box with no CUDA toolkit the first device call of any kind --",
        "# a reduction, a sort, the first cuBLAS load behind a matmul --",
        "# dies with 'Failed to find CUDA headers'.  Reinstalling the",
        "# cupy wheel or the woof GPU extra CANNOT fix that: it",
        "# reinstalls the compiler, which is already here.",
        "#",
        "# The header tree has to MATCH that compiler's major, and CuPy",
        "# publishes exactly that as an extra of its own wheel:",
    ]
    if major is None:
        lines += [
            "# this box's CUDA major could not be read, and the wheel is",
            "# chosen by it, so read it first:",
            "nvidia-smi",
            "# ...then run the line that matches what it prints:",
            f"#   {_cupy_ctk_install(12)}",
            f"#   {_cupy_ctk_install(13)}",
        ]
        action = ("nvidia-smi  # then pip install "
                  "'cupy-cuda12x[ctk]' or 'cupy-cuda13x[ctk]' to match")
    else:
        lines += [
            f"# this box's driver serves CUDA {box_major}, so it is this one",
            _cupy_ctk_install(major),
        ]
        action = _cupy_ctk_install(major)
    lines += [
        "#",
        "# That pulls the CUDA runtime and its headers, NVRTC, cuBLAS,",
        "# cuFFT, cuRAND, cuSOLVER and cuSPARSE as wheels: about 1.6 GiB",
        "# installed, measured on a driver-only box.  woof's own GPU",
        "# extras ask for the same set from 2.5.0 on, so a FRESH install",
        "# needs nothing beyond the documented pip line; this repairs an",
        "# environment that already exists, or one holding a CuPy older",
        "# than 14.0, which has no [ctk] extra to ask for.",
        "#",
        "# ALTERNATIVES, for an estate that installs a real toolkit",
        "# instead.  conda puts one into the active environment on both",
        "# platforms without administrator rights:",
        f"#   conda install -c nvidia cuda-toolkit{pin}",
        f"#   conda install -c conda-forge cuda-toolkit{pin}",
        "# ...and if CUDA_PATH did not get set on activation, or the",
        "# toolkit came from NVIDIA's own installer, point it at the",
        "# toolkit root yourself:",
        point_at_conda,
        "# ...or take the NVIDIA wheels on their own, which carry the",
        "# same header tree:",
        "#   " + _cuda_wheel_install(_CUDA_TOOLKIT_PACKAGES, major),
        "#   python -c \"import nvidia.cuda_runtime as r,pathlib;"
        "print(pathlib.Path(r.__file__).parent)\"",
    ]
    return "\n".join(lines), action


#: Markers that name the TOOLKIT HEADERS as the missing piece, wherever
#: the message surfaced -- an import, the cuBLAS pairing probe, or a
#: kernel compile.  CuPy says "Failed to find CUDA headers" in those
#: words, and names the ``CUDA_PATH``/``nvcc``/header file it consulted
#: looking for them.
#:
#: This is a FAILURE CLASS, not a probe: CuPy 14 raises the same error
#: from the first device call of any kind, so the branch that owns the
#: cuBLAS probe meets it too and used to call it a wrong wheel.
_HEADERS_FAILURE_MARKERS = ("cuda headers", "cuda_path", "cuda_home",
                            "nvcc", "cuda toolkit", "cuda_fp8.hpp",
                            "cuda_runtime.h")


def _names_missing_cuda_headers(text: object) -> bool:
    """Does this failure message name the toolkit-headers class?"""

    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _HEADERS_FAILURE_MARKERS)


#: Messages that say the probed CARD had no memory left, not that the
#: install is wrong.  Named breakage: on a shared 4-card node whose card 0
#: another process had filled (96.9 of 97.9 GB), `bash install.sh` exited 1
#: and doctor printed "MISSING cupy ... -> pip install 'recast-woof[gpu-cu13]'",
#: "MISSING CUDA kernel headers" and "MISSING radar-DA eigensolver" for a
#: correct install, because every probe opens the first visible card and
#: an allocation of 512 bytes failed there.
_DEVICE_FULL_MARKERS = ("out of memory", "outofmemoryerror",
                        "cudaerrormemoryallocation",
                        "cuda_error_out_of_memory")


def _names_full_device(*texts: object) -> bool:
    """Does any of these failure messages say the card was out of memory?"""

    return any(marker in str(text or "").lower()
               for text in texts for marker in _DEVICE_FULL_MARKERS)


def _full_device_check(name: str, evidence: object, *,
                       status: str = "info") -> Check:
    """Not judged: the probed card is full, which says nothing of the install."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    card = (f"CUDA_VISIBLE_DEVICES={visible}" if visible
            else "the first card (CUDA_VISIBLE_DEVICES is unset)")
    return Check(
        name, status,
        f"not judged -- the probe ran on {card}, which had no memory to "
        f"give: {_short(str(evidence), 96)}.  Another process holds that "
        "card; the install is not what failed",
        "# point the probe at a card with free memory, then re-run doctor:\n"
        "nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv\n"
        "CUDA_VISIBLE_DEVICES=<a free index or GPU-uuid> woof doctor",
        action="CUDA_VISIBLE_DEVICES=<free card> woof doctor",
        brief="card full, not judged", blocking=False)


#: The markers that name the WHEEL: a shared library the wheel
#: hard-codes and the loader could not produce.  (Their opposite number,
#: the toolkit-headers class, is :data:`_HEADERS_FAILURE_MARKERS`, which
#: lives beside the remedy it selects because three checks read it now.)
_WHEEL_IMPORT_MARKERS = ("no module named", "cannot open shared object",
                         "libcublas", "libnvrtc", "libcudart",
                         "dll load failed")


def _cupy_import_failure_remedy(evidence: str,
                                box_major: int | None) -> tuple[str, str]:
    """``(remedy, action)`` for a CuPy that is INSTALLED and will not import.

    THE 2.2.1 FIELD CASE, on Ubuntu.  Doctor printed ``MISSING cupy --
    ... Failed to find CUDA headers ...`` and offered ``pip install
    'recast-woof[gpu-cu13]'``.  The wheel was already installed; what was absent
    was the CUDA TOOLKIT, and no woof extra has ever carried one, so the
    reader watched pip report success and the fault survive.  The line that
    actually fixed that box was a toolkit install.

    The two faults are separable from the import's own message, cheaply
    and without a second probe: CuPy names the header tree or the
    ``CUDA_PATH``/``nvcc`` it consulted when the toolkit is what is
    missing, and names a shared library it could not load when the wheel
    is.  Where neither signature appears the symptom really is ambiguous,
    so BOTH remedies print, each LABELLED by the symptom it answers --
    never one of them silently chosen, which is how this bug read to the
    user who hit it.  The single action stays the toolkit line, because
    that is the branch a fresh box lands on and the branch the wheel
    remedy cannot fix.
    """

    lowered = evidence.lower()
    if _names_missing_cuda_headers(evidence):
        return _cuda_headers_remedy(box_major)
    if any(marker in lowered for marker in _WHEEL_IMPORT_MARKERS):
        return _gpu_extra_hint(box_major)
    headers_remedy, headers_action = _cuda_headers_remedy(box_major)
    wheel_remedy, _ = _gpu_extra_hint(box_major)
    return "\n".join([
        "# This import failed for one of two reasons and its message names",
        "# neither, so both remedies are printed.  Match yours by symptom.",
        "#",
        "# SYMPTOM: the message names a header, CUDA_PATH or nvcc.  The CUDA",
        "# TOOLKIT is missing, and no wheel or woof extra supplies one:",
        headers_remedy,
        "#",
        "# SYMPTOM: the message names a shared library or a missing module.",
        "# The CuPy wheel itself is absent or wrong for this box:",
        wheel_remedy,
    ]), headers_action


def _cuda_headers_check(*, reuse_bound=False) -> Check:
    """Can CuPy compile and run here, and which measured stage failed?

    A check rather than a footnote because the gap is silent by
    construction: cupy imports, cuBLAS loads, a matmul returns the right
    answer, and doctor called that verified -- while the first uncached
    reduction of a real run died on a missing header.  Everything cheaper
    than a cold compile passes on a box that cannot compile.

    A proven failure blocks forecast readiness and setup's final verdict.
    Preprocessing remains usable. An unavailable or timed-out probe is
    explicitly unjudged; ordinary measured ``check`` requires verification.
    """

    name = "CUDA kernel headers"
    if no_local_gpu():
        return Check(
            name, "info",
            "not judged -- compiling a kernel is device contact and "
            "GPUWM_NO_LOCAL_GPU is set",
            brief="device not touched", blocking=False)
    result = (_nvrtc_header_probe(reuse_bound=True) if reuse_bound
              else _nvrtc_header_probe())
    if "cupy" in result:
        return Check(
            name, "info",
            f"not judged -- kernels compile through cupy, which is "
            f"unavailable here ({result['cupy']})",
            brief="needs cupy", blocking=False)
    if "devices" in result:
        reason = result.get("device_error") or "no CUDA device visible"
        return Check(
            name, "info",
            f"not judged -- compiling needs a device: {_short(str(reason), 96)}",
            brief="no device", blocking=False)
    if "slow" in result:
        return Check(
            name, "info",
            f"not judged -- the cold compile did not finish "
            f"({result['slow']}); a first compile on a busy box is slow",
            brief="compile timed out", blocking=False)
    if "probe" in result:
        return Check(
            name, "missing",
            f"the compile probe could not be run: {result['probe']}",
            "woof doctor --explain\n# Read the probe process error above before choosing a repair.",
            action="woof doctor --explain",
            brief="CUDA probe unavailable", blocking=True)

    self_contained = result.get("self_contained", "did not report")
    headers = result.get("toolkit_headers", "did not report")
    cuda_path = result.get("cuda_path") or "unset"
    if _names_full_device(self_contained if self_contained != "ok" else "",
                          headers if headers != "ok" else ""):
        return _full_device_check(
            name, self_contained if self_contained != "ok" else headers)
    if self_contained == "ok" and headers == "ok":
        cache = result.get("readiness_cache")
        detail = (
            "a self-contained kernel and a cupy reduction ran again; "
            "prior cold compilation bound to unchanged compiler/header bytes"
            if cache == "bound-warm" else
            "a self-contained kernel and a cupy reduction compiled and ran "
            "from a COLD cache")
        return Check(
            name, "verified", f"{detail} (CUDA_PATH {cuda_path})",
            brief="kernels compile and run", blocking=False)
    stage = result.get("self_contained_stage")
    if stage in {"allocation", "launch", "synchronize", "readback", "validation"}:
        return _cuda_probe_failure(name, str(stage), self_contained, compiled=True)
    if self_contained != "ok" and _names_missing_cuda_headers(self_contained):
        # CuPy 14 raises the headers error BEFORE it compiles anything,
        # so a kernel with no ``#include`` at all fails too and the
        # two-kernel split cannot separate the pair on its own.  Falling
        # through to the wheel branch here is how a driver-only box was
        # told "NVRTC ships inside the cupy wheel, so this is the wheel"
        # and sent to reinstall the wheel that is present and correct.
        # The message names the class; the class picks the remedy.
        box_major = _driver_cuda_major()
        remedy, action = _cuda_headers_remedy(box_major)
        return Check(
            name, "missing",
            f"no kernel compiles on this box, including one with no "
            f"#include at all: {_short(str(self_contained), 120)}.  That "
            f"names the toolkit HEADER tree, which no cupy wheel ships and "
            f"no wheel reinstall supplies (CUDA_PATH {cuda_path})",
            remedy, action=action, brief="toolkit headers missing",
            blocking=True)
    if self_contained == "ok" and _names_missing_cuda_headers(headers):
        # THE DISTINCTION THIS CHECK EXISTS FOR.  NVRTC works, so the
        # wheel is fine and reinstalling it is wasted advice; what is
        # missing is the header tree NVRTC was asked to read.
        box_major = _driver_cuda_major()
        remedy, action = _cuda_headers_remedy(box_major)
        return Check(
            name, "missing",
            f"NVRTC works -- a self-contained kernel compiled -- but a cupy "
            f"reduction did not: {_short(str(headers), 120)}.  That is the "
            f"toolkit HEADER tree, which no cupy wheel ships and no wheel "
            f"reinstall supplies (CUDA_PATH {cuda_path})",
            remedy, action=action, brief="toolkit headers missing",
            blocking=True)
    if self_contained == "ok":
        return _cuda_probe_failure(name, str(result.get("toolkit_headers_stage", "reduction")),
                                   headers, compiled=True)
    if not any(marker in str(self_contained).lower() for marker in _WHEEL_IMPORT_MARKERS):
        return _cuda_probe_failure(name, str(stage or "compile"), self_contained,
                                   compiled=bool(result.get("self_contained_compiled")))
    # The compiler's library/module loader actually names a missing wheel
    # component. Execution failures never reach this installer branch.
    box_major = _driver_cuda_major()
    remedy, action = _gpu_extra_hint(box_major)
    return Check(
        name, "missing",
        f"cupy could not compile even a self-contained kernel: "
        f"{_short(str(self_contained), 120)}.  NVRTC ships inside the cupy "
        f"wheel, so this is the wheel rather than the toolkit headers",
        remedy, action=action, brief="nvrtc unusable", blocking=True)


def _cuda_probe_failure(name: str, stage: str, evidence: object, *, compiled: bool) -> Check:
    progress = "The self-contained kernel compiled; " if compiled else ""
    return Check(
        name, "missing", f"{progress}CUDA {stage} failed: {_short(str(evidence), 160)}",
        "nvidia-smi\n# Inspect the reported CUDA driver/device error, then rerun woof doctor.",
        action="nvidia-smi", brief=f"CUDA {stage} failed", blocking=True)


#: The CUDA libraries cuSOLVER needs present TOGETHER.  Every one is a
#: dependency OF cusolver, not a nicety: cusolver links cublas and cusparse,
#: and on Windows a wheel's DLLs are only visible to a compiled extension
#: through ``os.add_dll_directory``, never through PATH.
_CUSOLVER_PACKAGES = ("nvidia-cusolver", "nvidia-cublas", "nvidia-cusparse")

#: The toolkit pieces a header gap needs from pip when the estate has no
#: conda: the CUDA runtime, whose wheel carries the header tree CuPy's own
#: reduction kernels include, and NVRTC beside it so the compiler and the
#: headers it reads come from the same major.
_CUDA_TOOLKIT_PACKAGES = ("nvidia-cuda-runtime", "nvidia-cuda-nvrtc")


def _cuda_wheel_install(packages: tuple[str, ...], major: int | None) -> str:
    """The pip line that installs these CUDA libraries for ``major``.

    NVIDIA HAS DEPRECATED THE ``-cuXX`` SUFFIXED NAMES -- ``-cu12`` as well
    as ``-cu13`` -- and the suffixed distributions survive as tombstones:
    ``pip install nvidia-cusolver-cu13`` resolves, downloads, reports
    success, and installs a package whose entire content is a notice saying
    to use ``nvidia-cusolver`` instead.  A remedy that prints one leaves the
    box exactly as broken as before while the transcript says it was fixed
    -- the NVRTC shadow trap with extra steps.  So every name printed here
    is the unsuffixed one, at EVERY major; a remedy that spelled the suffix
    for 12 and dropped it for 13 was one NVIDIA deprecation away from being
    wrong again, and that deprecation has now landed.

    The major does not disappear with the suffix, it MOVES: out of the
    package name and into a version pin, ``nvidia-cusolver=="12.*"``, so
    the line still matches the box it was generated for.  The pin is
    derived from the detected driver and never hardcoded; with no major in
    hand the line carries no pin and the caller prints the lookup that
    finds one.  Double quotes, because ``12.*`` is a glob to sh and a
    string to PowerShell only when it is quoted, and the same spelling has
    to survive both.  ``--no-deps`` is what keeps the unsuffixed packages
    from dragging those same tombstones back in as dependencies.
    """

    if major is None:
        return "pip install --no-deps " + " ".join(packages)
    pins = " ".join(f'"{name}=={major}.*"' for name in packages)
    return "pip install --no-deps " + pins


def _cusolver_hint(box_major: int | None) -> tuple[str, str]:
    """``(remedy, action)`` for a device whose cuSOLVER is missing.

    The install line is a property of the BOX's CUDA major, read from the
    driver, for the same reason the CuPy extra is: until this release the
    line was a frozen ``-cu12`` spelling, so a CUDA-13 box asking doctor
    how to get its eigensolver back was handed three tombstone packages.
    The names are unsuffixed now at every major (see
    :func:`_cuda_wheel_install`), so the major rides in the version pin;
    with no major in hand both pins are printed and neither is presented
    as the default, because a silent default is exactly how a CUDA-13 box
    ends up installing CUDA 12 libraries.
    """

    preamble = (
        "# only needed for eigensolver='library', or for an ensemble larger\n"
        "  # than this project's own kernel supports.  The default analysis\n"
        "  # does not use cuSOLVER at all, so this is optional.\n"
        "  #\n"
        "  # FIRST check whether it is installed but merely unreachable, which\n"
        "  # is the commoner fault and which installing again will not fix:\n"
        "  python -c \"import sys,pathlib;print([str(p) for p in pathlib.Path(sys.prefix).rglob('*cusolver*') if p.is_dir()])\"\n"
        "  # If that prints a directory, the library is present and the loader\n"
        "  # cannot see it.  On Linux put that directory on $LD_LIBRARY_PATH;\n"
        "  # on Windows PATH is not enough -- since Python 3.8 a compiled\n"
        "  # extension resolves its dependants through os.add_dll_directory()\n"
        "  # only.  If it prints nothing, install it:")
    tombstone = (
        "  # (the -cu12 and -cu13 spellings of these are deprecation\n"
        "  # tombstones: they install cleanly and supply nothing)")
    if box_major is None:
        return ("\n".join([
            preamble,
            "  # NVIDIA has deprecated the -cuXX suffixes, so the package",
            "  # names no longer carry the major -- the version pin does,",
            "  # and it has to match the major this box serves.  Read it:",
            "  nvidia-smi",
            "  # then, for a CUDA 12.x box:",
            f"  {_cuda_wheel_install(_CUSOLVER_PACKAGES, 12)}",
            "  # or, for a box whose CUDA is 13-only:",
            f"  {_cuda_wheel_install(_CUSOLVER_PACKAGES, 13)}",
            tombstone,
        ]), "nvidia-smi  # then install the CUDA libraries for that major")
    install = _cuda_wheel_install(_CUSOLVER_PACKAGES, box_major)
    return ("\n".join([
        preamble,
        f"  # this box's driver serves CUDA {box_major}, so:",
        tombstone,
        f"  {install}",
    ]), install)


#: Solves one tiny symmetric eigenproblem two ways and reports which worked.
#: Deliberately the SAME two calls the analysis makes, because the fault this
#: check exists for is invisible to anything cheaper: cupy's core loads
#: independently of cuBLAS and cuSOLVER, so "cupy imports" and even "cupy
#: multiplies arrays" are both true on a box where the factorisation cannot
#: run at all.
#:
#: TWO PROPERTIES OF THIS SCRIPT ARE ESSENTIAL.  Do not tidy them away.
#:
#: 1. It runs in a FRESH PROCESS (see :func:`_eigensolver_probe`), and
#: 2. it calls NOTHING from ``cupy.linalg`` before ``eigh``.
#:
#: Measured on a rented Ada node, 2026-08-05, cupy 14.1.1, where the wheel's
#: ``libcusolver.so.11`` was installed but off the loader path:
#:
#:     $ python -c "import cupy; cupy.linalg.eigh(A)"
#:     ImportError: libcusolver.so.11: cannot open shared object file
#:
#:     $ python -c "import cupy; cupy.linalg.inv(A); cupy.linalg.eigh(A)"
#:     ok
#:
#: ``inv`` reaches cuSOLVER through ``cupy_backends.cuda.libs.cusolver``,
#: whose own link resolution finds the wheel; batched ``eigh`` reaches it
#: through ``cupyx.cusolver``, whose does not -- and once the first call has
#: pulled the library into the process, the second one finds it already
#: loaded.  So cuSOLVER's availability on such a box is a property of what
#: the process happened to do first, which is why "it worked yesterday" is a
#: real thing people say about this fault.  A probe that shares a process
#: with anything else, or that warms up with a different factorisation,
#: reports green on a box where the analysis will fail.
_EIGENSOLVER_PROBE = """
import json, numpy, sys
out = {}
try:
    import cupy
except Exception as error:
    sys.stdout.write(json.dumps({"cupy": repr(error)}))
    raise SystemExit(0)
rs = numpy.random.RandomState(0)
b = rs.standard_normal((8, 6, 6))
a = b @ b.transpose(0, 2, 1) + 6.0 * numpy.eye(6)
a = 0.5 * (a + a.transpose(0, 2, 1))
# The second batch is above 64, the global-work tier an ensemble of more
# than 64 members takes; both tiers must solve for the default to be whole.
c = rs.standard_normal((2, 66, 66))
c = c @ c.transpose(0, 2, 1) + 66.0 * numpy.eye(66)
c = 0.5 * (c + c.transpose(0, 2, 1))
try:
    from woof.core.jacobi_eigh import batched_eigh
    residual = 0.0
    for m in (a, c):
        w, v = batched_eigh(cupy.asarray(m))
        got = cupy.asnumpy(v @ (w[:, :, None] * v.transpose(0, 2, 1)))
        residual = max(residual,
                       float(numpy.abs(got - m).max() / numpy.abs(m).max()))
    out["jacobi"] = "ok" if residual < 1e-12 else f"residual {residual:.2e}"
except Exception as error:
    out["jacobi"] = f"{type(error).__name__}: {error}"
try:
    cupy.linalg.eigh(cupy.asarray(a))
    out["library"] = "ok"
except Exception as error:
    out["library"] = f"{type(error).__name__}: {error}"
sys.stdout.write(json.dumps(out))
"""


def _eigensolver_probe() -> dict:
    """Run :data:`_EIGENSOLVER_PROBE` out of process; ``{}`` if it could not."""
    if find_spec("cupy") is None:
        return {"cupy": "not installed"}
    return _json_probe(_EIGENSOLVER_PROBE)


def _da_eigensolver_check() -> Check:
    """Can the radar-DA analysis factor its matrices, and with what?

    This is the one line in the report that exists because of a specific
    field failure rather than a category of them.  Twice -- once on a box
    whose NVIDIA wheels were installed but invisible to a compiled extension,
    once on a rented node whose CUDA install simply shipped without it -- the
    DA path died inside cuSOLVER, and both times the surrounding evidence
    said the GPU was healthy, because for everything except the
    factorisation it was.  ``woof doctor`` now answers the question
    directly instead of leaving it to be inferred.

    Not blocking: an install that never runs data assimilation never touches
    either solver, and an install that does has this project's own kernel.
    """
    name = "radar-DA eigensolver"
    # The install spelling is the BOX's property, not a constant: see
    # _cusolver_hint.  Read once, so all four branches agree.
    cusolver_remedy, cusolver_action = _cusolver_hint(_driver_cuda_major())
    result = _eigensolver_probe()
    if "cupy" in result:
        return Check(
            name, "info",
            "not judged -- data assimilation runs on the device and cupy is "
            f"unavailable here ({result['cupy']})",
            brief="needs cupy", blocking=False)
    if "probe" in result:
        return Check(
            name, "missing",
            f"the probe could not be run: {result['probe']}",
            cusolver_remedy, action=cusolver_action,
            brief=_short(result["probe"]), blocking=False)

    jacobi = result.get("jacobi", "did not report")
    library = result.get("library", "did not report")
    if jacobi != "ok" and library != "ok" and _names_full_device(jacobi, library):
        return _full_device_check(name, jacobi)
    if jacobi == "ok":
        # The supported path works, so cuSOLVER is genuinely optional and
        # its absence is news rather than a gap.
        extra = ("cuSOLVER also available"
                 if library == "ok"
                 else f"cuSOLVER unavailable ({_short(library)}), which the "
                      "default analysis does not need")
        return Check(
            name, "verified",
            f"this project's batched Jacobi kernel solved and reconstructed "
            f"a test batch; {extra}",
            None if library == "ok" else cusolver_remedy,
            action=None if library == "ok" else cusolver_action,
            brief=("jacobi kernel ok, cuSOLVER ok" if library == "ok"
                   else "jacobi kernel ok, no cuSOLVER"),
            blocking=False)
    if library == "ok":
        return Check(
            name, "present",
            f"cuSOLVER works, but this project's own kernel did not: "
            f"{_short(jacobi)}.  Analyses still run, through "
            "eigensolver='library'",
            "# report this: the bundled kernel should build wherever cupy\n"
            "  # does.  Meanwhile set LetkfConfig(eigensolver='library')\n"
            "  # or run the analysis on the host",
            action="set eigensolver='library'",
            brief=_short(jacobi), blocking=False)
    return Check(
        name, "missing",
        f"neither solver works here -- bundled kernel: {_short(jacobi)}; "
        f"cuSOLVER: {_short(library)}.  Data assimilation cannot run on "
        "this device; forecasts are unaffected",
        cusolver_remedy, action=cusolver_action,
        brief="no device eigensolver", blocking=False)


# ---------------------------------------------------------------------------
# The pip extras: enumerated from the INSTALLED distribution, judged by import
# ---------------------------------------------------------------------------
#
# Doctor probed cupy, wrf and matplotlib and NOTHING ELSE on the Python
# side.  A 2026-08-14 reachability audit ran `woof doctor --explain` on
# a bare install and measured the consequence: 40,351 characters of
# report in which the strings `scipy`, `pyshp`, `shapefile`, `rasterio`
# and `pyproj` appeared ZERO times.  Four extras -- `obs`, `dealias`,
# `geog`, and half of `render` -- were invisible to the one command
# whose whole job is telling a user what their install cannot do.  The
# same report called `[render]` "wrf-rust + matplotlib" while the extra
# is wrf-rust + pyshp and matplotlib is a BASE dependency, so the one
# line a reader could have followed to the shapefile reader said the
# wrong package.
#
# The enumeration below is therefore NOT a transcription of
# pyproject.toml.  It is read from the installed distribution's own
# metadata, because that is what pip resolved onto this box, and a
# checkout sitting beside a wheel is exactly the configuration where a
# transcription and the truth part company.  What this module still
# declares by hand is the thing metadata cannot carry: which documented
# command each extra is the difference between running and not.

#: The distribution whose metadata declares the extras.  Named once.
_DISTRIBUTION = "recast-woof"

#: The leading name of a ``Requires-Dist`` line, and the ``extra ==``
#: marker that assigns it to one.
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA_MARKER = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")


def _canonical(name: str) -> str:
    """PEP 503 normalisation: one spelling two tables can agree on."""

    return re.sub(r"[-_.]+", "-", name).strip().lower()


#: pip distribution -> the name ``import`` actually uses, for every
#: package this project declares.  The two differ often enough that
#: guessing is how a check reports green on an absent package:
#: ``pyshp`` imports as ``shapefile``, ``wrf-rust`` as ``wrf``,
#: ``cupy-cuda12x`` as ``cupy``.  A distribution missing from this table
#: is still CHECKED -- by its installed metadata -- but the report says
#: that is all it was, rather than claiming an import nobody attempted.
#: ``tests/test_doctor_extras.py`` fails if this project declares a
#: requirement this table does not name.
_IMPORT_NAME = {
    "cupy-cuda12x": "cupy",
    "cupy-cuda13x": "cupy",
    "wrf-rust": "wrf",
    "pyshp": "shapefile",
    "scipy": "scipy",
    "rasterio": "rasterio",
    "pyproj": "pyproj",
    "numpy": "numpy",
    "netcdf4": "netCDF4",
    "matplotlib": "matplotlib",
    "jsonschema": "jsonschema",
    "threadpoolctl": "threadpoolctl",
    "pytest": "pytest",
    "pytest-xdist": "xdist",
    "psutil": "psutil",
    # The YAML reader ([dev] extra): the release contract tests read
    # the publication workflow through it.
    "pyyaml": "yaml",
    # The wheel-gate tests' build backend ([dev] extra): they run setup.py
    # with the battery's interpreter.
    "setuptools": "setuptools",
    # The undefined-name gate's linter ([dev] extra):
    # tests/test_no_undefined_names.py runs it.
    "ruff": "ruff",
    # The CDS client ([era5] extra): distribution and module share the name.
    "cdsapi": "cdsapi",
    "huggingface-hub": "huggingface_hub",
    "h5py": "h5py",
    # The companion distribution: pure reference tables (RRTMGP,
    # Thompson), pinned `==` and installed by a bare `pip install
    # woof`.  The probe imports it like any other, which is exactly
    # what a reader needs -- an importable woof_data is what
    # woof.data_assets resolves against.
    "recast-woof-data": "woof_data",
    # The 7z decoder ([hydro-conus] extra) for EPA's NHDPlus V2.1 archives.
    "py7zr": "py7zr",
}


#: Distribution -> the check that already judges it, deeply.  Listed
#: here so the extras/base block NAMES the package and points at the
#: verdict instead of publishing a second one.  Two lines disagreeing
#: about one package is worse than one line saying less.
#: The CuPy wheels are deliberately NOT here.  Their extras report
#: presence without a verdict (:func:`_mutually_exclusive_extra_check`),
#: so naming which wheel pip resolved is the whole content of those two
#: lines; filtering them out left the line saying " not installed" about
#: nothing at all.
_DEEP_CHECKED = {
    "rasterio": "geography stack (rasterio + pyproj)",
    "pyproj": "geography stack (rasterio + pyproj)",
    # Importability is not the question for the companion -- a skewed
    # one imports perfectly.  :func:`_companion_pair_check` compares the
    # versions, so the base-dependency line names the pin and points
    # there instead of printing `ok` over an install that is not.
    "recast-woof-data": "recast-woof-data",
}


@dataclass(frozen=True)
class _Requirement:
    """One ``Requires-Dist`` line, split into the parts a check needs."""

    #: PEP 503 name, which is what pip installed it under.
    distribution: str
    #: The requirement without its environment marker -- the half a
    #: reader recognises (``scipy>=1.11``).
    specifier: str
    #: ``recast-woof[render]`` and friends: an extra that pulls another extra
    #: rather than a package.  Reported as the alias it is.
    alias: bool
    #: The environment marker MINUS its ``extra ==`` clause -- the half
    #: that decides whether pip installs this requirement on THIS
    #: interpreter at all.  ``""`` when there is none.
    marker: str = ""


#: The ``extra == "..."`` clause and the boolean glue around it.  Removed
#: rather than evaluated: the extra is how the requirement was filed in
#: the first place, so what is left is the part that answers "would pip
#: install this here".
_EXTRA_CLAUSE = re.compile(
    r"""\s*(?:and\s+|or\s+)?extra\s*==\s*['"][^'"]+['"]\s*""")


def _parse_requirement(text: str) -> _Requirement | None:
    match = _REQUIREMENT_NAME.match(text)
    if match is None:
        return None
    distribution = _canonical(match.group(1))
    head, _, marker = text.partition(";")
    return _Requirement(distribution, head.strip(),
                        distribution == _canonical(_DISTRIBUTION),
                        _EXTRA_CLAUSE.sub(" ", marker).strip(" and or\t"))


def _marker_gap(requirement: "_Requirement") -> str | None:
    """The sentence that names why this build excludes ``requirement``.

    A marker tells a reader that pip skipped something; it does not tell
    them WHY, and "why" is the whole content of a gap report.  This is
    how the project says it out loud.  Anything excluded with no answer
    here still reports the marker verbatim -- an unnamed exclusion is a
    defect to surface, not one to hide.

    Asked, not tabled.  The one gap this project has ever recorded is the
    science core's interpreter ceiling, and that ceiling is an
    availability fact about an index that moves: wrf-rust 0.2.39 closed
    the 3.14 gap that 0.2.38 opened.  A table entry frozen at import
    would have gone on printing the closed one, so the answer is computed
    from :data:`science_core.SCIENCE_CORE_PYTHON_CEILING` at the moment a
    reader asks, and only when this interpreter is genuinely over it.
    """

    if requirement.distribution != _canonical(
            science_core.SCIENCE_CORE_DISTRIBUTION):
        return None
    found = _python_version()
    if science_core.python_supports_science_core(found):
        # Excluded for some reason that is NOT the ceiling; naming the
        # ceiling here would be a confident wrong answer.
        return None
    return science_core.python_gap_sentence(found)


def _python_version() -> tuple[int, ...]:
    """This interpreter's version, as one patchable seam.

    A function so the marker branches can be exercised from a suite that
    is not running on the interpreter under test; patching ``sys``
    itself would reach every other consumer in the process.
    """

    return tuple(sys.version_info[:3])


def _marker_excludes_this_interpreter(requirement: _Requirement) -> bool:
    """Would pip SKIP this requirement here, marker and all?

    ``packaging`` answers it exactly and is present on every install
    (matplotlib, a base dependency, requires it), so it is tried first.
    The fallback covers only ``python_version`` comparisons, which is the
    one marker this project authors, and is documented as partial rather
    than presented as an evaluator: a marker it cannot read reports
    "not excluded", which is the pre-marker behaviour and never invents
    an exclusion that pip did not make.
    """

    marker = requirement.marker
    if not marker:
        return False
    found = _python_version()
    try:
        from packaging.markers import Marker

        # The interpreter comes from the same seam the rest of this
        # module reports, never from ``packaging``'s own view of the
        # process: a check that judges one interpreter and reports
        # another is the class of defect this file exists to catch.
        return not Marker(marker).evaluate({
            "python_version": f"{found[0]}.{found[1]}",
            "python_full_version": ".".join(str(part) for part in found)})
    except ImportError:
        pass
    except Exception:
        return False
    found = found[:2]
    for operator, literal in re.findall(
            r"""python_version\s*(<=|>=|==|!=|<|>)\s*['"]([0-9.]+)['"]""",
            marker):
        want = tuple(int(part) for part in literal.split(".") if part.isdigit())
        here = found[:len(want)]
        satisfied = {
            "<": here < want, "<=": here <= want, ">": here > want,
            ">=": here >= want, "==": here == want, "!=": here != want,
        }[operator]
        if not satisfied:
            return True
    return False


def declared_requirements() -> tuple[tuple[_Requirement, ...],
                                     dict[str, tuple[_Requirement, ...]]] | None:
    """``(base requirements, extra -> requirements)`` for THIS install.

    ``None`` when the interpreter has no installed ``woof``
    distribution to read -- a source tree on ``PYTHONPATH`` and nothing
    else.  That case is reported as unanswered rather than guessed at,
    because the alternative is reading a checkout's ``pyproject.toml``
    and calling it the estate, which is the exact substitution this
    module exists to refuse.
    """

    try:
        declared = importlib.metadata.requires(_DISTRIBUTION) or []
        metadata = importlib.metadata.metadata(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None
    base: list[_Requirement] = []
    extras: dict[str, list[_Requirement]] = {
        _canonical(name): []
        for name in (metadata.get_all("Provides-Extra") or [])}
    for text in declared:
        requirement = _parse_requirement(text)
        if requirement is None:
            continue
        marker = _EXTRA_MARKER.search(text)
        if marker is None:
            base.append(requirement)
            continue
        extras.setdefault(_canonical(marker.group(1)), []).append(requirement)
    return tuple(base), {name: tuple(items)
                         for name, items in extras.items()}


@dataclass(frozen=True)
class _ExtraFacts:
    """What one extra is the difference between running and not.

    ``doors`` are the documented commands a box WITHOUT the extra
    cannot run at all.  ``still_works`` is what such a box can still do,
    and it is not decoration: a reader who sees ``[render]`` reported
    MISSING beside a working ``woof render`` deserves to be told why
    both are true (the rust engine needs no Python package; the
    matplotlib engine and ``woof enprod`` do).

    ``blocking`` follows ONE rule, stated in :func:`blocking_gaps`: an
    extra blocks exactly when a command the product's own README or
    FIRST-LIGHT tells a user to run cannot run without it.
    """

    doors: tuple[str, ...]
    still_works: str | None
    blocking: bool
    severity: str
    #: Named when another check owns the verdict, so the extras section
    #: reports the install line without double-counting the gap.
    deferred_to: str | None = None
    #: One of a set where installing every member would be the fault:
    #: the CuPy pair, one wheel per CUDA major.  Such a line never
    #: carries a gap, because a correctly installed box is missing the
    #: other one by definition.
    mutually_exclusive: bool = False
    #: Absent is this extra's NORMAL state, so its absence is context
    #: rather than a gap.  `[dev]` is the case: a user install is
    #: supposed not to have pytest, and printing MISSING at every reader
    #: who never intends to run the suite spends the word MISSING on
    #: nothing.  The line still prints, with its install command.
    expected_absent: bool = False


#: Which extra unlocks what.  Hand-declared because no metadata carries
#: it, and audited against the code in ``tests/test_doctor_extras.py``:
#: every door named here must be a real door, and every extra the
#: installed metadata declares must appear here.
_EXTRA_FACTS: dict[str, _ExtraFacts] = {
    # The LGPL-2.1+ 7z decoder for EPA's NHDPlus V2.1 source archives.  Absent
    # is normal: `--hydro on` over CONUS then prepares the worldwide fabric
    # and says why (woof.hydro.acquire), so nothing stops.
    "hydro-conus": _ExtraFacts(
        doors=("woof go --hydro on over CONUS with the NHDPlus V2.1 provider",),
        still_works="every forecast door, and --hydro on itself, which falls "
                    "back to the worldwide fabric",
        blocking=False, severity=SEVERITY_DEGRADED,
        deferred_to="py7zr (7z decoder)",
        expected_absent=True),
    "gpu-cu12": _ExtraFacts(
        doors=_GPU_DOORS, still_works=_GPU_STILL_WORKS,
        blocking=True, severity=SEVERITY_UNREACHABLE,
        deferred_to="cupy (GPU runtime)", mutually_exclusive=True),
    "gpu-cu13": _ExtraFacts(
        doors=_GPU_DOORS, still_works=_GPU_STILL_WORKS,
        blocking=True, severity=SEVERITY_UNREACHABLE,
        deferred_to="cupy (GPU runtime)", mutually_exclusive=True),
    # wrf-rust and pyshp gate DIFFERENT doors, which is why the old
    # single "render extra" line could not tell the truth about either:
    # wrf-rust is what `woof enprod` and `woof render --engine
    # matplotlib` import, and pyshp is the shapefile reader every DA
    # nowcast basemap draws with.  Nothing under `gpuwm/` imports
    # pyshp at all -- the rust renderer reads the shapefiles itself --
    # so a reader who is told "the render extra" and watches `woof
    # render` work anyway is entitled to think the report is wrong.
    "render": _ExtraFacts(
        doors=("woof enprod", "woof render --engine matplotlib",
               "woof go's render stage (it gates on the wrf package)",
               "python -m tools.da_nowcast render and the launcher's "
               "basemap endpoint (pyshp)"),
        still_works="woof render's DEFAULT rust engine, which draws from "
                    "the rw_wrfbatch binary and needs no Python package "
                    "from this extra",
        blocking=True, severity=SEVERITY_UNREACHABLE),
    "obs": _ExtraFacts(
        doors=("python -m tools.obs_battery_score",
               "python -m tools.obs_precampaign_controls",
               "python -m tools.obs_battery_registration"),
        still_works="every forecast and preprocessing route; only scoring "
                    "a run against observations needs it",
        blocking=False, severity=SEVERITY_DEGRADED),
    # NOT the default engine.  region-global became the shipped
    # --dealias-engine on 2026-08-12 and is the Rust library this report
    # checks on its own line; scipy labels the gate regions for the OTHER
    # engine only.  Saying otherwise would send a reader to pip for a
    # library their default path never loads.
    "dealias": _ExtraFacts(
        doors=("--dealias-engine vad-region, on python -m "
               "tools.obs_radar_grid_build / tools.obs_radar_grid_from_pack "
               "/ tools.da_nowcast run",),
        still_works="the shipped default --dealias-engine region-global, "
                    "which is the Rust library reported above",
        blocking=False, severity=SEVERITY_DEGRADED),
    # EMPTY from 2.3.3: rasterio and pyproj became runtime dependencies,
    # and the extra is retained so `pip install 'recast-woof[geog]'` -- written
    # down in 2.3.2's own docs and in other people's scripts -- keeps
    # resolving.  It is still reported, because "this extra now installs
    # nothing, and here is where its packages went" is exactly what a
    # reader following an older doc needs to be told.
    "geog": _ExtraFacts(
        doors=("a run whose config sets [static.highres] enabled = true "
               "(woof run / resume / static and both prepared runners)",),
        still_works="the standard WPS_GEOG static-field route, which is "
                    "what every config without that table uses",
        blocking=False, severity=SEVERITY_OPT_IN,
        deferred_to="geography stack (rasterio + pyproj)"),
    "dev": _ExtraFacts(
        doors=("pytest -- this project's own test suite",),
        still_works="everything a user runs; this extra is for developing "
                    "woof, not for using it",
        blocking=False, severity=SEVERITY_OPT_IN, expected_absent=True),
    # Maintainer-only, and absent is its normal state for the same reason
    # `[dev]`'s is: the one command behind it needs write credentials on a
    # dataset repository nobody but the maintainers owns, so no user path
    # can reach it.  It is reported rather than hidden because it is a
    # declared extra, and an inventory that quietly skips one is an
    # inventory a reader cannot trust.
    "publish": _ExtraFacts(
        doors=("python -m tools.publish_geog_mirror upload -- pushing the "
               "WPS_GEOG mirror snapshot to Hugging Face",),
        still_works="every route that READS the mirror, which is every "
                    "user-facing one; this extra is for writing it",
        blocking=False, severity=SEVERITY_OPT_IN, expected_absent=True),
}

#: Fold keys for the extras block.  Two, not one: a run of aliases
#: and a run of real extras fold to the same word otherwise, and a
#: report that prints "pip extras (3)" three times reads as one thing
#: said three times rather than three different things.
_GROUP_EXTRAS = "pip extras"
_GROUP_EXTRA_ALIASES = "pip extra aliases"

#: Groups the brief folds into ONE line wherever their members appear,
#: rather than only when they happen to be adjacent.  Exactly one group
#: qualifies, on one property: every line in it carries no information
#: a reader can act on (an alias that resolves, no gap, no command), so
#: the position of each member is not a fact worth three lines.  Nothing
#: with a remedy belongs here -- folding those would hide WHERE in the
#: estate the gap is.
_SCATTERED_FOLD_GROUPS = frozenset({_GROUP_EXTRA_ALIASES})


def _extra_install_line(extra: str) -> str:
    """The one command that installs ``extra``, as it must be typed."""

    return f"pip install 'recast-woof[{extra}]'"


def _package_evidence(requirement: _Requirement,
                      probe: dict[str, tuple[bool, str]]
                      ) -> tuple[str, str, str]:
    """``(state, label, evidence)`` for one required package.

    ``state`` is one of ``"ok"``, ``"absent"``, ``"broken"``,
    ``"untested"`` or ``"excluded"``.  ``untested`` is not a euphemism
    for ok: it is the answer for a distribution whose import name this
    module does not know, where the accurate report is that its metadata
    was read and nothing was imported.

    ``excluded`` is the state pip put the package in, not one this box
    can leave: an environment marker on the requirement means pip never
    tried to install it here.  Reporting that as ``absent`` produces the
    circular remedy in its second home -- ``wrf-rust ... not installed``
    over ``pip install 'recast-woof[render]'``, on an interpreter where that
    command installs nothing, reports success, and leaves doctor saying
    MISSING forever.
    """

    module = _IMPORT_NAME.get(requirement.distribution)
    try:
        version: str | None = importlib.metadata.version(
            requirement.distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version is None and _marker_excludes_this_interpreter(requirement):
        # Only when it is genuinely absent: a marker is about what pip
        # WOULD install, and a package that is here anyway (a source
        # build, a newer upstream release) is judged on what it does.
        gap = _marker_gap(requirement)
        return ("excluded", requirement.specifier,
                gap or (f"this build's metadata excludes it on this "
                        f"interpreter ({requirement.marker}), and the "
                        f"project has not recorded why"))
    if module is None:
        if version is None:
            return ("absent", requirement.specifier, "not installed")
        return ("untested", requirement.specifier,
                f"{version} installed per distribution metadata; NOT "
                "imported (doctor does not know this package's import "
                "name)")
    if module not in probe:
        probe[module] = _import_probe(module, requirement.distribution)
    ok, evidence = probe[module]
    label = (requirement.specifier if module == requirement.distribution
             else f"{requirement.specifier} (imports as {module})")
    if version is None:
        # The DISTRIBUTION is what pip resolved, and an import cannot
        # see it: cupy-cuda12x and cupy-cuda13x both import as `cupy`,
        # so a box with the wrong wheel for its CUDA major answers
        # `import cupy` perfectly and is still the failure this whole
        # module was rewritten around.  Metadata first, import second.
        if ok:
            return ("absent", label,
                    f"not installed -- `import {module}` does succeed, but "
                    "from a different distribution; this one is not what "
                    "pip resolved here")
        return ("absent", label, "not installed")
    if ok:
        return ("ok", label, f"{evidence} (distribution {version})")
    if _import_probe_incomplete(evidence):
        return ("untested", label, evidence)
    return ("broken", label, evidence)


#: CUDA major -> the extra whose wheel serves it.  The inverse of
#: :data:`_GPU_EXTRA_BY_MAJOR`, used to say which of a mutually
#: exclusive pair THIS box's driver can run.
_MAJOR_BY_GPU_EXTRA = {extra: major
                       for major, extra in _GPU_EXTRA_BY_MAJOR.items()}


def _driver_fit(major: int | None, box_major: int | None) -> str:
    """Can the driver read off this box run an extra's CUDA major?

    The number a driver reports is the NEWEST CUDA it serves, and a
    driver runs every older major as well: a CUDA 13 driver runs the
    cupy-cuda12x wheel, and cuBLAS loads and kernels compile on it.
    Reading that number as the one major the box accepts called a
    working ``[gpu-cu12]`` install the wrong extra on every CUDA 13
    box, while the runtime checks on the same report said it worked.
    Only an extra NEWER than the driver cannot run here.
    """

    if box_major is None:
        return ("this box's driver CUDA version could not be read, so "
                "neither of the pair is named as the one for this box")
    if major is None:
        return f"this box's driver serves CUDA {box_major} and older"
    if major > box_major:
        other = _GPU_EXTRA_BY_MAJOR.get(box_major, f"cuda-{box_major}")
        return (f"this box's driver serves CUDA {box_major} at most, "
                f"older than this extra's CUDA {major} wheel needs; the "
                f"extra that can run here is [{other}]")
    if major == box_major:
        return (f"this box's driver serves CUDA {box_major}, so this "
                f"extra's CUDA {major} wheel can run here")
    return (f"this box's driver serves CUDA {box_major} and older, so this "
            f"extra's CUDA {major} wheel can run here; a working install "
            "needs no switch")


def _mutually_exclusive_extra_check(name: str, extra: str, install: str,
                                    facts: _ExtraFacts,
                                    states: list) -> Check:
    """One of a pair where installing BOTH would be the fault.

    ``[gpu-cu12]`` and ``[gpu-cu13]`` are alternatives, one per CUDA
    major, and a healthy CUDA-12 box has exactly one of them.  Reporting
    the other as a MISSING gap would fail every correctly installed
    machine, so these lines never carry a verdict: they say which wheel
    this extra installs, whether pip resolved it here, whether this
    box's own driver can run its CUDA major, and the exact install
    line.  The verdict, the remedy and the exit code stay on the one
    deep check that judges the wheel against the box.
    """

    installed = [label for _i, state, label, _w in states if state == "ok"]
    absent = [label for _i, state, label, _w in states if state != "ok"]
    fit = _driver_fit(_MAJOR_BY_GPU_EXTRA.get(extra), _driver_cuda_major())
    held = (f"{', '.join(installed)} installed" if installed
            else f"{', '.join(absent)} not installed")
    return Check(
        name, "info",
        f"{held}.  The pair are alternatives, one CuPy per environment: "
        f"{fit}.  Install line: {install}.  Needed by: "
        + ", ".join(facts.doors)
        + f".  The verdict and the remedy are on the `{facts.deferred_to}` "
          "line, which judges the installed wheel against this box",
        brief=_short(f"{held}; {fit}"), group=_GROUP_EXTRA_ALIASES)


def _extra_check(extra: str, requirements: tuple[_Requirement, ...],
                 probe: dict[str, tuple[bool, str]]) -> Check:
    """One extra: what it holds, what it unlocks, and how to install it."""

    name = f"pip extra [{extra}]"
    install = _extra_install_line(extra)
    aliases = [item for item in requirements if item.alias]
    if aliases and len(aliases) == len(requirements):
        resolved = ", ".join(sorted(item.specifier for item in aliases))
        return Check(
            name, "info",
            f"an alias: {install} resolves to {resolved}.  The verdict is "
            "on those extras' own lines",
            brief=f"alias for {resolved}", group=_GROUP_EXTRA_ALIASES)
    facts = _EXTRA_FACTS.get(extra)
    # A package a deeper check already judges is NAMED here and judged
    # there.  An older install's metadata can still carry rasterio and
    # pyproj inside [geog]; the geography-stack line is the one verdict
    # about them either way, so this block never publishes a second.
    owned = sorted({f"{item.specifier} -> `{_DEEP_CHECKED[item.distribution]}`"
                    for item in requirements
                    if not item.alias and item.distribution in _DEEP_CHECKED})
    states = [(item, *_package_evidence(item, probe))
              for item in requirements
              if not item.alias and item.distribution not in _DEEP_CHECKED]
    packages = ", ".join(label for _item, _state, label, _why in states)
    if facts is not None and facts.mutually_exclusive:
        return _mutually_exclusive_extra_check(
            name, extra, install, facts, states)
    if not states:
        # An extra with nothing left for this block to judge.  `[geog]`
        # became one in 2.3.3, deliberately: rasterio and pyproj moved
        # into the runtime dependencies and the empty extra was kept so
        # the install line printed in 2.3.2's own documentation keeps
        # resolving.  Silence here would leave a reader following that
        # documentation with no way to learn where its packages went.
        if owned:
            held = (f"its package(s) are judged elsewhere: "
                    + ", ".join(owned))
        else:
            held = (f"declares no packages on this build, so {install} "
                    "installs nothing and cannot fail")
        where = (f"  The verdict is on the `{facts.deferred_to}` line"
                 if facts is not None and facts.deferred_to and not owned
                 else "")
        return Check(
            name, "info",
            f"{held}.  The extra is retained so install lines that name "
            f"it keep resolving.{where}",
            brief=_short(held), group=_GROUP_EXTRA_ALIASES)
    absent = [(label, why) for _i, state, label, why in states
              if state == "absent"]
    broken = [(label, why) for _i, state, label, why in states
              if state == "broken"]
    untested = [(label, why) for _i, state, label, why in states
                if state == "untested"]
    excluded = [(label, why) for _i, state, label, why in states
                if state == "excluded"]

    if facts is None:
        # An extra this build declares and this module has never been
        # taught.  Reported loudly rather than skipped: a silent extra is
        # the whole defect this section closes.
        unlocks = ("what it unlocks is NOT RECORDED in woof doctor -- "
                   "this build declares an extra doctor has not been "
                   "taught")
        blocking, severity = False, SEVERITY_DEGRADED
    else:
        unlocks = "needed by: " + ", ".join(facts.doors)
        blocking, severity = facts.blocking, facts.severity

    if broken:
        detail = ("; ".join(f"{label}: {why}" for label, why in broken)
                  + f".  {unlocks}")
        return Check(
            name, "missing", detail,
            f"# an installed package of [{extra}] does not import; "
            "reinstall it:\n" + install,
            action=install, brief=_short(detail), group=_GROUP_EXTRAS,
            blocking=True, severity=SEVERITY_BROKEN)

    if excluded:
        # A gap this box cannot close, so it prints no command.  MISSING
        # over an install line that installs nothing is worse than
        # silence: it sends a reader round a loop the project already
        # knows has no exit, which is why the sentence names the exit
        # (an interpreter this requirement HAS wheels for) and who owns
        # it.  Non-blocking for the same reason: `pip install recast-woof`
        # did everything it was asked, and an installer that trusts the
        # exit code must not fail a box for an upstream wheel matrix.
        detail = ("; ".join(f"{label}: {why}" for label, why in excluded)
                  + f".  {unlocks}")
        if facts is not None and facts.still_works:
            detail += f".  This box still has {facts.still_works}"
        return Check(
            name, "info", "not installable on this interpreter -- " + detail,
            brief=_short("; ".join(why for _label, why in excluded)),
            group=_GROUP_EXTRAS, blocking=False)

    if absent:
        missing = ", ".join(label for label, _why in absent)
        detail = f"{missing} not installed.  {unlocks}"
        if facts is not None and facts.still_works:
            detail += f".  Without it this box still has {facts.still_works}"
        remedy = install
        if facts is not None and facts.deferred_to:
            remedy = (f"# the verdict and the CUDA-major-specific remedy are "
                      f"on the `{facts.deferred_to}` line above\n{install}")
        if facts is not None and facts.expected_absent:
            return Check(
                name, "info",
                f"{missing} not installed, which is the normal state of an "
                f"install that is not developing woof.  {unlocks}.  "
                f"Install line: {install}",
                brief=f"{missing} not installed (expected)",
                group=_GROUP_EXTRAS)
        return Check(
            name, "missing", detail, remedy, action=install,
            brief=f"{missing} not installed", group=_GROUP_EXTRAS,
            blocking=blocking, severity=severity)

    if untested:
        detail = ("; ".join(f"{label}: {why}" for label, why in untested)
                  + f".  {unlocks}")
        return Check(name, "untested", "not tested -- " + detail,
                     brief=_short(f"metadata only: {packages}"),
                     group=_GROUP_EXTRAS)

    return Check(name, "verified",
                 f"{packages} imported in subprocesses.  {unlocks}",
                 brief=_short(packages), group=_GROUP_EXTRAS)


def _base_dependency_check(base: tuple[_Requirement, ...],
                           probe: dict[str, tuple[bool, str]]) -> Check:
    """The dependencies pip installs with no extra asked for.

    Here because ``matplotlib`` used to be reported as half of the
    ``[render]`` extra, which it has never been: it is a base
    requirement, already installed on every box, and naming it in the
    render remedy hid the package that extra actually carries.  A base
    dependency that is absent or unimportable is not an opt-in -- it is
    a broken install, and it blocks.

    Requirements a DEEPER check already owns are named here and judged
    there.  ``rasterio``/``pyproj`` became runtime dependencies in
    2.3.3 and have their own geography-stack line; probing them twice
    would put two verdicts about one fact in one report, which is how a
    reader learns to believe neither.
    """

    name = "base dependencies (installed by `pip install recast-woof`)"
    states = [(item, *_package_evidence(item, probe))
              for item in base
              if not item.alias and item.distribution not in _DEEP_CHECKED]
    elsewhere = sorted({f"{item.specifier} -> `{_DEEP_CHECKED[item.distribution]}`"
                        for item in base
                        if item.distribution in _DEEP_CHECKED})
    faults = [(label, why) for _i, state, label, why in states
              if state in ("absent", "broken")]
    if faults:
        detail = "; ".join(f"{label}: {why}" for label, why in faults)
        return Check(
            name, "missing", detail,
            "# these are not optional and not extras: they are what\n"
            "  # `pip install recast-woof` itself installs, so an absent one\n"
            "  # means a damaged install rather than a choice --\n"
            "pip install --force-reinstall --no-deps woof\n"
            "  # then, if it persists, reinstall with its dependencies:\n"
            "pip install --force-reinstall woof",
            action="pip install --force-reinstall woof",
            brief=_short(detail), blocking=True, severity=SEVERITY_BROKEN)
    incomplete = [(label, why) for _i, state, label, why in states
                  if state == "untested"]
    if incomplete:
        detail = "; ".join(f"{label}: {why}" for label, why in incomplete)
        return Check(name, "untested", detail, brief=_short(detail), blocking=False)
    packages = ", ".join(label for _i, _s, label, _w in states)
    detail = f"{packages} imported in subprocesses"
    if elsewhere:
        detail += ("; also required and reported on its own line: "
                   + ", ".join(elsewhere))
    return Check(name, "verified", detail, brief=_short(packages))


#: Packages no extra and no base requirement NAMES, which a documented
#: door nonetheless imports.  Each arrives transitively, and that is
#: exactly why it needs reporting: a transitive dependency is nobody's
#: declared contract, so nothing fails when it goes away.
#:
#: Pillow is here because `woof render --pair`'s own refusal calls it
#: "installed with the render extra", and it is not: `[render]` is
#: wrf-rust + pyshp, and Pillow rides in with matplotlib, a BASE
#: dependency.  A reader following that message installs an extra that
#: cannot supply the package they are missing.
_TRANSITIVE_CONSUMERS = (
    ("PIL", "Pillow", "woof render --pair",
     "arrives with matplotlib (a base dependency), NOT with the "
     "[render] extra"),
)


def _transitive_dependency_check() -> Check:
    """Packages a documented door imports that no requirement names."""

    name = "transitive dependencies (named by no requirement)"
    faults = []
    evidence = []
    for module, distribution, door, provenance in _TRANSITIVE_CONSUMERS:
        ok, why = _import_probe(module, distribution)
        if ok:
            evidence.append(f"{distribution} {why} ({provenance})")
        else:
            faults.append(f"{distribution}: {why} -- {door} needs it; "
                          f"it {provenance}")
    if faults:
        detail = "; ".join(faults)
        return Check(
            name, "missing", detail,
            "# these are not in any woof extra, so no extra of any\n"
            "  # name installs them.  Reinstall the base dependency that\n"
            "  # carries them, or name the package directly:\n"
            "pip install --force-reinstall matplotlib",
            action="pip install --force-reinstall matplotlib",
            brief=_short(detail), blocking=False,
            severity=SEVERITY_DEGRADED)
    return Check(name, "verified", "; ".join(evidence),
                 brief=_short(", ".join(
                     distribution
                     for _m, distribution, _d, _p in _TRANSITIVE_CONSUMERS)))


def _extras_checks() -> list[Check]:
    """Every extra this INSTALLED distribution declares, plus the base.

    Order is the metadata's own, which is pyproject's declaration order
    through the wheel: the GPU pair, render, geog, obs, dealias, dev,
    then the aliases.  A reader comparing this report against the
    install lines in the documentation is reading them in the same
    order the packaging declares them.
    """

    declared = declared_requirements()
    if declared is None:
        return [Check(
            "pip extras", "untested",
            "not tested -- this interpreter has no installed woof "
            "distribution to read extras from (a source tree on "
            "PYTHONPATH declares none).  Which extras exist, and which "
            "of them are installed, is a property of an INSTALL",
            "# install the package, even from the checkout, so its\n"
            "  # metadata exists to be read:\n"
            "pip install -e .",
            action="pip install -e .",
            brief="no installed distribution metadata")]
    base, extras = declared
    probe: dict[str, tuple[bool, str]] = {}
    checks = [_base_dependency_check(base, probe),
              _transitive_dependency_check()]
    for extra in extras:
        checks.append(_extra_check(extra, extras[extra], probe))
    return checks


def _geog_stack_check() -> Check:
    """Is the high-resolution path's PARITY FALLBACK available here?

    The check `woof doctor` did not have in 2.3.2, which is why the
    failure had to be discovered by running the feature: rasterio and
    pyproj lived in an extra nothing documented, and the only thing that
    reported their absence was a traceback after a 160.7 MiB download.
    Doctor's whole job is to answer that before anything is run.

    What the row reports moved when the warp substrate flipped onto the
    Rust ``static-fields`` library.  The DEFAULT engine is that library,
    reported by the ``static builder`` row; these two are what
    ``WOOF_STATIC_PYTHON=1`` runs on.  So the row is ``info`` rather
    than ``missing`` when they are absent: a wheel install with the
    library staged builds high-resolution terrain perfectly without
    them, and calling that MISSING would be doctor crying about a thing
    the shipped configuration does not read.  It becomes a real problem
    only when the library is also unavailable, and the ``static
    builder`` row is what says so.

    Deliberately a REAL import in a subprocess, not ``find_spec``.  Both
    libraries are thin Python over large native stacks (GDAL, PROJ), and
    the interesting failure on a box that has them installed is the one
    where the shared library will not load -- an ABI mismatch, a conda
    and pip GDAL fighting, a half-removed dist-info.  ``find_spec`` calls
    all of those green.  The front-door refusal uses the cheap probe
    because it runs on every build; doctor is where the expensive, accurate
    answer belongs.
    """
    from woof.static.geog_stack import GEOG_EXTRA, GEOG_MODULES

    results = {name: _import_probe(name) for name, _role in GEOG_MODULES}
    broken = {name: evidence for name, (ok, evidence) in results.items()
              if not ok}
    title = "geography stack (rasterio + pyproj, the highres fallback)"
    if not broken:
        versions = ", ".join(
            f"{name} {evidence}" for name, (_, evidence) in results.items())
        return Check(title, "verified",
                     f"imported in subprocesses ({versions}); "
                     "WOOF_STATIC_PYTHON=1 can run the pure-Python "
                     "high-resolution fallback",
                     brief=_short(versions))
    detail = "; ".join(f"{name}: {evidence}"
                       for name, evidence in sorted(broken.items()))
    return Check(
        title, "info",
        f"{detail} -- the pure-Python high-resolution fallback "
        "(WOOF_STATIC_PYTHON=1) cannot run here.  The DEFAULT engine is "
        "the Rust static-fields library; see the `static builder` row",
        GEOG_STACK_HINT, action=f"pip install 'recast-woof[{GEOG_EXTRA}]'",
        brief=_short(detail), blocking=False)


# ---------------------------------------------------------------------------
# Bridge executables: probe-execute, not stat()
# ---------------------------------------------------------------------------

def _exec_probe(path: Path) -> tuple[bool, str]:
    """Launch ``path`` once; can this binary execute at all?

    The bridges are austere fail-closed CLIs without ``--version``:
    given a lone probe argument each prints its usage/error diagnostic
    and exits 1 or 2.  That observable -- the process launches, emits a
    diagnostic, exits with an orderly code -- separates a runnable
    executable from an empty, truncated, or wrong-platform file, which
    refuses to launch (OSError) or dies with an abnormal status
    (Windows NTSTATUS / signal) and no diagnostic of its own.

    The header is checked BEFORE anything is launched, because on
    Windows the expensive way of asking this question can never answer:
    ``subprocess.run``'s timeout bounds the wait and not
    ``CreateProcess``, and a file with a corrupt image header can raise
    a modal loader dialog inside that call which no timeout reaches.
    Two release batteries froze there, on a file containing sixteen
    ASCII characters.  :func:`woof.bridges.launchable` answers the same
    question from the bytes; the launch that follows is bounded by the
    timeout AND by an error mode that makes the loader fail rather than
    prompt.  Nothing about a genuine bridge's verdict changed.
    """

    ok, evidence = bridges.launchable(path)
    if not ok:
        return False, f"{evidence} -- corrupt, truncated, or built for " \
                      "another platform"
    try:
        with bridges.quiet_loader_errors():
            # stdin closed: rw_globe_pack reads a JSON request from stdin, so
            # with an inherited open stdin (a terminal, the container build's
            # pipe) it waited for input until the timeout and the probe called
            # it broken (2.9.2 cut, 2026-10-17).  At EOF it prints its
            # diagnostic and exits 1, which the verdict below accepts.
            probe = subprocess.run(
                [str(path), "--version"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S,
                stdin=subprocess.DEVNULL)
    except OSError as error:
        return False, f"exists but failed to execute: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"probe invocation did not exit within "
                       f"{_PROBE_TIMEOUT_S} s")
    if probe.returncode == 0:
        return True, "probe invocation exited 0"
    diagnostic = bool((probe.stderr or probe.stdout or "").strip())
    if probe.returncode in (1, 2) and diagnostic:
        return True, (f"executes (probe exit {probe.returncode} with its "
                      "usage diagnostic)")
    return False, (f"probe invocation exited {probe.returncode} without a "
                   "usage diagnostic -- corrupt or built for another "
                   "platform")


def _bridge_checks() -> list[Check]:
    checks: list[Check] = []
    crate = bridges.crate_dir() / "Cargo.toml"
    for name, consumer in _BRIDGE_CONSUMERS.items():
        try:
            found = bridges.find_bridge(name)
        except FileNotFoundError as error:
            checks.append(Check(
                f"bridge {name}", "missing", str(error),
                f"# {bridges.BRIDGE_ENV[name]} names a missing "
                "executable: point it at a real build, or unset it "
                "and build one --\n"
                + bridges.bridge_remedy(name),
                action=f"unset {bridges.BRIDGE_ENV[name]}, or point it "
                       "at a real build",
                brief=f"{bridges.BRIDGE_ENV[name]} names a missing file",
                group=_GROUP_BRIDGES))
            continue
        if found is not None:
            ok, evidence = _exec_probe(found)
            if ok:
                # It launches.  That was the whole check, and it is not
                # enough: the wheel ships no Rust, so an upgrade of the
                # Python half leaves yesterday's binaries in place, and
                # a bridge built before a contract change still launches
                # and still prints its usage line.  1.1.0 moved the GFS
                # series file to three columns and doctor blessed every
                # 1.0.1 bridge as `ok`, after which each preparation
                # died blaming the series file woof had just written.
                ok, evidence = bridges.bridge_abi_matches(name, found)
            build = bridges.checkout_build_status(found) if ok else None
            if build is not None and not build.current:
                checks.append(Check(
                    f"bridge {name}", "missing", build.describe(),
                    "# this checkout built it, and its sources have moved\n"
                    f"# since; needed by: {consumer} --\n"
                    f"  {build.remedy()}",
                    action="rebuild this checkout's bridges",
                    brief="checkout build is older than its own sources",
                    group=_GROUP_BRIDGES))
                continue
            if ok:
                checks.append(Check(
                    f"bridge {name}", "verified", f"{found} -- {evidence}",
                    brief=_short(evidence), group=_GROUP_BRIDGES))
            else:
                checks.append(Check(
                    f"bridge {name}", "missing", f"{found} -- {evidence}",
                    f"# this one has to be replaced -- needed by: "
                    f"{consumer}\n"
                    + bridges.install_aware_build_hint(
                        bridges.CARGO_BUILD_HINT),
                    action=_build_action(), brief=_short(evidence),
                    group=_GROUP_BRIDGES))
        elif crate.is_file():
            checks.append(Check(
                f"bridge {name}", "missing",
                f"not built yet (checkout crate: {crate.parent})",
                f"# needed by: {consumer}\n{bridges.CARGO_BUILD_HINT}",
                action=bridges.CARGO_BUILD_HINT, brief="not built yet",
                group=_GROUP_BRIDGES))
        else:
            checks.append(Check(
                f"bridge {name}", "missing",
                "no checkout crate and no prebuilt executable "
                f"(searched {', '.join(str(c) for c in bridges.bridge_candidates(name))})",
                bridges.bridge_remedy(name)
                + f"\n  # needed by: {consumer}",
                action=_build_action(), brief="not staged",
                group=_GROUP_BRIDGES))
    return checks


#: Why an ``unreachable`` finding can still exit 0.  Printed on the two
#: that do, so the summary's severity census and its blocking count are
#: never in unexplained disagreement.
_REFERENCE_DOOR_NOTE = (
    "This door is not one of the README's or FIRST-LIGHT's own first-run "
    "steps, so it reports UNREACHABLE without failing the exit code")

#: The two GRIB2 tools, and the directory the mapped/20CRv3 routes
#: build them in.  ``mapped_source._build_grib2_tools`` spells this
#: path as ``Path(__file__).resolve().parents[1] / "tools" /
#: "grib1_bridge"`` from a module that lives in the same package
#: directory as this one, so the two expressions are the same path by
#: construction -- and ``tests/test_doctor_mapped_route.py`` binds them
#: by capturing the ``cwd`` that function would hand to cargo.
_GRIB2_ROUTE_TOOLS = ("grib2_inventory", "grib2_dump")
_GRIB2_ROUTE_CRATE_RELATIVE = ("tools", "grib1_bridge")


def _grib2_route_crate() -> Path:
    """The directory the mapped/20CRv3 routes run ``cargo build`` in."""

    return Path(__file__).resolve().parents[1].joinpath(
        *_GRIB2_ROUTE_CRATE_RELATIVE)


def _packaged_grib2_source_ids() -> tuple[str, ...]:
    """The packaged-profile GRIB2 sources, named for the line above.

    Derived from the adapter table and the packaged profiles rather than
    hand-listed, because the list GROWS: every new GRIB2 model landed as
    a packaged profile joins it on the day its rows land, and a
    hand-written enumeration would still be naming the first two.  Empty
    on any read problem -- this feeds a report line, never a gate.

    Since the compose port these sources are the ones that do NOT launch
    the two executables on a bare run: their preparation composes inside
    ``gpuwm_mapped_engine``.  They are still enumerated because a reader
    of this line is asking "does MY source need these tools?", and the
    answer for every one of them is worth printing rather than implying.
    """

    try:
        from woof.source_adapters import source_adapters
        from woof.source_authorities import packaged_profile

        ids = []
        for adapter in source_adapters():
            if (adapter.packaged_profile is None
                    or adapter.runner != "mapped_composition_v1"):
                continue
            try:
                declared = packaged_profile(str(adapter.packaged_profile))
            except (KeyError, FileNotFoundError, RuntimeError):
                continue
            if declared["source_format"] == "grib2":
                ids.append(adapter.source_id)
        return tuple(ids)
    except Exception:  # a broken adapter table is its own finding
        return ()


def _mapped_grib2_route_check() -> Check:
    """Can the mapped/20CRv3 GRIB2 routes reach the tools they need?

    THIS IS THE ``ok`` THIS MODULE WAS CAUGHT PRINTING, and it is now the
    line that reports the fix.  Through 2.3.3 ``woof fetch-bridges``
    staged ``grib2_inventory`` and ``grib2_dump``, ``_bridge_checks``
    probe-executed both and printed ``ok bridge grib2_inventory``, and
    then::

        $ woof-mapped-inspect --mapping <the wheel's own authority> \\
              --input FILE.grib2
        NotADirectoryError: [WinError 267] The directory name is invalid

    Both of doctor's statements about the FILES were true.  Neither was a
    statement about the ROUTE, and the route is what the reader was
    asking about: ``mapped_source._build_grib2_tools()`` never consulted
    the staged copies, it shelled ``cargo build`` in ``<package
    parent>/tools/grib1_bridge``, a directory no wheel contains.

    That function consults :func:`woof.bridges.find_bridge` FIRST now,
    the way ``ingest/grib.py`` always did, so the staged copies ARE what
    the default route uses.  And it is the resolver of RECORD: the
    ``rw-wps``/``woof prep`` front door resolves the same two tools
    through the same function for every GRIB2 route it dispatches --
    mapped, 20CRv3, and each packaged-profile source -- so a bare
    ``woof prep --source hrrr-prs`` needs no tool flags on a box this
    line calls verified.  The check applies the resolver's own static
    contract gate too, because "resolves" that hands the route a stale
    decoder is the same green-over-a-hole one release later.  This
    check still answers the route's question rather than the file's --
    it just has a third answer it did not have before:

    * both tools resolve through the ladder, and then the default route
      reaches them.  ``verified``.  This is the state a wheel install
      lands in after ``woof fetch-bridges``, and reporting it as a gap
      would be the same defect in the opposite direction.
    * neither resolves, and a crate is present (a source checkout), and
      then doctor CANNOT answer: judging the build would mean running
      cargo, which this module never does.  ``untested``, and it says so.
    * neither resolves and there is no crate to build in.  ``missing``,
      and the remedy is ``woof fetch-bridges`` with nothing appended to
      it, because staging is now sufficient on its own.

    It does not block.  See :func:`blocking_gaps`: the doors it closes
    (``woof-mapped-inspect``, ``woof adapt --descriptor``, the 20CRv3
    direct route) live in the CLI reference and the examples, not in the
    README's or FIRST-LIGHT's own first-run path.
    """

    name = "mapped/20CRv3 GRIB2 route (the tools it resolves)"
    # WHICH doors, exactly.  Before the compose port this line could say
    # "every packaged-profile GRIB2 source", because every mapped prep
    # ran the Python engine and launched these two executables.  It no
    # longer does: `compose` is declared for grib2, so a BARE prep of a
    # packaged-profile source resolves the engine and no subprocess tool
    # at all (`woof prep --source hrrr-prs ... --dry-run` prints neither
    # flag).  Saying otherwise here would tell a reader their bytes are
    # decoded somewhere they are not, which is the same defect
    # `ENGINE_GAPS` exists to prevent, in a report line instead of a
    # table.  The doors below are the ones measured to still resolve
    # them: `woof adapt` (adapt.py), `woof-member-prep`'s identity
    # verification (member_prep.py), and any mapped run that asks for
    # the Python engine or names a tool.  The `--source 20crv3` runner
    # left this list when it moved onto `decode_composed_source`: a bare
    # member prep composes in the engine and reads its raw
    # record-inventory surface in process, so it resolves neither tool.
    packaged = _packaged_grib2_source_ids()
    doors = ("woof adapt --descriptor/--input, woof-member-prep, and "
             "any mapped run that asks for the Python engine with "
             "--mapped-engine python or names a tool"
             + (f" (the packaged GRIB2 profiles -- {', '.join(packaged)} -- "
                "20crv3's member route included, compose in the engine on "
                "a bare run and resolve neither)"
                if packaged else ""))
    staged: dict[str, Path | None] = {}
    override_faults: list[str] = []
    stale_notes: list[str] = []
    for tool in _GRIB2_ROUTE_TOOLS:
        try:
            resolved = bridges.find_bridge(tool)
        except FileNotFoundError as error:
            # A set environment override naming a missing file.  The
            # route raises on it too, by design, so it is a gap here
            # whatever else resolves.
            staged[tool] = None
            override_faults.append(str(error))
            continue
        if resolved is not None:
            # The same static contract gate the resolver of record
            # (mapped_source._build_grib2_tools) applies: a stale hit
            # is a hit the route will not accept, so reporting it
            # resolved would be the green-over-a-hole this check exists
            # to stop -- in its newest costume.
            ok, evidence = bridges.bridge_abi_matches(tool, resolved)
            if not ok:
                stale_notes.append(f"{tool} at {resolved} {evidence}")
                resolved = None
        staged[tool] = resolved
    crate = _grib2_route_crate()
    have_crate = (crate / "Cargo.toml").is_file()
    found = {tool: path for tool, path in staged.items() if path is not None}

    if (len(found) == len(_GRIB2_ROUTE_TOOLS) and not override_faults
            and not stale_notes):
        return Check(
            name, "verified",
            f"{', '.join(str(path) for path in found.values())} resolve "
            f"through the same ladder {doors} use, so the default route "
            "reaches them with no flags",
            brief="resolves both tools", group=_GROUP_BRIDGES)

    if have_crate:
        absent = [tool for tool in _GRIB2_ROUTE_TOOLS if tool not in found]
        note = (f"{', '.join(absent)} do not resolve through the ladder, and "
                f"the route falls back to a cargo build in {crate}.  Doctor "
                "never runs cargo (see this module's docstring), so whether "
                "that build succeeds here is unknown")
        if stale_notes:
            note += "; " + "; ".join(stale_notes)
        if override_faults:
            note += "; " + "; ".join(override_faults)
        return Check(name, "untested", f"not tested -- {note}",
                     brief="not tested; the route falls back to cargo",
                     group=_GROUP_BRIDGES)

    absent = [tool for tool in _GRIB2_ROUTE_TOOLS if tool not in found]
    detail = (
        f"{', '.join(absent)} not resolvable, and this install has no "
        f"tools/grib1_bridge crate to build them in ({crate}), so "
        f"{doors} cannot resolve the tools")
    if stale_notes:
        detail += "; " + "; ".join(stale_notes)
    if override_faults:
        detail += "; " + "; ".join(override_faults)
    remedy_lines = [
        "woof fetch-bridges",
        "  # stages both tools where the mapped route already looks;",
        "  # --grib2-inventory / --grib2-dump override the resolved paths",
        "  # and are not needed for the default route."]
    if override_faults:
        remedy_lines.insert(0, "# unset the environment override(s) below, "
                               "or point them at a real build:")
    return Check(name, "missing", f"{detail}.  {_REFERENCE_DOOR_NOTE}",
                 "\n".join(remedy_lines),
                 action="woof fetch-bridges",
                 brief=f"{', '.join(absent)} not resolvable by this route",
                 group=_GROUP_BRIDGES,
                 blocking=False, severity=SEVERITY_UNREACHABLE)


#: One next command for every obs front door the bundle cannot supply:
#: they share a clone and a cargo build, so a copy per door reads as
#: several problems.  Reached only when an artifact is absent from the
#: bundle; every one of them is IN it as of this release, so the live
#: remedy is `woof fetch-bridges`.  Kept because the branch that needs
#: it is the branch a future front door lands on first.
_OBS_FRONT_DOOR_ACTION = ("build the obs front doors from a clone "
                          "of the renderer workspace")


#: Every bundled artifact, and the check that reports it.  A SET, kept
#: by hand and guarded by a test, because the alternative is the defect
#: it exists to stop: each of these checks is written per artifact, so
#: an artifact added to the bundle is invisible here until someone
#: remembers to write its check -- which is exactly how ``rw_nexrad``
#: and then the three observation front doors came to be absent from a
#: report that called the estate green.
#:
#: :func:`_bundle_coverage_checks` turns the set into a live sweep, so a
#: new artifact is REPORTED (as untested, never as ok) from the moment
#: the bundle carries it, rather than waiting for this map to catch up.
#: ``tests/test_doctor_route_accuracy.py`` fails when they disagree.
_CHECKED_ARTIFACTS = {
    "nextgen_hydrofabric": "the `NextGen nextgen_hydrofabric` line",
    "arwen_nextgen_forcing": "the `NextGen arwen_nextgen_forcing` line",
    "rw_nextgen_verify": "the `NextGen rw_nextgen_verify` line",
    "rw_nextgen_output": "the `NextGen rw_nextgen_output` line",
    "rw_hydro_world": "the `hydro door rw_hydro_world` line",
    "rw_hydro_params": "the `hydro door rw_hydro_params` line",
    "rw_hydrofabric": "the `hydro door rw_hydrofabric` line",
    "nextgen-reservoir-prepare": "the `hydro door nextgen-reservoir-prepare` line",
    "rw_hydro_init": "the `hydro door rw_hydro_init` line",
    "rw_hail": "the `obs front door` lines",
    "rw_fire_indices": "the `fire danger indices` line",
    "rw_ice_verify": "the `ice observation binding` line",
    "rw_ocean": "the `ocean preparation` line",
    "rw_bathy": "the `surge bathymetry` line",
    "rw_ocean_verify": "the `ocean SST verification` line",
    "rw-wind-table": "the `wind turbine tables` line",
    "grib1_bridge": "the `bridge ...` lines",
    "gfs_grib2_bridge": "the `bridge ...` lines",
    "hrrr_grib2_bridge": "the `bridge ...` lines",
    "grib2_inventory": "the `bridge ...` and mapped/20CRv3 route lines",
    "grib2_dump": "the `bridge ...` and mapped/20CRv3 route lines",
    "gdt101_remap": "the `bridge ...` lines",
    "met_intermediate": "the `bridge ...` lines",
    "gpuwm_preprocess_cpu": "the `cpu preprocess library` line",
    "rw_fetch": "the `fetch backbone` line",
    "rw_wrfbatch": "the `renderer` line",
    "rw_simradar": "the `simulated radar` line",
    "rw_soundings": "the `sounding window renderer` line",
    "rw_tc_fields": "the `cyclone field reader` line",
    "rw_mpas_mesh": "the `mesh generator` line",
    "rw_mpas_static": "the `mesh static builder` line",
    "rw_nexrad": "the `radar front door` line",
    "region_global_dealias": "the `region-global dealiasing engine` line",
    "rw_odim": "the `obs front door` lines",
    "rw_radar_feeds": "the `obs front door` lines",
    "rw_mrms": "the `obs front door` lines",
    "rw_stage4": "the `obs front door` lines",
    "rw_asos": "the `obs front door` lines",
    "rw_airnow": "the `obs front door` lines",
    "rw_goes": "the `obs front door` lines",
    "rw_opera": "the `obs front door` lines",
    "rw_igra2": "the `obs stream door` lines",
    "rw_amv": "the `obs stream door` lines",
    "rw_ndbc": "the `obs stream door` lines",
    "rw_gnssro": "the `obs stream door` lines",
    "rw_wis2": "the `obs stream door` lines",
    "rw_prepbufr": "the `obs stream door` lines",
    "rw_madis": "the `obs stream door` lines",
    "rw_netcdf": "the `NetCDF decoder` line",
    "rw_zarr": "the `era5 route fetch transport` line",
    "netcdf_writer": "the `NetCDF writer` line",
    "gpuwm_mapped_engine": "the `mapped decode engine` line",
    "static_fields": "the `static builder` line",
    "obs_regrid": "the `observation remap` line",
    "obs_score": "the `observation scoring` line",
    "rw_isobaric": "the `isobaric height reader` line",
    "rw_superob": "the `radar superob` line",
    "rw_mpas_mesh": "the `MPAS binary` lines",
    "rw_mpas_init": "the `MPAS binary` lines",
    "rw_mpas_geometry": "the `MPAS binary` lines",
    "rw_mpas_hostprep": "the `MPAS binary` lines",
    "rw_mpas_convert": "the `MPAS binary` lines",
    "rw_mpas_lbc": "the `MPAS binary` lines",
    "rw_hex_points": "the `MPAS binary` lines",
    "rw_mpas_init_v831": "the `MPAS binary` lines",
    "rw_mpas_gsl_oro": "the `MPAS binary` lines",
    "rw_tempo": "the `TEMPO initialization library` line",
    "rw_projected_intermediate": "the `projected MET intermediate` line",
    "rw_mlexport": "the `ML dataset exporter` line",
    "rw_icechunk": "the `Icechunk native history` line",
    "rw_grib2export": "the `GRIB2 exporter` line",
    "rw_verify": "the `observation verification engine` line",
    "rw_compare": "the `comparison engine` line",
    "rw_ensbatch": "the `ensemble reducer` line",
    "rw_nowcast_frames": "the `nowcast frames converter` line",
    "rw_globe_pack": "the `global globe packer` line",
    "rw_probgrid": "the `probability map renderer` line",
    "rw_scorecard": "the `observation scorecard` line",
    "rw_scoreboard": "the `forecast verification scorer` line",
}


def _bundle_coverage_checks() -> list[Check]:
    """Is every artifact the bundle carries reported by some check?

    The one check in this module that is ABOUT the report.  Each of the
    others is written per artifact, so the estate's completeness has
    always depended on somebody remembering -- and twice it did not:
    ``rw_nexrad`` shipped outside both audited sets and doctor passed on
    boxes where every radar route was dead, and the three observation
    front doors did the same thing again.  A bundle that grows a
    fourteenth artifact must not need a third incident.

    An uncovered artifact is resolved through the same ladder its
    consumer would use and then reported ``untested`` when it is there
    -- present, unprobed, and saying so -- because doctor knows no
    contract for a binary nobody has taught it about.  Never ``ok``.
    """

    try:
        from woof import bridge_assets

        bundled = tuple(bridge_assets.BUNDLED_ARTIFACTS)
    except Exception as error:                   # noqa: BLE001 - reported
        return [Check(
            "bundled artifact coverage", "untested",
            f"not tested -- the bundle manifest could not be read "
            f"({type(error).__name__}: {error}), so doctor cannot say "
            "whether every artifact it carries is reported above",
            "# reinstall so the packaged bundle manifest loads:\n"
            + REINSTALL_HINT,
            action="reinstall woof", brief="bundle manifest unreadable")]

    uncovered = [artifact for artifact in bundled
                 if artifact.name not in _CHECKED_ARTIFACTS]
    if not uncovered:
        owners = sorted({_CHECKED_ARTIFACTS[artifact.name]
                         for artifact in bundled})
        return [Check(
            "bundled artifact coverage", "verified",
            f"all {len(bundled)} artifact(s) this release's bundle carries "
            f"are reported by a check above ({'; '.join(owners)})",
            brief=f"{len(bundled)} of {len(bundled)} bundled artifacts "
                  "reported")]

    checks: list[Check] = []
    for artifact in uncovered:
        name = f"bundled artifact {artifact.name} (no check of its own)"
        try:
            filename = bridge_assets.artifact_filename(
                artifact, bridge_assets.host_platform() or "linux-x86_64")
            found = bridges.find_artifact(artifact.env_var, filename)
        except Exception as error:               # noqa: BLE001 - reported
            checks.append(Check(
                name, "untested",
                f"not tested -- it is in this release's bundle and doctor "
                f"has no check for it; resolving it also failed "
                f"({type(error).__name__}: {error})",
                brief="in the bundle, unresolvable, unchecked"))
            continue
        if found is None:
            checks.append(Check(
                name, "missing",
                f"in this release's bundle and not staged here; needed by: "
                f"{artifact.consumer}.  Doctor has no contract probe for "
                "this artifact, so staging it is all this line can ask for"
                ,
                "woof fetch-bridges\n"
                "  # stages every pinned artifact, this one included",
                action="woof fetch-bridges",
                brief="bundled, not staged, and unchecked",
                group=_GROUP_BRIDGES, blocking=False,
                severity=SEVERITY_DEGRADED))
            continue
        checks.append(Check(
            name, "untested",
            f"not tested -- {found} is staged, and doctor has no contract "
            f"probe for this artifact (needed by: {artifact.consumer}).  "
            "Its presence was checked; its contract was not",
            brief="staged; contract not probed", group=_GROUP_BRIDGES))
    return checks


#: The staged-estate line's name, referenced by tests and by the
#: coverage bookkeeping below.  One string, because a report line whose
#: name is spelled twice is a line two readers can disagree about.
_STAGED_ESTATE_NAME = ("staged bridge estate (the bytes the routes "
                       "resolve vs this release's pins)")


def _hex_helpers_check() -> Check | None:
    """Where the hex model's compiled helpers come from on this install.

    A PyPI platform wheel carries their pins, not the helpers (WOOF 1.1.0
    r16, PyPI's per-file limit): the first hex door that needs one fetches
    the release's helper archive and verifies every file
    (:mod:`woof.hex.helpers`).  Not staged yet is the expected state of a
    fresh install, so it is reported and never blocks.
    """

    try:
        from woof.hex import helpers
        state = helpers.status()
    except Exception:                                # noqa: BLE001
        return None
    if state["state"] == "absent":
        return None
    if state["state"] == "packaged":
        return Check("hex helpers", "verified",
                     f"{state['files']} programs carried in {state['dir']}",
                     brief=f"{state['files']} carried")
    if state["state"] == "staged":
        return Check("hex helpers", "verified",
                     f"{state['files']} programs staged at {state['dir']}, verified against this "
                     f"release's pins ({state['bundle']})",
                     brief=f"{state['files']} staged")
    return Check("hex helpers", "info",
                 f"on demand: {state['files']} programs in {state['bundle']} "
                 f"({state['bundle_bytes'] / 1e6:.1f} MB) are fetched from {state['url']} and verified by "
                 f"the first hex command that needs one; `woof fetch-bridges` stages them now into "
                 f"{state['dir']}",
                 action="woof fetch-bridges",
                 brief="on demand, not staged yet")


def _staged_estate_check() -> Check:
    """Are the artifacts this box RESOLVES the bytes this release pinned?

    Every other artifact line in this report asks a question of one
    binary: is it there, does it launch, does it declare this contract.
    test host A answered yes to all three on an estate staged in August by an
    earlier release -- the files were present, they launched, and
    :data:`woof.bridges.BRIDGE_ABI_MARKERS` only moves when a contract
    version bumps, which it had not -- so ``woof doctor`` printed no
    gaps and exited 0, and the first ``woof prep`` refused with rc 78.

    The question this check asks is the one ``woof fetch-bridges``
    asks and doctor did not: are these the exact bytes -- size and
    SHA-256 -- that the release this wheel came from published?  It is
    asked of the file each consumer's own resolution ladder would pick
    (:func:`woof.bridges.find_artifact`), not of a directory listing,
    because what a route runs is what the ladder returns.

    Three verdicts, and the difference between them is who owns the
    bytes:

    * A mismatch on a file under :func:`woof.bridges.default_bridge_dir`
      is a gap.  That directory is exactly what ``woof fetch-bridges``
      writes, so one command closes it.
    * A mismatch on a file resolved anywhere else -- an environment
      override, a checkout's ``target/release`` -- is reported and does
      NOT block.  A cargo build never matches a release pin, and
      offering to re-fetch a directory that binary does not live in
      would name a breakage that is not there.
    * Absent is somebody else's line.  Every bundled artifact already
      has a check that names it when it is not staged, and printing the
      same gap twice gives a reader two counts of one estate.

    The doors ask the same question now, at the moment they pick a file
    (:func:`woof.bridges.require_release_pin`, over the same pins and
    the same size-and-SHA-256 comparison), which is what this line used
    to say alone: test host A's report named the skew and every route went on
    running the August bytes anyway.  This check keeps its own job --
    saying what the estate IS, all of it, before anything opens a door
    -- and stays a reader: it resolves inside
    :func:`woof.bridges.inspection_only` so measuring the box never
    repairs it.
    """

    try:
        from woof import bridge_assets
    except Exception as error:                   # noqa: BLE001 - reported
        return Check(
            _STAGED_ESTATE_NAME, "untested",
            f"not tested -- the packaged pins could not be loaded "
            f"({type(error).__name__}: {error}), so doctor cannot say "
            "whether the staged artifacts are this release's bytes",
            "# reinstall so the packaged pins load:\n" + REINSTALL_HINT,
            action="reinstall woof", brief="pins module unreadable",
            group=_GROUP_BRIDGES)

    platform = bridge_assets.host_platform()
    try:
        pins = bridge_assets.load_pins()
    except Exception as error:                   # noqa: BLE001 - reported
        return Check(
            _STAGED_ESTATE_NAME, "untested",
            f"not tested -- the packaged pins document is unusable "
            f"({type(error).__name__}: {error})",
            "# reinstall so the packaged pins load:\n" + REINSTALL_HINT,
            action="reinstall woof", brief="pins document unusable",
            group=_GROUP_BRIDGES)
    bundle = pins.bundle_for(platform)
    if bundle is None:
        # A source checkout, or a platform no bundle is published for.
        # Both are normal and neither is a gap; saying which one it is
        # keeps the reader from reading silence as a pass.
        reason = (f"this build's packaged pins declare no release "
                  f"(a checkout stamps them at cut time)"
                  if pins.release is None else
                  f"this release's pins carry no bundle for "
                  f"{bridge_assets.host_platform_description()}")
        return Check(
            _STAGED_ESTATE_NAME, "info",
            f"not compared -- {reason}, so there is nothing to check the "
            "staged artifacts against.  A released wheel carries pins and "
            "this line compares every artifact's bytes against them.  "
            + bridges.staging_location_note(),
            brief="no pins to compare against", group=_GROUP_BRIDGES,
            blocking=False)

    by_name = {artifact.name: artifact
               for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    staged_dir = bridges.default_bridge_dir()
    current: list[str] = []
    stale: list[str] = []
    foreign: list[str] = []
    absent: list[str] = []
    for pin in bundle.binaries:
        artifact = by_name.get(pin.artifact)
        if artifact is None:
            # A pins document from a release this code does not know.
            # Fall back to the staged directory, which is the rung the
            # pins are about.
            resolved = staged_dir / pin.filename
            resolved = resolved if resolved.is_file() else None
        else:
            try:
                # Resolve as a READER.  Outside this scope the same call
                # refreshes, or refuses on, exactly the artifacts this
                # line exists to enumerate, and a check that repairs what
                # it is measuring can only ever report a clean estate.
                # `collect_checks` already holds this scope; it is
                # repeated here because tests and embedders reach this
                # function directly.
                with bridges.inspection_only():
                    resolved = bridges.find_artifact(artifact.env_var,
                                                     pin.filename)
            except FileNotFoundError:
                # The override names a missing file.  Its own check
                # already reports that by name.
                absent.append(pin.filename)
                continue
        if resolved is None:
            absent.append(pin.filename)
        elif bridge_assets.matches_pin(resolved, pin):
            current.append(pin.filename)
        elif (_under(resolved, staged_dir)
              or _under(resolved, bridges.legacy_bridge_dir())):
            where = ("" if _under(resolved, staged_dir)
                     else f" at {resolved}, left in place")
            stale.append(f"{pin.filename} ({resolved.stat().st_size:,} B, "
                         f"pinned {pin.bytes:,} B{where})")
        else:
            foreign.append(f"{pin.filename} at {resolved}")

    assets_note = ""
    try:
        held, stale_assets, absent_assets = bridge_assets.classify_assets(
            staged_dir, bundle)
    except Exception:                            # noqa: BLE001 - optional
        held, stale_assets, absent_assets = [], [], []
    if (stale_assets or absent_assets) and _map_assets_resolve_elsewhere(
            staged_dir):
        # The staged copy is what fetch-bridges writes, and the renderer
        # reads it only when nothing earlier resolves.  A wheel install
        # draws from the recast-woof-data companion, so an unstaged copy here
        # costs no picture a coastline; demanding it anyway was the
        # MISSING line every bare pip install printed.
        stale_assets, absent_assets = [], []
    if bundle.assets and (stale_assets or absent_assets):
        assets_note = (
            f"; {len(stale_assets) + len(absent_assets)} of "
            f"{len(bundle.assets)} pinned map asset file(s) under "
            f"{staged_dir / bridge_assets.ASSET_ROOT} are missing or stale, "
            "so the renderer draws plots without coastlines or borders")

    census = (f"{len(current)} of {len(bundle.binaries)} artifact(s) match "
              f"the bytes {pins.release} published; "
              + bridges.staging_location_note())
    if stale:
        return Check(
            _STAGED_ESTATE_NAME, "missing",
            f"{len(stale)} artifact(s) staged at {staged_dir} are NOT this "
            f"release's bytes: {', '.join(stale)}.  They exist and they "
            "launch, so every other line above passes them.  The doors no "
            "longer do: a resolution that lands on one of these re-fetches "
            "this release's bundle before the run continues and refuses by "
            "name when it cannot, so what this line predicts is a run that "
            "pauses to refresh -- or stops -- never one that silently "
            f"diverges ({census}){assets_note}",
            "woof fetch-bridges\n"
            "  # replaces every artifact whose bytes do not match the\n"
            "  # packaged pins, then re-verifies size and SHA-256.\n"
            "  # Doing it here is doing it deliberately, once, rather\n"
            "  # than inside the next door that resolves one of them",
            action="woof fetch-bridges",
            brief=f"{len(stale)} staged artifact(s) are not "
                  f"{pins.release}'s bytes",
            group=_GROUP_BRIDGES, severity=SEVERITY_BROKEN)
    if stale_assets or absent_assets:
        return Check(
            _STAGED_ESTATE_NAME, "missing",
            f"every staged artifact matches this release's pins ({census})"
            + assets_note,
            "woof fetch-bridges\n"
            "  # stages the pinned map assets beside the binaries",
            action="woof fetch-bridges",
            brief=f"{len(stale_assets) + len(absent_assets)} map asset "
                  "file(s) missing or stale",
            group=_GROUP_BRIDGES, blocking=False,
            severity=SEVERITY_DEGRADED)
    if foreign:
        return Check(
            _STAGED_ESTATE_NAME, "info",
            f"{len(foreign)} artifact(s) resolve from outside "
            f"{staged_dir} and are not the bytes {pins.release} published, "
            "which is what a local build or an environment override looks "
            f"like: {'; '.join(foreign)}.  `woof fetch-bridges` would not "
            f"change what these routes run ({census})",
            brief=f"{len(foreign)} artifact(s) built or overridden locally",
            group=_GROUP_BRIDGES, blocking=False)
    if not current:
        return Check(
            _STAGED_ESTATE_NAME, "info",
            f"no pinned artifact is staged at {staged_dir}; each one's own "
            "line above names what it costs and how to get it",
            brief="nothing staged to compare", group=_GROUP_BRIDGES,
            blocking=False)
    return Check(
        _STAGED_ESTATE_NAME, "verified",
        f"{census}"
        + (f"; {len(absent)} not staged (reported by their own lines above)"
           if absent else "")
        + (f"; {len(held)} map asset file(s) verified" if held else ""),
        brief=f"{len(current)} of {len(bundle.binaries)} match the pins",
        group=_GROUP_BRIDGES)


def _map_assets_resolve_elsewhere(staged_dir: Path) -> bool:
    """Does the renderer draw its maps from somewhere other than
    ``staged_dir``?  Never raises; ``False`` when it cannot tell."""

    try:
        from woof import rustwx

        with bridges.inspection_only():
            renderer = rustwx.find_renderer()
        resolved = rustwx.resolve_basemap_dir(renderer)
    except Exception:                            # noqa: BLE001 - optional
        return False
    return resolved is not None and not _under(resolved, staged_dir)


def _under(path: Path, directory: Path) -> bool:
    """Is ``path`` inside ``directory``?  Never raises."""

    try:
        return path.resolve().is_relative_to(directory.resolve())
    except (OSError, ValueError):
        return False


def _obs_front_door_checks() -> list[Check]:
    """Every observation front door, reported by name.

    ``woof/obs/frontdoor.py`` resolves each of these through the same
    ladder as ``rw_nexrad`` and refuses by name when one is absent.
    Through 2.3.3 none of them was in ``BUNDLED_ARTIFACTS``, so ``woof
    fetch-bridges`` printed "all artifacts already staged and pin-valid"
    and the refusal repeated verbatim -- and doctor, which audited
    exactly the bundled set, reported a fully green estate on a box
    where every obs front door was absent.

    Both halves of that are fixed now and this function is where they
    meet.  The binaries are REPORTED, so the estate is no longer green
    over them; and the remedy is a FUNCTION of the bundle manifest
    rather than an assumption, so it reads ``woof fetch-bridges``
    exactly when that command can supply the artifact.  As of this
    release every door below is bundled, so it always can -- but the
    branch that says otherwise is kept, because the next front door to
    land will land on it before it lands in a bundle.

    Non-blocking: these doors are named in no README and no FIRST-LIGHT
    step.
    """

    try:
        from woof.obs import frontdoor
    except ImportError as error:                 # pragma: no cover - partial
        return [Check(
            "observation front doors (MRMS / Stage-IV / ASOS)", "missing",
            f"woof.obs is not importable ({error}) -- blocks every "
            "observation front door and means this install is incomplete",
            "# reinstall so the observation stack imports:\n"
            + REINSTALL_HINT,
            action="reinstall woof", brief="obs stack not importable",
            group=_GROUP_ENGINES, severity=SEVERITY_BROKEN)]
    try:
        from woof.bridge_assets import BUNDLED_ARTIFACTS

        bundled = {artifact.name for artifact in BUNDLED_ARTIFACTS}
    except Exception:                            # noqa: BLE001 - reported
        bundled = set()

    # The door NAMED per instrument is the product subcommand, because
    # that is the one a reader can type after `pip install recast-woof`.  The
    # campaign drivers under tools/ are the same binaries reached the
    # other way and are named beside the three that have one; `woof
    # obs <instrument>` is what every one of them has.
    doors = {"mrms": "woof obs mrms (and python -m tools.obs_fetch_mrms)",
             "stage4": "woof obs stage4 (and python -m "
                       "tools.obs_fetch_stage4)",
             "asos": "woof obs asos (and python -m tools.obs_fetch_asos)",
             "goes": "woof obs goes",
             "airnow": "woof obs airnow",
             "opera": "woof obs opera",
             "odim": "woof obs odim, and every `woof obs radar` "
                     "subcommand",
             "radar-feeds": "woof obs radar-feeds",
             "hail": "woof obs hail"}
    # Read from the resolver rather than enumerated here, so a door added
    # to FRONT_DOORS and not to `doors` above is a loud KeyError in the
    # test suite instead of a silently unreported binary -- which is the
    # exact failure this whole function exists to have stopped.
    unnamed = sorted(set(frontdoor.FRONT_DOORS) - set(doors))
    assert not unnamed, (
        f"woof.obs.frontdoor gained {unnamed} and woof.doctor does not "
        "name the door(s) they unlock")
    checks: list[Check] = []
    for instrument, door in sorted(doors.items()):
        front = frontdoor.FRONT_DOORS[instrument]
        name = f"obs front door {front.name} ({front.subject})"
        in_bundle = front.name in bundled
        try:
            found = front.find()
        except FileNotFoundError as error:
            checks.append(Check(
                name, "missing", f"{error} -- {door} cannot run",
                f"# {front.env_var} names a missing executable: point it "
                "at a real build, or unset it --\n"
                + bridges.install_aware_build_hint(
                    frontdoor.CARGO_BUILD_HINT, bridges.RUSTWX_CRATE_RELATIVE),
                action=f"unset {front.env_var}, or point it at a real build",
                brief=f"{front.env_var} names a missing file",
                group=_GROUP_ENGINES, blocking=False,
                severity=SEVERITY_UNREACHABLE))
            continue
        if found is None:
            # The remedy is a FUNCTION of whether the bundle can supply
            # this artifact, read from the bundle manifest rather than
            # assumed -- and assuming is the defect being fixed.  The
            # resolver's own refusal opens with `woof fetch-bridges`
            # because *a* bundle exists for the platform, without ever
            # asking whether THIS artifact is in it, so a reader ran it,
            # was told all pinned artifacts were staged, and met the
            # identical refusal again.  When a release does start
            # bundling these, this line becomes that one command with no
            # edit here.
            if in_bundle and bridges.on_demand_fetch_offered(front.name):
                # An on-demand artifact (the platform wheel leaves it to the
                # bundle; its first use stages it): the same row the stream
                # doors get.  Breakage (2.9.2 cut, Windows wheel smoke,
                # 2026-10-17): rw_radar_feeds, on demand since 2.9.2, was
                # reported "not staged; woof obs radar-feeds cannot run"
                # on a fresh install, which is false: its first use runs.
                checks.append(Check(
                    name, "missing",
                    "not staged yet -- it is fetched on demand: the platform "
                    f"wheel does not carry it, and the first `{door}` that "
                    "needs it stages it from this release's bundle",
                    f"woof fetch-bridges --partial --artifact {front.name}",
                    action=f"woof fetch-bridges --partial --artifact {front.name}",
                    brief="on demand; staged by its first fetch",
                    group=_GROUP_ENGINES, blocking=False,
                    severity=SEVERITY_OPT_IN))
                continue
            if in_bundle:
                bundle_note = ("it IS in this release's prebuilt bundle, so "
                               "one command stages it")
                remedy = ("woof fetch-bridges\n"
                          "  # stages every pinned artifact for this "
                          "platform, this one included")
                action = "woof fetch-bridges"
            else:
                bundle_note = (
                    "and `woof fetch-bridges` will NOT supply it: this "
                    "artifact is not in the bundle at all, so that command "
                    "reports every pinned artifact staged and this door "
                    "stays shut")
                remedy = (
                    f"# {front.name} is not a bundled artifact.  The only\n"
                    "  # route to it is a clone and a build of the renderer\n"
                    "  # workspace:\n"
                    + bridges.install_aware_build_hint(
                        frontdoor.CARGO_BUILD_HINT,
                        bridges.RUSTWX_CRATE_RELATIVE)
                    + f"\n  # then copy "
                    f"{bridges.executable_name(front.name)} "
                    f"into {bridges.default_bridge_dir()},\n"
                    f"  # or set {front.env_var} to its full path")
                # One action for all three, so the terse report folds
                # them into a single line: they share one clone, one
                # cargo build, and one reason.
                action = _OBS_FRONT_DOOR_ACTION
            checks.append(Check(
                name, "missing",
                f"not built and not staged -- {door} cannot run, "
                f"{bundle_note}.  {_REFERENCE_DOOR_NOTE}",
                remedy, action=action,
                brief=f"not staged; {door} cannot run",
                group=_GROUP_ENGINES, blocking=False,
                severity=SEVERITY_UNREACHABLE))
            continue
        ok, evidence = front.probe(found)
        if not ok:
            checks.append(Check(
                name, "missing", f"{found} -- {evidence}; {door} cannot run",
                "# REBUILD it -- this is a record-contract change, so\n"
                "  # another copy of the same vintage fails identically --\n"
                + bridges.install_aware_build_hint(
                    frontdoor.CARGO_BUILD_HINT, bridges.RUSTWX_CRATE_RELATIVE),
                action=f"rebuild {front.name}", brief=_short(evidence),
                group=_GROUP_ENGINES, blocking=False,
                severity=SEVERITY_UNREACHABLE))
            continue
        checks.append(Check(name, "verified", f"{found} -- {evidence}",
                            brief=_short(evidence), group=_GROUP_ENGINES))
    return checks


def _obs_stream_door_checks() -> list[Check]:
    """The observation-stream doors (``woof obs streams``) and prepbufr.

    They are on-demand artifacts: pinned in this release's bundle, not in
    the platform wheel, staged by the first fetch that needs one.  So an
    absent one on a pip install whose pins hold it is the expected state and
    reads ``opt-in``; anywhere a download cannot supply it (a checkout, no
    bundle for this platform) it reads ``unreachable`` with the build route,
    the severity an absent obs front door has.  Resolution only: doctor
    never downloads.
    """

    try:
        from woof.obs import frontdoor
    except ImportError:                          # pragma: no cover - partial
        return []
    doors = [*frontdoor.STREAM_DOORS.values(), frontdoor.PREPBUFR]
    checks: list[Check] = []
    for front in doors:
        name = f"obs stream door {front.name} ({front.subject})"
        try:
            found = front.find()
        except FileNotFoundError as error:
            checks.append(Check(
                name, "missing", f"{error} -- its observation streams cannot be fetched",
                f"# unset {front.env_var} or point it at a real build:\n" + front.remedy(),
                action=f"unset {front.env_var}, or point it at a real build",
                brief=f"{front.env_var} names a missing file",
                group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_UNREACHABLE))
            continue
        if found is None:
            if bridges.on_demand_fetch_offered(front.name):
                checks.append(Check(
                    name, "missing",
                    "not staged yet -- it is fetched on demand: the platform "
                    "wheel does not carry it, and the first stream fetch that "
                    "needs it stages it from this release's bundle",
                    front.remedy(),
                    action=f"woof fetch-bridges --partial --artifact {front.name}",
                    brief="on demand; staged by its first fetch",
                    group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_OPT_IN))
            else:
                checks.append(Check(
                    name, "missing",
                    "not built and not staged -- its observation streams cannot be fetched",
                    front.remedy(), action=_OBS_FRONT_DOOR_ACTION,
                    brief="not staged; its streams cannot be fetched",
                    group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_UNREACHABLE))
            continue
        ok, evidence = front.probe(found)
        if not ok:
            checks.append(Check(
                name, "missing", f"{found} -- {evidence}; its streams cannot be fetched",
                front.remedy(), action=f"rebuild {front.name}", brief=_short(evidence),
                group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_UNREACHABLE))
            continue
        checks.append(Check(name, "verified", f"{found} -- {evidence}",
                            brief=_short(evidence), group=_GROUP_ENGINES))
    return checks


def _fetch_backbone_check() -> Check:
    """The vendored Rust fetch backbone: probe-execute, not stat().

    ``woof fetch --engine auto`` routes HRRR through this binary
    exactly when the check passes.  A missing one is not a gap in the
    estate -- the stdlib Python transport is the documented, always
    available fallback -- so an unbuilt backbone is ``info`` rather than
    ``missing``.  A binary that *exists* but reports a different
    fetch-record ABI is a different matter: that one fails after the
    download rather than before it, so it is reported ``missing``.
    """

    name = f"fetch backbone {rustwx_fetch.FETCH_NAME} (rust download engine)"
    try:
        found = rustwx_fetch.find_fetch_bin()
    except FileNotFoundError as error:
        return Check(
            name, "missing", str(error),
            f"# {rustwx_fetch.FETCH_ENV} names a missing executable: "
            "point it at a real build, or unset it and build one --\n"
            + rustwx_fetch.fetch_remedy(),
            action=f"unset {rustwx_fetch.FETCH_ENV}, or point it at a "
                   "real build",
            brief=f"{rustwx_fetch.FETCH_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        crate = rustwx_fetch.crate_dir() / "Cargo.toml"
        if crate.is_file():
            detail = f"not built yet (checkout crate: {crate.parent})"
        else:
            detail = (
                "no checkout crate and no prebuilt executable (searched "
                + ", ".join(str(c)
                            for c in rustwx_fetch.fetch_candidates()) + ")")
        return Check(
            name, "info",
            detail + " -- woof fetch falls back to the Python transport, "
            "which has no whole-file branch: every object arrives as "
            "hundreds of serial .idx range GETs, measured at 560 s for "
            "one 419 MB HRRR file against 27-35 s for the same file "
            "taken whole (~16x).  A run that pays it records "
            "engine_selection='python-fallback' in its fetch manifest",
            bridges.install_aware_build_hint(
                rustwx_fetch.CARGO_BUILD_HINT, "tools/rustwx")
            + "\n  # enables woof fetch --engine rust: parallel range "
            "GETs,\n  # the cross-process NOMADS rate governor, and "
            "--mode full-file",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not built; woof fetch pays a measured ~16x transport "
                  "tax",
            group=_GROUP_ENGINES)
    ok, evidence = rustwx_fetch.probe_fetch_bin(found)
    if not ok:
        return Check(
            name, "missing", f"{found} -- {evidence}",
            "# it has to be replaced:\n" + bridges.install_aware_build_hint(
                rustwx_fetch.CARGO_BUILD_HINT, "tools/rustwx"),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(evidence), group=_GROUP_ENGINES)
    return Check(name, "verified", f"{found} -- {evidence}",
                 brief=_short(evidence), group=_GROUP_ENGINES)


#: The door each MPAS binary unlocks, by artifact name.  What a reader
#: can type after ``pip install recast-woof``, never a crate path.
_MPAS_DOORS = {
    "rw_mpas_mesh": "woof mesh (MPAS mesh generation)",
    "rw_mpas_static": "woof mesh (the matching MPAS static)",
    "rw_mpas_init": "MPAS initial conditions from a grid and a static file",
    "rw_mpas_geometry": "MPAS reconstruction geometry",
    "rw_mpas_hostprep": "MPAS forecast host preparation",
    "rw_mpas_convert": "MPAS history onto the renderer's tape",
    "rw_mpas_lbc": "MPAS lateral boundaries for a limited-area mesh",
    "rw_hex_points": "soundings and meteograms from MPAS history "
                     "(the woof hex product door)",
    "rw_mpas_init_v831": "RRFS-MPAS initial conditions and boundaries "
                         "(woof hex init --physics-suite hrrrv5)",
    "rw_mpas_gsl_oro": "GSL orographic drag statistics for the RRFS suites",
}

#: One action for all five: they share a clone, a cargo build and a
#: reason, so the terse report folds them into a single line.
_MPAS_ACTION = ("build the MPAS binaries from a clone "
                f"({bridges.RUSTWX_CRATE_RELATIVE})")


def _mpas_bridge_checks() -> list[Check]:
    """The five MPAS binaries, reported by name.

    ``rw_mpas_convert`` is the artifact this module's other comments
    keep naming as the precedent: written, committed, resolved by name,
    carried by no bundle, reported by no check.  ``rw_mpas_mesh`` --
    the generator that reproduces the published NCAR meshes to 1e-11 --
    joined it in that state the day it was built, with no bundle row,
    no contract marker, no environment variable, no resolver and no
    Python that called it.  A capability with no front door is not a
    feature, and an estate that calls itself green over four absent
    binaries is the reason this file has a bundle-coverage check.

    Non-blocking.  These are a capability a bare install does not
    otherwise have, not a default path that silently degrades: nothing
    else in woof makes a mesh, so their absence closes ``woof mesh``
    and touches nothing else.  Staleness is judged as well as presence,
    statically, out of the bytes -- the same check
    ``woof fetch-bridges`` applies before it installs one.
    """

    try:
        from woof import mpas_mesh
    except ImportError as error:                 # pragma: no cover - partial
        return [Check(
            "MPAS binaries (mesh, static, init, convert, lbc)", "missing",
            f"woof.mpas_mesh is not importable ({error}) -- closes "
            "`woof mesh` and means this install is incomplete",
            "# reinstall so the mesh door imports:\n" + REINSTALL_HINT,
            action="reinstall woof", brief="mesh door not importable",
            group=_GROUP_ENGINES, severity=SEVERITY_BROKEN)]

    unnamed = sorted(set(mpas_mesh.BRIDGES) - set(_MPAS_DOORS))
    assert not unnamed, (
        f"woof.mpas_mesh gained {unnamed} and woof.doctor does not name "
        "the door(s) they unlock")

    checks: list[Check] = []
    for name, door in sorted(_MPAS_DOORS.items()):
        bridge = mpas_mesh.BRIDGES[name]
        label = f"MPAS binary {name} ({bridge.subject})"
        try:
            found = bridge.find()
        except FileNotFoundError as error:
            checks.append(Check(
                label, "missing", f"{error} -- {door} cannot run",
                f"# {bridge.env_var} names a missing executable: point it "
                "at a real build, or unset it --\n"
                + bridges.install_aware_build_hint(
                    mpas_mesh.CARGO_BUILD_HINT,
                    bridges.RUSTWX_CRATE_RELATIVE, name),
                action=f"unset {bridge.env_var}, or point it at a build",
                brief=f"{bridge.env_var} names a missing file",
                group=_GROUP_ENGINES, blocking=False,
                severity=SEVERITY_UNREACHABLE))
            continue
        if found is None:
            row = _on_demand_row(label, name, door)
            if row is not None:
                checks.append(row)
                continue
            checks.append(Check(
                label, "missing",
                f"not built and not staged -- {door} cannot run",
                bridge.remedy(), action=_MPAS_ACTION,
                brief=f"not staged; {door} cannot run",
                group=_GROUP_ENGINES, blocking=False,
                severity=SEVERITY_UNREACHABLE))
            continue
        ok, evidence = bridges.bridge_abi_matches(name, found)
        if not ok:
            checks.append(Check(
                label, "missing", f"{found} -- {evidence}; {door} would "
                "mis-parse the request it is given",
                "# this one is STALE, so re-point does not help; rebuild "
                "it:\n" + bridges.install_aware_build_hint(
                    mpas_mesh.CARGO_BUILD_HINT,
                    bridges.RUSTWX_CRATE_RELATIVE, name),
                action=_MPAS_ACTION, brief="stale build; rebuild it",
                group=_GROUP_ENGINES, severity=SEVERITY_BROKEN))
            continue
        checks.append(Check(label, "verified", f"{found} -- {evidence}"))

    # The sizing table is package data the mesh door reads on every
    # invocation: it holds the per-card footprint model every refusal is
    # built on and the smoothness bound. A wheel that lost it turns
    # `woof mesh --card` into a traceback rather than a refusal.
    try:
        sizing = mpas_mesh.load_sizing()
        measured = ", ".join(c.key for c in sizing.measured_cards())
        checks.append(Check(
            "MPAS mesh sizing table", "verified",
            f"{sizing.source} -- {len(sizing.cards)} card row(s), "
            f"measured: {measured or 'none'}; smoothness bound "
            f"{sizing.smoothness.refuse_above_percent_per_cell:.2f} %/cell "
            f"({sizing.smoothness.status})"))
    except Exception as error:                   # noqa: BLE001 - reported
        checks.append(Check(
            "MPAS mesh sizing table", "missing",
            f"unreadable ({error}) -- `woof mesh` cannot size a request "
            "against any card and cannot apply the smoothness bound",
            "# it is package data; a reinstall replaces it:\n"
            + REINSTALL_HINT,
            action="reinstall woof", brief="mesh sizing table unreadable",
            group=_GROUP_ENGINES, blocking=False,
            severity=SEVERITY_UNREACHABLE))
    return checks


def _icechunk_check() -> Check:
    from woof import output_formats

    def find():
        # Doctor reads the estate; it never stages an on-demand artifact.
        with bridges.inspection_only():
            return output_formats.find_binary()

    label = "Icechunk native history (rw_icechunk)"
    if not os.environ.get("WOOF_RW_ICECHUNK"):
        try:
            absent = find() is None
        except FileNotFoundError:
            absent = False
        if absent:
            row = _on_demand_row(label, "rw_icechunk", "--output-format icechunk",
                                 crate_relative="tools/rw_icechunk")
            if row is not None:
                return row
    return _byte_contract_door_check(
        label=label, name="rw_icechunk",
        env_var="WOOF_RW_ICECHUNK", find=find,
        remedy=output_formats.remedy, door="--output-format icechunk",
        missing_severity=SEVERITY_UNREACHABLE)


def _ml_export_check() -> Check:
    """``rw_mlexport``, the binary behind ``woof ml-export``.

    Non-blocking: it is a capability (turning history files into a
    training dataset), not a default path, so its absence closes
    ``ml-export`` and touches nothing else.  Staleness is judged out of
    the bytes, as ``woof fetch-bridges`` judges it.
    """

    from woof import ml_export

    binary = ml_export.BINARY
    # On demand since 2.9.11 (PyPI per-file limit): the first export stages it.
    row = _absent_on_demand_row(f"ML dataset exporter ({binary.name})", binary.name,
                                binary.env_var, binary.find, "woof ml-export")
    if row is not None:
        return row
    return _byte_contract_door_check(
        label=f"ML dataset exporter ({binary.name})", name=binary.name,
        env_var=binary.env_var, find=binary.find, remedy=binary.remedy,
        door="woof ml-export", missing_severity=SEVERITY_UNREACHABLE)


def _byte_contract_door_check(*, label, name, env_var, find, remedy,
                              door, missing_severity) -> Check:
    """Resolve an executable and judge the contract marker in its bytes."""
    try:
        found = find()
    except FileNotFoundError as error:
        return Check(
            label, "missing", f"{error} -- {door} cannot run",
            f"# {env_var} names a missing executable: point it at a "
            "real build, or unset it --\n" + remedy(),
            action=f"unset {env_var}, or point it at a build",
            brief=f"{env_var} names a missing file",
            group=_GROUP_ENGINES, blocking=False,
            severity=missing_severity)
    if found is None:
        return Check(
            label, "missing", f"not built and not staged -- {door} cannot run",
            remedy(), action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=f"not staged; {door} cannot run",
            group=_GROUP_ENGINES, blocking=False,
            severity=missing_severity)
    ok, evidence = bridges.bridge_abi_matches(name, found)
    if not ok:
        return Check(
            label, "missing", f"{found} -- {evidence}; {door} would hand it a "
            "request it does not speak",
            "# this one is STALE, so re-point does not help; rebuild it:\n"
            + remedy(),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="stale build; rebuild it", group=_GROUP_ENGINES,
            blocking=False, severity=SEVERITY_BROKEN)
    return Check(label, "verified", f"{found} -- {evidence}",
                 group=_GROUP_ENGINES)


def _rustwx_door_check(*, label: str, env_var: str, find, remedy: str,
                       doors: tuple[tuple[str, object], ...],
                       missing_severity: str = SEVERITY_UNREACHABLE) -> Check:
    """One ``tools/rustwx`` executable behind its doors.

    ``find`` is the resolver the doors call and each entry of ``doors``
    is ``(door, probe)`` with the probe THAT door refuses by, imported
    from the door's own module and never restated here.  Breakage this
    prevents: the first version of these lines judged every binary by the
    static :data:`woof.bridges.BRIDGE_ABI_MARKERS` byte search while
    ``woof render --compare`` judged ``rw_compare`` by its ``--abi``
    line, so a build carrying the reference-input literal and an older
    ``--abi`` line was reported verified here and refused at the door.
    Verified means every door would accept the build; otherwise the line
    names each door that would refuse it and that door's own evidence.
    Non-blocking: each closes only its own doors.
    """

    build = _build_action(bridges.RUSTWX_CRATE_RELATIVE)
    every_door = " and ".join(door for door, _ in doors)
    try:
        found = find()
    except (FileNotFoundError, RuntimeError) as error:
        return Check(
            label, "missing", f"{error} -- {every_door} cannot run",
            f"# {env_var} names a missing executable: point it at a real "
            "build, or unset it --\n" + remedy,
            action=f"unset {env_var}, or point it at a build",
            brief=f"{env_var} names a missing file",
            group=_GROUP_ENGINES, blocking=False,
            severity=missing_severity)
    if found is None:
        return Check(
            label, "missing",
            f"not built and not staged -- {every_door} cannot run",
            remedy, action=build, brief="not staged; its doors cannot run",
            group=_GROUP_ENGINES, blocking=False, severity=missing_severity)
    accepted: list[str] = []
    refused: list[str] = []
    for door, probe in doors:
        ok, evidence = probe(found)
        (accepted if ok else refused).append(f"{door}: {evidence}")
    if refused:
        return Check(
            label, "missing",
            f"{found} -- STALE for {len(refused)} of {len(doors)} door(s); "
            + "; ".join(f"{line} (refused)" for line in refused)
            + "".join(f"; {line}" for line in accepted),
            "# this one is STALE, so re-point does not help; rebuild it:\n"
            + remedy,
            action=build, brief="stale build; rebuild it",
            group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_BROKEN)
    return Check(label, "verified", f"{found} -- " + "; ".join(accepted),
                 group=_GROUP_ENGINES)


def _verification_engine_check() -> Check:
    """``rw_verify``, the native observation verification engine.

    Degraded rather than unreachable when absent: a completed run still
    finishes, but the observation verification it draws at finish by
    default (``WOOF_VERIFY_VISUALS``) and ``woof verify-visuals`` have
    nothing to score with.  Judged by
    :func:`woof.rustwx.probe_verification_binary`, the probe
    :func:`woof.rustwx.verification_binary` refuses by.
    """

    from woof import rustwx

    filename = bridges.executable_name("rw_verify")
    return _rustwx_door_check(
        label="observation verification engine (rw_verify)",
        env_var=rustwx.VERIFICATION_ENV,
        find=rustwx.find_verification_binary,
        remedy=bridges.artifact_remedy(
            env_var=rustwx.VERIFICATION_ENV, filename=filename,
            subject="the observation verification engine",
            crate_relative=bridges.RUSTWX_CRATE_RELATIVE,
            one_liner=bridges.rustwx_build_hint(), artifact="rw_verify"),
        doors=((("woof verify-visuals and the observation verification a "
                 "completed run draws"), rustwx.probe_verification_binary),),
        missing_severity=SEVERITY_DEGRADED)


def _probe_ensemble_reducer(path) -> tuple[bool, str]:
    """The refusal :func:`woof.rustwx.require_ensemble_diagnostic_reducer`
    applies before any ensemble member starts, as a door probe."""

    import subprocess

    from woof import rustwx

    try:
        rustwx.require_ensemble_diagnostic_reducer(path)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return False, str(error)
    return True, ("--ensemble-diagnostic-reduce-abi "
                  + rustwx.CPU_ENSEMBLE_REDUCTION_ABI)


def _ensemble_reducer_check() -> Check:
    """``rw_ensbatch``, the native ensemble diagnostic reducer.

    The bundle carries it (woof.bridge_assets, WOOF_ENSEMBLE_RENDERER)
    and the ensemble route refuses before its members start when it is
    missing or older than the reduction contract; with no line here a box
    whose copy was stale showed a green estate until an ensemble run was
    already queued.  Degraded rather than unreachable: single forecasts do
    not use it.
    """

    from woof import rustwx

    filename = bridges.executable_name("rw_ensbatch")
    return _rustwx_door_check(
        label="ensemble reducer (rw_ensbatch)",
        env_var="WOOF_ENSEMBLE_RENDERER",
        find=lambda: rustwx.find_ensemble_product_renderer(None),
        remedy=bridges.artifact_remedy(
            env_var="WOOF_ENSEMBLE_RENDERER", filename=filename,
            subject="the ensemble reducer",
            crate_relative=bridges.RUSTWX_CRATE_RELATIVE,
            one_liner=bridges.rustwx_build_hint(), artifact="rw_ensbatch"),
        doors=(("ensemble diagnostic reduction and aggregate product "
                "rendering", _probe_ensemble_reducer),),
        missing_severity=SEVERITY_DEGRADED)


def _comparison_engine_check() -> Check:
    """``rw_compare``, the reference-model comparison engine.

    Two doors, two contracts: ``woof render --compare`` refuses by
    :func:`woof.rustwx_lanes.probe_compare_bin` (the ``--abi`` line) and
    the reference panels beside a run's observation verification refuse
    by :func:`woof.rustwx_lanes.probe_compare_reference_bin` (the
    reference-input receipt).  The two landed on parallel branches, so a
    build can pass either alone, and this line runs both.
    """

    from woof import rustwx_lanes

    return _rustwx_door_check(
        label=f"comparison engine ({rustwx_lanes.COMPARE_NAME})",
        env_var=rustwx_lanes.COMPARE_ENV,
        find=rustwx_lanes.find_compare_bin,
        remedy=rustwx_lanes.compare_remedy(),
        doors=(("woof render --compare", rustwx_lanes.probe_compare_bin),
               ("the reference verification panels",
                rustwx_lanes.probe_compare_reference_bin)))


def _grib2_export_check() -> Check:
    """``rw_grib2export``, the binary behind ``woof export-grib2``.

    Held out of 2.8.6 with its binary (30dda9aeb dropped this line then);
    2.8.7 bundles the clean-room exporter again, so the line returns.
    Without it the native GRIB2 history export, and the live GRIB2 output a
    run writes as frames land, cannot run; nothing else is touched.  Judged
    by the exporter's own ``--abi`` probe, the one its door refuses by.
    """

    from woof import grib2_export

    binary = grib2_export.BINARY
    return _rustwx_door_check(
        label=f"GRIB2 exporter ({binary.name})",
        env_var=binary.env_var, find=binary.find, remedy=binary.remedy(),
        doors=(("woof export-grib2 and a run's live --grib2 output",
                binary.probe),))


def _gpu_post_kernels_check() -> Check:
    """Do the GPU post kernels load on this card?

    Breakage this names: through 2.9.0 the exporter's PTX (NVRTC 13.4,
    PTX ISA 9.4) was refused by drivers older than CUDA 13.4 (measured on
    595.91, CUDA 13.2: ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``), so on
    every card without an embedded CUBIN ``--grib2`` and ``export-grib2``
    computed every product on the CPU and said so only in manifest.json.
    Judged by the exporter's own ``--probe-post-device``, which opens the
    card the way a run does.  Degraded, never blocking: the CPU path
    writes the same bytes, only slower.
    """

    from woof import grib2_export

    name = "GPU post kernels (GRIB2 products on the GPU)"
    if no_local_gpu():
        return Check(name, "untested",
                     "not tested: GPUWM_NO_LOCAL_GPU is set, so the card is not opened",
                     brief="not tested (GPUWM_NO_LOCAL_GPU)", group=_GROUP_ENGINES)
    try:
        exe = grib2_export.BINARY.find()
    except (FileNotFoundError, RuntimeError):
        exe = None
    if exe is None:
        return Check(name, "untested",
                     "not tested: the GRIB2 exporter is not staged (see its own line)",
                     brief="not tested (no exporter)", group=_GROUP_ENGINES)
    probe = grib2_export.probe_post_device(exe)
    if probe is None:
        return Check(name, "untested",
                     f"not tested: {exe} has no --probe-post-device (an older build)",
                     brief="not tested (older exporter)", group=_GROUP_ENGINES)
    floor = probe.get("ptx_driver_floor") or "CUDA 12.0"
    if probe.get("ok"):
        return Check(name, "verified", str(probe.get("device")),
                     brief=str(probe.get("device")), group=_GROUP_ENGINES)
    if not probe.get("driver"):
        return Check(name, "info",
                     f"no NVIDIA driver: GRIB2 products are computed on the CPU ({probe.get('error')})",
                     brief="no NVIDIA driver; GRIB2 post on the CPU", group=_GROUP_ENGINES)
    if probe.get("operation") not in (None, "load post kernels", "resolve post kernel"):
        # No card to judge (none visible, or it would not open): not a
        # driver-floor gap, and the forecast's own GPU lines say why.
        return Check(name, "untested",
                     f"not tested: no CUDA device opened ({probe.get('error')}); GRIB2 products "
                     "would be computed on the CPU",
                     brief="not tested (no CUDA device opened)", group=_GROUP_ENGINES)
    return Check(
        name, "missing",
        f"the card did not load the post kernels, so --grib2 and export-grib2 compute every "
        f"product on the CPU (the same bytes, slower): {probe.get('error')}",
        "# GRIB2 output stays correct on the CPU.  To compute it on the GPU, install an\n"
        f"# NVIDIA driver at or above {floor}, then rerun `woof doctor`.",
        action=f"update the NVIDIA driver ({floor} or newer)",
        brief="kernels refused by the driver; GRIB2 post on the CPU",
        group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_DEGRADED)


def _nowcast_frames_check() -> Check:
    """``rw_nowcast_frames``, the NetCDF nowcast to frames-root converter.

    One door, ``rw_nowcast_frames from-netcdf``, judged by
    :func:`woof.rustwx_lanes.probe_nowcast_frames_bin` (the ``--abi``
    line), the probe :func:`woof.rustwx_lanes.require_nowcast_frames_bin`
    refuses by.  Degraded rather than unreachable when absent: forecasts,
    renders and the in-tree nowcast runner (which writes the frames
    contract itself) are untouched; only a NetCDF nowcast cannot reach a
    comparison sheet.
    """

    from woof import rustwx_lanes

    return _rustwx_door_check(
        label=f"nowcast frames converter ({rustwx_lanes.NOWCAST_FRAMES_NAME})",
        env_var=rustwx_lanes.NOWCAST_FRAMES_ENV,
        find=rustwx_lanes.find_nowcast_frames_bin,
        remedy=rustwx_lanes.nowcast_frames_remedy(),
        doors=(("rw_nowcast_frames from-netcdf (NetCDF nowcasts onto the "
                "comparison sheets)", rustwx_lanes.probe_nowcast_frames_bin),),
        missing_severity=SEVERITY_DEGRADED)


def _observation_scorecard_check() -> Check:
    """``rw_scorecard``, the native standing observation scorecard.

    One door, ``woof obs scorecard``, judged by the bundle contract
    marker :func:`woof.obs.scorecard.binary` refuses by.  Degraded rather
    than unreachable when absent: forecasts and renders are untouched;
    only the recorded-case scorecard cannot run.
    """

    from woof.obs import scorecard

    def probe(path):
        return bridges.bridge_abi_matches("rw_scorecard", path)

    filename = bridges.executable_name("rw_scorecard")
    return _rustwx_door_check(
        label="observation scorecard (rw_scorecard)",
        env_var="WOOF_RW_SCORECARD",
        find=scorecard.find_binary,
        remedy=bridges.artifact_remedy(
            env_var="WOOF_RW_SCORECARD", filename=filename,
            subject="the observation scorecard",
            crate_relative=bridges.RUSTWX_CRATE_RELATIVE,
            one_liner=bridges.rustwx_build_hint(), artifact="rw_scorecard"),
        doors=(("woof obs scorecard", probe),),
        missing_severity=SEVERITY_DEGRADED)


def _forecast_scoreboard_check() -> Check:
    """``rw_scoreboard``, the forecast verification scorer.

    One door, ``woof scoreboard --request``, judged by
    :func:`woof.scoreboard.probe` (the ``--abi`` line), the refusal that
    door applies.  Degraded rather than unreachable when absent: only the
    scoreboard requests cannot run.
    """

    from woof import scoreboard

    label = "forecast verification scorer (rw_scoreboard)"
    row = _absent_on_demand_row(label, scoreboard.BINARY.name, scoreboard.BINARY.env_var,
                                scoreboard.BINARY.find, "woof scoreboard --request")
    if row is not None:
        return row
    return _rustwx_door_check(
        label=label,
        env_var=scoreboard.BINARY.env_var,
        find=scoreboard.BINARY.find,
        remedy=scoreboard.BINARY.remedy(),
        doors=(("woof scoreboard --request", scoreboard.probe),),
        missing_severity=SEVERITY_DEGRADED)


def _tc_fields_check() -> Check:
    """``rw_tc_fields``, the cyclone field reader behind ``woof tc-tracks``.

    Resolved through the same ladder :func:`woof.tc_tracks.binary` walks
    (``WOOF_RW_TC_FIELDS``, the staged bundle, the rustwx build) and judged
    stale out of its bytes, as ``woof fetch-bridges`` judges it.  Degraded
    rather than unreachable when absent: forecasts and renders are
    untouched; only cyclone tracking on finished output cannot run.
    """

    name = "rw_tc_fields"
    env_var = "WOOF_RW_TC_FIELDS"
    filename = bridges.executable_name(name)
    label = f"cyclone field reader ({name})"
    door = "woof tc-tracks"
    remedy = bridges.artifact_remedy(
        env_var=env_var, filename=filename,
        subject="the cyclone field reader",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE,
        one_liner=bridges.rustwx_build_hint(), artifact=name)
    row = _absent_on_demand_row(label, name, env_var, lambda: bridges.find_artifact(
        env_var, filename, crate_relative=bridges.RUSTWX_CRATE_RELATIVE), door)
    if row is not None:
        return row
    return _byte_contract_door_check(
        label=label, name=name, env_var=env_var,
        find=lambda: bridges.find_artifact(
            env_var, filename, crate_relative=bridges.RUSTWX_CRATE_RELATIVE),
        remedy=lambda: remedy, door=door, missing_severity=SEVERITY_DEGRADED)


#: The 2.9.4 capability lanes' tools that are not obs front doors, with the
#: door each one serves (cap-fire-indices, cap-fzra, cap-waves-couple,
#: cap-slab-ocean, cap-surge).
_CAPABILITY_TOOLS = {
    "rw_fire_indices": ("fire danger indices", "the stateful FWI/KBDI door (woof.fire_danger_state)"),
    "rw_ice_verify": ("ice observation binding", "freezing-rain verification (woof.ensemble.ice_accretion verify)"),
    "rw_ocean": ("ocean preparation", "the coupled ocean door (woof.ocean.door)"),
    "rw_bathy": ("surge bathymetry", "the storm-surge door (woof.ocean.surge)"),
    "rw_ocean_verify": ("ocean SST verification", "ocean SST verification (woof.ocean.mixed_layer.verify)"),
    # labs/windfarm (2.9.11): the turbine-inventory decoder behind --wind-farms.
    "rw-wind-table": ("wind turbine tables", "the wind-farm doors (woof go --wind-farms, woof.wind_inputs)"),
}


def _capability_tool_checks() -> list[Check]:
    """The capability lanes' five non-front-door tools, one line each.

    Breakage it prevents: the 2.9.4 capability lanes bundled rw_fire_indices,
    rw_ice_verify, rw_ocean, rw_ocean_verify and rw_bathy with no doctor line, so an
    install whose fire danger, ice verification, ocean or surge doors could not run
    read green (tests/test_doctor_route_accuracy.py::
    test_every_bundled_artifact_is_covered_by_a_check).  Each tool is resolved
    through the artifact ladder without fetching, an absent on-demand tool gets
    the on-demand row, and a present one is judged stale out of its bytes
    against its BRIDGE_ABI_MARKERS contract.  Degraded, never blocking:
    forecasts are untouched; only that door cannot run.
    """
    from woof import bridge_assets

    checks = []
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        if artifact.name not in _CAPABILITY_TOOLS:
            continue
        subject, door = _CAPABILITY_TOOLS[artifact.name]
        label = f"{subject} ({artifact.name})"
        filename = bridge_assets.artifact_filename(
            artifact, bridge_assets.host_platform() or "linux-x86_64")
        remedy = bridges.artifact_remedy(
            env_var=artifact.env_var, filename=filename, subject=subject,
            crate_relative=artifact.crate, artifact=artifact.name)
        with bridges.inspection_only():
            candidates = bridges.artifact_candidates(
                artifact.env_var, filename, crate_relative=artifact.crate)
        found = next((candidate for candidate in candidates
                      if candidate.is_file() and os.access(candidate, os.X_OK)), None)
        if found is None:
            row = (_on_demand_row(label, artifact.name, door, crate_relative=artifact.crate)
                   if artifact.on_demand and not os.environ.get(artifact.env_var) else None)
            if row is not None:
                checks.append(row)
                continue
            checks.append(Check(
                label, "missing", f"{filename} not built and not staged -- {door} cannot run",
                remedy, action=_build_action(artifact.crate),
                brief=f"not staged; {door} cannot run", group=_GROUP_ENGINES,
                severity=SEVERITY_DEGRADED, blocking=False))
            continue
        ok, evidence = bridges.bridge_abi_matches(artifact.name, found)
        if not ok:
            checks.append(Check(
                label, "missing", f"{found} -- {evidence}",
                "# this one is STALE, so re-point does not help; rebuild it:\n" + remedy,
                action=_build_action(artifact.crate), brief="stale build; rebuild it",
                group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_BROKEN))
            continue
        checks.append(Check(label, "verified", f"{found} -- {evidence}", group=_GROUP_ENGINES))
    return checks


def _earth2_runtime_checks() -> list[Check]:
    """The Earth2Studio interpreter and the TempestExtremes tools ``woof tc-tracks`` runs.

    Breakage these lines prevent (2.9.3 door 6, G12, 2026-10-20): on a fresh install every
    tc-tracks backend refused and earth2-diagnose stopped, while this report checked only
    ``rw_tc_fields`` and said nothing.  Resolved through :mod:`woof.earth2_env`, the same lookup
    the doors use.  Absent while the doors may build the managed environment on first use, a row is
    opt-in, as an on-demand bridge is before its first fetch; with WOOF_EARTH2_FETCH=0 the doors
    refuse instead, so it is degraded.  Breakage the opt-in prevents (2.9.4 R15 at 2147c15faf,
    RTX 4090, 2026-10-22): every bare pip install reported both rows degraded and failed the defaults
    leg's doctor row although the first tc-tracks or earth2-diagnose call builds them.
    """
    from woof import earth2_env

    absent = SEVERITY_OPT_IN if earth2_env.fetch_allowed() else SEVERITY_DEGRADED
    checks = []
    label = "Earth2Studio interpreter (tc-tracks wuduan/vitart, earth2-diagnose)"
    try:
        found, notes = earth2_env.find_python(("cyclone",))
    except Exception as error:  # a broken table must still print a line
        found, notes = None, [f"{type(error).__name__}: {error}"]
    if found:
        checks.append(Check(label, "verified", f"{found} -- earth2studio with the [cyclone] extra"))
    else:
        detail = "not found -- " + ("; ".join(notes) or "no candidate")
        checks.append(Check(
            label, "missing", detail, "# " + earth2_env.remedy(("cyclone",)),
            action=earth2_env.SETUP_COMMAND,
            brief="missing; built on first use or by " + earth2_env.SETUP_COMMAND,
            blocking=False, severity=absent))
    label = "TempestExtremes DetectNodes/StitchNodes (tc-tracks --backend tempestextremes)"
    try:
        tools = [earth2_env.find_tool(name) for name in earth2_env.TEMPEST_TOOLS]
        problem = None
    except ValueError as error:
        tools, problem = [None, None], str(error)
    if all(tools):
        checks.append(Check(label, "present", " ".join(tools)))
    else:
        missing = [n for n, t in zip(earth2_env.TEMPEST_TOOLS, tools) if not t]
        checks.append(Check(
            label, "missing", problem or f"{', '.join(missing)} not found (WOOF_DETECTNODES/"
            f"WOOF_STITCHNODES, PATH, {earth2_env.managed_tool_dir()})",
            "# " + earth2_env.tempest_remedy(), action=earth2_env.SETUP_COMMAND,
            brief="missing; built on first use or by " + earth2_env.SETUP_COMMAND,
            blocking=False, severity=absent))
    return checks


def _absent_on_demand_row(label: str, artifact: str, env_var: str, find, door: str) -> Check | None:
    """The on-demand row when ``artifact`` resolves nowhere and no override names it, else None.

    Breakage it prevents: rw_scoreboard and rw_tc_fields went on demand in 2.9.4 (PyPI per-file
    limit), and their rows would read "not built and not staged" on a fresh install whose first
    use stages them.  A set override, a present build or a missing pin falls through to the
    ordinary line.
    """
    if os.environ.get(env_var):
        return None
    try:
        with bridges.inspection_only():
            found = find()
    except (FileNotFoundError, RuntimeError):
        return None
    if found is not None:
        return None
    return _on_demand_row(label, artifact, door)


def _on_demand_row(label: str, artifact: str, door: str, *,
                   crate_relative: str = bridges.RUSTWX_CRATE_RELATIVE) -> Check | None:
    """The on-demand row for an absent artifact the platform wheel leaves to the bundle, or None.

    The row the obs front doors get (``on demand; staged by its first fetch``), for the other
    artifacts that are on demand.  Breakage it prevents (2.9.3 pass 23 R12, Windows wheel smoke):
    rw_soundings and rw_nextgen_verify went on demand in 3b66d4c13ce, but their rows still said
    "not staged; its doors cannot run", which a fresh install contradicts: the first use stages them.
    """
    if not bridges.on_demand_fetch_offered(artifact, crate_relative):
        return None
    return Check(
        label, "missing",
        "not staged yet -- it is fetched on demand: the platform wheel does not carry it, and "
        f"the first `{door}` that needs it stages it from this release's bundle",
        f"woof fetch-bridges --partial --artifact {artifact}",
        action=f"woof fetch-bridges --partial --artifact {artifact}",
        brief="on demand; staged by its first fetch",
        group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_OPT_IN)


def _sounding_renderer_check() -> Check:
    """``rw_soundings``, the website-style sounding window renderer.

    One door, ``woof soundings``, judged by :func:`woof.soundings.probe`
    (the ``--abi`` line), the refusal that door applies.  Degraded rather
    than unreachable when absent: forecasts and renders are untouched; only
    the sounding product cannot be drawn.

    A build the door accepts is then asked to create its render device
    (:func:`woof.soundings.device`).  Breakage this prevents: the ``--abi``
    line alone read green on a headless server without the Vulkan loader
    where every render panicked creating its device.  The line names the
    device (GPU adapter, or the CPU rasteriser and why no GPU was usable).
    """

    from woof import soundings

    label = "sounding window renderer (rw_soundings)"
    filename = bridges.executable_name("rw_soundings")
    try:
        absent = soundings.find_binary() is None
    except FileNotFoundError:
        absent = False          # a named override that is missing: the door check reports it
    if absent:
        row = _on_demand_row(label, "rw_soundings", "woof soundings")
        if row is not None:
            return row
    door = _rustwx_door_check(
        label=label,
        env_var=soundings.ENV,
        find=soundings.find_binary,
        remedy=bridges.artifact_remedy(
            env_var=soundings.ENV, filename=filename,
            subject="the sounding window renderer",
            crate_relative=bridges.RUSTWX_CRATE_RELATIVE,
            one_liner=bridges.rustwx_build_hint(), artifact="rw_soundings"),
        doors=(("woof soundings", soundings.probe),),
        missing_severity=SEVERITY_DEGRADED)
    if door.status != "verified":
        return door
    found = soundings.find_binary()
    ok, evidence = soundings.device(found)
    if ok:
        return Check(label, "verified", f"{door.detail}; render device: {evidence}",
                     group=_GROUP_ENGINES)
    return Check(
        label, "missing", f"{found} -- cannot create a render device: {evidence}",
        "# install the Vulkan loader for GPU rendering, or let the CPU rasteriser draw:\n"
        "sudo apt-get install -y libvulkan1\n"
        f"unset {soundings.RENDERER_ENV}",
        action="sudo apt-get install -y libvulkan1",
        brief="no render device", group=_GROUP_ENGINES, blocking=False,
        severity=SEVERITY_DEGRADED)


def _simulated_radar_check() -> Check:
    """Check the optional native radar capability without touching a GPU.

    Every gap carries its next command, as the sibling bridge checks do:
    the one-line doctor layer otherwise showed a radar gap that led nowhere.
    """
    from woof.rustwx import SIMULATED_RADAR_ENV, simulated_radar_binary
    label = "simulated radar (rw_simradar)"
    remedy = bridges.artifact_remedy(
        env_var=SIMULATED_RADAR_ENV, filename=bridges.executable_name("rw_simradar"),
        subject="the simulated radar engine", crate_relative=bridges.RUSTWX_CRATE_RELATIVE,
        one_liner=bridges.rustwx_build_hint(), artifact="rw_simradar")
    build = _build_action(bridges.RUSTWX_CRATE_RELATIVE)
    try:
        found = simulated_radar_binary()
    except (FileNotFoundError, RuntimeError) as error:
        return Check(label, "missing", str(error), remedy,
                     action=f"unset {SIMULATED_RADAR_ENV}, or point it at a build",
                     brief=f"{SIMULATED_RADAR_ENV} names a missing file",
                     group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_UNREACHABLE)
    if found is None:
        # On demand since 2.9.11 (PyPI per-file limit): a fresh install's
        # first radar door stages it, so the row says that, not "cannot run".
        row = (None if os.environ.get(SIMULATED_RADAR_ENV) else
               _on_demand_row(label, "rw_simradar",
                              "woof simulated-radar or [simulated_radar] run"))
        if row is not None:
            return row
        return Check(label, "missing",
                     "not staged; simulated radar cannot run" ,
                     remedy, action=build, brief="not staged; simulated radar cannot run",
                     group=_GROUP_ENGINES, blocking=False,
                     severity=SEVERITY_UNREACHABLE)
    ok, evidence = bridges.bridge_abi_matches("rw_simradar", found)
    return Check(label, "verified" if ok else "missing", f"{found}: {evidence}",
                 None if ok else remedy, action=None if ok else build,
                 brief=None if ok else "stale build; rebuild it",
                 group=_GROUP_ENGINES, blocking=False,
                 severity=None if ok else SEVERITY_BROKEN)


def _nexrad_front_door_check() -> Check:
    """The radar front door: is it here, and is it current?

    A bundled binary with **no fallback**.  A missing fetch backbone
    leaves the Python transport, so that one is ``info``; there is no
    second way to turn a radar volume into observations, so an absent
    ``rw_nexrad`` is the difference between a box that can assimilate
    and a box that cannot.  (It is not the ONLY such binary, which this
    docstring used to claim: ``region_global_dealias`` has no fallback
    either -- ``vad-region`` is a different solver, not a substitute --
    and the renderer's "matplotlib fallback" turned out to need the
    ``[render]`` extra as well.  Both were measured, 2026-08-14.)
    It is reported ``missing`` and it blocks, for the same reason the
    GRIB bridges do: it ships in the bundle ``woof fetch-bridges``
    stages, so its absence means an incomplete install rather than an
    unexercised option.

    This check exists because the report used to pass without it.
    ``rw_nexrad`` was outside both audited sets -- not a GRIB bridge,
    not a render engine -- so ``woof doctor`` printed a clean estate on
    installs where every radar route was dead, and the first news came
    from the launcher refusing to print a plan.

    Staleness is judged, not just presence.  The wrapper pins the exact
    ``--abi`` contract line it was written against, so a binary built
    before the live-chunk route fails the probe here rather than at the
    first live fetch -- and the remedy for that one is *rebuild*, never
    re-point, which is why the stale branch says so in those words.
    """

    # Imported here rather than at module scope, and the reason is what
    # this module IS.  `woof.obs.nexrad` needs only the standard library,
    # but reaching it executes `woof/obs/__init__.py`, which imports the
    # gridding stack and therefore numpy.  doctor is the tool for
    # diagnosing a broken or partial install, so a doctor that cannot be
    # imported without the full scientific stack cannot diagnose the
    # installs it exists for -- and the release pipeline proved it: the
    # bridges job imports this module for `_exec_probe` alone, installs no
    # dependencies, and died on `import numpy` three jobs deep.
    #
    # The failure branch is NOT a quiet skip.  This check was added
    # because the report used to pass without it, so a version that can
    # vanish when an import fails would reintroduce exactly the
    # silent-green hole it closes.  An unimportable obs stack is itself a
    # broken install, and it is reported as one.
    try:
        from woof.obs import nexrad
    except ImportError as error:
        return Check(
            "radar front door (radar observation ingest)", "missing",
            f"woof.obs is not importable ({error}) -- blocks ALL radar "
            "observation ingest, and means this install is incomplete "
            "rather than merely missing the binary",
            "# reinstall so the observation stack imports:\n"
            f"{REINSTALL_HINT}",
            action="reinstall woof", brief="obs stack not importable",
            group=_GROUP_BRIDGES)

    name = f"radar front door {nexrad.NEXRAD_NAME} (radar observation ingest)"
    blocks = ("blocks ALL radar observation ingest -- every DA nowcast "
              "route, live and archived")
    try:
        found = nexrad.find_nexrad_bin()
    except FileNotFoundError as error:
        return Check(
            name, "missing", f"{error} -- {blocks}",
            f"# {nexrad.NEXRAD_ENV} names a missing executable: point it "
            "at a real build, or unset it and get one --\n"
            + nexrad.nexrad_remedy(),
            action=f"unset {nexrad.NEXRAD_ENV}, or point it at a real build",
            brief=f"{nexrad.NEXRAD_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        crate = nexrad.crate_dir() / "Cargo.toml"
        if crate.is_file():
            detail = f"not built yet (checkout crate: {crate.parent})"
        else:
            detail = ("no checkout crate and no prebuilt executable "
                      "(searched "
                      + ", ".join(str(c)
                                  for c in nexrad.nexrad_candidates()) + ")")
        return Check(
            name, "missing", f"{detail} -- {blocks}",
            nexrad.nexrad_remedy()
            + "\n  # without it nothing can read a radar volume: no "
            "superobs,\n  # no analysis, no nowcast",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not staged; no radar ingest", group=_GROUP_ENGINES)
    ok, evidence = nexrad.probe_nexrad_bin(found)
    if not ok:
        return Check(
            name, "missing", f"{found} -- {evidence}; {blocks}",
            "# REBUILD it -- do not re-point WOOF_RW_NEXRAD at another\n"
            "  # copy: this is a contract change, so every binary older\n"
            "  # than it fails the same way --\n"
            + bridges.install_aware_build_hint(
                nexrad.CARGO_BUILD_HINT, "tools/rustwx"),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(evidence), group=_GROUP_ENGINES)
    return Check(name, "verified", f"{found} -- {evidence}",
                 brief=_short(evidence), group=_GROUP_ENGINES)


def _netcdf_decoder_check() -> Check:
    """The NetCDF decoder, which every NetCDF source now needs.

    ``missing`` rather than ``info``, for the same reason ``rw_nexrad``
    is: since NetCDF decode moved off ``netCDF4.Dataset`` and onto the
    vendored ``netcrust`` reader, an absent ``rw_netcdf`` does not cost
    an option -- it costs every NetCDF source the product accepts, and
    there is deliberately no Python decoder behind it to quietly take
    over.  A prerequisite of a shipped path belongs in the blocking
    group, and the remedy is the build, never a workaround flag.
    """

    name = "NetCDF decoder (every NetCDF source)"
    blocks = ("blocks every NetCDF source -- woof adapt, the mapped "
              "routes, rw-wps --source netcdf")
    try:
        from woof import netcdf_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"woof.netcdf_bridge is not importable ({error}) -- "
                     f"{blocks}",
                     "# reinstall so the NetCDF front door imports:\n"
                     "  pip install --force-reinstall woof",
                     brief="netcdf_bridge not importable",
                     group=_GROUP_ENGINES)
    try:
        found = netcdf_bridge.find_netcdf_bin()
    except netcdf_bridge.NetcdfDecodeError:
        return Check(
            name, "missing",
            "the staged reader cannot supply numeric and character records, "
            f"including WRF Times -- {blocks}",
            netcdf_bridge.netcdf_remedy(),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="reader predates character records",
            group=_GROUP_ENGINES)
    except FileNotFoundError as error:
        return Check(
            name, "missing", f"{error} -- {blocks}",
            f"# {netcdf_bridge.NETCDF_ENV} names a missing executable: "
            "point it at a real build, or unset it and get one --\n"
            + netcdf_bridge.netcdf_remedy(),
            action=f"unset {netcdf_bridge.NETCDF_ENV}, or point it at a "
                   f"real build",
            brief=f"{netcdf_bridge.NETCDF_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        return Check(
            name, "missing",
            "no prebuilt executable (searched "
            + ", ".join(str(c) for c in netcdf_bridge.netcdf_candidates())
            + f") -- {blocks}",
            netcdf_bridge.netcdf_remedy()
            + "\n  # without it no NetCDF source can be read at all: woof\n"
            "  # decodes NetCDF in Rust and keeps no Python fallback",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not staged; no NetCDF source can be read",
            group=_GROUP_ENGINES)
    build = bridges.checkout_build_status(found)
    if build is not None and not build.current:
        return Check(
            name, "missing", f"{build.describe()} -- {blocks}",
            "# this checkout built it, and its sources have moved since;\n"
            "# rebuild before the next run reaches it --\n"
            f"  {build.remedy()}",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="checkout build is older than its own sources",
            group=_GROUP_ENGINES)
    return Check(name, "verified", str(found),
                 brief="staged", group=_GROUP_ENGINES)


def _mapped_engine_check() -> Check:
    """The default mapped decoder (Rust since 10ef633138d), with its gaps."""

    name = "mapped decode engine (woof prep --source mapped)"
    try:
        from woof import mapped_engine_bridge as engine_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"woof.mapped_engine_bridge is not importable ({error})",
                     "# reinstall so the mapped seam imports:\n"
                     "  pip install --force-reinstall woof",
                     brief="mapped_engine_bridge not importable",
                     group=_GROUP_ENGINES)
    default = engine_bridge.DEFAULT_ENGINE
    try:
        found = engine_bridge.find_engine()
    except FileNotFoundError as error:
        return Check(
            name, "missing", str(error),
            f"# {engine_bridge.ENGINE_PATH_ENV} names a missing "
            "executable: point it at a real build, or unset it --\n"
            + engine_bridge.engine_remedy_lines(
                "the override names no file"),
            action=f"unset {engine_bridge.ENGINE_PATH_ENV}, or point it "
                   "at a real build",
            brief=f"{engine_bridge.ENGINE_PATH_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        detail = (
            "no prebuilt executable (searched "
            + ", ".join(str(c) for c in engine_bridge.engine_candidates())
            + ")")
        return Check(
            name, "missing",
            f"{detail} -- blocks every mapped source, which is the "
            "default decode path",
            engine_bridge.engine_remedy_lines("it is not staged"),
            action=_build_action(engine_bridge.ENGINE_CRATE_RELATIVE),
            brief="not staged; no mapped source can be read",
            group=_GROUP_ENGINES)
    matches, reason = bridges.bridge_abi_matches(
        engine_bridge.ENGINE_NAME, found)
    if not matches:
        return Check(
            name, "missing",
            f"{found} {reason}",
            engine_bridge.engine_remedy_lines(reason),
            action=_build_action(engine_bridge.ENGINE_CRATE_RELATIVE),
            brief="staged build predates this release's frameset contract",
            group=_GROUP_ENGINES, blocking=True)
    # The paths still routed to Python are named HERE rather than
    # left for a reader to discover from a receipt: this line is
    # what a user reads to learn where their bytes are decoded, and
    # "the Rust engine is the default" without the exceptions would
    # be a true sentence that leaves a false impression.
    suffix = "".join(f"; {gap}" for gap in engine_bridge.ENGINE_GAPS)
    return Check(
        name, "verified", f"{found} speaks {engine_bridge.FRAMESET_SCHEMA}"
        + suffix,
        brief=f"staged; default engine is {default}",
        group=_GROUP_ENGINES)


def _library_check(name, bridge, *, resolve, abi, remedy, consequence,
                   brief, missing_errors=(FileNotFoundError,)) -> Check:
    """Resolve and load a library through its own seam, retaining ABI evidence."""
    try:
        path = getattr(bridge, resolve)()
    except missing_errors as error:
        return Check(
            name, "missing", f"{error} -- {consequence}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=f"not staged; {brief}", group=_GROUP_ENGINES)
    reason = bridge.unavailable_reason()
    if reason is not None:
        return Check(
            name, "missing", f"{path} -- {reason} -- {consequence}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=f"not loadable; {brief}", group=_GROUP_ENGINES)
    return Check(name, "verified", f"{path} -- ABI {getattr(bridge, abi)}",
                 brief="staged", group=_GROUP_ENGINES)


def _ncwrite_check() -> Check:
    """The NetCDF WRITER, which two default engines now need.

    ``missing`` rather than ``info``, for the same reason ``rw_netcdf``
    is: since the product tape and then the wrfinput/wrfbdy pair flipped
    onto the Rust classic writer, an absent ``netcdf_writer`` library
    does not cost an option -- it costs every history write AND every
    WRF-input export, and both refuse loudly rather than silently
    downgrading a container with build state.  The remedy is the build;
    ``WOOF_WRFOUT_WRITER=python`` and ``WOOF_WRFINPUT_WRITER=python``
    are named as what they are, documented workarounds onto the netCDF4
    writer, not fallbacks the product takes on its own.
    """

    name = "NetCDF writer (default wrfout and wrfinput engine)"
    blocks = ("blocks every history write and every WRF-input export -- "
              "woof sim, woof go, woof prep, woof-wrf-init, rw-wps, "
              "the tilestream history path")
    try:
        from woof.io import nc_writer_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"woof.io.nc_writer_bridge is not importable "
                     f"({error}) -- {blocks}",
                     "# reinstall so the writer seam imports:\n"
                     "  pip install --force-reinstall woof",
                     brief="nc_writer_bridge not importable",
                     group=_GROUP_ENGINES)
    # Install-aware: `woof fetch-bridges` first on a wheel, the in-tree
    # build for a checkout -- then name the workaround AS a workaround.
    remedy = bridges.artifact_remedy(
        env_var=nc_writer_bridge.NCWRITE_BRIDGE_ENV,
        filename=nc_writer_bridge.library_names()[0],
        subject="the NetCDF writer",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE) + (
        "\n  # WOOF_WRFOUT_WRITER=python (history) and\n"
        "  # WOOF_WRFINPUT_WRITER=python (the WRF input pair) select the\n"
        "  # netCDF4 writer as an explicit workaround, each onto its own\n"
        "  # HDF5 container (NETCDF4_CLASSIC / NETCDF4)")
    return _library_check(
        name, nc_writer_bridge, resolve="resolve_ncwrite_bridge", abi="NCWRITE_ABI",
        remedy=remedy, consequence=blocks,
        brief="default history/WRF-input writes blocked")


def _static_builder_check() -> Check:
    """The static-field builder, which the default build path now needs.

    ``missing`` rather than ``info``, the ``netcdf_writer`` posture:
    since ``build_static`` and the ProjectedGrid array methods flipped
    onto the Rust crate by default (the static-rust-port lanes), an
    absent ``static_fields`` library does not cost an option -- it turns
    every static build into a reported WORKAROUND onto the numpy
    fallback.  The fallback still produces byte-identical fields (the
    dual-run parity record in docs/dev/static-rust-port.md), so the
    breakage this row names is the degradation itself: a wheel whose
    every prepare prints a workaround line is not running the shipped
    configuration, and at corridor scale the fallback is measured ~11x
    slower.
    """

    name = "static builder (default static-field engine)"
    degrades = ("every static build -- woof go's prepare stage, nest "
                "initialization, the statics corridor, and the whole "
                "[static.highres] overlay including its GeoTIFF decode "
                "and warp -- runs the pure-Python fallback as a reported "
                "WORKAROUND, which needs the [geog] extra installed")
    try:
        from woof.static import rust_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"woof.static.rust_bridge is not importable "
                     f"({error}) -- {degrades}",
                     "# reinstall so the builder seam imports:\n"
                     "  pip install --force-reinstall woof",
                     brief="rust_bridge not importable",
                     group=_GROUP_ENGINES)
    remedy = bridges.artifact_remedy(
        env_var=rust_bridge.STATIC_BRIDGE_ENV,
        filename=rust_bridge.library_names()[0],
        subject="the static-field builder",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE) + (
        "\n  # WOOF_STATIC_PYTHON=1 selects the numpy fallback as an\n"
        "  # explicit workaround (byte-identical fields, measured ~11x\n"
        "  # slower at corridor scale)")
    return _library_check(
        name, rust_bridge, resolve="resolve_static_bridge", abi="STATIC_ABI",
        remedy=remedy, consequence=degrades,
        brief="static builds degrade to the fallback")


def _obs_regrid_check() -> Check:
    """The observation remap, which the battery's default now needs.

    ``missing`` rather than ``info``, the ``static_fields`` posture:
    since `woof.verify.obs.regrid` flipped onto the Rust crate by
    default, an absent ``obs_regrid`` library does not cost an option --
    every observation score becomes a reported WORKAROUND onto the scipy
    fallback.

    The breakage this row names is not slowness.  On the scipy path two
    source cells exactly equidistant from one destination cell are
    resolved by whichever the cKDTree traversal reached first (measured:
    232 of 400 trials answered the lowest index, 168 answered another),
    so a remap plan stops being a function of its two grids alone.  The
    battery's premise is that every arm is remapped by the identical
    integer array, which is what makes arm scores comparable; a plan
    whose ties turn on traversal order is that premise, unenforced.
    """

    name = "observation remap (default battery remap engine)"
    degrades = ("every observation score -- python -m "
                "tools.obs_battery_score, tools.obs_precampaign_controls, "
                "tools.obs_battery_registration -- runs the scipy fallback "
                "as a reported WORKAROUND, where exactly-tied nearest "
                "neighbours are resolved by cKDTree traversal order rather "
                "than by lowest source index")
    try:
        from woof import obs_regrid_bridge as regrid_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"woof.obs_regrid_bridge is not importable "
                     f"({error}) -- {degrades}",
                     "# reinstall so the remap seam imports:\n"
                     "  pip install --force-reinstall woof",
                     brief="regrid_bridge not importable",
                     group=_GROUP_ENGINES)
    remedy = bridges.artifact_remedy(
        env_var=regrid_bridge.OBSREGRID_BRIDGE_ENV,
        filename=regrid_bridge.library_names()[0],
        subject="the observation remap engine",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE) + (
        "\n  # WOOF_OBSREGRID_PYTHON=1 selects the scipy fallback as an\n"
        "  # explicit workaround (bit-identical everywhere except exact\n"
        "  # neighbour ties, where scipy has no defined answer)")
    return _library_check(
        name, regrid_bridge, resolve="resolve_obsregrid_bridge", abi="OBSREGRID_ABI",
        remedy=remedy, consequence=degrades,
        brief="observation scores degrade to the fallback")


def _noah_init_check() -> Check:
    """A native host preparation must be able to partition frozen soil water."""
    from woof import noah_init_bridge

    reason = noah_init_bridge.unavailable_reason()
    if reason is not None:
        return Check(
            "Noah soil initialization", "missing", reason,
            bridges.install_aware_build_hint(bridges.CARGO_BUILD_HINT),
            action=_build_action(), brief="frozen soil cannot be prepared",
            group=_GROUP_ENGINES)
    return Check("Noah soil initialization", "verified",
                 "native float64 frozen-soil solve is available",
                 brief="staged", group=_GROUP_ENGINES)


def _obs_score_check() -> Check:
    """The numerical scoring library required by observation verification."""
    name = "observation scoring"
    try:
        from woof import obs_score_bridge as score_bridge
    except ImportError as error:
        return Check(name, "missing", f"scoring seam is not importable: {error}",
                     "pip install --force-reinstall woof", brief="seam missing",
                     group=_GROUP_ENGINES)
    remedy = bridges.artifact_remedy(
        env_var=score_bridge.OBSSCORE_BRIDGE_ENV,
        filename=score_bridge.library_names()[0],
        subject="the observation scoring engine",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE)
    try:
        path = score_bridge.resolve_obsscore_bridge()
    except FileNotFoundError as error:
        return Check(name, "missing", str(error), remedy,
                     action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
                     brief="scores cannot be computed", group=_GROUP_ENGINES)
    reason = score_bridge.unavailable_reason()
    if reason is not None:
        return Check(name, "missing", f"{path}: {reason}", remedy,
                     action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
                     brief="scores cannot be computed", group=_GROUP_ENGINES)
    return Check(name, "verified", f"{path}: ABI {score_bridge.OBSSCORE_ABI}",
                 brief="staged", group=_GROUP_ENGINES)


def _globe_pack_check() -> Check:
    """``rw_globe_pack``, the global globe frame and statics packer (2.8.9).

    The global worker publishes globe frames and resolves verified baked
    statics through it, judged by its ABI marker. Non-blocking and degraded
    when absent: regional, hex and ensemble forecasts never call it, but a
    global run falls back to building its statics natively (about 2,000 s
    at T255) and cannot pack its globe frames.
    """

    name = "global globe packer (rw_globe_pack)"
    filename = bridges.executable_name("rw_globe_pack")
    remedy = bridges.artifact_remedy(
        env_var="WOOF_RW_GLOBE_PACK", filename=filename,
        subject="the global globe packer",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE,
        one_liner=bridges.rustwx_build_hint(), artifact="rw_globe_pack")
    degrades = ("global runs build their statics natively and cannot pack "
                "globe frames")
    # The packer is built in tools/rustwx (the remedy's own command), so a
    # checkout's tools/rustwx/target rung comes before the shared resolver,
    # whose checkout rung is tools/grib1_bridge.  Without it a fresh
    # `bash install.sh` built rw_globe_pack and doctor still reported it
    # missing, with a remedy that rebuilt the same file.
    try:
        path = None
        if not os.environ.get("WOOF_RW_GLOBE_PACK"):
            for built in (rustwx.crate_dir() / "target" / "release" / filename,
                          rustwx.crate_dir() / "target" / "debug" / filename):
                if built.is_file():
                    path = bridges.accept_resolved(built.resolve())
                    break
        if path is None:
            path = bridges.find_artifact("WOOF_RW_GLOBE_PACK", filename)
    except FileNotFoundError as error:
        path, reason = None, str(error)
    else:
        reason = None if path is not None else f"{filename} is not staged"
    if path is None:
        return Check(name, "missing",
                     f"{reason} -- {degrades}", remedy,
                     action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
                     brief="not staged; global statics build natively",
                     group=_GROUP_ENGINES, blocking=False,
                     severity=SEVERITY_DEGRADED)
    marker = bridges.BRIDGE_ABI_MARKERS["rw_globe_pack"]
    try:
        carries = marker in path.read_bytes()
    except OSError as error:
        carries, reason = False, f"unreadable ({error})"
    if not carries:
        return Check(name, "missing",
                     f"{path} -- {reason or 'does not carry the ' + marker.decode() + ' contract'} "
                     f"-- {degrades}", remedy,
                     action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
                     brief="stale build; global statics build natively",
                     group=_GROUP_ENGINES, blocking=False,
                     severity=SEVERITY_DEGRADED)
    return Check(name, "verified", f"{path} -- ABI {marker.decode()}",
                 brief="staged", group=_GROUP_ENGINES)


def _probgrid_check() -> Check:
    """``rw_probgrid``, the fixed-ladder probability map renderer (2.9.9).

    Draws a precomputed probability field (a post-processed severe
    probability) on a fixed ladder through the production panel path,
    judged by its ABI marker. Non-blocking and degraded when absent:
    forecasts and every other map are untouched; only those probability
    maps cannot be drawn.
    """

    name = "probability map renderer (rw_probgrid)"
    filename = bridges.executable_name("rw_probgrid")
    remedy = bridges.artifact_remedy(
        env_var="WOOF_RW_PROBGRID", filename=filename,
        subject="the probability map renderer",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE,
        one_liner=bridges.rustwx_build_hint(), artifact="rw_probgrid")
    degrades = "fixed-ladder probability maps are not drawn"
    # Built in tools/rustwx like rw_globe_pack, so a checkout's
    # tools/rustwx/target rung comes before the shared resolver.
    try:
        path = None
        if not os.environ.get("WOOF_RW_PROBGRID"):
            for built in (rustwx.crate_dir() / "target" / "release" / filename,
                          rustwx.crate_dir() / "target" / "debug" / filename):
                if built.is_file():
                    path = bridges.accept_resolved(built.resolve())
                    break
        if path is None:
            path = bridges.find_artifact("WOOF_RW_PROBGRID", filename)
    except FileNotFoundError as error:
        path, reason = None, str(error)
    else:
        reason = None if path is not None else f"{filename} is not staged"
    if path is None:
        if bridges.on_demand_fetch_offered("rw_probgrid"):
            return Check(
                name, "missing",
                "not staged yet -- it is fetched on demand: the platform wheel does not carry it; "
                "`woof fetch-bridges --partial --artifact rw_probgrid` stages it from this release's bundle",
                "woof fetch-bridges --partial --artifact rw_probgrid",
                action="woof fetch-bridges --partial --artifact rw_probgrid",
                brief="on demand; staged by woof fetch-bridges",
                group=_GROUP_ENGINES, blocking=False, severity=SEVERITY_OPT_IN)
        return Check(name, "missing",
                     f"{reason} -- {degrades}", remedy,
                     action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
                     brief="not staged; probability maps are not drawn",
                     group=_GROUP_ENGINES, blocking=False,
                     severity=SEVERITY_DEGRADED)
    marker = bridges.BRIDGE_ABI_MARKERS["rw_probgrid"]
    try:
        carries = marker in path.read_bytes()
    except OSError as error:
        carries, reason = False, f"unreadable ({error})"
    if not carries:
        return Check(name, "missing",
                     f"{path} -- {reason or 'does not carry the ' + marker.decode() + ' contract'} "
                     f"-- {degrades}", remedy,
                     action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
                     brief="stale build; probability maps are not drawn",
                     group=_GROUP_ENGINES, blocking=False,
                     severity=SEVERITY_DEGRADED)
    return Check(name, "verified", f"{path} -- ABI {marker.decode()}",
                 brief="staged", group=_GROUP_ENGINES)


def _isobaric_reader_check() -> Check:
    """The isobaric-height reader the Python height consumers call.

    ``missing``, because there is no Python implementation behind it: the
    vortex tracker on a host state, the GNSS-RO refractivity operator, the
    verification maps and the flagship products refuse their isobaric
    heights by name without it (a layer-mean height at the mass-level
    pressure, the only other reading, is 4 to 6 m high at 500 hPa).
    """

    name = "isobaric height reader (rw-isobaric)"
    degrades = ("the vortex tracker on a host state, the GNSS-RO operator, "
                "the verification maps and the flagship products refuse "
                "their isobaric heights")
    try:
        from woof import isobaric_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"woof.isobaric_bridge is not importable ({error}) "
                     f"-- {degrades}",
                     "# reinstall so the reader seam imports:\n"
                     "  pip install --force-reinstall woof",
                     brief="isobaric_bridge not importable",
                     group=_GROUP_ENGINES)
    remedy = bridges.artifact_remedy(
        env_var=isobaric_bridge.ISOBARIC_BRIDGE_ENV,
        filename=isobaric_bridge.library_names()[0],
        subject="the isobaric height reader",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE)
    return _library_check(
        name, isobaric_bridge, resolve="resolve_isobaric_bridge", abi="ISOBARIC_ABI",
        remedy=remedy, consequence=degrades, brief="isobaric height reads refuse")


def _tempo_library_check() -> Check | None:
    """``rw_tempo``, the TEMPO tempo_init library (2.9.0).

    ``woof.core.tempo_rust`` loads it through ctypes whenever a run selects
    TEMPO, and there is no Python implementation behind it.  Until 2.9.0 no
    wheel and no bundle carried it, so TEMPO -- and the rrfs-v2-hex-tempo
    preset -- stopped at "build it from a checkout" on every pip install.
    Judged by loading it and reading its ABI, the way the loader does.

    ``None`` where the seam itself is absent: the preparation-only RW-WPS
    package stages no forecast physics (tools/build_rw_wps_release.py), runs
    no TEMPO and bundles no TEMPO library, so a line there would report a
    missing file for a door that package does not have.
    """

    name = "TEMPO initialization library (rw-tempo)"
    degrades = ("a run selecting TEMPO microphysics, and the "
                "rrfs-v2-hex-tempo preset, cannot build TEMPO's tables")
    try:
        from woof.core import tempo_rust
    except ImportError:
        return None
    remedy = bridges.artifact_remedy(
        env_var=tempo_rust.TEMPO_BRIDGE_ENV,
        filename=tempo_rust.library_name(),
        subject="the TEMPO initialization library",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE)
    try:
        path = tempo_rust.resolve()
    except FileNotFoundError as error:
        return Check(
            name, "missing", f"{error} -- {degrades}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not staged; TEMPO cannot initialize",
            group=_GROUP_ENGINES)
    try:
        tempo_rust.load()
    except (OSError, RuntimeError, AttributeError) as error:
        return Check(
            name, "missing", f"{path} -- {error} -- {degrades}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not loadable; TEMPO cannot initialize",
            group=_GROUP_ENGINES)
    return Check(name, "verified",
                 f"{path} -- ABI {tempo_rust.ABI_VERSION}",
                 brief="staged", group=_GROUP_ENGINES)


#: The four entries woof hex resolves from the projected MET library.
_PROJECTED_INTERMEDIATE_ENTRIES = (
    "gpuwm_projected_intermediate_v1", "gpuwm_projected_geometry_v1",
    "gpuwm_projected_geometry_release_v1", "gpuwm_projected_window_v1")


def _projected_intermediate_check() -> Check:
    """``rw_projected_intermediate``, the projected MET library (2.9.1).

    ``woof hex intermediate`` loads it through ctypes for HRRR and every
    other projected source, and has no Python field route behind it.  On
    2.9.0 no wheel and no bundle carried it, so a pip install could start
    hex only from lat-lon sources.  Judged the way woof hex loads it: the
    file resolves, its ABI answers, and the four entries it binds exist.
    Not blocking: a WOOF run never calls it.
    """

    import ctypes

    from woof import bridge_assets

    name = "projected MET intermediate (rw_projected_intermediate)"
    artifact = next(a for a in bridge_assets.BUNDLED_ARTIFACTS
                    if a.name == "rw_projected_intermediate")
    filename = bridge_assets.artifact_filename(
        artifact, bridge_assets.host_platform() or "linux-x86_64")
    degrades = ("woof hex cannot start a hex point run from HRRR or any "
                "other projected source (lat-lon sources are unaffected)")
    remedy = bridges.artifact_remedy(
        env_var=artifact.env_var, filename=filename,
        subject="the projected MET intermediate library",
        crate_relative=artifact.crate, artifact=artifact.name)
    # woof hex's own ladder: the bridge ladder, then a checkout build in
    # the engine workspace (woof.hex.hrrr_intermediate._native_projected_entry).
    workspace = bridges._package_parent() / artifact.crate / "target"
    try:
        path = bridges.find_artifact(artifact.env_var, filename) or next(
            (candidate for candidate in (workspace / "release" / filename,
                                         workspace / "debug" / filename)
             if candidate.is_file()), None)
    except FileNotFoundError as error:
        path, reason = None, str(error)
    else:
        reason = f"{filename} is not staged"
    if path is None:
        return Check(name, "missing", f"{reason} -- {degrades}",
                     remedy, action=_build_action(artifact.crate),
                     brief="not staged; hex from HRRR refuses",
                     group=_GROUP_ENGINES, blocking=False,
                     severity=SEVERITY_DEGRADED)
    symbol, expected = bridge_assets.library_abi_for(artifact.name)
    try:
        library = ctypes.CDLL(str(path))
        probe = getattr(library, symbol)
        probe.restype = ctypes.c_uint32
        version = int(probe())
        absent = [entry for entry in _PROJECTED_INTERMEDIATE_ENTRIES
                  if not hasattr(library, entry)]
    except (OSError, AttributeError) as error:
        return Check(name, "missing", f"{path} -- not loadable ({error}) -- {degrades}",
                     remedy, action=_build_action(artifact.crate),
                     brief="not loadable; hex from HRRR refuses",
                     group=_GROUP_ENGINES, blocking=False,
                     severity=SEVERITY_DEGRADED)
    if version != expected or absent:
        detail = (f"ABI {version}, this release speaks {expected}" if version != expected
                  else f"lacks {', '.join(absent)}")
        return Check(name, "missing", f"{path} -- stale build: {detail} -- {degrades}",
                     remedy, action=_build_action(artifact.crate),
                     brief="stale build; hex from HRRR refuses",
                     group=_GROUP_ENGINES, blocking=False,
                     severity=SEVERITY_DEGRADED)
    return Check(name, "verified", f"{path} -- ABI {version}",
                 brief="staged", group=_GROUP_ENGINES)


def _nextgen_checks() -> list[Check]:
    """Probe the shipped hydro artifacts through their consumer ladders.

    This is a CPU ABI/entrypoint check, not a fabric-coverage or skill verdict.
    Inspection must not fetch, rebuild or refresh an estate to make it pass.
    """
    from woof import bridge_assets, hydrofabric
    from woof.hydro import forcing_inputs, verify_usgs, output

    probes = {
        "nextgen_hydrofabric": (hydrofabric._library,
                               ("arwen_hydrofabric_read_into",)),
        "arwen_nextgen_forcing": (forcing_inputs._library,
                                 ("arwen_forcing_prepare", "arwen_forcing_read_packet")),
        "rw_nextgen_verify": (verify_usgs._binary, ()),
        "rw_nextgen_output": (output._library, ("rw_nextgen_output_call", "rw_nextgen_output_free")),
    }
    checks = []
    with bridges.inspection_only():
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            if artifact.name not in probes:
                continue
            loader, entries = probes[artifact.name]
            filename = bridge_assets.artifact_filename(
                artifact, bridge_assets.host_platform() or "linux-x86_64")
            remedy = bridges.artifact_remedy(
                env_var=artifact.env_var, filename=filename,
                subject=f"NextGen {artifact.name}",
                crate_relative=artifact.crate, artifact=artifact.name)
            try:
                loaded = loader()
                if artifact.kind == "library":
                    path = loaded._name
                    symbol, expected = bridge_assets.library_abi_for(artifact.name)
                    version = int(getattr(loaded, symbol)())
                    if version != expected:
                        raise RuntimeError(f"ABI {version}; this release speaks {expected}")
                    for entry in entries:
                        getattr(loaded, entry)
                    detail = f"{path} -- ABI {version}, required entrypoints present"
                else:
                    path = Path(loaded)
                    result = subprocess.run([str(path), "--version"], capture_output=True,
                                            text=True, timeout=10, check=True)
                    marker = "nextgen-gauge-table-v1"
                    if marker not in result.stdout:
                        raise RuntimeError(f"{path}: --version lacks {marker}")
                    detail = f"{path} -- {result.stdout.strip()}"
            except (OSError, AttributeError, RuntimeError, ValueError,
                    subprocess.SubprocessError) as error:
                row = (_on_demand_row(f"NextGen {artifact.name}", artifact.name,
                                      ("woof go (coupled hydro)" if artifact.name == "rw_nextgen_output"
                                       else "woof hydro verify-usgs"), crate_relative=artifact.crate)
                       if artifact.on_demand and not os.environ.get(artifact.env_var) else None)
                if row is not None:
                    checks.append(row)
                    continue
                checks.append(Check(
                    f"NextGen {artifact.name}", "missing",
                    f"{error} -- this hydro consumer cannot run", remedy,
                    action=_build_action(artifact.crate), group=_GROUP_ENGINES,
                    severity=SEVERITY_DEGRADED, blocking=False))
            else:
                checks.append(Check(f"NextGen {artifact.name}", "verified", detail,
                                    brief="CPU ABI/entrypoints checked",
                                    group=_GROUP_ENGINES))
    return checks


#: The default-on hydro door's provider tools, bundled on demand by
#: e81d769e735 (2.9.3 improve-1).
_HYDRO_DOOR_TOOLS = ("rw_hydro_world", "rw_hydro_params", "rw_hydrofabric",
                     "nextgen-reservoir-prepare", "rw_hydro_init")


def _hydro_door_tool_checks() -> list[Check]:
    """The five tools the default-on hydro door runs to build a covering fabric.

    Breakage it prevents: e81d769e735 bundled these tools on demand so the
    default hydro door could find them (it had refused with "builder
    executable rw_hydro_world not found"), and no doctor line reported any of
    them, so an install whose hydro door could not prepare a fabric read
    green.  Each tool is resolved through the ladder the door uses
    (:func:`woof.hydro.door._executable`) but never fetched -- inspection
    reports, it does not stage -- and a present tool must answer a bare call
    with its own usage line, the entrypoint the door's probe calls.  Not a
    fabric-coverage verdict.
    """
    from woof import bridge_assets

    checks = []
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        if artifact.name not in _HYDRO_DOOR_TOOLS:
            continue
        label = f"hydro door {artifact.name}"
        filename = bridge_assets.artifact_filename(
            artifact, bridge_assets.host_platform() or "linux-x86_64")
        with bridges.inspection_only():
            candidates = bridges.artifact_candidates(
                artifact.env_var, filename, crate_relative=artifact.crate)
        path = next((candidate for candidate in candidates
                     if candidate.is_file() and os.access(candidate, os.X_OK)), None)
        if path is None:
            row = (_on_demand_row(label, artifact.name, "woof run (default-on hydro)",
                                  crate_relative=artifact.crate)
                   if artifact.on_demand and not os.environ.get(artifact.env_var) else None)
            if row is not None:
                checks.append(row)
                continue
            checks.append(Check(
                label, "missing",
                f"{filename} not found -- the default-on hydro door cannot prepare "
                "a fabric, so forecasts run without hydro and the receipt names the gap",
                bridges.artifact_remedy(
                    env_var=artifact.env_var, filename=filename, subject=label,
                    crate_relative=artifact.crate, artifact=artifact.name),
                action=_build_action(artifact.crate), group=_GROUP_ENGINES,
                severity=SEVERITY_DEGRADED, blocking=False))
            continue
        marker = f"usage: {artifact.name}"
        try:
            answer = subprocess.run([str(path)], capture_output=True, text=True, timeout=10)
            text = answer.stdout + answer.stderr
        except (OSError, subprocess.SubprocessError) as error:
            text = f"did not answer: {error}"
        if marker not in text:
            lines = text.strip().splitlines()
            checks.append(Check(
                label, "missing",
                f"{path} does not answer a bare call with '{marker}' "
                f"({lines[-1] if lines else 'no output'}) -- the hydro door cannot drive it",
                bridges.artifact_remedy(
                    env_var=artifact.env_var, filename=filename, subject=label,
                    crate_relative=artifact.crate, artifact=artifact.name),
                action=_build_action(artifact.crate), group=_GROUP_ENGINES,
                severity=SEVERITY_DEGRADED, blocking=False))
            continue
        checks.append(Check(label, "verified", f"{path} -- answers with its usage line",
                            brief="present; entrypoint answers", group=_GROUP_ENGINES))
    return checks


def _hydro_proj_checks() -> list[Check]:
    """Load each hydro bridge's PROJ and transform one EPSG point.

    A bare usage call never loaded the former dlopen dependency, so doctor
    reported green on hosts whose default-on hydro silently turned off.
    """
    from woof import bridge_assets
    import ctypes
    wanted = {"rw_hydro_world", "rw_hydrofabric", "rw_hydro_init", "nextgen_hydrofabric"}
    checks = []
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        if artifact.name not in wanted:
            continue
        label = f"hydro PROJ {artifact.name}"
        filename = bridge_assets.artifact_filename(artifact, bridge_assets.host_platform() or "linux-x86_64")
        with bridges.inspection_only():
            candidates = bridges.artifact_candidates(artifact.env_var, filename, crate_relative=artifact.crate)
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            # An on-demand hydro bridge the platform wheel leaves to the bundle: its first use stages it, and
            # the projection is proved then.  Breakage it prevents (2.9.7 pass 5 R12, Windows wheel smoke): a
            # fresh install showed "MISSING rust engines (3)" for the three hydro executables that hydro, opt-in
            # since 2.9.4.1, had never needed to stage.
            row = _on_demand_row(label, artifact.name, "woof run --hydro on", crate_relative=artifact.crate)
            if row is not None:
                checks.append(row)
                continue
        try:
            if path is None:
                raise FileNotFoundError(f"{filename} is not staged")
            if artifact.name == "nextgen_hydrofabric":
                with bridges.quiet_loader_errors():
                    library = ctypes.CDLL(str(path))
                probe = library.arwen_hydrofabric_proj_probe
                probe.argtypes, probe.restype = [], ctypes.c_int
                if probe() != 0:
                    raise ValueError("the loaded library's EPSG:4326 -> EPSG:5070 transform failed")
            else:
                answer = subprocess.run([str(path), "--proj-probe"], capture_output=True, text=True, timeout=30)
                if answer.returncode:
                    raise ValueError((answer.stderr or answer.stdout).strip()[-1000:])
                result = json.loads(answer.stdout)
                if (result.get("proj_version") != "9.7.1" or result.get("network") is not False
                        or result.get("resources") != "embedded-only"):
                    raise ValueError(f"unexpected projection probe: {result}")
            checks.append(Check(label, "verified", f"{path} -- transforms (-97, 35) from EPSG:4326 to EPSG:5070; "
                                "static PROJ 9.7.1, embedded database, network off",
                                brief="loaded; one point transformed", group=_GROUP_ENGINES))
        except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as error:
            checks.append(Check(label, "missing", f"{error} -- this bridge cannot prove its hydro projection",
                                bridges.artifact_remedy(env_var=artifact.env_var, filename=filename,
                                    subject=label, crate_relative=artifact.crate, artifact=artifact.name),
                                action=_build_action(artifact.crate), group=_GROUP_ENGINES,
                                severity=SEVERITY_DEGRADED, blocking=False))
    return checks


def _radar_superob_check() -> Check:
    """The radar superob the default radar observation build calls.

    ``missing``, because the stage refuses without it rather than falling
    back: the numpy module runs only on the stated opt-out
    ``WOOF_SUPEROB_PYTHON=1``, so an absent library stops every radar
    observation build (and with it every radar DA cycle) by name.
    """

    name = "radar superob (rw-superob)"
    degrades = ("every radar observation build refuses its superob stage, "
                "so radar DA cycles stop at observation prep")
    try:
        from woof.obs import superob_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"woof.obs.superob_bridge is not importable ({error}) "
                     f"-- {degrades}",
                     "# reinstall so the superob seam imports:\n"
                     "  pip install --force-reinstall woof",
                     brief="superob_bridge not importable",
                     group=_GROUP_ENGINES)
    remedy = bridges.artifact_remedy(
        env_var=superob_bridge.SUPEROB_BRIDGE_ENV,
        filename=superob_bridge.library_name(),
        subject="the radar superob",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE) + (
        "\n  # WOOF_SUPEROB_PYTHON=1 runs the numpy reference as an\n"
        "  # explicit workaround")
    return _library_check(
        name, superob_bridge, resolve="resolve_superob_bridge", abi="SUPEROB_ABI",
        remedy=remedy, consequence=degrades, brief="radar observation builds refuse",
        missing_errors=(superob_bridge.SuperobBridgeMissing, FileNotFoundError))


def _region_dealias_check() -> Check:
    """The region-global dealiasing engine, which the default now needs.

    ``missing``, and the change of status from ``info`` is the finding:
    ``region-global`` became the shipped ``--dealias-engine`` on
    2026-08-12, so an absent library no longer costs an OPTION -- it
    costs every dealiased velocity in the DA nowcast, on the path a run
    reaches an hour in.  A default that needs a build is a default whose
    build blocks, the same rule that puts ``rw_nexrad`` here while the
    renderer and the fetch backbone stay informational.

    What it does NOT do is send the operator to a workaround: the remedy
    is the build, and ``--dealias-engine vad-region`` is named as what it
    is, a different solver with different decisions, not as a way around
    a missing file.
    """

    name = "region-global dealiasing engine (default --dealias-engine)"
    blocks = ("blocks the default velocity dealiasing path -- every "
              "--dealias run in the DA nowcast, live and archived")
    try:
        from woof.obs import dealias_region
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"woof.obs is not importable ({error}) -- {blocks}",
                     "# reinstall so the observation stack imports:\n"
                     f"{REINSTALL_HINT}",
                     action="reinstall woof",
                     brief="obs stack not importable",
                     group=_GROUP_ENGINES)
    try:
        found = dealias_region.find_region_bridge()
    except FileNotFoundError as error:
        return Check(
            name, "missing", str(error),
            f"# {dealias_region.REGION_DEALIAS_ENV} names a missing "
            "library: point it at a real build, or unset it --\n"
            + dealias_region.region_bridge_remedy(),
            action=(f"unset {dealias_region.REGION_DEALIAS_ENV}, or point it "
                    "at a real build"),
            brief=f"{dealias_region.REGION_DEALIAS_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        return Check(
            name, "missing", f"not staged or built -- {blocks}",
            dealias_region.region_bridge_remedy()
            + "\n  # or `woof fetch-bridges`, which stages it with the\n"
            "  # other prebuilt artifacts.  --dealias-engine vad-region\n"
            "  # runs without it, but it is a DIFFERENT solver: it\n"
            "  # abstains where this one resolves, and the engine that\n"
            "  # ran is recorded with the velocities it made",
            action=_build_action(dealias_region.CRATE_RELATIVE),
            brief="not staged; default dealiasing blocked",
            group=_GROUP_ENGINES)
    try:
        engine = dealias_region.load_region_dealiaser(found)
    except Exception as error:                   # noqa: BLE001 - reported
        return Check(
            name, "missing", f"{found} -- {error}",
            "# REBUILD it: this is a contract mismatch, so re-pointing\n"
            f"  # {dealias_region.REGION_DEALIAS_ENV} at another copy of the\n"
            "  # same vintage fails identically --\n"
            + dealias_region.region_bridge_remedy(),
            action=_build_action(dealias_region.CRATE_RELATIVE),
            brief=_short(str(error)), group=_GROUP_ENGINES)
    evidence = (f"ABI {engine.abi_version}, refinement API "
                f"{engine.rift_api_version}, upstream "
                f"{dealias_region.UPSTREAM_COMMIT[:12]}")
    return Check(name, "verified", f"{found} -- {evidence}",
                 brief=_short(evidence), group=_GROUP_ENGINES)


def _matplotlib_engine_note() -> str:
    """What ``woof render --engine matplotlib`` would do, on THIS install.

    This function exists because the sentence it replaces was false.
    Doctor said "matplotlib remains the documented fallback" in three
    places, and the matplotlib engine imports the ``wrf`` package at the
    top of its render loop -- every product it draws is a ``wrf.getvar``
    call -- so on an install without ``[render]`` there was no fallback
    at all.  The sentence has to be a question, asked here.

    It is no longer a FALLBACK in either arm.  ``--engine auto``
    refuses when ``rw_wrfbatch`` is unusable rather than degrading,
    because the project render law ( 2026-08-06) reserves
    weather-field product plots for that binary and names exactly one
    permitted fallback, which is not this engine (audit F7).  So the
    accurate doctor line is about a WORKAROUND a reader can type, never
    about a safety net that catches them.
    """

    ok, _evidence = _import_probe("wrf", "wrf-rust")
    if ok:
        return ("`woof render --engine matplotlib` is available here as "
                "a NAMED WORKAROUND (the wrf package imports); the "
                "default --engine auto refuses rather than degrading to "
                "it, because weather fields come from rw_wrfbatch")
    if not science_core.python_supports_science_core(_python_version()):
        # Not "not installed": not INSTALLABLE.  Telling a reader over
        # the ceiling to run pip for a package with no wheel for their
        # interpreter is the same dead loop the extras line used to
        # print.  The sentence is derived, so the day upstream publishes
        # the wheels this branch stops firing on its own -- which is
        # what 0.2.39 did for 3.14.
        return ("`woof render --engine matplotlib` cannot run on this "
                "interpreter, and pip cannot supply what it needs: "
                + science_core.python_gap_sentence(_python_version())
                + ".  The DEFAULT rust engine is unaffected -- it draws "
                  "from the rw_wrfbatch binary and imports no Python "
                  "science core at all")
    return ("woof render has NO engine left here: the matplotlib "
            "workaround imports the wrf package from the [render] "
            "extra, which is not installed (see the `pip extra "
            "[render]` line)")


def _rust_renderer_check() -> Check:
    """The vendored Rusty Weather renderer: probe-execute, not stat().

    ``woof render`` defaults to this engine exactly when the check
    passes.  Whether its absence is a gap depends on something this
    check cannot see on its own -- the matplotlib engine needs the
    ``[render]`` extra too -- so the status stays ``info``/``missing``
    and non-blocking here, and the sentence about the fallback is
    computed (:func:`_matplotlib_engine_note`) rather than asserted.
    The blocking verdict on an absent ``[render]`` belongs to the extras
    block, which owns that package; two lines blocking on one fact would
    count one problem twice.
    """

    name = f"renderer {rustwx.RENDERER_NAME} (rust render engine)"
    try:
        found = rustwx.find_renderer()
    except FileNotFoundError as error:
        return Check(
            name, "missing", str(error),
            f"# {rustwx.RENDERER_ENV} names a missing executable: "
            "point it at a real build, or unset it and build one --\n"
            + rustwx.renderer_remedy(),
            action=f"unset {rustwx.RENDERER_ENV}, or point it at a real "
                   "build",
            brief=f"{rustwx.RENDERER_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        crate = rustwx.crate_dir() / "Cargo.toml"
        if crate.is_file():
            detail = f"not built yet (checkout crate: {crate.parent})"
        else:
            detail = ("no checkout crate and no prebuilt executable "
                      f"(searched {', '.join(str(c) for c in rustwx.renderer_candidates())})")
        note = _matplotlib_engine_note()
        return Check(
            name, "info", f"{detail} -- {note}",
            bridges.install_aware_build_hint(
                rustwx.CARGO_BUILD_HINT, "tools/rustwx")
            + "\n  # enables --engine rust and makes it the default",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(f"not built; {note}"),
            group=_GROUP_ENGINES)
    ok, evidence = rustwx.probe_renderer(found)
    if not ok:
        # Non-blocking here on purpose, and NOT because "matplotlib
        # remains the fallback" -- that sentence was false on a base
        # install.  The extras block owns the [render] verdict; this
        # line says what is true of the binary and names the state of
        # the other engine.
        note = _matplotlib_engine_note()
        return Check(
            name, "missing", f"{found} -- {evidence}; {note}",
            "# it has to be replaced:\n" + bridges.install_aware_build_hint(
                rustwx.CARGO_BUILD_HINT, "tools/rustwx"),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(evidence), group=_GROUP_ENGINES,
            blocking=False, severity=SEVERITY_DEGRADED)
    # Ask the question the renderer answers, in the renderer's own
    # order.  Probing woof's checkout path alone reported "NO basemap
    # assets found" on every pip install -- including the ones where
    # rw_wrfbatch resolves the assets from its own build directory and
    # draws the coastlines the report says are missing.
    basemap = rustwx.resolve_basemap_dir(found)
    if basemap is None:
        # A gap with its fix, not a verified line with a footnote: a
        # renderer with no map assets draws every picture with no
        # coastlines, borders or state lines and exits 0, so this line
        # is the one place before a run that says so.  Non-blocking:
        # the fields still draw.
        remedy = rustwx.basemap_remedy()
        return Check(
            name, "missing",
            f"{found} -- {evidence}; NO basemap assets resolve, so every "
            "picture is drawn with no coastlines, borders or state lines",
            f"{remedy}\n  # the recast-woof-data package carries the map assets",
            action=remedy, brief="built; no basemap assets",
            group=_GROUP_ENGINES, blocking=False,
            severity=SEVERITY_DEGRADED)
    if os.environ.get("RUSTWX_BASEMAP_DIR") or os.environ.get(
            "RUSTWX_ASSETS_DIR"):
        basemap_note = f"basemaps {basemap} (RUSTWX_* environment override)"
    else:
        basemap_note = f"basemaps {basemap}"
    return Check(name, "verified", f"{found} -- {evidence}; {basemap_note}",
                 brief="built, with basemap assets",
                 group=_GROUP_ENGINES)


def _provenance_check() -> Check:
    """Which tree is executing, and do its version claims agree?

    The estate report's own subject.  Every other check here asks
    whether some artifact beside woof is usable; this one asks whether
    the woof doing the asking can name itself -- which is the question
    that went unasked while a checkout sat 2,370 commits behind with an
    editable install reporting a version from a different tree.

    ``missing`` and blocking only for a genuine contradiction, which is
    :func:`woof.provenance_gate.version_identity_refusal`'s definition
    and not "anything unusual": a wheel with no git, a clone nobody
    installed, and a checkout with a borrowed-but-agreeing number are
    all reported ``verified``, because none of them is wrong about
    anything.
    """

    from woof.explain import split
    from woof.provenance import resolve
    from woof.provenance_gate import (executing_version,
                                       version_identity_refusal)

    name = "install provenance (which tree is executing)"
    try:
        prov = resolve()
        refusal = version_identity_refusal(prov)
    except Exception as error:                          # noqa: BLE001
        return Check(name, "missing",
                     f"provenance could not be resolved: {error}",
                     REINSTALL_HINT, action="pip install -e .",
                     brief=_short(str(error)))
    if refusal is not None:
        action, _ = split(refusal)
        return Check(
            name, "missing", action,
            f"# the two claims have to be re-bound:\n"
            f"  pip install -e {prov.source_root}\n"
            "  # (`woof version` prints the same finding with the\n"
            "  #  upgrade path for a wheel install)",
            action=f"pip install -e {prov.source_root}",
            brief="version claims disagree")
    detail = f"{prov.install_kind} {executing_version()} at {prov.source_root}"
    git = prov.git or {}
    if git.get("commit"):
        detail += (f", git {git['commit']} on "
                   f"{git.get('branch') or 'a detached HEAD'}"
                   f" ({'dirty' if git.get('dirty') else 'clean'})")
    if prov.metadata_is_borrowed:
        # Not a refusal -- the digits agree -- but a reader comparing
        # receipts has to know the number came from elsewhere.
        detail += ("; NOTE its version string is read from another "
                   "install's metadata (no distribution provides this "
                   "code), and happens to agree")
    return Check(name, "verified", detail, brief=_short(detail))


def _renderer_tree_check() -> Check:
    """Does the resolved render engine belong to THIS tree?

    Separate from :func:`_rust_renderer_check`, which asks whether a
    renderer exists and runs.  Existence was the whole gate until now,
    and existence is what let a checkout borrow the engine some other
    tree had staged in the shared bridge directory -- a substitution
    with no message anywhere and a different product catalog at the far
    end of it.

    Non-blocking, deliberately.  ``woof render --engine rust``
    REFUSES a foreign engine and ``--engine auto`` degrades to
    matplotlib, so the estate is not broken; this line is how a reader
    finds out before they run a render rather than after.
    """

    from woof.provenance import resolve
    from woof.provenance_gate import bridge_tree_match

    # A wheel install is not a checkout, and its row said "vs this
    # checkout" beside a verdict about the installed wheel.
    prov = resolve()
    against = ("this wheel install" if prov.install_kind == "wheel"
               else "this checkout")
    name = f"renderer tree match ({rustwx.RENDERER_NAME} vs {against})"
    try:
        found = rustwx.find_renderer()
    except FileNotFoundError as error:
        # _rust_renderer_check already reports this one in full.
        return Check(name, "info", str(error), brief=_short(str(error)),
                     group=_GROUP_ENGINES)
    match = bridge_tree_match(found, env_var=rustwx.RENDERER_ENV, prov=prov)
    if match.verdict == "absent":
        return Check(name, "info",
                     "no renderer resolved, so there is nothing to match",
                     brief="no renderer resolved", group=_GROUP_ENGINES)
    if match.matched:
        return Check(name, "verified", f"{match.verdict}: {match.basis}",
                     brief=match.verdict, group=_GROUP_ENGINES)
    return Check(
        name, "missing",
        f"{match.bridge} is from another tree -- {match.basis}, while "
        f"this checkout is at {(match.engine_commit or '')[:12]}",
        # Every non-comment line here must be a pasteable command --
        # `tests/test_doctor.py` enforces it, and a bare `VAR=value` is
        # neither a command nor portable between PowerShell and sh.  So
        # the declaration route is printed the way `bridges
        # .artifact_remedy` prints its own env-var route: commented.
        "# build the engine this tree matches:\n"
        + bridges.install_aware_build_hint(
            rustwx.CARGO_BUILD_HINT, "tools/rustwx")
        + "\n  # ... or, if that binary IS the one you mean, declare it:\n"
        f"  #   {rustwx.RENDERER_ENV}={match.bridge}",
        action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
        brief="renderer built from another commit",
        group=_GROUP_ENGINES, blocking=False)


def _cpu_library_check() -> Check:
    from woof.ingest.cpu_backend import (
        CPU_BACKEND_ABI, CpuPreprocessBackend)

    # The tail is a `#` block, not prose fused onto the build command.
    # It used to read `... --offline  then copy it into <dir> or set
    # WOOF_CPU_PREPROCESS_BRIDGE`, which a reader pastes whole and the
    # shell hands to cargo as arguments -- the exact failure the remedy
    # contract exists to stop, and a test host G field finding.
    remedy = (bridges.install_aware_build_hint(bridges.CARGO_BUILD_HINT)
              + "\n  # then copy the built library into "
              f"{bridges.default_bridge_dir()},\n"
              "  # or set WOOF_CPU_PREPROCESS_BRIDGE to its full path")
    try:
        backend = CpuPreprocessBackend()
    except FileNotFoundError as error:
        return Check(
            "cpu preprocess library", "missing",
            "gpuwm_preprocess_cpu shared library not found "
            "(--preprocess-backend cpu needs it, and so does "
            "--preprocess-backend auto wherever CUDA is unusable; every "
            "preparation that maps soil, snow, skin temperature or sea ice "
            "needs it under either backend; the "
            "radar dealiaser's coarse VAD search falls back to NumPy "
            f"without it): {error}", remedy,
            action=_build_action(), brief="not staged",
            group=_GROUP_BRIDGES, blocking=False,
            severity=SEVERITY_DEGRADED)
    except (OSError, RuntimeError, AttributeError) as error:
        return Check(
            "cpu preprocess library", "missing",
            f"found but not loadable as ABI v{CPU_BACKEND_ABI}: {error}",
            "# it has to be replaced:\n" + remedy,
            action=_build_action(),
            brief=f"not loadable as ABI v{CPU_BACKEND_ABI}",
            group=_GROUP_BRIDGES)
    path, abi = backend.path, backend.abi_version
    indexed = backend.indexed_donor_interp
    from woof.ingest.cpu_backend import (
        MASKED_NEAREST_ENTRY, MASKED_STENCIL_ENTRY, WPS_MASKED_CHAIN_ENTRY,
        WPS_MASKED_REACH_ENTRY)

    lacking = [name for name, present in (
        (WPS_MASKED_CHAIN_ENTRY, backend.wps_masked_chain_entry),
        (WPS_MASKED_REACH_ENTRY, backend.wps_masked_chain_entry),
        (MASKED_STENCIL_ENTRY, backend.masked_stencil_entry),
    ) if not present]
    lacking.extend(getattr(backend, "water_blend_missing", ()))
    if not getattr(backend, "masked_nearest_entry", False):
        lacking.append(MASKED_NEAREST_ENTRY)
    backend.close()
    if lacking:
        # Not a note on the line: the masked surface fields have no other
        # route under either backend, so a library without these entries
        # cannot prepare any source with a land-sea mask.
        return Check(
            "cpu preprocess library", "missing",
            f"{path} loaded via ctypes, ABI v{abi}, but without "
            f"{' and '.join(lacking)}: the masked surface fields (soil "
            "moisture and temperature, snow, skin temperature, sea ice) "
            "map only through the library's masked chain and soil "
            "stencil, and lake and sea water temperatures only through "
            "its water blends, under both preprocessing backends, so no "
            "source with a land-sea mask can be prepared",
            "# it has to be rebuilt or re-fetched:\n" + remedy,
            action=_build_action(),
            brief="predates the masked surface chain",
            group=_GROUP_BRIDGES)
    # The same library carries the dealiaser's coarse VAD search.  A
    # library built before that entry point existed still serves every
    # interpolation call, so this is a note on the line and not a second
    # verdict -- but it is the difference between a radar volume
    # dealiased in seconds and one dealiased in half a minute, and a
    # user who cannot see which one they have cannot ask why.  It is a
    # note on BOTH verdicts below: the indexed-donor entry point and the
    # VAD search are separate additions to the same library, and a
    # reader told about one of them still has to be told about the other.
    from woof.obs.coarse_cost import unavailable_reason

    reason = unavailable_reason()
    search = ("radar coarse VAD search: native"
              if reason is None else
              f"radar coarse VAD search: NumPy ({reason})")
    if not indexed:
        # A staged library can be older than the checkout driving it, and
        # the ABI integer cannot say so: it describes the calls that
        # already existed, and this one is an addition.  Absence is not a
        # fault -- the projected route keeps its NumPy mirror and warns
        # at the first plan -- but it is a whole preparation stage's
        # worth of wall clock, so the estate report says it rather than
        # calling the library simply "verified".
        return Check(
            "cpu preprocess library", "info",
            f"{path} loaded via ctypes, ABI v{abi}, but without "
            "gpuwm_indexed_interp_f32: projected-source horizontal "
            "mapping falls back to the single-core NumPy mirror, which "
            f"dominates nested preparation; {search}",
            "# it has to be rebuilt from this checkout:\n" + remedy,
            action=_build_action(),
            brief="ABI v{0}, projected mapping on the NumPy mirror".format(
                abi),
            group=_GROUP_BRIDGES)
    return Check("cpu preprocess library", "verified",
                 f"{path} loaded via ctypes, ABI v{abi}, "
                 f"indexed-donor horizontal entry present; {search}",
                 brief=f"ABI v{abi}", group=_GROUP_BRIDGES)


# ---------------------------------------------------------------------------
# Packaged tables: the model's own validators, not directory counts
# ---------------------------------------------------------------------------

def _companion_pair_check() -> Check:
    """Are ``woof`` and its ``recast-woof-data`` companion the same cut?

    The breakage this prevents, measured on the published 2.6.1 wheel
    with recast-woof-data downgraded to 2.6.0 beside it.  Doctor printed
    ``ok  base dependencies ... recast-woof-data==2.6.1`` -- a green line
    asserting the ``==`` pin was met over an install where it was not --
    and said nothing else about the pair anywhere in the report.  Then,
    on any box whose Thompson tables are not already staged under
    ``~/.woof``, the run died in a nine-frame ``ImportError`` raised
    out of :func:`_thompson_tables_check`, because that check catches
    file errors and the companion lock refuses with ``ImportError``.
    Doctor's contract is a row and a remedy, never a traceback, so the
    skew gets a row of its own and the pin's own line points here.

    Metadata, not presence: a companion cut for another release has
    every file and opens cleanly.  It is a different numerical setup
    wearing the right filenames -- 2.6.1's radiation and microphysics
    reading 2.6.0's k-distribution and lookup tables -- and it produces
    numbers rather than an error, which is why nothing downstream can
    see it and why it belongs in this report.
    """

    from woof import data_assets

    required = data_assets._required_companion_version()
    distribution = data_assets.COMPANION_DISTRIBUTION
    if required == data_assets._UNKNOWN_VERSION:
        # woof itself is a source tree no distribution provides.  There
        # is no version to hold the companion to, so there is no skew to
        # find and a gap here would name no breakage.
        return Check(
            distribution, "info",
            f"woof runs from a source tree no distribution provides, "
            f"so there is no version to hold {distribution} to",
            brief="source tree; no version to compare")
    try:
        found: str | None = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        found = None
    if found is None:
        try:
            importlib.import_module(data_assets.COMPANION_PACKAGE)
        except ImportError:
            return Check(
                distribution, "missing",
                f"woof {required} pins {distribution}=={required} and no "
                f"{distribution} distribution is installed; every packaged "
                f"reference table (the RRTMGP k-distributions, the "
                f"Thompson lookup tables) resolves through it",
                data_assets.companion_install_command(),
                action=data_assets.companion_install_command(),
                brief="not installed", severity="broken")
        # Importable but not installed: a source checkout on sys.path.
        # data_assets deliberately does not compare in that shape, so
        # neither does this row -- `verified` is not a word for a
        # question nobody asked.
        return Check(
            distribution, "untested",
            f"not tested: {data_assets.COMPANION_PACKAGE} imports from a "
            f"source checkout that no distribution provides, so there is "
            f"no companion version to compare against woof {required}",
            brief="source checkout; version not comparable")
    if data_assets._public_version(found) == required:
        return Check(
            distribution, "verified",
            f"woof {required} and {distribution} {found} are the same "
            f"cut, so every packaged table this install reads is the one "
            f"its physics was built against",
            brief=f"{distribution} {found} matches")
    return Check(
        distribution, "missing",
        f"woof {required} pins {distribution}=={required}, found "
        f"{found}.  The two are cut from one commit, so this install was "
        f"edited.  Running {required}'s radiation and microphysics "
        f"against {found}'s k-distribution and lookup tables is a "
        f"different numerical setup that produces numbers instead of an "
        f"error, and no certification capsule can see it",
        data_assets.companion_install_command(),
        action=data_assets.companion_install_command(),
        brief=f"{distribution} is {found}, needs {required}",
        severity="broken")


def _thompson_tables_check() -> Check:
    from woof.core.thompson_contract import validate_table_assets
    from woof.physics_compat import thompson_table_root
    from woof.table_assets import (
        classify_assets, missing_externalized_assets)

    try:
        root = thompson_table_root()
    except ImportError as error:
        # The companion refused: skewed, or absent with no staged root
        # to fall to.  Resolving the root sits OUTSIDE the try below on
        # purpose -- that one catches the file errors a validator raises
        # -- and an ImportError crossing this frame is what turned
        # `woof doctor` into a traceback.  :func:`_companion_pair_check`
        # above owns the verdict and the remedy; this row says only that
        # it could not run, because two lines publishing one remedy is
        # how a reader learns to read neither.
        return Check(
            "thompson tables", "untested",
            f"not tested: the packaged table root could not be resolved "
            f"-- {error}",
            brief=f"companion refused; see the "
                  f"{_companion_distribution_name()} line",
            blocking=False, severity="degraded")
    try:
        assets = validate_table_assets(root)
    except (FileNotFoundError, ValueError, OSError) as error:
        # The externalized assets (woof.table_assets: the two largest
        # Thompson tables) are published as release assets rather than
        # shipped in the wheel; their absence has a one-command fix
        # that is not "reinstall".
        fetchable = missing_externalized_assets(Path(root))
        _valid, invalid, absent = classify_assets(Path(root))
        if fetchable and not invalid and len(absent) == len(fetchable):
            total_mib = sum(a.bytes for a in fetchable) / (1024 * 1024)
            # Without a named root the first mp_physics=8 run stages the
            # set itself (table_assets.require_thompson_tables), so a
            # fresh install is not broken: the default works, it just
            # has not been paid for yet.  A named root is the operator's
            # mirror, never filled for them, and stays a blocking gap.
            from woof.table_assets import auto_stage_applies
            automatic = auto_stage_applies()
            return Check(
                "thompson tables", "missing",
                f"{', '.join(a.filename for a in fetchable)} not staged "
                f"at {root} (externalized: published as a release asset, "
                "not shipped in the package)"
                + ("; the first mp_physics=8 run stages them "
                   "automatically" if automatic else ""),
                "woof fetch-tables\n"
                f"  # one {total_mib:.0f} MiB download, SHA-256 verified "
                "against the\n"
                "  # packaged pins before install; --from stages it from a\n"
                "  # local directory instead, offline",
                action="woof fetch-tables",
                brief=f"{len(fetchable)} externalized table(s) not staged "
                      f"({total_mib:.0f} MiB)",
                blocking=not automatic,
                severity="degraded" if automatic else None)
        return Check(
            "thompson tables", "missing", str(error),
            REINSTALL_HINT
            + "\n  # if WOOF_THOMPSON_TABLE_ROOT is set, point it at a"
            "\n  # byte-identical mirror of the packaged tables, or unset it",
            action="pip install -e .", brief=_short(str(error)))
    # The mp=28 activation table is read out of the same root, and was
    # never looked at here: an empty CCN_ACTIVATE.BIN reported "verified"
    # and failed the first aerosol-aware forecast at load.  Located the
    # way the mp=28 loader locates it, so its file override is honoured.
    from woof.core.thompson_aerosol_contract import (
        AEROSOL_TABLE_ASSETS, AEROSOL_TABLE_PATH_ENV,
        resolve_ccn_activation_path, validate_ccn_activation_asset)
    try:
        aerosol = (validate_ccn_activation_asset(
            resolve_ccn_activation_path(None, root)),)
    except FileNotFoundError:
        names = ", ".join(a.filename for a in AEROSOL_TABLE_ASSETS)
        if os.environ.get(AEROSOL_TABLE_PATH_ENV):
            return Check(
                "thompson tables", "missing",
                f"{AEROSOL_TABLE_PATH_ENV} names "
                f"{os.environ[AEROSOL_TABLE_PATH_ENV]}, which is not a file",
                f"  # point {AEROSOL_TABLE_PATH_ENV} at a verified "
                f"{names}, or unset it",
                brief=f"{AEROSOL_TABLE_PATH_ENV} names no file",
                blocking=False, severity="degraded")
        return Check(
            "thompson tables", "missing",
            f"{len(assets)} classic assets at {root} byte-validated, but "
            f"{names} (the aerosol-aware Thompson activation table, "
            "mp_physics=28) is not in that root",
            "woof fetch-tables\n"
            "  # copies the verified packaged table into the root",
            action="woof fetch-tables",
            brief=f"{names} not staged (mp_physics=28 only)",
            blocking=False, severity="degraded")
    except (ValueError, OSError) as error:
        # Degraded, not blocking, exactly like the file being absent:
        # only mp_physics=28 reads it, and the classic tables every other
        # Thompson run needs were verified above.  A damaged copy used to
        # fail the whole check for mp_physics=8 users who never open it.
        names = ", ".join(a.filename for a in AEROSOL_TABLE_ASSETS)
        return Check(
            "thompson tables", "missing",
            f"{len(assets)} classic assets at {root} byte-validated, but "
            f"{names} (the aerosol-aware Thompson activation table, "
            f"mp_physics=28) is damaged: {error}",
            "woof fetch-tables\n"
            "  # replaces a damaged copy in ~/.woof with the verified\n"
            "  # packaged table; in a root named by\n"
            "  # WOOF_THOMPSON_TABLE_ROOT, delete the file first",
            action="woof fetch-tables",
            brief=f"{names} damaged (mp_physics=28 only)",
            blocking=False, severity="degraded")
    assets = (*assets, *aerosol)
    return Check(
        "thompson tables", "verified",
        f"{len(assets)} assets at {root} byte-validated (exact size + "
        f"SHA-256, {sum(asset.bytes for asset in assets):,} B), the same "
        "validation every mp8 and mp28 run performs at load",
        brief=f"{len(assets)} assets byte-validated")


def _p3_table_check() -> Check:
    """Can an mp_physics=50 run load its ice lookup table, right now?

    Same shape as the Thompson check above and for the same reason: it
    runs the model's OWN validator, so a green line here means the
    validation every mp=50 run performs at load has already passed on
    this install -- not merely that a file of the right name exists.

    Not externalized: at 1.6 MiB the table ships inside the wheel, so its
    absence means a broken install (or a bad WOOF_P3_TABLE_ROOT), never
    a pending download.  There is no `fetch-tables` arm here on purpose.

    Why it needs its own line: P3's process rates ARE this table, the
    loader refuses without it, and until this check existed the only
    place that refusal could surface was the first microphysics call of a
    forecast -- after fetch and preprocessing had been paid for.
    """
    from woof.core.p3_tables import (
        P3_TABLE_ROOT_ENV,
        P3TableVersionError,
        TABLE_1_2MOM_ASSET,
        TABLE_1_2MOM_VERSION,
        UPSTREAM_WRF_TABLE_1_2MOM_SHA256,
        _validate_asset_bytes,
        p3_table_root,
    )

    root = p3_table_root()
    path = Path(root) / TABLE_1_2MOM_ASSET.filename
    try:
        _validate_asset_bytes(path, TABLE_1_2MOM_ASSET)
    except P3TableVersionError as error:
        # A DIFFERENT table, not a damaged one: the fix is to stage the
        # right version, and "reinstall" alone would be wrong advice.
        return Check(
            "p3 table", "missing", str(error),
            f"# stage p3_lookupTable_1.dat-v{TABLE_1_2MOM_VERSION}, or "
            f"unset {P3_TABLE_ROOT_ENV}" + "\n" + REINSTALL_HINT,
            action="pip install -e .",
            brief=f"table is not v{TABLE_1_2MOM_VERSION}")
    except (FileNotFoundError, ValueError, OSError) as error:
        hint = (
            REINSTALL_HINT
            + "\n  # if " + P3_TABLE_ROOT_ENV + " is set, point it at a"
            + "\n  # byte-identical mirror of the packaged table, or unset it"
            + "\n  # note: WRF's own run/ copy is the SAME table with LF line"
            + "\n  # endings (sha-256 "
            + UPSTREAM_WRF_TABLE_1_2MOM_SHA256[:16] + "...), which does not"
            + "\n  # match this pin; see woof/data/p3/PROVENANCE.md")
        return Check(
            "p3 table", "missing", str(error), hint,
            action="pip install -e .", brief=_short(str(error)))
    return Check(
        "p3 table", "verified",
        f"p3_lookupTable_1.dat-v{TABLE_1_2MOM_VERSION} at {root} "
        "byte-validated (header version, exact size and SHA-256, "
        f"{TABLE_1_2MOM_ASSET.size:,} B), the same validation every mp50 "
        "run performs at load",
        brief=f"v{TABLE_1_2MOM_VERSION} byte-validated")


def _arbitrary_input_check() -> Check:
    """Can this install ingest a data source nobody blessed in advance?

    The whole reason this check exists: `woof adapt` was a working
    arbitrary-input front door that no user could find and, on a wheel
    install, could not run.  A feature nobody can reach is not shipped, so
    doctor states the surface rather than leaving it to be discovered.
    """

    from woof.adapt import _SUPPORTED_FORMATS

    formats = list(_SUPPORTED_FORMATS)
    # NetCDF authoring is reachable when the RUST decoder is, not when
    # netCDF4 is.  Probing the Python library here was correct while
    # `adapt` read NetCDF through `netCDF4.Dataset`; since the decode
    # moved to `rw_netcdf` the old probe was wrong in both directions --
    # it reported the door shut on an install that can open it, and open
    # on an install whose bridge is unstaged.  `_netcdf_decoder_check`
    # is the blocking verdict on that executable; this check states what
    # the door does with it.
    try:
        from woof import netcdf_bridge
        netcdf_decoder = netcdf_bridge.find_netcdf_bin()
    except (ImportError, FileNotFoundError, RuntimeError) as error:
        return Check(
            "arbitrary input", "missing",
            f"NetCDF authoring is unreachable: the Rust NetCDF decoder is "
            f"not usable ({error}); `woof adapt --descriptor ...` cannot "
            "read a NetCDF source without it",
            REINSTALL_HINT, action="pip install -e .",
            brief=_short(f"rw_netcdf unusable: {error}"))
    # Both formats need a decoder now.  Report the split rather than one
    # verdict, because a missing GRIB2 decoder does not stop a NetCDF
    # source from running, and vice versa.
    missing = []
    for name in ("grib2_inventory", "grib2_dump"):
        try:
            if bridges.find_bridge(name) is None:
                missing.append(name)
        except FileNotFoundError:
            missing.append(name)
    if netcdf_decoder is None:
        grib_state = (
            f"GRIB2 authoring is unavailable too ({', '.join(missing)} not "
            "resolvable)." if missing else "GRIB2 authoring is unaffected."
        )
        return Check(
            "arbitrary input", "missing",
            "NetCDF authoring is unreachable: rw_netcdf is not staged, and "
            f"woof keeps no Python NetCDF decoder behind it. {grib_state} "
            "The format contract is docs/intermediate-format.md.",
            netcdf_bridge.netcdf_remedy(),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="rw_netcdf not staged")
    detail = (
        "`woof adapt` authors arbitrary sources from your own descriptor, "
        f"no source name required; formats: {', '.join(formats)}. NetCDF is "
        f"decoded by rw_netcdf ({netcdf_decoder}). "
        "The format contract is docs/intermediate-format.md."
    )
    if missing:
        return Check(
            "arbitrary input", "present",
            detail + f" GRIB2 authoring is unavailable: {', '.join(missing)} "
            "not resolvable; NetCDF authoring is unaffected.",
            bridges.bridge_remedy(missing[0]),
            action="woof fetch-bridges",
            brief=f"netcdf ready; grib2 needs {', '.join(missing)}",
            blocking=False)
    return Check(
        "arbitrary input", "verified", detail,
        brief=f"woof adapt authors {', '.join(formats)} from a descriptor")


def _ocean_climatology_check() -> Check:
    """Is the default mixed-layer ocean's WOA18 MLD climatology staged here?

    The ocean is on by default for forecasts of 48 h and more and starts
    from one month of this climatology.  Unstaged, a forecast fetches its
    month from NCEI at its first step, so an NCEI outage stops it there
    (2026-10-22: THREDDS 503 for an hour, a run died after 60 min of
    preparation).  Degraded, not blocking: the default still works while
    NCEI answers.  Reads only local files (size + SHA-256 per month).
    """
    try:
        from woof.ocean.mixed_layer import climatology
        root, valid, damaged = climatology.staged_state()
    except Exception as error:                          # noqa: BLE001
        return Check("ocean climatology", "missing",
                     f"cannot resolve the ocean climatology cache: {error}",
                     "woof fetch-tables --ocean",
                     action="woof fetch-tables --ocean",
                     brief=_short(str(error)), blocking=False,
                     severity="degraded")
    if len(valid) == 12:
        return Check(
            "ocean climatology", "verified",
            f"12 NOAA WOA18 monthly MLD files at {root} "
            "(size + SHA-256 per pinned month)",
            brief="12 months staged")
    absent = [m for m in range(1, 13) if m not in valid and m not in damaged]
    parts = []
    if absent:
        parts.append(f"{len(absent)} month(s) not staged")
    if damaged:
        parts.append(f"month(s) {', '.join(f'{m:02d}' for m in damaged)} "
                     "do not match their pins (moved aside and refetched "
                     "on use)")
    return Check(
        "ocean climatology", "missing",
        f"{'; '.join(parts)} at {root}; the default mixed-layer ocean "
        "(48 h and longer) then fetches its month from NCEI at the "
        "forecast's first step, and fails there while NCEI is down",
        "woof fetch-tables --ocean\n"
        "  # 12 files, about 85 MB, SHA-256 verified before install;\n"
        "  # --from DIR stages them from a local copy, offline",
        action="woof fetch-tables --ocean",
        brief=f"{12 - len(valid)} of 12 WOA18 months not staged",
        blocking=False, severity="degraded")


def _noah_tables_check() -> Check:
    try:
        from woof.core.landuse import load_landuse_table
        from woof.core.noah import load_tables

        tables = load_tables()
        landuse = load_landuse_table()
    except Exception as error:  # any parse/read failure is the finding
        return Check("noah tables", "missing",
                     f"packaged tables failed to parse: {error}",
                     REINSTALL_HINT, action="pip install -e .",
                     brief=_short(f"failed to parse: {error}"))
    return Check(
        "noah tables", "verified",
        "VEGPARM/SOILPARM/GENPARM parsed by the SOIL_VEG_GEN_PARM "
        f"transcription ({tables.lucats} vegetation / {tables.slcats} "
        f"soil categories); LANDUSE.TBL parsed ({landuse.lucats} "
        "categories)",
        brief="VEGPARM/SOILPARM/GENPARM/LANDUSE parsed")


# ---------------------------------------------------------------------------
# Data roots
# ---------------------------------------------------------------------------

def geography_gaps(geog: Path) -> list[Check]:
    """The reasons a staged WPS_GEOG tree cannot build static fields.

    Empty means usable.  Published because ``woof go`` needs the same
    answer doctor gives, and the alternative -- a second existence test
    written at the call site -- is how the two came to disagree in the
    first place: doctor named the remedy while ``go`` passed
    ``--geog-root`` through unexamined and let ``rw-wps`` fail on a
    missing ``index`` file, after the fetch stage had downloaded the
    forcing.  One check, two readers.

    Scoped to the WRF consumer, because ``go`` runs the WRF static path
    and nothing else.  The same tree also carries a mesh-scoped verdict
    -- ``woof mesh`` opens land-use pins the WRF builder never
    touches -- and folding that in here would refuse a WRF run over a
    download its own preprocessing has no reader for.
    """

    from woof.geog_assets import GEOG_CONSUMER_WRF

    return [check for check in
            _geog_tree_checks(geog, consumers=(GEOG_CONSUMER_WRF,))
            if check.status == "missing"]


def _geog_tree_checks(geog: Path, *,
                      consumers: tuple[str, ...] | None = None) -> list[Check]:
    """Check one staged WPS_GEOG tree, wherever it was resolved from.

    The dataset list comes from :mod:`woof.geog_assets` -- the module
    that stages the tree -- rather than from the domain wizard, which
    also declares it.  The two are one list: ``geog_assets`` derives its
    order from :data:`woof.geog_assets.GEOG_ARCHIVES` and
    ``tests/test_fetch_geog.py`` asserts it equals
    ``woof.domain_wizard.GEOG_DATASETS`` exactly, so this is the same
    nine names read from the half doctor's remedy names.

    Reading them from the wizard instead was also the one thing that
    made this report unreachable from a preprocessing-only install: the
    wizard imports the memory preflight and ``woof.cli``, so the
    standalone RW-WPS wheel could not stage a module that named it, and
    its package-boundary scan said so.
    """

    from woof.geog_assets import (
        GEOG_CONSUMER_MESH, GEOG_CONSUMER_WRF, GEOG_CONSUMERS,
        datasets_required_by, default_geog_consumers, archive_for,
        packaged_recipe_datasets)

    wanted = default_geog_consumers() if consumers is None else consumers

    # The WRF door's own column, not the whole fetchable inventory: this
    # tree also carries datasets that builder never opens, and reporting
    # them here would call a working WRF install broken.
    # By directory, as the builder opens them: the soil categories are two
    # children of the soilgrids container, whose index sits in each child.
    from woof.geog_assets import WRF_DEFAULT_GEOG_DIRS as GEOG_DATASETS

    checks: list[Check] = []
    if not geog.is_dir():
        # A tree nobody has fetched yet is the documented state of a
        # fresh install (the download is an explicit opt-in), so it
        # reports but does not fail the exit code.  A *partial* tree
        # below stays blocking -- that one is a corrupted download.
        return [Check(
            "WPS_GEOG", "missing",
            f"{geog} does not exist (the default geog_root).  Nothing "
            "that builds static fields can run without it",
            GEOG_HINT, action="woof fetch-geog",
            brief=f"{geog} does not exist", blocking=False)]
    if GEOG_CONSUMER_WRF in wanted:
        absent = sorted(name for name in GEOG_DATASETS
                        if not (geog / name).is_dir())
        unindexed = sorted(
            name for name in GEOG_DATASETS
            if (geog / name).is_dir()
            and not (geog / name / "index").is_file())
        if absent or unindexed:
            problems = []
            if absent:
                problems.append(f"missing dataset directorie(s): "
                                f"{', '.join(absent)}")
            if unindexed:
                problems.append(
                    "dataset(s) without their WPS `index` file (empty or "
                    f"partial download): {', '.join(unindexed)}")
            checks.append(Check(
                "WPS_GEOG", "missing", f"{geog}: " + "; ".join(problems),
                GEOG_HINT, action="woof fetch-geog",
                brief=f"{len(absent) + len(unindexed)} of "
                      f"{len(GEOG_DATASETS)} dataset(s) absent or unindexed"))
        else:
            checks.append(Check(
                "WPS_GEOG", "verified",
                f"{geog} carries all {len(GEOG_DATASETS)} datasets the WRF "
                "static builder opens, each with its WPS index file",
                brief=f"all {len(GEOG_DATASETS)} WRF datasets indexed"))
        # The packaged recipes' terrain drag reads pins no consumer column
        # requires (orogwd3_10m for the CONUS HRRR recipe's gwd_opt = 3).
        # Defect: doctor called the tree complete while `woof go` refused
        # that recipe for them.  Not blocking: a configuration without
        # that drag option never opens them.  Whole-tree report only:
        # geography_gaps (the go preflight) asks for the WRF consumer alone,
        # and go checks the drag datasets per configuration itself.
        recipe_missing, recipe_pins = [], []
        for name in (packaged_recipe_datasets() if consumers is None else ()):
            for child in archive_for(name).index_subdirs or ("",):
                if not (geog / name / child / "index").is_file():
                    recipe_missing.append(str(Path(name) / child))
                    if name not in recipe_pins:
                        recipe_pins.append(name)
        if recipe_missing:
            action = "woof fetch-geog --datasets " + ",".join(recipe_pins)
            checks.append(Check(
                "WPS_GEOG recipes", "missing",
                f"{geog}: the packaged recipes' terrain drag reads "
                f"datasets that are not indexed: {', '.join(recipe_missing)}",
                action, action=action,
                brief=f"{len(recipe_missing)} packaged-recipe index(es) missing",
                blocking=False))

    if GEOG_CONSUMER_MESH in wanted:
        checks.extend(_mesh_geog_checks(geog))
    # Optional doors are checked only when requested. Index locations and
    # membership come from pins, including any future multi-dataset archive.
    for consumer in wanted:
        if consumer in (GEOG_CONSUMER_WRF, GEOG_CONSUMER_MESH):
            continue
        missing = []
        for name in datasets_required_by(consumer):
            pin = archive_for(name)
            for child in pin.index_subdirs or ("",):
                if not (geog / name / child / "index").is_file():
                    missing.append(str(Path(name) / child))
        action = f"woof fetch-geog --datasets {consumer}"
        checks.append(Check(
            f"WPS_GEOG {consumer}", "missing" if missing else "verified",
            f"{geog}: missing indexed datasets: {', '.join(missing)}" if missing
            else f"{geog}: all {consumer} datasets indexed",
            action if missing else "", action=action if missing else None))
    return checks


def _mesh_geog_checks(geog: Path) -> list[Check]:
    """The mesh door's own completeness verdict over the same tree.

    `woof mesh` builds MPAS-A's native static (79 variables, 2.9.0) and
    five of them are the Noah-MP soil group, which the WRF static builder
    never opens; its land use and GWD mask come from two archives the WRF
    builder does not read either.  The
    two doors therefore need different datasets out of one tree, and this
    report used to have one verdict for both: without this row doctor
    called the tree complete and `woof mesh` then refused it, which is
    the report being wrong rather than the tree.

    It blocks only when MPAS support was requested: woof hex is installed
    or WOOF_RW_MPAS_STATIC names a builder by hand.  Every wheel bundles
    `rw_mpas_static`, so resolving that binary alone does not establish
    intent.  Without an MPAS request the row stays missing but opt-in:
    WRF never reads these datasets and its complete geography must pass.
    """

    from woof import rustwx_static

    absent = rustwx_static.missing_geog_datasets(geog)
    if not absent:
        return []
    # An override also requests MPAS when its path is broken; the binary
    # check separately reports that missing executable.
    requested = (bool(os.environ.get(rustwx_static.STATIC_ENV))
                 or find_spec("hexcore") is not None)
    # The bare command, not the dataset flag.  The archives these come
    # from are part of the default fetch set, so naming the flag would
    # teach a narrower command than the one that works.  The size is the
    # size of the archives actually absent, read off the pin table: it
    # was a literal "~13 GB" (the soil archive's) on every tree, so a
    # WRF-only tree (`--datasets wrf`, the WOOF image's set) short of only
    # the two land-use archives -- ~0.05 GB to fetch, ~2 GB unpacked --
    # was quoted the 13 GB soil archive it already carries.
    from woof.geog_assets import size_phrase

    carrier_names = rustwx_static.fetchable_for(absent)
    carriers = ", ".join(carrier_names)
    size = size_phrase(tuple(carrier_names))
    remedy = ("woof fetch-geog\n"
              f"  # the default set stages {carriers} ({size}),\n"
              "  # the archive(s) these come from; a re-run fetches only\n"
              "  # what is absent; `--datasets mesh` narrows the default\n"
              "  # set to what woof mesh reads\n"
              "  # or `woof mesh --no-static` to write the grid alone")
    return [Check(
        "WPS_GEOG (MPAS static)", "missing",
        f"{geog} cannot build an MPAS static: "
        f"{', '.join(absent)} are absent. `woof mesh` writes the grid and "
        "refuses the static half of the pair, and the mesh registry admits "
        "neither on its own, so the door cannot be opened. Nothing on the "
        "WRF side reads these, so a WRF-only run is unaffected",
        remedy, action="woof fetch-geog",
        brief=f"{len(absent)} dataset(s) short of an MPAS static",
        blocking=requested,
        severity=SEVERITY_UNREACHABLE if requested else SEVERITY_OPT_IN)]


def _case_data_root_check() -> list[Check]:
    """The case-data root AND the static geography under it.

    v1.0.0 returned a single ``info`` when GPUWM_CASE_DATA_ROOT was
    unset and never looked for WPS_GEOG at all, so doctor printed "no
    gaps; every check verified" on a machine with zero static geography
    -- contradicting the README, which says doctor requires each
    dataset's index file, and greenlighting a box on which nothing
    downstream could run.  An unset variable is not an excuse to skip
    the check: ``fetch-geog`` and every config default resolve to the
    same place through :func:`woof.geog_assets.default_geog_root`, so
    doctor looks exactly there.
    """

    from woof.geog_assets import default_geog_root

    raw = os.environ.get("GPUWM_CASE_DATA_ROOT")
    if not raw:
        return [
            Check(
                "GPUWM_CASE_DATA_ROOT", "info",
                "not set.  Layout when you set it: the root is the "
                "directory that CONTAINS your case bundles and (by "
                "default) WPS_GEOG -- configs reference "
                "${GPUWM_CASE_DATA_ROOT}/<bundle>/... and geog_root "
                "defaults to ${GPUWM_CASE_DATA_ROOT}/WPS_GEOG",
                brief=f"not set; geog_root defaults to "
                      f"{default_geog_root()}"),
            *_geog_tree_checks(default_geog_root()),
        ]
    root = Path(raw)
    if not root.is_dir():
        return [Check(
            "GPUWM_CASE_DATA_ROOT", "missing",
            f"set to {raw} but that directory does not exist",
            "# point GPUWM_CASE_DATA_ROOT at the directory that CONTAINS\n"
            "  # your case bundles and WPS_GEOG -- there is no command to\n"
            "  # print here, only your path",
            action="point GPUWM_CASE_DATA_ROOT at an existing directory",
            brief=f"set to {raw}, which does not exist")]
    return [
        Check("GPUWM_CASE_DATA_ROOT", "present",
              f"{root} (directory exists; its datasets are checked "
              "individually below)", brief=str(root)),
        *_geog_tree_checks(root / "WPS_GEOG"),
    ]


def _distribution_manifest_check() -> Check:
    """The sealed manifest: the WHOLE schema, then the artifacts.

    This used to check ``schema`` and ``status`` and stop.  A field user
    hand-authored a two-key document to get past an unrelated bug, got a
    green line here, and died several minutes into a preparation at
    ``contract.platform.backends`` -- a key doctor had never looked at.
    The validator it calls now is the same one every consumer calls, and
    it names every missing field at once.
    """

    from woof.runtime_manifest import RUNTIME_SCHEMA, manifest_defects

    name = "WOOF_NATIVE_DISTRIBUTION_MANIFEST"
    raw = os.environ.get(name)
    if not raw:
        return Check(
            name, "info",
            "not set (normal for source clones and pip installs; only "
            "sealed runtime archives set it to bind their decoder "
            "inventory)",
            brief="not set (normal for clones and pip installs)")
    path = Path(raw)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return Check(
            name, "missing",
            f"set to {raw} but not a readable JSON document: {error}",
            f"# unset {name} unless you are running a sealed runtime\n"
            "  # archive, whose installer sets it correctly",
            action=f"unset {name}", brief=_short(f"unreadable: {error}"))
    defects = manifest_defects(payload)
    if defects:
        shown = "; ".join(defects[:4])
        more = f" (+{len(defects) - 4} more)" if len(defects) > 4 else ""
        return Check(
            name, "missing",
            f"{path}: not a complete {RUNTIME_SCHEMA} document -- "
            f"{len(defects)} problem(s): {shown}{more}",
            f"# unset {name} unless you are running a sealed runtime\n"
            "  # archive, whose installer writes this document beside its\n"
            "  # artifacts.  A clone or a pip install binds its identity\n"
            "  # from git or from the installed wheel and needs no manifest.",
            action=f"unset {name}",
            brief=f"{len(defects)} schema problem(s): {_short(defects[0], 40)}")
    # A non-empty per-artifact inventory is part of the required schema
    # now, so the old "READY but declares no hashes -- presence only"
    # branch is unreachable: such a document is refused above, by name.
    declared = payload["payload"]
    root = path.resolve().parent
    failures: list[str] = []
    verified = 0
    for relative, record in sorted(declared.items()):
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(expected, str):
            failures.append(f"{relative}: malformed manifest record")
            continue
        artifact = root / relative
        if not artifact.is_file():
            failures.append(f"{relative}: missing")
            continue
        expected_bytes = record.get("bytes")
        if (isinstance(expected_bytes, int)
                and artifact.stat().st_size != expected_bytes):
            failures.append(f"{relative}: size mismatch")
            continue
        if _sha256(artifact) != expected:
            failures.append(f"{relative}: sha256 mismatch")
            continue
        verified += 1
    if failures:
        shown = "; ".join(failures[:5])
        more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
        return Check(
            name, "missing",
            f"{path}: {len(failures)} of {len(declared)} declared "
            f"artifacts failed revalidation: {shown}{more}",
            "# re-extract the sealed runtime archive (its installer wrote\n"
            f"  # this manifest beside its artifacts), or unset {name}",
            action="re-extract the sealed runtime archive, or unset "
                   f"{name}",
            brief=f"{len(failures)} of {len(declared)} artifacts failed "
                  "revalidation")
    return Check(
        name, "verified",
        f"{path}: READY; all {verified} declared artifacts re-hashed "
        "and match",
        brief=f"READY; {verified} artifacts re-hashed")


# ---------------------------------------------------------------------------
# The paths a RUN resolves: provenance, importability, per-source decoders
# ---------------------------------------------------------------------------

#: The console script every printed command in this product begins with.
#: Its launcher is what a shell has to be able to find.
_FRONT_DOOR_SCRIPT = "woof"


def _console_script_dirs() -> tuple[Path, ...]:
    """Directories this interpreter's installs put console scripts in.

    Both are asked for, because the two answers differ exactly where the
    defect lives: a ``pip install --user`` (which pip picks by itself on
    a system Python it cannot write to) lands scripts in the USER
    scheme's directory, while ``sysconfig``'s default names the
    interpreter's own.
    """

    import sysconfig

    found: list[Path] = []
    for scheme in (sysconfig.get_default_scheme(),
                   "nt_user" if os.name == "nt" else "posix_user"):
        try:
            raw = sysconfig.get_path("scripts", scheme)
        except (KeyError, ValueError):
            continue
        if not raw:
            continue
        path = Path(raw)
        if path not in found:
            found.append(path)
    return tuple(found)


def _console_scripts_check() -> Check:
    """Can a shell RUN ``woof``, not merely import it?

    Every command this product prints -- the wizard's fetch line, the
    forecast line, this report's own remedies -- begins with a console
    script.  Walked on Windows: an editable install wrote the launchers
    into the user scheme's Scripts directory while PATH carried the
    interpreter's own, which held none of them, so every one of those
    printed commands answered "is not recognized" and a working install
    read as a broken one.  Importing the package proves nothing about
    that: the failure is PATH resolution, so PATH is what is asked.
    """

    name = "console scripts on PATH (every printed command starts with one)"
    directories = _console_script_dirs()
    holders = [
        directory for directory in directories
        if any((directory / candidate).exists()
               for candidate in ((f"{_FRONT_DOOR_SCRIPT}.exe",
                                  f"{_FRONT_DOOR_SCRIPT}-script.py")
                                 if os.name == "nt"
                                 else (_FRONT_DOOR_SCRIPT,)))]
    if not holders:
        return Check(
            name, "untested",
            "not tested: no "
            f"{_FRONT_DOOR_SCRIPT} launcher was found in this "
            f"interpreter's script directories ({len(directories)} looked "
            "at), so there is nothing to resolve yet",
            "# install the package so its console scripts exist:\n"
            "pip install -e .",
            action="pip install -e .",
            brief="no launcher installed yet")
    on_path = [Path(entry) for entry in
               os.environ.get("PATH", "").split(os.pathsep) if entry]
    reachable = [directory for directory in holders
                 if any(_same_dir(directory, entry) for entry in on_path)]
    if reachable:
        return Check(name, "verified",
                     f"{_FRONT_DOOR_SCRIPT} resolves from {reachable[0]}",
                     brief=_short(str(reachable[0])))
    directory = holders[0]
    # The SHELL decides the remedy's syntax, and it is declared once for
    # the whole report -- `os.name` above answers a different question
    # (what the launcher file on disk is called).
    if bridges.WINDOWS_SHELL:
        remedy = (
            "# this PowerShell session:\n"
            f'$env:Path = "{directory};" + $env:Path\n'
            "# permanently: add that directory to your account's Path under\n"
            "# Settings, Environment Variables")
        action = f'$env:Path = "{directory};" + $env:Path'
    else:
        remedy = ("# this shell session:\n"
                  f'export PATH="{directory}:$PATH"\n'
                  "# permanently: add that line to your shell profile")
        action = f'export PATH="{directory}:$PATH"'
    return Check(
        name, "missing",
        f"{_FRONT_DOOR_SCRIPT} is installed in {directory}, which is not on "
        "PATH: every command this product prints will answer that "
        f"{_FRONT_DOOR_SCRIPT} is not recognized",
        remedy, action=action, brief=f"{directory} not on PATH")


def _same_dir(left: Path, right: Path) -> bool:
    """Do two path spellings name one directory?  Never raises."""

    try:
        if left.samefile(right):
            return True
    except OSError:
        pass
    try:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(
            str(right.resolve()))
    except OSError:
        return False


def _install_identity_check() -> Check:
    """Can this install name itself?  Every run's receipt asks it.

    Not a nicety.  ``_source_identity`` runs before any data is read and
    used to be a bare ``git rev-parse`` with the working directory set
    to the directory holding the package -- ``site-packages`` on a pip
    install, where git exits 128.  A field user hit
    ``CalledProcessError: returned non-zero exit status 128`` on the
    first preparation of a green install; nothing in the estate report
    had asked the question that fails.  So doctor asks it, here, by
    calling the exact resolver the run calls.
    """

    from woof.runtime_manifest import IdentityError, provenance

    root = Path(__file__).resolve().parent.parent
    name = "run provenance (identity every receipt binds)"
    try:
        identity = provenance(root)
    except IdentityError as error:
        return Check(
            name, "missing", str(error),
            REINSTALL_HINT
            + "\n  # a run records what produced it; with no distribution\n"
            "  # metadata and no checkout there is nothing to record",
            action="pip install -e .", brief=_short(str(error)))
    except Exception as error:  # a manifest defect, reported in full below
        return Check(
            name, "missing",
            f"the bound distribution manifest is unusable: {error}",
            "# see the WOOF_NATIVE_DISTRIBUTION_MANIFEST line below",
            action="unset WOOF_NATIVE_DISTRIBUTION_MANIFEST",
            brief=_short(str(error)))
    source = identity["identity_source"]
    if source == "git":
        evidence = f"git checkout {str(identity['git_commit'])[:12]} at {root}"
    elif source == "gpuwm-native-distribution-manifest":
        evidence = ("sealed runtime manifest "
                    f"{str(identity['git_commit'])[:12]}")
    elif source == "installed-source-content":
        content = identity["installed_source_content"] or {}
        missing = content.get("missing_file_count") or 0
        evidence = (f"source tree {content.get('source_version')} at {root} "
                    f"with no .git, bound by the content of its "
                    f"{content.get('source_file_count')} listed files "
                    f"({str(content.get('content_sha256'))[:12]}"
                    + (f"; {missing} listed file(s) missing" if missing else "")
                    + ")")
    elif source == "installed-editable-source":
        editable = identity["installed_editable"] or {}
        git = editable.get("git") or {}
        # A dirty editable tree is a normal developer install, not a
        # defect -- but the check must SAY so, because the whole point
        # of binding the source tree is that the receipt stops claiming
        # a clean identity the tree does not have.
        # ``dirty is None`` means git could not be launched and the tree
        # was never inspected.  Reporting that as "clean" would be the
        # one claim this whole binding exists to stop.
        if git.get("dirty") is None:
            state = "worktree unverified"
        else:
            state = "dirty" if git.get("dirty") else "clean"
        evidence = (f"editable install of {editable.get('distribution_name')} "
                    f"{editable.get('distribution_version')} at "
                    f"{editable.get('source_root')} "
                    f"(git {str(git.get('commit_full'))[:12]} on "
                    f"{git.get('branch') or 'a detached HEAD'}, {state})")
    else:
        wheel = identity["installed_wheel"] or {}
        evidence = (f"installed wheel {wheel.get('distribution_name')} "
                    f"{wheel.get('distribution_version')} "
                    f"({wheel.get('record_file_count')} RECORD entries, "
                    f"aggregate "
                    f"{str(wheel.get('record_aggregate_sha256'))[:12]})")
    return Check(name, "verified", f"{source}: {evidence}",
                 brief=_short(f"{source}: {evidence}"))


#: Modules a data route launches in a fresh interpreter, from whatever
#: directory the user happens to be standing in.
_ROUTE_ENTRY_MODULES = (
    "tools.prepare_hrrr_wrf",
    "tools.hrrr_single_domain_benchmark",
)


def _non_git_import_check() -> Check:
    """Do the route entry points import from a directory that is not a repo?

    The wrapper scripts are packaged modules, but they resolve paths and
    provenance relative to the installed tree, and one of them used to
    do that with a subprocess whose working directory decided the
    answer.  Importing them from a scratch directory -- which is where a
    user actually runs -- is the cheapest possible proof that neither
    the current directory nor the absence of a repository around it
    changes whether they load.
    """

    import tempfile

    name = "route entry points (import from a non-repository directory)"
    code = "".join(f"import {module}\n" for module in _ROUTE_ENTRY_MODULES)
    # The probe's import path is the directory that holds the ``woof``
    # package, which is what BOTH installs give a run: site-packages is
    # already on ``sys.path`` for a wheel, and a checkout root is what
    # every documented developer invocation supplies.  Seeding it keeps
    # the question about the working directory -- the thing that broke
    # -- rather than about whether a clone happens to be pip-installed.
    root = str(Path(__file__).resolve().parent.parent)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        root + os.pathsep + environment.get("PYTHONPATH", "")).rstrip(
            os.pathsep)
    with tempfile.TemporaryDirectory(prefix="gpuwm-doctor-cwd-") as scratch:
        try:
            probe = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True,
                errors="replace", cwd=scratch, env=environment,
                timeout=_PROBE_TIMEOUT_S * 4)
        except (OSError, subprocess.TimeoutExpired) as error:
            return Check(
                name, "missing", f"the import probe failed to run: {error}",
                REINSTALL_HINT, action="pip install -e .",
                brief=_short(str(error)))
    if probe.returncode == 0:
        return Check(
            name, "verified",
            f"{', '.join(_ROUTE_ENTRY_MODULES)} imported from a scratch "
            "directory outside any git repository",
            brief=f"{len(_ROUTE_ENTRY_MODULES)} modules import cleanly")
    tail = [line for line in (probe.stderr or "").strip().splitlines()
            if line.strip()]
    reason = tail[-1] if tail else f"exit {probe.returncode}"
    return Check(
        name, "missing", f"failed to import: {reason}",
        REINSTALL_HINT
        + "\n  # the preparation wrappers must load from any directory;\n"
        "  # an install that only works inside a checkout is incomplete",
        action="pip install -e .", brief=_short(reason))


#: Doctor's per-source route checks, by the ``--source`` name.  These are
#: public data products, not cases.
#:
#: DERIVED from the two declarations that between them name every route
#: this build serves, never transcribed.  It read ``SOURCE_DECODERS``
#: alone, which is the three routes with a dedicated Rust decoder, so
#: ``--source`` accepted era5/gfs/hrrr and refused ``hrrr-prs`` -- a
#: route ``woof fetch`` and ``woof prep`` both serve, and the one the
#: persona walk actually used (UX finding N17).  The registry is the
#: other half: a model joins this list by having a row, which is the
#: same act that makes it reachable from every other door.
#:
#: Resolved LAZILY, through the module ``__getattr__`` at the bottom of
#: this file.  ``woof.doctor`` must import on an interpreter that has
#: no scientific stack -- that is what a reader runs when the install is
#: broken, and a pinned test proves it -- and the source registry pulls
#: numpy in behind its coverage windows.  Code inside this module calls
#: :func:`_doctor_sources`; everything outside reads the name.
def _doctor_sources() -> tuple[str, ...]:
    """Every route name ``--source`` accepts, resolved once."""

    from woof import source_adapters

    return tuple(sorted(
        set(bridges.SOURCE_DECODERS)
        | {adapter.source_id
           for adapter in source_adapters.source_adapters()}))

#: Fold key for one source's route findings.
_GROUP_ROUTE = "route"


def _decoder_route_check(source: str) -> Check:
    """THE decoder that source's preparation will launch, resolved here.

    Resolved by calling :func:`woof.bridges.resolve_source_decoder` --
    the function the preparation wrapper itself calls -- rather than by
    re-deriving a path.  Doctor reporting one path while preparation
    used another is not a hypothetical: the wrapper resolved a source
    checkout's cargo workspace under ``site-packages``, so a wheel
    install got "no gaps" here and ``HRRR decoder is missing`` there.
    """

    name = f"{source} route decoder"
    try:
        found = bridges.resolve_source_decoder(source)
    except FileNotFoundError as error:
        message = str(error)
        headline, _, remedy = message.partition("\n")
        return Check(
            name, "missing", headline,
            remedy or bridges.bridge_remedy(bridges.SOURCE_DECODERS[source]),
            action=_build_action(), brief="not resolvable",
            group=_GROUP_ROUTE)
    except bridges.DecoderContractError as error:
        # The resolver owns the contract check now, so this arm reports
        # what preparation would refuse rather than re-deriving it.  A
        # stale binary reaches exactly one verdict, from one place.
        headline, _, remedy = str(error).partition("\n")
        return Check(
            name, "missing", headline,
            "# this one has to be replaced before the "
            f"{source} route can run:\n" + remedy,
            action=_build_action(),
            brief=_short("does not speak this release's contract"),
            group=_GROUP_ROUTE)
    executable = bridges.SOURCE_DECODERS[source]
    # Still asked, still reported: the resolver guarantees this is True,
    # and the evidence string is what the verified line says out loud.
    _, evidence = bridges.bridge_abi_matches(executable, found)
    return Check(
        name, "verified",
        f"preparation will launch {found} ({evidence})",
        brief=_short(str(found)), group=_GROUP_ROUTE)


def _hrrr_fetch_path_check() -> Check:
    """Which byte transport ``woof fetch --source hrrr`` will use.

    Reported because the difference is 16x on a real link and was
    invisible until after the download: the Python fallback can only do
    ``.idx`` range subsets, and one 419 MB file took 560 s that way
    against 27-35 s taken whole.  Not a gap -- the fallback is the
    documented always-available route and the fetch-backbone line above
    already carries the remedy -- so this is ``info`` when it is slow
    and ``verified`` when it is not.
    """

    from woof import fetch, rustwx_fetch

    name = "hrrr route fetch transport"
    try:
        found = rustwx_fetch.find_fetch_bin()
        usable = found is not None and rustwx_fetch.probe_fetch_bin(found)[0]
    except FileNotFoundError:
        found, usable = None, False
    if usable:
        return Check(
            name, "verified",
            f"engine rust ({found}), default --mode {fetch.HRRR_DEFAULT_MODE}"
            " (the .idx record subset in parallel range GETs; a whole "
            "object where the index cannot carry it)",
            brief=f"rust, {fetch.HRRR_DEFAULT_MODE}", group=_GROUP_ROUTE)
    return Check(
        name, "info",
        "engine python (the rust backbone is not usable here), which can "
        "only do .idx range subsets -- correct, and roughly an order of "
        "magnitude slower per file",
        "woof setup\n"
        "  # stages the rust fetch backbone; whole-file transfers need it",
        action="woof setup", brief="python transport (.idx subsets only)",
        group=_GROUP_ROUTE)


def _era5_fetch_path_check() -> Check:
    """Probe local retrieval prerequisites without reading keys or contacting CDS."""
    from woof import fetch, zarr_bridge

    cds_imports, cds_evidence = _import_probe("cdsapi", distribution="cdsapi")
    try:
        credentials = (bool(os.environ.get("CDSAPI_URL"))
                       and bool(os.environ.get("CDSAPI_KEY"))) or fetch.cds_credentials_path().is_file()
    except OSError:
        credentials = False
    with bridges.inspection_only():
        try:
            path = zarr_bridge.resolve_zarr_bin()
            arco, arco_evidence = bridges.bridge_abi_matches("rw_zarr", path)
        except (OSError, ValueError, bridges.DecoderContractError) as error:
            arco, arco_evidence = False, type(error).__name__
    cds_ready = cds_imports and credentials
    detail = (
        "CDS --retrieve: " + ("local prerequisites verified" if cds_ready else
            "needs " + ", ".join(item for item, ready in
                (("an importable cdsapi", cds_imports),
                 ("a credential file or CDSAPI_URL/CDSAPI_KEY pair", credentials)) if not ready))
        + f" ({cds_evidence}); --era5-provider arco: "
        + ("keyless ARCO reader is locally available" if arco else
           "native rw_zarr reader is missing or ABI-incompatible")
        + f" ({arco_evidence}); request document fallback is always available. "
          "Credential contents, authorization, provider availability and "
          "live retrieval are not tested by this local report.")
    return Check(
        "era5 route fetch transport", "verified" if cds_ready else "info", detail,
        # Every line of a remedy is a command that runs as printed or a `#`
        # comment (the paste contract in tests/test_doctor.py); a prose
        # sentence fused into a paste is the defect that contract exists
        # for, and this check arrived carrying one.
        None if cds_ready else
            "# --retrieve needs an importable cdsapi and CDS credentials\n"
            "pip install cdsapi\n"
            "# then write ~/.cdsapirc, or set CDSAPI_URL and CDSAPI_KEY\n"
            "# the keyless road instead, which stages the native zarr reader\n"
            "woof setup\n"
            "# with neither, woof fetch still writes the CDS request document",
        brief=("CDS local prerequisites verified" if cds_ready else
               "keyless ARCO locally available" if arco else "CDS request document available"),
        group=_GROUP_ROUTE)


def _gfs_fetch_path_check() -> Check:
    """Which byte transports ``woof fetch --source gfs`` can use.

    Two first-class transports, reported the way the HRRR line is: the
    default NOMADS grib-filter crop (governed stdlib HTTP, always
    available) and ``--mode full-file`` (whole pgrb2.0p25 objects from
    the S3 archive, preferring the rust backbone's parallel range
    GETs).  The report that prompted this line showed doctor naming the
    hrrr route while ``gfs_grib2_bridge`` sat verified two rows up with
    no route reported around it.
    """

    from woof import rustwx_fetch

    name = "gfs route fetch transport"
    try:
        found = rustwx_fetch.find_fetch_bin()
        usable = found is not None and rustwx_fetch.probe_fetch_bin(found)[0]
    except FileNotFoundError:
        found, usable = None, False
    if usable:
        return Check(
            name, "verified",
            "default: NOMADS grib-filter crop (governed stdlib HTTP); "
            f"--mode full-file: whole S3 objects, engine rust ({found})",
            brief="cgi-subset default; full-file via rust",
            group=_GROUP_ROUTE)
    return Check(
        name, "verified",
        "default: NOMADS grib-filter crop (governed stdlib HTTP); "
        "--mode full-file: whole S3 objects through the stdlib "
        "transport (the rust backbone is not usable here; `woof "
        "setup` stages the faster parallel-range-GET engine)",
        brief="cgi-subset default; full-file via python",
        group=_GROUP_ROUTE)


def _registry_transport(source: str) -> str:
    """What ``woof fetch --source SOURCE`` would move, in one clause."""

    from woof import fetch_routes

    try:
        route = fetch_routes.route_for(source)
    except ValueError as error:
        return str(error).splitlines()[0]
    hosts = ", ".join(host.name for host in route.hosts)
    return (f"fetch takes whole objects from {hosts} "
            f"({route.label}) through the packaged route table")


def _registry_route_check(source: str, engine: Check) -> Check:
    """One line for a route with no decoder executable of its own.

    A route without a dedicated binary is not a route without an
    answer: the registry row says whether an implementation exists, the
    packaged profile says WHAT the preparation decodes it through, and
    the fetch route table says whether ``woof fetch`` can go and get
    it.  All three are table lookups, so the twenty-eight routes this
    added to the default estate cost no probes -- ``engine`` is the one
    real resolution, made once by the caller and shared across them.

    ONE line rather than the decoder/transport pair the three
    executable-backed routes print, because both halves are one table
    read here and a second line would double the length of the default
    report to repeat a lookup.

    The engine's own verdict stays on the engine's own line.  This one
    DEFERS to it (``info``, never ``missing``) so that one unbuilt
    binary produces one gap with one remedy instead of twenty-eight
    copies of the same gap -- which is the shape that makes a report
    unreadable rather than more complete.
    """

    from woof.source_adapters import get_source_adapter

    adapter = get_source_adapter(source)
    name = f"{source} route"
    transport = _registry_transport(source)
    if not adapter.runnable:
        return Check(
            name, "info",
            "declared in the registry with no strict implementation "
            f"route on this release (status {adapter.status.value}); "
            f"`woof prep --show-source {source}` prints the row and "
            f"what it would need.  {transport}",
            brief="declared, no implementation route", group=_GROUP_ROUTE,
            blocking=False)
    if adapter.packaged_profile is None:
        return Check(
            name, "info",
            f"decoded through a mapping the caller supplies (runner "
            f"{adapter.runner}), so there is no packaged profile for "
            f"this report to resolve.  {transport}",
            brief="caller-supplied mapping", group=_GROUP_ROUTE,
            blocking=False)
    if engine.status == "verified":
        return Check(
            name, "verified",
            f"packaged profile {adapter.packaged_profile}, decoded on "
            f"{engine.brief or 'the mapped engine'}.  {transport}",
            brief=_short(adapter.packaged_profile), group=_GROUP_ROUTE)
    return Check(
        name, "info",
        f"packaged profile {adapter.packaged_profile}, decoded on the "
        "mapped engine -- which this estate has not resolved; see the "
        f"`{engine.name}` line for the verdict and the remedy.  "
        f"{transport}",
        engine.remedy, action=engine.action,
        brief="mapped engine unresolved; see the engine line",
        group=_GROUP_ROUTE, blocking=False)


def _source_route_checks(source: str, engine: Check | None = None
                         ) -> list[Check]:
    """Everything one data route resolves before it reads a byte.

    ``engine`` is the mapped decode engine's own check, resolved once
    per report by :func:`collect_checks` and shared across every
    registry route; omitted, it is resolved here, which is what a
    caller asking about one route wants.
    """

    known = _doctor_sources()
    if source not in known:
        raise ValueError(f"unknown doctor source {source!r}; known: "
                         f"{list(known)}")
    if source not in bridges.SOURCE_DECODERS:
        return [_registry_route_check(
            source,
            engine if engine is not None else _mapped_engine_check())]
    checks = [_decoder_route_check(source)]
    if source == "hrrr":
        checks.append(_hrrr_fetch_path_check())
    if source == "gfs":
        checks.append(_gfs_fetch_path_check())
    if source == "era5":
        checks.append(_era5_fetch_path_check())
    return checks


def _default_route_order() -> tuple[str, ...]:
    """Every route, ordered so the terse report can fold the tail.

    Three blocks, and the order is a stated grouping rather than a
    sort of the findings: the routes with a decoder executable of their
    own (which print a decoder line and a transport line each), then
    the registry routes that have an implementation, then the registry
    rows that are declared and not yet runnable.  Adjacent same-verdict
    lines fold, so the twenty-eight routes `--source` gained do not
    turn the default report into a page of alternating one-word
    verdicts; every one of them is still a named line under
    ``--explain`` and in ``--json``, and `--source NAME` prints its own.
    """

    from woof.source_adapters import get_source_adapter

    executable = sorted(bridges.SOURCE_DECODERS)
    rest = [source for source in _doctor_sources()
            if source not in bridges.SOURCE_DECODERS]

    def rank(source: str) -> tuple[int, str]:
        adapter = get_source_adapter(source)
        if not adapter.runnable:
            return (2, source)
        # Runnable through a packaged profile, then runnable only
        # through a mapping the caller brings: three tiers, so each one
        # folds to a line instead of interrupting its neighbour's run.
        return (0 if adapter.packaged_profile is not None else 1, source)

    return tuple(executable) + tuple(sorted(rest, key=rank))


#: Windows' classic path ceiling, measured rather than assumed.  On this
#: project's own reference box (LongPathsEnabled = 0) a 259-character
#: full path opened and a 260-character one failed with
#: ERROR_PATH_NOT_FOUND -- errno 2, which surfaces as
#: ``FileNotFoundError`` and reads as "the file is not there" rather than
#: "the name is too long".  That is the same number
#: :data:`woof.render_layout._MAX_PATH` documents for the render tree;
#: it is restated as an import below, not as a second literal.
_LONG_PATH_ROW = "Windows long paths (install root depth)"


def _long_paths_enabled() -> int | None:
    """``LongPathsEnabled`` from the machine hive, or None if unreadable.

    Absence is the same answer as 0: the value is created by whoever
    turns the feature on, and a box that never turned it on has the
    MAX_PATH ceiling.  None is reserved for "could not look", which is
    not judged.
    """

    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            value, _kind = winreg.QueryValueEx(key, "LongPathsEnabled")
            return int(value)
    except FileNotFoundError:
        return 0
    except OSError:                          # noqa: BLE001 - reported
        return None


def _longest_kernel_source_path() -> "Path | None":
    """The longest full path NVRTC's sources are read from, or None."""

    try:
        from woof.core import kernels

        kdir = Path(kernels.__file__).parent
        sources = [path for path in kdir.iterdir()
                   if path.suffix in (".cu", ".cuh")]
    except Exception:                        # noqa: BLE001 - reported
        return None
    if not sources:
        return None
    return max(sources, key=lambda path: len(str(path)))


def _is_windows() -> bool:
    """The platform test the long-path row consults, as its own seam.

    Tests force the Windows arm through THIS function, never through
    ``os.name``: ``pathlib.Path`` chooses ``WindowsPath`` from
    ``os.name`` at construction, so a test that rewrote the global
    ``os.name`` to ``"nt"`` on a Linux box made every ``Path()`` in the
    process raise ``NotImplementedError`` (Python 3.11), including the
    one pytest builds to format the failure, and the 2.6.2 publish run's
    test job died in pytest's own reporter.  A Windows box was blind to
    it, since the patch is a no-op there.
    """

    return os.name == "nt"


def _long_path_install_root_check() -> Check:
    """A deep install root that no other row would ever name.

    The concrete breakage, and it is silent: every CUDA kernel this
    model compiles is read off disk as text out of
    ``woof/core/kernels`` and handed to NVRTC as one source string.
    There are no include paths to lengthen -- the length is the INSTALL
    ROOT's.  Put the venv somewhere deep on a Windows box that never
    enabled long paths and the read fails with ``FileNotFoundError`` on
    a file that ``rglob`` had just listed, so the report reads as a
    broken install rather than a path too long by four characters.
    Nothing else in doctor measures the install root's depth, so the
    box had no row to read.
    """

    if not _is_windows():
        return Check(
            _LONG_PATH_ROW, "info",
            "not judged -- the MAX_PATH ceiling is a Windows filesystem "
            f"limit and this box is {sys.platform}",
            brief="not Windows", blocking=False)

    from woof.render_layout import _MAX_PATH

    longest = _longest_kernel_source_path()
    if longest is None:
        return Check(
            _LONG_PATH_ROW, "info",
            "not judged -- the packaged kernel sources could not be "
            "listed, so there is no path here to measure",
            brief="no kernel sources", blocking=False)

    length = len(str(longest))
    enabled = _long_paths_enabled()
    headroom = _MAX_PATH - 1 - length
    if enabled == 1:
        return Check(
            _LONG_PATH_ROW, "verified",
            f"LongPathsEnabled=1, so the {_MAX_PATH}-character ceiling "
            f"does not apply; longest kernel source path {length} chars",
            brief=f"long paths on ({length} chars)")
    if enabled is None:
        return Check(
            _LONG_PATH_ROW, "info",
            "not judged -- LongPathsEnabled could not be read from the "
            f"machine hive; longest kernel source path {length} chars, "
            f"{_MAX_PATH} is the ceiling when the value is 0",
            brief="registry unreadable", blocking=False)
    if length < _MAX_PATH:
        return Check(
            _LONG_PATH_ROW, "verified",
            f"longest kernel source path {length} chars, {headroom} "
            f"under the {_MAX_PATH}-character ceiling "
            "(LongPathsEnabled=0)",
            brief=f"{headroom} chars of headroom")
    detail = (f"the install root is too deep: {longest.name} sits at "
              f"{length} characters and this box has LongPathsEnabled=0, "
              f"so the {_MAX_PATH}-character ceiling applies.  Reading "
              "that file raises FileNotFoundError, so every kernel "
              "compile fails as a missing file rather than a long path")
    remedy = ("# Two fixes; either one is enough." "\n"
              "  # 1. Reinstall into a shorter root, e.g. C:\\gpuwm, "
              "and the" "\n"
              "  #    whole tree comes back under the ceiling." "\n"
              "  # 2. Or lift the ceiling machine-wide, once, as "
              "administrator:" "\n"
              "  #    set LongPathsEnabled to 1 (DWORD) under" "\n"
              "  #    HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
              "and reboot.")
    return Check(
        _LONG_PATH_ROW, "missing", detail, remedy,
        action=("reinstall under a shorter root, or set "
                "LongPathsEnabled=1 and reboot"),
        brief=f"{length} chars, over the {_MAX_PATH} ceiling",
        blocking=False, severity=SEVERITY_DEGRADED)



def collect_checks(sources: tuple[str, ...] | None = None,
                   progress: Callable[[str], None] | None = None
                   ) -> list[Check]:
    """The estate as it IS, which is the whole job of this command.

    A thin wrapper over :func:`_collect_checks` holding one contract:
    every resolution below happens inside
    :func:`woof.bridges.inspection_only`, so a staged artifact that is
    not this release's bytes is REPORTED here -- see
    :func:`_staged_estate_check`, which names it and offers ``woof
    fetch-bridges`` -- and never refreshed or refused underneath the
    report.  A diagnostic that repairs the box while measuring it can
    no longer say what the box was, and one that dies on the defect it
    was run to find is worse than no diagnostic at all.
    """

    with bridges.inspection_only():
        return _collect_checks(sources, progress)


def _collect_checks(sources: tuple[str, ...] | None = None,
                    progress: Callable[[str], None] | None = None
                    ) -> list[Check]:
    """The estate, plus every named data route's own resolution.

    ``sources=None`` means every route this build knows, which is what
    a bare ``woof doctor`` runs: the report that said "no gaps" before
    a route died on a path it had never resolved is the reason these
    are in the default estate rather than behind a flag.

    ``progress`` is called with the name of each phase as that phase
    begins, and is how the caller narrates a wait.  Measured on a first
    contact: 7.0 s wall, 6.1 s of it silent -- a subprocess import per
    declared package (1.4 s here) and the non-repository import probe
    (2.1 s here) between them own most of it, and neither said what it
    was doing.  The phases are coarse on purpose: one line per second
    or so of work, not one per check, because thirty lines of narration
    for a four-second command is the same wall by another name.

    It is a callback rather than a print because this function is the
    library surface every test and every embedder calls, and a
    diagnostic that writes to a stream nobody asked for is not one.
    :func:`doctor_main` is what routes it to stderr, where it cannot
    enter the report.
    """

    def phase(name: str) -> None:
        if progress is not None:
            progress(name)

    selected = (_default_route_order() if sources is None
                else tuple(sources))
    checks: list[Check] = []
    # Its own phase, not folded into the packages one below, because on
    # a box that HAS a GPU these three are the expensive ones: each
    # spawns a short-lived device subprocess (the cuBLAS pairing, a
    # cold-cache NVRTC compile, the eigensolver), and a reader whose
    # driver is wedged waits here longest of all.
    phase("the GPU runtime (cupy wheel, kernel headers, radar-DA solver)")
    version = ".".join(str(v) for v in sys.version_info[:3])
    if sys.version_info >= (3, 11):
        checks.append(Check("python", "verified", f"{version} (>= 3.11)",
                            brief=f"{version} (>= 3.11)"))
    else:
        checks.append(Check(
            "python", "missing", f"{version} is below the 3.11 floor",
            "# install Python 3.11 or newer -- which installer is right\n"
            "  # here depends on how this Python was installed",
            action="install Python 3.11 or newer",
            brief=f"{version} is below the 3.11 floor"))
    checks.append(_host_threads_check())
    checks.append(_cupy_check())
    # Between the wheel and the solver, because that is the order the
    # three fail in: a wheel that will not load, then a wheel that
    # loads but cannot compile, then a compile that cannot factor.
    checks.append(_cuda_headers_check())
    checks.append(_da_eigensolver_check())
    # Beside the extras, because they are the same question asked of a
    # different feature: can this install actually run the thing its
    # documentation describes?  This one keeps its own line because
    # rasterio/pyproj stopped being an extra in 2.3.3 -- they are
    # runtime dependencies now, and the extras block below defers to
    # this check rather than probing them a second time.
    checks.append(_geog_stack_check())
    # The measured slow span on a box with no GPU: one short-lived
    # subprocess import per package this install's metadata declares.
    phase("python packages this install declares (one import each)")
    # Every extra this INSTALL declares, in the packaging's own order.
    # Through 2.3.2 the only Python-package line here was a single
    # "render extra" that probed wrf and matplotlib, so `obs`,
    # `dealias` and pyshp were absent from a 40 KB report.
    checks.extend(_extras_checks())
    phase("rust engines, bridges and the staged estate")
    checks.append(_rust_renderer_check())
    checks.append(_renderer_tree_check())
    checks.append(_fetch_backbone_check())
    checks.append(_nexrad_front_door_check())
    # The three front doors no bundle carries.  Absent from this report
    # through 2.3.3, which is how a box with every radar-adjacent
    # observation route dead could print a fully green estate.
    checks.extend(_obs_front_door_checks())
    checks.extend(_obs_stream_door_checks())
    # The five MPAS binaries no bundle carried and no check reported.
    checks.extend(_mpas_bridge_checks())
    checks.append(_ml_export_check())
    checks.append(_icechunk_check())
    checks.append(_grib2_export_check())
    checks.append(_gpu_post_kernels_check())
    checks.append(_verification_engine_check())
    checks.append(_comparison_engine_check())
    checks.append(_ensemble_reducer_check())
    checks.append(_nowcast_frames_check())
    checks.append(_observation_scorecard_check())
    checks.append(_forecast_scoreboard_check())
    checks.append(_simulated_radar_check())
    checks.append(_sounding_renderer_check())
    checks.append(_tc_fields_check())
    checks.extend(_earth2_runtime_checks())
    checks.append(_netcdf_decoder_check())
    checks.append(_mapped_engine_check())
    checks.append(_ncwrite_check())
    checks.append(_static_builder_check())
    checks.append(_obs_regrid_check())
    checks.append(_obs_score_check())
    checks.append(_noah_init_check())
    checks.append(_isobaric_reader_check())
    tempo_check = _tempo_library_check()
    if tempo_check is not None:
        checks.append(tempo_check)
    checks.append(_projected_intermediate_check())
    checks.extend(_nextgen_checks())
    checks.extend(_hydro_door_tool_checks())
    checks.extend(_hydro_proj_checks())
    checks.extend(_capability_tool_checks())
    checks.append(_globe_pack_check())
    checks.append(_probgrid_check())
    checks.append(_radar_superob_check())
    checks.append(_region_dealias_check())
    checks.extend(_bridge_checks())
    # ABOUT the report, not about an artifact: does every artifact the
    # bundle carries have a line above?  Twice it did not, and both
    # times doctor called the estate green.
    checks.extend(_bundle_coverage_checks())
    # After every per-artifact line, because it is the question none of
    # them answers: are these the bytes this release published?  A
    # staged estate from an earlier release passes all of them and then
    # refuses at the first preparation.
    checks.append(_staged_estate_check())
    hex_helpers = _hex_helpers_check()
    if hex_helpers is not None:
        checks.append(hex_helpers)
    # After the bridges, because it is the question the five `ok bridge`
    # lines above do NOT answer: whether the route that needs two of
    # them can actually reach them.
    checks.append(_mapped_grib2_route_check())
    checks.append(_cpu_library_check())
    phase("physics tables and static inputs")
    # Ahead of every table line, because it is the question none of them
    # answers: a skewed companion passes each validator and is still the
    # wrong tables.  A reader meets the cause before the symptom.
    checks.append(_companion_pair_check())
    checks.append(_thompson_tables_check())
    checks.append(_p3_table_check())
    checks.append(_noah_tables_check())
    checks.append(_ocean_climatology_check())
    checks.append(_arbitrary_input_check())
    checks.extend(_case_data_root_check())
    phase("install identity and provenance")
    checks.append(_distribution_manifest_check())
    checks.append(_provenance_check())
    checks.append(_install_identity_check())
    checks.append(_console_scripts_check())
    checks.append(_non_git_import_check())
    checks.append(_long_path_install_root_check())
    # Resolved ONCE for every registry route below.  Twenty-eight of
    # the routes decode on the same mapped engine, and asking that
    # binary the same question twenty-eight times would put the whole
    # expansion of `--source` on doctor's wall clock.
    engine = _mapped_engine_check()
    if selected:
        phase(f"{len(selected)} data route(s): decoder and byte transport")
    for source in selected:
        checks.extend(_source_route_checks(source, engine))
    return checks


#: Column the first remedy line starts at: ten spaces of gutter plus
#: the ``remedy: `` label itself.  Continuation lines match it.
_REMEDY_LABEL = "          remedy: "

#: Status -> the label both reports print.  One vocabulary, so a reader
#: who runs the terse form and then ``--explain`` is reading the same
#: four words in the same column.
#: ``UNTESTED`` is one character wider than its four siblings and stays
#: that way on purpose: it is the status that must not be skimmed past
#: as though it were ``present``.
_LABELS = {"verified": "ok     ", "present": "present",
           "untested": "UNTESTED", "missing": "MISSING", "info": "info   "}


def _remedy_block(remedy: str) -> list[str]:
    """The remedy, every physical line aligned under ``remedy:``.

    Only the first line used to be indented; the rest arrived with
    whatever leading whitespace the composer happened to give them (0,
    2 or 4 spaces, three different composers), so a report showed
    commands hanging at column 0 under a label at column 18 and read as
    if the block had ended.  Both shells ignore leading whitespace, so
    aligning the whole block costs the reader nothing on paste.
    """

    lines = remedy.splitlines() or [remedy]
    block = [_REMEDY_LABEL + lines[0].strip()]
    block += [" " * len(_REMEDY_LABEL) + line.strip() if line.strip() else ""
              for line in lines[1:]]
    return block


def _install_headline() -> str:
    """Which copy of woof produced this report.

    A report that diagnoses an estate has to say whose estate, and the
    version number alone does not: an editable install shadows every
    wheel pip writes, so a reader can be looking at findings from code
    that is months older than the version they think they upgraded to.
    No network here -- `woof version` owns the index comparison; this
    is the local identity only, and it never fails a report it is only
    the header of.
    """

    try:
        from woof.version_cli import headline

        return headline()
    except Exception as error:                          # noqa: BLE001
        return f"(install identity unavailable: {type(error).__name__})"


def format_report(checks: list[Check]) -> str:
    """The full report: every finding's evidence and whole remedy block.

    This is what ``woof doctor --explain`` prints, unchanged from when
    it was the only thing doctor printed.  The remedy blocks in
    particular are not summarized anywhere: they are the text a reader
    pastes, and a paraphrase of a command is not a command.
    """

    lines = ["woof doctor: runtime estate"]
    for check in checks:
        lines.append(f"  {_LABELS[check.status]} {check.name}: "
                     f"{check.detail}")
        if check.remedy:
            lines.extend(_remedy_block(check.remedy))
    gaps = sum(1 for check in checks if check.status == "missing")
    blocking = len(blocking_gaps(checks))
    opt_in = gaps - blocking
    presence_only = sum(1 for check in checks if check.status == "present")
    if gaps:
        # Not "every remedy is copy-pasteable": a remedy can be a
        # sequence of steps, and on a pip install the bridge remedy is a
        # clone-and-build rather than a single line.  Claiming one line
        # when six were printed is the kind of small lie that costs the
        # reader their trust in the other five.
        #
        # And the count is SPLIT, because the exit code splits it.  A
        # summary reading "1 gap(s)" out of a process that exits 0 is a
        # report disagreeing with itself, and a fleet node comparing
        # 1.3.1 (exit 1) with 1.4.0 (exit 0) on the identical gap text
        # reasonably read the pair as a silent regression.  The exit code
        # is the right one -- an opt-in nobody opted into is not a fault
        # -- so the severity is what has to be visible.
        lines.append(
            f"woof doctor: {gaps} gap(s), {blocking} of them blocking "
            f"(the exit code is {1 if blocking else 0})"
            + _severity_clause(checks)
            + (f"; the other {opt_in} is/are pieces this install has not "
               "staged or installed, each with its own command above"
               if opt_in else "")
            + ".  Every remedy line above is either a command to run as "
              "printed, in the order printed, or a '#' comment.")
    elif presence_only:
        lines.append(f"woof doctor: no gaps ({presence_only} check(s) "
                     "presence-only as labeled, the rest verified).")
    else:
        lines.append("woof doctor: no gaps; every check verified.")
    return "\n".join(lines)


def _fold(checks: list[Check]) -> list[tuple[Check, int]]:
    """Collapse runs of same-status, same-group, same-action checks.

    A pip install gaps every Rust artifact at once, and the untidy
    consequence is six identical ``-> woof fetch-bridges`` lines that
    read as six separate problems needing six separate fixes.  Folding
    is presentation only: the fold is keyed on the three things that
    make two lines say the same thing, and each member survives intact
    in ``--explain`` and ``--json``.

    Only ADJACENT checks fold, so the report keeps the order
    :func:`collect_checks` chose -- a group interrupted by an unrelated
    check stays two entries rather than silently reordering the report.

    :data:`_SCATTERED_FOLD_GROUPS` is the one exception, and it is
    narrow: the extras block declares aliases in packaging order, which
    interleaves them with the real extras, so a walk read three
    consecutive ``info pip extra aliases (3)`` lines as three findings
    (UX finding N24).  A group listed there folds into its FIRST
    appearance wherever its members land, because every line in it says
    the same nothing -- the alias resolves, there is no gap and no
    command.  Every member still survives by name in ``--explain`` and
    ``--json``.
    """

    folded: list[tuple[Check, int]] = []
    scattered: dict[tuple, int] = {}
    for check in checks:
        if check.group in _SCATTERED_FOLD_GROUPS:
            key = (check.group, check.status, check.action)
            index = scattered.get(key)
            if index is not None:
                first, count = folded[index]
                folded[index] = (first, count + 1)
                continue
            scattered[key] = len(folded)
            folded.append((check, 1))
            continue
        if folded and check.group is not None:
            last, count = folded[-1]
            if (last.group == check.group and last.status == check.status
                    and last.action == check.action):
                folded[-1] = (last, count + 1)
                continue
        folded.append((check, 1))
    return folded


def _setup_summary(gaps: list[Check]) -> str | None:
    """``woof setup`` as the next step, when it is genuinely shorter.

    Named only when more than one gap would be closed by it -- which is
    what a fresh install looks like from here.  A single gap already
    printed its own one command above; sending that reader to a wrapper
    that runs two steps to fix one is longer, not shorter.
    """

    wanted = {check.action for check in gaps
              if check.action in SETUP_ACTIONS}
    if len(wanted) < 2:
        return None
    return ("woof setup runs " + " then ".join(
        action for action in SETUP_ACTIONS if action in wanted)
        + " in order.")


def format_brief(checks: list[Check]) -> str:
    """The default report: one line per finding, and the next command.

    Everything :func:`format_report` prints is still true and still
    reachable; this is the same findings at the width of a glance.  The
    rule for each line is what a reader in a hurry needs: the status,
    what it is about, and -- when there is something to do -- the single
    command that does it.
    """

    lines = ["woof doctor"]
    for check, count in _fold(checks):
        name = check.name if count == 1 else f"{check.group} ({count})"
        line = f"  {_LABELS[check.status]} {name}"
        if count == 1 and check.brief:
            line += f": {check.brief}"
        if check.action:
            line += f"  -> {check.action}"
        lines.append(line)
    gaps = [check for check in checks if check.status == "missing"]
    blocking = blocking_gaps(checks)
    if gaps:
        # Say how many of them the exit code is about.  See the note in
        # format_report: "1 gap(s)" beside exit 0 is the whole of the
        # reported 1.3.1 -> 1.4.0 "regression".
        summary = (f"woof doctor: {len(gaps)} gap(s), {len(blocking)} "
                   f"of them blocking (exit "
                   f"{1 if blocking else 0})"
                   + _severity_clause(checks) + ".")
        setup = _setup_summary(gaps)
        if setup:
            summary += " " + setup
        lines.append(summary)
        lines.append("Run `woof doctor --explain` for the full remedy "
                     "for each line, with the evidence behind it.")
    else:
        lines.append("woof doctor: no gaps.")
    return "\n".join(lines)


def blocking_gaps(checks: list[Check]) -> list[Check]:
    """The gaps that justify a nonzero exit.  THE severity rule, once.

    A finding is one of four severities, and exactly two of them fail
    the process:

    ``broken`` (blocks)
        Something is present and wrong: a truncated executable, a table
        that fails its hash, a manifest artifact that fails
        revalidation, an installed package that will not import, a CuPy
        wheel whose cuBLAS will not load on this box.  Nobody chose
        this, and nothing downstream can work around it.

    ``unreachable`` (blocks)
        A command the product's own README or FIRST-LIGHT tells a user
        to run cannot run at all on this box, and no documented
        fallback covers it.  CuPy absent is the type case: ``woof
        run`` is step 5 of First light, and without CuPy it dies in a
        raw ``ModuleNotFoundError`` after fetching gigabytes.  The
        ``[render]`` extra absent is the other: ``woof enprod`` and
        ``woof render --engine matplotlib`` are both README doors and
        both import the ``wrf`` package.

    ``degraded`` (does not block)
        An optional mode, engine or door is unavailable while a
        DOCUMENTED default still works: scipy's ``vad-region``
        dealiasing beside the shipped ``region-global`` engine, the
        rust fetch backbone beside the Python transport, cuSOLVER
        beside the bundled Jacobi kernel.

    ``opt-in`` (does not block)
        An explicit choice the operator has not made: the ~16 GB
        WPS_GEOG download, the ``dev`` extra.

    Two boundary cases are worth stating because they were argued.  A
    door that is genuinely unreachable but lives only in the CLI
    reference or the examples -- ``woof-mapped-inspect``, ``python -m
    tools.obs_fetch_mrms`` -- is reported ``unreachable`` and does NOT
    block: the exit code tracks the paths the product's own front page
    promises, and widening it to every module door would fail every
    install for routes most users never open.  And a ``present`` or
    ``untested`` finding never blocks, because neither is a claim that
    something is wrong.

    What changed on 2026-08-14, and why: CuPy's absence used to carry
    ``blocking=False``.  A bare install therefore printed ``3 gap(s), 0
    of them blocking (exit 0)`` -- a green light over a box that could
    not run a forecast -- and ``woof run-plan --probe`` inherited it
    as ``"ready": true``.  An installer script that trusts an exit code
    shipped that box.  The 1.3.1 -> 1.4.0 ``woof setup`` regression
    this function was originally written for is still handled, by the
    ``opt-in`` severity: an unfetched WPS_GEOG tree still exits 0.
    """

    return [check for check in checks
            if check.status == "missing" and check.blocking]


def severity_census(checks: list[Check]) -> dict[str, int]:
    """How many gaps of each severity, in the report's own vocabulary.

    Printed in the summary because "how many" and "how bad" are
    different questions, and a single number answering both is what let
    ``0 of them blocking`` sit over a broken box.
    """

    census: dict[str, int] = {}
    for check in checks:
        # Every gap carries a severity: Check.__post_init__ translates
        # a legacy `blocking` flag into one, so there is no unclassified
        # bucket to hide a finding in.
        if check.status != "missing" or check.severity is None:
            continue
        census[check.severity] = census.get(check.severity, 0) + 1
    return census


def _severity_clause(checks: list[Check]) -> str:
    """``" -- 1 unreachable, 2 opt-in"``, or ``""`` when there are none."""

    census = severity_census(checks)
    ordered = [severity for severity in
               (SEVERITY_BROKEN, SEVERITY_UNREACHABLE, SEVERITY_DEGRADED,
                SEVERITY_OPT_IN) if census.get(severity)]
    if not ordered:
        return ""
    return " -- " + ", ".join(f"{census[severity]} {severity}"
                              for severity in ordered)


#: Where doctor remembers which version it last reported on, so that
#: the NEXT run after an upgrade can say what moved.  Beside the bridges
#: and the staged tables, in the directory this product already owns;
#: the environment variable exists so a test -- and an operator with a
#: read-only home -- can put it somewhere else.
DOCTOR_STATE_ENV = "WOOF_DOCTOR_STATE"

#: Schema of that file.  One key today; named so a later key can be
#: added without a reader's file being unreadable.
DOCTOR_STATE_SCHEMA = "gpuwm-doctor-state-v1"


def state_path() -> Path | None:
    """The state file's path, or ``None`` when there is nowhere to put it."""

    override = os.environ.get(DOCTOR_STATE_ENV)
    if override:
        return Path(override)
    try:
        return Path.home() / ".woof" / "doctor-state.json"
    except (RuntimeError, OSError):              # pragma: no cover - no home
        return None


def upgrade_note(installed: str | None, path: Path | None = None
                 ) -> str | None:
    """What changed since the version doctor last reported here.

    Returns the block to print, or ``None`` -- for a first run (nothing
    recorded is not a change), for an unchanged version, for a
    downgrade, and for any release with nothing user-visible to say.
    Either way the current version is recorded, so the note is said
    ONCE: an upgrade announcement that repeats at every doctor run is
    noise the reader learns to skip past, which is how the announcement
    stops working.

    Never raises.  This is a courtesy line on a diagnostic command, and
    a state file that cannot be read or written must not cost a reader
    the report they actually asked for.
    """

    from woof import whats_changed

    path = state_path() if path is None else Path(path)
    if path is None or not installed:
        return None
    previous = None
    try:
        if path.is_file():
            recorded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(recorded, dict):
                previous = recorded.get("version")
    except (OSError, ValueError):
        previous = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema": DOCTOR_STATE_SCHEMA,
                        "version": str(installed)}, indent=2) + "\n",
            encoding="utf-8")
    except OSError:                              # pragma: no cover - read-only
        pass
    lines = whats_changed.since(previous, str(installed))
    if not lines:
        return None
    block = [f"woof doctor: this install is {installed}; the last report "
             f"here was from {previous}.  What changed for you:"]
    block += [f"  * {line}" for line in lines]
    block.append("  (said once -- this install is now the recorded one.)")
    return "\n".join(block)


def _progress_to_stderr(phase: str) -> None:
    """Narrate one phase on stderr, flushed, prefixed by the command.

    stderr and not stdout, for the reason the report's own header is
    printed outside :func:`format_report`: the two report layers are
    pinned verbatim by golden tests and ``--json`` is a machine
    surface.  Progress belongs beside the report, never inside it, and
    stderr is where a reader watching a terminal sees it and a pipeline
    reading the document does not.

    Flushed on every line because block buffering is what made this
    silent in the first place: a redirected stream holds narration
    until exit, which is the same wall the reader was already staring
    at.
    """

    print(f"woof doctor: checking {phase}...", file=sys.stderr,
          flush=True)


def upgrade_note_since(previous: str, installed: str | None = None
                       ) -> str | None:
    """The upgrade note for a reader coming from ``previous``, on demand.

    ``woof doctor --since 2.6.5``: the same lines :func:`upgrade_note`
    prints once after an upgrade, without reading or writing the state
    file, so a reader who missed the one-time note (or is deciding
    whether to upgrade a second machine) can ask for it again.  ``None``
    when nothing recorded lies in ``(previous, installed]``.
    """

    from woof import whats_changed

    if installed is None:
        from woof.version_cli import install_shape

        installed = install_shape().get("version")
    if not installed:
        # A bare source tree has no distribution to report a version
        # from; the code's own declaration (pyproject.toml) is what the
        # provenance receipt already reads for exactly this case.
        try:
            from woof.provenance import resolve

            installed = resolve().code_version
        except Exception:                               # noqa: BLE001
            installed = None
    if not installed:
        return None
    lines = whats_changed.since(previous, str(installed))
    if not lines:
        return None
    block = [f"woof doctor: this install is {installed}; what changed "
             f"since {previous}:"]
    block += [f"  * {line}" for line in lines]
    return "\n".join(block)


def doctor_main(args) -> int:
    since = getattr(args, "since", None)
    if since:
        # A query, not a diagnosis: print the release notes the reader
        # asked for and stop.  Running the estate checks behind a question
        # about release notes would cost a minute and hide the answer
        # under the report.
        note = upgrade_note_since(str(since))
        print(note if note is not None
              else f"woof doctor: nothing recorded as changed since {since}.")
        return 0
    sources = getattr(args, "source", None) or None
    from woof.forecast_gpu import FORECAST_GPU_REQUIREMENT

    # Keep driver contact isolated, like the deeper runtime checks below:
    # a broken driver must not wedge the diagnostic's own interpreter.
    availability = _json_probe(
        "import json; from woof.forecast_gpu import forecast_gpu_gap; "
        "print(json.dumps({'gap': forecast_gpu_gap()}))")
    gap = availability.get('gap')
    if 'gap' not in availability:
        gap = f"GPU availability probe did not complete ({availability.get('probe', 'no answer')})"
    if gap is not None:
        # stderr preserves --json's machine surface. Print before inventory
        # so a no-device install gets the forecast/CPU boundary first.
        print(f"woof doctor: ForecastGPURequired: {FORECAST_GPU_REQUIREMENT} "
              f"Detected: {gap}.", file=sys.stderr, flush=True)
    checks = collect_checks(tuple(sources) if sources else None,
                            progress=_progress_to_stderr)
    if getattr(args, "json", False):
        print(json.dumps(
            [check.__dict__ for check in checks], indent=2))
        return 1 if blocking_gaps(checks) else 0
    # WHICH copy of woof produced this report.  Printed here rather
    # than folded into format_report/format_brief because those two are
    # pinned verbatim by a golden test, and rightly: --explain's promise
    # is that the long form comes back unchanged.  A header is a
    # property of the printed command, not of the report text.
    #
    # It goes FIRST because it reframes everything under it: an estate
    # diagnosed from an editable install months behind the version the
    # reader thinks they upgraded to is a different report.
    print(_install_headline())
    # AFTER the identity line, because the note is about that install,
    # and BEFORE the findings, because "the run folder moved" reframes
    # every path in the report under it.  Only on the human path: the
    # JSON door is a machine surface and gets no prose, and leaving its
    # state alone keeps the note for the reader who will read it.
    note = _upgrade_note_for_this_install()
    if note is not None:
        print(note)
    print(format_report(checks) if explain_enabled(args)
          else format_brief(checks))
    return 1 if blocking_gaps(checks) else 0


def _upgrade_note_for_this_install() -> str | None:
    """:func:`upgrade_note` for whatever woof is running.  Never fails."""

    try:
        from woof.version_cli import install_shape

        version = install_shape().get("version")
        if not version:
            from woof.provenance import resolve

            version = resolve().code_version
        return upgrade_note(version)
    except Exception:                                   # noqa: BLE001
        return None


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="verify the runtime estate for real (a subprocess import of "
             "every package this install's own metadata declares, base "
             "and extras alike; probe executions of every bridge, front "
             "door and render engine; a ctypes load of the CPU library; "
             "table hash/parse validation; sealed-manifest re-hashing; "
             "each data route's decoder and byte transport; WPS_GEOG "
             "index files) and print one line per item with the command "
             "that closes each gap (--explain for the full remedies).  "
             "Exits 1 when a finding is broken or unreachable, 0 when "
             "the only gaps are degraded or opt-in")
    parser.add_argument("--json", action="store_true",
                        help="emit the checks as JSON")
    parser.add_argument(
        "--since", metavar="VERSION",
        help="print what changed for an existing user between VERSION and "
             "this install (the results that move on a bare configuration, "
             "and the checkpoints and namelists that stop loading), then "
             "exit 0 without running the estate checks.  The same note is "
             "printed once, automatically, on the first doctor run after "
             "an upgrade")
    parser.add_argument(
        "--source", action="append", choices=_doctor_sources(),
        metavar="SOURCE",
        help="report only this data route's own resolution (repeatable) "
             "alongside the shared estate: what its preparation will "
             "decode with, and the byte transport its fetch will use.  "
             "The choices are the source registry -- the same list "
             "`woof fetch` and `woof prep` take.  Omitted, every route "
             "this build knows is reported")
    parser.set_defaults(func=doctor_main)
    return parser


def main(argv: list[str] | None = None) -> int:
    """``python -m woof.doctor``: the same report, the same exit code.

    This module had no ``__main__`` guard, so the module door imported
    it and exited 0 having printed nothing -- and the reader who typed
    it was, by definition, someone whose install was already confusing
    them.  A silent 0 from a diagnostic is the worst answer available:
    it reads as "no gaps".

    It builds its surface from THE registrar the console script uses --
    :func:`register_cli` for the flags this subcommand owns, and
    :func:`woof.explain.add_explain_flag` for the one
    :func:`woof.cli.build_parser` puts on every subcommand -- so the
    two doors cannot drift into taking different arguments.  It does not
    import :mod:`woof.cli`, and that is deliberate rather than
    incidental: the standalone RW-WPS distribution stages this module
    and does NOT stage the CLI, so a door that reached for it would
    answer a reader on that wheel with an ImportError traceback, which
    is the same silence in a louder font.
    """

    import argparse

    from woof.explain import add_explain_flag, explain_scope

    holder = argparse.ArgumentParser(add_help=False)
    parser = register_cli(holder.add_subparsers())
    add_explain_flag(parser)
    parser.prog = "python -m woof.doctor"
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    # The same bounded lifetime `woof.cli.main` gives the flag: library
    # code that warns reads it from module state, and leaving it set
    # would hand the next invocation in this interpreter an explanation
    # layer nobody asked for.
    with explain_scope(explain_enabled(args)):
        return doctor_main(args)


def __getattr__(name: str):
    """``doctor.DOCTOR_SOURCES``, resolved on first read and cached.

    PEP 562, for one name and one reason: the registry that answers it
    imports numpy, and this module has a pinned promise that it imports
    on an interpreter with no scientific stack -- because a reader with
    a broken install runs `woof doctor` FIRST, and a diagnostic that
    cannot import on the estate it diagnoses is worse than no
    diagnostic.  Callers inside this file use ``_doctor_sources()``;
    a module-level ``__getattr__`` does not answer a module's own
    global lookups.
    """

    if name == "DOCTOR_SOURCES":
        value = _doctor_sources()
        globals()["DOCTOR_SOURCES"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Check", "DOCTOR_SOURCES", "DOCTOR_STATE_ENV",  # noqa: F822 -- module __getattr__
           "DOCTOR_STATE_SCHEMA",
           "SETUP_ACTIONS", "SEVERITY_BROKEN", "SEVERITY_DEGRADED",
           "SEVERITY_OPT_IN", "SEVERITY_UNREACHABLE", "blocking_gaps",
           "collect_checks", "declared_requirements", "doctor_main",
           "format_brief", "format_report", "geography_gaps", "main",
           "register_cli", "severity_census", "state_path",
           "upgrade_note", "upgrade_note_since"]


if __name__ == "__main__":                    # pragma: no cover - the door
    raise SystemExit(main())
