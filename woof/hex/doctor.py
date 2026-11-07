"""``woof hex doctor``: what this install can actually reach, and what to run.

A wheel for this project is deliberately partial, and saying so is the
point.  The Python driver, the front doors and the tables ship in it.
The things that do the work do not:

* the Rust engines (``rw_mpas_init``, ``rw_mpas_convert``,
  ``rw_wrfbatch``) are built from the woof ``tools/rustwx`` workspace
  and staged onto a machine -- ``woof fetch-bridges`` is the one
  command that does it;
* the CUDA lane needs a CuPy wheel whose CUDA major the box's driver
  serves (CUDA 12 or 13; the GPU doors refuse a runtime below 12000), and
  no pip extra can check what the driver actually serves, so this module
  reads it and says;
* the card itself: every architecture this install's NVRTC compiles for
  runs, anchored or not, and this module names the card, its compute
  capability and its anchor status, or the one refusal a card can meet;
* the meshes and their static fields are external assets with no fetch
  path in this distribution;
* the forecast runs on the installed woof's sixteen pinned seam files,
  which every engine from 2.5.8 carries in site-packages (measured
  2026-08-28 against the wheel PyPI serves: checked=16, matched=16,
  moved=(), absent=()); an older engine lacks ``docs/mpas-seam.md`` and
  cannot satisfy the pin on its own.

Every one of those is a place a fresh install meets a wall.  This module
exists so the wall has a sign on it: each gap prints THE command that
closes it, on this platform, spelled as it is typed.  A bare
``ImportError`` three commands later is the failure mode being replaced.

Statuses.  ``verified`` means the deep check ran and passed -- the module
imported in a short-lived subprocess, the binary was found on a named
rung.  ``present`` is for what can only be checked by existence.
``missing`` is a gap, and a gap always carries a remedy.  ``info`` is
context that is never a gap.

Exit status is 1 when any REQUIRED finding is missing, 0 otherwise.
Required means: a door this distribution advertises cannot open, or the
install itself is wrong.  The three Python dependencies and the three
engines are required in the first sense; an installed woof whose seam
bytes are not the ones this port pins is required in the second, and that
clause is new.  CuPy, the card's architecture, the seam-source note and
the mesh assets are reported and do not fail the process, because a user
who only wants the
render door should not be told the install is broken by the absence of a
mesh.

THE BREAKAGE THE SECOND CLAUSE PREVENTS, measured on a real install on
2026-08-27 (``evidence/userwalk-20260827/``): ``pip install woof hex``
resolved gpuwm 2.5.7, whose bytes the seam pin refuses; the forecast door
then stopped with two SHA-256 digests and no version number, and THIS
REPORT said ``Every check passed`` and exited 0 -- because its woof line
compared a version against a floor and never looked at a byte.  Fifteen of
the sixteen pinned paths are inside site-packages, so the drift was
visible here, at install time, without a card and without a checkout.
Nothing looked.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from . import DISTRIBUTION_NAME
from . import engines


#: How long a single import probe may take before it is called wedged.
_PROBE_TIMEOUT_S = 90

VERIFIED = "verified"
PRESENT = "present"
MISSING = "missing"
INFO = "info"
#: Works, but slower than it should: never blocking, never "Every check passed".
WARN = "warn"


@dataclass
class Finding:
    """One checked thing: what it is, what was found, what to run."""

    subject: str
    status: str
    detail: str
    #: Commands that close the gap.  Every line is a command as typed or a
    #: ``#`` comment -- never prose fused onto a command.
    remedy: str = ""
    #: Whether a gap here closes every front door.
    required: bool = False
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
            "required": self.required,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def _import_probe(module: str) -> tuple[bool, str]:
    """Import ``module`` in a short-lived subprocess and read its version.

    A subprocess rather than an import here for two reasons that have
    both been measured on this stack: a broken native dependency (a
    netCDF4 or CuPy whose shared libraries do not load) can take the
    interpreter down rather than raise, and a partially initialised
    module poisons every later check in the same process.  The report
    must survive the thing it is reporting on.
    """

    code = (
        "import importlib, json, sys\n"
        f"m = importlib.import_module({module!r})\n"
        "print(json.dumps({'version': getattr(m, '__version__', None),"
        " 'file': getattr(m, '__file__', None)}))\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"the import probe could not run: {error}"
    if probe.returncode != 0:
        tail = (probe.stderr or "").strip().splitlines()
        return False, tail[-1] if tail else f"import exited {probe.returncode}"
    try:
        facts = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return True, "imported (version not reported)"
    version = facts.get("version") or "version not reported"
    return True, f"imported, {version}"


def _distribution_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as distribution_version

    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def check_interpreter() -> list[Finding]:
    from . import __version__

    return [
        Finding(
            subject="distribution",
            status=INFO,
            detail=f"{DISTRIBUTION_NAME} {__version__}",
            evidence={
                "import_package": __package__ or "woof.hex",
                "location": str(Path(__file__).resolve().parent),
            },
        ),
        Finding(
            subject="interpreter",
            status=INFO,
            detail=(
                f"Python {platform.python_version()} on "
                f"{platform.system()} {platform.machine()}"
            ),
            evidence={"executable": sys.executable},
        ),
    ]


#: Module -> the distribution that installs it, where they differ.
_REQUIRED_MODULES = (
    ("numpy", "numpy", "arrays; 48 modules import it at line one"),
    ("netCDF4", "netCDF4", "reads and writes every mesh, static and history file"),
    ("scipy", "scipy", "the regridder's spatial index"),
)


def check_python_dependencies() -> list[Finding]:
    findings: list[Finding] = []
    for module, distribution, why in _REQUIRED_MODULES:
        ok, detail = _import_probe(module)
        findings.append(
            Finding(
                subject=f"{module} ({why})",
                status=VERIFIED if ok else MISSING,
                detail=detail,
                remedy="" if ok else f"  pip install {distribution}",
                required=True,
            )
        )
    return findings


#: CUDA major -> the pip extra that installs its CuPy wheel.  The same table
#: ``woof.doctor`` carries, and deliberately the same: a user who runs both
#: doctors on one box must not be told two different things.
_GPU_EXTRA_BY_MAJOR = {12: "gpu-cu12", 13: "gpu-cu13"}


def cuda_runtime_floor() -> int | None:
    """The CUDA runtime version every GPU door here requires, READ not restated.

    Taken off :func:`woof.hex.cuda_backend.runtime.require_cuda`'s own
    default, because that function is the thing that refuses and a second
    copy of the number in this file is a second thing to keep true.  The
    number this reads is the one the refusal quotes.

    ``None`` when the port's own CUDA module cannot be imported at all,
    which is a broken install rather than a missing card; the caller says so
    rather than guessing a floor.
    """

    try:
        from inspect import signature

        from .cuda_backend.runtime import require_cuda

        default = signature(require_cuda).parameters["min_runtime_version"].default
        return int(default)
    except Exception:  # pragma: no cover - a broken install
        return None


def _driver_library_names() -> tuple[str, ...]:
    """The CUDA driver library this platform would load, by name."""

    if sys.platform == "win32":
        return ("nvcuda.dll",)
    if sys.platform == "darwin":
        return ()
    return ("libcuda.so.1", "libcuda.so")


def _no_local_gpu() -> bool:
    for variable in ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(variable, "") not in ("", "0"):
            return True
    return False


def _driver_cuda_major() -> int | None:
    """The CUDA major this box's DRIVER serves, or ``None`` if unknown.

    Read with ``ctypes`` straight off the driver library rather than through
    CuPy, because the case that needs the answer most is the box that has NO
    CuPy yet -- which is exactly where every CuPy-based probe is by
    definition unavailable.  ``cuDriverGetVersion`` is the one entry point
    that answers without ``cuInit``: no context, no device opened, nothing
    that could disturb a card another process is holding.  It is still
    driver contact, so the no-local-GPU declaration suppresses it.

    Ported from ``woof.doctor._driver_cuda_major``, which had already
    solved this on the same boxes.  Every failure is a ``None``: a machine
    with no NVIDIA driver is the ordinary case here, not an error.
    """

    if _no_local_gpu():
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


def _cupy_runtime_probe() -> dict[str, object] | None:
    """CuPy's CUDA RUNTIME version and wheel name, in a short-lived process.

    Separate from :func:`_import_probe` because importing cleanly is the
    thing that misled everybody: a CuPy built for the wrong CUDA major
    imports, reports a version, allocates, and runs cuBLAS.  What tells the
    two apart is ``runtimeGetVersion``, and nothing was reading it.

    A subprocess for the same reason every probe here uses one -- a CuPy
    whose shared libraries do not load takes the interpreter down rather
    than raising.
    """

    code = (
        "import json, sys\n"
        "out = {}\n"
        "try:\n"
        "    import cupy\n"
        "    out['cupy_version'] = getattr(cupy, '__version__', None)\n"
        "    out['runtime_version'] = int(cupy.cuda.runtime.runtimeGetVersion())\n"
        "except Exception as error:\n"
        "    out['error'] = f'{type(error).__name__}: {error}'\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        document = json.loads((probe.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def _installed_cupy_wheels() -> list[str]:
    """Every installed CuPy distribution, by the name pip would uninstall."""

    from importlib import metadata

    names: list[str] = []
    try:
        distributions = list(metadata.distributions())
    except Exception:  # pragma: no cover - depends on the environment
        return names
    for distribution in distributions:
        try:
            name = distribution.metadata["Name"]
        except Exception:  # pragma: no cover - a malformed dist-info
            continue
        if name and str(name).lower().startswith("cupy"):
            names.append(str(name))
    return sorted(set(names))


def check_gpu_runtime() -> list[Finding]:
    from woof.gpu_smi import is_amd
    if is_amd() or os.environ.get('WOOF_BACKEND')=='hip-nvidia':
        from woof.hip_doctor import run
        report=run()
        return [Finding('HIP runtime', VERIFIED if report['ok'] else MISSING,
            f"Experimental AMD; ROCm {report['rocm_version']}; gfx {report['gfx_target']}; wave {report['wave_size']}; kernel smoke {report['kernel_smoke']}",
            'woof hex doctor --backend amd',
            required=True,evidence=report)]
    """CuPy, and the CUDA major this distribution's own runtime floor requires.

    THE DEFECT THIS REPLACES, and it is worth stating because it cost a
    measured cross-machine walk (``evidence/xmachine-20260827`` section 5,
    ``evidence/userwalk-20260827`` section 4.2).  The old check asked one
    question -- does ``import cupy`` work -- and printed a remedy naming both
    extras with a comment telling the user to match their driver's CUDA
    major.  ``render``'s compact view, which is what ``doctor`` prints by
    default, filters comment lines out and keeps the FIRST command, so the
    one line a user actually saw was ``pip install "recast-woof[gpu-cu12]"``.
    Under the 13000 floor of the time, that installed a CuPy every GPU door
    refused: ``cuda.runtime_version=12090 < required 13000``.  Doctor then
    reported ``verified`` on that same CuPy, because it imported.

    So this check reads three things instead of one: the runtime floor
    ``require_cuda`` will enforce, the CUDA major the DRIVER serves, and --
    when a CuPy is installed -- the CUDA runtime that CuPy actually carries.
    A CuPy the floor refuses is reported as a GAP with the exact refusal the
    forecast door would raise, before a card is ever opened, which is the
    difference between failing early by name and failing at the first real
    device call.

    THE SECOND DEFECT, measured on an RTX PRO 6000 whose driver serves CUDA
    12.8: ``[gpu]`` then installed cupy-cuda13x, which imports and reports
    runtime 13020 -- above the floor -- so this check said ``verified`` and
    doctor printed "Every check passed", while every device call on that box
    stopped with ``cudaErrorInsufficientDriver``.  A CuPy whose CUDA major is
    newer than the driver's is now a GAP naming that error, and every remedy
    names the extra for the major the driver serves.
    """

    floor = cuda_runtime_floor()
    floor_major = None if floor is None else floor // 1000
    box_major = _driver_cuda_major()
    # The extra for the major this driver serves; with the driver unread, the
    # `gpu` alias, whose CUDA 12 wheel opens a card on a CUDA 12 or 13 driver.
    if (
        box_major is not None
        and floor_major is not None
        and box_major >= floor_major
        and box_major in _GPU_EXTRA_BY_MAJOR
    ):
        matching = _GPU_EXTRA_BY_MAJOR[box_major]
    else:
        matching = "gpu"
    evidence: dict[str, object] = {
        "required_runtime_version": "unreadable" if floor is None else floor,
        "driver_cuda_major": "unknown" if box_major is None else box_major,
    }

    ok, detail = _import_probe("cupy")
    if not ok:
        if floor is None:
            return [
                Finding(
                    subject="cupy (the CUDA lane)",
                    status=MISSING,
                    detail=detail,
                    remedy=(
                        "  # this install could not import its own CUDA runtime\n"
                        "  # module, so the required CUDA major is unreadable\n"
                        f'  pip install --force-reinstall "{DISTRIBUTION_NAME}"'
                    ),
                    evidence=evidence,
                )
            ]
        if box_major is not None and box_major < floor_major:
            return [
                Finding(
                    subject="cupy (the CUDA lane)",
                    status=MISSING,
                    detail=(
                        f"this box's driver serves CUDA {box_major} and every "
                        f"GPU door here requires runtime {floor}: no CuPy "
                        f"wheel closes that. "
                    )
                    + detail,
                    remedy=(
                        f"  # NOT a pip problem.  cupy-cuda{floor_major}x needs a "
                        f"driver serving\n"
                        f"  # CUDA {floor_major}; this one serves CUDA {box_major}.  "
                        f"Update the NVIDIA driver,\n"
                        f"  # or run the CUDA lane on a machine that has one.  "
                        f"Installing\n"
                        f"  # cupy-cuda{box_major}x instead would import, probe "
                        f"clean, and then be\n"
                        f"  # refused by name: cuda.runtime_version < required "
                        f"{floor}."
                    ),
                    evidence=evidence,
                )
            ]
        install = f'  pip install "{DISTRIBUTION_NAME}[{matching}]"'
        if box_major is not None:
            why = (
                f"  # this box's driver serves CUDA {box_major}, so the CUDA-"
                f"{box_major} wheel\n"
            )
        else:
            why = (
                "  # the driver's CUDA major was not read; the CUDA 12 wheel "
                "runs on a\n  # CUDA 12 or CUDA 13 driver\n"
            )
        return [
            Finding(
                subject="cupy (the CUDA lane)",
                status=MISSING,
                detail=detail,
                remedy=why + install,
                evidence=evidence,
            )
        ]

    probe = _cupy_runtime_probe() or {}
    runtime = probe.get("runtime_version")
    evidence["cupy_runtime_version"] = (
        "unreadable" if not isinstance(runtime, int) else runtime
    )
    if floor is None or not isinstance(runtime, int):
        return [
            Finding(
                subject="cupy (the CUDA lane)",
                status=PRESENT,
                detail=detail,
                evidence=evidence,
            )
        ]
    if runtime < floor:
        wheels = _installed_cupy_wheels()
        evidence["installed_cupy_distributions"] = ", ".join(wheels) or "unknown"
        removal = (
            f"  pip uninstall -y {' '.join(wheels)}"
            if wheels
            else f"  pip uninstall -y cupy-cuda{runtime // 1000}x"
        )
        return [
            Finding(
                subject="cupy (the CUDA lane)",
                status=MISSING,
                detail=(
                    f"cupy imports but carries CUDA runtime {runtime}; every "
                    f"GPU door here refuses below {floor}"
                ),
                remedy=(
                    f"  # the forecast door will raise, by name:\n"
                    f"  #   CudaRefusal: cuda.runtime_version={runtime} < "
                    f"required {floor}\n"
                    f"  # remove the wrong-major wheel FIRST -- pip will leave "
                    f"both\n"
                    f"  # installed and import order, not intent, picks the "
                    f"winner\n"
                    f"{removal}\n"
                    f'  pip install "{DISTRIBUTION_NAME}[{matching}]"'
                ),
                evidence=evidence,
            )
        ]
    if box_major is not None and runtime // 1000 > box_major:
        wheels = _installed_cupy_wheels()
        evidence["installed_cupy_distributions"] = ", ".join(wheels) or "unknown"
        removal = (
            f"  pip uninstall -y {' '.join(wheels)}"
            if wheels
            else f"  pip uninstall -y cupy-cuda{runtime // 1000}x"
        )
        return [
            Finding(
                subject="cupy (the CUDA lane)",
                status=MISSING,
                detail=(
                    f"cupy imports but carries CUDA runtime {runtime}, and this "
                    f"box's driver serves CUDA {box_major}: every device call "
                    f"stops with cudaErrorInsufficientDriver"
                ),
                remedy=(
                    f"  # the wheel's CUDA major is newer than the driver's; "
                    f"install the\n"
                    f"  # wheel for CUDA {box_major}, removing this one FIRST -- "
                    f"pip leaves both\n"
                    f"  # installed and import order, not intent, picks the "
                    f"winner\n"
                    f"{removal}\n"
                    f'  pip install "{DISTRIBUTION_NAME}[{matching}]"'
                ),
                evidence=evidence,
            )
        ]
    return [
        Finding(
            subject="cupy (the CUDA lane)",
            status=VERIFIED,
            detail=f"{detail}; CUDA runtime {runtime}, at or above {floor}",
            evidence=evidence,
        )
    ]


_ARCHITECTURE_SUBJECT = "GPU architecture (the card and NVRTC)"


def _card_probe() -> dict[str, object] | None:
    """The card's name and compute capability, and NVRTC's target list.

    Read in a short-lived process, for the reason every probe here uses
    one.  It reads device 0's properties and asks NVRTC which targets it
    generates code for, through the same function the forecast door asks
    (``arch_admission.nvrtc_target_archs``).  It compiles nothing and
    launches nothing.  ``None`` when the probe itself could not run.
    """

    code = (
        "import json, sys\n"
        "out = {}\n"
        "try:\n"
        "    import cupy\n"
        "    if cupy.cuda.runtime.is_hip:\n"
        "        raise RuntimeError('CUDA is the default; select --backend amd for experimental ROCm checks')\n"
        "    out['device_count'] = int(cupy.cuda.runtime.getDeviceCount())\n"
        "    if out['device_count'] > 0:\n"
        "        p = cupy.cuda.runtime.getDeviceProperties(0)\n"
        "        name = p['name']\n"
        "        if isinstance(name, bytes):\n"
        "            name = name.decode('utf-8', 'replace')\n"
        "        out['name'] = str(name)\n"
        "        out['compute'] = [int(p['major']), int(p['minor'])]\n"
        "except Exception as error:\n"
        "    out['error'] = f'{type(error).__name__}: {error}'\n"
        "if 'compute' in out:\n"
        "    try:\n"
        f"        from {__package__}.cuda_backend.arch_admission import nvrtc_target_archs\n"
        "        out['nvrtc_targets'] = list(nvrtc_target_archs())\n"
        "    except Exception as error:\n"
        "        out['nvrtc_error'] = f'{type(error).__name__}: {error}'\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        document = json.loads((probe.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def check_gpu_architecture() -> list[Finding]:
    """The card: its name, compute capability and anchor status.

    Every architecture NVRTC can compile the kernels for runs this port by
    default (:mod:`woof.hex.cuda_backend.arch_admission`).  So an unanchored
    card is never a gap here: it is reported with its status, as the
    forecast door prints it and every receipt records it.  The one gap is
    the forecast door's own NVRTC refusal, a card whose target this
    install's NVRTC does not list (or an NVRTC that cannot be asked at all),
    because on it no kernel could be built.  A card that cannot be read at
    all is context, not a gap: the CuPy finding above says why.
    """

    from woof.hip_backend import active
    if active():
        return []  # The HIP runtime finding includes gfx target and wave size.
    from .cuda_backend.arch_admission import architecture_status, nvrtc_refusal

    if _no_local_gpu():
        return [
            Finding(
                subject=_ARCHITECTURE_SUBJECT,
                status=INFO,
                detail=(
                    "not read: this box declares that no GPU work happens on "
                    "it (WOOF_HEX_NO_LOCAL_GPU / GPUWM_NO_LOCAL_GPU)"
                ),
            )
        ]
    probe = _card_probe()
    if probe is None or "compute" not in probe:
        if probe is None:
            why = "the card probe could not run"
        elif probe.get("error"):
            why = str(probe["error"])
        else:
            why = "no CUDA device is visible to this process"
        return [
            Finding(
                subject=_ARCHITECTURE_SUBJECT,
                status=INFO,
                detail=f"no card read: {why}",
                evidence={} if probe is None else dict(probe),
            )
        ]
    compute = (int(probe["compute"][0]), int(probe["compute"][1]))
    name = str(probe.get("name") or "unnamed device")
    status = architecture_status(compute)
    evidence: dict[str, object] = dict(status.as_dict())
    evidence["name"] = name
    card = (
        f"{name}, compute capability {compute[0]}.{compute[1]} ({status.sm})"
    )
    floor = cuda_runtime_floor()
    floor_major = None if floor is None else floor // 1000
    if "nvrtc_targets" not in probe:
        error = probe.get("nvrtc_error", "no answer")
        evidence["nvrtc_error"] = error
        extra = _GPU_EXTRA_BY_MAJOR.get(floor_major) if floor_major else None
        remedy = (
            f"  # NVRTC could not be asked which targets it compiles for, and\n"
            f"  # every kernel is compiled by NVRTC when the run starts; the\n"
            f"  # CUDA extra installs CuPy with the NVRTC it loads\n"
        )
        if extra:
            remedy += f'  pip install "{DISTRIBUTION_NAME}[{extra}]"'
        return [
            Finding(
                subject=_ARCHITECTURE_SUBJECT,
                status=MISSING,
                detail=(
                    f"{card}: NVRTC could not be asked which targets it "
                    f"compiles for ({error}), so no kernel could be built on "
                    f"this install"
                ),
                remedy=remedy.rstrip("\n"),
                evidence=evidence,
            )
        ]
    targets = tuple(int(value) for value in probe["nvrtc_targets"])
    evidence["nvrtc_targets"] = [f"compute_{value}" for value in targets]
    target = compute[0] * 10 + compute[1]
    if target not in targets:
        if targets and target > max(targets):
            newer = f"CUDA {floor_major} " if floor_major else ""
            remedy = (
                f"  # this NVRTC generates code up to compute_{max(targets)}, "
                f"older than the card;\n"
                f"  # a newer {newer}NVRTC lists compute_{target}\n"
                f"  pip install --upgrade nvidia-cuda-nvrtc"
            )
        else:
            remedy = (
                f"  # NOT a pip problem.  The CUDA toolkit this port requires "
                f"generates no\n"
                f"  # code for compute_{target}; run the CUDA lane on a card "
                f"it lists"
            )
        return [
            Finding(
                subject=_ARCHITECTURE_SUBJECT,
                status=MISSING,
                detail=f"{name}: {nvrtc_refusal(compute, targets)}",
                remedy=remedy,
                evidence=evidence,
            )
        ]
    detail = f"{card}: admitted, {status.describe()}"
    if not status.anchored:
        detail += (
            "; the forecast door measures the numeric route on this card "
            "before a run"
        )
    return [
        Finding(
            subject=_ARCHITECTURE_SUBJECT,
            status=VERIFIED,
            detail=detail,
            evidence=evidence,
        )
    ]


#: The engine range, DERIVED rather than restated.  ``engine_pin`` computes
#: it from a table of measured verdicts; the version numbers live there and
#: nowhere else, because this constant used to be a second copy of a number
#: the declaration also carried, and copies drift.
def _requirement() -> str:
    from . import engine_pin

    return engine_pin.gpuwm_requirement()


_SEAM_SUBJECT = "woof seam bytes (the sixteen-file pin)"


def _check_seam_bytes(installed: str) -> Finding:
    """Do the installed engine's bytes match the ones the port pins?

    A VERSION comparison cannot answer this and a byte comparison can: the
    version is what the finding NAMES so a user can act, the bytes are what
    it knows.  The two are kept apart deliberately, because the walk's
    victim was a report that compared 2.5.7 against a floor of 2.5.5, found
    it higher, and called the estate healthy.
    """

    from . import engine_pin

    wanted = engine_pin.wanted_version()
    root = engine_pin.installed_root()
    if root is None:
        return Finding(
            subject=_SEAM_SUBJECT,
            status=INFO,
            detail=(
                "woof is installed but its files could not be located on "
                "disk, so the seam bytes were not compared.  The forecast "
                "hashes them at launch and records the result in its receipt."
            ),
        )
    try:
        inspection = engine_pin.inspect_seam(root)
    except Exception as error:  # pragma: no cover - a broken manifest module
        return Finding(
            subject=_SEAM_SUBJECT,
            status=INFO,
            detail=(
                f"the seam manifest could not be read ({error}), so the "
                "installed engine's bytes were not compared here"
            ),
        )

    total = len(engine_pin.seam_manifest())
    if inspection.misdeclared:
        # INFO, not a blocking gap, since 2026-10-10: the byte match is
        # evidence a receipt records (engine_pin.identity_of) and the
        # forecast runs on this install.
        return Finding(
            subject=_SEAM_SUBJECT,
            status=INFO,
            detail=(
                f"woof {installed} is installed and its {inspection.checked} "
                f"seam files are, byte for byte, the measured row of the "
                f"admitted engine woof {inspection.engine}.  An admitted "
                "engine is keyed by its version string and its bytes "
                "together, so a forecast on this install runs and its "
                f"receipt names it as woof {installed} with no measured "
                f"row, not as that engine and not as the pin ({wanted})."
            ),
            evidence={
                "installed": installed,
                "bytes_of": inspection.engine,
                "wanted": wanted,
                "root": str(root),
            },
        )
    if not inspection.drifted:
        row = (
            f" -- the measured row of the admitted engine woof "
            f"{inspection.engine}, not the public pin ({wanted})"
            if inspection.admitted
            else ""
        )
        return Finding(
            subject=_SEAM_SUBJECT,
            status=VERIFIED,
            detail=(
                f"woof {installed}: {inspection.checked} of {total} pinned "
                f"files are in this install and all {inspection.checked} match"
                f"{row}"
            ),
            evidence={
                "root": str(root),
                "matched": inspection.checked,
                "not_in_site_packages": list(inspection.absent),
                "wanted": wanted,
                "engine": inspection.engine,
                "admitted": inspection.admitted,
            },
        )

    named = installed or engine_pin.version_from_moved(inspection.moved) or "this"
    # INFO, not a blocking gap, since 2026-10-10 (owner ruling): this finding used to say "the forecast lane
    # refuses at launch", and it did.  A byte difference from the measured
    # engine names no breakage, so the forecast runs and its receipt records
    # the moved files (engine_pin.identity_of).
    return Finding(
        subject=_SEAM_SUBJECT,
        status=INFO,
        detail=(
            f"woof {named} is installed and the engine this port was "
            f"measured against is {wanted}: {len(inspection.moved)} of "
            f"{inspection.checked} checkable seam files differ.  The "
            "forecast runs on this install and its receipt records the "
            "difference (arwen_engine_pin).  Moved: "
            + ", ".join(inspection.moved)
        ),
        evidence={
            "installed": installed,
            "wanted": wanted,
            "declared": _requirement(),
            "moved": list(inspection.moved),
            "root": str(root),
        },
    )


def check_physics_seam() -> list[Finding]:
    """woof: the installed distribution, its BYTES, and the checkout the pin needs.

    Three findings, because they are three different facts with three
    different remedies, and folding them into one is how "woof is
    installed" got read as "the forecast lane will run".

    The middle one is the check the 2026-08-27 walk found missing.  It is a
    byte comparison, not a version comparison: the version is what the
    report NAMES, and the bytes are what it knows.  Every pinned path the
    installed engine carries is hashed here -- at 2.5.8 that is all sixteen,
    at 2.5.7 and below fifteen -- so this costs one hash of small files and
    needs no card, no network and no checkout.
    """

    from . import engine_pin

    findings: list[Finding] = []
    installed = _distribution_version("recast-woof")
    if installed is None:
        findings.append(
            Finding(
                subject="woof (the physics seam)",
                status=MISSING,
                detail="no woof distribution is installed in this interpreter",
                remedy=engine_pin.remedy(),
                required=True,
            )
        )
    else:
        findings.append(
            Finding(
                subject="woof (the physics seam)",
                status=PRESENT,
                detail=f"woof {installed} is installed",
                evidence={"version": installed, "declared": _requirement()},
            )
        )
        findings.append(_check_seam_bytes(installed))
    findings.append(_check_checkout_reach())
    return findings


def _check_checkout_reach() -> Finding:
    """Whether the installed woof can serve as the forecast's seam source.

    THE BREAKAGE THIS SHAPE PREVENTS, measured 2026-08-28 on a virtualenv
    holding only the published 2.5.8 wheels: this finding was a CONSTANT
    string saying "an installed woof satisfies pip and does not satisfy the
    pin", and it printed one line under an OK line reading "16 of 16 pinned
    files are in this install and all 16 match".  Two adjacent lines, one of
    them false, and the false one is the one that tells a user to go and
    clone something.  A reason that is typed once and never re-measured
    outlives the defect it was written for, so this one is DERIVED from the
    same inspection the line above it prints.

    Since 0.3.2 the driver's ``verify_arwen_checkout_git`` names an
    install by its version and pip's RECORD digest where it names a clone by
    its commit, so an install that carries all sixteen paths IS the seam
    source and no clone is asked for.
    """

    from . import engine_pin

    subject = "woof seam source (the forecast only)"
    tail = (
        "  The init and render doors need neither: they import no woof and "
        "drive Rust binaries instead."
    )
    wanted = engine_pin.wanted_version()
    root = engine_pin.installed_root()
    absent: tuple[str, ...] = ()
    total = len(engine_pin.seam_manifest())
    if root is not None:
        try:
            absent = engine_pin.inspect_seam(root).absent
        except Exception:  # pragma: no cover - a broken manifest module
            absent = ()
    if root is None or absent:
        missing = ", ".join(absent) if absent else "the pinned paths"
        count = len(absent) or total
        return Finding(
            subject=subject,
            status=INFO,
            detail=(
                f"{count} of the {total} pinned paths "
                f"{'is' if count == 1 else 'are'} not in this install "
                f"({missing}), so the installed distribution "
                "cannot satisfy the seam pin on its own.  Install woof "
                f"{wanted}, or pass a git clone of woof at v{wanted} as "
                "--gpuwm-checkout." + tail
            ),
            evidence={"absent": list(absent), "wanted": wanted},
        )
    return Finding(
        subject=subject,
        status=INFO,
        detail=(
            f"all {total} pinned paths resolve from this install, so the "
            "forecast runs on it with no --gpuwm-checkout and its receipts "
            "name the install by version and pip's RECORD digest.  A git "
            f"clone of woof at v{wanted} passed as --gpuwm-checkout is "
            "named by commit instead." + tail
        ),
        evidence={"absent": [], "wanted": wanted},
    )


def check_engines() -> list[Finding]:
    """Required Rust binaries and the command that stages them."""

    findings: list[Finding] = []
    # Asked once, of the INSTALLED woof, and reported plainly: whether the
    # one-command staging route can supply these engines at all on this box.
    # woof's published 2.5.2 bundles rw_wrfbatch and none of the MPAS
    # binaries, so "run woof fetch-bridges" is a complete answer for one of
    # the three engines here and no answer at all for the other two.  A
    # report that does not distinguish those sends a user round a loop.
    supplied = {spec.name: engines.gpuwm_bundles(spec) for spec in engines.ENGINES}
    unsupplied = sorted(name for name, ok in supplied.items() if ok is False)
    if unsupplied:
        findings.append(
            Finding(
                subject="woof fetch-bridges coverage",
                status=INFO,
                detail=(
                    "the woof installed here bundles no "
                    f"{', '.join(unsupplied)}, so `{engines.FETCH_COMMAND}` "
                    "cannot supply "
                    f"{'them' if len(unsupplied) > 1 else 'it'}.  A later "
                    "woof release adds these artifacts; until then they come "
                    "from a source build."
                ),
                evidence={"not_bundled": unsupplied},
            )
        )
    for spec in engines.ENGINES:
        path, source = engines.locate(spec)
        if path is not None:
            findings.append(
                Finding(
                    subject=f"{spec.name} ({spec.subject})",
                    status=VERIFIED,
                    detail=f"found via {source}: {path}",
                    required=True,
                    evidence={"path": str(path), "resolved_from": source},
                )
            )
            continue
        findings.append(
            Finding(
                subject=f"{spec.name} ({spec.subject})",
                status=MISSING,
                detail=(
                    f"{source}.  Without it {spec.what_breaks}.  "
                    f"Looked at: {engines.resolution_order(spec)}"
                ),
                remedy=engines.chmod_remedy(source) or engines.remedy(spec),
                required=True,
                evidence={"resolution_order": engines.resolution_order(spec)},
            )
        )
    findings.append(_check_observation_scorer())
    return findings


def _check_observation_scorer() -> Finding:
    """The observation scorer: absent only from a build of the sdist without a woof checkout."""

    from . import observation_scorecard

    subject = "hex_obs_scorecard (the observation scorer)"
    try:
        path = observation_scorecard.binary("hex_obs_scorecard")
    except FileNotFoundError as error:
        return Finding(subject=subject, status=MISSING, required=False,
                       detail=f"{error}.  Forecasts still run; `woof hex score` and the forecast's "
                              "automatic scorecard record NOT_MEASURED.")
    return Finding(subject=subject, status=VERIFIED, detail=f"found: {path}", evidence={"path": path})


def check_assets() -> list[Finding]:
    """The mesh pair, which this distribution ships and fetches neither."""

    return [
        Finding(
            subject="mesh grid + static pair",
            status=INFO,
            detail=(
                "external assets.  This distribution carries no mesh and has "
                "no fetch path for one, so every door that needs a mesh takes "
                "--grid and --static explicitly and refuses rather than guess "
                "a default.  Two routes exist: generate a mesh of any "
                "resolution with the staged rw_mpas_mesh and rw_mpas_static "
                "binaries, or supply an existing grid/static pair."
            ),
            evidence={"bridge_directory": str(engines.USER_BRIDGE_DIR)},
        )
    ]


def check_on_demand_assets() -> list[Finding]:
    """Report pinned cache bytes without fetching or changing the cache."""
    from . import assets
    try:
        return [Finding(subject=row['name'],
                        status=MISSING if row['status'] == 'invalid' else VERIFIED if row['status'] == 'staged' else INFO,
                        detail=row['detail'], required=row['status'] == 'invalid',
                        remedy=f"woof hex fetch-assets --only {row['name']}" if row['status'] != 'staged' else None,
                        evidence={'path': row['path'], 'sha256': row['sha256']})
                for row in assets.staged_rows()]
    except (OSError, ValueError, KeyError) as error:
        return [Finding(subject='on-demand assets', status=MISSING, required=True,
                        detail=str(error), remedy='reinstall woof hex')]


def _compiler_identity_probe() -> dict[str, object] | None:
    """The loaded NVRTC's identity (``woof.hex.cuda_backend.aot.compiler_identity``), in a short-lived process.

    It loads NVRTC and reads its version and library digest; it compiles nothing and touches no device.
    """

    code = (
        "import json, sys\n"
        "try:\n"
        "    from woof.hex.cuda_backend.aot import compiler_identity\n"
        "    out = {'identity': compiler_identity()}\n"
        "except Exception as error:\n"
        "    out = {'error': f'{type(error).__name__}: {error}'}\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    try:
        probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               errors="replace", timeout=_PROBE_TIMEOUT_S)
        document = json.loads((probe.stdout or "").strip() or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def check_kernel_pack() -> list[Finding]:
    """Whether the loaded NVRTC has a pinned prebuilt kernel pack.

    THE BREAKAGE (woof hex 0.4.12 post-publish check, 2026-11-05): a fresh ``[gpu-cu13]`` install pulled NVRTC
    13.5.37, published after the gate; no pack was pinned for it, so the first forecast compiled every kernel
    through NVRTC for about 100 s before its first step, and doctor printed "Every check passed".  A mismatch is
    a WARN: forecasts still run, only the first start of each kernel is slow.
    """

    subject = "prebuilt kernel pack"
    probe = _compiler_identity_probe()
    identity = (probe or {}).get("identity")
    if not isinstance(identity, dict) or not identity.get("nvrtc"):
        reason = (probe or {}).get("error") or "the compiler probe did not answer"
        return [Finding(subject=subject, status=INFO,
                        detail=f"no CUDA compiler loaded here ({reason}); packs apply once a CUDA extra is installed")]
    from . import assets
    try:
        pins = assets.load_pins()
        name, row, reason = assets.compatible_pack(identity, pins)
    except (OSError, ValueError, KeyError) as error:
        return [Finding(subject=subject, status=MISSING, required=False, detail=f"asset pins unavailable: {error}",
                        remedy="reinstall woof hex")]
    loaded = ".".join(str(part) for part in identity.get("nvrtc", ()))
    evidence = {"loaded_nvrtc": loaded, "nvrtc_package": identity.get("nvrtc_package"),
                "nvrtc_library_sha256": identity.get("nvrtc_library_sha256")}
    if name is not None:
        return [Finding(subject=subject, status=VERIFIED,
                        detail=f"loaded NVRTC {loaded} ({identity.get('nvrtc_package')}) matches {name}",
                        evidence={**evidence, "pack": name, "sha256": row["sha256"]})]
    major = identity["nvrtc"][0]
    same_major = [(n, r) for n, r in pins["assets"].items()
                  if r.get("kind") == "kernel-pack" and r.get("select", {}).get("nvrtc_major") == major]
    pinned = assets.pinned_compilers(same_major)
    newest = max((r for _, r in same_major), key=lambda r: tuple(r.get("compiler", {}).get("nvrtc", ())),
                 default=None)
    package = (newest or {}).get("compiler", {}).get("nvrtc_package")
    remedy = (assets.nvrtc_remedy(identity) + f"\n# the NVRTC the newest pinned CUDA {major} pack was built with"
              if package else assets.nvrtc_remedy(identity))
    return [Finding(subject=subject, status=WARN,
                    detail=(f"{reason}.  Pinned: {', '.join(pinned) or 'none'}.  Forecasts still run; the first "
                            "start of each kernel compiles through NVRTC (about 100 s on a fresh install)."),
                    remedy=remedy, evidence={**evidence, "pinned": pinned})]


def check_physics_resources() -> list[Finding]:
    """Report MPAS cache readiness without fetching, compiling or opening a GPU."""
    from .mpasphys import resources_build as resources
    from .mpasphys import resources_fetch

    remedy = 'woof hex physics-resources fetch'
    subjects = {'thompson-tables': 'MPAS physics DATA (Thompson + upstream tables)',
                'physics-resources': 'MPAS physics packet',
                'pool-reader': 'MPAS physics pool reader'}
    try:
        installed = resources.read_installed()['resources']
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [Finding(subject=subject, status=MISSING, required=True,
                        detail=f'invalid physics resource record: {error}', remedy=remedy)
                for subject in subjects.values()]
    findings = []
    for key, subject in subjects.items():
        if key not in installed:
            findings.append(Finding(subject=subject, status=INFO,
                                    detail='not staged; the MPAS-physics route (the hex default) fetches it on first use',
                                    remedy=remedy,
                                    evidence={'installed_record': str(resources.installed_record_path())}))
            continue
        try:
            path = resources.resolve_resource(key)
            evidence = resources_fetch.verify_installed_data() if key == 'thompson-tables' else {
                'path': str(path), 'sha256': installed[key]['sha256']}
        except (OSError, ValueError, KeyError, TypeError) as error:
            findings.append(Finding(subject=subject, status=MISSING, required=True,
                                    detail=str(error), remedy=remedy))
        else:
            findings.append(Finding(subject=subject, status=VERIFIED,
                                    detail='installed bytes match their recorded SHA-256 pins',
                                    evidence=evidence))
    return findings


def check_geography() -> list[Finding]:
    """Check the same default archive and datasets used by the static door."""
    try:
        from woof import rustwx_static
    except ImportError:
        return []  # The dependency check already reports the missing engine.
    root = rustwx_static.default_geog_root()
    refusal = rustwx_static.geog_refusal(root)
    if refusal is None:
        return [Finding("WPS_GEOG (mesh static)", VERIFIED,
                        f"{root} has the datasets required by the static builder")]
    return [Finding("WPS_GEOG (mesh static)", MISSING,
                    refusal.splitlines()[0],
                    remedy="woof fetch-geog --datasets mesh", required=True)]


def collect() -> list[Finding]:
    """Every finding, in the order a reader should meet them."""

    findings: list[Finding] = []
    findings.extend(check_interpreter())
    findings.extend(check_python_dependencies())
    findings.extend(check_physics_seam())
    findings.extend(check_gpu_runtime())
    findings.extend(check_gpu_architecture())
    findings.extend(check_engines())
    findings.extend(check_assets())
    findings.extend(check_geography())
    findings.extend(check_on_demand_assets())
    findings.extend(check_kernel_pack())
    findings.extend(check_physics_resources())
    return findings


def blocking_gaps(findings: list[Finding]) -> list[Finding]:
    """The findings that make the exit status 1.  The rule, stated once."""

    return [f for f in findings if f.status == MISSING and f.required]


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
_LABEL = {
    VERIFIED: "OK      ",
    PRESENT: "PRESENT ",
    MISSING: "MISSING ",
    INFO: "INFO    ",
    WARN: "WARN    ",
}


def render(findings: list[Finding], *, explain: bool) -> str:
    """One line per finding, or the whole evidence and remedy block."""

    lines: list[str] = []
    for finding in findings:
        label = _LABEL.get(finding.status, finding.status)
        if explain:
            lines.append(f"{label}{finding.subject}")
            lines.append(f"          {finding.detail}")
            if finding.evidence:
                for key, value in sorted(finding.evidence.items()):
                    lines.append(f"          {key}: {value}")
            if finding.remedy:
                lines.append("")
                lines.extend(finding.remedy.splitlines())
            lines.append("")
            continue
        headline = finding.detail.splitlines()[0] if finding.detail else ""
        if len(headline) > 96:
            headline = headline[:93] + "..."
        lines.append(f"{label}{finding.subject}: {headline}")
        if finding.remedy:
            first = [
                line
                for line in finding.remedy.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if first:
                lines.append(f"          {first[0].strip()}")
    return "\n".join(lines)


def summarise(findings: list[Finding]) -> str:
    blocking = blocking_gaps(findings)
    if not blocking:
        gaps = [f for f in findings if f.status in (MISSING, WARN)]
        if gaps:
            return (
                f"\nEvery required check passed.  {len(gaps)} optional gap(s) "
                f"remain; each prints its own command above."
            )
        return "\nEvery check passed."
    names = ", ".join(f.subject.split(" (")[0] for f in blocking)
    return (
        f"\n{len(blocking)} required item(s) missing: {names}.\n"
        f"Run `woof hex doctor --explain` for the full remedy for each."
    )


def add_doctor_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "doctor",
        help="report what this install can reach, and the command for each gap",
        description=(
            "Check the runtime estate this distribution needs and cannot "
            "carry: the Python dependencies, the physics seam, the CUDA "
            "lane, and the Rust engines the front doors drive.  Every gap "
            "prints the command that closes it."
        ),
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print the evidence and the whole pasteable remedy for each finding",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the findings as JSON instead of the text report",
    )
    parser.set_defaults(handler=run_doctor)


def run_doctor(arguments: argparse.Namespace) -> int:
    from .assets import inspection_only
    with inspection_only():
        findings = collect()
    if getattr(arguments, "json", False):
        print(
            json.dumps(
                {
                    "distribution": DISTRIBUTION_NAME,
                    "findings": [finding.as_dict() for finding in findings],
                    "blocking": [
                        finding.subject for finding in blocking_gaps(findings)
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render(findings, explain=getattr(arguments, "explain", False)))
        print(summarise(findings))
    return 1 if blocking_gaps(findings) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="woof hex doctor")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--json", action="store_true")
    return run_doctor(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Finding",
    "INFO",
    "MISSING",
    "PRESENT",
    "VERIFIED",
    "add_doctor_parser",
    "blocking_gaps",
    "collect",
    "main",
    "render",
    "run_doctor",
    "summarise",
]
