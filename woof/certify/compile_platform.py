"""Identity of the toolchain that turns this project's ``.cu`` sources into SASS.

Every acceptance gate stated in ULP against a Fortran oracle is a measurement
of *the compiled kernel*, not of the source.  The source is pinned by
:mod:`woof.certify.kernel_manifest` (``source_sha256``); this module pins the
other half -- which compiler produced the image the numbers were measured on.

Why it exists.  On 2026-08-04 this project's kernel compiler changed underneath
a certified table without a single tracked input moving: ``cupy`` 14.0.1,
``numpy`` 2.2.6, CPython 3.13.7 and the display driver were all still on their
pinned versions, and the ``.cu`` file was byte-identical, yet the Shin-Hong
CUDA column's distance from WRF moved on two fields.  The mover was NVRTC:
CuPy had been resolving the system CUDA v13.0 toolkit (NVRTC/ptxas 13.0.48)
and, once ``nvidia-cuda-nvrtc-cu12`` 12.9.86 appeared in ``site-packages``, it
resolved that instead.  The published pin row for that item reported only
``nvrtc.getVersion()`` -- ``(12, 9)`` -- which cannot separate 12.9.41 from
12.9.86, and the failure surfaced as a bare number mismatch with no named
cause.  What follows is the missing resolution.

Nothing here needs a device: NVRTC compiles an empty translation unit without
one, and the reported build string is exactly what CuPy folds into its own
kernel-cache key, so it is the same discriminator the cache already trusts.

Every item carries a status, the same contract :mod:`woof.certify.pins` uses:
``resolved`` means something measured it in this process, ``unavailable``
means nothing did and says why.  A fingerprint is never filled with a
plausible value.
"""

from __future__ import annotations

from woof.backend import compilation_target as _compilation_target

import hashlib
import os
import re
import sys
from typing import Any, Mapping

#: Status of an item this process measured.
STATUS_RESOLVED = "resolved"

#: Status of an item nothing measured, carrying the reason.
STATUS_UNAVAILABLE = "unavailable"

#: Value stored for an item that could not be resolved.  Deliberately a
#: string, not ``None``: a fingerprint round-trips through JSON receipts and
#: must compare equal to itself after that trip.
UNRESOLVED = "unavailable"

_BUILD_RE = re.compile(
    r"Cuda compilation tools, release [\d.]+, V(?P<build>[\d.]+)")
_BUILD_ID_RE = re.compile(r"Compiler Build ID: (?P<id>\S+)")


def nvrtc_banner() -> str:
    """The comment banner NVRTC puts at the head of the PTX it emits.

    Compiling the empty translation unit is the cheapest complete statement of
    compiler identity available: it names the release, the four-part build and
    the internal changelist, and needs no device.  Returns ``""`` if NVRTC is
    not importable or refuses the empty program.
    """
    try:
        from cupy_backends.cuda.libs import nvrtc
    except Exception:
        return ""
    from woof.compiler_metadata import nvrtc_banner as retained_banner
    # Load the library before resolving its bytes. getVersion compiles nothing.
    try:
        getattr(nvrtc, "getVersion", lambda: None)()
        return retained_banner(nvrtc, nvrtc_library())
    except Exception:
        return ""


def nvrtc_build(banner: str | None = None) -> str:
    """The four-part NVRTC build, e.g. ``"12.9.86"`` or ``"13.0.48"``.

    This is the discriminator ``nvrtc.getVersion()`` is too coarse to be:
    12.9.41 and 12.9.86 both report ``(12, 9)`` and do not generate the same
    code.  ``UNRESOLVED`` when NVRTC did not answer.
    """
    text = nvrtc_banner() if banner is None else banner
    match = _BUILD_RE.search(text)
    return match.group("build") if match else UNRESOLVED


def nvrtc_build_id(banner: str | None = None) -> str:
    """NVRTC's internal changelist, e.g. ``"CL-36037853"``, or ``UNRESOLVED``."""
    text = nvrtc_banner() if banner is None else banner
    match = _BUILD_ID_RE.search(text)
    return match.group("id") if match else UNRESOLVED


def nvrtc_library() -> dict[str, str]:
    """Path and SHA-256 of the NVRTC shared library this process loaded.

    The build string above is what the compiler says about itself; this is
    what the filesystem says about the file, and the two are recorded together
    so a same-version rebuild -- or a second copy of the same version shadowing
    the first -- cannot pass as the image the numbers were measured on.
    """
    path = _loaded_library_path()
    if not path:
        return {"status": STATUS_UNAVAILABLE,
                "reason": "no NVRTC shared library is loaded in this process",
                "path": UNRESOLVED, "sha256": UNRESOLVED}
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    except OSError as error:
        return {"status": STATUS_UNAVAILABLE,
                "reason": f"{type(error).__name__}: {error}",
                "path": path, "sha256": UNRESOLVED}
    return {"status": STATUS_RESOLVED, "path": path, "sha256": digest}


def _loaded_library_path() -> str:
    """Absolute path of the loaded NVRTC library, or ``""``.

    Windows and Linux keep this in different places and neither exposes it
    through CuPy, so both are read directly.  Failure is a plain ``""``: this
    is provenance, and provenance must never be able to break a run.
    """
    try:
        if sys.platform == "win32":
            return _loaded_library_path_win32()
        with open("/proc/self/maps", encoding="ascii", errors="replace") as fh:
            for line in fh:
                path = line.rsplit(" ", 1)[-1].strip()
                name = os.path.basename(path)
                if name.startswith("libnvrtc.so") and "builtins" not in name:
                    return path
    except Exception:
        return ""
    return ""


def _loaded_library_path_win32() -> str:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleFileNameW.argtypes = [
        ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    # CuPy's CUDA 12 and 13 builds resolve different SONAMEs; ask for both
    # rather than assuming which wheel is installed.
    for name in ("nvrtc64_120_0.dll", "nvrtc64_130_0.dll"):
        handle = kernel32.GetModuleHandleW(name)
        if not handle:
            continue
        buffer = ctypes.create_unicode_buffer(4096)
        if kernel32.GetModuleFileNameW(handle, buffer, 4096):
            return buffer.value
    return ""


#: The fingerprint's key set, in the order a diff should read: the two items
#: that actually generate code first, then the runtime that loads it, then the
#: host libraries that feed it.  Declared once so the measurement, the
#: projection out of a capsule, and the completeness test cannot drift apart --
#: a key that only one of the three knows about is a key nothing compares.
FINGERPRINT_KEYS: tuple[str, ...] = (
    "nvrtc_build",
    "nvrtc_build_id",
    "nvrtc_library_sha256",
    "cuda_driver_version",
    "device_compute_capability",
    "cupy_version",
    "numpy_version",
)


def compile_platform_fingerprint() -> dict[str, str]:
    """Everything that decides what SASS this process runs, in one dict.

    Ordered so a diff reads outward from the compiler: the two items that
    actually generate code first, then the runtime that loads it, then the
    host libraries that feed it.  Values are strings so the dict survives a
    JSON round trip into a receipt and still compares equal.
    """
    from woof.hip_backend import active, compile_fingerprint
    if active(): return compile_fingerprint()
    banner = nvrtc_banner()
    library = nvrtc_library()
    fingerprint = {
        "nvrtc_build": nvrtc_build(banner),
        "nvrtc_build_id": nvrtc_build_id(banner),
        "nvrtc_library_sha256": library["sha256"],
        "cuda_driver_version": _driver_version(),
        "device_compute_capability": _compute_capability(),
        "cupy_version": _cupy_version(),
        "numpy_version": _package_version("numpy"),
    }
    return fingerprint


def _cupy_version() -> str:
    """Reuse the pin table's resolver rather than re-deriving it.

    CuPy ships one wheel name per CUDA major, and a second enumeration of
    those names is how two receipts end up disagreeing about the same box.
    :mod:`woof.certify.pins` already owns that enumeration.
    """
    try:
        from woof.certify.pins import _installed_cupy_version
        return _installed_cupy_version()
    except Exception:
        return UNRESOLVED


def _driver_version() -> str:
    try:
        from cupy.cuda import runtime
        return str(runtime.driverGetVersion())
    except Exception:
        return UNRESOLVED


def _compute_capability() -> str:
    try:
        import cupy
        return str(_compilation_target(xp=cupy))
    except Exception:
        return UNRESOLVED


def _package_version(name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return UNRESOLVED


def describe_drift(recorded, measured) -> list[str]:
    """Human-readable ``key: recorded -> measured`` lines for what differs.

    Empty when the two agree on every key they share.  Keys present in only
    one side are reported too: a fingerprint that grew a field is itself a
    change worth naming rather than silently ignoring.

    An empty return means "these two agree", which is NOT the same claim as
    "the platform is verified": two fingerprints that resolved nothing agree
    on every key.  :func:`compile_platform_agreement` is the caller that owes
    the distinction, and it refuses to reach a verdict at all until
    :func:`unresolved_fingerprint_items` is empty on both sides.
    """
    lines = []
    for key in sorted(set(recorded) | set(measured)):
        was = recorded.get(key, "(absent)")
        now = measured.get(key, "(absent)")
        if was != now:
            lines.append(f"  {key}: recorded {was} -> measured {now}")
    return lines


def unresolved_fingerprint_items(fingerprint: Mapping[str, Any] | None
                                 ) -> tuple[str, ...]:
    """Fingerprint keys nothing measured, in declared order.

    Positive evidence of work, stated as its absence.  A key is unresolved
    when it is missing, empty, ``None`` or carries :data:`UNRESOLVED`.  An
    empty return is the only thing that licenses a comparison: without it a
    ``describe_drift`` of ``[]`` is "compared nothing", not "found nothing".
    """
    if not isinstance(fingerprint, Mapping):
        return FINGERPRINT_KEYS
    unresolved = []
    for key in FINGERPRINT_KEYS:
        value = fingerprint.get(key)
        if value is None or value == "" or value == UNRESOLVED:
            unresolved.append(key)
    return tuple(unresolved)


def recorded_compile_platform(capsule: Mapping[str, Any]) -> dict[str, str]:
    """The fingerprint a capsule already recorded, in fingerprint shape.

    The capsule does not carry this dict verbatim and must not start to: the
    same values are already published as pins, and a second copy is how two
    halves of one receipt end up disagreeing.  This projects the pins the
    published table names -- the NVRTC block of ``cuda_toolkit_nvrtc``, the
    driver, the device's compute capability, CuPy and NumPy -- onto the key
    set :func:`compile_platform_fingerprint` measures, so the two can be
    compared key for key.

    A pin whose status is not ``resolved``, or whose value does not carry the
    item, projects to :data:`UNRESOLVED`.  Nothing is inferred and nothing is
    omitted: a missing key would silently shrink the comparison, which is the
    failure this whole module exists to make impossible.
    """
    stack = capsule.get("numerical_stack")
    if not isinstance(stack, Mapping):
        stack = {}

    def pin_value(key: str) -> Any:
        entry = stack.get(key)
        if not isinstance(entry, Mapping):
            return None
        if entry.get("status") != STATUS_RESOLVED:
            return None
        return entry.get("value")

    def text(value: Any) -> str:
        if value is None or isinstance(value, (Mapping, list, tuple)):
            return UNRESOLVED
        rendered = str(value)
        return rendered if rendered and rendered != UNRESOLVED else UNRESOLVED

    toolkit = pin_value("cuda_toolkit_nvrtc")
    toolkit = toolkit if isinstance(toolkit, Mapping) else {}
    identity = pin_value("gpu_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    return {
        "nvrtc_build": text(toolkit.get("nvrtc_build")),
        "nvrtc_build_id": text(toolkit.get("nvrtc_build_id")),
        "nvrtc_library_sha256": text(toolkit.get("nvrtc_library_sha256")),
        # The pin stores the CUDA driver version as the integer CuPy reports;
        # the fingerprint stores every value as text so it survives a JSON
        # round trip.  Same measurement, one rendering.
        "cuda_driver_version": text(pin_value("cuda_driver_version")),
        "device_compute_capability": text(identity.get("compute_capability")),
        "cupy_version": text(pin_value("cupy_version")),
        "numpy_version": text(pin_value("numpy_version")),
    }


def compile_platform_agreement(recorded: Mapping[str, Any] | None,
                               measured: Mapping[str, Any] | None
                               ) -> dict[str, Any]:
    """Whether a recorded compile platform and a measured one are the same.

    Fail-closed, and -- the part that matters -- **verdict-suppressing**.
    ``drift`` is ``None``, not ``[]``, whenever either side is incomplete:
    two fingerprints that resolved nothing are equal, so a comparison run on
    partial data would report "no drift" and launder an unverified platform
    as a verified one.  That is the shape this project has already been burned
    by (a nonzero delta from two arms that executed no steps), so the
    comparison is not attempted rather than attempted on whatever is there.

    ``agreed`` is true only when both sides resolved every item AND nothing
    differs.  The returned block is what a verdict binds, so a reader can see
    the measurement that licensed the answer instead of only the answer.
    """
    recorded_missing = unresolved_fingerprint_items(recorded)
    measured_missing = unresolved_fingerprint_items(measured)
    recorded_block = {key: str((recorded or {}).get(key, UNRESOLVED))
                      for key in FINGERPRINT_KEYS}
    measured_block = {key: str((measured or {}).get(key, UNRESOLVED))
                      for key in FINGERPRINT_KEYS}
    if recorded_missing or measured_missing:
        return {
            "recorded": recorded_block,
            "measured": measured_block,
            "recorded_unresolved": list(recorded_missing),
            "measured_unresolved": list(measured_missing),
            "drift": None,
            "comparable": False,
            "agreed": False,
        }
    drift = describe_drift(dict(recorded_block), dict(measured_block))
    return {
        "recorded": recorded_block,
        "measured": measured_block,
        "recorded_unresolved": [],
        "measured_unresolved": [],
        "drift": drift,
        "comparable": True,
        "agreed": not drift,
    }


def describe_fingerprint(fingerprint: Mapping[str, Any] | None) -> str:
    """One line naming the compiler a fingerprint identifies.

    Used in a satisfied condition's detail so a PASS carries the evidence
    that something was measured, rather than a bare boolean a reader has to
    take on trust.
    """
    values = fingerprint or {}
    return (f"nvrtc_build {values.get('nvrtc_build', UNRESOLVED)}, "
            f"build id {values.get('nvrtc_build_id', UNRESOLVED)}, "
            f"library sha256 "
            f"{str(values.get('nvrtc_library_sha256', UNRESOLVED))[:16]}, "
            f"cupy {values.get('cupy_version', UNRESOLVED)}, "
            f"numpy {values.get('numpy_version', UNRESOLVED)}")


__all__ = [
    "FINGERPRINT_KEYS",
    "STATUS_RESOLVED",
    "STATUS_UNAVAILABLE",
    "UNRESOLVED",
    "compile_platform_agreement",
    "compile_platform_fingerprint",
    "describe_drift",
    "describe_fingerprint",
    "nvrtc_banner",
    "nvrtc_build",
    "nvrtc_build_id",
    "nvrtc_library",
    "recorded_compile_platform",
    "unresolved_fingerprint_items",
]
