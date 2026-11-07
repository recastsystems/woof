"""The pinned numerical environment, one entry per published pin.

``docs/public/DETERMINISM.md`` section 3 states which parts of the stack must
be identical between two runs for a byte comparison to mean what the document
says it means.  That table is the contract; this module is its machine-readable
twin.  Two of the published rows name two things each -- the CuPy version and
``CUPY_ACCELERATORS``, and the netCDF4 and HDF5 versions -- so eleven rows carry
thirteen items, and each item is resolved and reported separately.

Every item is emitted with a status.  ``resolved`` means something measured it
in this process; ``unavailable`` means nothing did, and carries the reason.  A
pin is never omitted and never filled with a plausible value: refusing an
unresolved pin is the certification command's job, and it can only do that job
if the receipt admits which pins went unmeasured.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from woof.gpu_stack_identity import gpu_cuda_stack_identity

#: Path, relative to the repository root, of the document this table mirrors.
PIN_TABLE_DOC = "docs/public/DETERMINISM.md"

#: Heading of the section that carries the published pin table.
PIN_TABLE_HEADING = "Fixed numerical environment"

STATUS_RESOLVED = "resolved"
STATUS_UNAVAILABLE = "unavailable"
STATUSES = (STATUS_RESOLVED, STATUS_UNAVAILABLE)


@dataclass(frozen=True)
class Pin:
    """One item of the published pin set."""

    #: Capsule key.  Generic: it names a property of the stack, never a case.
    key: str
    #: Verbatim first column of the published row this item comes from.
    doc_row: str
    #: Which half of a compound row this is, or ``None`` for a whole row.
    doc_row_part: str | None
    #: Where a resolved value comes from.
    source: str


PINS: tuple[Pin, ...] = (
    Pin("gpu_identity",
        "Physical GPU (model, SM version, and (for a same-card screen) "
        "the same UUID)",
        None, "woof.gpu_stack_identity.gpu_cuda_stack_identity + cupy device"),
    Pin("cuda_driver_version", "Driver version", None,
        "woof.gpu_stack_identity.gpu_cuda_stack_identity"),
    Pin("cuda_toolkit_nvrtc", "CUDA toolkit / NVRTC used to compile kernels",
        None, "cupy.cuda.runtime.runtimeGetVersion + cupy.cuda.nvrtc"),
    Pin("cupy_version", "CuPy version, and `CUPY_ACCELERATORS`",
        "CuPy version", "importlib.metadata"),
    Pin("cupy_accelerators", "CuPy version, and `CUPY_ACCELERATORS`",
        "CUPY_ACCELERATORS", "os.environ"),
    Pin("numpy_version", "NumPy version", None, "importlib.metadata"),
    Pin("netcdf4_version", "netCDF4 / HDF5 versions", "netCDF4",
        "importlib.metadata"),
    Pin("hdf5_version", "netCDF4 / HDF5 versions", "HDF5",
        "netCDF4.__hdf5libversion__"),
    Pin("arwen_version_and_commit", "WOOF version and git commit", None,
        "woof.__version__ + woof.supervisor.git_commit"),
    Pin("config_bytes", "Configuration file bytes", None, "run context"),
    Pin("input_artifact_bytes", "Every input artifact's bytes", None,
        "run context"),
    Pin("runner_route_and_io_mode", "Runner route and I/O mode", None,
        "run context"),
    Pin("output_and_diagnostic_mode",
        "Output and diagnostic mode, history cadence, checkpoint cadence",
        None, "run context"),
)

#: Capsule keys of the pin set, in published order.
PIN_KEYS: tuple[str, ...] = tuple(pin.key for pin in PINS)

#: Published rows that carry two items each.
COMPOUND_DOC_ROWS: tuple[str, ...] = tuple(
    sorted({pin.doc_row for pin in PINS if pin.doc_row_part is not None}))

#: Pin keys no probe can fill; the caller's run context is the only source.
RUN_CONTEXT_PINS: tuple[str, ...] = tuple(
    pin.key for pin in PINS if pin.source == "run context")

_PIN_BY_KEY = {pin.key: pin for pin in PINS}


def _resolved(pin: Pin, value: Any) -> dict[str, Any]:
    return {"value": value, "source": pin.source, "status": STATUS_RESOLVED}


def _unavailable(pin: Pin, reason: str) -> dict[str, Any]:
    return {"value": None, "source": pin.source,
            "status": STATUS_UNAVAILABLE, "reason": reason}


def _distribution_version(name: str) -> str:
    from importlib.metadata import version

    return str(version(name))


_CUPY_DISTRIBUTION = re.compile(r"^cupy(-cuda\d+x)?$")


def _installed_cupy_version() -> str:
    """Version of whichever CuPy distribution is actually installed.

    CuPy ships one wheel per CUDA major (``cupy-cuda12x``,
    ``cupy-cuda13x``) plus the source-built plain ``cupy``; which name
    pip resolved is a property of the box.  This pin used to look up the
    literal ``cupy-cuda12x``, so a CUDA-13 box -- the very box the
    ``gpu-cu13`` extra installs -- certified with ``cupy_version``
    unavailable while running CuPy kernels.  The value stays a bare
    version string: the FTZ receipt and its claim-site anchor render it
    as one.

    Same enumeration as ``woof.doctor._installed_cupy_wheels``,
    transcribed rather than imported because the capsule path does not
    load the CLI diagnostics stack.  If several CuPy distributions are
    installed at once (a broken but observable pip state) the
    highest-sorting wheel name wins, deterministically, rather than
    raising a capsule out of existence.
    """
    from importlib.metadata import distributions

    found: dict[str, str] = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip().lower()
        if _CUPY_DISTRIBUTION.match(name):
            found[name] = str(dist.version)
    if not found:
        raise ModuleNotFoundError(
            "no CuPy distribution (cupy, cupy-cudaNNx) is installed")
    return found[sorted(found)[-1]]


#: A CUDA device UUID is 16 bytes.  ``cudaUUID_t`` is ``char bytes[16]``
#: with no terminator, so its length is the array bound and nothing else.
CUDA_UUID_BYTES = 16


def device_uuid_hex(raw):
    """Hex of the 16 bytes that are the UUID, and of no other bytes.

    The buffer CuPy hands back for ``getDeviceProperties()['uuid']`` is
    not bounded by that array: the conversion stops at the first zero
    byte rather than at the array end, and ``cudaDeviceProp`` places
    ``luid`` immediately after ``uuid``, so an unbounded read can carry
    bytes of the neighbouring member.  Those bytes are not device
    identity -- Windows re-issues the LUID when it re-enumerates the
    adapter, so a pin regenerated on the same physical card could
    differ: churn that reads exactly like the run having moved to
    different silicon, which is the one question ``gpu_identity``
    exists to answer.  Same defect, same bound as
    ``tools/ftz_receipt/probe.device_uuid_hex``.
    """
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)[:CUDA_UUID_BYTES].hex()
    return raw


def _gpu_pins(entries: dict[str, dict[str, Any]], *, require_gpu: bool) -> None:
    gpu_identity = _PIN_BY_KEY["gpu_identity"]
    driver = _PIN_BY_KEY["cuda_driver_version"]
    toolkit = _PIN_BY_KEY["cuda_toolkit_nvrtc"]
    if not require_gpu:
        reason = "the caller declared this route does not probe a device"
        entries["gpu_identity"] = _unavailable(gpu_identity, reason)
        entries["cuda_driver_version"] = _unavailable(driver, reason)
        entries["cuda_toolkit_nvrtc"] = _unavailable(toolkit, reason)
        return
    try:
        block = gpu_cuda_stack_identity(require_gpu=True)
        import cupy as cp

        # This is the visible process-local device.  The run supervisor masks
        # its selected physical UUID before the worker imports CuPy.
        device = cp.cuda.Device(0)
        properties = cp.cuda.runtime.getDeviceProperties(0)
        uuid = device_uuid_hex(properties.get("uuid"))
        identity = {
            "name": block["device_0_name"],
            "compute_capability": __import__("woof.backend", fromlist=["compilation_target"]).compilation_target(device.id, xp=cp),
            "uuid": None if uuid is None else str(uuid),
            "device_count": block["device_count"],
        }
    except Exception as error:  # the probe fails closed; the pin says why
        reason = f"{type(error).__name__}: {error}"
        entries["gpu_identity"] = _unavailable(gpu_identity, reason)
        entries["cuda_driver_version"] = _unavailable(driver, reason)
        entries["cuda_toolkit_nvrtc"] = _unavailable(toolkit, reason)
        return
    entries["gpu_identity"] = _resolved(gpu_identity, identity)
    entries["cuda_driver_version"] = _resolved(
        driver, int(block["driver_version"]))
    if getattr(cp.cuda.runtime, 'is_hip', False):
        from woof.hiprtc import identity as hiprtc_identity
        entries["cuda_toolkit_nvrtc"] = _resolved(toolkit, {
            "backend": "hip", "hip_runtime_version": int(block["runtime_version"]),
            "hiprtc": hiprtc_identity(),
        })
        return
    try:
        from cupy.cuda import nvrtc

        nvrtc_version = ".".join(str(part) for part in nvrtc.getVersion())
    except Exception as error:
        nvrtc_version = f"unavailable: {type(error).__name__}: {error}"
    # ``getVersion()`` is major.minor, and that is not enough resolution for
    # the row's own claim.  12.9.41 and 12.9.86 both report "12.9" and do not
    # generate the same code; on 2026-08-04 this box moved from NVRTC 13.0.48
    # to 12.9.86 and a certified ULP table moved with it while this pin read
    # unchanged.  The four-part build and the library digest are recorded
    # alongside, never instead: the coarse value is what the published table
    # quotes and it keeps its key.
    from woof.certify.compile_platform import (
        nvrtc_banner, nvrtc_build, nvrtc_build_id, nvrtc_library)

    banner = nvrtc_banner()
    library = nvrtc_library()
    entries["cuda_toolkit_nvrtc"] = _resolved(toolkit, {
        "cuda_runtime_version": int(block["runtime_version"]),
        "nvrtc_version": nvrtc_version,
        "nvrtc_build": nvrtc_build(banner),
        "nvrtc_build_id": nvrtc_build_id(banner),
        "nvrtc_library_sha256": library["sha256"],
    })


def _package_pins(entries: dict[str, dict[str, Any]]) -> None:
    for key, resolve in (
            ("cupy_version", _installed_cupy_version),
            ("numpy_version", lambda: _distribution_version("numpy")),
            ("netcdf4_version", lambda: _distribution_version("netCDF4"))):
        pin = _PIN_BY_KEY[key]
        try:
            entries[key] = _resolved(pin, resolve())
        except Exception as error:
            entries[key] = _unavailable(
                pin, f"{type(error).__name__}: {error}")

    accelerators = _PIN_BY_KEY["cupy_accelerators"]
    raw = os.environ.get("CUPY_ACCELERATORS")
    entries["cupy_accelerators"] = _resolved(accelerators, {
        "present": raw is not None,
        "value": raw,
    })

    hdf5 = _PIN_BY_KEY["hdf5_version"]
    try:
        import netCDF4

        entries["hdf5_version"] = _resolved(
            hdf5, str(netCDF4.__hdf5libversion__))
    except Exception as error:
        entries["hdf5_version"] = _unavailable(
            hdf5, f"{type(error).__name__}: {error}")


def _code_pins(entries: dict[str, dict[str, Any]]) -> None:
    pin = _PIN_BY_KEY["arwen_version_and_commit"]
    import woof
    import woof.supervisor as supervisor

    commit = supervisor.git_commit()
    if commit.startswith("unavailable"):
        # A pip-installed tree is not a git repository, and this pin used
        # to say ``resolved`` around exactly that refusal -- the published
        # wheel's capsule carried status "resolved" wrapping the value
        # "unavailable: fatal: not a git repository".  Resolved is a claim
        # that something measured the value; a payload that says
        # unavailable refutes it in the same breath.  The pin is accurately
        # unavailable, and the half that DOES exist -- the version, which
        # ``woof.__version__`` reads from the installed distribution's
        # metadata -- stays bound in the entry.
        entries["arwen_version_and_commit"] = {
            "value": {"version": woof.__version__},
            "source": pin.source,
            "status": STATUS_UNAVAILABLE,
            "reason": ("the git commit half could not be resolved "
                       f"({commit}); the version half is bound from the "
                       "installed distribution's metadata"),
        }
        return
    entries["arwen_version_and_commit"] = _resolved(pin, {
        "version": woof.__version__,
        "git_commit": commit,
    })


def resolve_pins(run_context: Mapping[str, Any] | None = None, *,
                 require_gpu: bool = True) -> dict[str, dict[str, Any]]:
    """Resolve every published pin, in published order.

    ``run_context`` supplies the four pins nothing can probe -- the
    configuration bytes, the input-artifact bytes, the runner route with its
    I/O mode, and the output/diagnostic mode with its cadences.  Whatever it
    does not supply is reported ``unavailable`` with that as the reason, so a
    receipt built off a partial context is legible as one.
    """
    context = dict(run_context or {})
    unknown = sorted(set(context) - set(RUN_CONTEXT_PINS))
    if unknown:
        raise KeyError(f"run context carries non-pin keys {unknown}")
    entries: dict[str, dict[str, Any]] = {}
    _gpu_pins(entries, require_gpu=require_gpu)
    _package_pins(entries)
    _code_pins(entries)
    for key in RUN_CONTEXT_PINS:
        pin = _PIN_BY_KEY[key]
        if key in context and context[key] is not None:
            entries[key] = _resolved(pin, context[key])
        else:
            entries[key] = _unavailable(
                pin, "the run context did not supply this pin")
    return {key: entries[key] for key in PIN_KEYS}


def unresolved_pins(numerical_stack: Mapping[str, Any]) -> tuple[str, ...]:
    """Pin keys that are missing, or present without a resolved status."""
    missing = [key for key in PIN_KEYS if key not in numerical_stack]
    unresolved = [key for key in PIN_KEYS
                  if key in numerical_stack
                  and numerical_stack[key].get("status") != STATUS_RESOLVED]
    return tuple(missing + unresolved)
