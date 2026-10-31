"""The short-lived device probe: free VRAM, card load and the card's census.

One subprocess answers every question a caller asks the local card --
free and total memory, NVML utilization, and the local-memory census the
forecast fit prices against -- and exits, so the caller never stands up
a CUDA context of its own.

A LEAF MODULE ON PURPOSE: the standard library at module scope and one
function-local reach into :mod:`woof.local_gpu`, both of which the
standalone RW-WPS package stages.  That package does not carry the
forecast memory preflight (:mod:`woof.core.preflight`), which
re-exports every name here for its own callers.  While the probe lived
inside the preflight, the package's ``--preprocess-backend auto`` had no
load reading and kept a certified card another program held busy or
nearly full, where woof prepared on the CPU.  Anything this module
imports at module scope has to be staged by
``tools/build_rw_wps_release.py`` too.
"""

from __future__ import annotations

import json
import sys


def pool_lease_budget_bytes(device: int = 0) -> int | None:
    """This process's per-card pool-v3 reservation, never another owner's.

    The helper exports GPU_MUTEX_CARDS (physical indices in CUDA order),
    and records the budget in OWNER-N.json. Match a live ancestor's PID
    AND birth tick, just as the helper does, to avoid stale/reused PIDs.
    Exclusive leases have no declared budget and keep measured sizing.
    """
    import os
    from pathlib import Path

    cards = os.environ.get("GPU_MUTEX_CARDS", "").split(",")
    if device < 0 or device >= len(cards) or not cards[device].isdigit():
        return None
    try:
        ledger = json.loads((Path(os.environ["GPU_MUTEX_DIR"])
                             / f"OWNER-{cards[device]}.json").read_text())
        if ledger.get("schema") != "gpu-mutex.owner.v1":
            return None
        pid = os.getpid()
        while pid > 1:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            for owner in ledger["owners"]:
                budget = owner.get("budget_bytes")
                if (owner.get("pid") == pid and owner.get("birth") == fields[19]
                        and owner.get("mode") == "pool"
                        and type(budget) is int and budget > 0):
                    return budget
            pid = int(fields[1])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    return None


def sizing_free_memory(free_bytes: int, *, device: int = 0) -> tuple[int, str | None]:
    """Sizing availability under a lease, keeping telemetry itself intact."""
    budget = pool_lease_budget_bytes(device)
    if budget is None:
        return int(free_bytes), None
    return budget, (f"sizing against the pool lease budget {budget / 2**20:g} MiB, "
                    f"not the card's {free_bytes / 2**30:.2f} GiB free")


def cuda_device_identity(device: int) -> dict | None:
    """A physical card identity independent of CUDA visibility ordering.

    Prefer CUDA's UUID and retain the PCI reading already used to match
    CUDA with NVML below. PCI-only matches remain conservative across
    hosts; container hostnames cannot establish that two cards differ.
    Callers use this beside their existing device reading.
    """
    from woof.local_gpu import no_local_gpu
    if no_local_gpu():
        return None
    identity = {}
    try:
        import cupy as cp
        value = cp.cuda.runtime.getDeviceProperties(int(device)).get("uuid")
        if isinstance(value, bytes) and len(value) == 16:
            value = value.hex()
        if isinstance(value, str):
            value = value.strip().lower().removeprefix("gpu-").replace("-", "")
            if (len(value) == 32 and value != "0" * 32
                    and all(char in "0123456789abcdef" for char in value)):
                identity["uuid"] = value
    except Exception:  # noqa: BLE001 - older runtimes may expose only PCI
        pass
    try:
        import cupy as cp
        bus_id = cp.cuda.runtime.deviceGetPCIBusId(int(device))
        if isinstance(bus_id, bytes):
            bus_id = bus_id.decode("ascii")
        domain, bus, slot = str(bus_id).strip().lower().split(":")
        slot, function = slot.split(".")
        canonical = (f"{int(domain, 16):08x}:{int(bus, 16):02x}:"
                     f"{int(slot, 16):02x}.{int(function, 16):x}")
        identity["pci_bus_id"] = canonical
    except Exception:  # noqa: BLE001 - caller reserves conservatively
        pass
    return identity or None


#: What :func:`device_memory_probe_subprocess` runs in its short-lived
#: interpreter: both device questions -- the free/total VRAM the budget
#: subtracts from, and the local-memory profile the non-pool terms are
#: priced against -- answered in one process that then exits.
#:
#: TWO exit codes, not one.  Exit 3 is "a card could not be read"; exit
#: :data:`PROBE_EXIT_NO_RUNTIME` is "there is no CuPy here to read it
#: with", and the last stderr line names the module.  They were one code
#: until 2.3.3, and that is how `woof go`'s memory gate came to swallow
#: a missing GPU runtime: the probe exited 3, the gate read "no card
#: here", declined to refuse on a card it could not see, and let the
#: chain fetch gigabytes for a run that could never start.  A gate that
#: hides the reason a run cannot begin is worse than no gate.
_DEVICE_MEMORY_PROBE_SOURCE = """\
import json
import os
import subprocess
import sys

# SELF-CONTAINED ON PURPOSE.  This source runs in a bare interpreter to
# answer "is there a card, and what is it"; importing woof here would
# make the answer depend on the very install the caller may be asking
# about, and it did: importing one helper from woof.core.preflight
# turned the "no CuPy here" exit code into an ImportError traceback,
# which is the exact confusion PROBE_EXIT_NO_RUNTIME exists to end.

def _nvml_load():
    # (readings by bus ID, how many cards nvidia-smi listed).  The count is
    # every row printed, parsed or not: a card whose memory.used reads
    # [N/A] is still a card, and counting parsed rows made a two-card box
    # look like a one-card box to the rule in _nvml_sample.
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return {}, 0
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=pci.bus_id,memory.used,utilization.gpu,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return {}, 0
        readings = {}
        rows = [line for line in out.stdout.strip().splitlines()
                if line.strip()]
        for line in rows:
            fields = line.split(",")
            try:
                used = int(fields[1].strip()) * 1024 * 1024
            except (IndexError, ValueError):
                continue
            try:
                utilization = int(fields[2].strip())
            except (IndexError, ValueError):
                utilization = None
            try:
                total_mib = int(fields[3].strip())
            except (IndexError, ValueError):
                total_mib = None
            readings[fields[0].strip().lower().lstrip("0")] = (
                used, utilization,
                None if total_mib is None else total_mib * 1024 * 1024)
        return readings, len(rows)
    except Exception:
        return {}, 0


def _nvml_sample(device):
    # The pre-context row of the selected device, or None.  In its own
    # try: an error reading the bus ID used to throw away the whole fit
    # measurement, and with it the WDDM ceiling below.  nvidia-smi lists
    # every card the machine has, so when it LISTED one card that row IS
    # the selected device whatever the bus-ID read says; with several, a
    # row is used only when the bus IDs match (CUDA visibility can reorder
    # or hide cards), even when only one of their rows parsed.
    if not _readings:
        return None
    try:
        pci = cp.cuda.runtime.deviceGetPCIBusId(device)
        if isinstance(pci, bytes):
            pci = pci.decode("ascii")
        sample = _readings.get(pci.strip().lower().lstrip("0"))
    except Exception:
        sample = None
    if sample is None and _listed == 1:
        sample = next(iter(_readings.values()))
    return sample


def _is_cuda_runtime_error(error):
    # CuPy raises CUDARuntimeError for a failed runtime call, and its
    # message starts with the runtime's own code (cudaErrorMemoryAllocation,
    # cudaErrorDevicesUnavailable, ...).  Anything else raised in the try
    # below is this probe failing, not CUDA refusing the card.
    return (any(kind.__name__ == "CUDARuntimeError"
                for kind in type(error).__mro__)
            or "cudaError" in str(error))


# BEFORE cupy: this interpreter has no CUDA context yet, so the NVML
# reading here is the card without us on it, which is the free figure
# a run's own context is not yet charged against.
#
# THE PROFILE HALF IS THE CARD'S CENSUS AND NOTHING SAMPLED.  This probe
# used to take a second NVML reading after the context stood up and ship
# the delta as the card's bare context.  That delta is a card-wide figure
# in whole MiB: on one idle RTX 4090 one probe printed 395 MiB while
# every other field of this payload was identical across readings, and
# two earlier receipts of the same plan sit exactly 3 and 4 MiB below it
# -- three prices for one plan out of one document.  The
# context is priced from the census in the parent instead
# (preflight.MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD), which is
# also what the run door's own Machine prices, so two readings of one
# card are one profile.
_readings, _listed = _nvml_load()
try:
    import cupy as cp
except ImportError as error:
    sys.stderr.write("no-runtime: %s\\n" % (getattr(error, "name", None)
                                            or error))
    sys.exit(4)
_device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# Before setDevice: the bus-ID read needs no context, so the sample is
# attributed even when the context below cannot be made.
_sample = _nvml_sample(_device)
_before, _utilization = (None, None) if _sample is None else _sample[:2]
try:
    if _device:
        cp.cuda.runtime.setDevice(_device)
    free, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(_device)
    name = props["name"]
    _stack = int(cp.cuda.runtime.deviceGetLimit(0))
    # THE SMALLER OF THE TWO INSTRUMENTS, always.  On WDDM the display
    # driver can evict other processes' allocations, so cudaMemGetInfo
    # answers "free if everything else were paged out" -- measured
    # 2026-08-20 on an RTX 3080 with a loaded desktop, four consecutive
    # samples: memGetInfo said 9,097 MiB free while NVML said 3,375-3,405
    # MiB, a stable 5.7 GiB over-statement of a 10 GiB card.  A budget
    # built on the larger figure spends memory the run would have to
    # evict a desktop to get.  Same idiom as the device rail: an
    # ADDITIONAL ceiling, never a widening.  On Linux the two agree and
    # this is a no-op.
    # From the BEFORE reading -- the card without this probe's own
    # context on it.  The run's context is charged by the reserve, so
    # taking it out of free as well would bill it twice.
    _nvml_free = None if _before is None else max(0, int(total) - _before)
    _free = int(free) if _nvml_free is None else min(int(free), _nvml_free)
    # The compile platform, read the way woof.certify.compile_platform
    # reads it (NVRTC's own PTX banner names its four-part build; the
    # device names its architecture).  Both halves or neither: a half
    # that could not be read is left out, and the parent prices Noah-MP
    # on this card as unread rather than on a guessed platform.
    _platform = None
    try:
        import re as _re
        from cupy_backends.cuda.libs import nvrtc as _nvrtc
        _ptx = None
        # Load the two stdlib-only helpers by file. Importing the model package
        # here would make a bare runtime probe depend on that package's import.
        if len(sys.argv) > 2:
            try:
                import runpy as _runpy
                from pathlib import Path as _Path
                _root = _Path(sys.argv[2])
                _metadata = _runpy.run_path(str(_root / "compiler_metadata.py"))
                _platform_helpers = _runpy.run_path(str(_root / "certify" / "compile_platform.py"))
                _nvrtc.getVersion()
                _ptx = _metadata["nvrtc_banner"](_nvrtc, _platform_helpers["nvrtc_library"]())
            except Exception:
                # A standalone preparation kit can lack forecast provenance
                # helpers. It still measures the platform through the original
                # probe; never turn an unavailable cache into a missing card.
                _ptx = None
        if _ptx is None:
            _program = _nvrtc.createProgram("", "compile_platform_probe.cu", [], [])
            try:
                _nvrtc.compileProgram(_program, [])
                _ptx = _nvrtc.getPTX(_program)
            finally:
                _nvrtc.destroyProgram(_program)
        if isinstance(_ptx, bytes):
            _ptx = _ptx.decode("ascii", "replace")
        _build = _re.search(
            r"Cuda compilation tools, release [\\d.]+, V(?P<build>[\\d.]+)", _ptx)
        _capability = str(cp.cuda.Device(_device).compute_capability)
        if _build is not None and _capability:
            _platform = [_capability, _build.group("build")]
    except Exception:
        _platform = None
    payload = {
        "free_bytes": _free,
        "free_bytes_memgetinfo": int(free),
        "free_bytes_nvml": _nvml_free,
        "total_bytes": int(total),
        "utilization_gpu_percent": _utilization,
        "profile": {
            "name": (name.decode() if isinstance(name, bytes)
                     else str(name)),
            "multiprocessor_count": int(props["multiProcessorCount"]),
            "max_threads_per_multiprocessor": int(
                props["maxThreadsPerMultiProcessor"]),
            "default_stack_limit_bytes": _stack,
            "compile_platform": _platform,
        },
    }
except Exception as error:
    _error = " ".join(("%s: %s" % (type(error).__name__, error)).split())[:300]
    if not _is_cuda_runtime_error(error):
        # The probe's own failure, not CUDA's (a device property read
        # under another name, say).  Reported as a cuda_error it sent
        # automatic backend selection to the CPU saying "CUDA could not
        # open the card" about a card the preparation's own context would
        # have opened.
        print(json.dumps({"probe_error": _error}))
        sys.exit(3)
    # A card too full for a context, or held by another process in
    # exclusive-process mode, fails here.  Exit 3 still means "no card
    # answered" to the fit, but the NVML sample is printed before the exit:
    # it was discarded, so automatic backend selection kept CUDA for a
    # card that then could not allocate the preparation.
    print(json.dumps({
        "cuda_error": _error,
        "nvml": None if _sample is None else {
            "used_bytes": _sample[0],
            "total_bytes": _sample[2],
            "utilization_gpu_percent": _sample[1],
        },
    }))
    sys.exit(3)
print(json.dumps(payload))
"""

#: Long enough for a cold CuPy import plus context creation on a busy
#: box; a probe that cannot answer inside it reads as "no device", which
#: only ever under-promises (nothing refuses on a card it cannot see).
DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS = 120.0

#: The probe's exit code for "this interpreter has no CuPy at all",
#: distinct from the exit 3 that means "a card could not be read".
PROBE_EXIT_NO_RUNTIME = 4

#: :func:`device_memory_probe_reason` for :data:`PROBE_EXIT_NO_RUNTIME`,
#: the one reason whose remedy is installing CuPy.  A caller compares it
#: whole: other reasons quote an error, and an error that merely mentions
#: a cupy module is not a missing CuPy.
PROBE_REASON_NO_RUNTIME = "the GPU runtime (CuPy) is not installed"


def device_memory_probe_reason(*, run=None) -> str | None:
    """Why :func:`device_memory_probe_subprocess` has no numbers, or ``None``.

    ``None`` when the probe answered.  Otherwise one short phrase naming
    the CAUSE, so a caller can say something truer than "no card here"
    -- which is what the single-exit-code version forced every caller to
    say, including on a box whose only problem was an uninstalled
    runtime.
    """

    payload, reason = _device_memory_probe(run=run)
    return None if payload is not None else reason


#: The probe's exit code for "a card could not be read through CUDA"; its
#: last stdout line then names the error.  A CUDA runtime error carries the
#: selected card's pre-context NVML sample (:func:`_probe_failure_report`);
#: any other failure of the probe is a ``{"probe_error"}`` line instead
#: (:func:`_probe_error`).
PROBE_EXIT_CARD_UNREAD = 3


def _probe_last_line(stdout) -> dict | None:
    """The JSON object on an exit-3 probe's last stdout line, or ``None``."""

    lines = (stdout or "").strip().splitlines()
    if not lines:
        return None
    try:
        report = json.loads(lines[-1])
    except ValueError:
        return None
    return report if isinstance(report, dict) else None


def _probe_failure_report(stdout) -> dict | None:
    """The ``{"cuda_error", "nvml"}`` line an exit-3 probe printed, or ``None``."""

    report = _probe_last_line(stdout)
    if (report is None
            or not isinstance(report.get("cuda_error"), str)
            or not (report.get("nvml") is None
                    or isinstance(report.get("nvml"), dict))):
        return None
    return report


def _probe_error(stdout) -> str | None:
    """The error of an exit-3 probe that failed for a reason not CUDA's."""

    report = _probe_last_line(stdout)
    error = None if report is None else report.get("probe_error")
    return error if isinstance(error, str) and error.strip() else None


def _device_memory_probe(*, run=None, device=0, report_failure=False
                         ) -> tuple[dict | None, str | None]:
    """``(payload, reason)`` -- the probe result and, when absent, why.

    With ``report_failure`` a card the probe could not read through CUDA
    returns its failure report (see :func:`device_memory_probe_subprocess`)
    as the payload.
    """

    import subprocess

    from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu

    # The documented never-open-the-local-device switch, consulted
    # BEFORE anything spawns.  The probe subprocess IS device contact --
    # a CUDA primary context, memGetInfo, deviceGetLimit -- and the
    # 2.5.0 upgrader walk proved this path never asked: the variable was
    # set for every step and `woof go`'s memory gate still reported the
    # local card's free VRAM.  Under the switch there are no measured
    # numbers, on purpose; callers price the DECLARED budget and their
    # verdicts carry this reason so nobody mistakes "not read" for "not
    # there".
    if no_local_gpu():
        return None, (f"{NO_LOCAL_GPU_ENV} is set, so the local card was "
                      "not read")
    from pathlib import Path
    helper_root = str(Path(__file__).resolve().parents[1])
    runner = subprocess.run if run is None else run
    try:
        completed = runner(
            [sys.executable, "-c", _DEVICE_MEMORY_PROBE_SOURCE, str(device), helper_root],
            capture_output=True, text=True,
            timeout=DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"the probe subprocess did not run ({error})"
    if completed.returncode == PROBE_EXIT_NO_RUNTIME:
        return None, PROBE_REASON_NO_RUNTIME
    if completed.returncode != 0:
        if completed.returncode == PROBE_EXIT_CARD_UNREAD:
            error = _probe_error(completed.stdout)
            if error is not None:
                # The probe failed for a reason that is not CUDA's: no
                # failure report, because nothing says CUDA refuses this
                # card, and a reason that does not say it does.
                return None, ("the probe could not read the card through "
                              f"CUDA ({error})")
        report = (_probe_failure_report(completed.stdout)
                  if report_failure
                  and completed.returncode == PROBE_EXIT_CARD_UNREAD
                  else None)
        return report, "no CUDA device answered"
    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        return None, "the probe printed nothing"
    try:
        payload = json.loads(lines[-1])
    except ValueError:
        return None, "the probe printed something that is not its JSON"
    free = payload.get("free_bytes") if isinstance(payload, dict) else None
    if not isinstance(free, int) or isinstance(free, bool):
        return None, "the probe reported no free-memory figure"
    return payload, None


def device_memory_probe_subprocess(*, run=None, device=0,
                                   report_failure=False) -> dict | None:
    """Free/total VRAM and this card's local-memory profile, measured in
    a SHORT-LIVED subprocess; ``None`` when no card answered.

    ``cudaMemGetInfo`` and ``cudaDeviceGetLimit`` cannot be asked
    without standing up a CUDA primary context -- the same fact that
    keeps them out of estimator mode (see
    :func:`woof.core.preflight.device_physical_total_bytes`).  A process that asks them
    in-process therefore keeps that context, and its device memory, for
    the rest of its life.  ``woof check`` can afford that: it exits on
    the next line.  The ``woof go`` orchestrator cannot: after its
    memory gate it lives for the entire chain as the stage runner and
    progress printer, and the context it stood up to ask one question
    sat on the card for the whole run -- measured 0.486 GiB on the RTX
    5090 -- as a consumer no term of the budget it had just computed
    names.  Asked here, the context lives and dies inside the probe
    process and the caller never touches CUDA at all.

    The numbers are the same ones the in-process readers see (the probe
    runs ``sys.executable``, so it resolves the same CuPy), which is
    what keeps this gate and ``woof check`` from disagreeing about one
    card.  ``run`` is the ``subprocess.run`` seam, for tests.

    The numbers only.  A caller that must distinguish "no card" from "no
    runtime" -- the memory gate does, because those two answers licence
    opposite behaviour -- asks :func:`device_memory_probe_reason`.

    ``report_failure`` is for automatic preparation-backend selection.  A
    card the probe cannot read through CUDA (too full for a context, or
    held in exclusive-process mode) then returns the probe's failure
    report instead of ``None``: ``{"cuda_error": str, "nvml": {"used_bytes",
    "total_bytes", "utilization_gpu_percent"} | None}``, the NVML half
    being the selected card's pre-context sample.  Without it that card
    read as "no telemetry", auto kept CUDA, and the preparation then
    failed to allocate.  The fit never asks: to it such a card is still
    one that did not answer.  Only a CUDA runtime error makes that report;
    a failure of the probe's own (a device property read under another
    name, say) is still ``None``, missing telemetry, because nothing in
    it says the preparation's own context would fail.
    """

    return _device_memory_probe(
        run=run, device=device, report_failure=report_failure)[0]
