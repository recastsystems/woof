#!/usr/bin/env python3
"""Measure the real forecast door with full boundary receipts.

All forecast arguments after ``--`` go through ``woof.hex.cli.main``, with the
door's own defaults (MPAS-A v8.4.2's schedule and a mesh-derived timestep).
``--dt SECONDS`` is forwarded to the forecast as ``--dt`` when the forecast
arguments do not already carry one.  ``--measurement-mint`` is accepted for
older command lines and only labels the receipt: since 2026-10-10 no timestep
or regional configuration is refused for lacking an anchor, so there is no
gate for it to open.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import pstats
import sys
import sysconfig
import threading
import time
from typing import Any

def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def source_snapshot() -> dict[str, Any]:
    paths = (
        "src/hexcore/cli.py", "src/hexcore/forecast_door.py",
        "src/hexcore/dt_admission.py", "src/hexcore/cuda_regional_forecast_v841.py",
        "src/hexcore/drivers/mpas_mesh_binding.py",
        "src/hexcore/drivers/run_cuda_v841_forecast.py",
        "src/hexcore/drivers/run_cuda_v841_full_physics_x4.py",
        "src/hexcore/drivers/samehost_forecast_v841.py",
        "src/hexcore/drivers/samehost_identity_v841.py",
        "src/hexcore/drivers/samehost_checks_v841.py",
        "src/hexcore/drivers/samehost_serial_io_v841.py",
        "src/hexcore/partition_topology_v841.py",
        "src/hexcore/partition_field_halos_v841.py",
        "src/hexcore/partition_overlap_v841.py",
        "src/hexcore/partition_samehost_transport_v841.py",
        "src/hexcore/partition_samehost_pipeline_v841.py",
        "src/hexcore/partition_samehost_v841.py",
        "src/hexcore/cuda_driver.py", "src/hexcore/cuda_horizontal.py",
        "src/hexcore/drivers/forecast_receipt.py",
    )
    from woof.hex.shipped_sources import resolve
    return {name: file_record(resolve(name)) if resolve(name).is_file() else {"absent": True}
            for name in paths}


def runtime_snapshot(checkout: Path | None) -> dict[str, Any]:
    versions = {}
    for package in ("woof", "recast-woof-data", "cupy-cuda13x", "cupy-cuda12x",
                    "numpy", "netCDF4"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = {}
    if checkout is not None:
        from woof.hex.engine_pin import seam_manifest
        for name in seam_manifest():
            path = checkout / name
            if path.is_file():
                files[str(path)] = file_record(path)
    return {"distribution_versions": versions, "physics_source_files": files,
            "interpreter": {"version": sys.version,
                            "Py_GIL_DISABLED": sysconfig.get_config_var("Py_GIL_DISABLED"),
                            "gil_enabled": (sys._is_gil_enabled()
                                            if hasattr(sys, "_is_gil_enabled") else None)}}


def _protected_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compare immutable runtime provenance separately from live GIL state."""
    runtime = dict(snapshot["runtime"])
    if "interpreter" in runtime:
        interpreter = dict(runtime["interpreter"])
        interpreter.pop("gil_enabled", None)
        runtime["interpreter"] = interpreter
    return {**snapshot, "runtime": runtime}


def _parallel_runtime_qualification(runtime: dict[str, Any], *, required: bool) -> dict[str, Any]:
    interpreter = runtime.get("interpreter", {})
    compiled = interpreter.get("Py_GIL_DISABLED") == 1
    disabled = interpreter.get("gil_enabled") is False
    qualified = not required or (compiled and disabled)
    return {"required_for_samehost": required, "compiled_free_threaded": compiled,
            "gil_disabled_after_cli": disabled, "qualified": qualified,
            "reason": ("ordinary single-card control permits either interpreter build"
                       if not required else "free-threaded build and disabled GIL after CLI imports required")}


#: Input files hashed at once.  Each file is one sequential read on its own
#: thread (hashlib releases the interpreter while it digests a block).
INPUT_HASH_WORKERS = 8


def input_record(path: Path) -> dict[str, Any]:
    """The input's real SHA-256, computed once per file version (woof.hex.file_digest).

    THE COST THIS REMOVES, MEASURED (G3, 2026-10-12, conus3km 18 h preset,
    66 GB of grid, static, init and 7 boundary files): the instrument read
    every byte serially before the door started (127.1 s) and again after
    the run (105.5 s), and the door then re-hashed the same files for its
    own authority record.  A file version is its device, inode, size,
    modification and status-change time; any write changes the status-change
    time, which no user call can set back, so a changed input is still read
    again and its new digest still differs.  ``WOOF_HEX_DIGEST_CACHE=off``
    restores a full read on every call.
    """
    from woof.hex.file_digest import sha256_file

    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def input_snapshot(arguments: Any) -> dict[str, Any]:
    paths = [getattr(arguments, role, None) for role in ("grid", "static", "init")]
    lbc = getattr(arguments, "lbc_dir", None)
    if lbc is not None and lbc.is_dir():
        paths.extend(sorted(lbc.glob("lbc.*.nc")))
    files = [path for path in paths if path is not None and path.is_file()]
    if len(files) <= 1:
        return {str(path): input_record(path) for path in files}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(INPUT_HASH_WORKERS, len(files)),
                            thread_name_prefix="hex-input-digest") as pool:
        records = list(pool.map(input_record, files))
    return {str(path): record for path, record in zip(files, records)}


@contextmanager
def measurement_scope(*, enabled: bool, dt: float | None, card: str | None,
                      cumulus_scheme: str | None = "gf"):
    """Label a run as a measurement.  Opens nothing: there is no gate left.

    This used to enter a candidate timestep mint and swap the regional opener
    for the bare assembler, because the ordinary door refused unanchored
    timesteps and unminted regional classes.  Both refusals were retired on
    2026-10-10, so the scope only records what the caller declared.
    """
    if not enabled:
        yield None
        return
    yield {"measurement_mint": True, "dt_seconds": dt, "card": card,
           "cumulus_scheme": cumulus_scheme, "gate_opened": None}


@contextmanager
def boundary_receipts(door: Any, every: int):
    """Ask the existing driver to record boundaries without changing its step."""
    previous = door.build_driver_argv

    def build(request):
        return [*previous(request), "--fingerprint-every", str(every)]

    door.build_driver_argv = build
    try:
        yield
    finally:
        door.build_driver_argv = previous


class _FinalForcingObservation:
    """Keep only the latest committed next-step forcing until final hashing."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.latest = None
        self.metadata = {"enabled": enabled, "successful_steps_observed": 0,
                         "composite_step_restored": True, "wrapper_intact": True,
                         "model_time_seconds": None, "hashes": None}

    def capture_hashes(self, proof: Any, cp: Any = None) -> None:
        if self.latest is None:
            return
        if cp is None:
            import cupy as cp
        carrier = self.latest
        hashes = {}
        for name in ("rthdynten", "rqvdynten"):
            array = proof._host_array(getattr(carrier, name), cp)
            hashes[name] = {"dtype": array.dtype.str, "shape": list(array.shape),
                            "sha256": proof.array_sha256(array)}
        self.metadata["model_time_seconds"] = float(carrier.time_seconds)
        self.metadata["hashes"] = hashes


@contextmanager
def final_forcing_observer(*, enabled: bool, proof: Any = None):
    """Observe the unchanged composite result; perform no per-step transfers."""
    observation = _FinalForcingObservation(enabled)
    if not enabled:
        yield observation
        return
    if proof is None:
        from woof.hex.drivers.run_cuda_v841_forecast import proof
    previous = proof.execute_composite_step

    def execute(*args, **kwargs):
        result = previous(*args, **kwargs)
        observation.latest = result.dynamics_tendencies
        observation.metadata["successful_steps_observed"] += 1
        return result

    proof.execute_composite_step = execute
    try:
        yield observation
    finally:
        observation.metadata["wrapper_intact"] = proof.execute_composite_step is execute
        proof.execute_composite_step = previous
        observation.metadata["composite_step_restored"] = proof.execute_composite_step is previous
        observation.latest = None


def _profile_summary(stats: pstats.Stats) -> dict[str, Any]:
    functions = []
    for (filename, line, name), (primitive, calls, own, cumulative, _callers) in stats.stats.items():
        functions.append({"file": filename, "line": line, "function": name,
                          "primitive_calls": primitive, "calls": calls,
                          "self_seconds": own, "cumulative_seconds": cumulative})
    functions.sort(key=lambda entry: entry["cumulative_seconds"], reverse=True)
    categories = {
        "device_properties": ("getdeviceproperties", "getdeviceattribute"),
        "host_device_copies": ("asnumpy", "memcpy", "copyto"),
        "synchronization": ("synchronize", "wait_event", "waitevent"),
        "kernel_submission": ("launch", "rawkernel", "raw_kernel"),
        "thread_submission": ("submit", "future", "_worker"),
    }
    selected = {}
    for category, markers in categories.items():
        matches = [entry for entry in functions if any(marker in entry["function"].lower() for marker in markers)]
        selected[category] = {"calls": sum(entry["calls"] for entry in matches),
                              "self_seconds": sum(entry["self_seconds"] for entry in matches),
                              "functions": matches}
    return {"function_entries": len(functions), "total_calls": stats.total_calls,
            "total_self_seconds": stats.total_tt, "top_cumulative": functions[:60],
            "selected_functions": selected,
            "timing_kind": "host cProfile; cumulative times overlap and are not GPU kernel durations"}


def _write_profile(stats: pstats.Stats, root: Path, name: str) -> dict[str, Any]:
    path = root / f"{name}.pstats"
    stats.dump_stats(str(path))
    text = io.StringIO()
    stats.stream = text
    stats.sort_stats("cumulative").print_stats(80)
    (root / f"{name}.txt").write_text(text.getvalue(), encoding="utf-8")
    summary = _profile_summary(stats)
    (root / f"{name}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"pstats": str(path), "summary": summary}


class _WorkerProfile:
    """Per-thread host call timing in the standard pstats interchange format.

    The interpreter has one cProfile monitoring slot. This collector uses
    sys.setprofile's thread-local hook while the main cProfile owns that slot.
    It ignores unmatched activation returns rather than inventing a call.
    """

    def __init__(self):
        self._stack = []
        self._timings = {}
        self.stats = {}

    def _event(self, frame, event, argument):
        now = time.perf_counter()
        if event == "call":
            code = frame.f_code
            key = (code.co_filename, code.co_firstlineno, code.co_name)
            token = ("python", id(frame), key)
        elif event == "c_call":
            name = getattr(argument, "__qualname__", getattr(argument, "__name__", type(argument).__name__))
            module = getattr(argument, "__module__", None)
            key = ("~", 0, f"{module}.{name}" if module else str(name))
            token = ("c", id(frame), key)
        elif event in ("return", "c_return", "c_exception"):
            if not self._stack:
                return
            kind = "python" if event == "return" else "c"
            top = self._stack[-1]
            if top[0][0] != kind or top[0][1] != id(frame):
                return
            token, started, children, primitive, parent = self._stack.pop()
            key = token[2]
            cumulative = now - started
            own = max(0., cumulative - children)
            entry = self._timings.setdefault(key, [0, 0, 0., 0., {}])
            entry[0] += int(primitive)
            entry[1] += 1
            entry[2] += own
            entry[3] += cumulative
            if parent is not None:
                caller = entry[4].setdefault(parent, [0, 0, 0., 0.])
                caller[0] += int(primitive)
                caller[1] += 1
                caller[2] += own
                caller[3] += cumulative
            if self._stack:
                self._stack[-1][2] += cumulative
            return
        else:
            return
        primitive = not any(item[0][2] == key for item in self._stack)
        parent = self._stack[-1][0][2] if self._stack else None
        self._stack.append([token, now, 0., primitive, parent])

    def runcall(self, operation, *args):
        previous_hook = sys.getprofile()
        sys.setprofile(self._event)
        try:
            return operation(*args)
        finally:
            sys.setprofile(previous_hook)

    def create_stats(self):
        self.stats = {key: (*value[:4], {caller: tuple(counts) for caller, counts in value[4].items()})
                      for key, value in self._timings.items()}


_PROFILER_BUSY = "Another profiling tool is already active"


def _profiler_slot_taken() -> bool:
    """Whether a profiler already holds this thread's cProfile slot.

    Python 3.12+ runs cProfile on the single sys.monitoring PROFILER_ID slot
    and refuses a second enable with ValueError.  Python 3.11 runs it on the
    thread's sys.setprofile hook and a second enable silently REPLACES the
    first -- the caller's profile goes dead with no error.  THE BREAKAGE THE
    CHECK PREVENTS: on 3.11 a --profile run under an outside profiler took
    over that profiler's hook (the CI 3.11 leg caught it).  One answer on
    every interpreter, asked before anything is wrapped.
    """
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is not None:
        return monitoring.get_tool(monitoring.PROFILER_ID) is not None
    return sys.getprofile() is not None


@contextmanager
def profile_scope(directory: Path | None):
    """Profile the CLI and each worker invocation only under an explicit flag."""
    if directory is None:
        yield {"enabled": False, "worker_call_restored": True}
        return
    if _profiler_slot_taken():
        raise ValueError(f"{_PROFILER_BUSY}: --profile needs the interpreter's "
                         "profiler slot, and another profiler holds it")
    directory.mkdir(parents=True, exist_ok=False)
    # An e6 control tree has no same-host worker module. It still gets the
    # same CLI profile without importing a lane module from another checkout.
    try:
        from woof.hex.partition_samehost_v841 import SameHostRanks
    except ModuleNotFoundError as error:
        if error.name != "woof.hex.partition_samehost_v841":
            raise
        SameHostRanks = None
    previous = None if SameHostRanks is None else SameHostRanks._call
    previous_code = None if previous is None else hashlib.sha256(marshal.dumps(previous.__code__)).hexdigest()
    lock = threading.Lock()
    profiles = []
    active = 0
    result = {"enabled": True, "directory": str(directory), "worker_calls": 0,
              "worker_original_code_sha256": previous_code}

    def call(group, rank, operation):
        nonlocal active
        profile = cProfile.Profile()
        backend = "cProfile"
        with lock:
            active += 1
        try:
            try:
                profile.enable()
            except ValueError:
                if not _profiler_slot_taken():
                    raise
                profile = _WorkerProfile()
                backend = "sys.setprofile worker collector (cProfile monitoring slot already active)"
                return profile.runcall(previous, group, rank, operation)
            return previous(group, rank, operation)
        finally:
            if backend == "cProfile":
                profile.disable()
            with lock:
                profiles.append((int(rank), profile, backend))
                active -= 1

    main_profile = cProfile.Profile()
    main_profile.enable()
    if SameHostRanks is not None:
        SameHostRanks._call = call
    try:
        yield result
    finally:
        main_profile.disable()
        wrapper_intact = SameHostRanks is None or SameHostRanks._call is call
        if SameHostRanks is not None:
            SameHostRanks._call = previous
        with lock:
            completed = list(profiles)
            calls_active = active
        result["worker_calls"] = len(completed)
        result["workers_drained"] = calls_active == 0
        result["worker_call_restored"] = SameHostRanks is None or SameHostRanks._call is previous
        result["profile_wrapper_intact"] = wrapper_intact
        result["worker_restored_code_sha256"] = None if previous is None else hashlib.sha256(marshal.dumps(SameHostRanks._call.__code__)).hexdigest()
        result["main"] = _write_profile(pstats.Stats(main_profile, stream=io.StringIO()), directory, "main")
        workers = pstats.Stats(stream=io.StringIO())
        for _rank, profile, _backend in completed:
            workers.add(profile)
        result["ranks"] = _write_profile(workers, directory, "ranks")
        result["rank_profile_status"] = "measured" if completed else "no same-host worker calls reached"
        result["worker_profiler_backends"] = {backend: sum(1 for _rank, _profile, kind in completed if kind == backend)
                                              for backend in sorted({backend for _rank, _profile, backend in completed})}
        for rank in sorted({rank for rank, _profile, _backend in completed}):
            rank_stats = pstats.Stats(stream=io.StringIO())
            for candidate, profile, _backend in completed:
                if candidate == rank:
                    rank_stats.add(profile)
            _write_profile(rank_stats, directory, f"rank-{rank}")
        (directory / "profile-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if calls_active:
            raise RuntimeError("profile scope closed before same-host worker calls drained")


def _explicit_free_threaded_runtime() -> bool:
    return (sysconfig.get_config_var("Py_GIL_DISABLED") == 1
            and (os.environ.get("PYTHON_GIL") == "0"
                 or str(sys._xoptions.get("gil")) == "0"))


def _load_serial_io_policy():
    """Load the stdlib-only policy by path before any hex reader imports."""
    path = Path(__file__).with_name("samehost_serial_io_v841.py")
    spec = importlib.util.spec_from_file_location("_hex_samehost_serial_io_measurement", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"explicit no-GIL measurement requires the serial I/O helper at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.coordinator_only_extension_io


def _find_dt_admission(receipt: Any) -> Any:
    """The driver's recorded anchor evidence (``dt_admission``), wherever it sits."""
    if isinstance(receipt, dict):
        if "dt_admission" in receipt:
            return receipt["dt_admission"]
        for value in receipt.values():
            found = _find_dt_admission(value)
            if found is not None:
                return found
    return None


def _driver_result(out: Path, cuda_name: str):
    """Read the selected door's receipt, including native MPAS rank completion."""
    path = out / cuda_name
    if path.is_file():
        value = json.loads(path.read_text())
        return path, value, value.get('status')
    path = out / 'native-physics-receipt.json'
    if not path.is_file():
        return path, None, None
    value = json.loads(path.read_text())
    complete = value.get('schema') == 'mpas-native-forecast-v1' and value.get('completed') is True
    ranks = int(value.get('rank_execution', {}).get('ranks', 1))
    if ranks > 1:
        launch = out / 'samehost-rank-launch.json'
        if not launch.is_file():
            complete = False
        else:
            record = json.loads(launch.read_text())
            complete = complete and record.get('completed') is True and all(
                code == 0 for code in record.get('rank_process_exit_codes', {}).values())
        for rank in range(ranks):
            rank_path = out / f'native-physics-receipt.rank{rank}.json'
            if not rank_path.is_file():
                complete = False
            else:
                record = json.loads(rank_path.read_text())
                complete = complete and record.get('schema') == 'mpas-native-forecast-v1' and record.get('completed') is True
    return path, value, 'passed' if complete else 'incomplete'


def _run(arguments: argparse.Namespace, forecast_argv: list[str]) -> tuple[Path, dict[str, Any]]:
    from woof.hex import cli, forecast_door as door, time_ledger

    if forecast_argv and forecast_argv[0] == "--":
        forecast_argv = forecast_argv[1:]
    if not forecast_argv or forecast_argv[0] != "forecast":
        forecast_argv = ["forecast", *forecast_argv]
    if arguments.dt is not None and "--dt" not in forecast_argv:
        forecast_argv += ["--dt", repr(float(arguments.dt))]
    if arguments.devices is not None:
        if "--samehost-devices" in forecast_argv:
            raise ValueError("give devices once, through --devices or the forecast flag")
        forecast_argv += ["--samehost-devices", arguments.devices]
    parsed = cli.build_parser().parse_args(forecast_argv)
    if arguments.fingerprint_every < 1:
        raise ValueError("the identity instrument requires a positive fingerprint cadence")
    output = arguments.report or parsed.out.parent / (parsed.out.name + ".measurement.json")
    if output.exists():
        raise FileExistsError(f"measurement receipt already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Identity reads used to fall outside the instrument's wall timer; their
    # cost is in the same ledger.  Sources are read in full every time; large
    # inputs are digested once per file version (input_record).
    with time_ledger.stage("prep.instrument_identity"):
        before = {"source": source_snapshot(), "runtime": runtime_snapshot(parsed.gpuwm_checkout),
                  "inputs": input_snapshot(parsed)}
    previous_open = None
    previous_timesteps = None
    from woof.hex import dt_admission, cuda_regional_forecast_v841 as regional
    previous_open = regional.open_regional_forecast_v841
    previous_timesteps = dt_admission.ADMITTED_TIMESTEPS
    previous_builder = door.build_driver_argv
    started = time.monotonic()
    code, error = 2, None
    profile = {"enabled": False, "worker_call_restored": True}
    forcing = _FinalForcingObservation(False)
    try:
        with ExitStack() as contexts:
            contexts.enter_context(measurement_scope(
                enabled=arguments.measurement_mint, dt=arguments.dt, card=arguments.card,
                cumulus_scheme={"off": None, "ntiedtke": "ntiedtke"}.get(parsed.convection, "gf"),
            ))
            contexts.enter_context(boundary_receipts(door, arguments.fingerprint_every))
            profile = contexts.enter_context(profile_scope(getattr(arguments, "profile_dir", None)))
            forcing = contexts.enter_context(final_forcing_observer(
                enabled=getattr(parsed, "samehost_devices", None) is None))
            code = int(cli.main(forecast_argv))
            if forcing.enabled and forcing.latest is not None:
                from woof.hex.drivers.run_cuda_v841_forecast import proof
                forcing.capture_hashes(proof)
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
    with time_ledger.stage("post.instrument_identity"):
        after = {"source": source_snapshot(), "runtime": runtime_snapshot(parsed.gpuwm_checkout),
                 "inputs": input_snapshot(parsed)}
    restored = (regional.open_regional_forecast_v841 is previous_open
                and dt_admission.ADMITTED_TIMESTEPS is previous_timesteps
                and door.build_driver_argv is previous_builder
                and profile.get("worker_call_restored", False)
                and profile.get("profile_wrapper_intact", True)
                and forcing.metadata["composite_step_restored"]
                and forcing.metadata["wrapper_intact"])
    driver_path, driver_receipt, status = _driver_result(parsed.out, door.DRIVER_RECEIPT_NAME)
    completed = code == 0 and status == "passed"
    forecast_receipt = driver_receipt.get("forecast", {}) if driver_receipt else {}
    resolved_parallel = (len(forecast_receipt.get("devices", ())) > 1
                         or int((driver_receipt or {}).get('rank_execution', {}).get('ranks', 1)) > 1)
    final_forcing_hashes = (forcing.metadata["hashes"] if forcing.enabled and not resolved_parallel else
                           forecast_receipt.get("final_gf_tendency_hashes"))
    unchanged = _protected_snapshot(before) == _protected_snapshot(after)
    runtime_qualification = _parallel_runtime_qualification(
        after["runtime"], required=resolved_parallel or getattr(parsed, "samehost_devices", None) is not None)
    identity_eligible = completed and runtime_qualification["qualified"]
    receipt = {
        "schema": "mpas-port.cuda-v841-real-door-measurement/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "tip": arguments.tip,
        "command_argv": forecast_argv, "cli_exit_code": code, "driver_status": status,
        "driver_completed": completed, "exception": error,
        "run_identity_eligible": identity_eligible,
        "parallel_runtime_qualification": runtime_qualification,
        "wall_seconds": time.monotonic() - started,
        "measurement_mint": bool(arguments.measurement_mint),
        "requested_dt_seconds": arguments.dt,
        "card": arguments.card,
        "timestep_anchor": _find_dt_admission(driver_receipt),
        "fingerprint_every": arguments.fingerprint_every,
        "profile": profile,
        "final_forcing_observation": forcing.metadata,
        "final_gf_tendency_hashes": final_forcing_hashes,
        "source_runtime_inputs_before": before, "source_runtime_inputs_after": after,
        "protected_bytes_unchanged": unchanged, "ordinary_door_restored": restored,
        "driver_receipt": str(driver_path),
        "boundary_receipts": str(parsed.out / "boundary-fingerprints.jsonl"),
        "nonclaims": ["native MPAS whole-model identity"],
    }
    return output, receipt


def run(arguments: argparse.Namespace, forecast_argv: list[str]) -> int:
    """Write the primary receipt only after every selected policy restores."""
    enabled = _explicit_free_threaded_runtime()
    guard = None
    payload = None
    cleanup_error = None
    try:
        if enabled:
            policy = _load_serial_io_policy()
            with policy() as guard:
                payload = _run(arguments, forecast_argv)
        else:
            payload = _run(arguments, forecast_argv)
    except Exception as failure:
        if payload is None:
            raise
        cleanup_error = {"type": type(failure).__name__, "message": str(failure)}
    output, receipt = payload
    if enabled:
        serial_io = {"enabled": True, **guard.receipt()}
    else:
        serial_io = {"enabled": False, "module_entrypoints_restored": True,
                     "worker_refusals": 0,
                     "policy": "not selected: ordinary interpreter policy unchanged"}
    restored = serial_io.get("module_entrypoints_restored") is True
    qualified = restored and serial_io.get("worker_refusals") == 0 and cleanup_error is None
    serial_io["qualified"] = qualified
    receipt["serial_extension_io"] = serial_io
    receipt["ordinary_door_restored"] = receipt["ordinary_door_restored"] and restored
    receipt["run_identity_eligible"] = receipt["run_identity_eligible"] and qualified
    if cleanup_error is not None:
        receipt["instrument_cleanup_exception"] = cleanup_error
        if receipt["exception"] is None:
            receipt["exception"] = cleanup_error
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"measurement_receipt": str(output), "driver_completed": receipt["driver_completed"],
                      "run_identity_eligible": receipt["run_identity_eligible"],
                      "protected_bytes_unchanged": receipt["protected_bytes_unchanged"],
                      "ordinary_door_restored": receipt["ordinary_door_restored"],
                      "serial_extension_io_qualified": qualified}), flush=True)
    code = receipt["cli_exit_code"]
    unchanged, restored = receipt["protected_bytes_unchanged"], receipt["ordinary_door_restored"]
    return code if code and unchanged and restored else (0 if receipt["run_identity_eligible"] and unchanged and restored else 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-mint", action="store_true")
    parser.add_argument("--dt", type=float, default=None,
                        help="forwarded to the forecast as --dt when its arguments carry none")
    parser.add_argument("--card")
    parser.add_argument("--devices")
    parser.add_argument("--fingerprint-every", type=int, default=1)
    parser.add_argument("--profile-dir", type=Path,
                        help="explicit host cProfile run, separate from unprofiled benchmark arms")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tip")
    arguments, forecast = parser.parse_known_args(argv)
    return run(arguments, forecast)


if __name__ == "__main__":
    raise SystemExit(main())
