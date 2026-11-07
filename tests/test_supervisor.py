"""CPU/Windows gates for the Phase-5 fresh-process supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import woof.case_data as case_data
import woof.supervisor as supervisor
from woof.supervisor import (
    HEARTBEAT_SCHEMA, GPUAlreadyLockedError, GPUFileLock, GPUIdentity,
    GPUPreflightError, GPUProcess, Heartbeat, RollingStepWall,
    RuntimeHeartbeat, SupervisorError, atomic_publish_file, atomic_write_json,
    _current_transition_receipt, _parse_pmon_output, is_cuda_fatal,
    parse_compute_apps_output,
    preflight_exclusive_gpu,
    quarantine_file, read_heartbeat, stale_threshold_seconds,
    supervise_experiment, validate_manifest_checkpoint, write_heartbeat,
)


def _heartbeat(**updates):
    values = dict(
        schema=HEARTBEAT_SCHEMA, run_id="run-1", config_digest="a" * 64,
        pid=max(os.getpid(), 1), started_at_utc="2026-07-16T00:00:00Z",
        updated_at_utc="2026-07-16T00:00:01Z", status="integrating",
        model_elapsed_seconds=60.0, outer_step=1,
        last_durable_wrfout=None, last_checkpoint=None)
    values.update(updates)
    return Heartbeat(**values)


def test_heartbeat_exact_schema_round_trip(tmp_path):
    path = tmp_path / "run-progress.json"
    write_heartbeat(path, _heartbeat(last_durable_wrfout="wrfout.nc"))
    assert read_heartbeat(path) == _heartbeat(
        last_durable_wrfout="wrfout.nc")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema", "run_id", "config_digest", "pid", "started_at_utc",
        "updated_at_utc", "status", "model_elapsed_seconds", "outer_step",
        "last_durable_wrfout", "last_checkpoint",
    }


def test_supervisor_ignores_stale_transition_receipt(tmp_path):
    path = tmp_path / "microphysics-transitions.json"
    payload = {
        "schema": "gpuwm.microphysics-transitions/v1",
        "status": "PASS",
        "run_id": "old-run",
        "config_digest": "a" * 64,
        "transitions": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _current_transition_receipt(
        tmp_path, "new-run", "a" * 64) == (None, None)
    assert _current_transition_receipt(
        tmp_path, "old-run", "b" * 64) == (None, None)
    current, digest = _current_transition_receipt(
        tmp_path, "old-run", "a" * 64)
    assert current == path.resolve()
    assert digest is not None and len(digest) == 64


def test_heartbeat_remains_valid_json_when_writer_is_killed_mid_publish(
        tmp_path):
    """Terminate a separate Windows process after fsync but before replace."""
    final = tmp_path / "run-progress.json"
    ready = tmp_path / "ready"
    atomic_write_json(final, {"generation": 1})
    code = "\n".join((
        "import time",
        "from pathlib import Path",
        "from woof.supervisor import atomic_write_json",
        f"final = Path({str(final)!r})",
        f"ready = Path({str(ready)!r})",
        "def pause(_):",
        "    ready.write_text('ready', encoding='utf-8')",
        "    while True:",
        "        time.sleep(0.01)",
        "atomic_write_json(final, {'generation': 2}, _before_replace=pause)",
    ))
    process = subprocess.Popen(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, close_fds=True)
    deadline = time.monotonic() + 10.0
    while not ready.exists() and process.poll() is None:
        if time.monotonic() > deadline:
            process.kill()
            pytest.fail(process.stderr.read().decode("utf-8", errors="replace"))
        time.sleep(0.01)
    process.terminate()
    process.wait(timeout=10)
    assert json.loads(final.read_text(encoding="utf-8")) == {"generation": 1}
    orphans = list(tmp_path.glob("run-progress.json.tmp.*"))
    assert len(orphans) == 1
    assert quarantine_file(orphans[0], reason="kill-test").exists()


def test_atomic_publication_never_exposes_incomplete_final(tmp_path):
    final = tmp_path / "wrfout.nc"
    final.write_bytes(b"old-complete")

    def partial(path):
        path.write_bytes(b"readable-but-incomplete")

    def reject(path):
        assert path.read_bytes() == b"readable-but-incomplete"
        raise ValueError("missing completion attribute")

    with pytest.raises(ValueError, match="completion"):
        atomic_publish_file(final, partial, reject)
    assert final.read_bytes() == b"old-complete"
    assert not list(tmp_path.glob(".wrfout.nc.tmp.*"))
    quarantined = list((tmp_path / ".quarantine").iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"readable-but-incomplete"


def test_atomic_publication_handles_producer_exception(tmp_path):
    final = tmp_path / "wrfout.nc"

    def explode(path):
        path.write_bytes(b"prefix")
        raise OSError("injected disk failure")

    with pytest.raises(OSError, match="injected"):
        atomic_publish_file(final, explode, lambda _: None)
    assert not final.exists()
    assert list((tmp_path / ".quarantine").iterdir())


def test_heartbeat_replace_survives_a_concurrent_reader(tmp_path):
    """A Windows reader without FILE_SHARE_DELETE must not kill the writer."""
    path = tmp_path / "run-progress.json"
    write_heartbeat(path, _heartbeat(outer_step=1))
    opened = threading.Event()

    def hold_reader():
        with path.open("rb"):
            opened.set()
            time.sleep(0.08)

    reader = threading.Thread(target=hold_reader)
    reader.start()
    assert opened.wait(timeout=2.0)
    write_heartbeat(path, _heartbeat(
        updated_at_utc="2026-07-16T00:00:02Z", outer_step=2))
    reader.join(timeout=2.0)
    assert not reader.is_alive()
    assert read_heartbeat(path).outer_step == 2


def _try_lock_in_fresh_process(path: Path) -> subprocess.CompletedProcess:
    code = "\n".join((
        "import sys",
        "from woof.supervisor import GPUAlreadyLockedError, GPUFileLock",
        "try:",
        f"    lock = GPUFileLock('GPU-test', path={str(path)!r}).acquire()",
        "except GPUAlreadyLockedError:",
        "    sys.exit(23)",
        "else:",
        "    lock.release()",
        "    sys.exit(0)",
    ))
    return subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=15, close_fds=True)


def test_gpu_uuid_lock_excludes_a_second_process_on_windows(tmp_path):
    path = tmp_path / "gpu.lock"
    with GPUFileLock("GPU-test", path=path, run_id="owner"):
        blocked = _try_lock_in_fresh_process(path)
        assert blocked.returncode == 23, blocked.stderr
    acquired = _try_lock_in_fresh_process(path)
    assert acquired.returncode == 0, acquired.stderr


def test_lock_contention_raises_specific_error_in_process(tmp_path):
    # POSIX flock is process-scoped on some platforms; this assertion is only
    # about the Windows byte-range implementation used by the target box.
    if os.name != "nt":
        pytest.skip("Windows byte-range lock behavior")
    path = tmp_path / "gpu.lock"
    first = GPUFileLock("GPU-test", path=path).acquire()
    try:
        with pytest.raises(GPUAlreadyLockedError):
            GPUFileLock("GPU-test", path=path).acquire()
    finally:
        first.release()


def test_lock_error_classification_does_not_conflate_disk_errors():
    import errno
    import woof.supervisor as supervisor

    contention = errno.EACCES
    assert supervisor._is_lock_contention(OSError(contention, "held"))
    assert not supervisor._is_lock_contention(OSError(errno.EIO, "disk"))


def test_captured_wddm_compute_apps_fixture_parses_desktop_rows():
    fixture = (Path(__file__).with_name("data") /
               "nvidia_smi_wddm_compute_apps_61074.csv")
    processes = parse_compute_apps_output(
        fixture.read_text(encoding="utf-8"))
    assert len(processes) == 29
    assert all(process.uuid.startswith("GPU-") for process in processes)
    assert all(process.used_gpu_memory_mib is None for process in processes)
    assert any(process.process_name == "[Insufficient Permissions]"
               for process in processes)
    assert any(process.process_name.endswith("explorer.exe")
               for process in processes)


@pytest.mark.parametrize("result, expected", [
    (subprocess.CompletedProcess([], -11, "partial inventory", ""),
     (("SIGSEGV", "signal 11", "return code -11") if os.name == "posix"
      else ("status -11",)) + ("stdout: partial inventory",)),
    (subprocess.CompletedProcess([], 9, "Failed to initialize NVML", ""),
     ("status 9", "stdout: Failed to initialize NVML")),
    (subprocess.CompletedProcess([], -1073741819, "", "access violation"),
     ("status -1073741819", "stderr: access violation")),
    (FileNotFoundError("missing executable"), ("not found on PATH",)),
    (PermissionError("execution denied"),
     ("could not be started", "execution denied", "Check executable access")),
    (subprocess.TimeoutExpired(
        ["nvidia-smi"], 20, output=b"partial\xff", stderr=b"NVML stalled"),
     ("timed out after 20 seconds", "stdout: partial\ufffd", "stderr: NVML stalled")),
])
def test_nvidia_smi_failure_preserves_diagnostics_without_retry(
        monkeypatch, result, expected):
    arguments = ["--query-gpu=index,uuid,driver_version,name",
                 "--format=csv,noheader,nounits"]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 20
        assert kwargs["check"] is False
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    with pytest.raises(GPUPreflightError) as caught:
        supervisor.query_gpus()
    assert calls == [["nvidia-smi", *arguments]]
    message = str(caught.value)
    assert "failed closed" in message
    assert "GPU state checks did not complete" in message
    assert "nvidia-smi in the same shell" in message
    assert repr(arguments) in message
    for text in expected:
        assert text in message
    if os.name == "nt":
        assert "SIGSEGV" not in message


def test_nvidia_smi_success_still_requires_an_actual_gpu_identity(monkeypatch):
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(
                            args, 0, "0, GPU-test, 610.74, RTX 5090\n", ""))
    assert supervisor.query_gpus() == (
        GPUIdentity("GPU-test", "610.74", "RTX 5090", 0),)


_FOUR_CARDS = tuple(GPUIdentity(f"GPU-{i}aaa-card", "615.71", "RTX PRO 6000", i)
                    for i in range(4))


@pytest.mark.parametrize("mask, expected", [
    ("2", 2), ("GPU-3aaa-card", 3), ("GPU-1aaa", 1), (" 0 ", 0)])
def test_one_card_cuda_mask_selects_that_card_on_a_multi_gpu_node(
        monkeypatch, mask, expected):
    # Named breakage: `CUDA_VISIBLE_DEVICES=<card> woof run` on a 4-card
    # node was refused with "--gpu-uuid is required", nvidia-smi listing
    # every card whatever the mask said.
    monkeypatch.setattr(supervisor, "query_gpus", lambda: _FOUR_CARDS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    assert supervisor.select_gpu() == _FOUR_CARDS[expected]


@pytest.mark.parametrize("mask", [None, "", "0,1", "7", "GPU-nope", "MIG-x"])
def test_multi_gpu_node_without_one_named_card_still_refuses(monkeypatch, mask):
    monkeypatch.setattr(supervisor, "query_gpus", lambda: _FOUR_CARDS)
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(GPUPreflightError) as refused:
        supervisor.select_gpu()
    assert "GPU-0aaa-card" in str(refused.value)


def test_cuda_ordinal_on_a_mixed_node_needs_bus_order(monkeypatch):
    mixed = (GPUIdentity("GPU-a", "615.71", "RTX 5090", 0),
             GPUIdentity("GPU-b", "615.71", "RTX 3080", 1))
    monkeypatch.setattr(supervisor, "query_gpus", lambda: mixed)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    with pytest.raises(GPUPreflightError, match="PCI_BUS_ID"):
        supervisor.select_gpu()
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert supervisor.select_gpu() == mixed[1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b")
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    assert supervisor.select_gpu() == mixed[1]


def test_explicit_uuid_wins_over_the_cuda_mask(monkeypatch):
    monkeypatch.setattr(supervisor, "query_gpus", lambda: _FOUR_CARDS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert supervisor.select_gpu("GPU-1aaa-card") == _FOUR_CARDS[1]


@pytest.mark.parametrize("output", ["", "Failed to initialize NVML\n"])
def test_nvidia_smi_zero_exit_with_no_inventory_is_not_success(
        monkeypatch, output):
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, output, ""))
    with pytest.raises(GPUPreflightError):
        supervisor.query_gpus()


def test_captured_wddm_pmon_fixture_parses_exact_twelve_columns():
    fixture = (Path(__file__).with_name("data") /
               "nvidia_smi_wddm_pmon_61074.txt")
    modes = _parse_pmon_output(fixture.read_text(encoding="utf-8"))
    # Command names are anonymized; all twelve captured numeric columns remain.
    assert len(modes) == 27
    assert modes[632] == ("C+G", 0)
    assert modes[12888] == ("C+G", 0)
    assert modes[39444] == ("C", 0)


def _priced_device(monkeypatch, *, total_gib=32.0, used_gib=4.0):
    """Report one device's NVML total/used to the admission function."""

    import woof.core.preflight as preflight

    monkeypatch.setattr(
        preflight, "device_physical_total_bytes",
        lambda **kwargs: int(total_gib * 1024 ** 3))
    monkeypatch.setattr(
        preflight, "device_wide_used_bytes",
        lambda **kwargs: int(used_gib * 1024 ** 3))


@pytest.fixture
def warnings_seen():
    """Every woof.explain warning raised inside one test, as records."""

    from woof import explain

    seen = []
    explain.add_warning_observer(seen.append)
    try:
        yield seen
    finally:
        explain.remove_warning_observer(seen.append)


def test_wddm_preflight_prices_an_unmeasured_pure_cuda_cotenant(
        monkeypatch, warnings_seen):
    """Rewritten from test_wddm_preflight_rejects_pure_cuda_with_dash_or_zero_fb.

    That test pinned the rule this change retires: refuse because the
    per-process framebuffer is unmeasured.  Only the PER-PROCESS figure is
    unmeasured; the DEVICE-wide figure is measurable from the same tool and
    counts the co-tenant's memory either way, so the unmeasured branch is
    now the priced branch.
    """

    import woof.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    modes = _parse_pmon_output(
        "0 41001 C - - - - - - - 0 python.exe\n"
        "0 41005 C 25 0 - - - - 0 0 python.exe\n")
    assert modes == {41001: ("C", None), 41005: ("C", 0)}
    processes = tuple(
        GPUProcess(gpu.uuid, pid, "python.exe", memory, mode)
        for pid, (mode, memory) in modes.items())
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)
    _priced_device(monkeypatch)
    seen = warnings_seen

    receipt = preflight_exclusive_gpu(
        gpu.uuid, reservation_bytes=8 * 2 ** 30)
    assert receipt["verdict"] == "admitted"
    assert receipt["device_free_bytes"] == 28 * 2 ** 30
    assert len(seen) == 1
    message = seen[0]["action"]
    assert "8.00 GiB" in message
    assert "28.00 GiB" in message
    assert "pid=41001" in message
    assert "unsupported" not in message


def test_a_reservation_that_does_not_fit_is_refused_with_both_numbers(
        monkeypatch):
    import woof.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    processes = (GPUProcess(gpu.uuid, 41001, "python.exe", None, "C"),)
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)
    _priced_device(monkeypatch)

    with pytest.raises(GPUPreflightError) as caught:
        preflight_exclusive_gpu(gpu.uuid, reservation_bytes=30 * 2 ** 30)
    message = str(caught.value)
    assert "30.00 GiB" in message
    assert "28.00 GiB" in message
    assert "unsupported" not in message
    assert "pid=41001" in message

    # A run that has already started is re-measured and named, never
    # re-refused: the admission decision belongs before the first worker.
    receipt = preflight_exclusive_gpu(
        gpu.uuid, reservation_bytes=30 * 2 ** 30, decide=False)
    assert receipt["verdict"] == "admitted-already-started"


def test_nvidia_smi_failure_still_fails_closed_on_a_shared_card(monkeypatch):
    import woof.core.preflight as preflight
    import woof.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    processes = (GPUProcess(gpu.uuid, 41001, "python.exe", None, "C"),)
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)

    def _fails(**kwargs):
        raise GPUPreflightError("GPU preflight failed closed: nvidia-smi")

    monkeypatch.setattr(preflight, "device_wide_used_bytes", _fails)
    with pytest.raises(GPUPreflightError, match="failed closed"):
        preflight_exclusive_gpu(gpu.uuid, reservation_bytes=1 * 2 ** 30)


def test_wddm_preflight_permits_c_plus_g_pmon_rows(monkeypatch):
    import woof.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    modes = _parse_pmon_output(
        "0 41002 C+G - - - - - - - 0 explorer.exe\n"
        "0 41003 C+G - - - - - - 0 0 chrome.exe\n")
    processes = tuple(
        GPUProcess(gpu.uuid, pid, f"desktop-{pid}.exe", memory, mode)
        for pid, (mode, memory) in modes.items())
    assert modes == {41002: ("C+G", None), 41003: ("C+G", 0)}
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)
    preflight_exclusive_gpu(gpu.uuid)


def test_parse_pmon_output_rejects_malformed_column_count():
    with pytest.raises(GPUPreflightError, match="malformed pmon row"):
        _parse_pmon_output("0 41004 C - - - - - - 0 bad.exe\n")


def test_linux_pmon_idle_row_is_an_explicit_empty_sample():
    idle = (
        "# gpu pid type sm mem enc dec jpg ofa fb ccpm command\n"
        "0 - - - - - - - - - - -\n"
    )
    assert _parse_pmon_output(idle) == {}
    with pytest.raises(GPUPreflightError, match="malformed pmon PID"):
        _parse_pmon_output("0 - C - - - - - - - - python\n")


def test_wddm_preflight_permits_graphics_and_prices_large_pure_cuda(
        monkeypatch):
    """Rewritten from test_wddm_preflight_permits_graphics_and_rejects_large_pure_cuda.

    Graphics rows are still permitted outright.  A substantial pure-C row is
    still the thing that makes the card shared, but the verdict is now the
    priced one, and --allow-shared-gpu is redundant rather than the only way
    in.
    """

    import woof.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    processes = (
        GPUProcess(gpu.uuid, 101, "explorer.exe", None, "C+G"),
        GPUProcess(gpu.uuid, 102, "python.exe", 16, "C"),
        GPUProcess(gpu.uuid, 103, "python.exe", 512, "C"),
    )
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)
    _priced_device(monkeypatch)

    # A run that does not fit is refused, and pid=103 is named in it.
    with pytest.raises(GPUPreflightError, match="pid=103"):
        preflight_exclusive_gpu(gpu.uuid, reservation_bytes=30 * 2 ** 30)
    # The same card admits a run that fits, with no flag at all.
    assert preflight_exclusive_gpu(
        gpu.uuid, reservation_bytes=8 * 2 ** 30)["verdict"] == "admitted"
    # Approving the pid makes the card exclusive again, and an exclusive
    # card asks the device nothing.
    assert preflight_exclusive_gpu(
        gpu.uuid, approved_pids={103})["verdict"] == "exclusive"
    # The flag is now redundant: same verdict with and without it.
    assert preflight_exclusive_gpu(
        gpu.uuid, reservation_bytes=8 * 2 ** 30,
        allow_shared_gpu=True)["verdict"] == "admitted"


def test_stale_threshold_is_three_p99_with_120_second_floor():
    assert stale_threshold_seconds([1.0, 2.0, 40.0]) == 120.0
    assert stale_threshold_seconds([1.0, 2.0, 50.0]) == 150.0
    history = RollingStepWall(maxlen=3)
    for value in (1.0, 2.0, 3.0, 60.0):
        history.add(value)
    assert history.p99 == 60.0
    assert history.stale_threshold_seconds == 180.0


def test_runtime_heartbeat_durability_work_runs_once_per_new_artifact(
        monkeypatch, tmp_path):
    import woof.supervisor as supervisor

    wrfout = tmp_path / "wrfout"
    checkpoint = tmp_path / "checkpoint.npz"
    wrfout.write_bytes(b"complete")
    checkpoint.write_bytes(b"manifest-valid")
    calls = {"fsync": 0, "checkpoint": 0}

    def fsync(path):
        calls["fsync"] += 1
        return Path(path)

    def validate(path):
        calls["checkpoint"] += 1
        return Path(path).resolve()

    monkeypatch.setattr(supervisor, "fsync_file", fsync)
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", validate)
    progress = RuntimeHeartbeat(
        tmp_path / "run-progress.json", run_id="run",
        config_sha256="a" * 64, started_at_utc="2026-07-16T00:00:00Z")
    for step in (1, 2):
        progress(
            model_elapsed_seconds=60.0 * step, outer_step=step,
            last_durable_wrfout=wrfout, last_checkpoint=checkpoint)
    assert calls == {"fsync": 1, "checkpoint": 1}


def test_cuda_fatal_classification_policy():
    assert is_cuda_fatal("CUDA_ERROR_ILLEGAL_ADDRESS")
    assert is_cuda_fatal(RuntimeError("unspecified launch failure"))
    assert is_cuda_fatal("device-lost while synchronizing")
    assert not is_cuda_fatal(ValueError("ordinary bad configuration"))


def test_manifest_checkpoint_validation_reads_all_declared_members(tmp_path):
    arrays = {"state/u": np.arange(6, dtype=np.float32).reshape(2, 3)}
    header = {
        "array_manifest": {
            "state/u": {"shape": [2, 3], "dtype": "float32"}},
    }
    payload = {
        "__gpuwm_restart_header__": np.frombuffer(
            json.dumps(header).encode("utf-8"), dtype=np.uint8),
        **arrays,
    }
    path = tmp_path / "checkpoint.npz"
    with path.open("wb") as stream:
        np.savez(stream, **payload)
    assert validate_manifest_checkpoint(path) == path.resolve()

    with np.load(path, allow_pickle=False) as archive:
        tampered = {key: archive[key] for key in archive.files
                    if key != "state/u"}
    broken = tmp_path / "broken.npz"
    with broken.open("wb") as stream:
        np.savez(stream, **tampered)
    with pytest.raises(Exception, match="manifest"):
        validate_manifest_checkpoint(broken)


class _FakeClock:
    def __init__(self, step):
        self.now = 0.0
        self.step = float(step)

    def monotonic(self):
        return self.now

    def sleep(self, _seconds):
        self.now += self.step


class _ScriptedProcess:
    """Popen-shaped worker whose poll calls publish scripted heartbeats."""

    def __init__(self, command, env, script, pid):
        self.command = tuple(command)
        self.env = env
        self.script = list(script)
        self.pid = pid
        self.returncode = None
        self.terminated = False
        outdir = Path(command[command.index("--outdir") + 1])
        self.heartbeat_path = outdir / "run-progress.json"
        self.capsule_path = outdir / "failure-capsule.json"
        self.initial_checkpoint = (
            command[command.index("--restart") + 1]
            if "--restart" in command else None)
        self._generation = 0

    def _publish(self, event):
        self._generation += 1
        status = event["status"]
        checkpoint = event.get("checkpoint", self.initial_checkpoint)
        heartbeat_pid = int(event.get("heartbeat_pid", self.pid))
        write_heartbeat(self.heartbeat_path, Heartbeat(
            HEARTBEAT_SCHEMA, self.env["WOOF_RUN_ID"],
            self.env["WOOF_CONFIG_DIGEST"], heartbeat_pid,
            self.env["WOOF_STARTED_AT_UTC"],
            f"2026-07-16T00:00:{self._generation:02d}Z", status,
            float(event.get("model_seconds", event.get("step", 0) * 60.0)),
            int(event.get("step", 0)), None,
            None if checkpoint is None else str(Path(checkpoint).resolve()),
            restart=event.get("restart")))

    def _publish_capsule(self, capsule):
        """Leave a worker-authored failure capsule, as a real worker does.

        Bound to this attempt the same way the real one is -- the run id
        out of the worker's own environment and this process's pid -- so
        the supervisor's run/pid check is exercised rather than bypassed.
        """
        self.capsule_path.write_text(json.dumps({
            "schema": supervisor.FAILURE_CAPSULE_SCHEMA,
            "run_id": self.env["WOOF_RUN_ID"],
            "worker_pid": self.pid,
            "last_phase": capsule.get("phase", "preparing:load-config"),
            "last_step": 0,
            "exception": {
                "type": capsule["type"],
                "message": capsule["message"],
                "traceback": capsule.get("traceback", "Traceback ...\n"),
                "cuda_fatal": False,
            },
        }), encoding="utf-8")

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.script:
            event = self.script.pop(0)
            if "status" in event:
                self._publish(event)
            if "capsule" in event:
                self._publish_capsule(event["capsule"])
            if "exit" in event:
                self.returncode = int(event["exit"])
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminated = True
        self.returncode = -9


def _install_scripted_supervisor(monkeypatch, tmp_path, scripts, *,
                                 clock_step=1.0,
                                 publish_before_popen_returns=False,
                                 real_capsule_reader=False):
    import woof.supervisor as supervisor

    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname='fake'\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"fixture")
    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    processes = []
    scripts = [list(script) for script in scripts]
    clock = _FakeClock(clock_step)

    class _Lock:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def popen(command, *, env, **_kwargs):
        process = _ScriptedProcess(
            command, env, scripts[len(processes)], 41_000 + len(processes))
        if publish_before_popen_returns:
            process.poll()
        processes.append(process)
        return process

    def hash_inputs(path, **kwargs):
        assert Path(path) == config.resolve()
        assert kwargs["config_bytes"] == config.read_bytes()
        return {}

    monkeypatch.setattr(supervisor, "resolved_input_hashes", hash_inputs)
    monkeypatch.setattr(supervisor, "select_gpu", lambda _uuid: gpu)
    monkeypatch.setattr(supervisor, "preflight_exclusive_gpu",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "GPUFileLock", _Lock)
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint",
                        lambda path: Path(path).resolve())
    if not real_capsule_reader:
        # No capsule reaches disk under this harness, so the reader is
        # short-circuited rather than left to hunt for a file that is
        # never there.  `real_capsule_reader=True` lets a scripted worker
        # publish a genuine capsule and the real reader find it.
        monkeypatch.setattr(supervisor, "_read_worker_failure_capsule",
                            lambda *args, **kwargs: None)
    monkeypatch.setattr(
        supervisor, "write_failure_capsule",
        lambda path, **kwargs: Path(path))
    monkeypatch.setattr(supervisor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(supervisor.time, "sleep", clock.sleep)
    return config, checkpoint, processes


def test_supervise_masks_worker_to_the_selected_physical_uuid(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 1},
            {"status": "complete", "step": 2, "exit": 0},
        ]])
    result = supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert result.attempts == 1 and result.heartbeat.status == "complete"
    assert len(processes) == 1
    assert processes[0].env["CUDA_VISIBLE_DEVICES"] == "GPU-test"


def test_supervise_passes_one_create_only_captured_config_to_worker(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "complete", "step": 1, "exit": 0},
        ]])
    original = config.read_bytes()

    supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)

    command = processes[0].command
    payload_path = Path(
        command[command.index("--config-payload") + 1])
    expected_digest = hashlib.sha256(original).hexdigest()
    assert payload_path.read_bytes() == original
    assert processes[0].env["WOOF_CONFIG_DIGEST"] == expected_digest
    assert command[command.index("--config") + 1] == str(config.resolve())
    config.write_bytes(b"mutated original after parent capture\n")
    assert supervisor._validated_config_payload_bytes(
        payload_path, expected_digest) == original


def test_direct_supervise_does_not_snapshot_inputs_without_multi_run_opt_in(
        monkeypatch, tmp_path):
    monkeypatch.delenv(
        supervisor.SHARED_INPUT_AUTHORITY_ROOT_ENV, raising=False)
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "complete", "step": 1, "exit": 0},
        ]])
    monkeypatch.setattr(
        supervisor, "snapshot_resolved_input_files",
        lambda *_args, **_kwargs: pytest.fail(
            "ordinary woof run must not copy forcing into a CAS"))

    supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)

    assert supervisor.INPUT_AUTHORITIES_ENV not in processes[0].env
    assert not list((tmp_path / "out").glob("input-authorities-*"))


def test_captured_config_publication_is_create_only(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()
    path = supervisor._capture_config_payload(outdir, "run-id", b"first")

    with pytest.raises(FileExistsError):
        supervisor._capture_config_payload(outdir, "run-id", b"second")

    assert path.read_bytes() == b"first"


def test_transient_capture_paths_and_run_ids_do_not_diverge_capsules(tmp_path):
    from woof.certify.capsule import build_capsule, emit_capsule
    from woof.certify.dualrun import compare_capsule_files

    config = tmp_path / "source" / "experiment.toml"
    config.parent.mkdir()
    config_bytes = b"same captured config bytes\n"
    config.write_bytes(config_bytes)
    digest = hashlib.sha256(config_bytes).hexdigest()
    run_a = tmp_path / "arm-a"
    run_b = tmp_path / "arm-b"
    run_a.mkdir()
    run_b.mkdir()
    capture_a = supervisor._capture_config_payload(
        run_a, "independent-run-a", config_bytes)
    capture_b = supervisor._capture_config_payload(
        run_b, "independent-run-b", config_bytes)
    assert capture_a != capture_b

    context_a = supervisor._success_run_context(
        config, digest, {}, restart_interval_seconds=None)
    context_b = supervisor._success_run_context(
        config, digest, {}, restart_interval_seconds=None)
    assert context_a == context_b
    assert "captured_path" not in context_a["config_bytes"]

    common = {
        "emission_site": "supervisor:success",
        "run_context": context_a,
        "run_shape": {
            "route": "supervisor:woof run", "domain_count": 1,
            "run_seconds": 60.0},
        "output": {
            "frames": [{
                "path": "wrfout_d01_0000", "bytes": 4,
                "sha256": "8" * 64}],
            "trajectory_digest": {"d01": "9" * 64}},
        "receipts": {"run_progress": {"path": "run-progress.json"}},
        "require_gpu": False,
    }
    capsule_a = emit_capsule(run_a, build_capsule(**common))
    common["run_context"] = context_b
    capsule_b = emit_capsule(run_b, build_capsule(**common))

    assert compare_capsule_files(capsule_a, capsule_b).identical is True


def test_worker_rejects_captured_config_digest_mismatch_before_loader(
        monkeypatch, tmp_path):
    original = tmp_path / "source" / "experiment.toml"
    original.parent.mkdir()
    original.write_bytes(b"mutable original\n")
    payload = tmp_path / "out" / "captured.toml"
    payload.parent.mkdir()
    payload.write_bytes(b"tampered captured payload\n")
    stages = []

    class _Progress:
        last_phase = "worker-start"
        last_step = 0
        last_wrfout = None
        last_checkpoint = None

        def __init__(self, *_args, **_kwargs):
            pass

        def preparing(self, stage):
            self.last_phase = stage
            stages.append(stage)

        def failed(self):
            stages.append("failed")

    monkeypatch.setattr(supervisor, "RuntimeHeartbeat", _Progress)
    monkeypatch.setattr(
        case_data, "load_experiment_case_bytes",
        lambda *_args, **_kwargs: pytest.fail(
            "config loader must not run after digest mismatch"))
    monkeypatch.setattr(
        supervisor, "write_failure_capsule",
        lambda path, **_kwargs: Path(path))
    monkeypatch.setenv("WOOF_RUN_ID", "run-id")
    monkeypatch.setenv("WOOF_CONFIG_DIGEST", hashlib.sha256(
        b"expected captured payload\n").hexdigest())
    monkeypatch.setenv("WOOF_STARTED_AT_UTC", "2026-08-01T00:00:00Z")
    monkeypatch.setenv("WOOF_GPU_UUID", "GPU-test")
    monkeypatch.setenv("WOOF_GPU_DRIVER", "610.74")
    monkeypatch.setenv("WOOF_GPU_NAME", "RTX 5090")

    with pytest.raises(SupervisorError, match="config digest mismatch"):
        supervisor._worker_main(supervisor.argparse.Namespace(
            config=original, config_payload=payload, outdir=payload.parent,
            restart=None, health_debug=False))

    assert stages == ["worker-start", "validate-config", "failed"]


def test_captured_config_loader_keeps_original_source_and_relative_base(
        monkeypatch, tmp_path):
    source = tmp_path / "source" / "experiment.toml"
    source.parent.mkdir()
    captured = (
        b"[experiment]\nname='captured'\n"
        b"[case_data]\nforcing='relative.grib2'\n")
    observed = {}

    class _Experiment(str):
        # The door reads the built experiment's domain spacings for the
        # sub-km terrain default (resolve_static_highres); a stand-in with
        # no domains takes no default.
        domains = ()

    def build_experiment(raw, *, source):
        observed["experiment"] = (raw, source)
        return _Experiment("experiment")

    def build_case_data(raw, *, source, base_dir, require_inputs=True,
                        **kwargs):
        observed["case_data"] = (raw, source, base_dir)
        observed["require_inputs"] = require_inputs
        return "case-data"

    monkeypatch.setattr(case_data, "build_experiment", build_experiment)
    # The door reviews the experiment it just built (the pole footprint,
    # woof.experiment.review_root_footprint); this test hands it a string
    # in place of an experiment, so the review is stubbed with the builder.
    monkeypatch.setattr(case_data, "review_root_footprint",
                        lambda _experiment, _source: None)
    monkeypatch.setattr(case_data, "build_case_data", build_case_data)

    result = case_data.load_experiment_case_bytes(
        captured, source=str(source), base_dir=source.parent)

    assert result == ("experiment", "case-data")
    assert observed["experiment"][1] == str(source)
    assert observed["case_data"][1:] == (str(source), source.parent)
    assert observed["case_data"][0]["forcing"] == "relative.grib2"
    # A supervised run requires its declared inputs to exist.  Only the
    # planning callers turn that off, and never through this path.
    assert observed["require_inputs"] is True


def test_supervise_monitor_accepts_windows_redirector_child_pid(
        monkeypatch, tmp_path):
    child_pid = 52_001
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "preparing:worker-start", "step": 0,
             "heartbeat_pid": child_pid},
            {"status": "integrating", "step": 1,
             "heartbeat_pid": child_pid},
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": child_pid},
        ]])
    result = supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert result.heartbeat.pid == child_pid
    assert result.heartbeat.pid != processes[0].pid


def test_supervise_monitor_accepts_child_completion_before_first_poll(
        monkeypatch, tmp_path):
    child_pid = 52_001
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": child_pid},
        ]], publish_before_popen_returns=True)
    result = supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert result.heartbeat.pid == child_pid
    assert result.heartbeat.pid != processes[0].pid


def test_supervise_monitor_rejects_worker_pid_change(monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 1,
             "heartbeat_pid": 52_001},
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": 52_002},
        ]])
    with pytest.raises(SupervisorError, match="heartbeat identity violation"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(processes) == 1


def test_mismatched_final_heartbeat_cannot_supply_recovery_checkpoint(
        monkeypatch, tmp_path):
    mismatched_checkpoint = tmp_path / "checkpoint.npz"
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 1,
             "heartbeat_pid": 52_001, "checkpoint": None},
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": 52_002,
             "checkpoint": mismatched_checkpoint},
        ]])
    validated = []

    def validate(path):
        validated.append(Path(path).resolve())
        return Path(path).resolve()

    import woof.supervisor as supervisor
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", validate)
    with pytest.raises(SupervisorError, match="heartbeat identity violation"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert validated == []
    assert len(processes) == 1


def test_existing_child_failure_capsule_is_matched_by_effective_pid(
        monkeypatch, tmp_path):
    child_pid = 52_001
    config, _, _ = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 1,
             "heartbeat_pid": child_pid},
            {"exit": 7},
        ]])
    checked = []

    def read_capsule(path, *, run_id, worker_pid):
        checked.append((Path(path), run_id, worker_pid))
        return {"schema": supervisor.FAILURE_CAPSULE_SCHEMA,
                "run_id": run_id, "worker_pid": worker_pid,
                "exception": {"type": "RuntimeError",
                              "message": "the child's own words"}}

    def unexpected_write(*args, **kwargs):
        raise AssertionError("supervisor overwrote the worker failure capsule")

    import woof.supervisor as supervisor
    monkeypatch.setattr(supervisor, "_read_worker_failure_capsule",
                        read_capsule)
    monkeypatch.setattr(supervisor, "write_failure_capsule", unexpected_write)
    with pytest.raises(SupervisorError, match="no durable manifest-valid") \
            as caught:
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(checked) == 1
    assert checked[0][2] == child_pid
    # The capsule it matched is also the capsule it quotes.
    assert "the child's own words" in str(caught.value)


def test_supervise_monitor_validates_final_heartbeat_regression(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 5},
            {"status": "complete", "step": 1, "exit": 0},
        ]])
    with pytest.raises(SupervisorError, match="outer_step moved backward"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(processes) == 1


def test_supervise_monitor_rejects_change_after_failed_heartbeat(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "failed", "step": 1},
            {"status": "complete", "step": 2, "exit": 0},
        ]])
    with pytest.raises(SupervisorError, match="terminal status failed changed"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(processes) == 1


def test_rejected_final_mutation_cannot_reuse_earlier_complete_heartbeat(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "complete", "step": 1,
             "heartbeat_pid": 52_001},
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": 52_001},
        ]])
    with pytest.raises(SupervisorError, match="terminal status complete changed"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(processes) == 1


def test_supervise_monitor_crash_resumes_in_fresh_process(monkeypatch,
                                                          tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [
            [{"status": "integrating", "step": 2}, {"exit": 7}],
            [{"status": "integrating", "step": 3},
             {"status": "complete", "step": 4, "exit": 0}],
        ])
    # Passing an initial manifest-valid checkpoint proves every relaunch is a
    # resume and avoids any restart-from-scratch policy ambiguity.
    result = supervise_experiment(
        config, tmp_path / "out", restart=checkpoint,
        max_restarts=1, poll_seconds=0.05)
    assert result.attempts == 2
    assert all("--restart" in process.command for process in processes)


def test_supervise_monitor_crash_loop_hits_restart_cap(monkeypatch, tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [
            [{"status": "integrating", "step": 1}, {"exit": 11}],
            [{"status": "integrating", "step": 1}, {"exit": 12}],
        ])
    with pytest.raises(SupervisorError, match="exhausted 1"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=1, poll_seconds=0.05)
    assert len(processes) == 2


def test_supervise_monitor_kills_stale_integrating_worker(monkeypatch,
                                                          tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
        ]], clock_step=61.0)
    with pytest.raises(SupervisorError, match="became stale"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)
    assert processes[0].terminated


def test_supervise_will_not_relaunch_a_restore_the_worker_refused(
        monkeypatch, tmp_path):
    """A refused restore is deterministic: the same bytes refuse again.

    The 24 h reproduction spent three fresh processes rediscovering one
    refusal over one checkpoint, and each attempt's failure capsule
    overwrote the one before it -- so the evidence for what killed the
    FIRST worker was destroyed by the relaunches.  Nothing between
    attempts can change a restore's answer: same config digest, same
    file, same reader.
    """
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [
            [{"status": "preparing:restore-tree-checkpoint", "step": 0},
             {"capsule": {
                 "type": "RestartMismatchError",
                 "message": "tree restart is at 200 ticks but this "
                            "configuration stops at 100 ticks",
                 "phase": "preparing:restore-tree-checkpoint"},
              "exit": 1}],
            [{"status": "preparing:restore-tree-checkpoint", "step": 0},
             {"exit": 1}],
        ], real_capsule_reader=True)

    with pytest.raises(SupervisorError,
                       match="deterministic relaunch loop") as caught:
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=3, poll_seconds=0.05)

    assert len(processes) == 1
    assert "tree restart is at 200 ticks" in str(caught.value)


def test_supervise_still_relaunches_a_crash_that_is_not_a_refused_restore(
        monkeypatch, tmp_path):
    """CONTROL: fresh-process recovery is not being switched off.

    A worker that dies mid-integration may well succeed on a fresh CUDA
    context, which is what the recovery attempts exist for.
    """
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [
            [{"status": "integrating", "step": 3}, {"exit": 11}],
            [{"status": "integrating", "step": 3},
             {"status": "complete", "step": 6, "exit": 0}],
        ])

    result = supervise_experiment(
        config, tmp_path / "out", restart=checkpoint,
        max_restarts=3, poll_seconds=0.05)

    assert result.attempts == 2
    assert len(processes) == 2


def test_supervise_waits_out_a_worker_that_already_published_complete(
        monkeypatch, tmp_path):
    """A finished worker must not be killed on its way out the door.

    After the last integration step the worker still has durable work:
    drain the history writers, and hash every emitted frame into the
    success capsule.  On a 24 h four-domain run that is hundreds of GiB
    of reading with no model step left to beat on, and the integration
    watchdog cannot tell that from a hang -- so it SIGTERMed a worker
    that had already published its terminal ``complete`` record, and the
    supervisor then replayed the finished run as a restart loop.  A
    published terminal record retires the watchdog: there is no
    integration left to be stale.
    """
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "complete", "step": 2},
            {}, {}, {},
            {"exit": 0},
        ]], clock_step=61.0)

    result = supervise_experiment(
        config, tmp_path / "out", restart=checkpoint,
        max_restarts=0, poll_seconds=0.05)

    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert processes[0].terminated is False


def test_supervise_still_fails_loudly_when_a_worker_dies_after_complete(
        monkeypatch, tmp_path):
    """CONTROL: retiring the watchdog must not trust the record alone.

    ``complete`` is what the worker BELIEVES; the exit status is what
    happened.  A worker that publishes complete and then dies writing its
    receipts still failed, and still says so.
    """
    config, checkpoint, _ = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "complete", "step": 2},
            {"exit": 3},
        ]])

    with pytest.raises(SupervisorError, match="worker exited with status 3"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)


def test_heartbeat_admits_a_named_finalizing_phase():
    """Finalization is a published phase, not a silence."""
    assert _heartbeat(
        status="finalizing:hash-output-frames"
    ).status == "finalizing:hash-output-frames"
    with pytest.raises(ValueError, match="invalid heartbeat status"):
        _heartbeat(status="finalising")


def test_runtime_heartbeat_publishes_a_named_finalizing_phase(tmp_path):
    progress = RuntimeHeartbeat(
        tmp_path / "run-progress.json", run_id="run-1",
        config_sha256="b" * 64, started_at_utc="2026-07-16T00:00:00Z")
    progress.finalizing("hash-output-frames")
    record = read_heartbeat(tmp_path / "run-progress.json")
    assert record.status == "finalizing:hash-output-frames"
    assert progress.last_phase == "finalizing:hash-output-frames"


def test_finalizing_is_forward_progress_from_integrating_and_never_reverts():
    integrating = _heartbeat(
        status="integrating", updated_at_utc="2026-07-16T00:00:01Z")
    finalizing = _heartbeat(
        status="finalizing:drain-history-writers",
        updated_at_utc="2026-07-16T00:00:02Z")
    assert supervisor._heartbeat_regression(integrating, finalizing) is None

    for backward in ("integrating", "preparing:prepare-case"):
        later = _heartbeat(
            status=backward, updated_at_utc="2026-07-16T00:00:03Z")
        regression = supervisor._heartbeat_regression(finalizing, later)
        assert regression is not None and "finalizing" in regression


def test_supervise_monitor_slow_preparation_has_no_default_timeout(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "preparing:load-config", "step": 0},
            {"status": "preparing:prepare-case", "step": 0},
            {"status": "complete", "step": 0, "exit": 0},
        ]], clock_step=500.0)
    result = supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert result.attempts == 1
    assert not processes[0].terminated


def test_supervise_monitor_wedged_preparation_times_out_once(monkeypatch,
                                                             tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "preparing:prepare-case", "step": 0},
        ]], clock_step=31.0)
    with pytest.raises(SupervisorError, match="refusing a deterministic"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=3, prep_timeout_seconds=30.0, poll_seconds=0.05)
    assert len(processes) == 1
    assert processes[0].terminated


def test_supervise_monitor_detects_heartbeat_regression(monkeypatch, tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "integrating", "step": 1},
        ]])
    with pytest.raises(SupervisorError, match="outer_step moved backward"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=3, poll_seconds=0.05)
    assert len(processes) == 1
    assert processes[0].terminated


# --- directory-input identity ------------------------------------------
#
# The dual-run byte comparison is only as good as the binding between a
# run and its inputs.  Files are content-hashed; a *directory* input --
# in practice the static geography tree -- is bound by inventory
# (path/size/mtime) by default, which has one false-positive and one
# false-negative mode.  Both are gated here, at both mode values, because
# docs/public/DETERMINISM.md tells scientists exactly which one they get.


def _geography_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "nested").mkdir()
    (root / "index").write_bytes(b"projection=regular_ll\n")
    (root / "nested" / "00001-01200.00001-01200").write_bytes(
        bytes(range(256)) * 4)
    return root


def _stamp(root: Path, mtime_ns: int) -> None:
    for child in sorted(root.rglob("*")):
        if child.is_file():
            os.utime(child, ns=(mtime_ns, mtime_ns))


@pytest.mark.parametrize(
    "mode, copies_agree",
    [("inventory", False), ("content", True)])
def test_directory_hash_and_a_byte_identical_copy(tmp_path, mode,
                                                  copies_agree):
    """A staged copy with fresh mtimes: inventory false-positives."""
    left = _geography_tree(tmp_path / "left")
    right = _geography_tree(tmp_path / "right")
    _stamp(left, 1_500_000_000_000_000_000)
    _stamp(right, 1_600_000_000_000_000_000)

    left_digest = supervisor._hash_directory_manifest(left, mode=mode)
    right_digest = supervisor._hash_directory_manifest(right, mode=mode)
    assert (left_digest == right_digest) is copies_agree


@pytest.mark.parametrize(
    "mode, edit_is_seen",
    [("inventory", False), ("content", True)])
def test_directory_hash_and_a_size_and_mtime_preserving_edit(tmp_path, mode,
                                                             edit_is_seen):
    """A same-size edit with the mtime restored: inventory misses it."""
    tree = _geography_tree(tmp_path / "geog")
    _stamp(tree, 1_500_000_000_000_000_000)
    before = supervisor._hash_directory_manifest(tree, mode=mode)

    target = tree / "nested" / "00001-01200.00001-01200"
    payload = bytearray(target.read_bytes())
    payload[17] ^= 0x01                      # one bit, same length
    target.write_bytes(bytes(payload))
    _stamp(tree, 1_500_000_000_000_000_000)  # and the mtime is put back

    after = supervisor._hash_directory_manifest(tree, mode=mode)
    assert (before != after) is edit_is_seen


def test_directory_hash_inventory_digest_is_unchanged_by_the_mode_argument(
        tmp_path):
    """Old inventory digests stay comparable: no domain prefix was added."""
    tree = _geography_tree(tmp_path / "geog")
    _stamp(tree, 1_500_000_000_000_000_000)
    expected = hashlib.sha256()
    for relative in ("index", "nested/00001-01200.00001-01200"):
        child = tree / relative
        expected.update(
            f"{relative}\0{child.stat().st_size}\0"
            f"{child.stat().st_mtime_ns}\n".encode("utf-8"))
    assert supervisor._hash_directory_manifest(tree, mode="inventory") == \
        expected.hexdigest()


@pytest.mark.parametrize("mode", ["inventory", "content"])
def test_resolved_input_hashes_records_the_algorithm_it_used(
        monkeypatch, tmp_path, mode):
    tree = _geography_tree(tmp_path / "geog")
    forcing = tmp_path / "forcing.grib2"
    forcing.write_bytes(b"GRIB")

    class _Case:
        def resolved_inputs(self):
            return (case_data.ResolvedInput(role="forcing", path=forcing),
                    case_data.ResolvedInput(role="geog_root", path=tree))

    monkeypatch.setattr(case_data, "load_experiment_case",
                        lambda _path, **_kwargs: (None, _Case()))
    hashes = supervisor.resolved_input_hashes(
        tmp_path / "case.toml", directory_hash=mode)
    assert hashes[f"geog_root:{tree}"]["algorithm"] == \
        f"sha256-directory-{mode}"
    assert hashes[f"forcing:{forcing}"]["algorithm"] == "sha256"


def test_resolved_input_hashes_parses_the_parent_captured_bytes(
        monkeypatch, tmp_path):
    forcing = tmp_path / "forcing.grib2"
    forcing.write_bytes(b"GRIB")
    captured = b"captured config authority"
    observed = {}

    class _Case:
        def resolved_inputs(self):
            return (case_data.ResolvedInput(role="forcing", path=forcing),)

    def load_bytes(payload, *, source, base_dir):
        observed.update(
            payload=payload, source=source, base_dir=base_dir)
        return None, _Case()

    monkeypatch.setattr(case_data, "load_experiment_case_bytes", load_bytes)
    monkeypatch.setattr(
        case_data, "load_experiment_case",
        lambda _path: pytest.fail("mutable config path must not be reopened"))
    config = tmp_path / "source" / "case.toml"

    hashes = supervisor.resolved_input_hashes(
        config, config_bytes=captured)

    assert observed == {
        "payload": captured,
        "source": str(config),
        "base_dir": config.parent,
    }
    assert hashes[f"forcing:{forcing}"]["digest"] == hashlib.sha256(
        b"GRIB").hexdigest()


def test_worker_inventory_seal_preserves_roles_details_and_multiplicity(
        monkeypatch, tmp_path):
    shared = tmp_path / "shared-input.nc"
    shared.write_bytes(b"one shared payload")
    parent_records = (
        case_data.ResolvedInput(role="forcing", path=shared),
        case_data.ResolvedInput(
            role="source_orography", path=shared,
            detail="variable=HGT;domain=d01"),
        case_data.ResolvedInput(
            role="source_orography", path=shared,
            detail="variable=HGT;domain=d02"),
    )

    class _Case:
        def __init__(self, records):
            self.records = records

        def resolved_inputs(self):
            return self.records

    monkeypatch.setattr(
        case_data, "load_experiment_case",
        lambda _path: (None, _Case(parent_records)))
    hashes = supervisor.resolved_input_hashes(tmp_path / "case.toml")
    orography = hashes[f"source_orography:{shared.resolve()}"]
    assert len(orography["identities"]) == 2

    worker_records = (
        # The same source path is not enough: this record changed roles.
        case_data.ResolvedInput(role="vtable", path=shared),
        case_data.ResolvedInput(
            role="source_orography", path=shared,
            detail="variable=HGT;domain=d01"),
        # Multiplicity is unchanged, but the second duplicate's slot differs.
        case_data.ResolvedInput(
            role="source_orography", path=shared,
            detail="variable=HGT;domain=d03"),
    )
    with pytest.raises(SupervisorError) as caught:
        supervisor._validate_worker_resolved_input_inventory(
            _Case(worker_records), hashes)

    message = str(caught.value)
    assert "forcing:" in message and "vtable:" in message
    assert "domain=d02" in message and "domain=d03" in message


@pytest.mark.parametrize("control", ["shrink", "expand", "rename"])
def test_worker_refuses_forcing_glob_path_set_toctou_after_restore(
        monkeypatch, tmp_path, control):
    from woof import runtime

    source = tmp_path / "source"
    forcing_dir = source / "forcing"
    geog = source / "geog"
    outdir = tmp_path / "out"
    store = tmp_path / "plan-wide-cas"
    for directory in (forcing_dir, geog, outdir):
        directory.mkdir(parents=True)
    forcing_a = forcing_dir / "a.grib"
    forcing_b = forcing_dir / "b.grib"
    forcing_c = forcing_dir / "c.grib"
    outside_b = source / "b.grib.moved"
    forcing_a.write_bytes(b"forcing-a")
    forcing_b.write_bytes(b"forcing-b")
    (source / "Vtable").write_bytes(b"vtable")
    (source / "namelist.wps").write_bytes(b"wps")
    payload = (
        "[experiment]\n"
        "name = 'fixture'\n"
        "[case_data]\n"
        "forcing = 'forcing/*.grib'\n"
        "vtable = 'Vtable'\n"
        "wps_namelist = 'namelist.wps'\n"
        "geog_root = 'geog'\n"
        "sfcp_to_sfcp = true\n"
        "output_title = 'fixture'\n"
    ).encode("utf-8")
    config = source / "experiment.toml"
    captured = outdir / "captured.toml"
    config.write_bytes(payload)
    captured.write_bytes(payload)
    monkeypatch.setattr(
        case_data, "build_experiment",
        lambda *_args, **_kwargs: SimpleNamespace(domains=()))
    # A stand-in with no domains takes the experiment's place (so no sub-km
    # terrain default applies), and the door's plan review of it
    # (review_root_footprint) is stubbed beside the builder.
    monkeypatch.setattr(case_data, "review_root_footprint",
                        lambda _experiment, _source: None)

    # Parent parse, hashing, and CAS publication all see exactly {a, b}.
    parent_hashes = supervisor.resolved_input_hashes(
        config, config_bytes=payload)
    authorities = supervisor.snapshot_resolved_input_files(
        config, config_bytes=payload, input_hashes=parent_hashes,
        snapshot_root=store)
    parent_forcing = {
        identity["path"]
        for entry in parent_hashes.values()
        for identity in entry["identities"]
        if identity["role"] == "forcing"
    }
    assert parent_forcing == {
        str(forcing_a.resolve()), str(forcing_b.resolve())}

    events = []
    restored = False

    def mutate():
        if control == "shrink":
            forcing_b.rename(outside_b)
        elif control == "expand":
            forcing_c.write_bytes(b"forcing-c")
        else:
            forcing_a.rename(forcing_c)
        events.append("mutate")

    def restore():
        nonlocal restored
        if restored:
            return
        if control == "shrink":
            outside_b.rename(forcing_b)
        elif control == "expand":
            forcing_c.unlink()
        else:
            forcing_c.rename(forcing_a)
        restored = True
        events.append("restore")

    mutate()
    real_load = case_data.load_experiment_case_bytes

    def load_while_mutated(*args, **kwargs):
        events.append("parse")
        try:
            return real_load(*args, **kwargs)
        finally:
            # Restore before the worker performs its comparison.  The seal
            # must use what this parse resolved, not a second filesystem glob.
            restore()

    real_validate = supervisor._validate_worker_resolved_input_inventory

    def validate_after_restore(data, input_hashes):
        events.append("validate")
        assert restored
        assert forcing_a.is_file() and forcing_b.is_file()
        assert not forcing_c.exists() and not outside_b.exists()
        return real_validate(data, input_hashes)

    monkeypatch.setattr(
        case_data, "load_experiment_case_bytes", load_while_mutated)
    monkeypatch.setattr(
        supervisor, "_validate_worker_resolved_input_inventory",
        validate_after_restore)
    monkeypatch.setattr(
        runtime, "run_experiment",
        lambda *_args, **_kwargs: pytest.fail("runtime must never start"))
    monkeypatch.setattr(
        supervisor, "emit_run_capsule",
        lambda *_args, **_kwargs: pytest.fail(
            "a success capsule must never be emitted"))
    monkeypatch.setattr(supervisor, "git_commit", lambda: "test-commit")
    monkeypatch.setenv("WOOF_RUN_ID", f"glob-{control}")
    monkeypatch.setenv(
        "WOOF_CONFIG_DIGEST", hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv("WOOF_STARTED_AT_UTC", "2026-08-01T00:00:00Z")
    monkeypatch.setenv("WOOF_GPU_UUID", "GPU-test")
    monkeypatch.setenv("WOOF_GPU_DRIVER", "610.74")
    monkeypatch.setenv("WOOF_GPU_NAME", "RTX 5090")
    monkeypatch.setenv(
        "WOOF_INPUT_HASHES_JSON", json.dumps(parent_hashes))
    monkeypatch.setenv(
        supervisor.INPUT_AUTHORITIES_ENV, json.dumps(authorities))

    try:
        with pytest.raises(
                SupervisorError,
                match="worker-resolved input inventory") as caught:
            supervisor._worker_main(supervisor.argparse.Namespace(
                config=config, config_payload=captured, outdir=outdir,
                restart=None, health_debug=False))
    finally:
        restore()

    expected_missing = {
        "shrink": (forcing_b,),
        "expand": (),
        "rename": (forcing_a,),
    }[control]
    expected_extra = {
        "shrink": (),
        "expand": (forcing_c,),
        "rename": (forcing_c,),
    }[control]
    message = str(caught.value)
    for path in expected_missing:
        assert str(path.resolve()) in message.partition("; extra=")[0]
    for path in expected_extra:
        assert str(path.resolve()) in message.partition("; extra=")[2]
    assert events == ["mutate", "parse", "restore", "validate"]
    assert supervisor.read_heartbeat(
        outdir / supervisor.HEARTBEAT_NAME).status == "failed"
    failure = json.loads((
        outdir / supervisor.FAILURE_CAPSULE_NAME).read_text(encoding="utf-8"))
    assert failure["input_hashes"] == parent_hashes
    assert failure["last_phase"] == "preparing:validate-input-inventory"
    assert failure["exception"]["type"] == "SupervisorError"
    assert failure["exception"]["message"] == message
    assert not (outdir / "certification-capsule.json").exists()


def test_plan_cas_deduplicates_and_worker_consumes_all_file_snapshots(
        monkeypatch, tmp_path):
    forcing = tmp_path / "forcing.grib2"
    vtable = tmp_path / "Vtable"
    wps = tmp_path / "namelist.wps"
    orography = tmp_path / "orography.nc"
    geog = tmp_path / "geog"
    geog.mkdir()
    (geog / "index").write_bytes(b"geography")
    payloads = {
        forcing: b"ORIGINAL-FORCING",
        vtable: b"VTABLE-CONTENT",
        # Deliberately identical to Vtable: one CAS file must retain two
        # distinct declared provenance identities after remapping.
        wps: b"VTABLE-CONTENT",
        orography: b"OROGRAPHY-CONTENT",
    }
    for path, payload in payloads.items():
        path.write_bytes(payload)
    data = case_data.CaseDataConfig(
        forcing=(forcing,), vtable=vtable, wps_namelist=wps,
        geog_root=geog,
        source_orography=case_data.SourceOrography(
            path=orography, variable="HGT"),
        sfcp_to_sfcp=True, co2_vmr=None, output_title="fixture")
    monkeypatch.setattr(
        case_data, "load_experiment_case_bytes",
        lambda *_args, **_kwargs: (None, data))
    config_a = tmp_path / "case-a.toml"
    config_b = tmp_path / "case-b.toml"
    hashes = supervisor.resolved_input_hashes(
        config_a, config_bytes=b"captured")
    supervisor._validate_worker_resolved_input_inventory(data, hashes)
    copied = []
    real_copy = supervisor._copy_verified_authority

    def copy_once(source, destination, expected):
        copied.append(Path(source).resolve())
        return real_copy(Path(source), Path(destination), expected)

    monkeypatch.setattr(supervisor, "_copy_verified_authority", copy_once)
    store = tmp_path / "plan-wide-cas"
    first = supervisor.snapshot_resolved_input_files(
        config_a, config_bytes=b"captured", input_hashes=hashes,
        snapshot_root=store)
    second = supervisor.snapshot_resolved_input_files(
        config_b, config_bytes=b"captured", input_hashes=hashes,
        snapshot_root=store)

    assert first == second
    assert sorted(copied) == sorted(path.resolve() for path in (
        forcing, vtable, orography))
    assert set(first) == {str(path.resolve()) for path in payloads}
    sha_files = [path for path in store.iterdir()
                 if path.is_file() and len(path.name) == 64]
    assert len(sha_files) == len(set(
        hashlib.sha256(payload).hexdigest()
        for payload in payloads.values()))

    metadata = forcing.stat()
    mutated = b"MUTATED!-FORCING"
    assert len(mutated) == len(payloads[forcing])
    descriptor = os.open(forcing, os.O_WRONLY | os.O_TRUNC)
    try:
        os.write(descriptor, mutated)
    finally:
        os.close(descriptor)
    os.utime(forcing, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))

    real_path_open = Path.open
    original_forcing = forcing.resolve()

    def guarded_open(path, *args, **kwargs):
        if Path(path).resolve() == original_forcing:
            pytest.fail("worker reopened mutable forcing after snapshot")
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    replacements = supervisor._validated_worker_input_authorities(
        json.dumps(first), hashes)
    remapped = case_data.remap_case_data_files(data, replacements)
    assert remapped.forcing[0].read_bytes() == payloads[forcing]
    assert remapped.vtable == Path(first[str(vtable.resolve())]["snapshot"])
    assert remapped.wps_namelist == Path(first[str(wps.resolve())]["snapshot"])
    assert remapped.source_orography.path == Path(
        first[str(orography.resolve())]["snapshot"])
    assert all(path.resolve() not in payloads for path in (
        *remapped.forcing, remapped.vtable, remapped.wps_namelist,
        remapped.source_orography.path))
    assert {record.path for record in remapped.resolved_inputs()
            if record.role != "geog_root"} == {
                path.resolve() for path in payloads}
    capsule_context = supervisor._success_run_context(
        config_a, "f" * 64, hashes, restart_interval_seconds=None)
    assert str(store) not in json.dumps(capsule_context, sort_keys=True)

    descriptor = os.open(forcing, os.O_WRONLY | os.O_TRUNC)
    try:
        os.write(descriptor, payloads[forcing])
    finally:
        os.close(descriptor)
    os.utime(forcing, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert forcing.stat().st_mtime_ns == metadata.st_mtime_ns


def test_content_authority_publication_preserves_racing_winner(
        monkeypatch, tmp_path):
    source = tmp_path / "forcing.grib2"
    payload = b"one exact forcing payload"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    store = tmp_path / "cas"
    store.mkdir()
    real_link = supervisor.os.link
    calls = []

    def publish_winner_then_report_race(temporary, destination):
        calls.append((Path(temporary), Path(destination)))
        real_link(temporary, destination)
        raise FileExistsError("simulated racing create-only winner")

    monkeypatch.setattr(supervisor.os, "link",
                        publish_winner_then_report_race)
    authority = supervisor._content_authority_path(source, store, digest)

    assert authority == store / digest
    assert authority.read_bytes() == payload
    assert authority.stat().st_mode & stat.S_IWRITE == 0
    assert len(calls) == 1
    assert not list(store.glob(".*.tmp"))


@pytest.mark.parametrize(
    "requested, env, expected",
    [(None, None, "inventory"),
     (None, "content", "content"),
     ("content", None, "content"),
     ("inventory", "content", "inventory")])
def test_directory_hash_mode_resolution(monkeypatch, requested, env,
                                        expected):
    monkeypatch.delenv(supervisor.DIRECTORY_HASH_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(supervisor.DIRECTORY_HASH_ENV, env)
    assert supervisor.directory_hash_mode(requested) == expected


@pytest.mark.parametrize("bad", ["mtime", "sha256", ""])
def test_directory_hash_mode_refuses_an_unknown_mode(monkeypatch, bad):
    monkeypatch.delenv(supervisor.DIRECTORY_HASH_ENV, raising=False)
    with pytest.raises(ValueError, match="is not one of"):
        supervisor.directory_hash_mode(bad)
    monkeypatch.setenv(supervisor.DIRECTORY_HASH_ENV, bad)
    with pytest.raises(ValueError, match="environment variable"):
        supervisor.directory_hash_mode()
    with pytest.raises(ValueError, match="is not one of"):
        supervisor._hash_directory_manifest(Path("."), mode=bad)


def test_git_commit_survives_a_host_with_no_git_binary(monkeypatch):
    """A pip-install host need not have git at all; subprocess.run then
    raises FileNotFoundError, and before the 4090 stress wave that
    traceback reached the capsule builder.  The sentinel form is the
    same 'unavailable: <why>' the schema documents."""

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(supervisor.subprocess, "run", no_git)
    recorded = supervisor.git_commit()
    assert recorded.startswith("unavailable: ")
    assert "FileNotFoundError" in recorded


# --- A refusal must read as a refusal on the console --------------------
#
# "worker exited with status 1 ... (failure capsule: PATH)" is the same
# shell for a segfault and for a config the loader deliberately rejected.
# Through 1.8.0 the sentence that explained it sat one file away.


def _refusal_config(tmp_path):
    """A config the loader refuses: no [case_data] table at all."""
    payload = b"[experiment]\nname = 'fixture'\n"
    config = tmp_path / "experiment.toml"
    captured = tmp_path / "captured.toml"
    config.write_bytes(payload)
    captured.write_bytes(payload)
    return config, captured, payload


def test_a_config_load_refusal_capsule_carries_the_refusal_sentence(
        monkeypatch, tmp_path):
    """First: a REAL worker refusal produces the capsule we claim to read.

    Written against the genuine artifact rather than a synthetic capsule
    so the schema path is proven end to end -- a committed fixture in the
    wrong shape is exactly how this class of reader goes quietly blind
    (tests/test_report_bundle.py carries one).
    """
    outdir = tmp_path / "out"
    outdir.mkdir()
    config, captured, payload = _refusal_config(tmp_path)
    monkeypatch.setattr(supervisor, "git_commit", lambda: "test-commit")
    monkeypatch.setenv("WOOF_RUN_ID", "refusal-run")
    monkeypatch.setenv(
        "WOOF_CONFIG_DIGEST", hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv("WOOF_STARTED_AT_UTC", "2026-08-07T00:00:00Z")
    monkeypatch.setenv("WOOF_GPU_UUID", "GPU-test")
    monkeypatch.setenv("WOOF_GPU_DRIVER", "610.74")
    monkeypatch.setenv("WOOF_GPU_NAME", "RTX 5090")
    monkeypatch.setenv("WOOF_INPUT_HASHES_JSON", json.dumps({}))
    monkeypatch.delenv(supervisor.INPUT_AUTHORITIES_ENV, raising=False)

    with pytest.raises(ValueError, match=r"\[\[domain\]\]") as refused:
        supervisor._worker_main(supervisor.argparse.Namespace(
            config=config, config_payload=captured, outdir=outdir,
            restart=None, health_debug=False))

    capsule = json.loads((
        outdir / supervisor.FAILURE_CAPSULE_NAME).read_text(encoding="utf-8"))
    assert capsule["exception"]["type"] == "ValueError"
    assert capsule["last_phase"] == "preparing:load-config"
    assert capsule["exception"]["cuda_fatal"] is False
    headline = supervisor._capsule_headline(capsule)
    assert headline.startswith("ValueError: ")
    # The sentence the user can act on, verbatim from the loader.
    assert "must carry at least one [[domain]] table" in headline
    # Verbatim around the config path: a deep temporary root may shorten
    # the path itself (to its last components), never the sentence.
    line = str(refused.value).splitlines()[0].rstrip(".")
    before, found, after = line.partition(str(config))
    assert found, line
    assert headline.startswith(f"ValueError: {before}")
    assert headline.endswith(after)
    assert f"{config.parent.name}{os.sep}{config.name}" in headline

    from woof.explain import EXPLAIN_MARK
    assert EXPLAIN_MARK not in headline


def test_a_worker_that_dies_on_a_refusal_says_so_in_the_supervisor_error(
        monkeypatch, tmp_path):
    """THE requirement: the refusal sentence reaches the raised error.

    A guard refusal that prints only an exit status and a path is
    indistinguishable from a crash, and the user's actual problem -- the
    thing they can fix -- is in a file the message merely names.
    """
    refusal = ("experiment config fixture.toml carries no [case_data] "
               "table; the experiment runtime requires declared inputs "
               "(forcing, vtable, wps_namelist, geog_root, and policies).")
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "preparing:load-config"},
            {"capsule": {"type": "ValueError", "message": refusal}},
            {"exit": 1},
        ]], real_capsule_reader=True)
    del checkpoint          # no restart: the run never got far enough

    with pytest.raises(SupervisorError) as caught:
        supervise_experiment(
            config, tmp_path / "out", max_restarts=0, poll_seconds=0.05)

    message = str(caught.value)
    assert refusal.rstrip(".") in message
    assert "ValueError" in message
    # The old information is not traded away for the new: the exit
    # status, the reason no restart followed, and the capsule path all
    # survive, because each answers a different question.
    assert "worker exited with status 1" in message
    assert "no durable manifest-valid checkpoint" in message
    assert "failure-capsule.json" in message
    assert len(processes) == 1


def test_a_supervisor_authored_capsule_is_not_quoted_back_at_itself(
        monkeypatch, tmp_path):
    """WorkerExit's message IS "worker exited with status N"."""
    config, checkpoint, _ = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "preparing:load-config"},
            {"capsule": {"type": "WorkerExit",
                         "message": "worker exited with status 1"}},
            {"exit": 1},
        ]], real_capsule_reader=True)

    with pytest.raises(SupervisorError) as caught:
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)
    assert str(caught.value).count("worker exited with status 1") == 1


def test_a_capsule_from_another_attempt_is_not_quoted(monkeypatch, tmp_path):
    """The run/pid binding, which is why the reader validates it."""
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True)
    (outdir / "failure-capsule.json").write_text(json.dumps({
        "schema": supervisor.FAILURE_CAPSULE_SCHEMA,
        "run_id": "some-older-run", "worker_pid": 999,
        "exception": {"type": "ValueError", "message": "a stale sentence"},
    }), encoding="utf-8")
    config, checkpoint, _ = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[{"status": "preparing:load-config"}, {"exit": 1}]],
        real_capsule_reader=True)

    with pytest.raises(SupervisorError) as caught:
        supervise_experiment(
            config, outdir, restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)
    assert "a stale sentence" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    None, {}, {"exception": None}, {"exception": {}},
    {"exception": {"type": "", "message": ""}},
    {"exception": "not a mapping"},
])
def test_an_unusable_capsule_degrades_to_the_old_message(payload):
    """Never a second crash on the crash-reporting path."""
    assert supervisor._capsule_headline(payload) == ""


def test_the_headline_drops_the_explain_half_and_caps_the_length():
    """A layered refusal's sentinel is promised never to reach a terminal."""
    from woof.explain import layered

    action = "the short refusal sentence."
    message = layered(action, "The long explanation nobody pasted.")
    headline = supervisor._capsule_headline(
        {"exception": {"type": "ValueError", "message": message}})
    assert headline == f"ValueError: {action.rstrip(chr(46))}"
    assert "explanation nobody pasted" not in headline

    long = supervisor._capsule_headline(
        {"exception": {"type": "RuntimeError", "message": "x" * 4000}})
    assert len(long) < 300 and long.endswith("...")


@pytest.mark.parametrize("root", [
    "/srv/ci/" + "/".join(["deep-temporary-root"] * 12),
    "D:\\ci\\" + "\\".join(["deep-temporary-root"] * 12),
])
def test_a_deep_config_path_never_cuts_the_refusal_sentence(root):
    """The sentence after the path is what the user acts on.

    A CI runner's 120-character temporary root pushed the loader's
    "[[domain]] table (array of tables)" past the cap, which cut it
    (public CI rehearsal routes B and C, 2026-10-15).
    """
    sep = "\\" if "\\" in root else "/"
    path = f"{root}{sep}case-a{sep}experiment.toml"
    sentence = ("must carry at least one [[domain]] table (array of "
                "tables)")
    headline = supervisor._capsule_headline({"exception": {
        "type": "ValueError",
        "message": f"experiment config {path} {sentence}."}})
    assert len(path) > supervisor._CAPSULE_HEADLINE_CAP
    assert headline == (f"ValueError: experiment config "
                        f"...{sep}case-a{sep}experiment.toml {sentence}")


def test_worker_failure_capsule_embeds_the_exact_config_and_small_text_inputs(
        monkeypatch, tmp_path):
    """A capsule from a real failing run carries the run's own text bytes.

    Support burned a round trip asking a reporter for a sub-100-line TOML
    the capsule already identified by hash.  The v2 capsule embeds the
    captured config payload verbatim, plus the declared small-text inputs
    (Vtable, WPS namelist) keyed exactly as ``input_hashes`` keys them;
    forcing and geography stay hash-only.
    """
    from woof import runtime

    source = tmp_path / "source"
    forcing_dir = source / "forcing"
    geog = source / "geog"
    outdir = tmp_path / "out"
    for directory in (forcing_dir, geog, outdir):
        directory.mkdir(parents=True)
    (forcing_dir / "a.grib").write_bytes(b"forcing-a")
    vtable = source / "Vtable"
    wps = source / "namelist.wps"
    vtable_bytes = b"GRIB1| Level| From |  To  |\r\n  TT  | 100  |\n"
    wps_bytes = b"&share\n wrf_core = 'ARW',\n/\n"
    vtable.write_bytes(vtable_bytes)
    wps.write_bytes(wps_bytes)
    payload = (
        "[experiment]\n"
        "name = 'fixture'\n"
        "[case_data]\n"
        "forcing = 'forcing/*.grib'\n"
        "vtable = 'Vtable'\n"
        "wps_namelist = 'namelist.wps'\n"
        "geog_root = 'geog'\n"
        "sfcp_to_sfcp = true\n"
        "output_title = 'fixture'\n"
    ).encode("utf-8")
    config = source / "experiment.toml"
    captured = outdir / "captured.toml"
    config.write_bytes(payload)
    captured.write_bytes(payload)
    monkeypatch.setattr(
        case_data, "build_experiment",
        lambda *_args, **_kwargs: SimpleNamespace(domains=()))
    # A stand-in with no domains takes the experiment's place (so no sub-km
    # terrain default applies), and the door's plan review of it
    # (review_root_footprint) is stubbed beside the builder.
    monkeypatch.setattr(case_data, "review_root_footprint",
                        lambda _experiment, _source: None)
    parent_hashes = supervisor.resolved_input_hashes(
        config, config_bytes=payload)

    def injected_failure(*_args, **_kwargs):
        raise RuntimeError("injected CPU-path failure")

    monkeypatch.setattr(runtime, "run_experiment", injected_failure)
    monkeypatch.setattr(supervisor, "git_commit", lambda: "test-commit")
    monkeypatch.setenv("WOOF_RUN_ID", "embed-run")
    monkeypatch.setenv(
        "WOOF_CONFIG_DIGEST", hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv("WOOF_STARTED_AT_UTC", "2026-08-05T00:00:00Z")
    monkeypatch.setenv("WOOF_GPU_UUID", "GPU-test")
    monkeypatch.setenv("WOOF_GPU_DRIVER", "610.74")
    monkeypatch.setenv("WOOF_GPU_NAME", "RTX 5090")
    monkeypatch.setenv("WOOF_INPUT_HASHES_JSON", json.dumps(parent_hashes))
    monkeypatch.delenv(supervisor.INPUT_AUTHORITIES_ENV, raising=False)

    with pytest.raises(RuntimeError, match="injected CPU-path failure"):
        supervisor._worker_main(supervisor.argparse.Namespace(
            config=config, config_payload=captured, outdir=outdir,
            restart=None, health_debug=False))

    capsule = json.loads((
        outdir / supervisor.FAILURE_CAPSULE_NAME).read_text(encoding="utf-8"))
    assert capsule["schema"] == "gpuwm.failure-capsule/v3"
    config_text = capsule["config_text"]
    assert config_text["text"].encode("utf-8") == payload
    assert config_text["truncated"] is False
    assert config_text["size_bytes"] == len(payload)
    expected_keys = {
        f"vtable:{vtable.resolve()}", f"wps_namelist:{wps.resolve()}"}
    assert set(capsule["input_text"]) == expected_keys
    assert set(capsule["input_text"]) <= set(capsule["input_hashes"])
    embedded_vtable = capsule["input_text"][f"vtable:{vtable.resolve()}"]
    assert embedded_vtable["text"].encode("utf-8") == vtable_bytes
    assert embedded_vtable["truncated"] is False
    embedded_wps = capsule["input_text"][f"wps_namelist:{wps.resolve()}"]
    assert embedded_wps["text"].encode("utf-8") == wps_bytes
    assert not any(key.startswith(("forcing:", "geog_root:"))
                   for key in capsule["input_text"])


def test_failure_capsule_truncates_oversize_small_text_at_the_cap(
        monkeypatch, tmp_path):
    """The 64 KB cap bites with an explicit marker and the full true size."""
    monkeypatch.setattr(supervisor, "git_commit", lambda: "test-commit")
    cap = supervisor.FAILURE_CAPSULE_TEXT_CAP_BYTES
    vtable = tmp_path / "Vtable"
    vtable_bytes = b"V" * (cap + 4096)
    vtable.write_bytes(vtable_bytes)
    config = tmp_path / "case.toml"
    config_bytes = b"C" * (cap + 1)
    config.write_bytes(config_bytes)
    input_hashes = {
        f"vtable:{vtable.resolve()}": {
            "algorithm": "sha256",
            "digest": hashlib.sha256(vtable_bytes).hexdigest(),
            "detail": None,
            "identities": [{"role": "vtable",
                            "path": str(vtable.resolve()),
                            "detail": None}],
        },
    }

    path = supervisor.write_failure_capsule(
        tmp_path / "failure-capsule.json", run_id="cap-run",
        config_path=config, config_sha256="a" * 64,
        input_hashes=input_hashes,
        gpu=GPUIdentity("GPU-test", "610.74", "RTX 5090"),
        last_phase="stepping:outer-1", last_step=0,
        exception_type="RuntimeError", exception_message="boom",
        exception_traceback="trace", last_durable_wrfout=None,
        last_checkpoint=None, worker_pid=1234,
        config_bytes=config_bytes)

    capsule = json.loads(path.read_text(encoding="utf-8"))
    config_text = capsule["config_text"]
    assert config_text["truncated"] is True
    assert config_text["size_bytes"] == cap + 1
    assert config_text["text"].encode("utf-8") == config_bytes[:cap]
    embedded = capsule["input_text"][f"vtable:{vtable.resolve()}"]
    assert embedded["truncated"] is True
    assert embedded["size_bytes"] == cap + 4096
    assert embedded["text"].encode("utf-8") == vtable_bytes[:cap]


def test_failure_capsule_records_absent_text_instead_of_raising(
        monkeypatch, tmp_path):
    """The capsule writer must never be the thing that crashes a crash
    report: a deleted input, an unwritten config, or a mangled inventory
    entry each degrade to a recorded absence."""
    monkeypatch.setattr(supervisor, "git_commit", lambda: "test-commit")
    missing_vtable = tmp_path / "deleted" / "Vtable"
    input_hashes = {
        f"vtable:{missing_vtable}": {
            "algorithm": "sha256", "digest": "b" * 64, "detail": None,
            "identities": [{"role": "vtable",
                            "path": str(missing_vtable), "detail": None}],
        },
        "forcing:ignored": {
            "algorithm": "sha256", "digest": "c" * 64, "detail": None,
            "identities": [{"role": "forcing", "path": "x",
                            "detail": None}],
        },
        "mangled:entry": "not-an-object",
        "no-identities:entry": {"algorithm": "sha256", "digest": "d" * 64},
    }

    path = supervisor.write_failure_capsule(
        tmp_path / "failure-capsule.json", run_id="absent-run",
        config_path=tmp_path / "never-written.toml", config_sha256="a" * 64,
        input_hashes=input_hashes,
        gpu=GPUIdentity("GPU-test", "610.74", "RTX 5090"),
        last_phase="preparing:load-config", last_step=0,
        exception_type="RuntimeError", exception_message="boom",
        exception_traceback="trace", last_durable_wrfout=None,
        last_checkpoint=None, worker_pid=1234)

    capsule = json.loads(path.read_text(encoding="utf-8"))
    config_text = capsule["config_text"]
    assert config_text["text"] is None
    assert "FileNotFoundError" in config_text["error"]
    assert set(capsule["input_text"]) == {f"vtable:{missing_vtable}"}
    embedded = capsule["input_text"][f"vtable:{missing_vtable}"]
    assert embedded["text"] is None
    assert "FileNotFoundError" in embedded["error"]


def test_failure_capsule_identifies_the_code_that_ran_not_the_checkout(
        tmp_path):
    """A capsule must name the RUNNING distribution, not just a checkout.

    The support regression this pins: ``git_commit`` shells out from the
    package directory, so an install into
    ``<checkout>/.venv/lib/pythonX/site-packages`` makes git walk up and
    report the ENCLOSING CHECKOUT's HEAD.  A reporter who pulls a new tag
    without reinstalling then files a capsule naming the new commit while
    the old wheel produced every byte of the failure -- and triage spends
    its time on the wrong release.  ``installed`` answers the question
    ``git_commit`` cannot.
    """

    import woof

    config = tmp_path / "case.toml"
    config.write_text("[experiment]\nname = 'x'\n", encoding="utf-8")

    path = supervisor.write_failure_capsule(
        tmp_path / "failure-capsule.json", run_id="ident-run",
        config_path=config, config_sha256="b" * 64,
        input_hashes={},
        gpu=GPUIdentity("GPU-test", "610.74", "RTX 5090"),
        last_phase="stepping:outer-1", last_step=0,
        exception_type="FloatingPointError",
        exception_message="YSU returned non-finite dtheta tendency",
        exception_traceback="trace", last_durable_wrfout=None,
        last_checkpoint=None, worker_pid=4242)

    capsule = json.loads(path.read_text(encoding="utf-8"))

    assert "installed" in capsule, (
        "capsule cannot identify the running code: it carries only "
        f"git_commit={capsule.get('git_commit')!r}, which is the enclosing "
        "checkout's HEAD")
    installed = capsule["installed"]
    assert installed["distribution"] == woof.DISTRIBUTION_NAME
    assert installed["version"] == str(woof.__version__)
    # The import path is the stale-install signature: site-packages here
    # beside a checkout-derived git_commit means the two disagree.
    assert (Path(installed["package_path"])
            == Path(supervisor.__file__).resolve().parent)

    # The schema id is itself a version tell: it is emitted by the code
    # that ran, so it survives a misleading git_commit.
    assert capsule["schema"] == "gpuwm.failure-capsule/v3"


def test_git_commit_reads_the_enclosing_checkout_of_the_package(monkeypatch):
    """Pin WHY ``git_commit`` alone cannot identify the running code.

    It runs ``git rev-parse`` with the package's parent as the working
    directory and git searches upward from there, so the answer describes
    whatever repository encloses the install -- not the install.  Asserting
    the cwd keeps that documented and keeps a future reader from treating
    the field as the running version.
    """

    seen = {}

    class _Result:
        returncode = 0
        stdout = "deadbeef\n"
        stderr = ""

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["cwd"] = Path(kwargs["cwd"])
        return _Result()

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)

    assert supervisor.git_commit() == "deadbeef"
    assert seen["command"] == ["git", "rev-parse", "HEAD"]
    # Parent of the package directory -- NOT the package, and not a path
    # the running distribution is guaranteed to have anything to do with.
    assert seen["cwd"] == Path(supervisor.__file__).resolve().parents[1]
    assert seen["cwd"] != Path(supervisor.__file__).resolve().parent


def test_one_shared_card_is_named_once_per_process_not_once_per_call(
        monkeypatch, warnings_seen):
    """`woof.explain.warn` repeats; the run doors ask more than once.

    The stream controller asks this door before EVERY stage command, so
    the same sentence was printed per command on a shared card.  The
    sentence is said once; a device reading that CHANGED is news and is
    said again.
    """

    import woof.core.preflight as preflight
    import woof.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    processes = (GPUProcess(gpu.uuid, 41001, "python.exe", 4096, "C"),)
    device = {"used": 4 * 2 ** 30}
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)
    monkeypatch.setattr(supervisor, "_ADMISSION_WARNED", set())
    monkeypatch.setattr(
        preflight, "device_physical_total_bytes",
        lambda **kwargs: 32 * 2 ** 30)
    monkeypatch.setattr(
        preflight, "device_wide_used_bytes", lambda **kwargs: device["used"])

    for _ in range(4):
        assert preflight_exclusive_gpu(
            gpu.uuid, reservation_bytes=8 * 2 ** 30)["verdict"] == "admitted"
    assert len(warnings_seen) == 1

    device["used"] = 8 * 2 ** 30
    preflight_exclusive_gpu(gpu.uuid, reservation_bytes=8 * 2 ** 30)
    assert len(warnings_seen) == 2
    assert "24.00 GiB" in warnings_seen[-1]["action"]


def test_one_pricing_function_answers_from_a_path_or_a_loaded_experiment(
        tmp_path, monkeypatch):
    """Both run doors price the same configuration to the same number."""

    import woof.core.preflight as preflight
    import woof.supervisor as supervisor

    class _Estimate:
        peak_envelope_bytes = 7 * 2 ** 30

    loaded = object()
    monkeypatch.setattr(
        preflight, "admission_estimate",
        lambda experiment, **kwargs: _Estimate())
    config = tmp_path / "experiment.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "woof.experiment.load_experiment", lambda path: loaded)

    assert supervisor.priced_reservation_bytes(config) == 7 * 2 ** 30
    assert supervisor.priced_reservation_bytes(str(config)) == 7 * 2 ** 30
    assert supervisor.priced_reservation_bytes(loaded) == 7 * 2 ** 30
