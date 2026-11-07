"""Real subprocess coverage for the go forecast watchdog."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from woof import go_cli, supervisor


@pytest.fixture
def forecast_command(tmp_path, monkeypatch):
    module = tmp_path / "forecast_probe.py"
    module.write_text('''
import json
import sys
import time
from pathlib import Path

def main(argv=None, *, observer=None):
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[argv.index("--outdir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    evidence = out / "evidence"
    evidence.mkdir()
    (evidence / "progress.json").write_text(json.dumps({"status": "RUNNING"}))
    if observer is not None:
        observer(model_elapsed_seconds=60., outer_step=1,
                 last_durable_wrfout=None, last_checkpoint=None)
        observer.finalizing("drain-history-writers", work_bytes=1024)
    if "--stall" in argv:
        time.sleep(30)
    else:
        time.sleep(.4)
    (evidence / "progress.json").write_text(json.dumps({"status": "PASS"}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
''', encoding="utf-8")
    stage_env = go_cli._stage_env()
    stage_env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(go_cli.__file__).resolve().parents[1])])
    monkeypatch.setattr(go_cli, "_stage_env", lambda: stage_env)
    return [sys.executable, "-m", "forecast_probe", "--outdir", str(tmp_path / "run")]


def test_go_forecast_publishes_supervised_record(forecast_command):
    go_cli._run_stage("forecast", forecast_command, explain=False)
    out = Path(forecast_command[-1])
    beat = supervisor.read_heartbeat(out / "run-progress.json")
    assert beat.status == "complete"
    assert beat.outer_step == 1
    assert beat.model_elapsed_seconds == 60.
    assert json.loads((out / "evidence/progress.json").read_text())["status"] == "PASS"


def test_go_forecast_kills_stalled_finalization(forecast_command, monkeypatch):
    monkeypatch.setattr(supervisor, "finalization_stale_threshold_seconds",
                        lambda *args: .3)
    started = time.monotonic()
    with pytest.raises(go_cli.GoStageFailed) as caught:
        go_cli._run_stage("forecast", [*forecast_command, "--stall"], explain=False)
    assert caught.value.code == 124
    assert "finalizing:drain-history-writers" in caught.value.diagnostic
    assert time.monotonic() - started < 15
    out = Path(forecast_command[-1])
    assert json.loads((out / "evidence/progress.json").read_text())["status"] == "RUNNING"


@pytest.mark.parametrize("phase", [None, "preparing:restore", "integrating",
                                   "finalizing:hash-output-frames-1-of-2",
                                   "writing:history-d01"])
def test_watchdog_bounds_each_nonterminal_phase(tmp_path, monkeypatch, phase):
    from woof import forecast_supervisor

    now = [0.]
    monkeypatch.setattr(forecast_supervisor.time, "monotonic", lambda: now[0])
    watch = forecast_supervisor.ForecastWatchdog(
        [sys.executable, "-m", "probe", "--outdir", str(tmp_path)])
    beat = supervisor.RuntimeHeartbeat(
        watch.path, run_id=watch.run_id, config_sha256=watch.digest,
        started_at_utc=watch.started_at)
    work = (80 * 1024 * 1024
            if phase and phase.startswith(("finalizing:", "writing:")) else None)
    if phase is not None:
        beat._write(phase, work_bytes=work)
    phase = phase or "preparing:launch"
    assert watch.check(os.getpid()) is None
    if phase.startswith("preparing:"):
        # The default watchdog, as go builds it: preparation has no bound,
        # so a week of launch or restore is still not a stall.
        now[0] = 7 * 86400.
        assert watch.check(os.getpid()) is None
        return
    bound = 120. + (10 if work else 0)
    now[0] = bound - .1
    assert watch.check(os.getpid()) is None
    now[0] = bound + .1
    assert phase in watch.check(os.getpid())


# A stand-in for one streamed domain's history write, driven through the
# writer set every prepared runner submits to and supervised by the go
# watchdog with the runner's own heartbeat.  Two model steps are published
# first, so the write lands between two step records exactly as a frame at
# a history time does.  The stand-in writes nothing: it only takes the time
# a large frame takes.
_WRITE_PROBE = '''
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from woof.io.wrfout import PerDomainWrfoutWriters


class StandInWriter:
    global_attrs = {}
    pending_work_bytes = 0

    def __init__(self, seconds):
        self.seconds = seconds

    def submit(self, *args, **kwargs):
        pass

    def drain_staging(self):
        time.sleep(self.seconds)

    def drain(self):
        pass


class StandInStore:
    def history_fields(self, *, state=None):
        return {"T": np.zeros(1024 * 1024, dtype=np.float32)}


def main(argv=None, *, observer=None):
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[argv.index("--outdir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    seconds = float(argv[argv.index("--write-seconds") + 1])
    writers = object.__new__(PerDomainWrfoutWriters)
    writers.start_time = __import__("datetime").datetime(2024, 5, 20, 12)
    writers.output_dir = out / "wrfout"
    writers._episode_by_grid_id = {1: 0}
    writers._published_paths = set()
    writers._metadata_by_grid_id = {1: {}}
    writers._writers = {1: StandInWriter(seconds)}
    writers.attach_write_progress(observer)
    node = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), clock=SimpleNamespace(tick_den=1),
        state=SimpleNamespace(_streamed_domain=StandInStore()))
    for step in (1, 2):
        observer(model_elapsed_seconds=15. * step, outer_step=step,
                 last_durable_wrfout=None, last_checkpoint=None)
    writers.submit(node, 45)
    observer(model_elapsed_seconds=45., outer_step=3,
             last_durable_wrfout=None, last_checkpoint=None)
    return 0
'''


@pytest.fixture
def write_probe(tmp_path, monkeypatch):
    (tmp_path / "write_probe.py").write_text(_WRITE_PROBE, encoding="utf-8")
    stage_env = go_cli._stage_env()
    stage_env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(go_cli.__file__).resolve().parents[1])])
    monkeypatch.setattr(go_cli, "_stage_env", lambda: stage_env)
    # A step bound of half a second stands for the 120 s floor, and a
    # write floor of 8 MiB in five seconds for 8 MiB/s, so the stand-in's
    # 8 MiB frame (4 MiB written, then read back) buys five seconds.
    monkeypatch.setattr(supervisor.RollingStepWall, "stale_threshold_seconds",
                        property(lambda self: .5))
    monkeypatch.setattr(supervisor, "FINALIZATION_FLOOR_BYTES_PER_SECOND",
                        8 * 1024 * 1024 / 5.)
    return [sys.executable, "-m", "write_probe", "--outdir", str(tmp_path / "run")]


def test_a_history_write_past_the_step_bound_finishes(write_probe):
    # The 1132x906x55 streamed forecast: every step done, the last frame's
    # write longer than the step bound, and the worker was stopped (exit
    # 124) with nothing wrong.  The write is its own record now, bounded
    # by its bytes, and the step after it is timed from its end.
    go_cli._run_stage("forecast", [*write_probe, "--write-seconds", "1.5"],
                      explain=False)
    beat = supervisor.read_heartbeat(Path(write_probe[-1]) / "run-progress.json")
    assert (beat.status, beat.outer_step) == ("complete", 3)


def test_a_history_write_that_never_ends_is_still_stopped(write_probe):
    started = time.monotonic()
    with pytest.raises(go_cli.GoStageFailed) as caught:
        go_cli._run_stage("forecast", [*write_probe, "--write-seconds", "60"],
                          explain=False)
    assert caught.value.code == 124
    assert "forecast stalled in writing:history-d01" in caught.value.diagnostic
    assert time.monotonic() - started < 30


def test_a_write_hands_back_the_status_it_interrupted(tmp_path, monkeypatch):
    from woof import forecast_supervisor

    now = [0.]
    monkeypatch.setattr(forecast_supervisor.time, "monotonic", lambda: now[0])
    watch = forecast_supervisor.ForecastWatchdog(
        [sys.executable, "-m", "probe", "--outdir", str(tmp_path)])
    beat = forecast_supervisor.ForecastHeartbeat(
        watch.path, run_id=watch.run_id, config_sha256=watch.digest,
        started_at_utc=watch.started_at)
    # The analysis frame is written before the first step: preparation's
    # rules (no deadline) come back once it is written.
    beat(model_elapsed_seconds=0., outer_step=0, last_durable_wrfout=None,
         last_checkpoint=None)
    with supervisor.writing_progress(beat, "history-d01", work_bytes=1024):
        record = supervisor.read_heartbeat(watch.path)
        assert (record.status, record.work_bytes) == ("writing:history-d01", 1024)
    assert supervisor.read_heartbeat(watch.path).status == "preparing:first-step"
    assert watch.check(os.getpid()) is None
    now[0] += 7200.
    assert watch.check(os.getpid()) is None
    # Mid-run, the step bound comes back after the write.
    for step in (1, 2):
        beat(model_elapsed_seconds=60. * step, outer_step=step,
             last_durable_wrfout=None, last_checkpoint=None)
    with supervisor.writing_progress(beat, "checkpoint"):
        pass
    record = supervisor.read_heartbeat(watch.path)
    assert (record.status, record.outer_step) == ("integrating", 2)
    assert watch.check(os.getpid()) is None
    now[0] += 121.
    assert "stalled in integrating" in watch.check(os.getpid())


def test_watchdog_rejects_another_attempt(tmp_path):
    from woof import forecast_supervisor

    watch = forecast_supervisor.ForecastWatchdog(
        [sys.executable, "-m", "probe", "--outdir", str(tmp_path)])
    beat = supervisor.RuntimeHeartbeat(
        watch.path, run_id="previous-attempt", config_sha256=watch.digest,
        started_at_utc=watch.started_at)
    beat.starting()
    assert "identity violation" in watch.check(os.getpid())
    assert supervisor.read_heartbeat(watch.path).run_id == "previous-attempt"


def test_worker_refusal_preserves_existing_run(tmp_path, monkeypatch):
    from woof import forecast_supervisor
    from types import SimpleNamespace

    path = tmp_path / "run-progress.json"
    beat = supervisor.RuntimeHeartbeat(path, run_id="previous", config_sha256="",
                                        started_at_utc=supervisor.utc_now())
    beat.complete(60.)
    original = path.read_bytes()
    monkeypatch.setenv("WOOF_FORECAST_RUN_ID", "new")
    monkeypatch.setenv("WOOF_FORECAST_CONFIG_DIGEST", "")
    monkeypatch.setenv("WOOF_FORECAST_STARTED_AT", supervisor.utc_now())
    monkeypatch.setattr(forecast_supervisor.importlib, "import_module",
                        lambda _: SimpleNamespace(main=lambda *a, **k: 2))
    assert forecast_supervisor.main(["probe", "--outdir", str(tmp_path)]) == 2
    assert path.read_bytes() == original


def test_the_checkpoint_read_back_is_worded_as_a_check(tmp_path):
    """The read-back of a new checkpoint is timed as a write, not worded as one.

    Both progress lines said "writing verify checkpoint" while the run
    was reading back the checkpoint it had just written.
    """
    from io import StringIO

    from woof.progress import ForecastProgress

    assert go_cli._heartbeat_note("writing:verify-checkpoint") == (
        ", checking checkpoint")
    assert go_cli._heartbeat_note("writing:history-d01") == (
        ", writing history d01")
    assert go_cli._heartbeat_note("finalizing:write-receipts") == (
        ", finalizing write receipts")
    assert go_cli._heartbeat_note("integrating") is None

    stream = StringIO()
    reporter = ForecastProgress(stream=stream)
    beat = supervisor.RuntimeHeartbeat(
        tmp_path / supervisor.HEARTBEAT_NAME, run_id="run",
        config_sha256="0" * 64, started_at_utc="2026-09-28T00:00:00Z")
    beat.writing("verify-checkpoint", work_bytes=4096)
    reporter(supervisor.read_heartbeat(beat.path))
    beat.writing("checkpoint", work_bytes=4096)
    reporter(supervisor.read_heartbeat(beat.path))
    first, second = stream.getvalue().splitlines()
    assert "simulated; checking checkpoint |" in first, first
    assert "writing" not in first
    assert "simulated; writing checkpoint |" in second, second
