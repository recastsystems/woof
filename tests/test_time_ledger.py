"""C1 real emitters, process contention, shell failure and orchestration seams."""

import argparse
import copy
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from woof.timeline.ledger import Ledger
from woof.timeline.taxonomy import validate
from woof.stage_timing import (ForecastLedger, LedgerObserver, stage_timer, ledger_main, bind_ledger,
                                ledger_environment)


def _append(path, count):
    ledger = Ledger(path, run_id="unit")
    for index in range(count):
        assert ledger.emit("boot.concurrent", attrs={"index": index, "payload": "x" * 1024})


def _records(path):
    return [validate(json.loads(line)) for line in path.read_text().splitlines()]


def test_prepare_only_failure_keeps_original_error_and_closes_once(tmp_path, monkeypatch):
    import tomllib
    from woof import go_cli, runplan
    from woof.cli import build_parser
    fixture = Path(__file__).parent / 'fixtures/static_asset_configs/selected-source.toml'
    output = tmp_path / 'prepare'
    ledger = tmp_path / 'ledger.jsonl'
    monkeypatch.setenv('WOOF_TIME_LEDGER', str(ledger))
    args = build_parser().parse_args(['go', str(fixture), '--prepare-only',
                                     '--outdir', str(output), '--run-stamp', 'off'])
    def failed(plan, *, observer, **kwargs):
        observer.enter_stage('fetch', phase='fetch')
        raise runplan.PlanError('original preparation refusal: missing forcing field')
    monkeypatch.setattr(runplan, '_staged_chain', failed)
    monkeypatch.setattr(runplan, '_hrrr_chain', failed)
    @ledger_main
    def invoke():
        return go_cli._prepare_only_launch(args, config=fixture,
                    payload=tomllib.loads(fixture.read_text()))
    with pytest.raises(go_cli.GoRefusal, match='original preparation refusal'):
        invoke()
    events = [json.loads(line) for line in (output / 'preparation-events.jsonl').read_text().splitlines()]
    assert sum(row['event'] == 'stage_finished' for row in events) == 1
    assert sum(row['stage'] == 'finish' for row in _records(ledger)) == 1


def test_schema_and_environment_destination(tmp_path, monkeypatch):
    path = tmp_path / "rental-ledger.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(path))
    monkeypatch.setenv("WOOF_RENTAL_ID", "rental")
    monkeypatch.setenv("WOOF_RUN_ID", "run")
    monkeypatch.setenv("WOOF_PRODUCT", "regional")
    assert Ledger(tmp_path / "ignored.jsonl", card=2).emit("forecast.integrate", "begin")
    row, = _records(path)
    assert row["rental_id"] == "rental" and row["run_id"] == "run"
    assert row["card"] == 2 and row["pid"] == os.getpid()
    assert not (tmp_path / "ignored.jsonl").exists()
    for key, bad in (("product", "other"), ("phase", "start"), ("stage", "unknown.x"),
                     ("t_mono", float("nan")), ("card", True), ("attrs", [])):
        with pytest.raises(ValueError):
            validate({**row, key: bad})


def test_eight_processes_no_torn_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    path = tmp_path / "timeline.jsonl"
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_append, args=(path, 250)) for _ in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0
    rows = _records(path)
    assert len(rows) == 2000
    assert len({row["pid"] for row in rows}) == 8
    for pid in {row["pid"] for row in rows}:
        assert [row["attrs"]["index"] for row in rows if row["pid"] == pid] == list(range(250))


def test_unwritable_warns_once_names_path(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    # A directory is unwritable as a file even when tests run as root.
    target = tmp_path / "directory"
    target.mkdir()
    ledger = Ledger(target)
    for _ in range(20):
        assert not ledger.emit("boot.ready")
    warning = capsys.readouterr().err
    assert warning.count("time ledger:") == 1 and str(target) in warning


def test_serialization_failure_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    ledger = Ledger(tmp_path / "bad.jsonl")
    assert not ledger.emit("boot.ready", attrs={"bad": object()})
    assert not ledger.emit("boot.ready", attrs={"bad": float("nan")})
    assert capsys.readouterr().err.count("time ledger:") == 1


def _posix_bash():
    """A bash that can source tools/timeline/ledger.sh, or None.

    On Windows the first bash on PATH can be System32's WSL launcher, which
    is no shell without an installed distribution (the windows-2025 box:
    "Windows Subsystem for Linux must be updated"). Git for Windows ships the
    bash the emitter runs under there, found as tests/test_ruc_oracle_harness.py
    finds it.
    """
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if git:
        # git.exe sits in Git/cmd, Git/bin or Git/mingw64/bin; bash.exe in Git/bin.
        for root in list(Path(git).resolve().parents)[:3]:
            for candidate in (root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe"):
                if candidate.is_file():
                    return str(candidate)
    found = shutil.which("bash")
    if found and Path(found).resolve().parent.name.lower() != "system32":
        return found
    return None


def test_shell_schema_and_warn_once(tmp_path):
    bash = _posix_bash()
    if bash is None:
        pytest.skip("requires bash (Git Bash on Windows)")
    # Forward slashes: the emitter finds woof/timeline/ledger.py through its
    # own dirname, which a backslashed Windows path defeats.
    script = (Path(__file__).parents[1] / "tools/timeline/ledger.sh").as_posix()
    path = tmp_path / "shell.jsonl"
    env = {**os.environ, "WOOF_TIME_LEDGER": str(path)}
    command = 'set -euo pipefail; source "$1"; ledger_emit boot.ready begin; ledger_emit boot.ready end \'{"ok":true}\''
    result = subprocess.run([bash, "-c", command, "bash", script], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    rows = _records(path)
    assert [row["phase"] for row in rows] == ["begin", "end"]
    assert {k: v for k, v in rows[1]["attrs"].items() if k != "cpu_s"} == {"ok": True}
    env["WOOF_TIME_LEDGER"] = str(tmp_path)
    result = subprocess.run([bash, "-c", command, "bash", script], env=env, text=True, capture_output=True)
    assert result.returncode == 0 and result.stderr.count("time ledger: ") == 1
    assert str(tmp_path) in result.stderr


def test_existing_progress_timer_boundaries(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    # Use the actual runner's log writer, not fabricated event dictionaries.
    from woof.progress_log import StepLog
    from datetime import datetime
    path = tmp_path / "timeline.jsonl"
    relay = ForecastLedger(Ledger(path), tmp_path)
    relay.start()
    log = StepLog(start_time=datetime(2026, 1, 1), run_seconds=3600,
                  text_stream=None, jsonl_path=tmp_path / "progress.jsonl")
    log.domain_step(grid_id=1, step_count=1, model_seconds=10, step_wall_seconds=0.05)
    time.sleep(0.06)
    log.domain_step(grid_id=1, step_count=2, model_seconds=20, step_wall_seconds=0.04)
    log.close()
    relay.finish()
    rows = _records(path)
    first = [row for row in rows if row["stage"] == "forecast.first_step"]
    integration = [row for row in rows if row["stage"] == "forecast.integrate"]
    assert [row["phase"] for row in first] == ["begin", "end"]
    assert first[1]["t_mono"] - first[0]["t_mono"] == pytest.approx(0.05, abs=0.001)
    assert integration[0]["t_mono"] == first[0]["t_mono"]
    assert integration[-1]["t_mono"] > first[1]["t_mono"]


def test_main_restores_environment_and_closes_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    monkeypatch.delenv("WOOF_RUN_ID", raising=False)
    class Sink:
        hosts_forecast = False
    @ledger_main
    def main():
        observer = LedgerObserver(Sink(), bind_ledger(tmp_path), tmp_path)
        assert observer.hosts_forecast is False
        observer.stage_begin(label="prepare", command=[])
        raise RuntimeError("work failed")
    with pytest.raises(RuntimeError, match="work failed"):
        main()
    assert "WOOF_TIME_LEDGER" not in os.environ and "WOOF_RUN_ID" not in os.environ
    rows = _records(tmp_path / "timeline.jsonl")
    assert [(row["stage"], row["phase"]) for row in rows
            if row["stage"] not in ("post.orchestration", "boot.engine_admission")] == [
        ("prep.inputs", "begin"), ("prep.inputs", "end"), ("finish", "mark")]


def test_skeleton_parser_and_hooks_are_inert():
    from woof.go_cli import register_cli, SpeedHooks
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["go", "config.toml", "--stage-only", "fetch", "--warm-kernels",
                             "--static-cache-root", "/cache", "--static-cache-key", "key",
                             "--readiness-fold", "--live-products-width", "2"])
    plan = {"command": ["unchanged"], "nested": {"value": 1}}
    before_plan, before_args = copy.deepcopy(plan), copy.deepcopy(vars(args))
    for name in ("stage_only", "warm_kernels_spawn", "static_cache", "readiness_fold", "live_products_width"):
        assert getattr(SpeedHooks, name)(args, plan) is None
    assert plan == before_plan and vars(args) == before_args


def test_timer_preserves_work_exception(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    path = tmp_path / "timeline.jsonl"
    with pytest.raises(ValueError, match="work"):
        with stage_timer("prep.decode", ledger=Ledger(path)):
            raise ValueError("work")
    assert [row["phase"] for row in _records(path)] == ["begin", "end"]


def test_module_cli_has_no_double_import_warning(tmp_path):
    env = {**os.environ, "WOOF_TIME_LEDGER": str(tmp_path / "cli.jsonl")}
    result = subprocess.run([os.sys.executable, "-m", "woof.timeline.ledger", "boot.ready"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0 and result.stderr == ""
    assert len(_records(tmp_path / "cli.jsonl")) == 1
    result = subprocess.run([os.sys.executable, "-m", "woof.timeline.ledger", "--help"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0 and result.stderr == ""


def test_replaced_progress_log_opens_new_attempt(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    from woof.progress_log import StepLog
    from datetime import datetime
    path = tmp_path / "timeline.jsonl"
    relay = ForecastLedger(Ledger(path), tmp_path)
    relay.start(watch=False)
    for attempt in (1, 2):
        progress = tmp_path / "progress.jsonl"
        if attempt == 2:
            progress.rename(tmp_path / "old-progress.jsonl")
        log = StepLog(start_time=datetime(2026, 1, 1), run_seconds=3600,
                      text_stream=None, jsonl_path=progress)
        log.domain_step(grid_id=1, step_count=1, model_seconds=10, step_wall_seconds=0.05)
        log.close()
        relay.read()
    relay.finish()
    pairs = [row for row in _records(path) if row["stage"] == "forecast.integrate"]
    assert [(row["phase"], row["attrs"]["attempt"]) for row in pairs] == [
        ("begin", 1), ("end", 1), ("begin", 2), ("end", 2)]


def test_unnamed_progress_is_not_invented(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    path = tmp_path / "timeline.jsonl"
    relay = ForecastLedger(Ledger(path), tmp_path)
    relay.start(watch=False)
    relay.finish()
    rows = _records(path)
    assert all(row["stage"] != "forecast.integrate" for row in rows)
    assert rows[-1]["attrs"]["coverage"] == "unknown"


def test_launch_ledger_does_not_prepopulate_forecast_output(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    monkeypatch.delenv("WOOF_RUN_ID", raising=False)
    from woof.prepared_single_domain_forecast import claim_output_directory
    root = tmp_path / "launch"
    forecast = root / "run"
    # bind_ledger routes the process (WOOF_TIME_LEDGER, WOOF_RUN_ID) as a
    # launch does, and the delenv calls above record nothing to restore when
    # the variables were unset; without ledger_environment every later test
    # in the worker wrote its timeline here (test_obs_prepare_parallel found
    # no timeline.jsonl when it followed this file on test host 4).
    with ledger_environment():
        ledger = bind_ledger(root)
        assert ledger.emit("prep.inputs", "begin")
        assert (root / "timeline.jsonl").exists()
        assert not forecast.exists()
        assert Path(claim_output_directory(forecast)).resolve() == forecast.resolve()


def test_stage_only_seam_precedes_gpu_and_config_admission(monkeypatch):
    from woof.go_cli import SpeedHooks, go_main
    monkeypatch.setattr(SpeedHooks, "stage_only", lambda args, plan: 7)
    assert go_main(argparse.Namespace(config=Path("absent.toml"))) == 7


def test_hosted_poll_batches_but_completion_keeps_the_last_step(tmp_path, monkeypatch):
    from woof.progress_log import StepLog
    from datetime import datetime
    import woof.stage_timing as timing
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    clock = [99.0]
    monkeypatch.setattr(timing.time, "monotonic", lambda: clock[0])
    ledger_path = tmp_path / "timeline.jsonl"
    relay = ForecastLedger(Ledger(ledger_path), tmp_path)
    relay.start(watch=False)
    log = StepLog(start_time=datetime(2026, 1, 1), run_seconds=3600,
                  text_stream=None, jsonl_path=tmp_path / "progress.jsonl")
    log.domain_step(grid_id=1, step_count=1, model_seconds=15, step_wall_seconds=0.05)
    clock[0] = 100.0
    relay.poll()
    log.domain_step(grid_id=1, step_count=240, model_seconds=3600, step_wall_seconds=0.07)
    log.close()
    offset = relay.offset
    relay.poll()
    assert relay.offset == offset
    relay.finish()
    rows = _records(ledger_path)
    pairs = [row for row in rows if row["stage"] == "forecast.integrate"]
    assert [row["phase"] for row in pairs] == ["begin", "end"]
    last = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == "step"][-1]
    measured = datetime.fromisoformat(pairs[-1]["t_utc"].replace("Z", "+00:00")).timestamp()
    assert abs(measured - last["emitted_unix_ms"] / 1000) < 1e-6


def test_posting_wait_reaches_event_stream_before_close(tmp_path, monkeypatch):
    # Waiting until close hid a live forecast's source wait from its reader.
    import threading
    from woof import chain_events
    from woof.ingest.boundary_stream import WAIT_LOG_NAME

    (tmp_path / WAIT_LOG_NAME).write_text(json.dumps({
        "event": "source_wait_started", "lead": 6}) + "\n")
    chain = chain_events.GoChainEvents()
    carried = threading.Event()
    relay_waits = chain.relay_waits

    def relay():
        if relay_waits():
            carried.set()

    monkeypatch.setattr(chain, "relay_waits", relay)
    monkeypatch.setattr(chain_events, "LIVE_RELAY_SECONDS", 0.01)
    path = tmp_path / "events.jsonl"
    chain.open(path, plan={"run": tmp_path})
    try:
        assert carried.wait(10), "source wait was not relayed while the stream was open"
        assert sum(json.loads(line)["event"] == "source_wait_started"
                   for line in path.read_text().splitlines()) == 1
    finally:
        chain.close()
