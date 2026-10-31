"""woof.live_columns: the any-point column store written as frames land.

Breakage it prevents: a run whose soundings have no store for an output hour (a frame off the cadence that ended
the run, a grid whose frames were never queued), and a failed store write that stops the forecast instead of
being a warning.
"""
from __future__ import annotations

from pathlib import Path

from woof.live_columns import LiveColumns, columns_enabled, on_cadence


def frame(domain: int, stamp: str) -> Path:
    return Path(f"/run/wrfout_d{domain:02d}_{stamp}")


def test_whole_hours_of_every_grid_and_each_grids_last_frame():
    written = []

    def writer(path, out, threads=None):
        written.append(Path(path).name)
        return [{"path": str(out / Path(path).name), "bytes": 10, "ms": 1.0}]

    live = LiveColumns(Path("/run/columns"), writer=writer, minutes=60, threads=2)
    for d in (1, 2):
        for stamp in ("2024-04-26_18_00_00", "2024-04-26_18_15_00", "2024-04-26_19_00_00", "2024-04-26_19_45_00"):
            live.frame_committed(domain=d, valid_time=None, path=frame(d, stamp))
    summary = live.stop()
    assert sorted(written) == sorted(f"wrfout_d0{d}_{s}" for d in (1, 2)
                                     for s in ("2024-04-26_18_00_00", "2024-04-26_19_00_00", "2024-04-26_19_45_00"))
    assert summary["files"] == 6 and summary["bytes"] == 60 and not summary["failed"]


def test_a_failed_frame_is_a_warning_and_the_rest_are_written():
    warned = []

    def writer(path, out, threads=None):
        if "18_00" in Path(path).name:
            raise RuntimeError("column store not written: no PH")
        return [{"path": "x", "bytes": 1}]

    live = LiveColumns(Path("/run/columns"), writer=writer, minutes=0,
                       warn=lambda code, message, **f: warned.append(code))
    live.frame_committed(domain=1, valid_time=None, path=frame(1, "2024-04-26_18_00_00"))
    live.frame_committed(domain=1, valid_time=None, path=frame(1, "2024-04-26_18_10_00"))
    summary = live.stop()
    assert warned == ["columns_not_written"]
    assert summary["files"] == 1 and len(summary["failed"]) == 1


def test_on_by_default_and_switched_off_by_name():
    assert columns_enabled({}, environ={})
    assert not columns_enabled({}, environ={"WOOF_COLUMNS": "off"})
    assert not columns_enabled({"columns": False}, environ={})
    assert on_cadence(frame(1, "2024-04-26_19_00_00"), 60)
    assert not on_cadence(frame(1, "2024-04-26_19_30_00"), 60)
    assert on_cadence(frame(1, "2024-04-26_19_30_00"), 30)


def test_a_forced_restart_rewrites_the_rewound_hours_and_waits_for_the_frame_in_hand(tmp_path, monkeypatch):
    """Breakage: after a restart the store kept the failed attempt's columns (the writer skipped frame paths it had
    already written), and its writer could still be reading a frame the runner moves aside."""
    import threading

    from woof import live_columns, runplan

    written, reading = [], threading.Event()
    release = threading.Event()

    def writer(path, out, threads=None):
        if Path(path).name.endswith("19_00_00") and not release.is_set():
            reading.set()
            release.wait(10)          # the failed attempt's frame in hand when the restart comes
        written.append(Path(path).name)
        return [{"path": str(Path(out) / Path(path).name), "bytes": 1}]

    monkeypatch.setattr(live_columns, "_default_writer", lambda timeout: writer)

    class Events:
        def emit(self, event, **fields):
            pass

    observer = runplan.RunObserver(Events())
    observer.arm_columns(tmp_path / "columns")
    first = observer._columns
    h18, h19, h20 = (tmp_path / f"wrfout_d01_2024-04-26_{h}_00_00" for h in (18, 19, 20))
    observer.output_committed(domain=1, valid_time="2024-04-26T18:00:00", path=h18)
    observer.output_committed(domain=1, valid_time="2024-04-26T19:00:00", path=h19)
    observer.output_committed(domain=1, valid_time="2024-04-26T20:00:00", path=h20)   # queued behind 19
    assert reading.wait(10)
    threading.Timer(0.3, release.set).start()
    observer.restarting("forced")
    # The frame in hand finished before the restart returned; the queued one was dropped.
    assert written == [h18.name, h19.name]
    assert observer._columns is not None and observer._columns is not first
    assert observer._columns.out == tmp_path / "columns"
    # The new attempt commits the same paths again: each one is written again.
    observer.output_committed(domain=1, valid_time="2024-04-26T18:00:00", path=h18)
    observer.output_committed(domain=1, valid_time="2024-04-26T19:00:00", path=h19)
    observer.complete(7200)
    assert written == [h18.name, h19.name, h18.name, h19.name]


def test_a_stuck_writer_never_holds_the_stop(tmp_path):
    import threading
    import time

    warned, hold = [], threading.Event()

    def writer(path, out, threads=None):
        hold.wait(30)
        return []

    live = LiveColumns(tmp_path, writer=writer, minutes=0, warn=lambda code, message, **f: warned.append(code))
    live.frame_committed(domain=1, valid_time=None, path=frame(1, "2024-04-26_18_00_00"))
    live.frame_committed(domain=1, valid_time=None, path=frame(1, "2024-04-26_19_00_00"))
    started = time.monotonic()
    summary = live.stop(timeout=0.5)
    assert time.monotonic() - started < 5 and not summary["stopped"] and warned == ["columns_not_written"]
    hold.set()


def test_a_hung_rw_soundings_is_killed_at_its_timeout(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time

    import pytest

    from woof import soundings

    (tmp_path / "wrfout_d01_2024-04-26_18_00_00").write_bytes(b"")
    real_run = subprocess.run

    def run(argv, **kwargs):
        if "--abi" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=soundings.COLUMNS_ABI, stderr="")
        return real_run([sys.executable, "-c", "import time; time.sleep(120)"], **kwargs)   # rw_soundings hung

    monkeypatch.setattr(soundings, "binary", lambda: Path("rw_soundings"))
    monkeypatch.setattr(soundings.subprocess, "run", run)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not finish in 1 s"):
        soundings.write_columns(tmp_path, tmp_path / "columns", timeout=1)
    assert time.monotonic() - started < 30


def test_frames_between_full_history_leads_are_not_attempted_and_said_once():
    """Breakage (updraft factory, 2026-10-27): a `woof go` HRRR-recipe member with full history every 3 h
    (`[output] full_history_interval_s = 10800`, 15-min 2-D frames between) ran `rw_soundings --columns` on every
    hourly between-frame and warned "column store not written: variable not found in WRF file: P" for each. The
    store now takes the frames that carry the 3-D state, from the run's own [output] selection, and says so once."""
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    from woof.io.history_selection import HistorySelection
    from woof.live_columns import StateFrames

    output = HistorySelection.from_mapping({"full_history_interval_s": 10800.0, "between_preset": "minimal",
                                            "between_history_add": ["REFL_10CM", "UP_HELI_MAX"]})
    start = datetime(2022, 6, 15, 12)
    exp = SimpleNamespace(output=output, start_time=start,
                          domains=(SimpleNamespace(grid_id=1, output=None, start_time=None, quicklook_output=None),))
    full = {start + timedelta(hours=h) for h in range(0, 25, 3)}
    written, warned, reported = [], [], []

    def writer(path, out, threads=None):      # rw_soundings --columns on what the frame really holds
        if datetime.strptime(Path(path).name[11:], "%Y-%m-%d_%H_%M_%S") not in full:
            raise RuntimeError("column store not written: variable not found in WRF file: P")
        written.append(Path(path).name)
        return [{"path": "x", "bytes": 1}]

    live = LiveColumns(Path("/run/columns"), writer=writer, minutes=60, frames=StateFrames.for_experiment(exp),
                       report=reported.append, warn=lambda code, message, **f: warned.append(message))
    for quarter in range(24 * 4 + 1):
        valid = start + timedelta(minutes=15 * quarter)
        live.frame_committed(domain=1, valid_time=valid, path=frame(1, valid.strftime("%Y-%m-%d_%H_%M_%S")))
    summary = live.stop()
    assert warned == [] and not summary["failed"]
    assert sorted(written) == sorted(f"wrfout_d01_{t:%Y-%m-%d_%H_%M_%S}" for t in full)
    assert summary["skipped_without_state"] == 24 * 4 + 1 - len(full)
    notes = [r for r in reported if r.get("event") == "state_frames_only"]
    assert len(notes) == 1 and "every 10800 s" in notes[0]["note"]
    # A selection that keeps the state on every frame decides nothing away, and an offline child's standing
    # selection that sheds P is said the same way.
    assert StateFrames.for_experiment(SimpleNamespace(output=HistorySelection(), start_time=start, domains=exp.domains))(
        1, start + timedelta(hours=1), frame(1, "2022-06-15_13_00_00"))
    assert not StateFrames.for_selection(HistorySelection(preset="minimal"))(3, None, frame(3, "2022-06-15_13_00_00"))


def test_a_plain_go_run_owns_its_store_when_nobody_hosts_it(tmp_path, monkeypatch):
    """THE BREAKAGE: `woof go` runs the prepared runner as a subprocess with no observer, and only run-plan's
    RunObserver armed LiveColumns, so a plain go run wrote no .wcol (speed-output lane, 2026-10-28). The runner
    now owns the store unless a host owns it or WOOF_COLUMNS=off."""
    import woof.live_columns as live_columns
    from woof import prepared_single_domain_forecast as runner

    written = []

    def writer(path, out, threads=None):
        written.append(Path(path).name)
        return [{"path": str(out / Path(path).name), "bytes": 10, "ms": 1.0}]

    monkeypatch.setattr(live_columns, "_default_writer", lambda timeout: writer)
    monkeypatch.delenv("WOOF_COLUMNS", raising=False)
    columns = runner._route_owned_columns(outdir=tmp_path, observer=None)
    assert columns is not None and columns.out == tmp_path / "columns"
    runner._columns_sink(columns)(domain=1, valid_time=None, path=frame(1, "2024-04-26_18_00_00"))
    summary = runner._finish_owned_columns(columns)
    assert written == ["wrfout_d01_2024-04-26_18_00_00"] and summary["files"] == 1

    class Host:  # woof run-plan's RunObserver arms and stops its own store
        def arm_columns(self, *args, **kwargs):
            pass

    assert runner._route_owned_columns(outdir=tmp_path, observer=Host()) is None
    monkeypatch.setenv("WOOF_COLUMNS", "off")
    assert runner._route_owned_columns(outdir=tmp_path, observer=None) is None
