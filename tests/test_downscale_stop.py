"""Stopping a downscaled child is recorded as a stop.

WHAT BREAKAGE THESE PIN (gate law; all three present in 2.7.7).

* A SIGTERM left no record.  SIGTERM, which ``woof downscale`` tells
  its reader to send when Ctrl-C cannot reach it, had no handler on this
  route.  The child died at the default disposition (exit 143) with no
  event, no ``report.json``, no banner and a run manifest carrying only
  its start, so the folder read as a run still going.
* A Stop was published as a failure.  The desktop and terminal Stop
  (SIGINT to the process group) was published as a failure named
  ``KeyboardInterrupt: KeyboardInterrupt``, with ``result`` FAIL and a
  banner saying "Why it stopped: KeyboardInterrupt".
* Staging files were counted as pictures.  The renderer shared the
  child's process group, so the same Stop killed it between drawing its
  pictures under the engine's flat staging names and filing them, and
  the child then published and counted 155 to 171 flat ``rustwx_wrf_*``
  and ``var_wrf_*`` files as its pictures.

Each test starts the real engine door (``woof.offline_child_run.main``)
in a subprocess with a stub stepper in place of the CUDA forecast and a
stand-in renderer that stages and files its pictures the way ``woof
render`` does (``tests/downscale_stop_worker.py``), stops it the way the
desktop, the terminal app and a shell do, and reads what it left.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from woof import first_products, offline_child_run
from woof.runplan import read_events

_posix = pytest.mark.skipif(
    os.name != "posix",
    reason="process-group signals are how the desktop and the terminal "
           "app stop a run on Linux and macOS; Windows ends the tree")

_ROOT = Path(__file__).resolve().parents[1]
_WORKER = Path(__file__).resolve().with_name("downscale_stop_worker.py")
_FIRST = "wrfout_d02_2023-06-21_18_00_00"
_SECOND = "wrfout_d02_2023-06-21_18_15_00"


@pytest.mark.parametrize("outcome", ["stop", "pass", "fail"])
def test_columns_finish_before_the_terminal_event_without_pictures(
        tmp_path, monkeypatch, outcome):
    """Column cleanup must not turn a finished stream back into progress.

    No renderer is involved: even a successful column writer appended its
    summary after the stop, and a no-picture pass abandoned its columns.
    """
    from datetime import datetime

    config = tmp_path / "child.toml"
    config.write_text("# child\n")
    methods = []

    class Columns:
        def finish(self, method):
            methods.append(method)
            return dict(files=1, bytes=12, seconds=0.1, failed=[], out="columns")

        def halt(self):
            return self.finish("halt")

        def stop(self):
            return self.finish("stop")

    def forecast(args, progress):
        progress.start(outdir=tmp_path, child_config=config, ratio=3,
                       start_time=datetime(2023, 6, 21), parent={}, name="Column stop test")
        progress._columns = Columns()
        if outcome == "stop":
            raise KeyboardInterrupt
        if outcome == "fail":
            raise ValueError("forecast failed")
        return {"result": "PASS", "outputs": []}

    monkeypatch.setattr(offline_child_run, "_run", forecast)
    if outcome == "pass":
        offline_child_run.run(argparse.Namespace())
    else:
        with pytest.raises(KeyboardInterrupt if outcome == "stop" else ValueError):
            offline_child_run.run(argparse.Namespace())
    events = read_events(tmp_path / "events.jsonl")
    assert [e["event"] for e in events][-2:] == [
        "columns_written", "completed" if outcome == "pass" else "failed"]
    assert events[-2]["files"] == 1
    assert methods == ["halt" if outcome == "stop" else "stop"]


class _Child:
    """One stub child, started as the desktop's terminal app starts one."""

    def __init__(self, tmp_path: Path, *, products: str = "all",
                 hold: str = "", steps: int | None = None):
        self.outdir = tmp_path / "child"
        self.mark = tmp_path / "marks"
        self.mark.mkdir()
        config = tmp_path / "child.toml"
        config.write_text("# stub child configuration\n", encoding="utf-8")
        self.log = (tmp_path / "child.log").open("w", encoding="utf-8")
        environment = {**os.environ, "PYTHONPATH": str(_ROOT),
                       "STUB_HOLD": hold, "STUB_MARK": str(self.mark),
                       "STUB_STEPS": "" if steps is None else str(steps)}
        # Its own session with SIGINT at its default, as the terminal app
        # starts a job (tools/arwen-tui/src/job.rs), so a Stop can be the
        # process-group SIGINT that app sends.
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(_WORKER), "child", str(self.outdir),
             str(config), products],
            stdout=self.log, stderr=subprocess.STDOUT, env=environment,
            cwd=str(tmp_path), start_new_session=True,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        self.log_path = tmp_path / "child.log"

    def wait_for(self, predicate, what: str, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if self.process.poll() is not None:
                break
            time.sleep(0.05)
        self.finish()
        raise AssertionError(
            f"never saw {what}; the child said:\n{self.output()}")

    def stepping(self) -> None:
        self.wait_for(lambda: "child_step 2" in self.output(),
                      "the stub forecast stepping")

    def staged(self, frame: str) -> None:
        self.wait_for(lambda: (self.mark / f"staged-{frame}").exists(),
                      f"the render of {frame} staging its pictures")

    def output(self) -> str:
        # Read after finish() too, which closes this side of the log: the file itself holds every line.
        if not self.log.closed:
            self.log.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def finish(self, timeout: float = 20.0) -> int:
        try:
            code = self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            raise AssertionError(
                f"the child did not end {timeout:.0f} s after its stop:\n"
                f"{self.output()}")
        finally:
            self.log.close()
        return code

    def events(self) -> list[dict]:
        return read_events(self.outdir / "events.jsonl",
                           allow_partial_tail=True)

    def report(self) -> dict:
        return json.loads((self.outdir / "report.json")
                          .read_text(encoding="utf-8"))

    def manifest(self) -> dict:
        return json.loads((self.outdir / "run-manifest.json")
                          .read_text(encoding="utf-8"))

    def banner(self) -> str:
        return (self.outdir / "png" / first_products.DID_NOT_FINISH_BANNER
                ).read_text(encoding="utf-8")


def _pictures(render_dir: Path) -> list[str]:
    return sorted(path.relative_to(render_dir).as_posix()
                  for path in render_dir.rglob("*.png"))


@_posix
def test_a_sigterm_ends_the_child_with_a_stop_recorded(tmp_path):
    """A SIGTERM left no record.  ``kill -TERM <pid>``: the stream ends on
    the stop, the report says STOPPED, the banner says it was stopped by
    request, and the manifest says when it ended."""

    child = _Child(tmp_path)
    child.stepping()
    os.kill(child.process.pid, signal.SIGTERM)
    assert child.finish() == 143
    last = child.events()[-1]
    assert last["event"] == "failed"
    assert last["interrupted"] is True
    assert last["exit_code"] == 143
    assert last["signal"] == "SIGTERM"
    assert last["message"].startswith("Stopped by request (SIGTERM).")
    report = child.report()
    assert report["result"] == "STOPPED"
    assert report["stop"]["signal"] == "SIGTERM"
    assert report["stop"]["requested"] is True
    assert report["stop"]["stopped_at"]["step"] >= 2
    assert "failure" not in report
    banner = child.banner()
    assert banner.startswith("THIS FORECAST WAS STOPPED BEFORE IT FINISHED")
    assert "Why it stopped: it was stopped by request (SIGTERM)" in banner
    manifest = child.manifest()
    assert manifest["end_state"] == "stopped"
    assert manifest["ended_at_utc"] >= manifest["started_at_utc"]
    # The same report the supervisor's workers print, on this door.
    assert "SIGTERM (signal 15)" in child.output()
    assert "stopping; the run records the stop and ends." in child.output()


@_posix
def test_a_process_group_sigint_is_a_stop_and_no_record_names_the_exception(
        tmp_path):
    """A Stop was published as a failure.  The desktop's and the terminal
    app's Stop: SIGINT to the process group.  It exits 130, is recorded as
    a stop, and no record a run view reads says ``KeyboardInterrupt``."""

    child = _Child(tmp_path)
    child.stepping()
    os.killpg(child.process.pid, signal.SIGINT)
    assert child.finish() == 130
    last = child.events()[-1]
    assert last["event"] == "failed"
    assert last["interrupted"] is True
    assert last["exit_code"] == 130
    assert last["signal"] == "SIGINT"
    assert child.report()["result"] == "STOPPED"
    assert child.manifest()["end_state"] == "stopped"
    records = [child.outdir / "events.jsonl", child.outdir / "report.json",
               child.outdir / "run-manifest.json",
               child.outdir / "png" / first_products.DID_NOT_FINISH_BANNER,
               child.outdir / "png" / "render-summary.json"]
    for record in records:
        assert record.is_file(), record
        assert "KeyboardInterrupt" not in record.read_text(encoding="utf-8"), \
            record.name


@_posix
def test_a_stop_while_the_first_frame_is_drawn_leaves_no_staging_files(
        tmp_path):
    """Staging files were counted as pictures.  The Stop lands after the
    render has drawn the analysis frame's pictures under their flat
    staging names and before it filed them.  None of them reaches ``png/``, nothing is counted that no
    render receipt names, and the render is not left running."""

    child = _Child(tmp_path, hold=_FIRST)
    child.staged(_FIRST)
    os.killpg(child.process.pid, signal.SIGINT)
    assert child.finish() == 130
    render_dir = child.outdir / "png"
    pictures = _pictures(render_dir)
    assert not [name for name in pictures
                if "rustwx_wrf_" in name or "var_wrf_" in name], pictures
    receipted = first_products.receipted_pictures(render_dir)
    assert set(pictures) <= receipted
    products = child.report()["products"]
    assert products["pictures_on_disk"] == len(receipted) == 0
    summary = json.loads((render_dir / "render-summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["pictures_on_disk"] == 0
    assert "No pictures are in this folder" in child.banner()
    # Nothing the render worked in is left beside the pictures.
    assert not (render_dir / ".first-products-scratch").exists()
    events = [event["event"] for event in child.events()]
    assert "first_products_ready" not in events


@_posix
def test_a_stop_while_a_later_frame_is_drawn_keeps_the_filed_pictures(
        tmp_path):
    """The analysis frame was drawn and filed with its receipt; the
    second frame's render is stopped between staging and filing.  The
    organised tree of the first stays, the second's staging goes, and
    the count is the receipted pictures."""

    child = _Child(tmp_path, hold=_SECOND)
    child.staged(_SECOND)
    os.killpg(child.process.pid, signal.SIGINT)
    assert child.finish() == 130
    render_dir = child.outdir / "png"
    pictures = _pictures(render_dir)
    assert pictures and all(name.startswith("d02-250m/")
                            for name in pictures), pictures
    assert all("f000_" in name for name in pictures), pictures
    receipted = first_products.receipted_pictures(render_dir)
    assert set(pictures) == receipted
    assert child.report()["products"]["pictures_on_disk"] == len(pictures)
    assert f"{len(pictures)} pictures are in this folder" in child.banner()


def _exited(pid: int, timeout: float = 5.0) -> bool:
    """Whether process ``pid`` is gone within ``timeout`` seconds."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


@_posix
@pytest.mark.parametrize("stop", ["SIGINT", "SIGTERM"])
def test_a_stop_while_finalize_waits_for_the_last_frame_ends_its_render(
        tmp_path, stop):
    """The forecast finished and its finalize stage is waiting for the
    render of the last frame, drawn as it landed, when the stop lands.

    On CPython 3.11 and 3.12 the exception the stop raises inside that
    wait (``KeyboardInterrupt``, or the ``ChildStopped`` a SIGTERM
    becomes) marked the render's worker thread as ended while it was
    still waiting on its render, so the stop never ended the render.
    Measured on a real 250 m child: ``woof render`` and ``rw_wrfbatch``
    drew on after the child had exited and left their staging pictures
    in the folder.  The render is ended, and the pictures are the
    receipted ones.
    """

    # The stub commits its second frame at step 5; it finishes at step 7.
    child = _Child(tmp_path, hold=_SECOND, steps=7)
    child.staged(_SECOND)
    child.wait_for(lambda: any(
        event.get("event") == "stage_started"
        and event.get("stage") == "finalize" for event in child.events()),
        "the finalize stage waiting for the last frame's render")
    # Into the wait itself: the held render takes a minute to file.
    time.sleep(1.0)
    render = int((child.mark / f"pid-{_SECOND}").read_text(encoding="utf-8"))
    if stop == "SIGINT":
        os.killpg(child.process.pid, signal.SIGINT)
    else:
        os.kill(child.process.pid, signal.SIGTERM)
    assert child.finish() == (130 if stop == "SIGINT" else 143),         child.output()
    ended = _exited(render)
    if not ended:
        # Not left drawing past the test either (a group of its own).
        try:
            os.killpg(render, signal.SIGKILL)
        except OSError:
            pass
    assert ended, ("the render of the last frame was still drawing after "
                   "the stopped child exited")
    report = child.report()
    assert report["result"] == "PASS"
    assert (report["stop"]["stage"], report["stop"]["signal"]) == (
        "finalize", stop)
    render_dir = child.outdir / "png"
    pictures = _pictures(render_dir)
    assert pictures and all(name.startswith("d02-250m/")
                            for name in pictures), pictures
    assert set(pictures) == first_products.receipted_pictures(render_dir)
    assert report["products"]["pictures_on_disk"] == len(pictures)
    last = child.events()[-1]
    assert (last["event"], last["stage"], last["interrupted"]) == (
        "failed", "finalize", True)


def _drawn_early(monkeypatch):
    """The early render's subprocess, replaced by one that files a picture."""

    def runner(command, **_options):
        out = Path(command[command.index("--out") + 1])
        folder = out / "d02-250m" / "composite_reflectivity" / "20230621"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "f000_composite_reflectivity.png").write_bytes(b"PNG")
        return subprocess.CompletedProcess(list(command), 0, "", "")

    monkeypatch.setattr(first_products, "_run_render", runner)


def test_a_stop_during_the_finalize_render_keeps_the_forecast_verdict(
        tmp_path, monkeypatch):
    """Stopped while the finished forecast's pictures were being drawn.

    The forecast passed and its report says so; the stop is recorded
    beside that verdict, the render's flat staging files are removed,
    the pictures filed with a receipt stay, the stream ends on the stop
    at ``finalize``, and the door leaves as the Ctrl-C it was.
    """

    from datetime import datetime

    import woof.go_cli as go_cli

    _drawn_early(monkeypatch)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)

    def run_stage(label, command, **_):
        # The finalize render drew flat under the engine's staging names
        # and was stopped before it filed them or wrote its receipt.
        out = Path(command[command.index("--out") + 1])
        for product in ("composite_reflectivity", "var_wrf_isltyp_0123"):
            (out / f"rustwx_wrf_20230621_18z_f015_d02-250m_{product}.png"
             ).write_bytes(b"PNG")
        raise go_cli.GoInterrupted(label, None)

    monkeypatch.setattr(go_cli, "_run_stage", run_stage)
    config = tmp_path / "child.toml"
    config.write_text("# child\n", encoding="utf-8")
    outdir = tmp_path / "child-run"

    def fake_run(args, progress):
        outdir.mkdir(parents=True, exist_ok=True)
        progress.start(outdir=outdir, child_config=config, ratio=12,
                       start_time=datetime(2023, 6, 21, 18),
                       parent={"run_dir": str(tmp_path), "frames": 2},
                       name="Downscale of parent")
        progress.arm_render(outdir=outdir, render_products="all")
        progress.emit("stage_started", stage="forecast", phase="integrate")
        frames = []
        for minutes in (0, 15):
            frame = outdir / f"wrfout_d02_2023-06-21_18_{minutes:02d}_00"
            frame.write_bytes(b"CDF frame")
            frames.append(frame)
        # The analysis is drawn early; the second frame is left for the
        # finalize render, which is where this stop lands.
        progress.output_committed(
            domain=2, valid_time="2023-06-21T18:00:00Z",
            path=str(frames[0]), bytes=frames[0].stat().st_size)
        progress.wait_early_render()
        report = {"result": "PASS", "outputs": [str(f) for f in frames]}
        offline_child_run._publish_report(report, outdir)
        return report

    monkeypatch.setattr(offline_child_run, "_run", fake_run)
    with pytest.raises(KeyboardInterrupt):
        offline_child_run.run(argparse.Namespace(outdir=outdir,
                                                 render_products="all"))
    render_dir = outdir / "png"
    pictures = sorted(path.relative_to(render_dir).as_posix()
                      for path in render_dir.rglob("*.png"))
    assert pictures and not [name for name in pictures
                             if "rustwx_wrf_" in name], pictures
    assert set(pictures) == first_products.receipted_pictures(render_dir)
    report = json.loads((outdir / "report.json").read_text(encoding="utf-8"))
    assert report["result"] == "PASS"
    assert report["stop"]["stage"] == "finalize"
    assert report["stop"]["signal"] == "SIGINT"
    assert report["products"]["status"] == "STOPPED"
    assert report["products"]["pictures_on_disk"] == len(pictures)
    assert report["products"]["discarded_unfinished"] == 2
    last = read_events(outdir / "events.jsonl")[-1]
    assert (last["event"], last["stage"], last["interrupted"]) == (
        "failed", "finalize", True)
    assert "KeyboardInterrupt" not in last["message"]
    manifest = json.loads((outdir / "run-manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["end_state"] == "stopped"


def test_a_render_stage_that_answered_the_ctrl_c_is_a_stop(monkeypatch):
    """The render shares the child's process group on this stage and can
    exit 130 on the group's Ctrl-C before the child sees its own; that is
    the user's stop, not a render that failed."""

    import woof.go_cli as go_cli

    def stopped(*_args, **_kwargs):
        raise go_cli.GoStageFailed(130, "")

    monkeypatch.setattr("woof.runplan._finish_render", stopped)

    class _Progress:
        render_plan = {"render": "png", "render_products": "all"}

    with pytest.raises(go_cli.GoInterrupted):
        offline_child_run._finish_child_render(_Progress(), report={})
    assert offline_child_run._stopped_by_user(go_cli.GoInterrupted("r", 1))
    assert offline_child_run._stopped_by_user(offline_child_run.ChildStopped(15))
    assert not offline_child_run._stopped_by_user(SystemExit(0))
