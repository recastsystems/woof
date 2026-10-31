"""WRF-grade per-step progress: the log, its formatters, its markers.

The defect this file pins is a usability one, reported by a real user
driving this package from his own script: the simulation printed a
heartbeat every twenty seconds naming the most recently written FRAME,
never the model step, so a caller could not tell where the run was and
could not tell when a frame was safe to open.  WRF prints one line per
model time step and an explicit line for every output and restart it
writes, and that is the bar.

Everything here is CPU-only and touches no card.  The seam under test is
:mod:`woof.progress_log` itself plus the two places the runners call
it, which is deliberate: a formatter nobody reaches is not a feature.
"""

from __future__ import annotations

import ast
import json
import re
import threading
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from woof.progress_log import (FRAME_MARKER_DIRNAME, FRAME_MARKER_SCHEMA,
                                NEST_EVENTS, STEP_LOG_EVENTS,
                                STEP_LOG_FILENAME, STEP_LOG_SCHEMA,
                                STEP_LOG_SCHEMAS, StepLog,
                                format_domain_end_line,
                                format_domain_start_line,
                                format_output_line, format_restart_line,
                                format_run_end_line, format_step_line,
                                format_model_time, model_step_log,
                                open_step_log, publish_step_log,
                                read_step_log)

REPO = Path(__file__).resolve().parents[1]

START = datetime(2026, 8, 15, 0, 0, 0)


#: A regex written against WRF's OWN stdout, not against ours.  It is the
#: whole point of the text format: a script that already reads
#: ``rsl.out.0000`` keeps working when it is pointed at this runner.
WRF_MAIN = re.compile(
    r"Timing for main: time (\S+) on domain\s+(\d+):\s+([0-9.]+) "
    r"elapsed seconds")
WRF_WRITE = re.compile(
    r"Timing for Writing (\S+) for domain\s+(\d+):\s+([0-9.]+) "
    r"elapsed seconds")


def make_log(tmp_path, **kwargs):
    text = StringIO()
    log = StepLog(start_time=START, run_seconds=600.0, text_stream=text,
                  jsonl_path=tmp_path / STEP_LOG_FILENAME,
                  frame_marker_dir=tmp_path / FRAME_MARKER_DIRNAME,
                  **kwargs)
    return log, text


# ---------------------------------------------------------------------------
# The formatters, held against WRF's own grammar
# ---------------------------------------------------------------------------


def test_step_line_is_parseable_by_a_wrf_shaped_regex():
    line = format_step_line(domain=2, step=41,
                            valid_time=START + timedelta(seconds=12),
                            wall_seconds=0.0638249)
    match = WRF_MAIN.match(line)
    assert match, line
    assert match.group(1) == "2026-08-15_00:00:12"
    assert int(match.group(2)) == 2
    assert float(match.group(3)) == pytest.approx(0.06382, abs=1e-5)
    assert line.endswith("step 41")


def test_domain_and_run_lines_are_tagged_and_greppable():
    assert format_domain_start_line(
        domain=3, valid_time=START).startswith("d03 2026-08-15_00:00:00")
    assert "domain start" in format_domain_start_line(domain=3,
                                                      valid_time=START)
    end = format_domain_end_line(domain=3, valid_time=START, steps=17)
    assert "domain end" in end and "17" in end
    assert "SUCCESS COMPLETE" in format_run_end_line(
        status="SUCCESS", steps=17, wall_seconds=1.0)
    assert "SUCCESS" not in format_run_end_line(
        status="FAIL", steps=17, wall_seconds=1.0)


def test_model_time_refuses_a_sub_second_valid_time():
    """A frame name and a log line must agree; both are whole seconds."""

    with pytest.raises(ValueError):
        format_model_time(START + timedelta(microseconds=5))


# ---------------------------------------------------------------------------
# One line per model time step
# ---------------------------------------------------------------------------


def test_nests_get_their_own_lines_with_their_own_step_index(tmp_path):
    log, text = make_log(tmp_path)
    log.domain_step(grid_id=1, step_count=1, model_seconds=60.0,
                    step_wall_seconds=0.2)
    for step in (1, 2, 3, 4):
        log.domain_step(grid_id=2, step_count=step,
                        model_seconds=float(step * 15),
                        step_wall_seconds=0.05)
    log.close(status="SUCCESS")

    steps = [record for record in read_step_log(tmp_path / STEP_LOG_FILENAME)
             if record["event"] == "step"]
    assert [(r["domain"], r["step"]) for r in steps] == [
        (1, 1), (2, 1), (2, 2), (2, 3), (2, 4)]
    assert steps[-1]["valid_time"] == "2026-08-15_00:01:00"


def test_step_records_carry_the_four_required_fields(tmp_path):
    log, _ = make_log(tmp_path)
    log.domain_step(grid_id=4, step_count=9, model_seconds=15.0,
                    step_wall_seconds=0.125)
    log.close(status="SUCCESS")
    record = next(r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
                  if r["event"] == "step")
    assert record["domain"] == 4
    assert record["step"] == 9
    assert record["valid_time"] == "2026-08-15_00:00:15"
    assert record["step_wall_seconds"] == pytest.approx(0.125)
    assert record["model_seconds"] == pytest.approx(15.0)
    # A progress bar needs one division, not the whole run's arithmetic.
    assert record["fraction"] == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# Explicit events, not "the frame count changed"
# ---------------------------------------------------------------------------


def test_output_written_is_an_event_not_an_inference(tmp_path):
    log, text = make_log(tmp_path)
    frame = tmp_path / "wrfout_d01_2026-08-15_00_00_00"
    frame.write_bytes(b"x" * 128)
    log.output_committed(domain=1, valid_time=START, path=frame)
    log.close(status="SUCCESS")

    record = next(r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
                  if r["event"] == "output_written")
    assert record["domain"] == 1
    assert record["size_bytes"] == 128
    assert Path(record["path"]) == frame.resolve()
    match = WRF_WRITE.search(text.getvalue())
    assert match and match.group(1) == frame.name
    assert int(match.group(2)) == 1


def test_restart_written_is_an_event(tmp_path):
    log, text = make_log(tmp_path)
    restart = tmp_path / "restart_d01"
    restart.write_bytes(b"y" * 9)
    log.restart_written(domain=1, valid_time=START, path=restart)
    log.close(status="SUCCESS")
    record = next(r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
                  if r["event"] == "restart_written")
    assert record["size_bytes"] == 9
    assert "restart" in text.getvalue()
    assert WRF_WRITE.search(text.getvalue())


def test_run_end_states_the_outcome(tmp_path):
    log, text = make_log(tmp_path)
    log.domain_step(grid_id=1, step_count=1, model_seconds=1.0,
                    step_wall_seconds=0.1)
    log.close(status="FAIL", error="ran out of memory")
    record = next(r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
                  if r["event"] == "run_end")
    assert record["status"] == "FAIL"
    assert record["error"] == "ran out of memory"
    assert "SUCCESS" not in text.getvalue().splitlines()[-1]


def test_every_tag_emitted_is_declared(tmp_path):
    log, _ = make_log(tmp_path)
    frame = tmp_path / "wrfout_d01_2026-08-15_00_00_00"
    frame.write_bytes(b"x")
    log.phase("preflight_verify", 0.5)
    log.domain_step(grid_id=1, step_count=1, model_seconds=1.0,
                    step_wall_seconds=0.1)
    log.output_committed(domain=1, valid_time=START, path=frame)
    log.restart_written(domain=1, valid_time=START, path=frame)
    log.nest_spawned(domain=2, model_seconds=1.0, episode=1, parent=1)
    log.nest_moved(domain=2, model_seconds=1.0,
                   placement_from={"i_parent_start": 1, "j_parent_start": 1},
                   placement_to={"i_parent_start": 2, "j_parent_start": 1})
    log.containment_moved(
        domain=1, model_seconds=1.0, mover=2,
        placement_from={"i_parent_start": 1, "j_parent_start": 1},
        placement_to={"i_parent_start": 2, "j_parent_start": 1})
    log.track_fix(domain=2, model_seconds=1.0, lat=35.0, lon=-97.0)
    log.nest_retired(domain=2, model_seconds=1.0, episode=1)
    log.nest_rearmed(domain=2, model_seconds=1.0, episode=2)
    log.close(status="SUCCESS")
    seen = {r["event"] for r in read_step_log(tmp_path / STEP_LOG_FILENAME)}
    assert seen <= set(STEP_LOG_EVENTS)
    assert seen == set(STEP_LOG_EVENTS)


# ---------------------------------------------------------------------------
# The completion signal a script can trust
# ---------------------------------------------------------------------------


def test_frame_marker_lands_only_after_the_frame_is_readable(tmp_path):
    log, _ = make_log(tmp_path)
    frame = tmp_path / "wrfout_d01_2026-08-15_00_00_00"
    frame.write_bytes(b"z" * 64)
    log.output_committed(domain=1, valid_time=START, path=frame)
    log.close(status="SUCCESS")

    markers = sorted((tmp_path / FRAME_MARKER_DIRNAME).glob("*.json"))
    assert len(markers) == 1
    payload = json.loads(markers[0].read_text(encoding="utf-8"))
    assert payload["schema"] == FRAME_MARKER_SCHEMA
    assert Path(payload["path"]) == frame.resolve()
    assert payload["size_bytes"] == 64
    assert payload["domain"] == 1
    assert payload["valid_time"] == "2026-08-15_00:00:00"
    assert not list((tmp_path / FRAME_MARKER_DIRNAME).glob("*.tmp.*"))


def test_frame_marker_does_not_pollute_a_wrfout_glob(tmp_path):
    """The reason the marker lives in its own directory.

    A marker named ``wrfout_d01_....ready`` beside the frames would be
    matched by the same ``wrfout_d01_*`` glob every consumer of this
    tree already runs, and would be read as a frame.
    """

    log, _ = make_log(tmp_path)
    frame = tmp_path / "wrfout_d01_2026-08-15_00_00_00"
    frame.write_bytes(b"z")
    log.output_committed(domain=1, valid_time=START, path=frame)
    log.close(status="SUCCESS")
    assert sorted(p.name for p in tmp_path.glob("wrfout_d01_*")) == [
        "wrfout_d01_2026-08-15_00_00_00"]


def test_a_marker_for_a_missing_frame_is_refused_not_faked(tmp_path):
    """A marker is a POSITIVE signal; it may never outrun the data."""

    log, text = make_log(tmp_path)
    log.output_committed(domain=1, valid_time=START,
                         path=tmp_path / "never_written")
    log.close(status="SUCCESS")
    assert not (tmp_path / FRAME_MARKER_DIRNAME).exists() or not list(
        (tmp_path / FRAME_MARKER_DIRNAME).glob("*.json"))
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    written = [r for r in records if r["event"] == "output_written"]
    assert written and written[0]["marker"] is None


def test_the_marker_names_a_frame_the_real_writer_really_published(tmp_path):
    """Against the ARTIFACT: the shipped writer, not a stand-in file.

    Everything above writes bytes and calls them a frame.  This drives
    :class:`woof.io.wrfout.WrfoutWriter` -- the class that actually
    publishes history -- through its real close path (fsync,
    self-validation, atomic rename), raises the landing hook exactly
    where the async writer raises it, and then proves the two halves of
    the guarantee: the marker's size matches the published file, and the
    file the marker names opens as a complete history frame.

    It also pins what was ALREADY true, so the documentation cannot
    overclaim: while the writer is open the final name does not exist
    and the in-flight temporary is hidden behind a leading dot, so a
    ``wrfout_d01_*`` glob has never been able to see a partial frame.
    What the marker adds is the ANSWER TO "WHEN" -- one small file, and
    one event, instead of every consumer polling the frames themselves.
    """

    import netCDF4

    from woof.io.wrfout import WrfoutWriter

    frames = tmp_path / "wrfout"
    frames.mkdir()
    log, _ = make_log(tmp_path)
    path = frames / "wrfout_d01_2026-07-01_18_00_00"
    with WrfoutWriter(path, nx=3, ny=2, nz=2, dx=1000.0, dy=1000.0) as writer:
        writer.write_frame("2026-07-01_18:00:00",
                           {"T": np.zeros((2, 2, 3), np.float32)})
        # Mid-write: no final name, no marker, and the temporary is
        # hidden -- the property the docs are allowed to claim.
        assert not path.exists()
        assert [p.name for p in frames.glob("wrfout_d01_*")] == []
        assert [p.name for p in frames.iterdir()] != [], (
            "the writer wrote nothing at all; this test would then be "
            "asserting the absence of a file that was never started")
        assert not list((tmp_path / FRAME_MARKER_DIRNAME).glob("*.json"))

    # Where AsyncDomainWrfoutWriter raises its landing observer.
    log.output_committed(domain=1, valid_time=datetime(2026, 7, 1, 18),
                         path=path)
    log.close(status="SUCCESS")

    markers = list((tmp_path / FRAME_MARKER_DIRNAME).glob("*.json"))
    assert len(markers) == 1
    payload = json.loads(markers[0].read_text(encoding="utf-8"))
    assert Path(payload["path"]) == path.resolve()
    assert payload["size_bytes"] == path.stat().st_size
    # The guarantee, exercised rather than asserted: the file the marker
    # names is a readable, complete history frame.
    with netCDF4.Dataset(path) as ds:
        assert "T" in ds.variables
        assert ds.variables["Times"].shape[0] == 1


# ---------------------------------------------------------------------------
# The two streams must agree
# ---------------------------------------------------------------------------


def test_a_reordered_stream_is_refused_rather_than_read(tmp_path):
    path = tmp_path / STEP_LOG_FILENAME
    path.write_text(
        json.dumps({"schema": STEP_LOG_SCHEMA, "sequence": 1,
                    "event": "run_start"}) + "\n"
        + json.dumps({"schema": STEP_LOG_SCHEMA, "sequence": 3,
                      "event": "run_end"}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        read_step_log(path)


# ---------------------------------------------------------------------------
# Cadence, threads, and cost
# ---------------------------------------------------------------------------


def test_every_thins_steps_and_never_thins_events(tmp_path):
    log, _ = make_log(tmp_path, every=10)
    frame = tmp_path / "wrfout_d01_2026-08-15_00_00_00"
    frame.write_bytes(b"z")
    for step in range(1, 26):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=float(step), step_wall_seconds=0.01)
    log.output_committed(domain=1, valid_time=START, path=frame)
    log.close(status="SUCCESS")
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    steps = [r["step"] for r in records if r["event"] == "step"]
    # first step always, then every tenth, then the last one seen
    assert steps == [1, 10, 20, 25]
    assert len([r for r in records if r["event"] == "output_written"]) == 1


def test_concurrent_emits_never_tear_a_line(tmp_path):
    log, _ = make_log(tmp_path)
    frame = tmp_path / "wrfout_d01_2026-08-15_00_00_00"
    frame.write_bytes(b"z")

    def writer():
        for _ in range(200):
            log.output_committed(domain=1, valid_time=START, path=frame)

    thread = threading.Thread(target=writer)
    thread.start()
    for step in range(1, 201):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=float(step), step_wall_seconds=0.01)
    thread.join()
    log.close(status="SUCCESS")
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert len(records) == 1 + 1 + 200 + 200 + 1 + 1


def test_a_step_costs_far_less_than_a_step(tmp_path):
    """The print may not become the run.

    A GPU model step on the shapes this package runs is milliseconds at
    the very fastest.  The bound here is deliberately loose -- it is a
    catastrophic-regression tripwire (a per-step fsync, a per-step
    stat, a re-open) and not a benchmark.
    """

    log, _ = make_log(tmp_path)
    count = 2000
    started = time.perf_counter()
    for step in range(1, count + 1):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=float(step), step_wall_seconds=0.01)
    per_event = (time.perf_counter() - started) / count
    log.close(status="SUCCESS")
    assert per_event < 500e-6, f"{per_event * 1e6:.1f} us per step event"


def test_the_log_never_synchronises_a_gpu():
    """No timing print may add a device synchronisation.

    Source-level, because the defect it guards is invisible at runtime
    on a CPU-only test: a ``cp.cuda.Stream.null.synchronize()`` added to
    get a "true" per-step time would serialise the whole pipeline and
    still pass every assertion above.
    """

    source = (REPO / "woof" / "progress_log.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "synchronize", "progress_log synchronises"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names]
                     + [getattr(node, "module", "") or ""])
            for name in names:
                assert not name.startswith("cupy"), (
                    "progress_log imports cupy; the per-step wall time is "
                    "a perf_counter pair around the step the executor "
                    "already takes, never a device query")


def test_a_broken_sink_never_takes_the_run_down(tmp_path):
    class Exploding(StringIO):
        def write(self, _data):  # noqa: D102
            raise OSError("the terminal went away")

    log = StepLog(start_time=START, run_seconds=60.0,
                  text_stream=Exploding(),
                  jsonl_path=tmp_path / STEP_LOG_FILENAME,
                  frame_marker_dir=tmp_path / FRAME_MARKER_DIRNAME)
    log.domain_step(grid_id=1, step_count=1, model_seconds=1.0,
                    step_wall_seconds=0.01)
    log.close(status="SUCCESS")
    # The machine stream survived the human one.
    assert read_step_log(tmp_path / STEP_LOG_FILENAME)


def test_a_process_that_dies_before_closing_still_terminates_the_stream(
        tmp_path):
    """`run_start` then silence is indistinguishable from "still going".

    Driven as a real subprocess, because the guarantee IS an interpreter
    exit path: a forecast that dies in preflight never reaches the
    integration loop where the ordinary close lives.
    """

    import subprocess
    import sys
    import textwrap

    script = tmp_path / "die.py"
    script.write_text(textwrap.dedent(f"""
        from datetime import datetime
        from woof.progress_log import StepLog
        log = StepLog(start_time=datetime(2026, 8, 15), run_seconds=60.0,
                      text_stream=None,
                      jsonl_path=r"{tmp_path / STEP_LOG_FILENAME}",
                      frame_marker_dir=None)
        log.domain_step(grid_id=1, step_count=1, model_seconds=1.0,
                        step_wall_seconds=0.01)
        raise SystemExit(2)
    """), encoding="utf-8")
    import os

    # THIS tree's package, not whichever one is installed: a worktree
    # runs against its own checkout under pytest and a bare subprocess
    # would silently pick up the editable install instead.
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    done = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True,
                          cwd=str(REPO), env=env)
    assert done.returncode == 2, done.stderr
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert records[-1]["event"] == "run_end"
    assert records[-1]["status"] == "INCOMPLETE"


def test_closing_twice_says_nothing_twice(tmp_path):
    log, _ = make_log(tmp_path)
    log.close(status="SUCCESS")
    first = read_step_log(tmp_path / STEP_LOG_FILENAME)
    log.close(status="FAIL")
    assert read_step_log(tmp_path / STEP_LOG_FILENAME) == first


# ---------------------------------------------------------------------------
# Reachability: the front doors
# ---------------------------------------------------------------------------


def test_open_step_log_defaults_to_on_beside_the_outputs(tmp_path):
    log = open_step_log(outdir=tmp_path, start_time=START,
                        run_seconds=60.0, text_stream=StringIO())
    try:
        log.domain_step(grid_id=1, step_count=1, model_seconds=1.0,
                        step_wall_seconds=0.01)
    finally:
        log.close(status="SUCCESS")
    assert (tmp_path / STEP_LOG_FILENAME).is_file()
    assert (tmp_path / FRAME_MARKER_DIRNAME).is_dir()


def test_open_step_log_off_is_inert_but_still_an_object(tmp_path):
    log = open_step_log(outdir=tmp_path, start_time=START, run_seconds=60.0,
                        text_stream=StringIO(), progress_format="off",
                        frame_markers=False)
    log.domain_step(grid_id=1, step_count=1, model_seconds=1.0,
                    step_wall_seconds=0.01)
    log.close(status="SUCCESS")
    assert not (tmp_path / STEP_LOG_FILENAME).exists()
    assert not (tmp_path / FRAME_MARKER_DIRNAME).exists()


@pytest.mark.parametrize("module_name", [
    "woof.prepared_single_domain_forecast",
    "woof.prepared_domain_tree_forecast",
])
def test_both_simulation_doors_expose_the_progress_flags(module_name):
    """Engine-proven is not shipped: the flags must exist on the door."""

    import importlib

    module = importlib.import_module(module_name)
    parser = module.build_parser()
    flags = {action.option_strings[0] for action in parser._actions
             if action.option_strings}
    assert {"--progress-format", "--progress-every", "--progress-output",
            "--frame-markers"} <= flags


@pytest.mark.parametrize("module_name", [
    "woof.prepared_single_domain_forecast",
    "woof.prepared_domain_tree_forecast",
])
def test_both_runners_actually_wire_the_step_observer(module_name):
    """A formatter nobody calls is not a feature.

    Source-level because the call site is inside a GPU forecast this
    test cannot run: what is checked is that the runner hands the log's
    per-step hook to ``execute_experiment``, which is the ONLY way a
    step line can ever be produced.
    """

    import importlib

    module = importlib.import_module(module_name)
    source = Path(module.__file__).read_text(encoding="utf-8")
    # The log is built from the flags ...
    assert "ProgressOptions" in source, module_name
    # ... its per-step hook reaches the executor, which is the only
    # place a `Timing for main:` line can come from ...
    assert "step_observer=" in source, module_name
    # ... and its frame hook reaches the writer, which is the only
    # place a frame-ready marker may be published from.
    assert "step_log.output_committed" in source, module_name


def test_the_tree_door_publishes_its_log_where_the_runners_look():
    """The nest tags reach the stream through the MODEL, so the tree
    route -- the one door that runs relocation -- has to put its log
    there.  Source-level for the same reason as the cell above: the call
    site is inside a GPU forecast.  Without this line every relocation
    on the shipped route emits into the inert log and the live map draws
    nothing, with no error anywhere to say so.
    """

    import importlib

    module = importlib.import_module("woof.prepared_domain_tree_forecast")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "publish_step_log(model, step_log)" in source


@pytest.mark.parametrize("module_name", [
    "woof.prepared_single_domain_forecast",
    "woof.prepared_domain_tree_forecast",
])
@pytest.mark.parametrize("bad", ["0", "-3", "half"])
def test_a_nonsense_cadence_is_refused_not_quietly_clamped(
        module_name, bad, capsys):
    """Both doors, and the message must name the flag.

    NOT written as a bare ``parse_args`` + ``SystemExit``: every flag on
    these parsers is missing in such a call, so argparse would exit 2
    for a PERFECTLY VALID cadence too and the test would pass without
    testing anything.  The message is what distinguishes them.
    """

    import importlib

    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--progress-every", bad])
    assert "argument --progress-every" in capsys.readouterr().err, bad


@pytest.mark.parametrize("good", ["1", "60"])
def test_the_control_a_valid_cadence_is_not_the_thing_refused(good, capsys):
    """The negative control for the test above."""

    from woof.prepared_single_domain_forecast import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--progress-every", good])
    assert "argument --progress-every" not in capsys.readouterr().err


def test_a_record_pipe_and_stdout_sentences_are_refused_together():
    """Documented refusal: two grammars on one channel is not a mode."""

    with pytest.raises(ValueError, match="one channel"):
        open_step_log(outdir=Path("."), start_time=START, run_seconds=60.0,
                      progress_format="text", progress_output="-")
    # ... and the combination the page recommends is accepted.
    log = open_step_log(outdir=Path("."), start_time=START,
                        run_seconds=60.0, progress_format="jsonl",
                        progress_output="-", frame_markers=False)
    log.close(status="SUCCESS")


def test_the_pages_own_sample_lines_parse_as_the_code_writes_them():
    """The example in the docs must be output this code could produce.

    A hand-typed sample drifts silently -- a wrong step index in a
    fenced block is invisible to every other check and is exactly the
    thing a reader copies into a regex.  So the page's `Timing for
    main:` lines are held against the WRF-shaped parser AND against
    their own arithmetic: per domain, the step index must advance by
    one and the valid time by that domain's timestep.
    """

    page = (REPO / "docs" / "public" / "PROGRESS.md").read_text(
        encoding="utf-8")
    start = re.search(r"STARTING SIMULATION at (\S+) ", page)
    assert start, "the page shows no run_start line to anchor the sample"
    origin = datetime.strptime(start.group(1), "%Y-%m-%d_%H:%M:%S")

    seen: dict[int, list[tuple[int, datetime]]] = {}
    for line in page.splitlines():
        match = WRF_MAIN.match(line.strip())
        if not match:
            continue
        tail = line.strip().split("elapsed seconds", 1)[1].strip()
        assert tail.startswith("step "), line
        step = int(tail.rsplit("step ", 1)[1])
        stamp = datetime.strptime(match.group(1), "%Y-%m-%d_%H:%M:%S")
        seen.setdefault(int(match.group(2)), []).append((step, stamp))

    assert len(seen) >= 2, "the page shows no nested example at all"
    timesteps = {}
    for domain, rows in seen.items():
        indices = [row[0] for row in rows]
        assert indices == sorted(indices), f"d{domain:02d}: {indices}"
        # Anchored on the run's own start, so a single wrong index is
        # caught even where the page shows only two lines for a domain:
        # step N must land at start + N*dt for ONE dt.
        deltas = {(stamp - origin) / step for step, stamp in rows}
        assert len(deltas) == 1, (
            f"d{domain:02d} would need {sorted(deltas)} as its timestep; "
            "a domain has one, so an index or a valid time is wrong")
        timesteps[domain] = deltas.pop()
    # A nest's timestep divides its parent's a whole number of times.
    root = timesteps[min(timesteps)]
    for domain, step in timesteps.items():
        ratio = root / step
        assert ratio == int(ratio) and ratio >= 1, (
            f"d{domain:02d} at {step} against a root of {root} is not a "
            "whole-number time-step ratio")


def test_the_public_page_names_what_the_code_actually_publishes():
    """An enumeration binding: the page's nouns, from the code.

    A user-facing page that names a schema, a filename or an event tag
    is making a promise a consumer will code against, so every one of
    them is read out of the module rather than retyped.
    """

    page = (REPO / "docs" / "public" / "PROGRESS.md").read_text(
        encoding="utf-8")
    for needle in (STEP_LOG_SCHEMA, FRAME_MARKER_SCHEMA, STEP_LOG_FILENAME,
                   FRAME_MARKER_DIRNAME, *STEP_LOG_EVENTS):
        assert needle in page, f"PROGRESS.md does not name {needle!r}"
    # ... and every flag it tabulates is a flag the doors define.
    from woof.prepared_single_domain_forecast import build_parser

    flags = {option for action in build_parser()._actions
             for option in action.option_strings}
    for flag in ("--progress-format", "--progress-output",
                 "--progress-every", "--frame-markers",
                 "--no-frame-markers"):
        assert flag in page, f"PROGRESS.md does not document {flag}"
        assert flag in flags, f"{flag} is documented but not defined"


def test_go_asks_for_the_records_and_not_the_sentences(tmp_path):
    """`go` is the other caller, and it cannot take stdout sentences.

    Its subprocess arm captures the stage's stdout in memory and throws
    it away on success; its in-process arm is `woof run-plan`, whose
    whole contract is that stdout carries its event stream and nothing
    else.  Both still get every line -- in progress.jsonl.
    """

    from woof import go_cli

    plan = {"runner": "woof.prepared_single_domain_forecast",
            "source": "gfs", "prepared": tmp_path / "prep",
            "authority": tmp_path / "auth", "run": tmp_path / "run"}
    command = go_cli.forecast_command(
        plan, {"proof": "a" * 64, "source_manifest": "b" * 64,
               "prepared_content": "c" * 64})
    assert "--progress-format" in command
    assert command[command.index("--progress-format") + 1] == "jsonl"
    # ... and the flag it passes is one the runner really defines, in a
    # value the runner really accepts.
    from woof.prepared_single_domain_forecast import build_parser
    from woof.progress_log import PROGRESS_FORMATS

    action = next(a for a in build_parser()._actions
                  if "--progress-format" in a.option_strings)
    assert "jsonl" in action.choices
    assert "jsonl" in PROGRESS_FORMATS
    # The TREE command is built the same way.  Checked at the source
    # rather than by calling it, because composing that command reads a
    # real prepared hierarchy off disk; what matters here is that the
    # two builders share one answer instead of carrying two.
    import inspect

    assert "_PROGRESS_FLAGS" in inspect.getsource(
        go_cli.tree_forecast_command)


def test_execute_experiment_takes_a_step_observer():
    from woof.core.model import execute_experiment
    import inspect

    signature = inspect.signature(execute_experiment)
    assert "step_observer" in signature.parameters
    assert signature.parameters["step_observer"].default is None


# ---------------------------------------------------------------------------
# Through the REAL executor, on the REAL schedule
# ---------------------------------------------------------------------------
#
# The formatter tests above prove a grammar.  These prove the SEAM: the
# clocks, the schedule walk and the nest coupling are the shipped ones
# (``tests/test_model``'s scaffold, a ratio-1 two-domain nest over 60 s
# of model time at dt = 6 s), and only ``dycore.step`` -- the CUDA call
# -- is replaced.  A regression that stopped calling the observer, or
# called it once per PERIOD instead of once per DOMAIN STEP, fails here.


def _real_executor_model():
    from test_model import _model

    return _model()


def test_the_real_executor_calls_the_observer_once_per_domain_step(
        monkeypatch):
    from woof.core.model import execute_experiment

    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_args, **_kwargs: None)
    _exp, model = _real_executor_model()
    seen = []
    report = execute_experiment(
        model, validate_state=False, pool_trim_per_period=False,
        step_observer=lambda **event: seen.append(event))

    # 20 STEP ops on this schedule, ten per domain -- the executor's own
    # count, not a number this test chose.
    assert len(seen) == report.steps == 20
    assert [event["grid_id"] for event in seen[:4]] == [1, 2, 1, 2]
    assert [event["step_count"] for event in seen[:4]] == [1, 1, 2, 2]
    assert seen[0]["model_seconds"] == pytest.approx(6.0)
    assert seen[-1]["model_seconds"] == pytest.approx(60.0)
    # A real perf_counter difference: non-negative, and small because
    # the CUDA call is stubbed out.
    assert all(event["step_wall_seconds"] >= 0.0 for event in seen)


def test_the_real_executor_produces_wrf_lines_a_script_can_read(
        monkeypatch, tmp_path):
    from woof.core.model import execute_experiment

    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_args, **_kwargs: None)
    exp, model = _real_executor_model()
    text = StringIO()
    log = StepLog(start_time=exp.start_time,
                  run_seconds=float(exp.run_seconds), text_stream=text,
                  jsonl_path=tmp_path / STEP_LOG_FILENAME,
                  frame_marker_dir=tmp_path / FRAME_MARKER_DIRNAME)
    try:
        execute_experiment(model, validate_state=False,
                           pool_trim_per_period=False,
                           step_observer=log.step_observer)
    finally:
        log.close(status="SUCCESS")

    timing = [WRF_MAIN.match(line) for line in text.getvalue().splitlines()]
    timing = [match for match in timing if match]
    assert len(timing) == 20
    # d01 and d02 alternate on a ratio-1 nest, exactly as WRF's rsl.out
    # alternates them.
    assert [int(m.group(2)) for m in timing[:4]] == [1, 2, 1, 2]
    # The valid times are the model's, derived from the run's own start.
    assert timing[0].group(1) == "1982-05-20_00:00:06"
    assert timing[-1].group(1) == "1982-05-20_00:01:00"
    # ... and the machine stream says exactly the same thing.
    records = [r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
               if r["event"] == "step"]
    assert len(records) == 20
    assert [r["text"] for r in records] == [m.string for m in timing]
    assert records[0]["text"].endswith("step 1")
    assert records[2]["text"].endswith("step 2")
    # Both domains were announced, and both were closed.
    tags = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert [r["text"] for r in tags] == [
        line for line in text.getvalue().splitlines() if line.strip()]
    assert [r["sequence"] for r in tags] == list(range(1, len(tags) + 1))
    assert all(r["schema"] == STEP_LOG_SCHEMA for r in tags)
    assert "domain start" in text.getvalue() and "domain end" in text.getvalue()
    assert sorted(r["domain"] for r in tags
                  if r["event"] == "domain_start") == [1, 2]
    assert sorted(r["steps"] for r in tags
                  if r["event"] == "domain_end") == [10, 10]


def test_the_observer_costs_a_measurable_and_small_fraction(monkeypatch,
                                                            tmp_path):
    """The measurement the brief asked for, taken through the executor.

    Both arms integrate the SAME schedule through the SAME executor;
    only the observer differs.  Printed so the number is in the record
    rather than only in an assertion, and asserted only against a
    catastrophic bound -- a wall-clock ratio on a stubbed dycore is not
    a benchmark of anything but the log.
    """

    from woof.core.model import execute_experiment

    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_args, **_kwargs: None)

    def timed(step_observer):
        _exp, model = _real_executor_model()
        started = time.perf_counter()
        report = execute_experiment(
            model, validate_state=False, pool_trim_per_period=False,
            step_observer=step_observer)
        return time.perf_counter() - started, report.steps

    # The best of three paired readings: a shared Stage-1 host takes whole cores away for a moment (2.9.11
    # pass 9: 1.41 ms per step with 20 test parts and the cut's battery on a development machine's 24 cores, where the same
    # tree read under the bound in the pass before); the observer's own cost is the least of them.
    readings = []
    for attempt in range(3):
        bare, steps = timed(None)
        log = StepLog(start_time=datetime(1982, 5, 20), run_seconds=60.0,
                      text_stream=StringIO(),
                      jsonl_path=tmp_path / f"reading-{attempt}" / STEP_LOG_FILENAME,
                      frame_marker_dir=None)
        logged, _ = timed(log.step_observer)
        log.close(status="SUCCESS")
        readings.append(((logged - bare) / steps, bare, logged))
        if readings[-1][0] < 1e-3:
            break
    per_step, bare, logged = min(readings)
    print(f"\nper-step observer cost: {per_step * 1e6:.1f} us "
          f"({steps} steps; bare {bare * 1e3:.2f} ms, "
          f"logged {logged * 1e3:.2f} ms)")
    assert per_step < 1e-3, f"{per_step * 1e6:.1f} us per step"


# ---------------------------------------------------------------------------
# step-log/v2: the pre-sim phases, and the compile hiding inside step 1
# ---------------------------------------------------------------------------


def test_the_schema_is_v3_and_the_reader_still_takes_what_shipped(tmp_path):
    """A schema bump, and a reader that can replay what shipped.

    Older consumers refuse an unknown schema loudly and that is correct
    -- a `phase` record is a tag v1 was never told about, and the six
    lifecycle tags are ones v2 was never told about.  What must not
    happen is this tree losing the ability to read the streams its own
    published wheels wrote, so `read_step_log` takes every one.
    """

    assert STEP_LOG_SCHEMA == "gpuwm.step-log/v3"
    for shipped in ("gpuwm.step-log/v1", "gpuwm.step-log/v2"):
        assert shipped in STEP_LOG_SCHEMAS
    assert STEP_LOG_SCHEMA in STEP_LOG_SCHEMAS

    for version in ("v1", "v2"):
        legacy = tmp_path / f"{version}.jsonl"
        legacy.write_text(
            json.dumps({"schema": f"gpuwm.step-log/{version}", "sequence": 1,
                        "event": "run_start"}) + "\n"
            + json.dumps({"schema": f"gpuwm.step-log/{version}",
                          "sequence": 2, "event": "run_end"}) + "\n",
            encoding="utf-8")
        assert len(read_step_log(legacy)) == 2

    alien = tmp_path / "alien.jsonl"
    alien.write_text(
        json.dumps({"schema": "somebody.else/v1", "sequence": 1}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        read_step_log(alien)


def test_phase_is_an_event_tag_with_its_own_sentence(tmp_path):
    assert "phase" in STEP_LOG_EVENTS
    log, text = make_log(tmp_path)
    log.phase("preflight_verify", 1.25)
    log.phase("restore_prepared_cache", 3.5, road="store")
    log.close(status="SUCCESS")

    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    phases = [r for r in records if r["event"] == "phase"]
    assert [r["name"] for r in phases] == [
        "preflight_verify", "restore_prepared_cache"]
    assert phases[0]["wall_seconds"] == pytest.approx(1.25)
    assert phases[1]["road"] == "store"
    # One emit, two streams: the sentence the terminal got is inside the
    # record, exactly as every other event in this module.
    assert phases[0]["text"] in text.getvalue()
    assert "preflight_verify" in phases[0]["text"]


def test_a_phase_with_no_measurement_is_not_emitted(tmp_path):
    """``None`` seconds means "this road did not measure it".

    A phase record carrying null would read as "it took no time", which
    is the opposite of what an unmeasured stage means."""

    log, _ = make_log(tmp_path)
    log.phase("preflight_verify", None)
    log.close(status="SUCCESS")
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert not [r for r in records if r["event"] == "phase"]


def test_a_first_step_that_dwarfs_its_neighbours_is_named(tmp_path):
    """THE AUDIT'S 51 SECONDS.  Step 1 was 39x its neighbours and the
    only place that number lived was a step record nobody totalled."""

    log, _ = make_log(tmp_path)
    log.domain_step(grid_id=1, step_count=1, model_seconds=6.0,
                    step_wall_seconds=51.1)
    for step in range(2, 8):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=6.0 * step, step_wall_seconds=1.3)
    log.close(status="SUCCESS")

    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    end = records[-1]
    assert end["event"] == "run_end"
    assert end["first_step_excess_seconds"] == pytest.approx(49.8, abs=1e-6)


def test_an_ordinary_first_step_carries_no_excess(tmp_path):
    log, _ = make_log(tmp_path)
    for step in range(1, 8):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=6.0 * step, step_wall_seconds=1.3)
    log.close(status="SUCCESS")
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert records[-1]["first_step_excess_seconds"] is None


@pytest.mark.parametrize("first, steady", [
    (1.1841, [0.1268, 0.1278, 0.1227, 0.1235, 0.1240, 0.1259]),   # 9.5x
    (1.2683, [0.1281, 0.1252, 0.1270, 0.1206, 0.1215, 0.1228]),   # 10.2x
    (1.1634, [0.1330, 0.1264, 0.1253, 0.1276, 0.1204, 0.1229]),   # 9.2x
])
def test_a_healthy_warm_first_step_is_not_called_an_excess(tmp_path, first,
                                                           steady):
    """VALIDATE THE INSTRUMENT, against real measurements it must not
    fire on.

    These are the first seven step walls of three warm runs of the
    reference case on the reference box, 2026-08-16, with a fully warm
    kernel cache and nothing compiled.  Step 1 is 9-10x its neighbours
    on every one of them -- first-touch allocations, the first history
    alarm, the first load of each cached kernel -- which is why the
    obvious 10x threshold fired on a healthy run and had to be chosen
    against this distribution instead of guessed.
    """

    log, _ = make_log(tmp_path)
    for index, wall in enumerate([first, *steady], start=1):
        log.domain_step(grid_id=1, step_count=index,
                        model_seconds=6.0 * index, step_wall_seconds=wall)
    log.close(status="SUCCESS")
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert records[-1]["first_step_excess_seconds"] is None


def test_the_measured_cold_run_is_still_caught(tmp_path):
    """The other direction of the same instrument check.

    MEASURED cold: step 1 of 51.1 s against a 0.13 s steady state.
    A threshold chosen to clear the warm distribution must still leave
    this unmistakable.
    """

    log, _ = make_log(tmp_path)
    log.domain_step(grid_id=1, step_count=1, model_seconds=6.0,
                    step_wall_seconds=51.1)
    for step in range(2, 8):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=6.0 * step, step_wall_seconds=0.13)
    log.close(status="SUCCESS")
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert records[-1]["first_step_excess_seconds"] == pytest.approx(
        50.97, abs=1e-6)


def test_the_excess_survives_a_thinned_stream(tmp_path):
    """``--progress-every`` must not decide whether the compile is
    visible.  The wall of every step is recorded; only whether a ``step``
    RECORD is written is thinned."""

    log, _ = make_log(tmp_path, every=100)
    log.domain_step(grid_id=1, step_count=1, model_seconds=6.0,
                    step_wall_seconds=51.1)
    for step in range(2, 8):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=6.0 * step, step_wall_seconds=1.3)
    log.close(status="SUCCESS")
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert records[-1]["first_step_excess_seconds"] == pytest.approx(49.8)


def test_an_announced_compile_becomes_a_timed_phase_of_its_own(tmp_path):
    """Notify AND time.  The notice says it is coming; this says what it
    cost, in the same grammar as every other pre-sim phase, so a reader
    totalling ``phase`` records is not left with a 51 s hole labelled
    "step 1"."""

    log, _ = make_log(tmp_path)
    log.announce_kernel_compile(reason="architecture_missing",
                                compute_capability="86")
    log.domain_step(grid_id=1, step_count=1, model_seconds=6.0,
                    step_wall_seconds=51.1)
    for step in range(2, 8):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=6.0 * step, step_wall_seconds=1.3)
    log.close(status="SUCCESS")

    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    compiles = [r for r in records
                if r["event"] == "phase" and r["name"] == "kernel_compile"]
    assert len(compiles) == 1
    phase = compiles[0]
    assert phase["wall_seconds"] == pytest.approx(49.8)
    assert phase["reason"] == "architecture_missing"
    assert phase["compute_capability"] == "86"
    # ... and it lands before run_end, so a reader replaying the stream
    # in order learns the cost before it learns the run is over.
    assert records.index(phase) < len(records) - 1


def test_an_unannounced_slow_first_step_is_not_called_a_compile(tmp_path):
    """The excess is always reported; the NAME is only claimed when the
    cache said a compile was coming.  Calling any slow first step a
    kernel compile would be a guess wearing a receipt's clothes."""

    log, _ = make_log(tmp_path)
    log.domain_step(grid_id=1, step_count=1, model_seconds=6.0,
                    step_wall_seconds=51.1)
    for step in range(2, 8):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=6.0 * step, step_wall_seconds=1.3)
    log.close(status="SUCCESS")
    records = read_step_log(tmp_path / STEP_LOG_FILENAME)
    assert not [r for r in records
                if r["event"] == "phase" and r["name"] == "kernel_compile"]
    assert records[-1]["first_step_excess_seconds"] == pytest.approx(49.8)


def test_the_null_log_takes_the_new_calls_too(tmp_path):
    """``--progress-format off`` is a null OBJECT, not a None to guard.

    A new method that exists on one and not the other is a call site
    that crashes only for the reader who asked for silence."""

    log = open_step_log(outdir=tmp_path, start_time=START, run_seconds=60.0,
                        progress_format="off")
    log.phase("preflight_verify", 1.0)
    log.announce_kernel_compile(reason="cold_cache", compute_capability="86")
    log.close()


# ---------------------------------------------------------------------------
# step-log/v3: what the tree does to itself -- lifecycle and relocation
# ---------------------------------------------------------------------------
#
# The gap these close, stated once: a run could move a nest across half a
# state, retire an episode and re-arm the slot, and the per-step stream
# said nothing at all.  Every one of those decisions lived only in a
# receipt file written for a post-mortem, so a live map had no way to
# draw the tree it was watching -- the rectangles could not move and the
# track could not be drawn.  These six tags are that stream.


PLACEMENT_A = {"i_parent_start": 20, "j_parent_start": 18}
PLACEMENT_B = {"i_parent_start": 22, "j_parent_start": 18}


class FakeGrid:
    """The one method a placement is turned into a position through.

    ``ProjectedGrid.ij_to_latlon`` takes 1-BASED mass coordinates; this
    is the same surface :mod:`woof.core.storm_track_writer` reads, so a
    log that gets a position out of this gets one out of a real grid.
    """

    def __init__(self, lat0=35.0, lon0=-97.0, step=0.01, e_we=41, e_sn=41):
        self.lat0, self.lon0, self.step = lat0, lon0, step
        self.e_we, self.e_sn = e_we, e_sn

    def ij_to_latlon(self, x, y):
        return (self.lat0 + (y - 1.0) * self.step,
                self.lon0 + (x - 1.0) * self.step)


def test_the_lifecycle_vocabulary_is_declared_and_spelled_as_the_spec_says():
    """The six tags a live map binds to, by name.

    Studio's Live Run screen is being built against these spellings in a
    parallel lane, so a rename here is a rename of somebody else's
    parser.  Enumerated rather than described.
    """

    assert NEST_EVENTS == ("nest_spawned", "nest_retired", "nest_rearmed",
                           "nest_moved", "containment_moved", "track_fix")
    for tag in NEST_EVENTS:
        assert tag in STEP_LOG_EVENTS, tag


def test_a_spawn_is_an_event_with_a_domain_a_time_and_a_position(tmp_path):
    log, text = make_log(tmp_path)
    log.domain_step(grid_id=1, step_count=5, model_seconds=60.0,
                    step_wall_seconds=0.1)
    log.nest_spawned(domain=3, model_seconds=60.0, episode=1, parent=1,
                     placement=PLACEMENT_A, grid=FakeGrid(),
                     trigger="uh")
    log.close(status="SUCCESS")

    record, = [r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
               if r["event"] == "nest_spawned"]
    assert record["domain"] == 3
    assert record["parent"] == 1
    assert record["episode"] == 1
    assert record["trigger"] == "uh"
    assert record["placement"] == PLACEMENT_A
    assert record["valid_time"] == "2026-08-15_00:01:00"
    assert record["model_seconds"] == 60.0
    # A newborn has taken no step of its own, and says so rather than
    # borrowing its parent's count.
    assert record["step"] == 0
    # The grid's own centre: mass dimensions 40x40, 0-based centre 19.5,
    # which ij_to_latlon reads as the 1-based 20.5.
    assert record["lat"] == pytest.approx(35.195)
    assert record["lon"] == pytest.approx(-96.805)
    # One emit, two streams -- the same property every other tag has.
    assert record["text"] in text.getvalue()
    assert "nest_spawned" in record["text"]


def test_retire_and_rearm_are_separate_events_carrying_the_episode(tmp_path):
    log, _ = make_log(tmp_path)
    log.nest_retired(domain=3, model_seconds=2700.0, episode=1,
                     reason="retire", grid=FakeGrid())
    log.nest_rearmed(domain=3, model_seconds=3600.0, episode=2,
                     cooldown_seconds=900.0)
    log.close(status="SUCCESS")

    records = {r["event"]: r for r in
               read_step_log(tmp_path / STEP_LOG_FILENAME)}
    retired = records["nest_retired"]
    assert retired["domain"] == 3 and retired["episode"] == 1
    assert retired["reason"] == "retire"
    assert retired["valid_time"] == "2026-08-15_00:45:00"
    assert retired["lat"] is not None and retired["lon"] is not None

    rearmed = records["nest_rearmed"]
    # The episode the slot is armed FOR, not the one that ended: the
    # bound rearm counts against is stated in firings.
    assert rearmed["episode"] == 2
    assert rearmed["cooldown_seconds"] == 900.0
    # Nothing exists to have a position yet, and a re-arm says so rather
    # than repeating where the last episode happened to sit.
    assert rearmed["lat"] is None and rearmed["lon"] is None


def test_a_move_carries_the_placement_it_left_and_the_one_it_took(tmp_path):
    """Old AND new, because a map has to draw the origin ghost."""

    log, _ = make_log(tmp_path)
    log.domain_step(grid_id=2, step_count=40, model_seconds=600.0,
                    step_wall_seconds=0.1)
    log.nest_moved(domain=2, model_seconds=600.0,
                   placement_from=PLACEMENT_A, placement_to=PLACEMENT_B,
                   requested_shift=[3, 0], executed_shift=[2, 0],
                   clamped_by=["max_move_parent_cells"],
                   grid=FakeGrid(lat0=35.02),
                   lat_from=35.195, lon_from=-96.805)
    log.close(status="SUCCESS")

    record, = [r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
               if r["event"] == "nest_moved"]
    assert record["domain"] == 2
    assert record["step"] == 40
    assert record["placement_from"] == PLACEMENT_A
    assert record["placement_to"] == PLACEMENT_B
    assert record["requested_shift_parent_cells"] == [3, 0]
    assert record["executed_shift_parent_cells"] == [2, 0]
    assert record["clamped_by"] == ["max_move_parent_cells"]
    # The position it took, and the position it left.
    assert record["lat"] == pytest.approx(35.215)
    assert record["lat_from"] == pytest.approx(35.195)
    assert record["lon_from"] == pytest.approx(-96.805)


def test_a_containment_slide_names_the_slider_and_the_mover(tmp_path):
    """Two grid ids, and they are not interchangeable: the domain that
    MOVED is the parent, and the one it moved to keep contained did not
    move at all."""

    log, _ = make_log(tmp_path)
    log.containment_moved(domain=2, model_seconds=600.0, mover=3,
                          placement_from=PLACEMENT_A,
                          placement_to=PLACEMENT_B,
                          requested_shift=[5, 0], executed_shift=[2, 0],
                          clamped=True, mover_deviation_cells=[14, 0],
                          grid=FakeGrid())
    log.close(status="SUCCESS")

    record, = [r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
               if r["event"] == "containment_moved"]
    assert record["domain"] == 2 and record["mover"] == 3
    assert record["mover_deviation_cells"] == [14, 0]
    assert record["clamped"] is True
    assert record["placement_to"] == PLACEMENT_B


def test_a_track_fix_is_a_position_and_the_domain_it_steers(tmp_path):
    log, _ = make_log(tmp_path)
    log.track_fix(domain=2, model_seconds=600.0, lat=14.2, lon=-74.1,
                  found=True, refined_on=3)
    log.track_fix(domain=2, model_seconds=1200.0, lat=None, lon=None,
                  found=False)
    log.close(status="SUCCESS")

    first, second = [r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
                     if r["event"] == "track_fix"]
    assert first["domain"] == 2
    assert (first["lat"], first["lon"]) == (14.2, -74.1)
    assert first["found"] is True and first["refined_on"] == 3
    # A consultation that found nothing is still a record: the file's
    # time axis keeps its gap visible instead of leaving a reader to
    # infer one from a jump in the clock.
    assert second["found"] is False
    assert second["lat"] is None and second["lon"] is None


def test_a_position_that_cannot_be_derived_is_null_and_not_a_crash(tmp_path):
    """Telemetry never fails a run.

    An idealized tree carries no projection at all, and a node whose
    grid is a stand-in must produce a record with no position rather
    than an exception inside the integration loop.
    """

    log, _ = make_log(tmp_path)
    log.nest_moved(domain=2, model_seconds=60.0, placement_from=PLACEMENT_A,
                   placement_to=PLACEMENT_B, grid="not-a-grid")
    log.nest_spawned(domain=3, model_seconds=60.0, episode=1, grid=None)
    log.close(status="SUCCESS")

    records = [r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
               if r["event"] in NEST_EVENTS]
    assert len(records) == 2
    assert all(r["lat"] is None and r["lon"] is None for r in records)


def test_the_null_log_takes_the_lifecycle_calls_too(tmp_path):
    """``--progress-format off`` must not be the one road that crashes."""

    log = open_step_log(outdir=tmp_path, start_time=START, run_seconds=60.0,
                        progress_format="off")
    log.nest_spawned(domain=3, model_seconds=1.0, episode=1)
    log.nest_retired(domain=3, model_seconds=2.0, episode=1)
    log.nest_rearmed(domain=3, model_seconds=3.0, episode=2)
    log.nest_moved(domain=2, model_seconds=4.0, placement_from=PLACEMENT_A,
                   placement_to=PLACEMENT_B)
    log.containment_moved(domain=1, model_seconds=5.0, mover=2,
                          placement_from=PLACEMENT_A,
                          placement_to=PLACEMENT_B)
    log.track_fix(domain=2, model_seconds=6.0, lat=1.0, lon=2.0)
    log.close()


def test_a_model_with_no_log_published_still_answers_the_emitters():
    """The seam the runners reach the stream through.

    A relocation runner is handed the MODEL, not the log, so a route
    that opens no step log at all (or opened an inert one) must leave
    every emit site a no-op rather than a guard each call site has to
    remember.
    """

    from types import SimpleNamespace

    model = SimpleNamespace()
    log = model_step_log(model)
    assert log.enabled is False
    log.nest_moved(domain=2, model_seconds=1.0, placement_from=PLACEMENT_A,
                   placement_to=PLACEMENT_B)

    real = StepLog(start_time=START, run_seconds=60.0)
    publish_step_log(model, real)
    assert model_step_log(model) is real
    real.close(status="SUCCESS")


# ---------------------------------------------------------------------------
# The inertness gate: a run with no nests emits exactly what it always did
# ---------------------------------------------------------------------------

#: Which fields of a record differ between two identical runs on two
#: machines, keyed by the tag that carries them.  Everything NOT named
#: here is a property of the run and is compared byte for byte.
_VOLATILE = {
    "run_start": ("emitted_unix_ms", "pid", "frame_marker_dir"),
    "phase": ("emitted_unix_ms",),
    "domain_start": ("emitted_unix_ms",),
    "step": ("emitted_unix_ms",),
    "output_written": ("emitted_unix_ms", "path", "marker"),
    "restart_written": ("emitted_unix_ms", "path"),
    "domain_end": ("emitted_unix_ms",),
    # ``text`` joins the list on run_end alone: that sentence quotes the
    # run's wall clock, so it is the one printed line whose bytes are a
    # property of the machine.  Its grammar is pinned by
    # ``test_run_end_states_the_outcome`` instead.
    "run_end": ("emitted_unix_ms", "wall_seconds", "text"),
}

NEST_FREE_GOLDEN = Path(__file__).parent / "data" / (
    "progress-nest-free-stream.jsonl")


def nest_free_stream(tmp_path) -> str:
    """One nest-free run's whole stream, with the volatile fields fixed.

    Deliberately exercises every tag a nest-free forecast can reach, so
    the comparison below covers the whole surface rather than a sample.
    """

    log, _ = make_log(tmp_path)
    frame = tmp_path / "wrfout_d01_2026-08-15_00_00_00"
    frame.write_bytes(b"x" * 7)
    log.phase("preflight_verify", 0.5)
    for step in (1, 2):
        log.domain_step(grid_id=1, step_count=step,
                        model_seconds=12.0 * step, step_wall_seconds=0.25)
    log.output_committed(domain=1, valid_time=START, path=frame,
                         wall_seconds=0.5)
    log.restart_written(domain=1, valid_time=START, path=frame)
    log.close(status="SUCCESS")

    lines = []
    for record in read_step_log(tmp_path / STEP_LOG_FILENAME):
        fixed = dict(record)
        # The version stamp is the ONE difference this vocabulary is
        # allowed to make to a nest-free stream, and it is pinned by its
        # own cell above rather than smuggled through this one.
        fixed["schema"] = "<schema>"
        for key in _VOLATILE[record["event"]]:
            fixed[key] = "<volatile>"
        lines.append(json.dumps(fixed))
    return "\n".join(lines) + "\n"


def test_a_run_with_no_nests_emits_exactly_what_it_emitted_before(tmp_path):
    """THE INERTNESS GATE, and it is the point of the whole lane.

    A forecast with one domain, no relocation and no spawn must be
    unable to tell that the lifecycle vocabulary exists.  The golden
    beside this file was taken from the stream BEFORE these six tags
    landed; the only difference the change is permitted to make to it is
    the schema version stamp, which is fixed above and pinned on its
    own.  Anything else -- a new record, a new field on an old record, a
    changed value -- fails here, and that is the intent: a run that grew
    a byte because a feature it does not use exists is a regression.
    """

    assert NEST_FREE_GOLDEN.is_file(), (
        f"{NEST_FREE_GOLDEN} is the recorded nest-free stream and it is "
        "missing; without it there is nothing to be inert against")
    assert nest_free_stream(tmp_path) == NEST_FREE_GOLDEN.read_text(
        encoding="utf-8")


# ---------------------------------------------------------------------------
# The two WRF-input doors
# ---------------------------------------------------------------------------
#
# Both of them already PUBLISH markers -- they call ``run_prepared_tree``
# with ``progress_options`` omitted, and it resolves ``None`` to the
# defaults, which are on.  What they had was no way to say otherwise:
# not a format, not a cadence, not an output, not ``--no-frame-markers``,
# on either parser, and nothing carried through the ``--_worker``
# re-launch that does the actual run.  A behaviour with no control is
# not the same defect as a behaviour that is missing, and this pins both
# halves: the flags exist, and the default stays on.


def test_the_worker_doors_carry_the_progress_flags(tmp_path, monkeypatch):
    """The four flags reach both parsers, the runner and the worker argv."""

    import subprocess
    from types import SimpleNamespace

    from woof import go_cli, metem_door, metem_forecast, supervisor
    from woof import prepared_domain_tree_forecast, wrfinput_forecast
    from woof.progress_log import ProgressOptions

    four = ("--progress-format", "--progress-output", "--progress-every",
            "--no-frame-markers")

    def spelled(parser):
        return {flag for action in parser._actions
                for flag in action.option_strings}

    for flag in four:
        assert flag in spelled(wrfinput_forecast.build_parser()), flag
    for flag in four:
        assert flag in spelled(metem_forecast.build_parser()), flag

    recorded = {}
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(
        metem_door, "resolve_metem_run",
        lambda directory, **kwargs: SimpleNamespace(
            toml_text="", experiment=None,
            substitution_report=SimpleNamespace(substitutions=())))
    # Plan review reads the resolved experiment; these tests are about the
    # progress flags, so the review is answered without one.
    monkeypatch.setattr(metem_forecast, "resolve_metem_vertical",
                        lambda run, text, **kwargs: None)
    monkeypatch.setattr(metem_door, "check_analyzed_scalar_capabilities",
                        lambda run: None)
    inputs = SimpleNamespace(experiment=SimpleNamespace(
        start_time=datetime(2026, 5, 17, 18)), prepared_root=tmp_path / "prepared")
    # The substitution announcement reads the preparation's import record,
    # which the stubbed preparation never writes.
    monkeypatch.setattr(wrfinput_forecast, "announce_wrf_substitutions",
                        lambda run, path: None)
    monkeypatch.setattr(metem_forecast, "prepare_metem_run",
                        lambda run, directory, **kwargs: inputs)
    monkeypatch.setattr(metem_forecast, "MetemInitialization", lambda inputs: object())

    def run_prepared_tree(inputs, *, output_directory, **kwargs):
        recorded["options"] = kwargs.get("progress_options")
        return {"status": "ok"}

    monkeypatch.setattr(prepared_domain_tree_forecast, "run_prepared_tree",
                        run_prepared_tree)
    assert metem_forecast.run_metem_forecast(
        tmp_path / "met", tmp_path / "out", exclusive_gpu=False,
        render_products="none",
        progress_options=ProgressOptions(frame_markers=False)) == 0
    assert recorded["options"] == ProgressOptions(frame_markers=False)

    # ... and the supervised launch, which is where the run happens by
    # default, hands the child the same answer.
    class _Lock:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(supervisor, "select_gpu",
                        lambda uuid=None: SimpleNamespace(uuid="GPU-0"))
    monkeypatch.setattr(supervisor, "GPUFileLock", _Lock)
    monkeypatch.setattr(supervisor, "preflight_exclusive_gpu", lambda *a, **k: None)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: (
        recorded.__setitem__("argv", list(command)) or SimpleNamespace(returncode=0)))
    assert metem_forecast.run_metem_forecast(
        tmp_path / "met", tmp_path / "supervised",
        progress_options=ProgressOptions(frame_markers=False,
                                         progress_format="jsonl",
                                         progress_every=5)) == 0
    argv = recorded["argv"]
    assert "--no-frame-markers" in argv
    assert argv[argv.index("--progress-format") + 1] == "jsonl"
    assert argv[argv.index("--progress-every") + 1] == "5"


def test_the_worker_doors_publish_markers_with_no_flags_at_all(tmp_path, monkeypatch):
    """Default-on, pinned: absent flags are the defaults, and they are on."""

    from types import SimpleNamespace

    from woof import go_cli, metem_door, metem_forecast
    from woof import prepared_domain_tree_forecast, wrfinput_forecast
    from woof.progress_log import ProgressOptions

    parsed = wrfinput_forecast.build_parser().parse_args(
        ["--wrfinput", "w", "--outdir", str(tmp_path / "o")])
    assert ProgressOptions.from_args(parsed) == ProgressOptions()
    assert ProgressOptions.worker_flags(ProgressOptions.from_args(parsed)) == []

    recorded = {}
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(
        metem_door, "resolve_metem_run",
        lambda directory, **kwargs: SimpleNamespace(
            toml_text="", experiment=None,
            substitution_report=SimpleNamespace(substitutions=())))
    # Plan review reads the resolved experiment; these tests are about the
    # progress flags, so the review is answered without one.
    monkeypatch.setattr(metem_forecast, "resolve_metem_vertical",
                        lambda run, text, **kwargs: None)
    monkeypatch.setattr(metem_door, "check_analyzed_scalar_capabilities",
                        lambda run: None)
    inputs = SimpleNamespace(experiment=SimpleNamespace(
        start_time=datetime(2026, 5, 17, 18)), prepared_root=tmp_path / "prepared")
    # The substitution announcement reads the preparation's import record,
    # which the stubbed preparation never writes.
    monkeypatch.setattr(wrfinput_forecast, "announce_wrf_substitutions",
                        lambda run, path: None)
    monkeypatch.setattr(metem_forecast, "prepare_metem_run",
                        lambda run, directory, **kwargs: inputs)
    monkeypatch.setattr(metem_forecast, "MetemInitialization", lambda inputs: object())
    monkeypatch.setattr(
        prepared_domain_tree_forecast, "run_prepared_tree",
        lambda inputs, *, output_directory, **kwargs: recorded.__setitem__(
            "options", kwargs.get("progress_options")) or {"status": "ok"})
    assert metem_forecast.run_metem_forecast(
        tmp_path / "met", tmp_path / "bare", exclusive_gpu=False,
        render_products="none") == 0
    # None is the runner's own spelling of "the defaults", which are on.
    assert recorded["options"] is None
    assert ProgressOptions().frame_markers is True
