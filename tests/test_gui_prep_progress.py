"""A forecast's page says where its preparation is while it prepares.

The breakage: from the download to the first model step, the run page and the
list said only "Preparing the grid · forecast hour 0 of 15 · 12 / 3 / 1 km"
and nothing moved for minutes, although the engine was downloading, reading
the starting data, building each grid's start state and writing boundary
times.  Each test replays a real run's recorded events (tests/data/
gui-prep-progress, trimmed from runs on the test host 4 page) or records written
by the engine's own step reporters, into a live run folder, and reads what
the page server says about it; the last ones draw the words with the page's
own script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from woof import progress, runplan
from woof.gui import runs
from woof.gui.api import Api

from test_gui_server import make_run

DATA = Path(__file__).resolve().parent / "data" / "gui-prep-progress"
JS = Path(__file__).resolve().parents[1] / "woof" / "gui" / "static" / "js"


def recorded(name: str, *, through: int) -> list[str]:
    lines = (DATA / name).read_text(encoding="utf-8").splitlines()
    return [line for line in lines if json.loads(line)["sequence"] <= through]


def live_run(root: Path, name: str, lines: list[str], *, heartbeat: str | None = None) -> Path:
    """A run folder whose recorded process is this test's own, so it reads as running."""

    run = make_run(root, name, pid=os.getpid())
    (run / runs.EVENTS).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    if heartbeat is not None:
        (run / runs.HEARTBEAT).write_text(json.dumps({"status": heartbeat, "model_elapsed_seconds": 0.0}),
                                          encoding="utf-8")
    runs._CACHE.clear()
    return run


class Observer:
    """The run-plan observer's stream, which is all the relay reads."""

    def __init__(self, events):
        self.events = events


def engine_records(run: Path, act) -> None:
    """Append what the engine's own step reporters say while ``act`` runs, through the run's relay."""

    with runplan.EventStream(run / runs.EVENTS, mirror=None) as events:
        with runplan._preparation_relay(Observer(events)):
            act(events)
    runs._CACHE.clear()


def test_a_download_says_how_many_files_are_done_and_how_many_bytes(tmp_path):
    # The recorded HRRR download: six files in flight, then two of them land (407917152 and 428877471 bytes).
    run = live_run(tmp_path, "download", recorded("download.jsonl", through=73))
    status = runs.status(run)
    assert status["state"] == "running"
    prep = status["preparation"]
    assert prep["stage"] == "fetch"
    assert prep["fetch"]["files_done"] == 2
    assert prep["fetch"]["bytes"] == 407917152 + 428877471
    # The route declared no whole-request count or sizes, so no total is invented.
    assert prep["fetch"]["files_total"] is None and prep["fetch"]["bytes_total"] is None


def test_a_download_with_declared_sizes_says_bytes_against_the_total(tmp_path):
    run = live_run(tmp_path, "sized", recorded("download.jsonl", through=3))

    def fetch(_events):
        with progress.TransferMonitor("fetch source", ticker=False) as monitor:
            monitor.begin([("a.grib2", 3_000_000), ("b.grib2", 1_000_000)])
            monitor.start("a.grib2", expected_bytes=3_000_000)
            monitor.start("b.grib2", expected_bytes=1_000_000)
            monitor.finish("b.grib2", size=1_000_000)
            monitor.observe("a.grib2", 500_000)
            monitor.tick(force=True)

    # The fetch stage's own relay (runplan._run_fetch) puts every transfer record on the stream.
    with runplan.EventStream(run / runs.EVENTS, mirror=None) as events:
        with progress.event_sink(lambda event, **fields: events.emit(event, **fields)):
            fetch(events)
    runs._CACHE.clear()
    got = runs.status(run)["preparation"]["fetch"]
    assert (got["files_done"], got["files_total"]) == (1, 2)
    assert (got["bytes"], got["bytes_total"]) == (1_500_000, 4_000_000)


def test_an_arco_download_says_which_time_it_is_reading(tmp_path):
    from woof import era5_arco

    run = live_run(tmp_path, "arco", recorded("download.jsonl", through=3))
    said = []

    def fetch(events):
        relay = era5_arco._acquisition_progress(said.append, ["t0", "t1", "t2", "t3"], tmp_path / "missing.nc")
        with progress.event_sink(lambda event, **fields: events.emit(event, **fields)):
            # The native reader's own lines, as zarr_bridge relays them.
            relay("fetch: Zarr: 2011-05-22 12:00:00 1/4 geopotential")
            relay("fetch: Zarr: 2011-05-22 12:00:00 1/4 temperature")
            relay("fetch: Zarr: 2011-05-22 18:00:00 2/4 geopotential")

    with runplan.EventStream(run / runs.EVENTS, mirror=None) as events:
        fetch(events)
    runs._CACHE.clear()
    got = runs.status(run)["preparation"]["fetch"]
    assert (got["times_done"], got["times_total"]) == (1, 4)
    # Every line still reaches the fetch's own log.
    assert len(said) == 3


def test_a_nested_preparation_names_its_phase_step_and_grid(tmp_path):
    # Recorded: a 12 / 3 / 1 km ERA5 run inside prepare, compiling a GPU kernel.
    lines = recorded("nested-prepare.jsonl", through=6)
    run = live_run(tmp_path, "nested", lines, heartbeat="preparing:build-domain-tree")
    prep = runs.status(run)["preparation"]
    assert prep["stage"] == "prepare"
    assert prep["phase"] == "build-domain-tree"
    assert prep["compiling"] == 1
    assert prep["stage_seconds"] is not None and prep["stage_seconds"] >= 0

    # The builder then starts the 3 km grid's start state (woof.core.model.build_experiment).
    second = progress.prep_stage("domain_initialize", label="Start state and boundaries", index=2, count=3)

    def build(_events):
        with progress.prep_stage("source_decode", label="Read the starting data"):
            pass
        with progress.prep_stage("domain_initialize", label="Start state and boundaries", index=1, count=3):
            pass
        second.__enter__()

    engine_records(run, build)
    prep = runs.status(run)["preparation"]
    second.__exit__(None, None, None)
    assert prep["step"]["key"] == "domain_initialize"
    assert (prep["step"]["index"], prep["step"]["count"], prep["step"]["grid_km"]) == (2, 3, 3.0)
    assert prep["compiling"] is None


def test_a_real_nested_run_reads_step_by_step_from_download_to_the_first_model_step(tmp_path):
    # Recorded on test host 4 from this change's engine: the 8 GB 12 / 3 / 1 km ERA5 run an event page starts, from
    # its download to its third model step (a card with no kernel cache compiles through the first one).
    def at(sequence, heartbeat=None):
        run = live_run(tmp_path / str(sequence), "real", recorded("nested-run.jsonl", through=sequence),
                       heartbeat=heartbeat)
        return runs.status(run)["preparation"]

    fetch = at(6)["fetch"]
    assert (fetch["times_done"], fetch["times_total"]) == (2, 4)
    decode = at(11, heartbeat="preparing:build-domain-tree")
    assert decode["step"]["key"] == "source_decode" and decode["phase"] == "build-domain-tree"
    first = at(14)
    assert (first["step"]["index"], first["step"]["grid_km"], first["compiling"]) == (1, 12.0, 1)
    second = at(17)
    assert (second["step"]["index"], second["step"]["count"], second["step"]["grid_km"]) == (2, 3, 3.0)
    third = at(19)
    assert (third["step"]["index"], third["step"]["grid_km"]) == (3, 1.0)
    # The first model step, compiling: said, not "forecast hour 0" alone.
    compiling = at(33)
    assert compiling["stage"] == "forecast" and compiling["compiling"]
    # Stepping, with nothing left to prepare: the forecast's own line takes over.
    assert at(59) is None

def test_a_chained_forecast_says_how_many_boundary_times_are_ready(tmp_path):
    run = live_run(tmp_path, "chained", recorded("download.jsonl", through=3))

    def chain(events):
        events.emit("stage_started", stage="prepare", phase="prepare")
        with progress.prep_stage("root_initialize", label="Initialize root forcing states", count=1):
            pass
        events.emit("stage_finished", stage="prepare", wall_seconds=1.0, phases=["prepare"])
        events.emit("prepare_head_ready", head_sha256="0" * 64)
        events.emit("stage_started", stage="forecast", phase="forecast")
        for done in (1, 2, 3):
            progress.prep_progress("root_boundaries", label="Boundary times", done=done, count=15)

    engine_records(run, chain)
    prep = runs.status(run)["preparation"]
    assert prep["stage"] == "forecast" and prep["chained"]
    assert prep["boundaries"] == {"done": 3, "count": 15}

    engine_records(run, lambda events: events.emit("prepare_sealed", prepared_root="/p", prepared={}))
    assert runs.status(run)["preparation"] is None


def test_every_boundary_time_the_preparation_writes_is_said(monkeypatch, tmp_path, capsys):
    import woof.mapped_direct as mapped
    from test_mapped_direct import _install_prepare_fakes

    # The fakes' source holds two times: the start state and one boundary interval after it.
    args, _, _ = _install_prepare_fakes(monkeypatch, tmp_path, domain_count=1, backend="cpu")
    heard = []
    with progress.event_sink(lambda event, **fields: heard.append(fields.get("preparation"))):
        mapped.prepare_mapped_wrf(**args, stock_wrf_export="off")
    counted = [row for row in heard if row and row["stage"] == "root_boundaries" and row["event"] == "progress"]
    assert [(row["index"], row["count"]) for row in counted] == [(1, 1)]
    # The same record reaches a command-line host on stderr, beside the started and finished pair.
    assert capsys.readouterr().err.count('"event": "progress"') == 1


def test_the_staged_route_relays_its_in_process_preparation(tmp_path):
    # woof prep runs in the run-plan process on the staged route; its steps reached no listener before.
    with runplan.EventStream(tmp_path / "events.jsonl", mirror=None) as events:
        with runplan._preparation_relay(Observer(events)):
            with progress.prep_stage("root_static", label="Prepare root static fields"):
                pass
        with progress.prep_stage("root_static", label="outside the run"):
            pass
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    said = [(row["code"], row["preparation"]["event"]) for row in rows]
    assert said == [("preparation_progress", "started"), ("preparation_progress", "finished")]


# ---------------------------------------------------------------- steps of a preparer that is its own program

#: A preparer program's steps, reported the way the staged route's adapter reports them (woof.mapped_direct): each
#: one through prep_stage or prep_progress in the CHILD process, which reaches its parent only as a stderr line.
PREPARER = """
import os, sys
from woof.progress import prep_progress, prep_stage
if sys.argv[1] == "static":
    with prep_stage("root_static", label="Prepare root static fields"):
        pass
    step = prep_stage("root_initialize", label="Initialize root forcing states", backend="cpu")
    step.__enter__()
    # Gone with the step still open, as a preparer is while its parent watches: a normal exit would close it.
    os._exit(0)
else:
    for done in (1, 2, 3):
        prep_progress("root_boundaries", label="Boundary times", done=done, count=15)
"""


def test_the_staged_route_hears_the_steps_of_its_preparer_program(tmp_path):
    # The staged route runs `woof prep` in the run-plan process, and prep runs the preparer as its own program:
    # the program's steps reached that process only as stderr lines, which prep printed and dropped, so the run
    # page of a staged HRRR run showed no step and no boundary-time count for its whole preparation.
    import sys
    from woof import prep_output, source_cli

    run = live_run(tmp_path, "staged", recorded("download.jsonl", through=3))
    args = SimpleNamespace(output_root=tmp_path / "prepared", explain=False)

    def prepare(part):
        return lambda: source_cli._run_native_adapter([sys.executable, "-c", PREPARER, part])

    def head(events):
        events.emit("stage_started", stage="prepare", phase="prepare")
        assert prep_output.run_preparation(args, prepare("static")) == 0

    engine_records(run, head)
    heard = [row["preparation"] for row in runs.event_records(run / runs.EVENTS)
             if row.get("code") == "preparation_progress"]
    assert [(row["stage"], row["event"]) for row in heard] == [
        ("root_static", "started"), ("root_static", "finished"), ("root_initialize", "started")]
    step = runs.status(run)["preparation"]["step"]
    assert (step["key"], step["label"]) == ("root_initialize", "Initialize root forcing states")

    def chained(events):
        events.emit("stage_finished", stage="prepare", wall_seconds=1.0, phases=["prepare"])
        events.emit("prepare_head_ready", head_sha256="0" * 64)
        events.emit("stage_started", stage="forecast", phase="forecast")
        assert prep_output.run_preparation(args, prepare("boundaries")) == 0

    engine_records(run, chained)
    prep = runs.status(run)["preparation"]
    assert prep["stage"] == "forecast" and prep["chained"]
    assert prep["boundaries"] == {"done": 3, "count": 15}


def test_go_puts_its_preparer_programs_steps_on_the_run_and_says_each_once(tmp_path, monkeypatch, capsys):
    # `woof go` reads the preparer program's output itself; it printed each step and the run's events never
    # heard one, so a page watching that run had nothing to show.
    import sys
    from woof import source_cli
    from woof.cli import main
    from test_go_native_launch import allow_launch_resources, emit

    config = emit(tmp_path)
    allow_launch_resources(monkeypatch)

    def fetch(*_args, **_kwargs):
        # Where the staged route's `woof prep` launches its preparer, inside go's own run.
        assert source_cli._run_native_adapter([sys.executable, "-c", PREPARER, "static"]) == 0
        raise runplan.StageExitError("prepare", 70)

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(tmp_path / "run"), "--run-stamp", "off"]) == 1
    said = capsys.readouterr()
    assert said.out.count("go: Prepare root static fields\n") == 1
    assert said.out.count("go: Prepare root static fields: done") == 1
    assert said.out.count("go: Initialize root forcing states (cpu)\n") == 1
    # The step lines are words; only the stopped stage's replayed tail on stderr quotes its raw output.
    assert "GPUWM_PREP_EVENT" not in said.out
    stream = next((tmp_path / "run").rglob(runs.EVENTS))
    heard = [row["preparation"] for row in runs.event_records(stream) if row.get("code") == "preparation_progress"]
    assert [(row["stage"], row["event"]) for row in heard] == [
        ("root_static", "started"), ("root_static", "finished"), ("root_initialize", "started")]


# ---------------------------------------------------------------- the builder's steps, heard where the run is held

class _StopAtTheFirstNest(Exception):
    """Ends the builder once the first nest's start state has begun: every step it says is said by then."""


def _grid(grid_id, parent_id):
    return SimpleNamespace(grid_id=grid_id, parent_id=parent_id, run=SimpleNamespace(grid_id=grid_id))


def _builder_to_the_first_nest(monkeypatch):
    """A root, a dormant (spawn-declared) nest d02 and an active nest d03, through the real builder.

    Everything the builder calls is a stand-in except its step reports, which are what these tests read: no
    decode, no device and no case on disk.  The first nest's start state ends the build.
    """

    import woof.core.model as core_model

    d01, d02, d03 = _grid(1, 0), _grid(2, 1), _grid(3, 1)
    exp = SimpleNamespace(name="dormant-ahead", root=d01, domains=(d01, d02, d03), perturbation=None, feedback=0,
                          vertical=None)
    data = SimpleNamespace(source_orography=None, sfcp_to_sfcp=None)
    active = SimpleNamespace(domains=(d01, d03))
    clocks = {gid: SimpleNamespace(spec=SimpleNamespace(grid_id=gid, start_ticks=0)) for gid in (1, 3)}
    empty = SimpleNamespace(nbytes=0)
    estimate = SimpleNamespace(dycore_state_workspace_bytes=0, scratch_arena_bytes=0, workspace_bytes=0)
    prepared_root = SimpleNamespace(grid=None, initial_result=SimpleNamespace(state=SimpleNamespace(physics=None)))

    def stop(*_args, **_kwargs):
        raise _StopAtTheFirstNest

    for target, value in (
            ("woof.ingest.preflight.build_input_catalog", lambda data: object()),
            ("woof.core.model._adapt_experiment_vertical_for_case", lambda e, data, catalog: e),
            ("woof.runtime.forcing_snapshots", lambda *a, **k: {}),
            ("woof.runtime.forcing_schedule", lambda *a, **k: ()),
            ("woof.core.model._forcing_cadence_seconds", lambda catalog: 10800.0),
            ("woof.experiment.pre_spawn_experiment", lambda e: active),
            ("woof.vertical_adaptation.refuse_tree_off_the_experiment_coordinate", lambda *a, **k: None),
            # A real TickClock carries tick_den; the builder reads the declared tree's (dormant d02 included)
            # as the startup tree's tick floor since 801ccab134.
            ("woof.core.clock.resolve_clock", lambda *a, **k: SimpleNamespace(clocks=lambda: clocks, tick_den=1)),
            ("woof.core.clock.build_schedule", lambda *a, **k: None),
            ("woof.core.preflight.estimate_experiment", lambda *a, **k: estimate),
            ("woof.core.state.build_shared_dycore_state_workspace", lambda domains: empty),
            ("woof.core.state.build_shared_scratch_arena", lambda domains: empty),
            ("woof.core.model.uses_modern_rrtmgp_workspace", lambda e: False),
            ("woof.core.model.ModelMemoryLedger", lambda **k: None),
            ("woof.runtime.prepare_root_experiment_case", lambda *a, **k: prepared_root),
            ("woof.ingest.grib.clear_forcing_caches", lambda: None),
            ("woof.ingest.lateral_bc.bind_lateral_boundary_clock", lambda *a: None),
            ("woof.core.cam_ozone.configure_cam_ozone", lambda *a, **k: None),
            ("woof.core.radiation_composition.attach_modern_workspace", lambda *a: None),
            ("woof.ingest.nest_init.initialize_child", stop)):
        monkeypatch.setattr(target, value)

    def build():
        with pytest.raises(_StopAtTheFirstNest):
            core_model.build_experiment(exp, data)

    return exp, build


def test_a_grid_is_numbered_among_the_grids_that_start_with_the_run(monkeypatch):
    # A dormant nest d02 ahead of the active d03: the builder said d03 as "grid 3 of 2", and the page looked its
    # spacing up by that number.
    _, build = _builder_to_the_first_nest(monkeypatch)
    heard = []
    with progress.event_sink(lambda event, **fields: heard.append(fields.get("preparation"))):
        build()
    grids = [(row["event"], row["index"], row["count"], row.get("grid_id"))
             for row in heard if row and row["stage"] == "domain_initialize"]
    assert grids == [("started", 1, 2, 1), ("finished", 1, 2, 1), ("started", 2, 2, 3), ("failed", 2, 2, 3)]


def test_an_unsupervised_run_prints_no_raw_step_records(monkeypatch, tmp_path, capsys):
    # `woof run CONFIG --no-supervise` printed each of the builder's steps to the terminal as a raw
    # GPUWM_PREP_EVENT JSON line: the builder runs in the command that holds the run, and nothing reads its stderr.
    from woof import capabilities, cli, config, runtime
    import woof.case_data as case_data

    exp, build = _builder_to_the_first_nest(monkeypatch)
    path = tmp_path / "dormant-ahead.toml"
    path.write_text('[experiment]\nname = "dormant-ahead"\n', encoding="utf-8")
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(case_data, "load_experiment_case", lambda _path, **_kwargs: (exp, object()))
    monkeypatch.setattr(config, "validate_experiment_preparation", lambda _exp: None)

    def run(*_args, **_kwargs):
        build()
        return SimpleNamespace(wrfout_paths=(), completed_seconds=0.0, nan_free=True)

    monkeypatch.setattr(runtime, "run_experiment", run)
    capsys.readouterr()
    assert cli.main(["run", str(path), "--outdir", str(tmp_path / "out"), "--no-supervise"]) == 0
    said = capsys.readouterr()
    assert "GPUWM_PREP_EVENT" not in said.out + said.err
    assert "dormant-ahead" in said.out


def test_go_says_a_step_heard_in_its_process_as_a_step_not_a_warning(tmp_path, monkeypatch, capsys):
    # The builder's steps reach `woof go` only as the run's own preparation records, and go printed each one as
    # "warning: Start state and boundaries".
    from woof.cli import main
    from test_go_native_launch import allow_launch_resources, emit

    config = emit(tmp_path)
    allow_launch_resources(monkeypatch)
    started = {"schema": progress.PREP_EVENT_SCHEMA, "stage": "domain_initialize",
               "label": "Start state and boundaries", "index": 1, "count": 2, "grid_id": 1}

    def fetch(*_args, events=None, **_kwargs):
        # What a step taken in go's own process sends: the record alone, on the run's relay.
        with runplan._preparation_relay(Observer(events)):
            for event, extra in (("started", {}), ("finished", {"elapsed_seconds": 2.5})):
                progress.emit_event("warning", code="preparation_progress", phase="prepare",
                                    preparation={**started, "event": event, **extra}, message=started["label"])
        raise runplan.StageExitError("prepare", 70)

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    capsys.readouterr()
    # This callback supplies the preparation stream from the fetch itself,
    # which models the whole-cycle door.
    assert main(["go", str(config), "--outdir", str(tmp_path / "run"),
                 "--run-stamp", "off", "--whole-cycle"]) == 1
    said = capsys.readouterr()
    assert "warning: Start state and boundaries" not in said.err
    assert "go: Start state and boundaries\n" in said.out
    assert "go: Start state and boundaries: done (2.5 s)" in said.out


def test_a_grid_step_keeps_its_spacing_when_a_dormant_nest_is_ahead_of_it(tmp_path):
    # The recorded 12 / 3 / 1 km run, its 3 km d02 dormant: the builder on the 1 km d03, the second of two grids.
    run = live_run(tmp_path, "dormant", recorded("nested-prepare.jsonl", through=5),
                   heartbeat="preparing:build-domain-tree")
    # The builder's record for d03, as its step report sends it.
    started = {"schema": progress.PREP_EVENT_SCHEMA, "stage": "domain_initialize", "event": "started",
               "label": "Start state and boundaries", "index": 2, "count": 2, "grid_id": 3}
    engine_records(run, lambda _events: progress.emit_event(
        "warning", code="preparation_progress", phase="prepare", preparation=started, message=started["label"]))
    prep = runs.status(run)["preparation"]
    assert (prep["step"]["index"], prep["step"]["count"], prep["step"]["grid_km"]) == (2, 2, 1.0)


def test_the_time_on_a_step_is_the_steps_own(tmp_path):
    # The line put the stage's time after a grid's step, so a grid begun a second ago read as building for minutes.
    lines = recorded("nested-prepare.jsonl", through=5)
    run = live_run(tmp_path, "timed", lines)
    stage_begun = next(json.loads(line) for line in lines if json.loads(line)["event"] == "stage_started"
                       and json.loads(line)["stage"] == "prepare")["emitted_unix_ms"]
    step = progress.prep_stage("domain_initialize", label="Start state and boundaries", index=2, count=3)
    engine_records(run, lambda _events: step.__enter__())
    opened = [row for row in runs.event_records(run / runs.EVENTS) if row.get("code") == "preparation_progress"][-1]
    now_ms = opened["emitted_unix_ms"] + 4000
    prep = runs.preparation(runs.event_facts(run / runs.EVENTS), None, now_ms=now_ms)
    step.__exit__(None, None, None)
    assert prep["step_seconds"] == 4
    assert prep["stage_seconds"] == (now_ms - stage_begun) // 1000 > 4


# ---------------------------------------------------------------- steps of a `woof go` stage, read while it runs

#: A stage program reporting its steps the way the HRRR nest preparation (woof.native_hierarchy, run as
#: `python -m woof.hrrr_hierarchy_direct`) does: through prep_stage, which reaches its parent only as stderr lines.
#: "wait" leaves a step open and waits for the parent to have said it before it exits, so it passes only if the
#: parent hears a step while the stage is still running; "refuse" stops with a refusal after its steps.
STAGE_PROGRAM = """
import os, sys, time
from woof.progress import prep_stage
with prep_stage("child_initialize", label="Initialize child domains", count=1):
    pass
if sys.argv[1] == "refuse":
    with prep_stage("hierarchy_artifacts", label="Write hierarchy artifacts"):
        pass
    print("refused: the child domain leaves its parent", file=sys.stderr)
    sys.exit(2)
step = prep_stage("hierarchy_artifacts", label="Write hierarchy artifacts")
step.__enter__()
deadline = time.monotonic() + 20
while not os.path.exists(sys.argv[2]):
    if time.monotonic() > deadline:
        sys.exit(3)
    time.sleep(0.05)
os._exit(0)
"""

#: The GFS chain's preparation stage, `python -m woof.source_cli`, reduced to what matters here: its preparation
#: host (woof.prep_output) running the preparer program PREPARER.
PREP_HOST_PROGRAM = """
import sys
from pathlib import Path
from types import SimpleNamespace
from woof import prep_output, source_cli
args = SimpleNamespace(output_root=Path(sys.argv[1]), explain=False)
sys.exit(prep_output.run_preparation(
    args, lambda: source_cli._run_native_adapter([sys.executable, "-c", sys.argv[2], "static"])))
"""

REPO = Path(__file__).resolve().parents[1]


def _importable():
    """The environment a stage program needs to import this tree (stages run with PYTHONSAFEPATH)."""

    return {"PYTHONPATH": os.pathsep.join(filter(None, (str(REPO), os.environ.get("PYTHONPATH"))))}


def test_a_hrrr_nest_stage_s_steps_reach_the_run_page_while_it_runs(tmp_path):
    # Plain `hrrr` runs its nest preparation as a `woof go` stage, which gathered the stage's output with
    # communicate() and read it only once the stage ended: the run page showed no step for the whole preparation.
    from woof import go_cli

    run = live_run(tmp_path, "hrrr", recorded("download.jsonl", through=3))
    heard_file = tmp_path / "heard"
    seen = {}

    class Observer(runplan.RunObserver):
        def warn(self, code, message, **fields):
            super().warn(code, message, **fields)
            record = fields.get("preparation") or {}
            if (record.get("stage"), record.get("event")) == ("hierarchy_artifacts", "started"):
                runs._CACHE.clear()
                seen["page"] = runs.status(run)["preparation"]
                heard_file.write_text("", encoding="utf-8")

    with runplan.EventStream(run / runs.EVENTS, mirror=None) as events:
        go_cli._run_stage("prepare", [sys.executable, "-c", STAGE_PROGRAM, "wait", str(heard_file)],
                          explain=False, observer=runplan._GoObserver(Observer(events)), env=_importable())
    # Said while the stage was still running: the program waited for it before exiting.
    step = seen["page"]["step"]
    assert (step["key"], step["label"]) == ("hierarchy_artifacts", "Write hierarchy artifacts")
    heard = [row["preparation"] for row in runs.event_records(run / runs.EVENTS)
             if row.get("code") == "preparation_progress"]
    assert [(row["stage"], row["event"]) for row in heard] == [
        ("child_initialize", "started"), ("child_initialize", "finished"), ("hierarchy_artifacts", "started")]


def test_a_gfs_preparation_stage_passes_its_preparers_steps_to_the_run(tmp_path):
    # Plain `gfs` prepares in `python -m woof.source_cli`, a `woof go` stage whose preparation host kept each step
    # line its preparer wrote to its own log: nothing of the preparation reached the run page.
    from woof import go_cli

    run = live_run(tmp_path, "gfs", recorded("download.jsonl", through=3))
    with runplan.EventStream(run / runs.EVENTS, mirror=None) as events:
        go_cli._run_stage("prepare", [sys.executable, "-c", PREP_HOST_PROGRAM, str(tmp_path / "prepared"),
                                      PREPARER], explain=False,
                          observer=runplan._GoObserver(runplan.RunObserver(events)), env=_importable())
    heard = [row["preparation"] for row in runs.event_records(run / runs.EVENTS)
             if row.get("code") == "preparation_progress"]
    assert [(row["stage"], row["event"]) for row in heard] == [
        ("root_static", "started"), ("root_static", "finished"), ("root_initialize", "started")]
    runs._CACHE.clear()
    step = runs.status(run)["preparation"]["step"]
    assert (step["key"], step["label"]) == ("root_initialize", "Initialize root forcing states")


def test_a_gfs_forecast_beside_its_preparation_says_how_many_boundary_times_are_ready(tmp_path):
    # `woof go`'s chained GFS preparation said its head only to go's own hooks: a run page never knew the forecast
    # ran beside its preparation, so it said nothing of the boundary times written while the forecast stepped.
    run = live_run(tmp_path, "gfs-chained", recorded("download.jsonl", through=3))
    with runplan.EventStream(run / runs.EVENTS, mirror=None) as events:
        observer = runplan._GoObserver(runplan.RunObserver(events))
        observer.stage_begin(label="prepare", command=["prepare"])
        observer.prepare_head_ready(head_sha256="0" * 64)
        observer.stage_begin(label="forecast", command=["forecast"])
        for done in (1, 2, 3):
            observer.warn(**progress.prep_record_event({
                "schema": progress.PREP_EVENT_SCHEMA, "stage": "root_boundaries", "label": "Boundary times",
                "event": "progress", "index": done, "count": 15}))
        runs._CACHE.clear()
        prep = runs.status(run)["preparation"]
        assert prep["stage"] == "forecast" and prep["chained"]
        assert prep["boundaries"] == {"done": 3, "count": 15}
        # The preparation stage ends: sealed, and the forecast's own line takes over.
        observer.stage_end(label="prepare", exit_code=0, ok=True, elapsed_seconds=1.0, progress=None)
    runs._CACHE.clear()
    assert runs.status(run)["preparation"] is None


def test_a_preparation_host_keeps_its_step_lines_off_a_persons_terminal(tmp_path, monkeypatch, capfd):
    # With no parent reading step records (`woof prep` typed at a terminal) the raw lines stay in the log.
    from woof import prep_output, source_cli

    monkeypatch.delenv(progress.PREP_EVENT_PARENT_ENV, raising=False)
    args = SimpleNamespace(output_root=tmp_path / "prepared", explain=False)
    assert prep_output.run_preparation(
        args, lambda: source_cli._run_native_adapter([sys.executable, "-c", PREPARER, "static"])) == 0
    said = capfd.readouterr()
    assert "GPUWM_PREP_EVENT" not in said.out + said.err
    assert "prep: Prepare root static fields" in said.out


def test_a_failed_stages_tail_is_its_refusal_not_its_step_records(tmp_path, capsys):
    # The replayed tail and the stage_failed diagnostic (what the desktop shows) were the preparer's raw
    # GPUWM_PREP_EVENT lines, with the refusal pushed out of the 8-line diagnostic by a preparer with many steps.
    from woof import go_cli

    failures = []

    class Observer:
        def stage_failed(self, *, label, exit_code, diagnostic):
            failures.append(diagnostic)

    with pytest.raises(go_cli.GoStageFailed) as failed:
        go_cli._run_stage("prepare", [sys.executable, "-c", STAGE_PROGRAM, "refuse"], explain=False,
                          observer=Observer(), env=_importable())
    said = capsys.readouterr().out
    assert "refused: the child domain leaves its parent" in said
    assert "GPUWM_PREP_EVENT" not in said
    assert failures == [failed.value.diagnostic] == ["refused: the child domain leaves its parent"]


def test_go_leaves_step_records_out_of_a_failed_stages_tail_and_keeps_them_in_its_log(tmp_path, monkeypatch,
                                                                                       capsys):
    # `woof go` printed the last 8 lines of a failed stage's stderr, and on a preparation those were raw JSON.
    from woof import source_cli
    from woof.cli import main
    from test_go_native_launch import allow_launch_resources, emit

    config = emit(tmp_path)
    allow_launch_resources(monkeypatch)

    def fetch(*_args, **_kwargs):
        assert source_cli._run_native_adapter([sys.executable, "-c", PREPARER, "static"]) == 0
        raise runplan.StageExitError("prepare", 70)

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(tmp_path / "run"), "--run-stamp", "off"]) == 1
    said = capsys.readouterr()
    assert "GPUWM_PREP_EVENT" not in said.out + said.err
    assert "Details:" in said.err
    log = next((tmp_path / "run").rglob("launch.log")).read_text(encoding="utf-8")
    assert log.count("GPUWM_PREP_EVENT") == 3


def test_a_chained_preparation_runs_in_its_callers_context(tmp_path):
    # `woof go` sends every preparer's output to its launch log through a context variable, and the chained
    # preparation's thread started without it: its preparer ran behind a second host with a second log, and
    # `woof go --explain` said each step twice.
    import io
    from woof import command_output
    from woof.ingest.boundary_stream import run_chained

    streams = (io.StringIO(), io.StringIO())
    with command_output.redirect_adapter_output(*streams):
        prepared, forecast = run_chained(prepared_root=tmp_path / "prepared",
                                         prepare=command_output.ADAPTER_OUTPUT.get,
                                         forecast=lambda head: head)
    assert prepared == streams and forecast is None


# ---------------------------------------------------------------- the words, drawn by the page's own script

SCRIPT = r"""
const input = JSON.parse(await (await import("node:fs/promises")).readFile("input.json", "utf8"));
const { prepLine } = await import("./core.js");
console.log(JSON.stringify(input.statuses.map((st) => prepLine(st, input.words.screens.preparation))));
"""


@pytest.fixture(scope="module")
def node():
    found = shutil.which("node")
    if found is None:
        pytest.skip("Node is not installed")
    return found


def drawn(tmp_path, node, statuses) -> list[list[str]]:
    folder = tmp_path / "page"
    shutil.copytree(JS, folder)
    (folder / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    (folder / "input.json").write_text(json.dumps({"statuses": statuses, "words": Api.copy()}), encoding="utf-8")
    (folder / "t.mjs").write_text(SCRIPT, encoding="utf-8")
    done = subprocess.run([node, "t.mjs"], cwd=folder, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_page_says_the_step_grid_download_and_boundary_times_in_words(tmp_path, node):
    download = {"preparation": {"stage": "fetch", "stage_seconds": 75, "fetch": {
        "files_done": 2, "files_total": 6, "bytes": 836794623, "bytes_total": None}}}
    arco = {"preparation": {"stage": "fetch", "stage_seconds": 30, "fetch": {"times_done": 1, "times_total": 4}}}
    grid = {"preparation": {"stage": "prepare", "stage_seconds": 312, "step_seconds": 12, "phase": "build-domain-tree",
                            "step": {"key": "domain_initialize", "index": 2, "count": 3, "grid_km": 3.0}}}
    # A count heard without its step's start: the stage's time, said beside the stage's name rather than the step.
    untimed = {"preparation": {"stage": "prepare", "stage_seconds": 312, "step_seconds": None, "step": {
        "key": "root_boundaries", "done": 3, "count": 15}}}
    phase = {"preparation": {"stage": "prepare", "stage_seconds": 5, "phase": "build-domain-tree", "compiling": 2}}
    chained = {"preparation": {"stage": "forecast", "chained": True, "boundaries": {"done": 3, "count": 15},
                               "stage_seconds": None}}
    first_step = {"preparation": {"stage": "forecast", "compiling": 12, "stage_seconds": None}}
    assert drawn(tmp_path, node, [download, arco, grid, untimed, phase, chained, first_step]) == [
        ["2 of 6 files", "837 MB so far", "for 1 min 15 s"],
        ["time 1 of 4 read", "for 30 s"],
        ["Start state and boundaries, grid 2 of 3 (3 km)", "for 12 s"],
        ["for 5 min 12 s", "Boundary times 3 of 15"],
        ["for 5 s", "Building the grids", "compiling GPU kernels"],
        ["boundary times 3 of 15 ready"],
        ["compiling GPU kernels"],
    ]


def test_a_steps_time_is_said_right_after_the_step_before_the_kernel_compile(tmp_path, node):
    # "Start state and boundaries, grid 2 of 3 (3 km) · compiling GPU kernels · for 12 s" read as the compile's time.
    grid = {"preparation": {"stage": "prepare", "stage_seconds": 312, "step_seconds": 12, "compiling": 2,
                            "step": {"key": "domain_initialize", "index": 2, "count": 3, "grid_km": 3.0}}}
    named = {"preparation": {"stage": "prepare", "stage_seconds": 40, "step_seconds": 7, "compiling": 1,
                             "step": {"key": "root_static", "label": "Prepare root static fields"}}}
    assert drawn(tmp_path, node, [grid, named]) == [
        ["Start state and boundaries, grid 2 of 3 (3 km)", "for 12 s", "compiling GPU kernels"],
        ["Static fields", "for 7 s", "compiling GPU kernels"],
    ]


def test_a_bare_go_says_a_gfs_preparations_steps_in_the_terminal_and_its_stream(tmp_path, capsys):
    # `woof go` on the GFS chain in a terminal has GoChainEvents as its observer, which had no warn hook: every
    # step the preparer wrote was dropped, the terminal showed only the stage heartbeat and events.jsonl no step.
    from woof import go_cli
    from woof.chain_events import GoChainEvents, read_chain_events

    chain = GoChainEvents()
    chain.open(tmp_path / "events.jsonl")
    try:
        go_cli._run_stage("prepare", [sys.executable, "-c", PREP_HOST_PROGRAM, str(tmp_path / "prepared"),
                                      PREPARER], explain=False, observer=chain, env=_importable())
    finally:
        chain.close()
    said = capsys.readouterr()
    assert "     .. Prepare root static fields\n" in said.out
    assert "     .. Prepare root static fields: done (" in said.out
    assert "     .. Initialize root forcing states (cpu)\n" in said.out
    assert "GPUWM_PREP_EVENT" not in said.out + said.err
    heard = [row["preparation"] for row in read_chain_events(tmp_path / "events.jsonl")
             if row.get("code") == "preparation_progress"]
    assert [(row["stage"], row["event"]) for row in heard] == [
        ("root_static", "started"), ("root_static", "finished"), ("root_initialize", "started")]


def test_a_bare_go_still_says_a_render_warning_in_the_terminal(tmp_path, capsys):
    # A warning that is not a step, once the chain observer has a warn hook: on the stream and on stderr.
    from woof.chain_events import GoChainEvents, read_chain_events

    chain = GoChainEvents()
    chain.open(tmp_path / "events.jsonl")
    chain.warn("render_basemap_missing", "no map assets resolve", render_stage="finalize")
    chain.close()
    assert "warning: no map assets resolve" in capsys.readouterr().err
    [row] = [row for row in read_chain_events(tmp_path / "events.jsonl") if row.get("event") == "warning"]
    assert row["code"] == "render_basemap_missing" and row["render_stage"] == "finalize"
