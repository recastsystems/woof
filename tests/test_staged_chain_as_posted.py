"""The staged chain as posted (A136 L10): mapped sources stream through run-plan and go.

``woof run-plan`` and ``woof go`` run a packaged mapped source through
the staged chain (fetch, ``woof prep``, the prepared forecast).  As posted
(the default), the fetch runs beside the preparation: the fetch writes
``prep-arguments.json`` once the window's start needs and donors are in
(DESIGN A136 2.3 step 2), the preparation starts from it with
``--as-posted`` and waits on each later lead's marker, its seal writes the
input manifest beside the fetched files, and the forecast binds the head.
The preparation door forwards ``--as-posted`` to the mapped engine, and the
single-domain forecast binds a mapped as-posted head on what the head
carries.

CPU only; no device, no source data.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from conftest import requires_cupy
from woof import fetch_as_posted, fetch_routes, runplan, source_cli
from woof.runplan import (EVENTS_FILENAME, EventStream, execute_plan,
                           load_plan, read_events)

from test_runplan import _staged_plan


CYCLE = "2026-08-18T06"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _schedule(out: Path, leads=(0, 1, 2)) -> None:
    _write(out / "posting" / "schedule.json", {
        "schema": "gpuwm.posting-schedule.v1", "source": "icon-eu",
        "cycle": CYCLE, "table_sha256": "7" * 64,
        "start_needs": [{"role": "analysis", "source": "icon-eu", "lead": 0}],
        "leads": [{"lead": lead, "state": "scheduled",
                   "valid_time": f"2026-08-18T{6 + lead:02d}:00:00Z",
                   "expected_at": "2026-08-18T08:45:00Z",
                   "late_at": "2026-08-18T09:45:00Z", "last_answer": None}
                  for lead in leads]})


def _handoff(out: Path, argv, *, posted=True, unbound=()) -> None:
    document = {
        "schema": fetch_routes.PREP_ARGUMENTS_SCHEMA,
        "source": "icon-eu", "prep_source": "icon-eu", "cycle": CYCLE,
        "argv": list(argv), "unbound_supplement_roles": list(unbound),
        "member": None, "member_set": None,
    }
    if posted:
        document.update(as_posted=True, posting=str(out / "posting"))
    _write(out / fetch_routes.PREP_ARGUMENTS_NAME, document)


def _run(tmp_path, monkeypatch, *, fetch, prep, forecast=None):
    """Drive the staged chain with the fetch, preparation and forecast observed."""

    import woof.go_cli as go_cli
    import woof.stage_cli as stage_cli

    staged = []

    def resolve_bundle(prepared_root):
        return {"document": Path(prepared_root) / "proof.json",
                "schema": "probe", "source": "icon-eu",
                "layout": "single", "domains": 1, "payload": {}}

    def sim_command(bundle, **kw):
        staged.append(("sim_command", dict(kw)))
        return ["python", "-m", "runner", "--source", bundle["source"],
                "--outdir", str(kw["outdir"])]

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    monkeypatch.setattr(runplan, "_run_prep", prep)
    monkeypatch.setattr(runplan, "_staged_forecast", forecast or (
        lambda argv, *, layout, observer: staged.append(("forecast", argv))))
    monkeypatch.setattr(stage_cli, "resolve_bundle", resolve_bundle)
    monkeypatch.setattr(stage_cli, "sim_command", sim_command)
    monkeypatch.setattr(go_cli, "_render_stage",
                        lambda plan, **kw: staged.append(("render", {})))
    geog = tmp_path / "GEOG"
    geog.mkdir(exist_ok=True)
    plan = load_plan(_staged_plan(tmp_path, tmp_path / "run",
                                  run_options={"geog_root": str(geog),
                                               "render_products": "none"}))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        code = execute_plan(plan, events=events)
    return code, staged, read_events(plan.run_dir / EVENTS_FILENAME), plan


@requires_cupy
def test_the_staged_chain_prepares_beside_an_as_posted_fetch(
        tmp_path, monkeypatch):
    """The preparation starts from the handoff the fetch writes before its
    later leads move, with --as-posted and the seal's manifest beside the
    fetched files, while the fetch is still running."""

    argv = ["--source", "icon-eu", "--input-list", "inputs.txt",
            "--supplement", "surface=invariant.grib2",
            "--author-input-manifest", "inputs.json"]
    preparing = threading.Event()
    seen = {}

    def fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        _schedule(out)
        _handoff(out, argv)
        # The later leads move only once the preparation has started.
        seen["overlap"] = preparing.wait(20)
        seen["fetch_ended"] = time.monotonic()
        return {"source": "icon-eu"}

    def prep(arguments):
        seen["prepare"] = list(arguments)
        seen["prep_started"] = time.monotonic()
        preparing.set()
        prep_root = Path(arguments[arguments.index("--output-root") + 1])
        prep_root.mkdir(parents=True, exist_ok=True)
        (prep_root / "proof.json").write_text("{}", encoding="utf-8")

    code, staged, events, plan = _run(tmp_path, monkeypatch, fetch=fetch,
                                      prep=prep)
    assert code == 0
    assert seen["overlap"], "the preparation did not start beside the fetch"
    prepare = seen["prepare"]
    data = Path(prepare[prepare.index("--as-posted") + 1]).parent
    assert prepare[prepare.index("--as-posted") + 1] == str(data / "posting")
    manifest = Path(prepare[prepare.index("--author-input-manifest") + 1])
    assert manifest.parent == data
    assert manifest == runplan._posted_manifest_path(
        data, plan.run_dir / "chain" / "prep")
    assert manifest.name != "inputs.json"
    stages = [record["stage"] for record in events
              if record["event"] == "stage_started"]
    assert stages[:3] == ["fetch", "prepare", "forecast"]
    fetched = next(record for record in events
                   if record["event"] == "stage_finished"
                   and record["stage"] == "fetch")
    assert fetched["fetch"]["as_posted"] is True
    assert "posting_schedule" in [record["event"] for record in events]


@requires_cupy
def test_a_window_the_fetch_finds_whole_is_prepared_whole(
        tmp_path, monkeypatch):
    """A fetch that schedules no wait (its window here whole) ends first,
    and the preparation binds the whole window as before."""

    argv = ["--source", "icon-eu", "--input-list", "inputs.txt",
            "--author-input-manifest", "inputs.json"]
    seen = {}

    def fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        _handoff(out, argv, posted=False)
        return {"source": "icon-eu"}

    def prep(arguments):
        seen["prepare"] = list(arguments)
        prep_root = Path(arguments[arguments.index("--output-root") + 1])
        prep_root.mkdir(parents=True, exist_ok=True)
        (prep_root / "proof.json").write_text("{}", encoding="utf-8")

    code, _staged, _events, plan = _run(tmp_path, monkeypatch, fetch=fetch,
                                        prep=prep)
    assert code == 0
    assert "--as-posted" not in seen["prepare"]
    assert seen["prepare"][seen["prepare"].index(
        "--author-input-manifest") + 1] == str(
            plan.run_dir / "chain" / "inputs.json")


@requires_cupy
def test_a_preparation_that_fails_beside_the_fetch_stops_it(
        tmp_path, monkeypatch):
    """The run is over and so is its download: the fetch in this process
    ends at its next wait, and failed.json tells a preparation still
    waiting on a marker that no more will come."""

    argv = ["--source", "icon-eu", "--input-list", "inputs.txt",
            "--author-input-manifest", "inputs.json"]
    seen = {}

    def fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        seen["out"] = out
        _schedule(out)
        _handoff(out, argv)
        # The loop's wait, ended by the chain's stop (PostingLoop._pause).
        seen["stopped"] = fetch_as_posted.stop_event(out).wait(20)
        raise fetch_as_posted.FetchStopped("stopped")

    def prep(arguments):
        raise runplan.StageExitError("prepare", 1)

    code, _staged, events, _plan = _run(tmp_path, monkeypatch, fetch=fetch,
                                        prep=prep)
    assert code == 1
    assert seen["stopped"], "the fetch was not stopped"
    failed = json.loads((seen["out"] / "posting" / "failed.json").read_text())
    assert failed["code"] == "fetch_stopped"
    final = [record for record in events if record["event"] == "failed"]
    assert final and final[-1]["stage"] == "prepare"


@requires_cupy
def test_a_posted_handoff_naming_no_manifest_is_refused_and_its_fetch_stopped(
        tmp_path, monkeypatch):
    """Without a manifest path the preparation could not wait on the leads
    and would read a window still posting as whole; the run is refused
    before it prepares, and the fetch is not left downloading."""

    argv = ["--source", "icon-eu", "--input-list", "inputs.txt"]
    seen = {}

    def fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        _schedule(out)
        _handoff(out, argv)
        seen["stopped"] = fetch_as_posted.stop_event(out).wait(20)
        raise fetch_as_posted.FetchStopped("stopped")

    def prep(arguments):  # pragma: no cover - never reached
        raise AssertionError("prepared a window still posting as whole")

    code, _staged, events, _plan = _run(tmp_path, monkeypatch, fetch=fetch,
                                        prep=prep)
    assert code == 1
    assert seen["stopped"], "the fetch was not stopped"
    final = [record for record in events if record["event"] == "failed"]
    assert "names no --author-input-manifest" in final[-1]["message"]


@requires_cupy
@pytest.mark.parametrize("broken, message", [
    ("non_string_argument", "Preparation arguments must be strings"),
    ("unbound_supplement", "leaves the supplement role(s) ['surface'] unbound"),
])
def test_a_refused_posted_handoff_stops_its_fetch_and_posting_relay(
        tmp_path, monkeypatch, broken, message):
    """Validation after the handoff must end both background workers before
    the run returns, without reaching preparation or fetching more leads."""

    from woof import chain_events

    argv = ["--source", "icon-eu", "--input-list", "inputs.txt",
            "--author-input-manifest", "inputs.json"]
    if broken == "non_string_argument":
        argv[3] = 7
    seen = {}
    finished = threading.Event()
    posting_threads = []
    relays = []

    class ObservedRelay(chain_events.HostedPostingRelay):
        def start(self, **kwargs):
            super().start(**kwargs)
            posting_threads.append(self._relay._posting_thread)
            relays.append(self)

    monkeypatch.setattr(chain_events, "HostedPostingRelay", ObservedRelay)

    def fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        seen["stop"] = fetch_as_posted.stop_event(out)
        _schedule(out)
        _handoff(out, argv, unbound=(
            ("surface",) if broken == "unbound_supplement" else ()))
        try:
            seen["stopped"] = seen["stop"].wait(20)
            raise fetch_as_posted.FetchStopped("stopped")
        finally:
            finished.set()

    def prep(arguments):  # pragma: no cover - never reached
        raise AssertionError("prepared a refused handoff")

    try:
        code, _staged, events, _plan = _run(
            tmp_path, monkeypatch, fetch=fetch, prep=prep)
        assert code == 1
        assert seen.get("stopped"), "the refused handoff left the fetch running"
        assert finished.is_set()
        assert posting_threads and all(
            not thread.is_alive() for thread in posting_threads)
        final = [record for record in events if record["event"] == "failed"]
        assert message in final[-1]["message"]
    finally:
        # Keep the failure reproduction from leaving its background work
        # behind on a revision without the cleanup.
        if "stop" in seen:
            seen["stop"].set()
            finished.wait(5)
        for relay in relays:
            relay.stop()


@requires_cupy
def test_a_fetch_that_fails_before_its_handoff_is_the_chains_failure(
        tmp_path, monkeypatch):
    def fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        _schedule(out)
        raise runplan.StageExitError("fetch", 2)

    def prep(arguments):  # pragma: no cover - never reached
        raise AssertionError("prepared without a handoff")

    code, _staged, events, _plan = _run(tmp_path, monkeypatch, fetch=fetch,
                                        prep=prep)
    assert code == 1
    final = [record for record in events if record["event"] == "failed"]
    assert final[-1]["stage"] == "fetch"


def test_the_chain_names_what_keeps_it_whole_window_first():
    from types import SimpleNamespace

    one = SimpleNamespace(domains=(object(),))
    tree = SimpleNamespace(domains=(object(), object()))
    assert runplan._staged_beside_refusal(
        {"source": "hrrr-prs"}, one) is None
    assert runplan._staged_beside_refusal(
        {"source": "rap"}, one) is None
    assert "whole cycle" in runplan._staged_beside_refusal(
        {"source": "hrrr-prs", "as_posted": False}, one)
    # A domain tree prepares as posted too: its head prepares every child
    # under the input plan (WOOF run_9498bc66b5fdd08c waited 14.5 min in
    # fetch for the 06Z cycle's last lead before this guard was retired).
    assert runplan._staged_beside_refusal(
        {"source": "hrrr-prs"}, tree) is None
    assert "member" in runplan._staged_beside_refusal(
        {"source": "gefs"}, one)
    assert "normalizes" in runplan._staged_beside_refusal(
        {"source": "icon-global"}, one)
    # A separate initial analysis (the HRRR recipe's initial_source) is
    # checked against the whole window, so the chain fetches it whole first
    # instead of reaching the preparation's as-posted refusal.
    assert "separate initial analysis" in runplan._staged_beside_refusal(
        {"source": "rap-native"}, one, separate_initial=True)
    assert runplan._staged_beside_refusal(
        {"source": "rap-native"}, one, separate_initial=False) is None


# ---------------------------------------------------------------------------
# The fetch writes the preparation's handoff first, saying where it posts
# ---------------------------------------------------------------------------

def test_the_handoff_says_the_window_is_fetched_as_posted(tmp_path):
    plan = fetch_routes.resolve_request(
        "hrrr-prs", cycle=__import__("datetime").datetime(2026, 9, 30, 12),
        hours=3, cadence=1, start_hour=0, host=None, member=None,
        out=tmp_path)
    fetch_routes.write_handoff(plan, tmp_path, posting=tmp_path / "posting")
    document = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert document["as_posted"] is True
    assert document["posting"] == str((tmp_path / "posting").resolve())
    (tmp_path / "whole").mkdir()
    fetch_routes.write_handoff(plan, tmp_path / "whole")
    whole = json.loads((tmp_path / "whole" /
                        fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert "as_posted" not in whole and "posting" not in whole


# ---------------------------------------------------------------------------
# The preparation door forwards --as-posted to the mapped engine
# ---------------------------------------------------------------------------

def _mapped_args(tmp_path, **extra):
    argv = ["--source", "mapped", "--source-format", "grib2",
            "--composition", str(tmp_path / "c.json"),
            "--mapping", str(tmp_path / "m.json"),
            "--input-list", str(tmp_path / "inputs.txt"),
            "--supplement", f"surface={tmp_path / 's.grib2'}",
            "--provenance", f"surface_provenance={tmp_path / 'p.json'}",
            "--wps-namelist", str(tmp_path / "n.wps"),
            "--experiment-config", str(tmp_path / "e.toml"),
            "--geog-root", str(tmp_path / "geog"),
            "--output-root", str(tmp_path / "prepared")]
    for flag, value in extra.items():
        argv += [flag] if value is True else [flag, str(value)]
    return source_cli._parser().parse_args(argv)


def test_the_mapped_door_forwards_as_posted_without_a_manifest_digest(
        tmp_path):
    args = _mapped_args(tmp_path, **{
        "--as-posted": tmp_path / "posting",
        "--author-input-manifest": tmp_path / "inputs-run.json"})
    assert source_cli._required_mapped_args(args) == []
    # What the dispatch does as posted: the seal writes the manifest there.
    args.source_sha256s = Path(args.author_input_manifest).resolve()
    args.source_sha256s_sha256 = None
    command = source_cli._mapped_command(args)
    assert command[command.index("--as-posted") + 1] == str(
        tmp_path / "posting")
    assert command[command.index("--input-manifest") + 1] == str(
        (tmp_path / "inputs-run.json").resolve())
    assert "--input-manifest-sha256" not in command


@pytest.mark.parametrize("extra, said", [
    ({"--source-manifest": "m.json", "--source-manifest-sha256": "0" * 64},
     "takes no --source-manifest pair"),
    ({}, "needs --author-input-manifest PATH"),
    ({"--author-input-manifest": "i.json", "--author-only": True},
     "--author-only authors a manifest of the whole window"),
])
def test_the_mapped_door_refuses_what_as_posted_cannot_mean(
        tmp_path, extra, said):
    args = _mapped_args(tmp_path, **{"--as-posted": tmp_path / "posting",
                                     **extra})
    assert any(said in error for error in source_cli._required_mapped_args(args))


def test_a_source_that_normalizes_its_inputs_whole_does_not_prepare_as_posted():
    assert source_cli.prepares_as_posted("hrrr-prs")
    assert source_cli.prepares_as_posted("rap")
    refusal = source_cli.as_posted_refusal("icon-d2")
    assert refusal is not None and "normalizes every input file" in refusal
    assert not source_cli.prepares_as_posted("icon-global")


def test_the_sources_that_prepare_as_they_post_are_the_rolling_rows():
    # The CHANGELOG's as-posted row names these doors' sources; the rest say
    # why they wait for the whole window, and in their row's own words.
    for source in ("gfs", "hrrr", "hrrr-prs", "rap", "rrfs", "icon-eu", "gem-gdps"):
        assert source_cli.prepares_as_posted(source), source
    assert ("does not post lead by lead (its posting shape is whole_cycle)"
            in source_cli.as_posted_refusal("aifs"))
    assert ("its posting shape is donor_gated"
            in source_cli.as_posted_refusal("aigfs"))
    assert ("declares no posting row"
            in source_cli.as_posted_refusal("era5-l137"))
    assert ("prepares a whole fetched window"
            in source_cli.as_posted_refusal("era5"))


# ---------------------------------------------------------------------------
# The forecast binds a mapped as-posted head
# ---------------------------------------------------------------------------

def test_a_mapped_as_posted_head_binds_its_plan():
    from woof import prepared_single_domain_forecast as forecast
    from woof.ingest.boundary_stream import (
        as_posted_placeholder, input_plan_sha256)

    plan = {"manifest": {"schema": "x"}, "route_table_sha256": "1" * 64}
    digest = input_plan_sha256(plan)
    assert forecast._as_posted_head_binding(
        "hrrr-prs", {"input_plan": plan, "input_plan_sha256": digest},
        None) == as_posted_placeholder(digest)


def test_a_planned_mapped_manifest_row_waits_for_the_seal_only_as_posted():
    from woof import prepared_single_domain_forecast as forecast

    manifest = {
        "schema": forecast._SOURCE_SCHEMA["hrrr-prs"],
        "mapping_sha256": "1" * 64, "composition_sha256": "2" * 64,
        "primary_files": [
            {"path": "a.grib2", "bytes": 10, "sha256": "3" * 64},
            {"path": "b.grib2", "bytes": None, "sha256": None}],
        "supplements": {"surface": [
            {"path": "a.grib2", "bytes": 10, "sha256": "3" * 64}]},
        "provenance": {"p": {"path": "p.json", "bytes": 5,
                             "sha256": "4" * 64}},
        "decoders": {},
    }
    files = forecast._mapped_composition_manifest_file_specs(
        "hrrr-prs", manifest, planned=True)
    assert files["primary[1]"]["sha256"] is None
    with pytest.raises(ValueError, match="unsafe path or byte count"):
        forecast._mapped_composition_manifest_file_specs("hrrr-prs", manifest)


def test_a_mapped_as_posted_head_is_held_to_its_plans_decoders(tmp_path):
    from woof import prepared_single_domain_forecast as forecast

    config = tmp_path / "e.toml"
    namelist = tmp_path / "n.wps"
    config.write_text("x", encoding="utf-8")
    namelist.write_text("y", encoding="utf-8")

    def receipt(path):
        return {"path": path.name, "bytes": path.stat().st_size,
                "sha256": forecast._sha256(path)}

    engine = {"path": "/opt/engine", "bytes": 9, "sha256": "5" * 64}
    proof = {"execution_inputs": {
        "experiment_config": receipt(config),
        "wps_namelist": receipt(namelist),
        "decoders": {"engine": engine}}, "preprocessing": {"backend": "cpu"}}

    def bind(declared):
        return forecast._posted_mapped_head_authority(
            proof=proof, manifest={"decoders": declared},
            placeholder="as-posted:" + "6" * 64,
            experiment_config=config, wps_namelist=namelist,
            source="hrrr-prs", member_manifest=False, evidence_paths={},
            authority_sha256={"mapping": "1" * 64, "composition": "2" * 64})

    _paths, authority, _member = bind({"engine": dict(engine)})
    assert authority["receipt_content_sha256"] == "as-posted:" + "6" * 64
    assert authority["decoder_sha256"] == {"engine": "5" * 64}
    with pytest.raises(ValueError, match="decoder manifest differs"):
        bind({"engine": {**engine, "sha256": "7" * 64}})


@requires_cupy
def test_a_lead_past_its_budget_ends_the_staged_chain_with_75(
        tmp_path, monkeypatch):
    """The fetch writes source_behind and exits 75, the preparation ends on
    that record in its own process; the run says source_behind with the
    lead and exits 75 (DESIGN A136 3.6), not 1 for a failed stage."""

    argv = ["--source", "icon-eu", "--input-list", "inputs.txt",
            "--author-input-manifest", "inputs.json"]
    gate = threading.Event()

    def fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        _schedule(out)
        _handoff(out, argv)
        assert gate.wait(20)
        _write(out / "posting" / "failed.json", {
            "code": "source_behind", "source": "icon-eu", "cycle": CYCLE,
            "lead": 2, "valid_time": "2026-08-18T08:00:00Z",
            "expected_at": "2026-08-18T08:45:00Z",
            "late_at": "2026-08-18T09:45:00Z", "late_after_minutes": 60,
            "heard": True, "last_answer": "not_posted",
            "message": "icon-eu f002 has not posted"})
        raise runplan.StageExitError("fetch", 75)

    def prep(arguments):
        gate.set()
        deadline = time.monotonic() + 20
        posting = Path(arguments[arguments.index("--as-posted") + 1])
        while not (posting / "failed.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        raise runplan.StageExitError("prepare", 1)

    code, _staged, events, _plan = _run(tmp_path, monkeypatch, fetch=fetch,
                                        prep=prep)
    assert code == 75
    behind = [record for record in events if record["event"] == "source_behind"]
    assert behind and behind[-1]["lead"] == 2
    assert behind[-1]["last_answer"] == "not_posted"
    final = [record for record in events if record["event"] == "failed"]
    assert final[-1]["error_class"] == "SourceBehind"
    assert final[-1]["exit_code"] == 75


@requires_cupy
@pytest.mark.parametrize("later_fetch", ["success", "source_behind"])
@pytest.mark.parametrize("first_failure", ["forecast"])
def test_a_forecast_failure_before_a_later_lead_keeps_its_failure(
        tmp_path, monkeypatch, later_fetch, first_failure):
    """Waiting for the real producer's seal must not reorder failures.

    The forecast callback fails while only the head and first interval
    exist. The fetch answers a later lead only after ``run_chained``
    enters its hold for that failure, then preparation seals or fails.
    A later posting timeout remains a second fact, never the run's cause.
    """

    from woof import stage_cli
    from woof.ingest import boundary_stream
    from test_boundary_stream import (
        PROOF_HEAD, _chained_tree, _frames, _snapshots, _times)

    argv = ["--source", "icon-eu", "--input-list", "inputs.txt",
            "--author-input-manifest", "inputs.json"]
    held = threading.Event()
    fetch_finished = threading.Event()
    seen = {}
    real_hold = boundary_stream._stop_for_failed_forecast
    message = ("forecast became unstable before the later lead"
               if first_failure == "forecast"
               else "prepared head binding failed before the later lead")
    timeout_message = "icon-eu f002 has not posted"

    def hold(root, worker, error):
        # The real chain calls this only after its forecast callback raised.
        seen["held"] = time.monotonic()
        held.set()
        return real_hold(root, worker, error)

    def fetch(arguments, run_dir, **kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        _schedule(out)
        _handoff(out, argv)
        assert held.wait(20), "the forecast failure did not reach its seal hold"
        seen["later_fetch"] = time.monotonic()
        try:
            if later_fetch == "success":
                for lead in (0, 1, 2):
                    _write(out / "posting" / f"f{lead:03d}.json", {})
                return {"source": "icon-eu"}
            if later_fetch == "source_behind":
                _write(out / "posting" / "failed.json", {
                    "code": "source_behind", "source": "icon-eu",
                    "cycle": CYCLE, "lead": 2,
                    "valid_time": "2026-08-18T08:00:00Z",
                    "expected_at": "2026-08-18T08:45:00Z",
                    "late_at": "2026-08-18T09:45:00Z",
                    "late_after_minutes": 60,
                    "heard": True, "last_answer": "not_posted",
                    "message": timeout_message})
                raise runplan.StageExitError("fetch", 75)
            raise RuntimeError("later fetch failed after the forecast")
        finally:
            fetch_finished.set()

    def prep(arguments):
        root = Path(arguments[arguments.index("--output-root") + 1])
        root.parent.mkdir(parents=True, exist_ok=True)
        snapshots = _snapshots(4)
        writer, _ = _chained_tree(root.parent, snapshots,
                                  name=root.name, stop_after=0)
        assert fetch_finished.wait(20), "the later fetch did not answer"
        if later_fetch == "success":
            frames, times = _frames(snapshots), _times(len(snapshots))
            for index in (1, 2):
                writer.write_segment(index, frames.interval(index, times))
            receipt = writer.seal_cache()
            writer.publish({**PROOF_HEAD,
                "prepared_cache": {"content_sha256": receipt["content_sha256"]},
                "boundary_stream": writer.boundary_stream_proof()})
            return None
        error = runplan.StageExitError("prepare", 1)
        writer.fail(error)
        raise error

    def forecast(arguments, *, layout, observer):
        assert first_failure == "forecast", "forecast ran after its binding failed"
        seen["first_failure"] = time.monotonic()
        raise FloatingPointError(message)

    def resolve_head(root, digest):
        # This is a real digest-bound generic test head. Its injected
        # consumer leaves source-specific validation outside this test.
        if first_failure == "head_binding":
            seen["first_failure"] = time.monotonic()
            raise ValueError(message)
        return {"document": Path(root) / "boundary-stream" / "head.json",
                "root": Path(root), "schema": "probe", "source": "icon-eu",
                "layout": "single", "domains": 1, "payload": {}}

    monkeypatch.setattr(boundary_stream, "_stop_for_failed_forecast", hold)
    monkeypatch.setattr(stage_cli, "resolve_head_bundle", resolve_head)
    code, _staged, events, _plan = _run(
        tmp_path, monkeypatch, fetch=fetch, prep=prep, forecast=forecast)
    assert seen["first_failure"] <= seen["held"] < seen["later_fetch"]
    assert code == 1
    final = [record for record in events if record["event"] == "failed"][-1]
    assert final["error_class"] == (
        "FloatingPointError" if first_failure == "forecast" else "ValueError")
    assert final["message"] == message
    assert final["exit_code"] != 75
    if later_fetch == "source_behind":
        secondary = final["secondary_source_behind"]
        assert secondary["lead"] == 2
        assert secondary["message"] == timeout_message
        warnings = [record for record in events
                    if record["event"] == "warning"
                    and record.get("code") == "secondary_source_behind"]
        assert warnings and timeout_message in warnings[-1]["message"]
    else:
        assert "secondary_source_behind" not in final
