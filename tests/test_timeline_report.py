"""Real file adapters, CLI artifact checks, and interval arithmetic."""
import json
import subprocess
import sys

import pytest

from woof.timeline.report import Timeline, timestamp, union_seconds, waterfall


def event(stage, phase, at, *, card=None, run="run", mono=None, host="host", attrs=None, rental="rental"):
    return {"v": 1, "t_utc": at, "t_mono": at if mono is None else mono,
            "host": host, "rental_id": rental, "run_id": run, "product": "regional",
            "stage": stage, "phase": phase, "card": card, "pid": 7, "attrs": attrs or {}}


def read(tmp_path, rows):
    path = tmp_path / "ledger.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return Timeline().read(path), path


def test_two_card_union_and_allocated_idle_card(tmp_path):
    timeline, _ = read(tmp_path, [event("rental.requested", "mark", 0),
        event("forecast.integrate", "begin", 10, card=0, attrs={"card_busy": True}),
        event("forecast.integrate", "begin", 30, card=1, attrs={"card_busy": True}),
        event("forecast.integrate", "end", 60, card=0),
        event("forecast.integrate", "end", 80, card=1),
        event("rental.terminated", "mark", 100)])
    report = timeline.report(cards=4)
    assert report["ff"] == .7
    assert report["integrate_union_seconds"] == 70
    assert report["card_busy_fraction"] == .25
    assert report["integration_card_occupancy_fraction"] == .25
    assert report["request_to_first_step_seconds"] == 10
    assert report["tail_seconds"] == 20
    assert report["absolute_rental_wall_seconds"] == 100


def test_proxy_never_becomes_rental_ff_and_png_is_labelled(tmp_path, monkeypatch):
    timeline, _ = read(tmp_path, [event("forecast.integrate", "begin", 10, card=0, rental=None),
                                event("forecast.integrate", "end", 80, card=0, rental=None)])
    timeline.boundary("proxy.start", 0, "measured command start")
    timeline.boundary("proxy.end", 100, "last artifact")
    report = timeline.report(cards=1)
    assert report["mode"] == "proxy"
    assert report["proxy_ff"] == .7
    assert report["ff"] is None
    assert report["absolute_rental_wall_seconds"] is None
    assert report["request_to_first_step_seconds"] is None
    import matplotlib.axes
    titles = []
    original = matplotlib.axes.Axes.set_title
    def capture(self, text, *args, **kwargs):
        titles.append(text)
        return original(self, text, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", capture)
    path = tmp_path / "proxy.png"
    waterfall(report, path)
    assert path.read_bytes().startswith(b"\x89PNG")
    assert any("PROXY FF" in text for text in titles)


def test_protocol_proxy_marks_are_consumed_without_cli_boundary_overrides(tmp_path):
    timeline, _ = read(tmp_path, [event("worker.claim", "mark", 0, attrs={"basis": "bench proxy envelope"}, rental=None),
        event("forecast.integrate", "begin", 10, rental=None),
        event("forecast.integrate", "end", 80, rental=None),
        event("finish", "mark", 100, attrs={"basis": "bench proxy envelope"}, rental=None)])
    report = timeline.report()
    assert report["mode"] == "proxy"
    assert report["proxy_ff"] == .7
    assert report["ff"] is None


def test_append_order_wall_adjustment_and_host_clocks(tmp_path):
    rows = [event("forecast.integrate", "end", 50, card=0, mono=110),
            event("forecast.integrate", "begin", 10, card=0, mono=100),
            event("forecast.integrate", "end", 40, card=1, mono=5, host="other"),
            event("forecast.integrate", "begin", 30, card=1, mono=0, host="other"),
            event("rental.requested", "mark", 0), event("rental.terminated", "mark", 100)]
    report = read(tmp_path, rows)[0].report(cards=2)
    assert report["integrate_union_seconds"] == 15
    assert any("UTC adjusted" in message for message in report["warnings"])


def test_missing_end_not_invented_or_closed_by_parser_time(tmp_path):
    report = read(tmp_path, [event("forecast.integrate", "begin", 10),
                            event("rental.requested", "mark", 0)])[0].report()
    assert report["ff"] is None
    assert report["wall_seconds"] is None
    assert any("unmatched begin" in warning for warning in report["warnings"])


def test_shell_emitter_pid_changes_pair_only_unambiguous_context(tmp_path):
    rows = [event("boot.install", "begin", 10),
            {**event("boot.install", "end", 30), "pid": 8}]
    report = read(tmp_path, rows)[0].report()
    assert report["intervals"][0]["end"] - report["intervals"][0]["start"] == 20
    rows.insert(1, {**event("boot.install", "begin", 15), "pid": 9})
    report = read(tmp_path, rows)[0].report()
    assert report["intervals"] == []


def test_legacy_steps_use_all_domains_last_end_not_last_root(tmp_path):
    rows = [{"event": "step", "domain": 1, "step": 1, "step_wall_seconds": 2, "emitted_unix_ms": 12000},
            {"event": "step", "domain": 1, "step": 2, "step_wall_seconds": 2, "emitted_unix_ms": 20000},
            {"event": "step", "domain": 3, "step": 9, "step_wall_seconds": 1, "emitted_unix_ms": 25000},
            {"event": "output_written", "valid_time": "2099-01-01_00:00:00", "emitted_unix_ms": 30000, "durable_after_seconds": 5}]
    timeline = read(tmp_path, rows)[0]
    timeline.boundary("proxy.start", 0, "command start")
    report = timeline.report()
    assert report["integrate_union_seconds"] == 15
    assert report["proxy_ff"] == .5
    assert report["tail_seconds"] == 5
    assert report["card_busy_fraction"] is None


def test_legacy_engine_stage_not_mistaken_for_integration(tmp_path):
    rows = [{"event": "stage_started", "stage": "forecast", "emitted_unix_ms": 10000},
            {"event": "stage_finished", "stage": "forecast", "emitted_unix_ms": 80000,
             "started_unix_ms": 10000, "finished_unix_ms": 80000},
            {"event": "model_progress", "outer_step": 100, "wall_seconds": 65, "emitted_unix_ms": 78000}]
    timeline = read(tmp_path, rows)[0]
    timeline.boundary("proxy.start", 0, "start")
    timeline.boundary("proxy.end", 100, "artifact")
    report = timeline.report()
    assert report["proxy_ff"] is None
    assert report["legacy_pipeline_fraction_upper_bound"] == .7


def test_multiple_rentals_refuse_then_select(tmp_path):
    rows = [event("rental.requested", "mark", 0), event("rental.terminated", "mark", 100),
            event("rental.requested", "mark", 150, rental="second"),
            event("rental.terminated", "mark", 200, rental="second")]
    timeline = read(tmp_path, rows)[0]
    with pytest.raises(ValueError, match="multiple rentals"):
        timeline.report()
    assert timeline.report(rental_id="second")["wall_seconds"] == 50


def test_legacy_step_selection_respects_product_and_rental(tmp_path):
    rows = [{"event": "step", "rental_id": "first", "product": "regional",
             "step_wall_seconds": 5, "emitted_unix_ms": 10000},
            {"event": "step", "rental_id": "other", "product": "hex",
             "step_wall_seconds": 5, "emitted_unix_ms": 90000}]
    timeline = read(tmp_path, rows)[0]
    timeline.boundary("proxy.start", 0, "start")
    timeline.boundary("proxy.end", 100, "last artifact")
    assert timeline.report(rental_id="first", product="regional")["integrate_union_seconds"] == 5


def test_site_launch_denominator_is_explicitly_legacy(tmp_path):
    timeline = Timeline()
    timeline.receipt({"id": "run", "boxes": [{"launched": "2026-01-01T00:00:00Z", "ended": "2026-01-01T00:01:40Z"}],
                      "attempts": [{"forecastWallSeconds": 80}]}, "cost.json:1")
    report = timeline.report()
    assert report["mode"] == "legacy_rental"
    assert report["ff"] is None
    assert report["legacy_ff"] is None  # untimed work is not located integration
    assert report["legacy_rental_wall_seconds"] == 100
    assert report["unplaced"][0]["seconds"] == 80


def test_union_clips_and_canonical_replaces_legacy_view(tmp_path):
    rows = [event("rental.requested", "mark", 10), event("rental.terminated", "mark", 90),
            event("forecast.integrate", "begin", 0, card=0), event("forecast.integrate", "end", 100, card=0),
            {"event": "step", "run_id": "run", "card": 0, "step_wall_seconds": 5, "emitted_unix_ms": 90000}]
    report = read(tmp_path, rows)[0].report(cards=1)
    assert report["ff"] == 1
    assert len([r for r in report["intervals"] if r["stage"] == "forecast.integrate"]) == 1


def test_invalid_json_and_schema_refuse_with_evidence(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"v":')
    with pytest.raises(ValueError, match="broken.jsonl:1"):
        Timeline().read(path)
    row = event("forecast.integrate", "begin", 10)
    row["extra"] = 1
    with pytest.raises(ValueError, match="invalid C1 keys"):
        read(tmp_path, [row])
    with pytest.raises(ValueError, match="unsupported ledger version"):
        read(tmp_path, [{**event("forecast.integrate", "begin", 10), "v": 2}])
    assert timestamp("2026-01-01T00:00:00") is None
    assert union_seconds([(0, 10), (5, 15), (0, 10)]) == 15


def test_compressed_legacy_receipt_and_metadata_inference(tmp_path):
    import gzip
    path = tmp_path / "timing.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"schema": "gpuwm.timeline.sources.v1", "boundaries": {
            "proxy.start": {"at": 0}, "proxy.end": {"at": 100}}}, stream, indent=2)
    timeline = Timeline().read(path)
    timeline.receipt(event("forecast.integrate", "begin", 10, rental=None), "emitter:1")
    timeline.receipt(event("forecast.integrate", "end", 80, rental=None), "emitter:2")
    assert timeline.report()["product"] == "regional"
    assert timeline.report()["proxy_ff"] == .7
    for name in ("worker.env", ".env.production", "andrew_accessKeys.csv", ".cdsapirc", "auth.json"):
        with pytest.raises(ValueError, match="credential files"):
            Timeline().read(tmp_path / name)


def test_real_cli_default_artifacts_and_help(tmp_path):
    _, ledger = read(tmp_path, [event("rental.requested", "mark", 0), event("rental.terminated", "mark", 100),
                             event("forecast.integrate", "begin", 10), event("forecast.integrate", "end", 80)])
    output = tmp_path / "result.json"
    proc = subprocess.run([sys.executable, "-m", "woof.cli", "timeline", str(ledger), "--output", str(output)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(output.read_text())["ff"] == .7
    assert output.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    for args, phrase in ((["timeline", "--help"], "project"), (["timeline", "project", "--help"], "--reference"),
                         (["--help-all"], "timeline")):
        proc = subprocess.run([sys.executable, "-m", "woof.cli", *args], capture_output=True, text=True)
        assert proc.returncode == 0
        assert phrase in proc.stdout


def test_real_project_cli_and_startup_ledger_bridge(tmp_path):
    from pathlib import Path
    root = Path(__file__).parent / "data" / "timeline"
    output = tmp_path / "project.json"
    proc = subprocess.run([sys.executable, "-m", "woof.cli", "timeline", "project",
        "--reference", str(root / "projection-reference.json"), "--deltas", str(root / "projection-deltas.json"),
        "--output", str(output)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(output.read_text())["products"][0]
    assert result["seconds_saved"] == 8
    assert result["projected_proxy_ff"] == 100 / 112
    from tools.startup_timeline import main
    output = tmp_path / "bridge.json"
    assert main(["--ledger", str(root / "hex-source.json"), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["mode"] == "proxy"


def test_retained_historic_regional_fraction_and_ensemble_wall():
    from pathlib import Path
    root = Path(__file__).parent / "data" / "timeline"
    timeline = Timeline().read(root / "regional-cost.json").read(root / "regional-progress.jsonl")
    report = timeline.report()
    assert report["ff"] is None  # legacy source lacks rental.requested
    assert abs(report["legacy_ff"] - .905) / .905 < .01
    timeline = Timeline().read(root / "ensemble-boundaries.json").read(root / "ensemble-scheduler.json")
    report = timeline.report(cards=4)
    assert abs(report["legacy_rental_wall_seconds"] / 60 - 78.5) / 78.5 < .01
    assert report["ff"] is None
    assert report["legacy_ff"] is None  # preparation is inside the member spans


@pytest.mark.parametrize("name", ["hex-source.json", "cycle-source.json"])
def test_retained_historic_incomplete_step_boundaries_stay_unknown(name):
    from pathlib import Path
    report = Timeline().read(Path(__file__).parent / "data" / "timeline" / name).report()
    assert report["mode"] == "proxy"
    assert report["ff"] is None
    assert report["proxy_ff"] is None
    assert report["unplaced"]


def test_parallelism_sampling_keeps_live_children_when_a_task_exits(monkeypatch):
    """Regression: a task can exit between listing procfs and reading its children."""
    import io
    from woof.timeline import parallelism

    listdir, real_open = parallelism.os.listdir, open

    def tasks(path):
        return ["gone", "live"] if path == "/proc/123/task" else listdir(path)

    def children(path, *args, **kwargs):
        if path == "/proc/123/task/gone/children":
            raise FileNotFoundError(path)
        if path == "/proc/123/task/live/children":
            return io.StringIO("234 345\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(parallelism.os, "listdir", tasks)
    monkeypatch.setattr(parallelism, "open", children, raising=False)
    assert parallelism._children(123) == [234, 345]
