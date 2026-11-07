"""Region door (2026-10-29): a limited-area source's GRIB download runs beside the mesh stage.

The breakage this guards: a polygon region from HRRR downloaded its hours only after the mesh and cut finished
(MESHES.md 21.2 s + 86.3 s; build host 82 s then 136 s). The download starts with the mesh stage, the intermediate stage
joins it before its own fetch-and-decode, and a door that stops early stops the download.
"""
import json
from pathlib import Path

import pytest

from woof.hex import card_fit, regional_door as door
from woof.hex.cli import build_parser


def arguments(tmp_path, *extra):
    return build_parser().parse_args([
        "regional", "--point", "36.5,-97.5", "--cycle", "2024-05-06T18",
        "--hours", "3", "--work-dir", str(tmp_path / "run"), *extra])


class Handle:
    def __init__(self, events, argv, returncode=0):
        self.events, self.argv, self.returncode = events, argv, returncode

    def finish(self):
        self.events.append("prefetch.finish")
        return {"returncode": self.returncode, "wall_seconds": 30.0, "cpu_seconds": 4.0,
                "average_cpu_cores": 0.13, "log": "source-prefetch.log"}

    def cancel(self):
        self.events.append("prefetch.cancel")


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    events = []
    handles = []

    def start(argv, *, name, directory, env=None):
        events.append("prefetch.start")
        handles.append(Handle(events, [str(a) for a in argv]))
        return handles[-1]

    def cli(argv, status):
        events.append(f"cli.{argv[0]}")
        if argv[0] == "mesh-plan":
            mesh = Path(argv[argv.index("--out-dir") + 1])
            mesh.mkdir(parents=True, exist_ok=True)
            (mesh / "x.point-generate.json").write_text(json.dumps({"cull": {"row": "r"}}))
        return 0

    monkeypatch.setattr(card_fit, "start_stage", start)
    monkeypatch.setattr(door, "_cli", cli)
    monkeypatch.setattr(door, "_first_look", lambda *a, **k: None)
    rest = {name: (lambda name: lambda a, v, s: events.append(name) or {"returncode": 0})(name)
            for name in ("init", "lbc", "contract", "forecast", "products")}
    monkeypatch.setattr(door, "_RUNNERS", {**door._RUNNERS, **rest})
    return events, handles


def test_download_starts_with_the_mesh_and_is_joined_before_the_intermediate(tmp_path, wired):
    events, handles = wired
    args = arguments(tmp_path)
    assert door.run_regional(args) == 0
    assert events[:4] == ["prefetch.start", "cli.mesh-plan", "prefetch.finish", "cli.intermediate"]
    argv = handles[0].argv
    plan = json.loads((args.work_dir / "regional-plan.json").read_text())
    assert argv[argv.index("--prefetch-source") + 1:] == [
        plan["source"]["intermediate_row"], plan["paths"]["grib"], plan["cycle"], plan["paths"]["met"],
        str(max(door.PREFETCH_STREAMS, args.workers)), "0", "1", "2", "3"]
    receipt = json.loads((args.work_dir / "regional-receipt.json").read_text())
    assert receipt["stages"]["intermediate"]["source_prefetch"]["returncode"] == 0
    assert door._PREFETCHES == {}


def test_a_failed_download_leaves_the_stage_to_fetch(tmp_path, wired, monkeypatch):
    events, handles = wired
    original = card_fit.start_stage
    monkeypatch.setattr(card_fit, "start_stage", lambda argv, **k: (original(argv, **k), setattr(handles[-1], "returncode", 1))[0])
    args = arguments(tmp_path)
    assert door.run_regional(args) == 0
    assert "cli.intermediate" in events  # the stage's own fetch-and-decode still runs
    receipt = json.loads((args.work_dir / "regional-receipt.json").read_text())
    assert receipt["stages"]["intermediate"]["source_prefetch"]["returncode"] == 1


def test_a_door_that_stops_at_the_mesh_stops_the_download(tmp_path, wired, monkeypatch):
    events, _ = wired
    monkeypatch.setattr(door, "_cli", lambda argv, status: events.append(f"cli.{argv[0]}") or 3)
    assert door.run_regional(arguments(tmp_path)) == 3
    assert events == ["prefetch.start", "cli.mesh-plan", "prefetch.cancel"]
    assert door._PREFETCHES == {}


@pytest.mark.parametrize("extra", [("--stage", "mesh"), ("--skip", "intermediate")])
def test_no_download_when_the_intermediate_stage_will_not_run(tmp_path, wired, extra):
    events, _ = wired
    assert door.run_regional(arguments(tmp_path, *extra)) == 0
    assert "prefetch.start" not in events


def test_the_background_half_downloads_every_hour_at_once(tmp_path, monkeypatch):
    """Two hours each holding the fetch for 0.4 s finish in well under 0.8 s, with the stage's own request fields."""
    import threading
    import time
    from woof.hex import hrrr_intermediate

    seen, gate = [], threading.Barrier(2, timeout=5)

    def ensure(request, hour, log):
        seen.append((request.source.name, str(request.grib_dir), request.out_dir, request.workers, hour))
        gate.wait()  # both hours are in flight together, or this times out
        time.sleep(0.4)
        return request.out_dir

    monkeypatch.setattr(hrrr_intermediate, "_ensure_hour", ensure)
    started = time.monotonic()
    assert door._prefetch_source(["hrrr", str(tmp_path / "grib"), "2026-10-28T12", str(tmp_path / "met"), "16", "0", "1"]) == 0
    assert time.monotonic() - started < 0.75
    assert sorted(item[-1] for item in seen) == [0, 1] and {item[3] for item in seen} == {16}


def test_a_table_source_decodes_beside_the_mesh_and_the_stage_joins_it(tmp_path, wired, monkeypatch):
    """GFS and IFS read no mesh: mesh 84 s then intermediates 82 s on the regional hrrrv5 region (build host, 2026-10-31)."""
    events, handles = wired
    monkeypatch.setattr(door, "_stage_intermediate_from_table",
                        lambda *a: pytest.fail("the stage redid what the background stage wrote"))
    finished = Handle.finish

    def finish(self):
        plan = json.loads((args.work_dir / "regional-plan.json").read_text())
        met = Path(plan["paths"]["met"])
        met.mkdir(parents=True, exist_ok=True)
        (met / "intermediate-receipt.json").write_text("{}")
        return finished(self)

    monkeypatch.setattr(Handle, "finish", finish)
    args = arguments(tmp_path, "--source", "gfs")
    assert door.run_regional(args) == 0
    assert events[:3] == ["prefetch.start", "cli.mesh-plan", "prefetch.finish"]
    argv = handles[0].argv
    assert argv[argv.index("--decode-table-source") + 1] == str(args.work_dir / "regional-plan.json")
    receipt = json.loads((args.work_dir / "regional-receipt.json").read_text())
    assert receipt["stages"]["intermediate"]["source_prefetch"]["returncode"] == 0
    assert "reused" not in receipt["stages"]["intermediate"]


def test_a_failed_table_decode_leaves_the_stage_to_run_and_say_why(tmp_path, wired, monkeypatch):
    events, handles = wired
    original = card_fit.start_stage
    monkeypatch.setattr(card_fit, "start_stage", lambda argv, **k: (original(argv, **k), setattr(handles[-1], "returncode", 1))[0])
    monkeypatch.setattr(door, "_stage_intermediate_from_table",
                        lambda *a: events.append("stage.decode") or {"returncode": 0, "files": 4})
    args = arguments(tmp_path, "--source", "gfs")
    assert door.run_regional(args) == 0
    assert "stage.decode" in events


def test_the_background_half_runs_the_stage_function_on_the_plan(tmp_path, monkeypatch):
    plan = tmp_path / "regional-plan.json"
    plan.write_text(json.dumps({"hours": 3}))
    seen = []
    monkeypatch.setattr(door, "_stage_intermediate_from_table",
                        lambda args, value, status: seen.append((args.workers, value)) or status.note("ok"))
    assert door._decode_table_source([str(plan), "6"]) == 0
    assert seen == [(6, {"hours": 3})]
