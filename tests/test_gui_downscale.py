"""Downscale from a finished run's page: ``/api/runs/RUN/downscale`` against a live server with a fake runner.

The door runs the ``woof downscale`` line the terminal controller builds for the same answers.  The expected
argument lists below are written out by hand from the controller's own pin (``tools/arwen-tui/src/main.rs``,
``a_downscale_request_builds_exactly_the_guided_command``) and the panel's defaults
(``DownscalePanel::derived_payload``), with the boundary ceiling asked for as the parent grid's own frame cadence
rather than accepted; they are never computed by the code under test.
"""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path

import pytest

from woof import proc_identity
from woof.gui import api as gui_api, auth, runs
from woof.gui.jobs import Refused, Runner
from woof.gui.server import build_server, serve_in_thread

CYCLE = "2026-09-24"


class FakeRunner(Runner):
    """Stands where the engine does: a review writes the plan ``--dry-run`` writes, a start launches nothing."""

    kind = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.checked: list[tuple[list[str], Path | None]] = []
        self.detached: list[dict] = []
        self.queries: list[list[str]] = []
        self.refuse: str | None = None
        self.wrapper_pid = os.getpid()
        # the disk block woof downscale --dry-run writes (woof.offline_child_run.child_disk_projection)
        self.disk = {"total_bytes": 6 * 2**30, "free_bytes": 40 * 2**30, "fits": True,
                     "history_bytes": 4 * 2**30, "checkpoint_bytes": 1 * 2**30, "picture_bytes": 1 * 2**30,
                     "pictures_per_frame": 6, "keep_checkpoints": 1}

    def query(self, argv, *, cwd=None, timeout=0, log=None):
        self.queries.append(list(argv))
        if "downscale-parent" in argv:
            return {"schema": "gpuwm.downscale-parent.v1", "dx_m": 12000.0, "center_latlon": [39.5, -84.0]}
        return {"devices": []}

    def check(self, argv, *, cwd=None, timeout=0):
        self.checked.append((list(argv), cwd))
        if self.refuse:
            raise Refused(self.refuse)
        out = Path(argv[argv.index("--out") + 1])
        # The shapes woof downscale --dry-run writes: the outline keyed by compass corner beside parent_cells and
        # basis (woof.downscale._child_outline), each warning as woof.explain.warn's {"action", "why"} record.
        plan = {"schema": "gpuwm.downscale-plan.v1", "child_grid_id": 2, "parent_domain": 1,
                "child_grid": {"nx": 60, "ny": 60, "nz": 49, "dx": 4000.0, "dy": 4000.0, "dt": 20.0,
                               "run_seconds": 3600.0, "output_interval_s": 3600.0, "ratio": 3,
                               "i_parent_start": 11, "j_parent_start": 13},
                "child_outline": {"sw": [39.0, -85.0], "se": [39.0, -84.0], "ne": [40.0, -84.0],
                                  "nw": [40.0, -85.0],
                                  "parent_cells": {"i_first": 11, "i_last": 30, "j_first": 13, "j_last": 32},
                                  "basis": "parent mass points of the corner cells the child covers"},
                "memory": {"basis": "measured-local", "vram_gib": 16.0, "free_bytes": 15 * 2**30,
                           "budget_bytes": 14 * 2**30, "peak_envelope_bytes": 3 * 2**30, "fits": True},
                "streaming": {"mode": "resident", "why": "fits the card"},
                "warnings": [{"action": "child surface state interpolated from the parent's own history rather "
                                        "than built on the child grid",
                              "why": "the child's land-use is the parent's"}],
                "les_regime": None, "render_products": "all", "outdir": str(out), "disk": self.disk}
        (out.parent / f"{out.name}.downscale-plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (out.parent / f"{out.name}.child.toml").write_text("nx = 60\n", encoding="utf-8")
        return "woof downscale: derived child 60x60\n"

    def launch_detached(self, argv, *, cwd, outdir, kind, owner_file=None):
        if self.refuse:
            raise Refused(self.refuse)
        # the engine adopts only an empty --out: nothing of the page's may be in it when the command starts
        assert list(Path(outdir).iterdir()) == []
        jobs_dir = Path(cwd).parent / "fake-jobs" / f"job-{len(self.detached)}"
        jobs_dir.mkdir(parents=True)
        # A job record as the real launch writes it: the wrapper's PID and its identity.
        job = {"job_id": f"job-{len(self.detached)}", "jobs_dir": str(jobs_dir), "wrapper_pid": self.wrapper_pid,
               "wrapper_process": proc_identity.identify(self.wrapper_pid), "argv": list(argv)}
        self.detached.append({"argv": list(argv), "cwd": Path(cwd), "outdir": Path(outdir), "kind": kind, "job": job})
        return job

    def runtime_gap(self):
        # A start launches nothing here, so it needs no CuPy in this Python.
        return None


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    monkeypatch.setattr("woof.fetch._head_ok", lambda url: True)
    runner = FakeRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    yield server, runner
    server.shutdown()
    server.server_close()


def request(server, method, path, *, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    sent = {"Host": f"127.0.0.1:{server.port}", "Cookie": f"{auth.cookie_name(server.port)}={server.token}"}
    if method == "POST":
        sent[auth.TOKEN_HEADER] = server.token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        sent["Content-Type"] = "application/json"
    connection.request(method, path, body=data, headers=sent)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response, json.loads(raw)


def dead_pid() -> int:
    from woof.mcp.gpulock import _pid_alive

    pid = 999_999
    while _pid_alive(pid):
        pid -= 1
    return pid


def finished_run(root: Path, name: str = "gfs-parent", *, frames: int = 3, checkpoint: bool = True,
                 end: dict | None = None, pid: int | None = None, minutes: int = 60,
                 nest_minutes: int | None = None) -> tuple[Path, Path]:
    """A finished run-plan folder laid out as the prepared route writes it: frames under chain/NAME/run/wrfout,
    checkpoints in chain/NAME/run, both named by absolute paths in the event log.

    ``minutes`` is how often d01 saved a frame; ``nest_minutes`` adds a d02 nest saving on its own cadence over the
    same window, with its own checkpoint.
    """

    run = root / name
    work = run / "chain" / name / "run"
    history = work / "wrfout"
    history.mkdir(parents=True)
    records = [{"event": "resolved_plan", "configuration": {"experiment": {
        "run_seconds": 7200.0, "start_time": f"{CYCLE}T00:00:00",
        "projection": {"map_proj": "lambert", "ref_lat": 39.5, "ref_lon": -84.0},
        "domains": [{"grid_id": 1, "parent_id": 0, "i_parent_start": 1, "j_parent_start": 1,
                     "parent_grid_ratio": 1, "run": {"nx": 100, "ny": 90, "nz": 49, "dx": 12000.0}}]}}}]
    window = (frames - 1) * minutes
    grids = [(1, minutes)] + ([(2, nest_minutes)] if nest_minutes else [])
    for domain, step in grids:
        for at in range(0, window + 1, step):
            frame = history / f"wrfout_d{domain:02d}_{CYCLE}_{at // 60:02d}_{at % 60:02d}_00"
            frame.write_bytes(b"")
            records.append({"event": "output_committed", "domain": domain,
                            "valid_time": f"{CYCLE}T{at // 60:02d}:{at % 60:02d}:00", "path": str(frame)})
    if checkpoint:
        for domain, _ in grids:
            rst = work / f"gpuwmrst_d{domain:02d}_{CYCLE}_{window // 60:02d}_{window % 60:02d}_00__0123abcd.npz"
            rst.write_bytes(b"")
        records.append({"event": "model_progress", "model_seconds": window * 60.0, "last_checkpoint": str(rst)})
    records.append(end if end is not None else {"event": "completed"})
    (run / "events.jsonl").write_text("".join(json.dumps({"sequence": i, **r}) + "\n" for i, r in enumerate(records)),
                                      encoding="utf-8")
    # A manifest as ``woof run-plan`` writes it: the pid and, for a live one, that process's identity.
    manifest = {"pid": dead_pid() if pid is None else pid}
    if pid is not None:
        manifest["process"] = proc_identity.identify(pid)
    (run / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run, history


# ---------------------------------------------------------------- the command is the desktop's

def test_show_command_for_a_downscale_is_the_line_the_desktop_panel_sends(gui):
    # Before this door, POST /api/runs/RUN/downscale answered 404 "No such action." and a web GUI user had to
    # use a terminal or the desktop app to downscale a finished forecast.
    server, runner = gui
    _, history = finished_run(server.root)
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"mode": "run", "lat": 39.5, "lon": -84.0, "dry_run": True})
    assert response.status == 200, body
    assert body["dry_run"] is True
    out = server.root / "gfs-parent-d02-x3"
    # A request for a centre with the panel's defaults, as the controller's guide spells it: the positional parent
    # frames folder, --flag=value answers in the guide's order, the boundary ceiling where the guide asks for it,
    # --out DIR as two tokens, the measured card and every product drawn.
    # The page's engine commands run the installed package with -P (gpuwm.gui.jobs.engine_argv).
    assert body["argv"][1:5] == ["-P", "-m", "woof", "downscale"]
    assert body["argv"][5:] == [
        str(history), "--point=39.5,-84", "--parent-restart=latest", "--parent-domain=1", "--ratio=3",
        "--max-boundary-interval-seconds=3600", "--output-interval-seconds=3600", "--tiles=auto", "--out", str(out),
        "--auto-vram", "--render-products=all"]
    # Show command runs nothing and makes nothing
    assert not out.exists() and runner.detached == [] and runner.checked == []
    assert not (server.root / ".arwen-gui" / "starts").exists()

    # Review is the same line with the plan mode's --dry-run where the guide puts it
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"mode": "plan", "lat": 39.5, "lon": -84.0, "dry_run": True})
    assert response.status == 200
    args = body["argv"][5:]
    assert args[args.index("--out") + 2:] == ["--dry-run", "--auto-vram", "--render-products=all"]


def test_the_boundary_ceiling_is_the_cadence_the_chosen_grids_frames_have(gui):
    # The page asks the engine for the cadence the parent grid actually saved, the one the form shows, and never
    # sends --accept-parent-cadence: that flag records a person's acceptance of the archive's cadence as a
    # scientific choice, and nothing on the page asks for it.  So a default child's plan and report record the
    # ceiling the page asked for, and the engine refuses frames that are not at the cadence the page showed.
    server, _ = gui
    finished_run(server.root, "hourly")
    finished_run(server.root, "nested", nest_minutes=15)
    finished_run(server.root, "quarter", frames=5, minutes=15)

    def asked(run, **extra):
        response, body = request(server, "POST", f"/api/runs/{run}/downscale",
                                 body={"lat": 39.5, "lon": -84.0, "dry_run": True, **extra})
        assert response.status == 200, body
        args = body["argv"][5:]
        assert "--accept-parent-cadence" not in args
        return [arg for arg in args if arg.startswith("--max-boundary-interval-seconds")]

    assert asked("hourly") == ["--max-boundary-interval-seconds=3600"]
    assert asked("quarter") == ["--max-boundary-interval-seconds=900"]
    # a run with a nest starts from its finest grid, whose frames keep their own cadence; another grid picked on
    # the form is asked for at its own
    assert asked("nested") == ["--max-boundary-interval-seconds=900"]
    assert asked("nested", domain=1) == ["--max-boundary-interval-seconds=3600"]
    # the review is the same line with --dry-run, and Start runs what was reviewed
    response, body = request(server, "POST", "/api/runs/hourly/downscale",
                             body={"mode": "plan", "lat": 39.5, "lon": -84.0})
    assert response.status == 200, body
    assert "--max-boundary-interval-seconds=3600" in body["argv"]
    assert "--max-boundary-interval-seconds=3600" in body["start_argv"]
    assert "--accept-parent-cadence" not in body["start_argv"]

    # the form reads the cadence it shows from the grid rows
    response, facts = request(server, "GET", "/api/runs/nested/downscale")
    assert facts["domain"] == 2
    assert {row["id"]: row["interval_s"] for row in facts["domains"]} == {1: 3600.0, 2: 900.0}


def test_a_drawn_box_is_sized_in_child_cells_as_the_desktop_sizes_it(gui):
    # A 1 by 1 degree box at 39.5 N on a 12 km grid refined 3 times (4 km): 85.8 km across is 21.45 cells,
    # rounded to 21, a whole number of parent cells (7); 111.2 km tall is 27.8 cells, rounded to 28 and snapped to
    # the ratio's unit, 27.  The centre is the box's centre.
    server, _ = gui
    finished_run(server.root)
    box = {"west": -85.0, "south": 39.0, "east": -84.0, "north": 40.0}
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"box": box, "hours": 1, "card": "16gb", "products": "none", "name": "boxed",
                                   "dry_run": True})
    assert response.status == 200, body
    args = body["argv"][5:]
    assert args[1:5] == ["--point=39.5,-84.5", "--parent-restart=latest", "--parent-domain=1", "--ratio=3"]
    assert "--child-size=21,27" in args and "--hours=1" in args
    # a card size named on the page is the engine's own tier, in place of measuring this computer's card
    assert "--card=16gb" in args and "--auto-vram" not in args
    assert args[-1] == "--render-products=none"
    assert args[args.index("--out") + 1] == str(server.root / "boxed")
    assert body["child"]["dx_km"] == 4.0 and body["child"]["child_size"] == [21, 27]

    # a target spacing is the refinement that reaches it
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"lat": 39.5, "lon": -84.0, "dx_km": 3, "dry_run": True})
    assert response.status == 200 and "--ratio=4" in body["argv"]
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"lat": 39.5, "lon": -84.0, "dx_km": 3, "ratio": 3, "dry_run": True})
    assert response.status == 400 and "disagree" in body["message"]


# ---------------------------------------------------------------- review, then start

def test_review_shows_the_engines_plan_and_leaves_nothing_behind(gui):
    server, runner = gui
    finished_run(server.root)
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"mode": "plan", "lat": 39.5, "lon": -84.0, "hours": 1})
    assert response.status == 200, body
    review = body["review"]
    assert review["nx"] == 60 and review["dx_km"] == 4.0 and review["hours"] == 1.0
    # the engine's compass corners become the ring the map draws, in the desktop panel's order
    assert review["outline"] == [[39.0, -85.0], [39.0, -84.0], [40.0, -84.0], [40.0, -85.0]]
    assert "60 by 60 points at 4 km" in review["words"] and "resident" in review["words"]
    assert "Needs 3.0 GiB of the 14.0 GiB the card allows." in review["words"]
    # the disk the child will write, beside the free space where it writes it
    assert "Writes about 6.0 GiB to disk, with 40.0 GiB free there." in review["words"]
    assert review["disk"]["total_bytes"] == 6 * 2**30 and review["disk"]["fits"] is True
    assert review["warnings"] == ["child surface state interpolated from the parent's own history rather than "
                                  "built on the child grid"]
    argv, cwd = runner.checked[-1]
    assert "--dry-run" in argv
    # the review writes into a draft folder, which goes once the plan is read: nothing lands in the runs folder
    assert not cwd.exists() and list((server.root / ".arwen-gui" / "drafts").iterdir()) == []
    assert not (server.root / "gfs-parent-d02-x3").exists()
    assert not [p.name for p in server.root.iterdir() if p.name.endswith((".downscale-plan.json", ".child.toml"))]
    # the Start line is the reviewed one into the new run folder, without --dry-run
    assert "--dry-run" not in body["start_argv"]
    assert body["start_argv"][body["start_argv"].index("--out") + 1] == str(server.root / "gfs-parent-d02-x3")

    runner.refuse = ("warning: something first\n"
                     f"woof downscale: the child reaches past the parent interior at {server.root}/gfs-parent")
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"mode": "plan", "lat": 39.5, "lon": -84.0})
    assert response.status == 422
    assert body["message"].startswith("the child reaches past the parent interior")
    assert str(server.root) not in body["message"]


def test_review_says_when_the_disk_cannot_hold_the_child(gui):
    # A child that fits used to show no disk figure on the page at all; one that does not says both figures.
    server, runner = gui
    finished_run(server.root)
    runner.disk = dict(runner.disk, total_bytes=20 * 2**30, free_bytes=2 * 2**30, fits=False)
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"mode": "plan", "lat": 39.5, "lon": -84.0, "hours": 1})
    assert response.status == 200, body
    assert "Writes about 20.0 GiB to disk, more than the 2.0 GiB free there." in body["review"]["words"]
    assert body["review"]["disk"]["fits"] is False

    # a plan with no disk block (an engine older than the block) reviews as before, without disk words
    runner.disk = None
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"mode": "plan", "lat": 39.5, "lon": -84.0, "hours": 1})
    assert response.status == 200, body
    assert "to disk" not in body["review"]["words"] and body["review"]["disk"] is None


def test_start_launches_into_an_empty_folder_and_the_child_lists_as_running(gui):
    server, runner = gui
    parent, _ = finished_run(server.root)
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"mode": "run", "lat": 39.5, "lon": -84.0, "hours": 1, "name": "fine"})
    assert response.status == 200, body
    assert body["run"] == "fine" and body["url"] == "fine"
    launched = runner.detached[-1]
    assert launched["argv"] == body["argv"] and "--dry-run" not in launched["argv"]
    assert launched["cwd"] == server.root and launched["outdir"] == server.root / "fine"
    child = server.root / "fine"
    assert child.is_dir() and list(child.iterdir()) == []
    record = runs.start_record(child)
    assert record["job"]["job_id"] == "job-0" and record["parent"] == "gfs-parent"
    assert body["command"] in (parent / "commands.log").read_text(encoding="utf-8")

    # listed at once, running, with the command it ran; nothing is in its folder but what the engine writes
    response, listing = request(server, "GET", "/api/runs")
    row = next(r for r in listing["runs"] if r["id"] == "fine")
    assert row["status"]["state"] == "running"
    response, detail = request(server, "GET", "/api/runs/fine")
    assert "downscale" in detail["commands_log"]

    # the name is now taken
    response, again = request(server, "POST", "/api/runs/gfs-parent/downscale",
                              body={"mode": "run", "lat": 39.5, "lon": -84.0, "name": "fine"})
    assert response.status == 409


def test_a_start_the_engine_refused_before_writing_anything_reads_as_failed_in_its_words(gui):
    server, runner = gui
    finished_run(server.root)
    runner.wrapper_pid = dead_pid()
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"lat": 39.5, "lon": -84.0, "name": "refused"})
    assert response.status == 200
    jobs_dir = Path(runner.detached[-1]["job"]["jobs_dir"])
    (jobs_dir / "result.json").write_text(json.dumps({"exit_code": 2}), encoding="utf-8")
    (jobs_dir / "stderr.log").write_text(
        "woof downscale: the parent history carries no restart for domain 1; pass --parent-restart\n",
        encoding="utf-8")
    response, detail = request(server, "GET", "/api/runs/refused")
    status = detail["status"]
    assert status["state"] == "failed"
    assert status["end"]["message"] == "woof downscale: the parent history carries no restart for domain 1; " \
                                       "pass --parent-restart"


@pytest.mark.parametrize("step", ["commands-log", "start-record", "assistant"])
def test_a_downscale_the_engine_accepted_is_started_whatever_fails_after_it(gui, tmp_path, monkeypatch, step):
    # The defect: a step after the child's launch (the parent's commands.log line, the page's record of the start,
    # the assistant making room on the card) raised, so Start answered that the page server hit an error while the
    # child ran, and the card's OWNER line was let go while the child held the card.
    server, runner = gui
    owner = tmp_path / "card-queue" / "OWNER"
    owner.parent.mkdir()
    owner.write_text("", encoding="utf-8")
    queue = server.api.queue
    queue.owner_file, queue.owner_tag, queue.pid = str(owner), "gui-test", 424_242
    queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                          "memory_used_mib": 512}], "processes": []}
    parent, _ = finished_run(server.root)
    if step == "commands-log":
        def denied(*args, **kwargs):
            raise PermissionError(13, "Permission denied", "commands.log")

        monkeypatch.setattr(gui_api, "log_command", denied)
        said = "commands.log"
    elif step == "start-record":
        # a file where the page's folder of start records goes
        starts = runs.start_record_path(server.root / "fine").parent
        starts.parent.mkdir(parents=True, exist_ok=True)
        starts.write_text("", encoding="utf-8")
        said = "record of the start"
    else:
        def unanswered():
            raise RuntimeError("the model server did not answer")

        monkeypatch.setattr(server.api.assistant, "make_room", unanswered)
        said = "assistant's model"
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"mode": "run", "lat": 39.5, "lon": -84.0, "hours": 1, "name": "fine"})
    assert response.status == 200, body
    assert body["message"].startswith("Started.") and said in body["message"], body["message"]
    assert len(runner.detached) == 1
    # the child holds the card: the line is its wrapper's
    lines = owner.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and f" pid {runner.wrapper_pid} " in lines[0], lines
    if step != "commands-log":
        assert body["command"] in (parent / "commands.log").read_text(encoding="utf-8")


def test_the_review_reads_the_outline_and_warnings_in_every_form_the_engine_writes():
    # A review that read the outline only as a list drew nothing on the map for the engine's real plan, which keys
    # the corners by compass point; one that read warnings by "message" showed none of woof.explain.warn's.
    from woof.gui import downscale

    keyed = {"child_outline": {"sw": [35.05, -98.02], "se": [35.05, -96.98], "ne": [35.95, -96.98],
                               "nw": [35.95, -98.02], "parent_cells": {"i_first": 3}, "basis": "mass points"}}
    assert downscale.plan_outline(keyed) == [[35.05, -98.02], [35.05, -96.98], [35.95, -96.98], [35.95, -98.02]]
    assert downscale.plan_outline({"child_outline": [[1, 2], [3, 4], [5, 6]]}) == [[1.0, 2.0], [3.0, 4.0],
                                                                                   [5.0, 6.0]]
    # fewer than three finite corners outline nothing, in either form
    assert downscale.plan_outline({"child_outline": {"sw": [35.05, -98.02], "ne": [35.95, -96.98]}}) is None
    assert downscale.plan_outline({"child_outline": [[1, 2], [float("nan"), 4], [5, 6]]}) is None
    assert downscale.plan_outline({"child_outline": {"sw": [1, 2], "se": [3, 4], "ne": [5, 6], "nw": "missing"}}) \
        == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert downscale.plan_outline({}) is None

    plan = {"warnings": [{"action": "child extent 91x111 is adjusted to 90x111", "why": "the placement's unit"},
                         {"message": "an older record"}, "a bare sentence", 7, {"why": "no sentence"}],
            "memory_note": "this child could not be priced before the run: no card"}
    assert downscale.plan_warnings(plan) == ["child extent 91x111 is adjusted to 90x111", "an older record",
                                             "a bare sentence",
                                             "this child could not be priced before the run: no card"]


# ---------------------------------------------------------------- what cannot be downscaled, in words

def test_the_run_page_says_what_it_offers_and_why_not(gui):
    server, _ = gui
    finished_run(server.root)
    response, facts = request(server, "GET", "/api/runs/gfs-parent/downscale")
    assert response.status == 200
    assert facts["eligible"] is True and facts["domain"] == 1 and facts["ratio"] == 3
    assert facts["domains"] == [{"id": 1, "frames": 3, "restart_sets": 1, "interval_s": 3600.0, "dx_km": 12.0,
                                 "first": f"{CYCLE}T00:00:00Z", "last": f"{CYCLE}T02:00:00Z"}]
    assert facts["output_minutes"] == "60" and facts["window_hours"] == 2.0 and facts["name"] == "gfs-parent-d02-x3"

    finished_run(server.root, "one-frame", frames=1)
    response, facts = request(server, "GET", "/api/runs/one-frame/downscale")
    assert facts["eligible"] is False
    assert facts["reason"] == "Only one saved output frame, so this run cannot be downscaled."
    response, body = request(server, "POST", "/api/runs/one-frame/downscale", body={"lat": 39.5, "lon": -84.0})
    assert response.status == 409 and body["message"] == facts["reason"]

    finished_run(server.root, "no-checkpoint", checkpoint=False)
    response, body = request(server, "POST", "/api/runs/no-checkpoint/downscale", body={"lat": 39.5, "lon": -84.0})
    assert response.status == 409
    assert body["message"] == "No restart checkpoints saved, so this run cannot be downscaled."

    finished_run(server.root, "still-going", end={"event": "model_progress", "model_seconds": 3600.0},
                 pid=os.getpid())
    response, body = request(server, "POST", "/api/runs/still-going/downscale", body={"lat": 39.5, "lon": -84.0})
    assert response.status == 409 and "running" in body["message"]

    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"lat": 39.5, "lon": -84.0, "tiles": "on"})
    assert response.status == 400 and "tiles" in body["message"]
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale", body={"dry_run": True})
    assert response.status == 400 and "centre" in body["message"]


def test_a_copied_run_folder_is_read_where_it_now_is(gui, tmp_path):
    # The event log keeps the absolute paths the run had when it wrote them; a folder copied elsewhere keeps the
    # names under the run's own, so its frames are found in the copy.
    server, _ = gui
    elsewhere, _ = finished_run(tmp_path / "elsewhere")
    import shutil

    shutil.copytree(elsewhere, server.root / "gfs-parent")
    shutil.rmtree(elsewhere)
    response, body = request(server, "POST", "/api/runs/gfs-parent/downscale",
                             body={"lat": 39.5, "lon": -84.0, "dry_run": True})
    assert response.status == 200, body
    assert body["argv"][5] == str(server.root / "gfs-parent" / "chain" / "gfs-parent" / "run" / "wrfout")


# ---------------------------------------------------------------- a finished downscale's own facts

def finished_child(root: Path, name: str = "fine", *, plan_beside: bool = False) -> Path:
    """A finished downscale folder as ``woof downscale`` leaves it: the plan it wrote, and an event log whose
    ``resolved_plan`` names the child's configuration file instead of carrying it
    (woof.offline_child_run emits ``config_source`` and ``config_sha256`` only)."""

    child = root / name
    child.mkdir(parents=True)
    plan = {"schema": "gpuwm.downscale-plan.v1", "child_grid_id": 2, "parent_domain": 1,
            "parent": {"run_dir": str(root / "gfs-parent"), "domain": 1},
            "child_grid": {"nx": 72, "ny": 72, "nz": 49, "dx": 4000.0, "dy": 4000.0, "dt": 20.0,
                           "run_seconds": 3600.0, "output_interval_s": 3600.0, "ratio": 3,
                           "i_parent_start": 11, "j_parent_start": 13},
            "placement": {"ratio": 3, "i_parent_start": 11, "j_parent_start": 13},
            "initial_condition": {"GPUWM_INITIAL_CONDITION_MODEL_START_DATE": f"{CYCLE}_01:00:00"}}
    where = root / f"{name}.downscale-plan.json" if plan_beside else child / "downscale-plan.json"
    where.write_text(json.dumps(plan), encoding="utf-8")
    records = [{"event": "resolved_plan", "config_source": str(child / "child.toml"), "config_sha256": "a" * 64},
               {"event": "stage_started", "stage": "forecast"},
               {"event": "model_progress", "model_seconds": 3600.0},
               {"event": "completed", "stage": "forecast"}]
    (child / "events.jsonl").write_text("".join(json.dumps({"sequence": i, **r}) + "\n"
                                                for i, r in enumerate(records)), encoding="utf-8")
    (child / "run-manifest.json").write_text(json.dumps({"pid": dead_pid()}), encoding="utf-8")
    return child


@pytest.mark.parametrize("plan_beside", [False, True], ids=["plan-inside", "plan-beside"])
def test_a_finished_downscale_shows_its_grid_and_levels_and_no_damage_warning(gui, plan_beside):
    # The defect: the child's event log names its configuration rather than carrying it, which the page read as
    # damaged records ("grid details could not be read") and left Grid and Levels blank on the list, the run
    # page and the article, with or without pictures.
    server, _ = gui
    finished_run(server.root)
    finished_child(server.root, plan_beside=plan_beside)
    response, listing = request(server, "GET", "/api/runs")
    assert response.status == 200
    row = next(r for r in listing["runs"] if r["id"] == "fine")
    assert row["status"]["state"] == "finished"
    assert row["status"]["metadata_warnings"] == []
    assert row["card"]["dx_km"] == [4.0]
    assert row["card"]["levels"] == [49]
    response, detail = request(server, "GET", "/api/runs/fine")
    assert detail["status"]["metadata_warnings"] == []
    assert detail["status"]["grids"]["grids_label"] == "4 km"
    assert detail["card"]["dx_km"] == [4.0] and detail["card"]["levels"] == [49]
    response, article = request(server, "GET", "/api/wiki/run/fine")
    assert response.status == 200, article
    facts = {fact["id"]: fact for fact in article["facts"]}
    assert facts["grid"]["text"] == "4 km"
    assert facts["levels"]["text"] == "49"
    if not plan_beside:
        # cited to the file the facts were read from
        assert facts["grid"]["cite"] == ["run-file:downscale-plan.json"]
        assert facts["levels"]["cite"] == ["run-file:downscale-plan.json"]
    # the child is still placed in its parent's grid on the map, not on a grid of its own
    response, placed = request(server, "GET", "/api/runs/fine/map")
    assert [d.get("context", False) for d in placed["domains"]] == [True, False]
    assert placed["domains"][-1]["grid_id"] == 2 and placed["domains"][-1]["dx_km"] == 4.0


def test_a_reference_only_plan_without_its_downscale_plan_still_says_the_grid_is_unreadable(gui):
    # A resolved plan that carries no configuration and whose downscale plan is gone has no grid to show: the
    # warning is true there and stays.
    server, _ = gui
    child = finished_child(server.root)
    (child / "downscale-plan.json").unlink()
    response, detail = request(server, "GET", "/api/runs/fine")
    assert detail["status"]["metadata_warnings"] == [runs.GRID_WARNING]
    assert detail["card"]["levels"] == []
