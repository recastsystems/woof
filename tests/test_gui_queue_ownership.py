"""The queue's card lines belong to processes, not to numbers: handed over, kept and removed by identity.

The defects this file holds shut:

- A started forecast's OWNER line was handed to its wrapper's PID with no
  check that the PID was still the wrapper, and a run that had already
  ended was released by a bare PID check.
- The queue removed an ended run's line by PID alone, so another claim that
  happened to carry the same number (another tag, another time) went with
  it and two programs could start on one card.
- A start that failed left the queue's line in the OWNER file.
- A page server that stopped between claiming the card and handing its line
  to the run's wrapper left the line for good, since no record named it:
  harmless while no process held its number, and holding the card for
  whichever process was given that number next.  A line of the server's
  tag whose process had ended stayed the same way.
- A claim recorded before its line was written carried no time, so the
  sweep after that server stopped removed every line of its tag under its
  number: the live line of a run another page server had started, once
  that run was given the same number.
- A start the engine had accepted whose commands.log line or job record
  could not be written was put back in the queue and started a second time
  once the card was free, its card line was let go while it ran, and Start
  said the start had failed.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

import woof
from woof import proc_identity
from woof.gui import api as gui_api, jobs as gui_jobs, runs as gui_runs
from woof.gui.api import ApiError, Reply
from woof.gui.files import write_json
from woof.gui.jobs import Runner
from woof.gui.machines import LOCAL
from woof.gui.queue import MARKER_SCHEMA, QUEUE_SCHEMA
from woof.gui.server import build_server
from woof.machine_agent import release_line, retag_card
from test_gui_server import FakeRunner, make_run


@pytest.fixture()
def owned(tmp_path):
    owner = tmp_path / "card-queue" / "OWNER"
    owner.parent.mkdir()
    owner.write_text("", encoding="utf-8")
    server = build_server(tmp_path / "runs", port=0, runner=FakeRunner(), token="t" * 43)
    queue = server.api.queue
    queue.owner_file, queue.owner_tag, queue.pid = str(owner), "gui-test", 424_242
    queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                          "memory_used_mib": 512}], "processes": []}
    try:
        yield queue, owner
    finally:
        server.server_close()


def _sleeper(seconds: float = 60) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"], stdin=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _launched(process: subprocess.Popen, identity=None):
    identity = proc_identity.identify(process.pid) if identity is None else identity
    return lambda: Reply(200, {"ok": True, "job": {"wrapper_pid": process.pid, "wrapper_process": identity}})


def test_the_line_is_handed_to_the_wrapper_and_recorded_with_its_identity(owned):
    queue, owner = owned
    process = _sleeper()
    try:
        queue.launch_holding_card("run-a", _launched(process), 5.0)
        lines = owner.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1 and lines[0].startswith("gui-test ") and f" pid {process.pid} " in lines[0]
        row = queue._document()["owned"][0]
        assert row["process"] == proc_identity.identify(process.pid)
        assert row["line"] == {"tag": "gui-test", "utc": lines[0].split()[1], "pid": process.pid}
        # While the wrapper lives its line stays; once it has ended the line goes.
        queue._release_ended()
        assert owner.read_text(encoding="utf-8").splitlines() == lines
        process.kill()
        process.wait(timeout=10)
        queue._release_ended()
        assert owner.read_text(encoding="utf-8") == "" and queue._document()["owned"] == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_a_wrapper_that_is_not_its_process_any_more_is_never_handed_the_line(owned):
    queue, owner = owned
    process = _sleeper()
    try:
        stale = {**proc_identity.identify(process.pid), "start": "1"}   # its number, another process's identity
        queue.launch_holding_card("run-b", _launched(process, stale), 5.0)
        assert owner.read_text(encoding="utf-8") == ""
        assert queue._document()["owned"] == []
    finally:
        process.kill()
        process.wait(timeout=10)


def test_an_ended_runs_cleanup_removes_its_own_line_and_no_other_under_the_same_number(owned):
    queue, owner = owned
    process = _sleeper()
    try:
        queue.launch_holding_card("run-c", _launched(process), 5.0)
        ours = owner.read_text(encoding="utf-8")
        # The run's wrapper ended and its number is another process's now (the sleeper stands for that
        # process): the record's identity no longer names what holds the number.
        document = queue._document()
        document["owned"][0]["process"] = {**document["owned"][0]["process"], "start": "1"}
        queue._save(document)
        # That process has claimed the card under the same number (its own tag, its own time) meanwhile.
        theirs = f"other-tool 2026-09-27T00:00:00Z pid {process.pid} bounded 30 min\n"
        also = f"gui-test 2026-01-01T00:00:00Z pid {process.pid} bounded 30 min\n"
        owner.write_text(ours + theirs + also, encoding="utf-8")
        queue._release_ended()
        assert owner.read_text(encoding="utf-8") == theirs + also
        assert queue._document()["owned"] == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_a_start_that_fails_leaves_no_line(owned):
    queue, owner = owned

    def refused():
        raise ApiError(409, "The engine refused the start.", "")

    with pytest.raises(ApiError):
        queue.launch_holding_card("run-d", refused, 5.0)
    assert owner.read_text(encoding="utf-8") == "" and queue._document()["owned"] == []


# A page server on the same forecasts folder and OWNER file, stopped in the middle of a start: it has claimed
# the card and its run's wrapper does not exist yet, or (``claiming``) it has recorded its claim and not yet
# written the line.  ``stale`` records its claim with an identity that is not the process holding its number,
# which is what the record reads once that number has gone to another process.
_CLAIMING_SERVER = r"""
import sys
import time

from woof import machine_agent, proc_identity
from gpuwm.gui.server import build_server
from test_gui_server import FakeRunner

root, owner, stale, step = sys.argv[1], sys.argv[2], sys.argv[3] == "stale", sys.argv[4]
if stale:
    identify = proc_identity.identify
    proc_identity.identify = lambda pid: {**identify(pid), "start": "1"}
if step == "claiming":
    def claim_card(*args, **kwargs):
        print("claiming", flush=True)
        time.sleep(600)

    machine_agent.claim_card = claim_card
server = build_server(root, port=0, runner=FakeRunner(), token="t" * 43)
queue = server.api.queue
queue.owner_file, queue.owner_tag = owner, "gui-test"
queue._cards = lambda: {"devices": [], "processes": []}


def launch():
    print("claimed", flush=True)
    time.sleep(600)


queue.launch_holding_card("run-x", launch, 5.0)
"""


def _claiming_server(queue, owner, *, stale: bool = False, step: str = "claimed") -> subprocess.Popen:
    paths = [str(Path(woof.__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([*paths, os.environ.get("PYTHONPATH", "")])}
    child = subprocess.Popen([sys.executable, "-c", _CLAIMING_SERVER, str(queue.root), str(owner),
                              "stale" if stale else "recorded", step], stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    # Printed by its launch (the claim and its record are both written), or by the claim it is held in (only
    # the record is).
    if child.stdout.readline().strip() != step:
        child.kill()
        _, err = child.communicate(timeout=60)
        pytest.fail(f"the claiming page server did not reach {step!r}: " + err[-2000:])
    lines = owner.read_text(encoding="utf-8").splitlines()
    if step == "claimed":
        assert len(lines) == 1 and lines[0].startswith("gui-test ") and f" pid {child.pid} " in lines[0], lines
    else:
        assert lines == []
    rows = queue._document()["owned"]
    assert [row["pid"] for row in rows] == [child.pid], rows
    return child


def _end(child: subprocess.Popen) -> None:
    if child.poll() is None:
        child.kill()
    child.communicate(timeout=60)


def test_a_page_server_that_stops_between_claiming_and_handing_on_leaves_no_line(owned):
    queue, owner = owned
    child = _claiming_server(queue, owner)
    _end(child)
    # The next page server's sweep.
    queue._release_ended()
    assert owner.read_text(encoding="utf-8") == ""
    assert queue._document()["owned"] == []


def test_a_claim_whose_number_now_names_another_process_goes_and_another_tags_line_stays(owned):
    queue, owner = owned
    child = _claiming_server(queue, owner, stale=True)
    try:
        theirs = f"other-tool 2026-09-27T00:00:00Z pid {child.pid} bounded 30 min\n"
        with owner.open("a", encoding="utf-8") as stream:
            stream.write(theirs)
        queue._release_ended()
        assert owner.read_text(encoding="utf-8") == theirs
        assert queue._document()["owned"] == []
        assert child.poll() is None
    finally:
        _end(child)


def test_a_claim_stopped_before_its_line_was_written_removes_no_line_of_its_tag(owned):
    queue, owner = owned
    child = _claiming_server(queue, owner, stale=True, step="claiming")
    try:
        # Its number now names another process, and that process is a run started by another page server of
        # the same tag on this OWNER file (one on another forecasts folder), holding the card under its own
        # line.  The record names no line, since none was written, so that run's line stays.
        theirs = f"gui-test 2020-01-01T00:00:00Z pid {child.pid} bounded 720 min\n"
        owner.write_text(theirs, encoding="utf-8")
        queue._release_ended()
        assert owner.read_text(encoding="utf-8") == theirs
        assert queue._document()["owned"] == []
    finally:
        _end(child)


def test_a_line_of_this_tag_whose_process_ended_goes_and_no_other_line_does(owned):
    queue, owner = owned
    process = _sleeper()
    process.kill()
    process.wait(timeout=10)
    ended = f"gui-test 2026-09-27T00:00:00Z pid {process.pid} bounded 720 min\n"
    theirs = f"other-tool 2026-09-27T00:00:00Z pid {process.pid} bounded 30 min\n"
    live = f"gui-test 2026-09-27T00:00:00Z pid {os.getpid()} bounded 720 min\n"
    owner.write_text(ended + theirs + live, encoding="utf-8")
    queue._release_ended()
    assert owner.read_text(encoding="utf-8") == theirs + live


def test_retag_hands_over_only_this_servers_own_line(tmp_path):
    owner = tmp_path / "OWNER"
    owner.write_text("gui-test 2026-09-27T00:00:00Z pid 111 bounded 30 min\n"
                     "someone 2026-09-27T00:00:00Z pid 111 bounded 30 min\n", encoding="utf-8")
    line = retag_card(str(owner), 111, 222, tag="gui-test", keep=lambda: True)
    assert line == {"tag": "gui-test", "utc": "2026-09-27T00:00:00Z", "pid": 222}
    assert owner.read_text(encoding="utf-8") == ("gui-test 2026-09-27T00:00:00Z pid 222 bounded 30 min\n"
                                                 "someone 2026-09-27T00:00:00Z pid 111 bounded 30 min\n")
    release_line(str(owner), {"tag": "gui-test", "utc": "2026-09-27T00:00:00Z", "pid": 222})
    assert owner.read_text(encoding="utf-8") == "someone 2026-09-27T00:00:00Z pid 111 bounded 30 min\n"


def test_a_line_record_with_no_time_removes_no_line(tmp_path):
    owner = tmp_path / "OWNER"
    text = ("gui-test 2026-09-27T00:00:00Z pid 111 bounded 720 min\n"
            "gui-test 2026-09-27T01:00:00Z pid 111 bounded 720 min\n")
    owner.write_text(text, encoding="utf-8")
    # Tag and PID alone name every line of that tag under that number, a later claim's included.
    release_line(str(owner), {"tag": "gui-test", "utc": None, "pid": 111})
    release_line(str(owner), {"tag": "gui-test", "pid": 111})
    assert owner.read_text(encoding="utf-8") == text
    release_line(str(owner), {"tag": "gui-test", "utc": "2026-09-27T01:00:00Z", "pid": 111})
    assert owner.read_text(encoding="utf-8") == "gui-test 2026-09-27T00:00:00Z pid 111 bounded 720 min\n"


# A start the engine accepted is started, whatever the bookkeeping after it does.  The defect: the commands.log
# line or the job record written after the launch raised, so the queue put the running forecast back in line and
# started it again once the card was free, the card line was let go while it ran, and Start said the start had
# failed.

class SleeperRunner(FakeRunner):
    """The page's own launch, which writes the run's job record, over a fake engine start that is a real process
    the queue hands its line to."""

    launch = Runner.launch

    def __init__(self) -> None:
        super().__init__()
        self.processes: list[subprocess.Popen] = []

    def launch_detached(self, argv, *, cwd, outdir, kind, owner_file=None):
        process = _sleeper()
        self.processes.append(process)
        self.launched.append(list(argv))
        return {"job_id": f"job-{process.pid}", "jobs_dir": str(Path(cwd) / ".fake-job"),
                "wrapper_pid": process.pid, "wrapper_process": proc_identity.identify(process.pid), "argv": argv}


#: What each failing step is called in Start's reply.
SAID = {"commands-log": "commands.log", "job-record": "record of the start"}


@pytest.fixture()
def unloggable(request, tmp_path, monkeypatch):
    """A page server on a shared card where one step after an accepted launch fails: the run's commands.log line
    (the default), or the page's job record of the run."""

    step = getattr(request, "param", "commands-log")
    owner = tmp_path / "card-queue" / "OWNER"
    owner.parent.mkdir()
    owner.write_text("", encoding="utf-8")
    runner = SleeperRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    queue = server.api.queue
    queue.owner_file, queue.owner_tag, queue.pid = str(owner), "gui-test", 424_242
    queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                          "memory_used_mib": 512}], "processes": []}

    if step == "commands-log":
        def denied(*args, **kwargs):
            raise PermissionError(13, "Permission denied", "commands.log")

        monkeypatch.setattr(gui_api, "log_command", denied)
        monkeypatch.setattr(gui_jobs, "log_command", denied)
    else:
        written = gui_jobs.write_json

        def full(path, *args, **kwargs):
            if Path(path).name == gui_runs.JOB:
                raise OSError(28, "No space left on device", str(path))
            return written(path, *args, **kwargs)

        monkeypatch.setattr(gui_jobs, "write_json", full)
    try:
        yield server, runner, owner, SAID[step]
    finally:
        for process in runner.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        server.server_close()


@pytest.mark.parametrize("unloggable", list(SAID), indirect=True)
def test_a_queued_start_whose_bookkeeping_fails_leaves_the_line_once_and_keeps_its_card(unloggable):
    server, runner, owner, _ = unloggable
    queue = server.api.queue
    rundir = server.root / "q1"
    rundir.mkdir(parents=True)
    write_json(rundir / gui_runs.QUEUED, {"schema": MARKER_SCHEMA, "machine": LOCAL,
                                          "queued_utc": "2026-09-27T00:00:00Z", "held": None, "waiting": None})
    queue._save({"schema": QUEUE_SCHEMA, "order": ["q1"], "owned": []})
    first = queue.tick()
    running = runner.processes[0].poll() is None
    lines = owner.read_text(encoding="utf-8").splitlines()
    left = queue.order()
    # Once it has ended and the card is free, nothing starts it again.
    runner.processes[0].kill()
    runner.processes[0].wait(timeout=10)
    queue.tick()
    assert len(runner.launched) == 1, "the accepted start was launched a second time"
    assert first == ["q1"] and running
    assert left == [] and not (rundir / gui_runs.QUEUED).exists()
    # While it ran it held the card: the line was its wrapper's.
    assert len(lines) == 1 and f" pid {runner.processes[0].pid} " in lines[0], lines


@pytest.mark.parametrize("unloggable", list(SAID), indirect=True)
def test_start_says_started_when_only_a_step_after_it_failed(unloggable):
    server, runner, owner, said = unloggable
    make_run(server.root, "ready", plan=True)
    reply = server.api.handle("POST", "/api/runs/ready/start", {}, b"{}", "application/json")
    assert reply.status == 200, reply.body
    assert reply.body["message"].startswith("Started.") and said in reply.body["message"]
    assert len(runner.launched) == 1
    lines = owner.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and f" pid {runner.processes[0].pid} " in lines[0]


def test_a_queued_start_that_fails_after_its_launch_anywhere_is_not_started_again(unloggable, monkeypatch):
    # Past the page's own start: the card line's hand-over to the run's wrapper fails.  The forecast was started,
    # so the queue lets it go rather than starting it once more.
    import woof.machine_agent as machine_agent

    server, runner, owner, _ = unloggable

    def unwritable(*args, **kwargs):
        raise OSError(28, "No space left on device", "OWNER")

    monkeypatch.setattr(machine_agent, "retag_card", unwritable)
    queue = server.api.queue
    rundir = server.root / "q2"
    rundir.mkdir(parents=True)
    write_json(rundir / gui_runs.QUEUED, {"schema": MARKER_SCHEMA, "machine": LOCAL,
                                          "queued_utc": "2026-09-27T00:00:00Z", "held": None, "waiting": None})
    queue._save({"schema": QUEUE_SCHEMA, "order": ["q2"], "owned": []})
    first = queue.tick()
    runner.processes[0].kill()
    runner.processes[0].wait(timeout=10)
    queue.tick()
    assert len(runner.launched) == 1, "the started forecast was launched a second time"
    assert first == ["q2"] and queue.order() == [] and not (rundir / gui_runs.QUEUED).exists()
