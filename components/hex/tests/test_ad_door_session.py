# SPDX-License-Identifier: Apache-2.0
"""The session: a sensitivity program kept loaded ACROSS commands, and what a command reads once per plan.

THE GAP THESE HOLD SHUT: a later 3 km gradient asked as a new command took 399.7 s where the same gradient on the
program an earlier objective had left loaded took 283.7 s (card run of 2026-11-14): every command started its own
runner, and read the plan's 708 MB plan.json several times before it did. A program kept across commands must never
answer for another plan, build or setting, must never hand one command another's answer, must give the card back, and
must be replaced by a fresh start, with no word from the user, when it is gone.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_plan_facts as pf
from woof.hex import ad_session as s


@pytest.fixture(autouse=True)
def _stand_in_runner_on_hidden_cards(monkeypatch):
    # The battery hides the cards (conftest CUDA_VISIBLE_DEVICES=-1) and these tests start a stand-in runner that
    # opens no device, so the door's hidden-card refusal (refuse_hidden_card) is let through here only.
    from woof.hex import ad_doors as _d
    monkeypatch.setattr(_d, "refuse_hidden_card", lambda: None)

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"

# The runner's window mode with the loaded-program route (resident.rs), as in test_ad_door_speed.py, plus: a gradient
# takes FAKE_SLEEP seconds; a REQUEST whose out folder holds FAKE_DIE_ON ends the program with no answer (a crash).
FAKE_RUNNER = r'''#!PYTHON
import json, os, sys, time
from pathlib import Path
argv = sys.argv[1:]
assert argv[0] == "window", argv
words = {argv[i][2:]: argv[i + 1] for i in range(1, len(argv) - 1, 2)}
log = os.environ.get("FAKE_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps({"pid": os.getpid(), "words": sorted(words)}) + "\n")
MAY = {"out", "x0-planted", "objective", "dump-grad", "dump-state", "bstore", "spill-dir", "x0-dir", "x0-h"}
def gradient(w, name, index):
    out = Path(w["out"]); out.mkdir(parents=True, exist_ok=True)
    time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
    obj = json.loads(Path(w["objective"]).read_text())
    json.loads(Path(w["x0-dir"]).read_text())
    (out / "window.json").write_text(json.dumps({"objective": {"reverse_wall_s": 0.5, "name": obj.get("name")},
                                                 "words": w, "pid": os.getpid(),
                                                 "plan_seen": Path("plan.json").read_text(),
                                                 "resident": {"request": name, "index": index}}))
    return 0, None
rdir = words.get("resident")
if os.environ.get("FAKE_NO_RESIDENT"):
    rdir = None
name = words.get("resident-name", "r001")
plan_at_start = Path("plan.json").read_text()
rc, err = gradient(words, name, 1)
if rdir is None:
    sys.exit(rc)
rdir = Path(rdir)
idle = float(words.get("resident-idle-s", "900"))
def state(what, index):
    tmp = rdir / "state.tmp"
    tmp.write_text(json.dumps({"state": what, "pid": os.getpid(), "index": index, "served": index}))
    os.replace(tmp, rdir / "state.json")
def answer(name, rc, err, index):
    tmp = rdir / (name + ".done.tmp")
    tmp.write_text(json.dumps({"rc": rc, "error": err, "name": name, "index": index, "wall_s": 0.5, "pid": os.getpid()}))
    os.replace(tmp, rdir / (name + ".done"))
index = 1
answer(name, rc, err, index)
t0 = time.time()
state("idle", index)
while True:
    if (rdir / "stop").exists():
        state("ended", index); sys.exit(0)
    reqs = sorted(rdir.glob("*.req"))
    if reqs:
        p = reqs[0]; name = p.stem; taken = p.with_suffix(".taken"); os.rename(p, taken)
        req = json.loads(taken.read_text())
        index += 1
        state("busy", index)
        if os.environ.get("FAKE_DIE_ON") and os.environ["FAKE_DIE_ON"] in req.get("out", ""):
            os._exit(3)
        bad = [k for k in req if k not in MAY]
        if bad:
            answer(name, 1, "request words outside the request list: " + ",".join(bad), index); sys.exit(1)
        merged = dict(words); merged.update({k: v for k, v in req.items() if v != ""})
        rc, err = gradient(merged, name, index)
        answer(name, rc, err, index)
        t0 = time.time()
        state("idle", index)
        continue
    if time.time() - t0 > idle:
        state("ended", index); sys.exit(0)
    time.sleep(0.02)
'''


def _alive(pid: int) -> bool:
    return s._start_ticks(pid) is not None


def _wait_dead(pid: int, seconds: float = 10.0) -> bool:
    t0 = time.time()
    while _alive(pid) and time.time() - t0 < seconds:
        time.sleep(0.05)
    return not _alive(pid)


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A stand-in machine: the runner, a plan folder, a sessions folder of this test's own, and no card to read."""
    runner = tmp_path / "ad-binder"
    runner.write_text(FAKE_RUNNER.replace("#!PYTHON", "#!" + sys.executable))
    runner.chmod(0o755)
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "plan.json").write_text(json.dumps({"allocations": [], "physics": {"dt_seconds": 15}, "v": 1}))
    monkeypatch.setenv(s.SESSIONS_ENV, str(tmp_path / "sessions"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "starts.log"))
    for var in s.SCHEDULER_ENV + (s.MODE_ENV,):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(d, "refuse_no_local_gpu", lambda: None)
    monkeypatch.setattr(d, "process_card_mib", lambda pid: {"0000:01:00.0": 1000})
    monkeypatch.setattr(d, "admit_card", lambda *a, **k: {"free_mib": 96000})
    # the operator key of these stand-ins: the plan's file and its stamps, and the runner (ad_doors.check_key reads
    # the same from a real context, with the kernel set, the images and the row's words)
    monkeypatch.setattr(d, "check_key", lambda ctx: {"plan_sha256": d._sha256_file(ctx.plan / "plan.json"),
                                                     "plan_files": d._plan_files_sum(ctx.plan),
                                                     "runner_sha256": d._sha256_file(ctx.runner)})
    b = SimpleNamespace(tmp=tmp_path, runner=runner, plan=plan, log=tmp_path / "starts.log", pids=[], n=0)
    yield b
    monkeypatch.setenv(s.SESSIONS_ENV, str(tmp_path / "sessions"))
    s.stop_sessions(None, wait_s=5.0)
    for pid in b.pids:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


def _command(box, table, name="first", door="sensitivity", steps=3, idle=30.0, session=None, plan=None, **argwords):
    """One command's context as the door builds it up to the session's opening (no card, no asset root)."""
    box.n += 1
    out = box.tmp / f"cmd{box.n}"
    out.mkdir()
    (out / "empty-move.json").write_text(json.dumps({"name": "empty move", "terms": []}))
    objective = out / "objective.json"
    objective.write_text(json.dumps({"name": name, "terms": [[1, 0, 1.0]]}))
    args = argparse.Namespace(session=session, session_idle_s=idle, session_name=None, wait_card=0.0,
                              checked_forward=False, boundary_layer=None, **argwords)
    ctx = SimpleNamespace(args=args, door=door, cards=["0"], card="0", row_name=STORM, row=table["rows"][STORM],
                          table=table, steps=steps, plan=plan or box.plan, runner=box.runner, need_mib=lambda: 1)
    ctx.argv = [str(box.runner), "window", "--config", "plan.json", "--steps", str(steps), "--out",
                str(out / "window"), "--bstore", str(out / "window" / "bstore"), "--objective", str(objective),
                "--dump-grad", "3,46", "--checkpoints", "sqrt", "--x0-dir", str(out / "empty-move.json"),
                "--tangent-only", "1", "--no-tangent", "1"]
    ctx.out_window = out / "window"
    ctx.session = s.open_for(ctx)
    return ctx


def _gradient(box, table, ctx):
    line = d.gradient_line(ctx, ctx.args) if hasattr(ctx, "split") else {
        **table["doors"]["sensitivity"]["gradient_line"], "row": table["rows"][STORM]["gradient_line"]}
    program = ctx.session.program(line)
    assert program is not None
    doc = program.start(ctx.argv, {"AD_SNAP_REUSE": "1"}, ctx.plan, "0", "g", ctx.out_window)
    box.pids.append(program.proc.pid)
    return program, doc


@pytest.fixture
def table():
    return d.load_table()


# ---------------------------------------------------------------------------------------------- attach
def test_the_first_command_starts_fresh_and_leaves_its_program_and_the_next_attaches(box, table, capsys):
    one = _command(box, table, "first")
    first_line = capsys.readouterr().out.splitlines()[0]
    assert "session card0: STARTED FRESH" in first_line and "--session-stop card0" in first_line
    program, doc1 = _gradient(box, table, one)
    one.session.finish()
    said = capsys.readouterr().out
    assert "left loaded" in said and "30 s" in said
    pid = program.proc.pid
    assert _alive(pid), "the program ended with its command"
    assert doc1["objective"]["name"] == "first" and doc1["loaded_program"]["attached"] is False
    folder = box.tmp / "sessions" / "card0"
    assert stat.S_IMODE(os.lstat(folder).st_mode) == 0o700 and stat.S_IMODE(os.lstat(folder.parent).st_mode) == 0o700
    kept = json.loads((folder / "session.json").read_text())
    assert kept["pid"] == pid and kept["key"]["operator"]["plan_sha256"] and kept["idle_s"] == 30
    assert one.session.block()["left_loaded"] is True and one.session.block()["how"] == "started fresh"
    # its start move is the session's own copy: the first command's folder may be removed
    assert f"--x0-dir {folder / 'empty-move.json'}" in (folder / "cmd").read_text()

    two = _command(box, table, "second")
    first_line = capsys.readouterr().out.splitlines()[0]
    assert "session card0: ATTACHED" in first_line and f"pid {pid}" in first_line
    program2, doc2 = _gradient(box, table, two)
    two.session.finish()
    assert doc2["objective"]["name"] == "second" and doc2["pid"] == pid, "the later command started its own program"
    assert doc2["loaded_program"]["attached"] is True and doc2["loaded_program"]["session"] == "card0"
    assert len(box.log.read_text().splitlines()) == 1
    assert program2.stage_label(True, "g") == "session card0 attached"
    assert two.session.block()["how"] == "attached" and two.session.block()["left_loaded"] is True
    assert not list(folder.glob("*.done")) and not list(folder.glob("*.taken")) and not list(folder.glob("*.req"))
    # the answer is the request's own: the runner names the request it served, and the door takes no other
    assert doc2["resident"]["request"].endswith("-g") and doc2["resident"]["request"] != doc1["resident"]["request"]


def test_an_answer_under_another_requests_name_is_not_taken(box, table):
    one = _command(box, table, "first")
    program, _ = _gradient(box, table, one)
    with pytest.raises(d.AdDoorError, match="not taken"):
        program._tidy("q-mine", {"resident": {"request": "q-someone-elses"}, "args": {"out": "x"}}, box.tmp / "x")


# ------------------------------------------------------------------------- refusal on a changed plan or setting
def test_a_changed_plan_is_never_answered_by_the_loaded_program(box, table, capsys):
    one = _command(box, table, "first")
    program, doc1 = _gradient(box, table, one)
    one.session.finish()
    old = program.proc.pid
    (box.plan / "plan.json").write_text(json.dumps({"allocations": [], "physics": {"dt_seconds": 15}, "v": 2}))
    capsys.readouterr()
    two = _command(box, table, "second")          # the same command words: the plan's files decide
    program2, doc2 = _gradient(box, table, two)
    said = capsys.readouterr().out
    assert "NOT ATTACHED" in said and "plan" in said and "STARTED FRESH" in said
    assert doc2["pid"] != old and _wait_dead(old), "the program loaded for the old plan was left on the card"
    assert json.loads(doc2["plan_seen"])["v"] == 2, "a gradient of the old plan was handed to the new command"
    assert two.session.block()["how"] == "started fresh" and "plan" in two.session.block()["why"]
    assert len(box.log.read_text().splitlines()) == 2


def test_a_file_of_the_plan_touched_without_a_change_of_bytes_is_refused_too(box, table, capsys):
    one = _command(box, table)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    (box.plan / "extra-payload.bin").write_bytes(b"new file in the plan")
    two = _command(box, table, "second")
    _, doc2 = _gradient(box, table, two)
    assert "NOT ATTACHED" in capsys.readouterr().out and doc2["pid"] != program.proc.pid


def test_another_runner_build_or_runner_word_is_refused(box, table, capsys):
    one = _command(box, table)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    two = _command(box, table, "second")
    two.argv[two.argv.index("--checkpoints") + 1] = "all"      # a runner word that is not a request's own
    _, doc2 = _gradient(box, table, two)
    assert "runner words differ" in capsys.readouterr().out and doc2["pid"] != program.proc.pid
    two.session.finish()
    box.runner.write_text(box.runner.read_text() + "\n# another build\n")
    three = _command(box, table, "third")
    three.argv[three.argv.index("--checkpoints") + 1] = "all"
    _, doc3 = _gradient(box, table, three)
    assert "runner differ" in capsys.readouterr().out and doc3["pid"] != doc2["pid"]
    three.session.finish()
    assert three.session.block()["program_host_mib"] > 0
    (box.runner.parent / "libhexad_phys.so").write_bytes(b"another build of the library the runner loads")
    four = _command(box, table, "fourth")
    four.argv[four.argv.index("--checkpoints") + 1] = "all"
    _, doc4 = _gradient(box, table, four)
    assert "runner libraries differ" in capsys.readouterr().out and doc4["pid"] != doc3["pid"]


def test_other_command_words_start_fresh_in_the_first_line(box, table, capsys):
    one = _command(box, table, steps=3)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    capsys.readouterr()
    two = _command(box, table, "second", steps=59)
    first_line = capsys.readouterr().out.splitlines()[0]
    assert "STARTED FRESH" in first_line and "other words (steps)" in first_line
    assert _wait_dead(program.proc.pid) and two.session.live is None


def test_the_request_words_and_the_store_placement_are_not_part_of_the_key(box, table):
    one = _command(box, table)
    line = {"empty_move": {"name": "empty move", "terms": []}}
    a = s.program_key(one, one.argv + ["--bstore-ram-gib", "150"], {"AD_X": "1"}, line)
    other = [w.replace("cmd1", "elsewhere") for w in one.argv] + ["--bstore-ram-gib", "97"]
    assert s.program_key(one, other, {"AD_X": "1"}, line) == a
    assert s.key_differs(a, s.program_key(one, one.argv + ["--bstore-ram-gib", "150"], {"AD_X": "2"}, line)) == [
        "runner environment"]
    steps = list(one.argv)
    steps[steps.index("--steps") + 1] = "59"
    assert s.key_differs(a, s.program_key(one, steps + ["--bstore-ram-gib", "150"], {"AD_X": "1"}, line)) == [
        "runner words"]


# ------------------------------------------------------------------------------------------------ time-out
def test_an_idle_session_ends_by_itself_and_the_next_command_starts_fresh(box, table, capsys):
    one = _command(box, table, idle=0.4)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    assert "0.4 s" in capsys.readouterr().out
    assert _wait_dead(program.proc.pid, 10.0), "the idle program did not end: the card would stay held"
    assert s.list_sessions() == []
    two = _command(box, table, "second", idle=30.0)
    assert "STARTED FRESH: no program was loaded" in capsys.readouterr().out.splitlines()[0]
    _, doc2 = _gradient(box, table, two)
    assert doc2["objective"]["name"] == "second" and doc2["pid"] != program.proc.pid


# --------------------------------------------------------------------------------------- crash of the session
def test_a_session_that_crashed_between_commands_is_not_found(box, table, capsys):
    one = _command(box, table)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    os.kill(program.proc.pid, signal.SIGKILL)
    program.proc.wait(timeout=10)
    capsys.readouterr()
    two = _command(box, table, "second")
    assert "STARTED FRESH: no program was loaded" in capsys.readouterr().out.splitlines()[0]
    _, doc2 = _gradient(box, table, two)
    assert doc2["objective"]["name"] == "second"


def test_a_session_that_dies_while_it_serves_is_replaced_by_a_fresh_start(box, table, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_DIE_ON", "cmd2")
    one = _command(box, table)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    monkeypatch.delenv("FAKE_DIE_ON")              # the replacement program is healthy
    capsys.readouterr()
    two = _command(box, table, "second")
    assert "ATTACHED" in capsys.readouterr().out.splitlines()[0]
    program2, doc2 = _gradient(box, table, two)
    err = capsys.readouterr().err
    assert "ended before it answered" in err and "STARTED FRESH" in err
    assert doc2["objective"]["name"] == "second" and doc2["pid"] != program.proc.pid
    assert doc2["loaded_program"]["attached"] is False
    assert two.session.block()["how"] == "started fresh" and "ended before it answered" in two.session.block()["why"]
    assert not list((box.tmp / "sessions" / "card0").glob("*.req"))
    assert _alive(program2.proc.pid), "the replacement is the session now"
    assert len(box.log.read_text().splitlines()) == 2


def test_a_pid_taken_by_another_process_is_not_a_session(box, table):
    one = _command(box, table)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    folder = box.tmp / "sessions" / "card0"
    doc = json.loads((folder / "session.json").read_text())
    doc["pid"], doc["start_ticks"] = os.getpid(), s._start_ticks(os.getpid())   # a live process that serves nothing
    (folder / "session.json").write_text(json.dumps(doc))
    assert s.read_live(folder.parent, "card0") is None
    doc["pid"], doc["start_ticks"] = program.proc.pid, 12345                     # the right pid, another start time
    (folder / "session.json").write_text(json.dumps(doc))
    assert s.read_live(folder.parent, "card0") is None


# ------------------------------------------------------------------------------------- two commands at once
DRIVER = r'''
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
from woof.hex import ad_doors as d
from woof.hex import ad_session as s
tmp, name, runner, plan = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
d.refuse_no_local_gpu = lambda: None
d.refuse_hidden_card = lambda: None
d.process_card_mib = lambda pid: {}
d.admit_card = lambda *a, **k: {"free_mib": 96000}
d.check_key = lambda ctx: {"plan_sha256": d._sha256_file(ctx.plan / "plan.json"), "plan_files": d._plan_files_sum(ctx.plan)}
table = d.load_table()
STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
out = tmp / name
out.mkdir()
(out / "empty-move.json").write_text(json.dumps({"name": "empty move", "terms": []}))
(out / "objective.json").write_text(json.dumps({"name": name, "terms": []}))
args = argparse.Namespace(session=None, session_idle_s=30.0, session_name=None, wait_card=0.0, checked_forward=False,
                          boundary_layer=None)
ctx = SimpleNamespace(args=args, door="sensitivity", cards=["0"], card="0", row_name=STORM, row=table["rows"][STORM],
                      table=table, steps=3, plan=plan, runner=runner, need_mib=lambda: 1)
ctx.session = s.open_for(ctx)
argv = [str(runner), "window", "--config", "plan.json", "--steps", "3", "--out", str(out / "window"), "--bstore",
        str(out / "window" / "bstore"), "--objective", str(out / "objective.json"), "--dump-grad", "3", "--checkpoints",
        "sqrt", "--x0-dir", str(out / "empty-move.json"), "--tangent-only", "1", "--no-tangent", "1"]
line = {**table["doors"]["sensitivity"]["gradient_line"], "row": table["rows"][STORM]["gradient_line"]}
doc = ctx.session.program(line).start(argv, {}, plan, "0", "g", out / "window")
ctx.session.finish()
print("RESULT " + json.dumps({"name": doc["objective"]["name"], "pid": doc["pid"], "how": ctx.session.block()["how"]}))
'''


def test_two_commands_at_once_share_one_program_and_each_gets_its_own_answer(box, table, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "1.0")
    driver = box.tmp / "driver.py"
    driver.write_text(DRIVER)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    procs = []
    for name in ("alpha", "beta"):
        procs.append(subprocess.Popen([sys.executable, str(driver), str(box.tmp), name, str(box.runner),
                                       str(box.plan)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True))
        time.sleep(0.25)            # alpha holds the session lock when beta arrives
    outs = [p.communicate(timeout=60) for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outs
    results = {}
    for name, (out, _) in zip(("alpha", "beta"), outs):
        results[name] = json.loads([x for x in out.splitlines() if x.startswith("RESULT ")][0][7:])
        results[name]["out"] = out
    box.pids.append(results["alpha"]["pid"])
    assert results["alpha"]["name"] == "alpha" and results["beta"]["name"] == "beta", "an answer went to the other command"
    assert results["alpha"]["pid"] == results["beta"]["pid"] and len(box.log.read_text().splitlines()) == 1
    assert sorted(r["how"] for r in results.values()) == ["attached", "started fresh"]
    second = next(r for r in results.values() if r["how"] == "attached")     # whichever came second waited
    assert "is serving another command" in second["out"] and "waiting for it" in second["out"]
    lines = second["out"].splitlines()
    assert "ATTACHED" in lines[1], "the waiting command did not attach to the program the first one left"


# ------------------------------------------------------------------- a shared machine: the folder, stop, list
def test_the_sessions_folder_is_private_or_no_session_is_used(box, table, capsys, monkeypatch):
    real = box.tmp / "elsewhere"
    real.mkdir()
    link = box.tmp / "linked-sessions"
    link.symlink_to(real)
    monkeypatch.setenv(s.SESSIONS_ENV, str(link))
    ctx = _command(box, table)
    said = capsys.readouterr().out
    assert "session: not used" in said and "STARTED FRESH" in said.splitlines()[0]
    assert ctx.session.mode == "off" and ctx.session.program({"loaded_program": {}}) is None
    wide = box.tmp / "wide"
    wide.mkdir(mode=0o777)
    os.chmod(wide, 0o777)
    assert stat.S_IMODE(os.lstat(s.private_dir(wide)).st_mode) == 0o700


def test_a_refused_command_leaves_no_lock_file(box, table, capsys):
    """A command refused before its checks pass loaded no program: its card lock file is removed with the lock. A
    command that leaves a program loaded keeps the file, and a refusal beside that program does not remove it."""
    lock = box.tmp / "sessions" / "card0.lock"
    args = argparse.Namespace(session=None, session_idle_s=30.0, session_name=None, wait_card=0.0,
                              checked_forward=False, boundary_layer=None)
    ctx = SimpleNamespace(args=args, door="sensitivity", cards=["0"], card="0", row_name=STORM,
                          row=table["rows"][STORM], table=table, steps=3, plan=box.plan, runner=box.runner,
                          need_mib=lambda: 1)
    use = s.open_for(ctx, defer=True)
    assert use.mode == "use" and lock.is_file(), "the command did not take its card lock"
    use.refused()
    assert capsys.readouterr().out == "", "a refused command said it started a program"
    assert not lock.exists(), "a refused command left its card lock file"
    assert list((box.tmp / "sessions").iterdir()) == []
    # a command that leaves its program loaded keeps the file, and a refusal beside that program leaves it
    one = _command(box, table, "first")
    program, _ = _gradient(box, table, one)
    one.session.finish()
    assert lock.is_file() and _alive(program.proc.pid), "a loaded program lost its lock file"
    beside = s.open_for(ctx, defer=True)
    beside.refused()
    assert lock.is_file() and _alive(program.proc.pid), "a refusal beside a loaded program removed its lock file"
    s.stop_sessions(None, wait_s=5.0)
    assert _wait_dead(program.proc.pid)
    assert not lock.exists(), "a stopped session left its lock file"
    # a command that held the removed file open does not hold the lock of the next command's file
    held = s.open_for(ctx, defer=True)
    fd = os.open(lock, os.O_RDWR)
    held.refused()
    assert not lock.exists()
    again = s.open_for(ctx, defer=True)
    assert lock.is_file() and os.path.samestat(os.fstat(again._fd), os.stat(lock))
    assert not os.path.samestat(os.fstat(fd), os.stat(lock))
    os.close(fd)
    again.refused()
    assert not lock.exists()


def test_a_session_is_listed_and_stopped_by_name(box, table, capsys):
    one = _command(box, table)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    rows = s.list_sessions()
    assert [r["name"] for r in rows] == ["card0"] and rows[0]["pid"] == program.proc.pid and rows[0]["steps"] == 3
    parser = argparse.ArgumentParser()
    s.add_session_args(parser)
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--session-stop", "card0"])
    assert stopped.value.code == 0 and "session card0" in capsys.readouterr().out
    assert _wait_dead(program.proc.pid) and s.list_sessions() == []
    with pytest.raises(SystemExit):
        parser.parse_args(["--session-list"])
    assert "no session is loaded" in capsys.readouterr().out


def test_a_door_that_runs_its_own_program_stops_the_session_on_its_card(box, table, capsys):
    one = _command(box, table)
    program, _ = _gradient(box, table, one)
    one.session.finish()
    capsys.readouterr()
    other = _command(box, table, door="tangent")
    assert "session card0" in capsys.readouterr().out and other.session.mode == "off"
    assert _wait_dead(program.proc.pid), "the session kept the card a tangent command needs"
    off = _command(box, table, session="off")
    assert off.session.mode == "off" and off.session.block()["on"] is False


def test_a_batch_job_keeps_no_session_unless_asked(box, table, capsys, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    job = _command(box, table)
    assert job.session.mode == "off" and "batch job" in capsys.readouterr().out
    asked = _command(box, table, session="on")
    assert asked.session.mode == "use"
    asked.session.finish()


def test_a_window_of_the_commands_own_ends_the_session_and_the_next_gradient_starts_it_again(box, table, capsys):
    one = _command(box, table)
    program, _ = _gradient(box, table, one)
    program.end("this command's selfdot window needed the card")
    assert _wait_dead(program.proc.pid) and one.session.block()["left_loaded"] is False
    assert "selfdot" in one.session.block()["ended"]
    out2 = box.tmp / "again" / "window"
    objective = box.tmp / "again" / "objective.json"
    objective.parent.mkdir()
    objective.write_text(json.dumps({"name": "again", "terms": []}))
    doc = program.request("g", {"out": str(out2), "objective": str(objective), "dump-grad": "3",
                                "bstore": str(out2 / "bstore")}, out2)
    box.pids.append(program.proc.pid)
    assert doc["objective"]["name"] == "again" and "STARTED FRESH again" in capsys.readouterr().out
    assert one.session.block()["left_loaded"] is True and one.session.block()["ended"] is None


# ----------------------------------------------------------------------------- through the door's own functions
def test_one_objective_through_the_door_attaches_on_the_second_command(tmp_path, table, box, monkeypatch, capsys):
    from test_ad_door_speed import _ctx, _objective

    def command(name):
        ctx = _ctx(tmp_path, table)
        ctx.out = tmp_path / name
        ctx.out.mkdir()
        ctx.args.session, ctx.args.session_idle_s, ctx.args.session_name = None, 30.0, None
        ctx.runner, ctx.plan, ctx.arch_assets, ctx.kset_assets, ctx.assets = box.runner, box.plan, {"x": 1}, {}, {}
        for key in set(ctx.row["asset_args"].values()) | set(ctx.row["asset_env"].values()):
            p = tmp_path / "assets-files" / key
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
            ctx.assets[key] = p
        ctx.stages, ctx.windows, ctx.started = [], [], time.time()
        ctx.need_mib, ctx.class_label = (lambda: 1), (lambda: None)
        for meth in ("window_line", "window_loaded", "window_loaded_fallback", "_loaded_doc", "stage"):
            setattr(ctx, meth, types.MethodType(getattr(d.Context, meth), ctx))
        ctx.fmap = SimpleNamespace(resolve=lambda *a, **k: {"name": "rho", "copies": [(3, 0)], "units": "kg"},
                                   fields={}, doc={"plan_sha256": "ab" * 32})
        ctx.session = s.open_for(ctx)
        return ctx

    monkeypatch.setattr(d, "_sensitivity_inputs", lambda c: ([{"name": "rho", "copies": [(3, 0)]}], [3, 46]))
    monkeypatch.setattr(d, "host_available_mib", lambda: 200 * 1024)
    monkeypatch.setattr(d, "objective_file", lambda c, spec, dest: (
        {"name": c.out.name, "metric": "theta_m", "units": "K"}, _objective(dest, c.out.name), {"how": "made"}))
    monkeypatch.setattr(d, "strict_check", lambda c, *a: {"how": "stored", "pass": True})
    answered = []
    monkeypatch.setattr(d, "_sensitivity_answer", lambda c, args, objective, inputs, doc, strict: (
        answered.append({"doc": doc, "extra": dict(c.receipt_extra), "stages": [x["stage"] for x in c.stages]})
        or (0, {"rec": {}})))
    one = command("one")
    assert d._sensitivity(one, one.args)[0] == 0
    one.session.finish()
    two = command("two")
    assert d._sensitivity(two, two.args)[0] == 0
    two.session.finish()
    box.pids.append(answered[0]["doc"]["pid"])
    assert answered[0]["doc"]["pid"] == answered[1]["doc"]["pid"] and len(box.log.read_text().splitlines()) == 1
    assert [a["doc"]["objective"]["name"] for a in answered] == ["one", "two"]
    assert answered[0]["extra"]["session"]["how"] == "started fresh" and answered[1]["extra"]["session"]["how"] == "attached"
    assert answered[0]["stages"][-1] == "window (window, session card0 started fresh)"
    assert answered[1]["stages"][-1] == "window (window, session card0 attached)"
    assert answered[1]["doc"]["door_validity"]["boundaries_compared"] is False
    out = capsys.readouterr().out
    assert out.count("STARTED FRESH") == 1 and out.count("ATTACHED") == 1 and out.count("left loaded") == 2
    cmd = (tmp_path / "sessions" / "card0" / "cmd").read_text()
    assert "--resident " in cmd and "--resident-idle-s 30" in cmd and "--tape-gib 33" in cmd
    # a runner without the loaded-program route: its one answer is kept, no session, and the command says so
    s.stop_sessions(None, wait_s=5.0)
    monkeypatch.setenv("FAKE_NO_RESIDENT", "1")
    three = command("three")
    assert d._sensitivity(three, three.args)[0] == 0
    three.session.finish()
    assert answered[2]["doc"]["objective"]["name"] == "three" and answered[2]["extra"]["session"]["left_loaded"] is False
    assert "serves no requests" in answered[2]["extra"]["session"]["ended"]


def test_the_sensitivity_parser_has_the_session_words():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["sensitivity", "--out", "x", "--session", "off", "--session-idle-s", "45",
                              "--session-name", "loop-1"])
    assert args.session == "off" and args.session_idle_s == 45.0 and args.session_name == "loop-1"
    assert parser.parse_args(["sensitivity", "--out", "x"]).session is None
    with pytest.raises(s.SessionFolderError):
        s.session_name(argparse.Namespace(session_name="../up"), ["0"])
    assert s.session_name(None, ["0"]) == "card0" and s.session_name(None, ["GPU-1a/2"]) == "cardGPU_1a_2"


def test_memory_the_session_holds_is_counted_as_the_commands_own(box, table, monkeypatch, tmp_path):
    monkeypatch.setattr(d, "process_card_mib", lambda pid: {"0000:01:00.0": 89396})
    one = _command(box, table)
    _gradient(box, table, one)
    one.session.finish()
    assert d._SESSION_HELD == {"card": {}, "host": 0}
    two = _command(box, table, "second")
    assert d._SESSION_HELD["card"] == {"0": 89396} and d._SESSION_HELD["host"] >= 0
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.parent.mkdir()
    smi.write_text("#!/bin/sh\necho 7856\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{smi.parent}{os.pathsep}{os.environ['PATH']}")
    assert d.free_mib("0") == 7856 + 89396 and d.free_mib("1") == 7856
    two.session.finish()
    assert d.free_mib("0") == 7856


# ------------------------------------------------------------------------------- once per plan: the kept facts
@pytest.fixture
def facts_plan(tmp_path, monkeypatch):
    import gzip
    import numpy as np
    monkeypatch.setenv(pf.FACTS_ENV, str(tmp_path / "facts"))
    monkeypatch.setenv(d.OBJECTIVES_ENV, str(tmp_path / "objectives"))
    pf._MEMO.clear()
    pf._SUMS.clear()
    plan = tmp_path / "plan"
    (plan / "recording" / "payloads").mkdir(parents=True)
    root = np.arange(10, dtype="<f4").reshape(2, 5)                     # a field with a padding column (4 cells)
    (plan / "recording" / "payloads" / "root.gz").write_bytes(gzip.compress(root.tobytes()))
    (plan / "recording" / "payloads" / "copy.bin").write_bytes(np.ascontiguousarray(root[:, :4]).tobytes())
    (plan / "recording" / "payloads" / "other.bin").write_bytes(np.zeros(8, "<f4").tobytes())
    doc = {"allocations": [
        {"id": 3, "bytes": 40, "initial": {"path": "recording/payloads/root.gz", "encoding": "gzip"}},
        {"id": 7, "bytes": 32, "initial": {"path": "recording/payloads/copy.bin"}},
        {"id": 9, "bytes": 32, "initial": {"path": "recording/payloads/other.bin"}}],
        "physics": {"dt_seconds": 15}, "modules": [], "blocks": {}}
    (plan / "plan.json").write_text(json.dumps(doc))
    reads = []
    real = Path.read_text

    def read_text(self, *a, **k):
        if self.name == "plan.json":
            reads.append(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    return SimpleNamespace(plan=plan, doc=doc, reads=reads, tmp=tmp_path)


def _map(fp):
    doc = {"plan_sha256": d._plan_facts(fp.plan).sha256(), "fields": {"theta": {"allocation": 3, "shape": [2, 5]}}}
    return d.FieldMap(doc, doc["plan_sha256"], fp.plan / "alloc-map.json", mesh={"cells": 4, "edges": 99}, plan=fp.plan)


def test_the_plans_facts_are_read_once_per_plan_and_give_the_same_answers(facts_plan):
    fp = facts_plan
    first = {"dt": d._plan_dt(fp.plan), "allocs": d.PlanState(fp.plan).allocs, "found": _map(fp).found,
             "copies": _map(fp).fields["theta"]["copies"], "sha": d._plan_facts(fp.plan).sha256()}
    assert first["dt"] == 15.0 and sorted(first["allocs"]) == [3, 7, 9] and first["allocs"][7] == fp.doc["allocations"][1]
    assert first["found"] == [{"root": 3, "copies": [7], "shape": [2, 4],
                               "rule": "initial payload equals the root without its padding column"}]
    assert first["copies"] == [(3, 0), (7, 0)] and first["sha"] == d._sha256_file(fp.plan / "plan.json")
    assert len(fp.reads) == 1, "plan.json was read more than once by one process"
    assert d._plan_facts(fp.plan).how == "made"
    # a later command (another process: nothing in memory) reads no plan.json and no payload, and answers the same
    pf._MEMO.clear()
    fp.reads.clear()
    raw = d.PlanState.raw
    d.PlanState.raw = lambda self, alloc: pytest.fail("a later command read a start payload for the padding search")
    try:
        later = {"dt": d._plan_dt(fp.plan), "allocs": d.PlanState(fp.plan).allocs, "found": _map(fp).found,
                 "copies": _map(fp).fields["theta"]["copies"], "sha": d._plan_facts(fp.plan).sha256()}
    finally:
        d.PlanState.raw = raw
    assert later == first and fp.reads == [] and d._plan_facts(fp.plan).how == "kept"


def test_a_changed_plan_makes_its_facts_again(facts_plan):
    fp = facts_plan
    assert d._plan_dt(fp.plan) == 15.0 and _map(fp).found
    changed = dict(fp.doc, physics={"dt_seconds": 20}, allocations=fp.doc["allocations"][:1] + fp.doc["allocations"][2:])
    (fp.plan / "plan.json").write_text(json.dumps(changed))
    assert d._plan_dt(fp.plan) == 20.0, "a kept fact was taken for a changed plan (same process)"
    assert sorted(d.PlanState(fp.plan).allocs) == [3, 9] and _map(fp).found == []
    pf._MEMO.clear()
    assert d._plan_dt(fp.plan) == 20.0 and d._plan_facts(fp.plan).how == "kept"
    # another build of the door programs makes them again too
    pf._MEMO.clear()
    pf._CODE_SUM[:] = ["another build"]
    try:
        assert d._plan_dt(fp.plan) == 20.0 and d._plan_facts(fp.plan).how == "made"
    finally:
        pf._CODE_SUM.clear()


def test_a_facts_folder_that_cannot_be_written_changes_no_answer(facts_plan, monkeypatch):
    fp = facts_plan
    blocker = fp.tmp / "a-file"
    blocker.write_text("not a folder")
    monkeypatch.setenv(pf.FACTS_ENV, str(blocker / "sub"))
    assert d._plan_dt(fp.plan) == 15.0 and sorted(d.PlanState(fp.plan).allocs) == [3, 7, 9] and _map(fp).found
    assert d._plan_dt(fp.tmp / "no-such-plan") is None


def test_the_operators_file_sums_are_kept_by_stamp(facts_plan, monkeypatch):
    fp = facts_plan
    folder = fp.tmp / "kset"
    folder.mkdir()
    (folder / "a.cubin").write_bytes(b"a" * 100)
    (folder / "b.cubin").write_bytes(b"b" * 100)
    first = d._content_sum(folder)
    assert first == d._content_sum_read(folder) and d._content_sum(folder / "a.cubin") == d._sha256_file(folder / "a.cubin")
    pf._SUMS.clear()
    monkeypatch.setattr(d, "_sha256_file", lambda p: pytest.fail("an unchanged file was read again"))
    assert d._content_sum(folder) == first
    monkeypatch.undo()
    monkeypatch.setenv(pf.FACTS_ENV, str(fp.tmp / "facts"))
    (folder / "b.cubin").write_bytes(b"c" * 101)
    assert d._content_sum(folder) != first and d._content_sum(folder) == d._content_sum_read(folder)
    assert d._content_sum(fp.tmp / "absent") is None and d._content_sum(None) is None


def test_the_receipt_line_names_the_seconds_before_the_clock():
    assert d.used_line({"card_peak_mib": 1, "host_peak_mib": 2, "wall_s": 3.0}).endswith("wall 3.0 s")
    assert d.used_line({"card_peak_mib": 1, "host_peak_mib": 2, "wall_s": 3.0, "before_clock_s": 1.2}).endswith(
        "wall 3.0 s (and 1.2 s before that clock started)")
    age = pf.process_age_s()
    assert age is not None and 0.0 <= age < 3600.0
