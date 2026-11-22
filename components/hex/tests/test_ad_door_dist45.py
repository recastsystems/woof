"""dist45 door facts (processors only): the forward kept between gradients of one forecast (which commands keep it,
the host memory the command counts for it, what the command prints for a gradient that took the kept forward and for
one that ran its own, with a stand-in program that serves requests as the runner does), the exact route on the
default route's kernel set, the one-line warning for a long exact window, the session's quick comparison holding the
cloud word, and the measured cautions the commands print."""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_session
from woof.hex.ad import assets


@pytest.fixture(autouse=True)
def _stand_in_runner_on_hidden_cards(monkeypatch):
    # The battery hides the cards (conftest CUDA_VISIBLE_DEVICES=-1) and these tests start a stand-in runner that
    # opens no device, so the door's hidden-card refusal (refuse_hidden_card) is let through here only.
    from woof.hex import ad_doors as _d
    monkeypatch.setattr(_d, "refuse_hidden_card", lambda: None)

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
R120 = "hrrrv5-x1.40962"
KS = "storm-bs5-lean2-astra2-dynE5-sp74"


def _row():
    return d.load_table()["rows"][STORM]


def _ns(**kw):
    base = dict(plan=None, init=None, boundary_layer=None, cloud_derivative=None, checked_forward=False, row=STORM,
                mesh=None, keep_forward=None, objectives=None)
    base.update(kw)
    return SimpleNamespace(**base)


USE, OFF = SimpleNamespace(mode="use"), SimpleNamespace(mode="off")


def test_which_commands_keep_the_forward():
    row = _row()
    on = d.keep_forward_plan(row, _ns(), True, USE)                       # a later command can attach
    assert on["on"] and on["env"]["AD_KEEP_FORWARD"] == "2" and on["route"] == "tape-free"
    assert d.keep_forward_plan(row, _ns(objectives=Path("o.json")), True, OFF)["on"]   # several targets, one program
    one = d.keep_forward_plan(row, _ns(), True, OFF)                     # one target, the program ends with it
    assert not one["on"] and one["offered"] and "no later gradient" in one["why"]
    off = d.keep_forward_plan(row, _ns(keep_forward="off"), True, USE)
    assert not off["on"] and off["why"] == "--keep-forward off"
    exact = d.keep_forward_plan(row, _ns(boundary_layer="taped"), True, USE)   # not on the exact route in this build
    assert not exact["on"] and not exact["offered"] and "12 steps" in exact["why"]
    assert not d.keep_forward_plan(row, _ns(), False, USE)["on"]         # the gradient line off (--checked-forward)
    assert not d.keep_forward_plan(d.load_table()["rows"][R120], _ns(row=R120), True, USE)["on"]
    table = d.load_table()
    d.refuse_speed_words(_ns(keep_forward="off"), table)
    d.refuse_speed_words(_ns(keep_forward="on"), table)
    with pytest.raises(d.AdDoorError, match="--keep-forward on: row .*boundary layer taped: the forward is not kept"):
        d.refuse_speed_words(_ns(keep_forward="on", boundary_layer="taped"), table)
    with pytest.raises(d.AdDoorError, match="--keep-forward off: row hrrrv5-x1.40962"):
        d.refuse_speed_words(_ns(keep_forward="off", row=R120), table)


def test_the_host_need_counts_the_kept_forward():
    row = _row()
    plain = d.store_need(row, 59, 400_000, line_on=True, check_runs=False)
    kept = d.store_need(row, 59, 400_000, line_on=True, check_runs=False, keep_forward=True)
    assert plain["host_mib"] == 158_617 and plain["keep_forward_mib"] == 0
    assert kept["keep_forward_mib"] == 9 * int(row["boundary_mib"]) == 91_044      # eight checkpoints and the window end
    assert kept["host_mib"] == 158_617 + 91_044 == 249_661
    assert "forward kept" in kept["host_for"] and "9 more states (91,044 MiB)" in kept["how"]
    # with the answer check the call's peak is the check's: its windows run after the loaded program has ended
    check = d.store_need(row, 59, 400_000, line_on=True, check_runs=True, keep_forward=True)
    assert check["host_mib"] == 258_057 and check["host_gradient_mib"] == 249_661
    # a host whose store does not fit in memory: the program keeps no forward, and none is counted
    small = d.store_need(row, 59, 100_000, line_on=True, check_runs=False, keep_forward=True)
    assert small["ram_mib"] < small["store_mib"] and small["keep_forward_mib"] == 0
    assert small["host_mib"] == d.store_need(row, 59, 100_000, line_on=True, check_runs=False)["host_mib"]
    # 12 steps: four checkpoints and the window end
    assert d.store_need(row, 12, 400_000, True, False, keep_forward=True)["keep_forward_mib"] == 5 * int(row["boundary_mib"])
    with pytest.raises(d.HostMemoryError, match="needs about 249,661 MiB"):
        d.refuse_host_too_small(STORM, kept["host_mib"], 200_000, what=kept["host_for"])


def test_the_command_says_which_happened_for_each_gradient():
    plan = {"on": True}
    assert d.keep_forward_words({"keep_forward": {"used": True}}, {"on": False}) is None
    assert d.keep_forward_words({"keep_forward": {"used": True}}, None) is None
    took = d.keep_forward_words({"keep_forward": {"used": True, "forward_step_loop_s_not_run": 17.68},
                                 "loaded_program": {"first": False}}, plan)
    assert took.startswith("forward: KEPT, not run again") and "17.7 s of forward steps not run" in took
    first = d.keep_forward_words({"keep_forward": {"used": False, "keeping": True},
                                  "loaded_program": {"first": True}}, plan)
    assert first.startswith("forward: ran its own (this is its program's first gradient)") and "is kept for" in first
    moved = d.keep_forward_words({"keep_forward": {"used": False, "keeping": True},
                                  "loaded_program": {"first": False}}, plan)
    assert moved.startswith("forward: ran its own (the program held no forward of this start") and "another start" in moved
    files = d.keep_forward_words({"keep_forward": {"used": False, "keeping": False, "why_not": "a checkpoint is in files"},
                                  "loaded_program": {"first": True}}, plan)
    assert "it is not kept (a checkpoint is in files)" in files
    assert d.keep_forward_words({}, plan).startswith("forward: ran (the runner's receipt names no kept forward")


STAND_IN = textwrap.dedent('''
    """A stand-in for the runner's loaded-program route: it serves DIR/NAME.req as the runner does and keeps the
    "forward" of the request before, which it reuses only when the start file reads as it did then."""
    import hashlib, json, os, sys, time
    from pathlib import Path
    a = sys.argv[1:]
    def word(flag): return a[a.index(flag) + 1]
    rdir, name, out, start = Path(word("--resident")), word("--resident-name"), Path(word("--out")), Path(word("--start"))
    on = os.environ.get("AD_KEEP_FORWARD") == "2"
    kept, index = None, 0
    def serve(name, out):
        global kept, index
        index += 1
        key = hashlib.sha256(start.read_bytes()).hexdigest()
        used = on and kept == key
        doc = {"args": {"out": str(out)}, "resident": {"request": name, "index": index}}
        if on:
            doc["keep_forward"] = ({"used": True, "forward_step_loop_s_not_run": 1.5} if used
                                   else {"used": False, "keeping": True})
            kept = key
        out.mkdir(parents=True, exist_ok=True)
        (out / "window.json").write_text(json.dumps(doc))
        tmp = rdir / (name + ".done.part"); tmp.write_text(json.dumps({"rc": 0, "index": index, "wall_s": 0.01}))
        os.replace(tmp, rdir / (name + ".done"))
    serve(name, out)
    t0 = time.time()
    while time.time() - t0 < 60 and not (rdir / "stop").exists():
        for req in sorted(rdir.glob("*.req")):
            words = json.loads(req.read_text()); req.unlink()
            serve(req.name[:-4], Path(words["out"]))
        time.sleep(0.02)
''')


@pytest.mark.parametrize("word", ["2", None])
def test_a_stand_in_program_reuses_the_forward_for_the_same_start_and_not_for_a_changed_one(tmp_path, monkeypatch, word):
    """Three requests to one loaded program through the door's own LoadedProgram: the same start again takes the kept
    forward; a changed start runs its own; with the word off nothing is kept or said."""
    monkeypatch.setattr(d, "refuse_no_local_gpu", lambda: None)
    monkeypatch.setattr(d, "process_card_mib", lambda pid: {})
    monkeypatch.setattr(d, "library_path", lambda: "")
    script, start = tmp_path / "standin.py", tmp_path / "start.bin"
    script.write_text(STAND_IN)
    start.write_bytes(b"start A")
    plan = {"on": word is not None}
    env = {"AD_KEEP_FORWARD": word} if word else {}
    program = d.LoadedProgram(tmp_path / "program", idle_s=30, stop_wait_s=20)
    said = []
    try:
        o1 = tmp_path / "o1"
        doc = program.start([sys.executable, str(script), "--out", str(o1), "--start", str(start)], env, tmp_path, "0",
                            "o01", o1)
        said.append(d.keep_forward_words(doc, plan))
        req = lambda k: {"out": str(tmp_path / f"o{k}"), "objective": "x", "dump-grad": "1", "bstore": str(tmp_path / "b")}
        said.append(d.keep_forward_words(program.request("o02", req(2), tmp_path / "o2"), plan))      # the same start
        start.write_bytes(b"start B")
        said.append(d.keep_forward_words(program.request("o03", req(3), tmp_path / "o3"), plan))      # a changed start
        said.append(d.keep_forward_words(program.request("o04", req(4), tmp_path / "o4"), plan))      # B again
    finally:
        program.stop()
    if word is None:
        assert said == [None, None, None, None]
        return
    assert said[0].startswith("forward: ran its own (this is its program's first gradient)")
    assert said[1].startswith("forward: KEPT, not run again")
    assert said[2].startswith("forward: ran its own (the program held no forward of this start")
    assert said[3].startswith("forward: KEPT, not run again")


def test_the_exact_route_runs_the_default_routes_kernel_set_and_radiation_kernels():
    row, pins = _row(), assets.load_pins()
    exact, taken = d.route_row(row, _ns(boundary_layer="taped"))
    assert exact["assets"]["kernels"] == f"latest/kset/{KS}/kernels.json"
    # dist46: the cloud default's images are taken too (by the cloud word's route)
    assert {t["asset"] for t in taken if t["option"] == "--boundary-layer"} == {"kernels", "cubins"}
    assert all(t["route"] == "taped" for t in taken if t["option"] == "--boundary-layer")
    routes = row["gradient_line"]["boundary_layer"]["routes"]
    # dist48: the default route's bundle also carries its boundary-layer form's p3 images; the radiation pair is shared
    assert routes["taped"]["scheme_bundle"]["images"] == {r: i for r, i in routes["tape-free"]["scheme_bundle"]["images"].items()
                                                         if r.startswith("p7/")}
    # 0.6.0: the default route's set is the same set with three double-arithmetic dynamics derivative images (-ddr);
    # the exact route keeps the single-arithmetic images its exact reference bytes were read on
    assert {k: dict(v, path=v["path"].replace(f"{KS}-ddr/", f"{KS}/")) for k, v in routes["tape-free"]["assets"].items()} == routes["taped"]["assets"]
    assert routes["tape-free"]["assets"]["kernels"]["path"] == f"latest/kset/{KS}-ddr/kernels.json"
    assert "images" not in routes["taped"] and not routes["taped"].get("gradient_env")   # no tape-free image, no carried words
    want = assets.wanted(pins, d.example_reads(row, _ns(boundary_layer="taped")))
    assert f"ad-kset-{KS}-v1.tar.gz" in want and "ad-aux-rad-step-sm120-v1.tar.gz" in want
    assert "37096b881573276a" in routes["taped"]["what"] and "272.2 s" in routes["taped"]["what"]
    # the other doors of the row (tangent, adjoint-test, --checked-forward) keep the row's own six-kernel set
    assert row["assets"]["kernels"].endswith("dyn6/kernels.json")
    assert "exact_route_keeps" not in row["gradient_line"]["kernel_set"]["tape_free_route"]


def test_a_long_exact_window_is_warned_about_and_not_refused():
    routes = _row()["gradient_line"]["boundary_layer"]["routes"]
    dt = _row()["dt_s"]
    assert d.route_window_warning(routes["taped"], 59, dt) is None
    assert d.route_window_warning(routes["taped"], 120, dt) is None                    # 30 minutes: inside
    said = d.route_window_warning(routes["taped"], 121, dt)
    assert "about 30 minutes at 3 km (measured on this example)" in said and "still runs" in said
    assert "(asked: 30.25 min)" in said
    assert d.route_window_warning(routes["tape-free"], 239, dt) is None                # no limit on the default route
    assert "about 30 minutes at 3 km, measured on this example" in routes["taped"]["window_limit"]


def test_the_sessions_quick_comparison_holds_the_cloud_word_and_the_keep_forward_word():
    assert "cloud_derivative" in ad_session.COMMAND_WORDS and "keep_forward" in ad_session.COMMAND_WORDS
    base = dict(mesh=None, suite=None, oro=None, dt=None, plan=None, init=None, case=None, runner=None, assets=None,
                gradient_mesh=None, gradient_plan=None, checked_forward=False, boundary_layer=None, first_gradient=None,
                rank_plans=None, keep_forward=None)
    words = lambda cd: ad_session.command_words(SimpleNamespace(
        args=argparse.Namespace(cloud_derivative=cd, **base), row_name=STORM, steps=59, cards=["0"]))
    assert words(None) != words("gated") and words("gated") == words("gated")
    assert words("gated")["words"]["cloud_derivative"] == "gated"


def test_the_printed_figures_are_the_measured_ones():
    table = d.load_table()
    row = table["rows"][STORM]
    text = json.dumps(table)
    assert "16 times" not in text
    tf = row["gradient_line"]["boundary_layer"]["routes"]["tape-free"]
    assert "50 s" in tf["saves"] and "80 s" in tf["saves"]   # dist47: the newest card run's figures
    assert "172.6 s" in tf["keep_forward"]["says"] and "180.5 s" in tf["keep_forward"]["says"]   # measured since
    for figure in ("239.5 s", "195.7 s", "205.3 s", "153,044 MiB", "158,617 MiB", "247.3 s", "184.4 s", "176.3 s"):
        assert figure in row["gradient_line"]["measured"], figure
    (near,) = row["accuracy_notes"]
    assert "1.37 times the real model's change" in near
    assert "on the exact route" in near and "--boundary-layer guarded" in near   # dist48: the default is closure-live
    assert "The fix is not in this build" not in near and "ran away in one calm column" in near
    (far,) = table["rows"][R120]["accuracy_notes"]
    assert "0.09 of the real model's change" in far and "held or live alike" in far and "not in this build" in far
    cd = row["gradient_line"]["cloud_derivative"]
    assert cd["default"] == "number"                                    # 0.6.0: number-following default
    assert "2.06e-06 against 1.67e-06" in cd["routes"]["gated"]["default_when_before"]   # dist46: gated's reasons
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "16 times" not in readme and "287 s on the default route" not in readme and "238 GiB" not in readme
    for figure in ("239.5 s", "195.7 s", "205.3 s", "249,661 MiB", "184.4 s", "1.37 times", "0.09 of the real"):
        assert figure in readme, figure


def test_the_parser_takes_the_keep_forward_word():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="door"))
    assert parser.parse_args(["sensitivity", "--out", "x"]).keep_forward is None
    assert parser.parse_args(["sensitivity", "--out", "x", "--keep-forward", "off"]).keep_forward == "off"
    with pytest.raises(SystemExit):
        parser.parse_args(["sensitivity", "--out", "x", "--keep-forward", "maybe"])


def test_a_command_that_stopped_the_loaded_program_waits_for_the_card_it_frees(monkeypatch):
    """A command with another word than the loaded program's (the cloud word is now in the quick comparison) stops
    that program before it reads the card: it must wait for the memory the program is giving back, not be refused."""
    waits = []
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "admit_card", lambda card, need, wait_s, archs=None: waits.append(wait_s) or {})
    monkeypatch.setattr(d, "refuse_card_too_small", lambda card, need: waits.append("size only"))
    ctx = SimpleNamespace(row={}, cards=["0"], split=None, need_mib=lambda: 96000,
                          args=SimpleNamespace(wait_card=0.0), session=SimpleNamespace(stopped=[{"pid": 1}]))
    d.Context.check_cards(ctx)
    assert waits == [d.SESSION_RELEASE_WAIT_S]
    ctx.session = SimpleNamespace(stopped=[])            # nothing was stopped: refused at once as before
    d.Context.check_cards(ctx)
    assert waits[-1] == 0.0
    ctx.args = SimpleNamespace(wait_card=30.0)           # the user's own wait: the size rule only, as before
    d.Context.check_cards(ctx)
    assert waits[-1] == "size only"


ZOMBIE_LEADER = textwrap.dedent("""
    import ctypes, os, sys, threading, time
    def later():
        time.sleep(float(sys.argv[1]))
        os._exit(0)
    threading.Thread(target=later).start()
    print("up", flush=True)
    ctypes.CDLL(None).syscall(60, 0)      # the first thread ends alone (exit, not exit_group): the others run on
""")


@pytest.mark.skipif(not sys.platform.startswith("linux") or os.uname().machine != "x86_64", reason="/proc and syscall 60")
def test_a_program_whose_first_thread_has_ended_is_alive_until_its_last_thread_goes(tmp_path):
    """The stopped runner's first thread reads Z in /proc while its other threads still hold the card and the host
    memory: such a process is not gone, and a command waits for it."""
    import subprocess
    import time
    script = tmp_path / "leader.py"
    script.write_text(ZOMBIE_LEADER)
    proc = subprocess.Popen([sys.executable, str(script), "1.5"], stdout=subprocess.PIPE)
    try:
        assert proc.stdout.readline().strip() == b"up"
        t0 = time.monotonic()
        while ad_session._start_ticks(proc.pid) is not None and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        assert ad_session._start_ticks(proc.pid) is None           # the first thread has ended ...
        assert ad_session._threads_left(proc.pid) and ad_session._alive(proc.pid, None)   # ... the process has not
        handle = ad_session._Pid(proc.pid, None, tmp_path)
        assert handle.poll() is None
        handle.wait(timeout=10)
        assert time.monotonic() - t0 >= 1.0 and not ad_session._threads_left(proc.pid)
        assert not ad_session._alive(proc.pid, None)
    finally:
        proc.wait(timeout=10)
    assert not ad_session._alive(proc.pid, None) and not ad_session._threads_left(proc.pid)


def test_a_command_that_stopped_a_program_reads_the_host_memory_once_it_has_settled(monkeypatch):
    readings = iter([106_048, 114_986, 180_000, 249_500, 250_000, 250_100])
    monkeypatch.setattr(d, "host_available_mib", lambda: next(readings))
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    assert d.host_available_settled() == 250_000              # rising by more than 1 GiB a second: still coming back
    monkeypatch.setattr(d, "host_available_mib", lambda: 300_000)
    assert d.host_available_settled() == 300_000              # nothing coming back: one second, the same reading
    monkeypatch.setattr(d, "host_available_mib", lambda: None)
    assert d.host_available_settled() is None
    import inspect
    assert 'host_available_settled() if getattr(self.session, "stopped", None)' in inspect.getsource(d.Context.__init__)
