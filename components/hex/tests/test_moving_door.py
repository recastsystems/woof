"""The installed moving-region door: ``woof hex moving``.

THE BREAKAGE THESE PREVENT (release 0.4.0 readiness B6): the moving nest ran
only through the source tree's tools/moving48 shell controller, which was
bound to one rented host's card queue and work folders, so no install could
run it and the public export refused those files.  The door now lives in the
package; these checks hold its surface, its child commands and its process
handling, with no card and no assets.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex.cli import build_parser, main

TREE = Path(__file__).resolve().parents[1]
CASES = TREE / "verification" / "moving48" / "cases"


def _box_tokens() -> tuple[str, ...]:
    """A rented host's work root, and its card queue's names, which are the project's own: those are read from the
    private pattern file (``host_tokens``; HEX_PRIVATE_PATTERNS names it, and it is never packaged). THE BREAKAGE:
    this file spelled them in pieces, so every source package carried them."""
    tokens = ["/" + "work/"]
    path = os.environ.get("HEX_PRIVATE_PATTERNS")
    if path:
        tokens += json.loads(Path(path).read_text(encoding="utf-8"))["host_tokens"]
    return tuple(tokens)


BOX_TOKENS = _box_tokens()


def _case(tmp_path: Path, **changes) -> Path:
    case = json.loads((CASES / "tc-wp-2025110300.json").read_text())
    case.update(changes)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    return path


def test_every_moving_command_is_reachable_from_the_console_door():
    parser = build_parser()
    moving = parser._subparsers._group_actions[0].choices["moving"]
    commands = set(moving._subparsers._group_actions[0].choices)
    assert {"check", "stage", "prepare", "prepare-member", "run", "fixed", "summarize",
            "operator", "zone"} <= commands
    arguments = parser.parse_args(["moving", "run", "--case", "c.json", "--casedir", "d", "--out", "r",
                                   "--hours", "6"])
    assert arguments.engine is None and arguments.device is None and arguments.members is None
    assert arguments.hold_for_member_seconds == 1800


def test_the_console_door_still_imports_without_numpy():
    # The render door must open on a node whose Python has no numpy; adding the
    # moving parser must not pull it in at parser build time.
    code = ("import sys, builtins\n"
            "real = builtins.__import__\n"
            "def guard(name, *a, **k):\n"
            "    if name == 'numpy' or name.startswith('numpy.'):\n"
            "        raise ImportError('numpy blocked')\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = guard\n"
            "from woof.hex.moving_door import add_moving_parser\n"
            "import argparse\n"
            "p = argparse.ArgumentParser(); add_moving_parser(p.add_subparsers())\n"
            "print('ok')\n")
    env = dict(os.environ, PYTHONPATH=str(TREE / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_member_specs_come_from_the_case_alone(tmp_path):
    from woof.hex.moving_case import frame_spec, load_case, member_name, spec, stitch

    case = load_case(_case(tmp_path))
    lat, lon = case["storm"]["initial_center_deg"]
    body = spec(case, "start", lat, lon)
    # corridor: fixed rings about each corridor centre, then the moving 6 km ring and the fine cap
    fine = body["regions"][-1]
    assert fine["shape"]["center_deg"] == [lat, lon]
    assert fine["spacing_km"] == case["mesh"]["fine_km"]
    assert fine["shape"]["radius_km"] == case["mesh"]["fine_radius_km"]
    frame = frame_spec(case)
    assert frame["regions"] and body["regions"][:len(frame["regions"])] == frame["regions"]
    moving = body["regions"][len(frame["regions"]):]
    assert all(r["shape"]["center_deg"] == [lat, lon] for r in moving)
    assert [r["spacing_km"] for r in moving] == [6.0, case["mesh"]["fine_km"]]
    radii = stitch(case)
    assert radii["outer_radius_km"] > radii["inner_radius_km"] > case["mesh"]["fine_radius_km"]
    assert member_name(10.5, 128.8) == "mn01050e12880"
    assert member_name(-5.0, -60.0) == "ms00500e30000"


def test_case_refusals_name_what_is_missing(tmp_path):
    from woof.hex.moving_case import MovingCaseError, load_case

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"start_time": "2025-11-03_00:00:00"}))
    with pytest.raises(MovingCaseError, match="lacks dt_seconds, n_levels, vertical_spec, storm, mesh"):
        load_case(path)
    assert main(["moving", "fixed", "--case", str(path), "--casedir", str(tmp_path), "--out",
                 str(tmp_path / "r"), "--hours", "1"]) == 2


def test_vertical_spec_resolves_in_the_tree_and_from_the_package(tmp_path):
    from woof.hex.moving_case import load_case, resolve_vertical_spec

    in_tree = resolve_vertical_spec(load_case(CASES / "tc-wp-2025110300.json"))
    assert in_tree == TREE / "verification" / "vertical-specs" / "tc55-v1.json"
    # A case copied away from the tree finds the packaged spec of the same name.
    packaged = resolve_vertical_spec(load_case(_case(tmp_path)))
    assert packaged.parent.name == "countries" and packaged.name == "tc55-v1.json"
    assert json.loads(packaged.read_text()) == json.loads(in_tree.read_text())


def test_segments_are_this_interpreter_running_the_forecast_door(tmp_path):
    from woof.hex.moving_case import load_case
    from woof.hex.moving_run import segment_command

    case = load_case(_case(tmp_path))
    member = dict(name="start", mesh="start", grid="g.nc", static="s.nc", init="i.nc", dt_seconds=18.75)
    cmd = segment_command(case, member, tmp_path / "seg", 48.0, 60.0, None, init_source="gfs analysis")
    assert cmd[:4] == [sys.executable, "-m", "woof.hex.cli", "forecast"]
    assert "--gpuwm-checkout" not in cmd  # the installed engine by default
    assert cmd[cmd.index("--dt") + 1] == "18.75"
    assert cmd[-2:] == case["physics_args"]
    with_engine = segment_command(case, member, tmp_path / "seg", 1.0, 60.0, "/engine", init_source="x")
    assert with_engine[with_engine.index("--gpuwm-checkout") + 1] == "/engine"


def test_segment_environment_sets_run_state_and_keeps_knobs(tmp_path, monkeypatch):
    from woof.hex.moving_run import Controller

    monkeypatch.setenv("WOOF_HEX_MOVING_OPERATOR", "stale-operator.npz")
    monkeypatch.setenv("WOOF_HEX_MOVING_CHECKPOINT_IN", "stale.pkl")
    monkeypatch.setenv("WOOF_HEX_MOVING_BALANCE", "0")
    fake = SimpleNamespace(a=SimpleNamespace(seal_every_hours=12.0, device="1"))
    member = dict(mesh_rows=str(tmp_path / "rows.json"))
    env = Controller.segment_env(fake, member, tmp_path / "control", tmp_path / "b.pkl", None, None, None)
    assert "WOOF_HEX_MOVING_OPERATOR" not in env and "WOOF_HEX_MOVING_CHECKPOINT_IN" not in env
    assert env["WOOF_HEX_MOVING_BALANCE"] == "0"  # a comparison knob passes through
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["WOOF_HEX_MOVING_CHECKPOINT_OUT"] == str(tmp_path / "b.pkl")
    zone = Controller.segment_env(fake, member, tmp_path / "c", tmp_path / "b.pkl", tmp_path / "in.pkl",
                                  tmp_path / "a--b.zone.npz", None)
    assert zone["WOOF_HEX_MOVING_ZONE"].endswith("a--b.zone.npz")
    assert zone["WOOF_HEX_MOVING_CHECKPOINT_IN"] == str(tmp_path / "in.pkl")


@pytest.mark.parametrize('whole_domain', [False, True])
def test_corridor_transfer_selection_reaches_carrier_and_survives_resume(tmp_path, monkeypatch, whole_domain):
    """Corridor geometry used to force zone carry, bypassing the all-domain balance solver."""
    from woof.hex.moving_run import Controller
    from woof.hex import moving_members

    case = _case(tmp_path)
    monkeypatch.setattr(Controller, 'adopt_start_member', lambda self: 'start')
    monkeypatch.setattr(Controller, 'read_member', lambda *args: dict(mesh_rows='rows.json', status='ready'))
    monkeypatch.setattr(moving_members, 'read_stage', lambda path: dict(source='gfs'))
    argv = ['moving', 'run', '--case', str(case), '--casedir', str(tmp_path / 'case'),
            '--out', str(tmp_path / 'run'), '--hours', '6']
    if whole_domain:
        argv.append('--whole-domain-remap')
    controller = Controller(build_parser().parse_args(argv))
    assert controller.stitch and controller.a.zone_carry  # geometry is unchanged
    path = controller.operator_path('start', 'next')
    env = controller.segment_env(dict(mesh_rows='rows.json'), tmp_path / 'control', tmp_path / 'out.pkl',
                                 tmp_path / 'in.pkl', path, 'source.nc')
    assert ('WOOF_HEX_MOVING_OPERATOR' in env) == whole_domain
    assert ('WOOF_HEX_MOVING_ZONE' in env) != whole_domain
    resumed = Controller(build_parser().parse_args([*argv, '--resume']))
    assert resumed.state['transfer'] == controller.state['transfer']
    changed = [value for value in argv if value != '--whole-domain-remap']
    if not whole_domain:
        changed.append('--whole-domain-remap')
    with pytest.raises(SystemExit, match='retain the original transfer route'):
        Controller(build_parser().parse_args([*changed, '--resume']))


@pytest.mark.skipif(os.name != "posix" or shutil.which("sh") is None, reason="process groups are POSIX")
def test_stopping_a_child_stops_its_whole_tree(tmp_path):
    from woof.hex.moving_run import start_child, stop_process_tree

    marker = tmp_path / "grandchild.pid"
    with (tmp_path / "log").open("w") as log:
        proc = start_child(["sh", "-c", f"sleep 300 & echo $! > {marker}; wait"], stdout=log)
    for _ in range(100):
        if marker.is_file() and marker.read_text().strip():
            break
        time.sleep(0.05)
    grandchild = int(marker.read_text())
    stop_process_tree(proc, grace_seconds=10)
    assert proc.poll() is not None
    for _ in range(100):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(grandchild, signal.SIGKILL)
        pytest.fail("the grandchild outlived its stopped parent")


def test_no_moving_file_is_bound_to_a_host_queue_or_work_root():
    files = sorted((TREE / ".." / ".." / "woof" / "hex").glob("moving_*.py")) + [TREE / ".." / ".." / "woof" / "hex" / "storm_tracker.py"]
    files += sorted((TREE / "tools" / "moving48").glob("*"))
    files += [TREE / "tools" / "run_moving_patch.py"]
    files += sorted((TREE / "tools" / "moving_jolt").glob("*"))
    hits = [f"{path.relative_to(TREE)}: {token}" for path in files if path.is_file()
            for token in BOX_TOKENS if token in path.read_text(errors="replace")]
    assert hits == []
    # The host runtime file the shell controller sourced is gone, and nothing sources it.
    assert not (TREE / "tools" / "moving48" / "runtime.sh").exists()
    assert not [p.name for p in (TREE / "tools" / "moving48").glob("*.sh") if "runtime.sh" in p.read_text()]


def test_check_refuses_lts_on_the_cpu_and_a_missing_reference_arm(tmp_path):
    from woof.hex.moving_jolt_check import run_check

    with pytest.raises(ValueError, match="CUDA only"):
        run_check(tmp_path, device="cpu", lts=True)
    with pytest.raises(ValueError, match="fixed arm is the reference"):
        run_check(tmp_path, device="cpu", arms=("baseline", "balanced"))


def test_check_verdict_grades_the_balanced_arm_only():
    from woof.hex.moving_jolt_check import MOVE_STEPS, STEPS, evaluate

    def arm(scale, error=0.0):
        rows = [dict(w=1.0, dpdt=1.0) for _ in range(STEPS)]
        for step in MOVE_STEPS:
            rows[step] = dict(w=scale, dpdt=scale)
        budget = dict(before=1.0, after=1.0 + error, absolute_error=error)
        return dict(rows=rows, budgets=[dict(mass=budget, theta_mass=budget, tracers={"passive": budget})])

    good = evaluate(dict(fixed=arm(1.0), baseline=arm(16.0), balanced=arm(1.01, 3e-10)), limit=1.5, budget=1e-9)
    assert good["verdict"]["passed"] and good["verdict"]["baseline_worst_jolt"] == 16.0
    jolted = evaluate(dict(fixed=arm(1.0), balanced=arm(1.6)), limit=1.5, budget=1e-9)
    assert not jolted["verdict"]["passed"]
    leaky = evaluate(dict(fixed=arm(1.0), balanced=arm(1.0, 2e-9)), limit=1.5, budget=1e-9)
    assert not leaky["verdict"]["passed"]


def test_short_segment_preserves_whole_step_duration(tmp_path):
    from fractions import Fraction
    from woof.hex.moving_run import segment_command
    from woof.hex.mpasphys.forecast_v842 import integral_duration
    case = {'start_time':'2025-01-01_00:00:00', 'physics_args':[]}
    member = dict(mesh='start', grid='g', static='s', init='i', dt_seconds=3.75)
    cmd = segment_command(case, member, tmp_path, 5/60, 2.5, init_source='gfs')
    duration = Fraction(cmd[cmd.index('--hours')+1])*3600
    assert integral_duration(duration, Fraction('3.75')) == 300
