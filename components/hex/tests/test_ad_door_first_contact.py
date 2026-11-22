# SPDX-License-Identifier: Apache-2.0
"""What a new user hits first on the differentiable engine's doors, before any card work or download.

THE BREAKAGES (the dist47 checker's fresh install on a 32 GB card, 2026-11-15):
* the 3 km storm command printed "session card0: STARTED FRESH: ... this command leaves its program loaded for the
  next one" and was then refused for the card's size: no program was ever loaded;
* ``adjoint-test --steps 12`` on the 3-step 120 km example downloaded the example (2.4 GB) and then stopped in the
  runner with "physics step count or single-stream plan is invalid".
"""
import argparse
import types

import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_session as s
from woof.hex.ad import assets


@pytest.fixture(autouse=True)
def _stand_in_runner_on_hidden_cards(monkeypatch):
    # The battery hides the cards (conftest CUDA_VISIBLE_DEVICES=-1) and these tests start a stand-in runner that
    # opens no device, so the door's hidden-card refusal (refuse_hidden_card) is let through here only.
    from woof.hex import ad_doors as _d
    monkeypatch.setattr(_d, "refuse_hidden_card", lambda: None)

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def user(monkeypatch, tmp_path):
    """A card of the right family (32 GB unless a test says otherwise), a host and a disk that fit, a runner, a
    sessions folder of the test's own, and an asset root that must not be resolved."""
    for name in d.NO_LOCAL_GPU_ENV + s.SCHEDULER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("WOOF_HEX_AD_ASSETS", s.MODE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(s.SESSIONS_ENV, str(tmp_path / "sessions"))
    fetched = []

    def asset_root(cache=None, want=None):
        fetched.append("asset_root")
        raise AssertionError("the asset root was resolved (a first use downloads the example)")

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: fetched.append("_fetch"))
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 32607)
    monkeypatch.setattr(d, "free_mib", lambda card: 32000)
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 900_000)
    monkeypatch.setattr(d, "host_available_settled", lambda *a, **k: 900_000)
    monkeypatch.setattr(d, "_free_disk_mib", lambda folder: 9_000_000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)
    return fetched


STORM_CMD = ["sensitivity", "--row", STORM, "--steps", "59", "--circle", "40.6,-91.3,150", "--levels", "1-10"]


def test_a_refused_command_never_says_it_started_or_left_a_program(user, tmp_path, monkeypatch, capsys):
    # the README's 3 km command on a 32 GB card: refused for the card, and no session line before the refusal
    with pytest.raises(d.AdDoorError, match="more than the whole card"):
        d.Context(_parse(STORM_CMD + ["--out", str(tmp_path / "o")]), "sensitivity")
    out = capsys.readouterr().out
    assert "STARTED FRESH" not in out and "leaves its program loaded" not in out and "ATTACHED" not in out
    # a card that fits and a host that does not: refused for the host, and still no session line
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 50_000)
    monkeypatch.setattr(d, "host_available_settled", lambda *a, **k: 50_000)
    with pytest.raises(d.AdDoorError, match="MiB of memory available"):
        d.Context(_parse(STORM_CMD + ["--out", str(tmp_path / "o")]), "sensitivity")
    out = capsys.readouterr().out
    assert "STARTED FRESH" not in out and "leaves its program loaded" not in out
    # card and host fit, the run folder's disk does not: refused for the disk, and no session line
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 900_000)
    monkeypatch.setattr(d, "host_available_settled", lambda *a, **k: 900_000)
    monkeypatch.setattr(d, "_free_disk_mib", lambda folder: 1000)
    with pytest.raises(d.AdDoorError, match="disk with 1,000 MiB free"):
        d.Context(_parse(STORM_CMD + ["--out", str(tmp_path / "o")]), "sensitivity")
    out = capsys.readouterr().out
    assert "STARTED FRESH" not in out and "leaves its program loaded" not in out
    assert user == []


def test_the_session_line_is_said_once_the_checks_pass_and_before_any_download(user, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(STORM_CMD + ["--out", str(tmp_path / "o")]), "sensitivity")
    lines = capsys.readouterr().out.splitlines()
    assert ": STARTED FRESH: no program was loaded for this card" in lines[0] and "session card" in lines[0]
    assert sum("STARTED FRESH" in line for line in lines) == 1 and user == ["asset_root"]


def test_a_program_the_command_stopped_is_said_when_the_command_is_then_refused(tmp_path, capsys):
    ctx = types.SimpleNamespace(door="sensitivity", args=types.SimpleNamespace(session_idle_s=None), cards=["0"])
    use = s.Use(ctx, "use", None)
    use.defer, use.name = True, "card0"
    use.say("woof hex sensitivity: session card0: STARTED FRESH: the program found loaded (pid 41) was started "
            "with other words (steps) and was stopped; this command leaves its program loaded for the next one")
    use.stopped.append({"name": "card0", "pid": 41, "why": "started with other words: steps"})
    use._stopped_here.append(use.stopped[-1])
    use.refused()
    out = capsys.readouterr().out
    assert "STARTED FRESH" not in out and "leaves its program loaded" not in out
    assert "(pid 41) was stopped (started with other words: steps)" in out and "no program is loaded" in out
    use.said()   # nothing held is said after a refusal
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("row, steps, most", [(None, 12, 3), ("hrrrv5-full-x1.40962", 4, 3), (STORM, 240, 239)])
def test_a_step_count_the_example_does_not_cover_is_refused_before_anything(user, tmp_path, row, steps, most):
    argv = ["adjoint-test", "--steps", str(steps), "--out", str(tmp_path / "o")] + (["--row", row] if row else [])
    with pytest.raises(d.AdDoorError, match=rf"--steps {steps}: row \S+'s example plan covers windows of 1 to {most} "
                                            rf"steps .*pass --steps from 1 to {most}.*Nothing was downloaded"):
        d.Context(_parse(argv), "adjoint-test")
    assert user == [] and not (tmp_path / "o").exists()


def test_a_window_of_no_steps_is_refused_and_not_run_as_the_default(user, tmp_path):
    with pytest.raises(d.AdDoorError, match="--steps 0: a window has at least 1 step. Nothing was downloaded"):
        d.Context(_parse(["adjoint-test", "--steps", "0", "--out", str(tmp_path / "o")]), "adjoint-test")
    assert user == []


def test_a_count_the_example_covers_goes_on_and_a_plan_of_ones_own_is_the_runners_to_judge(user, tmp_path,
                                                                                          monkeypatch):
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(["adjoint-test", "--steps", "3", "--out", str(tmp_path / "a")]), "adjoint-test")
    plan = tmp_path / "plan"
    plan.mkdir()
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(["adjoint-test", "--steps", "12", "--plan", str(plan), "--out", str(tmp_path / "b")]),
                  "adjoint-test")
    assert user == ["asset_root", "asset_root"]


def test_every_example_plan_says_the_steps_it_covers():
    rows = d.load_table()["rows"]
    for name, row in rows.items():
        if row.get("example_plan"):
            assert int(row["example_plan_steps"]) >= int(row["default_steps"]) and row["example_plan_steps_what"], name


QUIET_RUNNER = r'''#!PYTHON
import json, sys, time
from pathlib import Path
argv = sys.argv[1:]
out = Path(argv[argv.index("--out") + 1])
print(json.dumps({"event": "window_native", "function": "mynn_reverse", "pass": "ad", "step": 12}), file=sys.stderr,
      flush=True)
time.sleep(5.0)   # quiet: the door's checks every 2 s see the log stand still
for _ in range(10):   # then a line every 0.4 s: the next check sees it move again, and none sees it stand still
    print(json.dumps({"event": "window_mark", "at": "done"}), file=sys.stderr, flush=True)
    time.sleep(0.4)
(out / "window.json").write_text(json.dumps({"event": "window_receipt"}))
'''


def test_a_runner_that_goes_quiet_is_said_with_where_it_stopped_and_the_time(tmp_path, monkeypatch, capsys):
    """THE BREAKAGE: a first adjoint-test on the 12 km cut went silent in its third window for ten minutes (the
    dist47 checker's run) and the door printed nothing. Now the door says, with the time, where the runner's log
    stopped, how busy the card is and what the runner holds; and that it writes again when it does."""
    import sys as _sys
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    runner = tmp_path / "runner"
    runner.write_text(QUIET_RUNNER.replace("PYTHON", _sys.executable))
    runner.chmod(0o755)
    monkeypatch.setenv("WOOF_HEX_AD_QUIET_S", "1")
    monkeypatch.setattr(d, "process_card_mib", lambda pid: {"00000000:01:00.0": 23550})
    monkeypatch.setattr(d, "card_busy_percent", lambda card: 100)
    monkeypatch.setattr(d, "window_validity", lambda out, doc, argv: {})
    plan = tmp_path / "plan"
    plan.mkdir()
    doc = d.run_window([str(runner), "window", "--out", str(tmp_path / "w")], {}, plan, tmp_path / "w", "0")
    err = capsys.readouterr().err
    assert "window w: the runner (pid" in err and "has written nothing for" in err
    assert "its log stops at pass ad, step 12 (window_native: mynn_reverse)" in err
    assert "card 0: 100% busy, 23,550 MiB held by the runner" in err and "the runner writes again" in err
    [note] = doc["door_quiet"]
    assert note["last"].startswith("pass ad, step 12") and note["quiet_s"] >= 1.0 and note["card_busy_percent"] == 100
