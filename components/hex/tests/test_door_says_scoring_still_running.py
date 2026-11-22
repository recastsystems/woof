"""A composed door's last line does not call scoring done while its scorer is still running.

THE BREAKAGE (0.6.0, a fresh install from the published files): `woof hex global --row global-120km ...` printed
`done text=global-120km 2026-10-11T12: mesh, static, init, forecast, products, export, score` and exited 0. The
scorer it had handed the run to (a background process on the CPU) worked for several more minutes, and
`scorecard.json` read QUEUED when the prompt came back.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import observation_scorecard as scorecard  # noqa: E402

STAGES = ["mesh", "static", "init", "forecast", "products", "export", "score"]


def test_a_running_scorer_is_named_as_running_with_its_pid_and_file():
    text = scorecard.stages_done_text(STAGES, {"joined": "automatic", "status": "QUEUED", "pid": 60214,
                                               "path": "/run/forecast/observation-scorecard/scorecard.json"})
    done, tail = text.split("; ", 1)
    assert done == "mesh, static, init, forecast, products, export done"
    assert "score" not in done
    assert "STILL RUNNING" in tail and "pid 60214" in tail
    assert "/run/forecast/observation-scorecard/scorecard.json" in tail and "no longer QUEUED" in tail


def test_scoring_that_waits_for_observations_is_not_called_done():
    for status in ("NOT_YET_OBSERVABLE", "DEFERRED"):
        text = scorecard.stages_done_text(["forecast", "score"], {"status": status, "path": "card.json"})
        assert text.startswith("forecast done; score STILL RUNNING") and status in text


def test_a_finished_scorecard_gives_its_status_and_count():
    text = scorecard.stages_done_text(STAGES, {"joined": "completed", "status": "PARTIALLY_MEASURED",
                                               "measured_metrics": 41, "path": "card.json"})
    assert text.endswith("export done; score (PARTIALLY_MEASURED, 41 measured metrics)")
    assert "RUNNING" not in text


def test_skipped_scoring_and_doors_without_a_score_stage():
    assert scorecard.stages_done_text(STAGES, {"joined": "skipped", "status": "SKIPPED"}).endswith("; score skipped")
    assert scorecard.stages_done_text(["mesh", "static"], None) == "mesh, static"
    assert scorecard.stages_done_text(["score"], {"status": "QUEUED", "path": "c"}).startswith("score STILL RUNNING")


def test_the_join_reports_a_live_scorer_as_queued_and_a_finished_one_with_its_count(tmp_path):
    run = tmp_path / "forecast"
    root = run / "observation-scorecard"
    root.mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    import hashlib
    (root / scorecard.SOURCE_CONFIG).write_text(
        json.dumps({"sha256": hashlib.sha256(config.read_bytes()).hexdigest()}), encoding="utf-8")
    (root / "automatic-retry-receipt.json").write_text(json.dumps({"status": "SCORED", "pid": os.getpid()}),
                                                       encoding="utf-8")
    (root / "scorecard.json").write_text(json.dumps({"status": "MEASURED", "measured_metrics": 7}), encoding="utf-8")
    joined = scorecard.join_automatic_scorer(run, config)
    assert joined["status"] == "MEASURED" and joined["measured_metrics"] == 7
    assert scorecard.stages_done_text(["forecast", "score"], joined) == "forecast done; score (MEASURED, 7 measured metrics)"


def test_the_terminal_gets_the_line_too(capsys):
    """The `done` line goes to the door's log file. The terminal's own last line must say it as well: a 0.6.0
    user saw "EXPORT done", then the prompt."""
    line = scorecard.say_scoring("global", STAGES, {"status": "QUEUED", "pid": 60214, "path": "/run/card.json"})
    out = capsys.readouterr().out.strip().splitlines()
    assert out == [line] and line.startswith("woof hex global: score STILL RUNNING in the background (pid 60214")
    assert "/run/card.json" in line and "no longer QUEUED" in line
    done = scorecard.say_scoring("country", ["forecast", "score"], {"joined": "completed", "status": "MEASURED",
                                                                    "measured_metrics": 12})
    assert done == "woof hex country: score (MEASURED, 12 measured metrics)"
    assert scorecard.say_scoring("global", ["mesh", "static"], None) is None
    assert capsys.readouterr().out.strip() == done
    for name in ("global_door.py", "country_door.py"):
        assert "say_scoring(" in (ROOT / "../../woof/hex" / name).read_text(encoding="utf-8"), name


def test_both_composed_doors_end_with_that_line():
    for name in ("global_door.py", "country_door.py"):
        source = (ROOT / "../../woof/hex" / name).read_text(encoding="utf-8")
        done = [line for line in source.splitlines() if "status.done(" in line and "members" not in line]
        assert done and all("stages_done_text(" in line for line in done), (name, done)
