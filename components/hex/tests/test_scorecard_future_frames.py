"""The automatic observation scorer never waits for frames valid in the future.

Breakage: a 3 km hex region from GFS f348 (valid 13 days ahead) started the scorecard watcher, which slept in the
leased forecast command's process tree for 75 min until it was killed by PID; the run's products step waited
behind it (2026-10-24). Now a run whose frames are all in the future starts no
watcher and says when to score, and a watcher never sleeps for a future frame.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from woof.hex import observation_scorecard as osc


def _frames(tmp_path, start, hours):
    return [{"history": str(tmp_path / f"h{h}.nc"),
             "valid_time": osc.stamp(start + timedelta(hours=h))} for h in hours]


def _config(start):
    return {"start": osc.stamp(start), "init": "i.nc", "references": [], "thresholds_mm": [1.0],
            "widths_km": [30.0], "observation_delay_hours": 7}


def test_future_run_starts_no_watcher_and_says_when(monkeypatch, tmp_path):
    start = datetime.now(timezone.utc) + timedelta(days=13)
    frames = _frames(tmp_path, start, [0, 1])
    monkeypatch.setattr(osc, "discover", lambda run, config=None: ({**_config(start), "run": str(run)}, frames))
    monkeypatch.setattr(osc, "queue_retry", lambda *a, **k: pytest.fail("a watcher was started for a future run"))
    monkeypatch.delenv(osc.SCORING_ENV, raising=False)
    monkeypatch.delenv("WOOF_HEX_OBSERVATION_CONFIG", raising=False)
    result = osc.forecast_hook(tmp_path, init=tmp_path / "i.nc", grid=tmp_path / "g.nc", start=osc.stamp(start))
    assert result["status"] == "NOT_YET_OBSERVABLE"
    root = tmp_path / "observation-scorecard"
    card = json.loads((root / "scorecard.json").read_text(encoding="utf-8"))
    assert card["status"] == "NOT_YET_OBSERVABLE" and "woof hex score" in card["reason"]
    assert all("in the future" in row["reason"] for row in card["metrics"])
    assert not (root / "pending-init.nc").exists() and not (tmp_path / ".keep").exists()
    # A door's score stage reports it at once instead of scoring (and fetching) a future run itself.
    config = tmp_path / "observation-scorecard.config.json"
    config.write_text("{}", encoding="utf-8")
    (root / osc.SOURCE_CONFIG).write_text(json.dumps({"path": str(config), "sha256": __import__("hashlib").sha256(b"{}").hexdigest()}),
                                          encoding="utf-8")
    (root / "automatic-retry.json").write_text("{}", encoding="utf-8")
    joined = osc.join_automatic_scorer(tmp_path, config)
    assert joined["status"] == "NOT_YET_OBSERVABLE" and "NOT_YET_OBSERVABLE" in joined["note"]


def test_watcher_never_sleeps_for_a_future_frame(monkeypatch, tmp_path):
    start = datetime.now(timezone.utc) - timedelta(hours=10)  # f000/f001 observable now, f020 not valid yet
    frames = _frames(tmp_path, start, [0, 1, 20])
    monkeypatch.setattr(osc, "discover", lambda run, config=None: ({**_config(start), "run": str(run)}, frames))
    scored = []

    def score(run, *, config, budget, leads):
        scored.append(set(leads))
        return {"status": "PARTIALLY_MEASURED"}

    monkeypatch.setattr(osc, "score", score)
    monkeypatch.setattr(osc.time, "sleep", lambda seconds: pytest.fail("the watcher waited for a future frame"))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_config(start)), encoding="utf-8")
    (tmp_path / "observation-scorecard").mkdir()
    assert osc.watch(tmp_path, plan) == 3
    assert scored == [{0, 1}]
    receipt = json.loads((tmp_path / "observation-scorecard" / "automatic-retry-receipt.json").read_text(encoding="utf-8"))
    assert "valid in the future" in receipt["reason"] and len(receipt["observable_from"]) == 1
