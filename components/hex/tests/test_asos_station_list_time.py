"""The scorer gives the ASOS station list the time the archive's pace needs, and a run freezes it once.

THE BREAKAGE (0.6.0, a fresh install from the published files): a 24 h global run's scorecard had all 500 ASOS rows
NOT_MEASURED, each "command timed out: .../rw_asos". Measured by hand: the globe is 266 archive networks, the
`rw_asos` of gpuwm 3.0 waits 2 s between its requests, so `rw_asos stations` took 533.6 s (the 2.9 binary: 21 to
30 s) and was given the 180 s every scorer command gets. One request timed out; its failure was shared by all 25
frames. One frame's report window took 40.1 s, inside the default.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import observation_scorecard as scorecard  # noqa: E402

GLOBE = {"west": -180.0, "south": -90.0, "east": 180.0, "north": 90.0}
VALID = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)


class Runner:
    """Stands in for the scorer's command runner: records each rw_asos call and writes what it would."""

    def __init__(self, out, timeout=180):
        self.out, self.timeout, self.calls, self.iem = Path(out), timeout, [], threading.Lock()

    def run(self, argv, *, json_output=True, acquisition=True, timeout=None):
        self.calls.append((argv[1], timeout))
        target = Path(argv[argv.index("--out") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"status": "READY", "stations": []}) if argv[1] == "stations" else "x\n",
                          encoding="utf-8")
        return {}


def test_the_measured_global_list_fits_and_a_small_region_keeps_the_default():
    networks = ",".join(f"N{i}_ASOS" for i in range(266))
    seconds = scorecard.station_list_seconds(networks, 180)
    assert seconds >= 533.6 * 1.25, "the globe's list took 533.6 s on woof 3.0's rw_asos; leave it room"
    assert seconds <= 1800, "it must still fit the score command's default acquisition budget"
    assert scorecard.station_list_seconds("IA_ASOS,IL_ASOS,MO_ASOS", 180) == 180
    assert scorecard.station_list_seconds("", 180) == 180


def test_the_scorer_asks_for_that_time_and_freezes_the_list_once_per_run(tmp_path, monkeypatch):
    pytest.importorskip("woof.obs.surface_networks")
    monkeypatch.setattr(scorecard, "binary", lambda name: name)
    runner = Runner(tmp_path / "observation-scorecard")
    runner.out.mkdir()
    scorecard.acquire_surface(VALID, tmp_path / "call-1/f012/asos", runner, region=GLOBE)
    assert [name for name, _ in runner.calls] == ["stations", "fetch", "table"]
    widened = dict(runner.calls)["stations"]
    assert widened is not None and widened >= 533.6 * 1.25, runner.calls
    assert dict(runner.calls)["fetch"] is None and dict(runner.calls)["table"] is None
    kept = list(runner.out.glob("asos-stations-*.json"))
    assert len(kept) == 1 and scorecard.station_table_ready(kept[0])
    # A later scoring call of the same run (a live cycle scores lead by lead) reads the frozen list.
    later = Runner(runner.out)
    scorecard.acquire_surface(VALID, tmp_path / "call-2/f013/asos", later, region=GLOBE)
    assert [name for name, _ in later.calls] == ["fetch", "table"]


def test_a_half_written_list_is_not_reused(tmp_path):
    path = tmp_path / "asos-stations-0.json"
    assert not scorecard.station_table_ready(path)
    path.write_text('{"status": "REA', encoding="utf-8")
    assert not scorecard.station_table_ready(path)
    path.write_text('{"status": "READY"}', encoding="utf-8")
    assert scorecard.station_table_ready(path)
