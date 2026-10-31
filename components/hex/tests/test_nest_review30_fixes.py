"""2.9.7 release review 30, findings 1 and 2, each tested exactly.

1. A trigger firing's card claim is made inside the parent forecast.  Recorded
   under the parent's PID, the claim counted the parent's ~18 GB as the nest's
   allocation, so it counted 0 unallocated bytes and a second firing one look
   later passed on memory the first had already claimed.
2. add-nest took every parent frame: a parent writing every 30 min fed a 6 h
   nest 7 frames covering 0-3 h, and the nest's posting waited for +4 h until
   source_behind.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import pytest

from woof.hex import add_nest_door as door
from woof.hex import export_door, nest_budget, render_door
from woof.hex.card_ledger import CardLedger

GIB = 2**30
PARENT = 4242


# --------------------------------------------------------------------------- 1. claims under the parent's PID

def test_two_trigger_firings_cannot_both_pass_beside_a_parent_that_holds_card_memory(tmp_path):
    held = {PARENT: 18 * GIB}  # the running hex parent, as nvidia-smi reports it on a bare VM
    ledger = CardLedger(tmp_path / "card.json", free_bytes=lambda: 8 * GIB,
                        used_bytes=lambda pids: sum(held.get(int(p), 0) for p in pids),
                        alive=lambda pid: int(pid) == PARENT)
    (tmp_path / nest_budget.FIRE_BUDGET_NAME).write_text(json.dumps({"nest_bytes": 4 * GIB}))
    fits = nest_budget.card_fit_hook(tmp_path, ledger, owner_pid=PARENT)
    first, second = {"index": 0}, {"index": 1}
    assert fits(first)[0]
    ok, why = fits(second)
    assert not ok, "4.4 GiB claimed of 8 GiB free: a second 4.4 GiB nest does not fit"
    assert "GiB free after other nests' claims" in why
    # The claim stays alive on the parent alone until add-nest adopts it, and
    # still counts in full once adopted by a process that holds no card memory.
    assert ledger.holds(first["card_reservation"])
    assert ledger.adopt(first["card_reservation"], [PARENT + 1])
    rows = json.loads(ledger.path.read_text())["reservations"]
    assert ledger.unallocated(rows) == int(4 * GIB * 1.1)


# --------------------------------------------------------------------------- 2. the parent's history cadence

def _fake_convert(convert_exe, *, histories, scratch, simulation_start, **_kwargs):
    # One frame per history, as rw_mpas_convert writes them: export --follow converts the frames that are ready in
    # one call (woof hex 0.4.11), not one call a frame.
    frames = Path(scratch) / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rows = []
    for history in histories:
        stamp = Path(history).stem.split(".")[-1]
        out = frames / f"wrfout_d01_{stamp}"
        out.write_bytes(b"frame")
        rows.append({"output": str(out), "output_sha256": "x" * 64, "valid_time": stamp})
    return {"frames": rows}, ["fake"]


def test_follow_takes_only_the_nests_hourly_leads_from_a_30_min_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(render_door, "convert_frames", _fake_convert)
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda value: Path("fake"))
    run = tmp_path / "run"
    (run / "forecast").mkdir(parents=True)
    start = dt.datetime(2025, 5, 19, 12)
    with (run / "forecast" / "history-ready.jsonl").open("w", encoding="utf-8") as stream:
        for k in range(13):  # 0-6 h every 30 min
            history = run / "forecast" / f"history.{start + dt.timedelta(minutes=30 * k):%Y-%m-%d_%H_%M_%S}.nc"
            history.write_bytes(b"h")
            stream.write(json.dumps({"step": k, "path": str(history)}) + "\n")
    args = argparse.Namespace(
        forecast_dir=str(run), mesh="grid.nc", init="init.nc", simulation_start="2025-05-19_12:00:00",
        follow=True, follow_frames=7, follow_timeout_minutes=0.02, follow_poll_seconds=0.01,
        follow_from="2025-05-19_12:00:00", follow_every_minutes=60.0,
        out=str(tmp_path / "export"), scratch=None, window="region-latlon:35.5,-97.5,1300,1300",
        remap="product", view="nest", field_set="full", nc_format="cdf5", model_label="hex",
        keep_scratch=False, overwrite=False, convert_exe=None)
    assert export_door.run_export(args) == 0
    rows = [json.loads(line) for line in
            (tmp_path / "export" / export_door.READY_NAME).read_text(encoding="utf-8").splitlines()]
    assert [row["valid_time"] for row in rows] == [f"2025-05-19_{12 + h}_00_00" for h in range(7)]


def _live_parent(tmp_path, history_minutes):
    run = tmp_path / "hex"
    (run / "forecast").mkdir(parents=True)
    (run / "forecast" / "forecast-start.json").write_text(json.dumps({
        "schema": "gpuwm-hex.forecast-start/1", "grid": "/w/g.nc", "init": "/w/i.nc",
        "start_time": "2025-05-19_12:00:00", "forecast_hours": 12.0,
        "history_every_minutes": history_minutes, "history_from_minutes": 0.0}))
    return run


def _args(run, tmp_path, *extra):
    parser = argparse.ArgumentParser()
    door.add_add_nest_parser(parser.add_subparsers())
    return parser.parse_args(["add-nest", str(run), "--lat", "35", "--lon", "-97", "--wait-minutes", "0",
                              "--out", str(tmp_path / "nests"), *extra])


def test_add_nest_from_a_30_min_parent_starts_on_the_hour_and_follows_hourly_leads(tmp_path, monkeypatch):
    run = _live_parent(tmp_path, 30.0)
    start = dt.datetime(2025, 5, 19, 12)
    closed = [(start + dt.timedelta(minutes=30 * k), Path(f"h{k}.nc")) for k in range(2)]  # 12:00, 12:30
    monkeypatch.setattr(door, "closed_frames", lambda forecast_dir: closed)
    monkeypatch.setattr(door, "disk_check", lambda *a, **k: {})
    monkeypatch.setattr(door, "nest_config", lambda *a, **k: tmp_path / "nest.json")
    monkeypatch.setattr(door, "nest_estimate_bytes", lambda config: None)
    monkeypatch.setattr(door, "_engine", lambda name: name)
    spawned = []

    class Spawned(Exception):
        pass

    def spawn(argv, log):
        spawned.append([str(a) for a in argv])
        raise Spawned

    monkeypatch.setattr(door, "_spawn", spawn)
    with pytest.raises(Spawned):
        door.add_nest(_args(run, tmp_path, "--start", "now", "--hours", "6"))
    export = spawned[0]
    assert export[export.index("--follow-from") + 1] == "2025-05-19_12:00:00", "12:30 is not an hour cycle"
    assert export[export.index("--follow-every-minutes") + 1] == "60"
    assert export[export.index("--follow-frames") + 1] == "7"


def test_add_nest_refuses_a_cadence_off_the_parents_history_spacing(tmp_path):
    run = _live_parent(tmp_path, 180.0)
    with pytest.raises(door.AddNestRefusal, match="not a whole multiple of the parent's 180 min"):
        door.add_nest(_args(run, tmp_path, "--start", "now"))
