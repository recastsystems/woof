"""C1 recording must preserve destination checks, bytes and measured clocks."""
import json
import os
from pathlib import Path

import pytest

from woof.hex import time_ledger


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    monkeypatch.delenv("WOOF_CARD", raising=False)
    monkeypatch.delenv("CARD_QUEUE_CARDS", raising=False)
    monkeypatch.setenv("WOOF_CARD_QUEUE_PREFIX", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    time_ledger.configure(None)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_default_retains_original_clocks_without_creating_destination(tmp_path):
    pytest.importorskip("woof.timeline.ledger", reason="the engine ships no woof.timeline (woof 2.8.7 and earlier)")
    out = tmp_path / "run"
    time_ledger.configure(out)
    time_ledger.emit("prep.host", "begin", t_mono=100, t_unix=1700000000)
    time_ledger.emit("prep.host", "end", t_mono=105, t_unix=1700000005)
    assert not out.exists()
    out.mkdir()
    time_ledger.emit("prep.destination")
    events = rows(out / "timeline.jsonl")
    assert [event["t_mono"] for event in events[:2]] == [100, 105]
    assert all(event["card"] == 4 and event["product"] == "hex" for event in events)
    assert len(events) == 3


def test_rental_destination_and_ids_are_inherited(tmp_path, monkeypatch):
    pytest.importorskip("woof.timeline.ledger", reason="the engine ships no woof.timeline (woof 2.8.7 and earlier)")
    path = tmp_path / "rental.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(path))
    monkeypatch.setenv("WOOF_RENTAL_ID", "rental-test")
    monkeypatch.setenv("WOOF_RUN_ID", "run-test")
    out = tmp_path / "absent"
    time_ledger.configure(out)
    time_ledger.emit("prep.door", "begin")
    assert not out.exists()
    event = rows(path)[0]
    assert event["run_id"] == "run-test" and event["rental_id"] == "rental-test"


def test_failure_warns_once_and_names_file(tmp_path, capsys):
    pytest.importorskip("woof.timeline.ledger", reason="the engine ships no woof.timeline (woof 2.8.7 and earlier)")
    path = tmp_path / "file"
    path.write_bytes(b"retained")
    time_ledger.configure(path / "child")
    os.environ["WOOF_TIME_LEDGER"] = str(path / "bad.jsonl")
    try:
        time_ledger.emit("finish")
        time_ledger.emit("finish")
    finally:
        os.environ.pop("WOOF_TIME_LEDGER")
    error = capsys.readouterr().err
    assert error.count("cannot append") == 1
    assert str(path / "bad.jsonl") in error
    assert path.read_bytes() == b"retained"


def test_unknown_physical_card_is_null(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-device-uuid")
    assert time_ledger.card_index() is None
    monkeypatch.setenv("WOOF_CARD", "7")
    assert time_ledger.card_index() == 7


def test_preparation_refusal_pairs_events_and_preserves_error(tmp_path, monkeypatch, capsys):
    pytest.importorskip("woof.timeline.ledger", reason="the engine ships no woof.timeline (woof 2.8.7 and earlier)")
    from woof.hex.cli import main

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(path))
    argv = ["lbc", "--grid", str(tmp_path / "missing-init.nc"),
            "--met-dir", str(tmp_path / "met"), "--out-dir", str(tmp_path / "lbc"),
            "--start-time", "2026-10-07_12:00:00", "--stop-time", "2026-10-07_13:00:00",
            "--nfglevels", "51", "--extrap-airtemp", "constant", "--use-spechumd", "yes"]
    assert main(argv) == 2
    error = capsys.readouterr().err
    assert "missing-init.nc" in error
    assert [(event["stage"], event["phase"]) for event in rows(path)] == [
        ("prep.lbc", "begin"), ("prep.lbc", "end")]

    # A broken destination must retain the same refusal, not a timing error.
    path.write_bytes(b"retained")
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(path / "child.jsonl"))
    assert main(argv) == 2
    assert "missing-init.nc" in capsys.readouterr().err
    assert path.read_bytes() == b"retained"


def test_cache_load_is_paired_even_on_a_miss(tmp_path, monkeypatch):
    pytest.importorskip("woof.timeline.ledger", reason="the engine ships no woof.timeline (woof 2.8.7 and earlier)")
    from woof.hex import prep_cache
    path = tmp_path / "rental.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(path))
    assert prep_cache._load(tmp_path, "0" * 64) is None
    events = rows(path)
    assert [(row["stage"], row["phase"]) for row in events] == [
        ("prep.cache_load", "begin"), ("prep.cache_load", "end")]


def test_emitted_original_clocks_reach_the_real_timeline_cli(tmp_path, monkeypatch):
    """A calculator compatibility fixture, not a measured forecast fraction."""
    pytest.importorskip("woof.timeline.ledger", reason="the engine ships no woof.timeline (woof 2.8.7 and earlier)")
    import subprocess
    import sys

    ledger = tmp_path / "contract.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(ledger))
    monkeypatch.setenv("WOOF_RUN_ID", "contract-test")
    time_ledger.configure(tmp_path)
    for stage, phase, offset in (
        ("forecast.construct", "begin", 0),
        ("forecast.construct", "end", 2),
        ("forecast.integrate", "begin", 2),
        ("forecast.first_step", "begin", 2),
        ("forecast.first_step", "end", 3),
        ("forecast.integrate", "end", 10),
        ("finish", "mark", 12),
    ):
        time_ledger.emit(stage, phase, t_mono=100 + offset, t_unix=1700000000 + offset)
    output = tmp_path / "timeline.json"
    proc = subprocess.run([
        sys.executable, "-m", "woof.cli", "timeline", "report", str(ledger),
        "--proxy-start", "1700000000", "--proxy-end", "1700000012",
        "--product", "hex", "--output", str(output),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(output.read_text())
    assert report["mode"] == "proxy"
    assert report["ff"] is None
    assert report["integrate_union_seconds"] == 8
    assert report["proxy_ff"] == pytest.approx(8 / 12)
    assert not any("unmatched" in warning for warning in report["warnings"])


def test_uuid_visibility_uses_the_mutex_physical_card_index(monkeypatch):
    monkeypatch.delenv("WOOF_CARD", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-opaque-assignment")
    monkeypatch.setenv("CARD_QUEUE_CARDS", "1")
    assert time_ledger.card_index() == 1
