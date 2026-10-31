"""Sub-hourly history the way HRRR writes it, and a writer that does not
block the step loop on an ordinary write cadence.

Breakage they prevent (V61, 8x RTX 5090, HRRR v4 recipe, 2026-10-21): the
recipe's 15-minute history wrote an 18.7 GB full frame every 15 model
minutes into a one-ticket queue; a frame landed in about 40 s while the
model made 15 minutes in about 15 s, so the step loop blocked 10-30 s per
write and fh3 ran at 171 s/fh against about 62 without output.
"""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.io.history_selection import FULL, HistorySelection
from woof.io import wrfout


def test_full_frames_at_the_interval_and_the_subset_between():
    selection = HistorySelection.from_mapping({
        "full_history_interval_s": 3600.0, "between_preset": "minimal",
        "between_history_add": ["REFL_10CM"]}, source="test")
    produced = ["T", "PHB", "U", "V", "QVAPOR", "REFL_10CM", "T2", "RAINNC"]
    assert selection.for_lead(0.0) is selection
    assert selection.for_lead(7200.0) is selection
    assert selection.for_lead(None) is selection
    between = selection.for_lead(900.0)
    assert set(between.select(produced)) == {"T", "PHB", "REFL_10CM", "T2", "RAINNC"}
    assert selection.writes_everything  # whole hours stay the full history


def test_no_interval_keeps_every_frame_as_before():
    assert FULL.for_lead(900.0) is FULL
    assert HistorySelection.from_mapping(None) is FULL


def test_a_between_subset_without_its_interval_is_refused():
    with pytest.raises(ValueError, match="full_history_interval_s"):
        HistorySelection.from_mapping({"between_history_add": ["REFL_10CM"]}, source="t")
    with pytest.raises(ValueError, match="not a wrfout history variable"):
        HistorySelection.from_mapping({"full_history_interval_s": 3600,
                                       "between_history_add": ["NOT_A_FIELD"]}, source="t")


def test_the_packaged_hrrr_recipe_writes_hourly_full_history():
    import tomllib
    root = Path(__file__).resolve().parents[1]
    for path in (root / "woof/data/recipes/hrrr_v4_gsd41.toml",
                 root / "configs/recipes/hrrr_v4_gsd41.toml"):
        table = tomllib.loads(path.read_text(encoding="utf-8"))["output"]
        selection = HistorySelection.from_mapping(table, source=str(path))
        assert selection.full_history_interval_s == 3600.0
        assert "REFL_10CM" in selection.for_lead(900.0).select(["REFL_10CM", "U"])



@pytest.mark.usefixtures("classic_history_frames")
def test_disk_admission_prices_the_subset_frames_as_the_subset(tmp_path):
    """The packaged HRRR recipe at NOAA's 15-minute cadence, 2 h: 3 full
    frames and 6 subset frames.

    Breakage: disk admission priced all 9 as full history (147.7 GiB) and
    refused a 2 h full-CONUS run on a 146.9 GiB disk it fits.
    """
    from woof import disk_budget
    from woof.experiment import load_experiment
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs/recipes/hrrr_v4_gsd41.toml").read_text(encoding="utf-8")
    assert text.count("\nhistory_interval_s = 3600.0\n") == 1
    text = text.replace("\nhistory_interval_s = 3600.0\n", "\nhistory_interval_s = 900.0\n")
    recipe = tmp_path / "hrrr_15min.toml"
    recipe.write_text(text, encoding="utf-8")
    exp = load_experiment(recipe)
    assert float(exp.run_seconds) == 7200.0
    domain = exp.domains[0]
    projection = disk_budget.projected_run_bytes(exp, keep_checkpoints=0, fetch=None,
                                                 chain=None, render=False)
    row = projection["domains"][0]
    assert row["history_frames"] == 9
    assert disk_budget.history_frames(exp, domain, 7200.0, full_interval=3600.0) == 3
    full = row["history_frame_bytes"]
    from woof.io.history_selection import resolve
    selection = resolve(getattr(exp, "output", None), getattr(domain, "output", None))
    between = disk_budget.history_frame_bytes(domain.run, selection.for_lead(1800.0))
    assert between < full / 5
    assert abs(row["history_bytes"] - (3 * full + 6 * between)) <= full

def _shell(monkeypatch, budget):
    monkeypatch.setenv("WOOF_HISTORY_QUEUE_BYTES", str(budget))
    writer = object.__new__(wrfout.AsyncDomainWrfoutWriter)
    writer._condition = threading.Condition()
    writer._pending = 0
    writer._pending_bytes = 0
    writer._queue = writer._new_ticket_queue()
    writer._thread = SimpleNamespace(is_alive=lambda: True)
    writer._abort_event = threading.Event()
    writer._failure = None
    return writer


def _ticket(nbytes):
    return wrfout._AsyncFrame(
        path=Path("frame"), time_str="t", event=None, device_refs=(), pinned_refs=(),
        fields={"X": np.zeros(nbytes // 4, dtype=np.float32)})


def test_frames_under_the_byte_budget_never_block_the_producer(monkeypatch):
    writer = _shell(monkeypatch, budget=10_000)
    started = time.perf_counter()
    for _ in range(3):  # a one-ticket queue blocked on the second of these
        writer._admit(_ticket(3_000))
    assert time.perf_counter() - started < 1.0
    assert writer._queue.qsize() == 3


def test_a_backlog_past_the_budget_waits_until_the_writer_frees_bytes(monkeypatch):
    writer = _shell(monkeypatch, budget=5_000)
    writer._admit(_ticket(4_000))
    admitted = threading.Event()
    late = _ticket(4_000)
    thread = threading.Thread(target=lambda: (writer._admit(late), admitted.set()))
    thread.start()
    assert not admitted.wait(0.3)  # 8 000 bytes would exceed the budget
    with writer._condition:  # the worker lands the first frame
        writer._pending -= 1
        writer._pending_bytes -= 4_000
        writer._condition.notify_all()
    assert admitted.wait(2.0)
    thread.join(2.0)
    assert late.admission_wait_s > 0.2
