"""The event page's layouts count their download and preparation, and time them from their own source.

Breakage these prevent, measured on test host 4 on 2026-09-26: the event page
projected the 2021-12-10 tornado's 12 GB layout at 15.2 GiB and picked it
for a disk with 24 GiB free, and the run had written 28.9 GB by its first
forecast step (18.9 GB of HRRR files and 9.6 GB of preparation); and the
page promised 15 minutes to all pictures for that layout, whose download
and preparation alone took 707 s, because every unmeasured row was given
the two minute mean of the measured ERA5 runs.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from woof import disk_budget, download_budget

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
#: The EF4 2021 run of the 12 GB layout: du of its run folder by part at
#: the first forecast step (downloads, chain/hrrr-root-prep, chain/run).
EF4_DOWNLOAD, EF4_PREPARATION, EF4_FIRST_FRAME = 18_879_788_730, 9_552_000_842, 547_618_443


def _ef4_floor() -> float:
    """What the run had written at its first step, plus the 15 frames and all pictures still to come."""
    return (EF4_DOWNLOAD + EF4_PREPARATION + EF4_FIRST_FRAME + 15 * EF4_FIRST_FRAME
            + disk_budget.projected_picture_bytes(386, 374, 15 * 3600, 3600))


def test_the_ef4_12gb_recipe_row_carries_the_whole_figure():
    document = json.loads((ROOT / "woof/gui/seed/recipes/tornado-2021-2112102054-01.json")
                          .read_text(encoding="utf-8"))
    row = next(r for r in document["recipes"][0]["cards"] if r["card_gb"] == 12)
    assert row["disk_gib"] * GIB >= _ef4_floor()
    assert "the download is extra" not in row["disk_basis"]
    assert row["download"]["source"] == "hrrr" and row["download"]["bytes"] >= 0.95 * EF4_DOWNLOAD


def test_every_recipe_row_counts_its_download_and_preparation():
    for path in sorted((ROOT / "woof/gui/seed/recipes").glob("*.json")):
        for recipe in json.loads(path.read_text(encoding="utf-8"))["recipes"]:
            for row in recipe["cards"]:
                if not row.get("fits"):
                    continue
                assert row["download"]["bytes"] is not None, (path.name, row["card_gb"])
                assert "the download is extra" not in row["disk_basis"], (path.name, row["card_gb"])
                assert row["disk_gib"] * GIB >= row["download"]["bytes"] + row["download"]["preparation_bytes"]


def _estimate_times():
    spec = importlib.util.spec_from_file_location(
        "estimate_times", ROOT / "tools" / "wiki_seed" / "estimate_times.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_lead_in_is_the_rows_own_download_and_preparation():
    times = _estimate_times()
    rates = times.lead_in_rates()
    document = json.loads((ROOT / "woof/gui/seed/recipes/tornado-2021-2112102054-01.json")
                          .read_text(encoding="utf-8"))
    row = next(r for r in document["recipes"][0]["cards"] if r["card_gb"] == 12)
    seconds, words = times.lead_in_seconds(row, rates)
    # 707 s measured before the first step (fetch 503 s, preparation 204 s), not the old 120 s.
    assert 0.75 * 707 <= seconds <= 1.25 * 707, seconds
    assert "HRRR" in words
    assert row["est_total_minutes"] * 60 >= seconds + row["est_minutes"] * 60 - 5 * 60


def test_an_era5_row_keeps_the_era5_lead_in():
    times = _estimate_times()
    rates = times.lead_in_rates()
    budget = download_budget.download_estimate(
        {"source": "era5", "cycle": "2023-10-24T12", "hours": 24, "cadence": 6,
         "area": "5.27,-109.23,26.27,-89.97", "era5_provider": "arco"})
    # The Otis 8 GB layout's grids, as it ran: its preparation scales with them, not with its small download.
    row = {"download": dict(budget, preparation_bytes=0, chain="experiment"),
           "domains": [{"nx": 136, "ny": 158, "nz": 49}, {"nx": 192, "ny": 272, "nz": 49}]}
    seconds, words = times.lead_in_seconds(row, rates)
    # Measured: fetch 153 s and preparation 23 s.
    assert 0.6 * 176 <= seconds <= 1.4 * 176, seconds
    assert "ERA5" in words
