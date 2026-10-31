from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from tools import download_hrrr_native_subset as download


def test_download_source_hours_keep_absolute_identity_and_cycle_horizon():
    assert download._hours(
        "12,13,14,15,16,17,18",
        cycle=datetime(2026, 7, 18, 5)) == tuple(range(12, 19))
    assert download._hours(
        "40,41,42,43,44,45,46",
        cycle=datetime(2026, 7, 18, 18)) == tuple(range(40, 47))
    with pytest.raises(ValueError, match="horizon f18"):
        download._hours("18,19", cycle=datetime(2026, 7, 18, 5))


def test_file_workers_overlap_products_but_preserve_receipt_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_product(
        request: download.ProductRequest, *, workers: int, retries: int,
    ) -> dict[str, object]:
        nonlocal active, maximum_active
        assert workers == 2
        assert retries == 3
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            # Vary completion order; Executor.map must still retain request
            # order in the sealed receipt.
            time.sleep(0.01 if request.kind == "soil" else 0.03)
            payload = request.destination.name.encode("ascii")
            request.destination.write_bytes(payload)
            request.index_path.write_text("fixture\n", encoding="ascii")
            return {
                "kind": request.kind,
                "subset_path": request.destination.name,
                "subset_bytes": len(payload),
                "subset_sha256": hashlib.sha256(payload).hexdigest(),
                "source_bytes": len(payload),
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(download, "_download_product", fake_product)
    output = tmp_path / "download"
    assert download.main([
        "--cycle", "2026-07-18_00:00:00",
        "--forecast-hours", "0,1",
        "--output-root", str(output),
        "--workers", "2",
        "--file-workers", "4",
        "--retries", "3",
    ]) == 0

    assert maximum_active == 4
    receipt = json.loads((output / "download-receipt.json").read_text())
    assert receipt["file_workers"] == 4
    assert receipt["workers"] == 2
    assert receipt["source_forecast_hours"] == [0, 1]
    assert [item["subset_path"] for item in receipt["products"]] == [
        "hrrr.t00z.wrfnatf00.grib2",
        "hrrr.t00z.soilf00.grib2",
        "hrrr.t00z.wrfnatf01.grib2",
        "hrrr.t00z.soilf01.grib2",
    ]


@pytest.mark.parametrize(
    ("workers", "file_workers", "message"),
    ((2, 17, "file-workers must be 1..16"),
     (16, 8, "workers \\* file-workers may not exceed 64")),
)
def test_file_worker_resource_bounds_fail_before_download(
    tmp_path: Path, workers: int, file_workers: int, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        download.main([
            "--cycle", "2026-07-18_00:00:00",
            "--forecast-hours", "0,1",
            "--output-root", str(tmp_path / "download"),
            "--workers", str(workers),
            "--file-workers", str(file_workers),
        ])


def test_hrrr_start_extras_select_the_number_concentrations_and_2m_rh():
    # mesh-gen-fast polygon from HRRR (2026-10-29): the HRRR start asks the decoder for QNC, QNI, QNR and
    # RH2, the 561-record subset carried none of them, and the decoder refused "missing required field QNC hybrid
    # level 1". The subset now selects NCONCD/NCCICE/SPNCR on all 50 levels and RH at 2 m once.
    from tools import download_hrrr_native_subset as subset

    extras = ("QNC", "QNI", "QNR", "RH2")
    assert subset.atmosphere_record_count(extras=extras) == subset.ATMOSPHERE_RECORD_COUNT + 3 * 50 + 1
    selectors = subset.atmosphere_selectors(extras)
    assert "NCONCD:1 hybrid level" in selectors and "NCCICE:50 hybrid level" in selectors
    assert "SPNCR:25 hybrid level" in selectors and "RH:2 m above ground" in selectors
    assert subset.atmosphere_record_count() == subset.ATMOSPHERE_RECORD_COUNT
    assert "RH:2 m above ground" not in subset.atmosphere_selectors()

    rows = []
    offset = 0

    def add(variable, level, raw_extra=""):
        nonlocal offset
        rows.append(subset.IndexRow(len(rows) + 1, offset, variable, level,
                                    f"{len(rows) + 1}:{offset}:d=2026102812:{variable}:{level}:anl:{raw_extra}"))
        offset += 100

    for name in subset.HYBRID_FIELDS + ("NCONCD", "NCCICE", "SPNCR"):
        for level in range(1, 51):
            add(name, f"{level} hybrid level")
    for variable, level in subset.SURFACE_FIELDS + (("RH", "2 m above ground"), ("RH", "0C isotherm")):
        add(variable, level)
    plain = subset._atmosphere_selection(tuple(rows))
    assert len(plain) == subset.ATMOSPHERE_RECORD_COUNT
    started = subset._atmosphere_selection(tuple(rows), extras=extras)
    assert len(started) == subset.ATMOSPHERE_RECORD_COUNT + 151
    picked = {(rows[i].variable, rows[i].level) for i in started}
    assert ("NCONCD", "1 hybrid level") in picked and ("RH", "2 m above ground") in picked
    assert ("RH", "0C isotherm") not in picked



def test_the_subset_takes_the_optional_surface_records_the_bridge_reads_from_a_whole_file():
    # Breakage this prevents: hrrr_grib2_bridge reads PBLH (HPBL) and the low/middle/high cloud cover from a
    # whole wrfnat whenever it publishes them (OPTIONAL_SURFACE_SPECS); a subset without them prepared a different
    # start (PBLH = 0, no cloud-cover seed) and a series mixing the two byte modes was refused 80-146 s into the
    # preparation.  The subset takes them where the index lists them, outside the required count bar.
    from tools import download_hrrr_native_subset as subset

    rows = []
    offset = 0

    def add(variable, level, raw_extra=""):
        nonlocal offset
        rows.append(subset.IndexRow(len(rows) + 1, offset, variable, level,
                                    f"{len(rows) + 1}:{offset}:d=2026102912:{variable}:{level}:anl:{raw_extra}"))
        offset += 100

    for name in subset.HYBRID_FIELDS:
        for level in range(1, 51):
            add(name, f"{level} hybrid level")
    for variable, level in subset.SURFACE_FIELDS:
        add(variable, level)
    without = tuple(rows)
    for variable, level in subset.OPTIONAL_SURFACE_FIELDS:
        add(variable, level)
    add("HPBL", "surface", "0-1 hr acc fcst")  # an accumulated twin is never taken
    index_text = "\n".join(row.raw for row in rows)

    assert subset.optional_surface_published(index_text) == subset.OPTIONAL_SURFACE_FIELDS
    assert subset.optional_surface_selectors(subset.optional_surface_published(index_text)) == (
        "HPBL:surface", "LCDC:low cloud layer", "MCDC:middle cloud layer", "HCDC:high cloud layer")
    picked = subset._atmosphere_selection(tuple(rows))
    assert len(picked) == subset.ATMOSPHERE_RECORD_COUNT + 4
    assert {(rows[i].variable, rows[i].level) for i in picked} >= set(subset.OPTIONAL_SURFACE_FIELDS)
    # A cycle that lists none of them takes exactly the records it always did.
    assert subset.optional_surface_published("\n".join(row.raw for row in without)) == ()
    assert len(subset._atmosphere_selection(without)) == subset.ATMOSPHERE_RECORD_COUNT
    # The required bar still fires: a missing surface field is refused whatever optional records are listed.
    with pytest.raises(subset.IndexInventoryError):
        subset._atmosphere_selection(tuple(row for row in rows if row.variable != "ICEC"))
