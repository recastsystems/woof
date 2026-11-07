"""The declared upper-air donor above a source that tops out low (ECCC's HRDPS at 50 hPa).

The breakage: the RRFS producer's case 7 (hrrrv5, lapse-rate mode) refused
every HRDPS start, "temperature extrapolation above the first-guess column
top is not implemented for the lapse-rate mode", because the RRFS v2 L60 grid
tops at 25.9 km and HRDPS's column stops at 50 hPa.  The composition now
declares GFS above it; these tests hold the plan, the WPS record append and
the blend that leaves no seam at the source's top.
"""

from __future__ import annotations

import json
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HRDPS_PA = [101500, 100000, 98500, 97000, 95000, 92500, 90000, 87500, 85000, 80000, 75000, 70000,
            65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000, 27500, 25000, 22500, 20000,
            17500, 15000, 10000, 5000]


def _donor(name="rw-wps-gem-hrdps-grib2"):
    from woof.ingest.upper_air_donor import validate_upper_air_donor

    composition = json.loads((ROOT / "woof" / "authorities" / f"{name}.composition.json")
                             .read_text(encoding="utf-8"))
    return validate_upper_air_donor(composition["upper_air_donor"])


def test_hrdps_gets_gfs_above_its_50_hpa_top_and_its_top_level_blended():
    from woof.ingest.upper_air_donor import plan_levels

    plan = plan_levels(HRDPS_PA, _donor())
    assert plan["source_top_pa"] == 5000.0
    assert plan["append_pa"][0] == 4000.0 and plan["append_pa"][-1] == 100.0
    assert len(plan["append_pa"]) == 10
    # 100 hPa (the blend bottom) is the source's own; 50 hPa (its top) is the donor's.
    assert plan["blend"] == [(5000.0, 1.0)]


def test_a_source_that_already_reaches_the_donor_levels_gets_nothing():
    from woof.ingest.upper_air_donor import plan_levels

    plan = plan_levels(HRDPS_PA + [3000, 2000, 1000, 700, 500, 300, 100], _donor())
    assert plan["append_pa"] == [] and plan["blend"] == []


def test_the_packaged_hrdps_composition_validates_with_its_donor():
    from woof.mapped_composition import load_composition

    authorities = ROOT / "woof" / "authorities"
    contract = load_composition(authorities / "rw-wps-gem-hrdps-grib2.composition.json",
                                authorities / "rw-wps-gem-hrdps-grib2.mapping.json")
    assert contract["upper_air_donor"]["source_id"] == "gfs"


def test_an_undeclared_key_is_refused():
    from woof.ingest.upper_air_donor import validate_upper_air_donor

    raw = json.loads((ROOT / "woof" / "authorities" / "rw-wps-gem-hrdps-grib2.composition.json")
                     .read_text(encoding="utf-8"))["upper_air_donor"]
    with pytest.raises(ValueError, match="unknown key"):
        validate_upper_air_donor({**raw, "levels": [100]})


def _group(field, level, data, *, nx, ny):
    def record(body):
        return struct.pack(">i", len(body)) + body + struct.pack(">i", len(body))
    header = bytearray(156)
    header[0:24] = b"2026-10-07_18:00:00".ljust(24)
    header[28:60] = b"gem-hrdps".ljust(32)
    header[60:69] = field.ljust(9).encode()
    header[69:94] = b"K".ljust(25)
    header[94:140] = field.ljust(46).encode()
    header[140:144] = struct.pack(">f", float(level))
    header[144:156] = struct.pack(">iii", nx, ny, 0)
    projection = b"SWCORNER" + struct.pack(">fffff", 45.0, -76.0, 0.5, 0.5, 6371.229)
    return b"".join(record(body) for body in (
        struct.pack(">i", 5), bytes(header), projection, struct.pack(">i", 1),
        np.asarray(data, dtype=">f4").tobytes()))


def test_the_box_gains_the_donor_column_and_its_top_meets_the_donor(tmp_path, monkeypatch):
    from woof.ingest import upper_air_donor
    from woof.ingest.upper_air_donor import _groups, append_upper_air_to_wps

    nx, ny = 3, 2
    path = tmp_path / "MET_2026-10-07_18"
    blob = b""
    for level in (10000, 5000):
        blob += _group("TT", level, np.full((ny, nx), 215.0), nx=nx, ny=ny)
        blob += _group("UU", level, np.full((ny, nx), 10.0), nx=nx, ny=ny)
        blob += _group("PRES", level, np.full((ny, nx), float(level)), nx=nx, ny=ny)
    blob += _group("TT", 200100, np.full((ny, nx), 290.0), nx=nx, ny=ny)
    path.write_bytes(blob)
    donor = _donor()
    calls = {}

    def fake_fetch(donor_, cycle, lead, root, records):
        calls["records"] = records
        calls["cycle"] = (cycle, lead)
        return tmp_path / "donor.grib2", {"url": "test"}

    def fake_decode(donor_, grib, wanted):
        planes = {name: {level: np.full((181, 360), 200.0 + level / 1000.0) for level in levels}
                  for name, levels in wanted.items()}
        return planes, np.linspace(90.0, -90.0, 181), np.arange(0.0, 360.0, 1.0)

    (tmp_path / "donor.grib2").write_bytes(b"GRIB7777")
    monkeypatch.setattr(upper_air_donor, "_fetch", fake_fetch)
    monkeypatch.setattr(upper_air_donor, "_decode", fake_decode)
    receipt = append_upper_air_to_wps(path, donor, valid_time=datetime(2026, 10, 7, 18))
    assert receipt["appended_levels"] == 10
    assert receipt["fields"] == ["TT", "UU", "PRES"]
    assert calls["cycle"] == (datetime(2026, 10, 7, 18), 0)
    # TT and UU at 50 hPa (the blend) and 40-1 hPa; PRES is the level itself, never fetched.
    assert {name for name, _ in calls["records"]} == {"TMP", "UGRD"}
    assert ("TMP", "50 mb") in calls["records"] and ("TMP", "1 mb") in calls["records"]
    groups = {(field, level): np.frombuffer(body[4], ">f4").reshape(ny, nx)
              for field, level, body in _groups(path.read_bytes())}
    assert groups[("TT", 10000.0)][0, 0] == pytest.approx(215.0)       # below the blend: the source's own
    assert groups[("TT", 5000.0)][0, 0] == pytest.approx(205.0)        # its top: the donor's
    assert groups[("TT", 100.0)][0, 0] == pytest.approx(200.1, abs=1e-4)
    assert groups[("PRES", 1000.0)][1, 2] == 1000.0
    assert groups[("TT", 200100.0)][0, 0] == 290.0
    assert len([key for key in groups if key[0] == "UU"]) == 12
