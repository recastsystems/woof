"""ECMWF open data as it was published, not only as it is today: a hex run from a past IFS cycle at a later hour.

THE BREAKAGE THESE TESTS PREVENT (WOOF run_cd20e718a3bfea1a, 2026-11-13): a hex region from the 2024-05-14 12Z IFS run,
started at f198, built its mesh and static on a rented box and then failed at its first boundary file with "no address
has it": the row asked every lead for the surface geopotential z of the cycle's step-0 file, which ECMWF published only
from the 2024-12-02 18Z run on. The same row also asked only for today's soil records (sot/vsw; the 2024 runs published
st, stl2-4 and swvl1-4), only for the /oper/ path (the 06/18 UTC runs before 2026-05-12 06Z are under /scda/), and
admitted leads to 360 h for runs that stopped at 240 h (06/18 UTC: 90 h) before 2024-11-12 06Z.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from woof.hex import init_sources as src
from woof.hex.init_sources import Cycle, InitSourceRefusal, load_table, plan_products, resolve_source
from woof.hex.source_horizon import beyond_source, last_posted_lead

OLD_SOIL = {"st", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4"}
REFERENCE = "20241203/00z/ifs/0p25/oper/20241203000000-0h-oper-fc"


def _ifs():
    return resolve_source(load_table(), "ifs")


def _plans(cycle, lead):
    return plan_products(_ifs(), Cycle.parse(cycle), None, lead)


def _role(plans, role):
    found = [plan for plan in plans if plan.role == role]
    assert len(found) == 1, [plan.role for plan in plans]
    return found[0]


def _sfc_params(plan):
    return {p for clause in plan.index_select if clause["levtype"] == "sfc" for p in clause["param"]}


@pytest.mark.parametrize("lead", [0, 198])
def test_every_lead_selects_both_soil_encodings_and_reads_terrain_from_the_donor(lead):
    plans = _plans("2024-05-14T12", lead)
    primary = _role(plans, "primary")
    assert OLD_SOIL <= _sfc_params(primary)
    assert {"sot", "vsw"} <= {p for c in primary.index_select if c["levtype"] == "sol" for p in c["param"]}
    assert "z" not in _sfc_params(primary)
    donor = _role(plans, "terrain_donor")
    assert donor.index_select == {"param": "z", "levtype": "sfc"} and donor.fields == ["SOILGEO"]
    addresses = [donor.url, *donor.url_mirrors]
    indexes = [donor.index_url, *donor.index_mirrors]
    assert len(addresses) == len(indexes) == 6
    assert "/20240514/12z/ifs/0p25/oper/20240514120000-0h-oper-fc.grib2" in addresses[0]
    assert all(REFERENCE in url for url in addresses[-2:] + indexes[-2:])


def test_a_cycle_without_z_takes_the_reference_orography(monkeypatch, tmp_path):
    """The run's own step-0 index answers but holds no z (2024-05-14 12Z): the donor reads the 2024-12-03 00Z z."""
    donor = _role(_plans("2024-05-14T12", 198), "terrain_donor")
    own = "\n".join(json.dumps({"levtype": "sfc", "param": p, "_offset": 100 * k, "_length": 100})
                    for k, p in enumerate(["2t", "sp", "st", "swvl1"]))
    reference = json.dumps({"levtype": "sfc", "param": "z", "_offset": 0, "_length": 8})
    asked = []

    def fake_get(url, *, byte_range=None, attempts=10):
        asked.append(url)
        if url.endswith(".index") and "/20240514/12z/" in url and "/oper/" in url:
            return own.encode()
        if url.endswith(".index") and REFERENCE in url:
            return reference.encode()
        if REFERENCE in url and byte_range is not None:
            return b"GRIB\x00\x00\x00\x02"
        raise InitSourceRefusal(f"{url} answered HTTP 404")

    monkeypatch.setattr(src, "_http_get", fake_get)
    receipt = src._fetch_selected(donor, tmp_path / donor.filename, 1)
    assert REFERENCE in receipt["url"] and receipt["records"] == 1
    assert (tmp_path / donor.filename).read_bytes().startswith(b"GRIB")
    assert any("/scda/" in url for url in asked)


def test_a_cycle_with_z_reads_its_own_step_zero():
    donor = _role(_plans("2026-10-23T12", 198), "terrain_donor")
    assert "/20261023/12z/ifs/0p25/oper/20261023120000-0h-oper-fc.grib2" in donor.url


def test_the_scda_path_is_an_address_of_every_product():
    for lead in (0, 6):
        for plan in _plans("2024-05-22T18", lead):
            if plan.role in ("surface-input", "snow_donor"):  # GFS companions (sea ice; snow before 2025-11-20 12Z)
                continue
            urls = [plan.url, *plan.url_mirrors]
            assert any("/20240522/18z/ifs/0p25/scda/" in u and "-scda-fc." in u for u in urls), plan.role
            assert urls[0].startswith("https://storage.googleapis.com/") and "/oper/" in urls[0]


@pytest.mark.parametrize("cycle,lead", [
    ("2024-05-14T12", 240), ("2024-05-14T12", 198), ("2024-05-14T06", 90), ("2024-11-12T06", 144),
    ("2024-11-12T12", 246), ("2024-11-12T12", 360), ("2026-10-23T12", 354),
])
def test_posted_hours_of_each_era_plan(cycle, lead):
    assert _plans(cycle, lead)


@pytest.mark.parametrize("cycle,lead,words", [
    ("2024-05-14T12", 246, "every 3 h to f144, then every 6 h to f240 (its range before the 2024-11-12T06 UTC run)"),
    ("2024-05-14T06", 93, "every 3 h to f090 (its range before the 2024-11-12T06 UTC run)"),
    ("2024-11-12T00", 300, "to f240"),
])
def test_hours_a_past_run_never_posted_are_refused_at_planning_in_words(cycle, lead, words):
    with pytest.raises(InitSourceRefusal, match="outside the row's forecast cadence/range") as refused:
        _plans(cycle, lead)
    assert words in str(refused.value)


def test_a_past_run_ends_where_it_ended():
    row = _ifs()
    from datetime import datetime, timezone
    old = datetime(2024, 5, 14, 12, tzinfo=timezone.utc)
    assert last_posted_lead(row, old) == 240
    assert last_posted_lead(row, datetime(2024, 5, 14, 18, tzinfo=timezone.utc)) == 90
    assert last_posted_lead(row, datetime(2025, 5, 14, 12, tzinfo=timezone.utc)) == 360
    assert last_posted_lead(row, 12) == 360  # the hour alone: today's range
    assert beyond_source(row, old, 198, 72)["source_last_lead_hours"] == 240


def test_the_customer_region_plans_five_boundary_files():
    """run_cd20e718a3bfea1a: 2024-05-14 12Z from f198 for 24 h reads f198..f222 every 6 h (all posted)."""
    from woof.hex.lbc_producer import source_endpoint_commands

    commands = source_endpoint_commands(source="ifs", cycle="2024-05-14T12", hours=24, out_dir="/tmp/x",
                                        lead_hours=198)
    leads = [int(c["argv"][c["argv"].index("--lead-hours") + 1]) for c in commands]
    assert leads == [198, 204, 210, 216, 222]
    assert commands[0]["valid_time"] == "2024-05-22_18:00:00"


# ---------------------------------------------------------------------------
# the 2024 soil records decode (the row's Vtable through the real met_intermediate)
# ---------------------------------------------------------------------------
def _signed(value: int) -> int:
    """GRIB2 sign-and-magnitude 32-bit integer."""
    return (abs(value) | 0x80000000) if value < 0 else value


def _grib2(discipline, category, number, first, second, values, nx=4, ny=3):
    """One GRIB2 message (lat-lon template 3.0, product 4.0, simple packing 5.0) with the 2024 soil encoding:
    ``first``/``second`` are (scale factor, scaled value) on depth surface 106; ``None`` value means missing."""
    sec1 = struct.pack(">IBHHBBBHBBBBBBB", 21, 1, 98, 0, 27, 0, 1, 2024, 5, 14, 12, 0, 0, 0, 1)
    grid = struct.pack(">BBIBIBIIIIIIIBIIII", 6, 0, 0, 0, 0, 0, 0, nx, ny, 0, 0xFFFFFFFF,
                       _signed(40_000_000), 260_000_000, 0x30, _signed(39_500_000),
                       260_750_000, 250_000, 250_000)
    grid += b"\x00"  # scanning mode
    sec3 = struct.pack(">IBBIBBH", 14 + len(grid), 3, 0, nx * ny, 0, 0, 0) + grid
    def level(pair):
        scale, value = pair
        return struct.pack(">BI", scale, 0xFFFFFFFF if value is None else value)
    tmpl = struct.pack(">BBBBBHBBI", category, number, 2, 0, 154, 0, 0, 1, 198)
    tmpl += struct.pack(">B", 106) + level(first) + struct.pack(">B", 106) + level(second)
    sec4 = struct.pack(">IBHH", 9 + len(tmpl), 4, 0, 0) + tmpl
    scaled = [round(v * 100) for v in values]
    reference = min(scaled)
    packed = b"".join(struct.pack(">H", s - reference) for s in scaled)
    sec5 = struct.pack(">IBIH", 21, 5, nx * ny, 0) + struct.pack(">fhhBB", float(reference), 0, 2, 16, 0)
    sec6 = struct.pack(">IBB", 6, 6, 255)
    sec7 = struct.pack(">IB", 5 + len(packed), 7) + packed
    body = sec1 + sec3 + sec4 + sec5 + sec6 + sec7 + b"7777"
    return b"GRIB" + struct.pack(">HBBQ", 0, discipline, 2, 16 + len(body)) + body


#: (name, discipline, category, number, first, second): as the 2024-05-14 12Z f198 file encodes them (grib2_inventory).
SOIL_2024 = [
    ("ST000007", 2, 0, 2, (0, 0), (2, 7)),
    ("ST007028", 192, 128, 170, (0, 7), (0, 28)),
    ("ST028100", 192, 128, 183, (0, 28), (0, 100)),
    ("ST100289", 192, 128, 236, (0, 100), (0, None)),
    ("SM000007", 192, 128, 39, (0, 0), (0, 7)),
    ("SM007028", 192, 128, 40, (0, 7), (0, 28)),
    ("SM028100", 192, 128, 41, (0, 28), (0, 100)),
    ("SM100289", 192, 128, 42, (0, 100), (0, None)),
]


def test_the_2024_soil_records_decode_to_the_four_layers(tmp_path):
    try:
        writer = src.resolve_writer(None)
    except Exception as error:  # an install without woof's bridges cannot run the writer
        pytest.skip(f"met_intermediate is not installed: {error}")
    grib = tmp_path / "soil-2024.grib2"
    with grib.open("wb") as handle:
        for k, (_, discipline, category, number, first, second) in enumerate(SOIL_2024):
            base = 290.0 + k if k < 4 else 0.20 + 0.01 * k
            handle.write(_grib2(discipline, category, number, first, second, [base + 0.01 * i for i in range(12)]))
    vtable = src.resolve_vtable(_ifs())
    out = tmp_path / "MET:2024-05-22_18"
    completed = subprocess.run([str(writer), "--vtable", str(vtable), "--date", "2024-05-22_18:00:00",
                                "--map-source", "ecmwf", "--out", str(out), str(grib)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads(completed.stdout)
    assert report["grib_messages_matched"] == len(SOIL_2024)
    assert {field["field"] for field in report["fields"]} == {name for name, *_ in SOIL_2024}
