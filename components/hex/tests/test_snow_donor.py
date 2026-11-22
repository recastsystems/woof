"""Snow at the start of a hex run from ECMWF IFS before 2025-11-20 12Z, and from AIFS at any cycle.

THE BREAKAGE THESE TESTS PREVENT (2026-11-13 walk of the hex region orders the site accepts): ECMWF open data published no
snow (sd) before the 2025-11-20 12Z run (IFS HRES and ENS indexes: none on 2025-11-19 12Z or 2025-11-20 06Z, there from
2025-11-20 12Z), and AIFS single and ENS publish none at all, so every such start began with no snow on the ground
anywhere. In the winter case of 2025-01-22 00Z (the evening after the Gulf Coast snowstorm) the snowless start was the
start every IFS region got. The ifs, ifs-ens, aifs and aifs-ens rows now take the same cycle's GFS 0.25-degree WEASD at
the same lead as a ``snow_donor`` product, decoded by the row's own Vtable to SNOW (kg m-2, not doubled); the IFS rows
stop taking it when their own sd begins (the product's ``until``).
"""
from __future__ import annotations

import json
import struct
import subprocess

import pytest

from woof.hex import init_sources as src
from woof.hex.init_sources import Cycle, InitSourceRefusal, load_table, plan_products, resolve_source, validate_row

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{day}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"


def _row(name):
    return resolve_source(load_table(), name)


def _donors(name, cycle, lead, member=None):
    return [p for p in plan_products(_row(name), Cycle.parse(cycle), member, lead) if p.role == "snow_donor"]


@pytest.mark.parametrize("name,member", [("ifs", None), ("ifs-ens", 7)])
@pytest.mark.parametrize("cycle,lead", [("2025-01-22T00", 0), ("2025-01-22T00", 12), ("2024-12-15T12", 198),
                                        ("2025-11-20T06", 6)])
def test_an_ifs_start_before_its_snow_was_published_takes_gfs_snow_at_the_same_lead(name, member, cycle, lead):
    (donor,) = _donors(name, cycle, lead, member)
    when = Cycle.parse(cycle).when
    assert donor.url == GFS.format(day=f"{when:%Y%m%d}", hh=f"{when:%H}", lead=lead)
    assert donor.index_url == donor.url + ".idx"
    assert donor.index_select == {"records": [["^WEASD$", "^surface$"]]}
    assert donor.filename.endswith(f".f{lead:03d}.weasd")


@pytest.mark.parametrize("name,member", [("ifs", None), ("ifs-ens", 7)])
@pytest.mark.parametrize("cycle,lead", [("2025-11-20T12", 0), ("2026-01-15T00", 198)])
def test_an_ifs_start_from_2025_11_20_12z_reads_its_own_snow(name, member, cycle, lead):
    assert _donors(name, cycle, lead, member) == []
    primary = [p for p in plan_products(_row(name), Cycle.parse(cycle), member, lead) if p.role == "primary"][0]
    assert "sd" in json.dumps(primary.index_select)


@pytest.mark.parametrize("name,member", [("aifs", None), ("aifs-ens", 3), ("aifs-ifs-soil", None)])
@pytest.mark.parametrize("cycle,lead", [("2025-03-05T00", 0), ("2026-11-12T00", 24)])
def test_every_aifs_start_takes_gfs_snow(name, member, cycle, lead):
    (donor,) = _donors(name, cycle, lead, member)
    assert donor.filename.endswith(f".f{lead:03d}.weasd")


def test_no_aifs_row_is_left_without_snow():
    # THE BREAKAGE (0.6.0 front-door run, 2026-11-15): the aifs-ifs-soil row (AIFS with IFS deep soil, the row an hrrrv5
    # start from AIFS uses) came from another branch than the snow donor and had none, so its start over the Canadian
    # prairies on 2026-11-15 12Z had SNOW = 0 in all 5,339 cells. Every row of an AIFS primary carries the donor, at
    # step 0 and at a lead, and says so in its limits.
    aifs_rows = [row for row in load_table()["sources"] if str(row["id"]).startswith("aifs")]
    assert {row["id"] for row in aifs_rows} >= {"aifs", "aifs-ens", "aifs-ifs-soil"}
    for row in aifs_rows:
        assert "snow_donor" in [p["role"] for p in row["products"]], row["id"]
        assert "snow_donor" in [p["role"] for p in row["forecast"]["products"]], row["id"]
        assert any("snow_donor" in limit for limit in row["limits"]), row["id"]


def test_the_snow_donor_never_enters_the_surface_input():
    for name in ("ifs", "ifs-ens", "aifs", "aifs-ens"):
        row = _row(name)
        assert row["surface_input"]["product_role"] != "snow_donor"
        for products in (row["products"], row["forecast"]["products"]):
            (donor,) = [p for p in products if p["role"] == "snow_donor"]
            assert ("until" in donor) == name.startswith("ifs")


def test_a_product_until_must_name_a_cycle():
    row = json.loads(json.dumps(_row("ifs")))
    for product in row["products"]:
        if product["role"] == "snow_donor":
            product["until"] = "2025-11-20"
    with pytest.raises(InitSourceRefusal, match="YYYY-MM-DDTHH"):
        validate_row(row)


# ---------------------------------------------------------------------------
# GFS WEASD through the rows' own Vtables and the real met_intermediate
# ---------------------------------------------------------------------------
def _signed(value: int) -> int:
    return (abs(value) | 0x80000000) if value < 0 else value


def _weasd(values, nx=4, ny=3):
    """One GFS-style GRIB2 WEASD message (0-1-13 on the surface, lat-lon 3.0, product 4.0, simple packing 5.0)."""
    sec1 = struct.pack(">IBHHBBBHBBBBBBB", 21, 1, 7, 0, 2, 1, 1, 2025, 1, 22, 0, 0, 0, 0, 1)
    grid = struct.pack(">BBIBIBIIIIIIIBIIII", 6, 0, 0, 0, 0, 0, 0, nx, ny, 0, 0xFFFFFFFF,
                       _signed(31_000_000), 270_000_000, 0x30, _signed(30_500_000),
                       270_750_000, 250_000, 250_000)
    grid += b"\x00"
    sec3 = struct.pack(">IBBIBBH", 14 + len(grid), 3, 0, nx * ny, 0, 0, 0) + grid
    tmpl = struct.pack(">BBBBBHBBI", 1, 13, 2, 0, 96, 0, 0, 1, 0)
    tmpl += struct.pack(">BBI", 1, 0, 0) + struct.pack(">BBI", 255, 255, 0xFFFFFFFF)
    sec4 = struct.pack(">IBHH", 9 + len(tmpl), 4, 0, 0) + tmpl
    scaled = [round(v * 100) for v in values]
    reference = min(scaled)
    packed = b"".join(struct.pack(">H", s - reference) for s in scaled)
    sec5 = struct.pack(">IBIH", 21, 5, nx * ny, 0) + struct.pack(">fhhBB", float(reference), 0, 2, 16, 0)
    sec6 = struct.pack(">IBB", 6, 6, 255)
    sec7 = struct.pack(">IB", 5 + len(packed), 7) + packed
    body = sec1 + sec3 + sec4 + sec5 + sec6 + sec7 + b"7777"
    return b"GRIB" + struct.pack(">HBBQ", 0, 0, 2, 16 + len(body)) + body


def _snow_values(path):
    data = path.read_bytes()
    pos = 0
    while pos < len(data):
        records = []
        for _ in range(5):
            (n,) = struct.unpack(">i", data[pos:pos + 4])
            records.append(data[pos + 4:pos + 4 + n])
            pos += 8 + n
        header = records[1]
        field = header[60:69].decode().strip()
        if field == "SNOW":
            return list(struct.unpack(f">{len(records[4]) // 4}f", records[4]))
    return None


@pytest.mark.parametrize("name", ["ifs", "aifs-ens"])
def test_gfs_weasd_decodes_to_snow_in_kg_per_square_metre(tmp_path, name):
    try:
        writer = src.resolve_writer(None)
    except Exception as error:  # an install without woof's bridges cannot run the writer
        pytest.skip(f"met_intermediate is not installed: {error}")
    values = [0.0, 0.0, 3.5, 14.9, 19.7, 0.0, 13.1, 6.4, 0.0, 2.25, 40.0, 0.0]
    grib = tmp_path / "gfs.t00z.pgrb2.0p25.f000.weasd"
    grib.write_bytes(_weasd(values))
    out = tmp_path / "MET:2025-01-22_00"
    completed = subprocess.run([str(writer), "--vtable", str(src.resolve_vtable(_row(name))), "--date",
                                "2025-01-22_00:00:00", "--map-source", "ecmwf", "--out", str(out), str(grib)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads(completed.stdout)
    assert report["grib_messages_matched"] == 1
    assert "SNOW" in {field["field"] for field in report["fields"]}
    assert _snow_values(out) == pytest.approx(values, abs=0.01)
