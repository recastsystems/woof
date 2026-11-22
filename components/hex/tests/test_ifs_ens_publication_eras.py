"""ECMWF ENS open data as it was published: one ifs-ens member from a past cycle.

THE BREAKAGE THESE TESTS PREVENT (woof hex 0.5.0, 2026-11-13): `init-source --source ifs-ens --member 7`
from the 2024-05-14 12Z cycle refused at f000, "no address has it ... the index .../oper/20240514120000-0h-oper-fc.index
has no record matching {'param': 'z', 'levtype': 'sfc'}", and from the 2024-05-22 18Z cycle at f006 with HTTP 404 on
both mirrors. The row's terrain donor asked only the cycle's own oper step-0 file for its surface geopotential z, which
ECMWF published only from the 2024-12-02 18Z run on, and only under /oper/ (the 06/18 UTC runs before 2026-05-12 06Z
are under /scda/). The member's records also named only today's soil (sot/vsw); the 2024 ENS published st, stl2-4 and
swvl1-4 among its surface fields, so an old member started with no soil. The ifs row was fixed the same way
(test_ifs_publication_eras.py); these are the ENS row's two donor products and its member select.
"""
from __future__ import annotations

import json

import pytest

from woof.hex import init_sources as src
from woof.hex.init_sources import Cycle, InitSourceRefusal, load_table, plan_products, resolve_source

OLD_SOIL = {"st", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4"}
REFERENCE = "20241203/00z/ifs/0p25/oper/20241203000000-0h-oper-fc"


def _ens():
    return resolve_source(load_table(), "ifs-ens")


def _plans(cycle, lead, member=7):
    return plan_products(_ens(), Cycle.parse(cycle), member, lead)


def _role(plans, role):
    found = [plan for plan in plans if plan.role == role]
    assert len(found) == 1, [plan.role for plan in plans]
    return found[0]


@pytest.mark.parametrize("cycle,lead", [("2024-05-14T12", 0), ("2024-05-14T12", 198), ("2024-05-22T18", 6),
                                        ("2026-10-23T12", 0)])
def test_the_donor_tries_oper_then_scda_then_the_reference_orography(cycle, lead):
    donor = _role(_plans(cycle, lead), "terrain_donor")
    assert donor.index_select == {"param": "z", "levtype": "sfc"} and donor.fields == ["SOILGEO"]
    addresses = [donor.url, *donor.url_mirrors]
    indexes = [donor.index_url, *donor.index_mirrors]
    assert len(addresses) == len(indexes) == 6
    day, hour = cycle[:10].replace("-", ""), cycle[11:13]
    assert addresses[0] == (f"https://storage.googleapis.com/ecmwf-open-data/{day}/{hour}z/ifs/0p25/oper/"
                            f"{day}{hour}0000-0h-oper-fc.grib2")
    assert all(f"/{day}/{hour}z/ifs/0p25/scda/{day}{hour}0000-0h-scda-fc." in u for u in addresses[2:4] + indexes[2:4])
    assert all(REFERENCE in url for url in addresses[-2:] + indexes[-2:])
    assert [u.rsplit(".", 1)[0] for u in addresses] == [u.rsplit(".", 1)[0] for u in indexes]


def test_the_member_selects_both_soil_encodings():
    for lead in (0, 198):
        primary = _role(_plans("2024-05-14T12", lead), "primary")
        params = set(primary.index_select["param"])
        assert OLD_SOIL <= params and {"sot", "vsw"} <= params and "z" not in params
        assert primary.index_select["type"] == "pf" and primary.index_select["number"] == "7"


def test_a_2024_member_selects_its_own_records_and_soil():
    """The 2024 ENS index lists each perturbed member's soil as st/stl2-4/swvl1-4 on levtype sfc and also carries a
    control (type cf): the select keeps member 7's records only, soil included."""
    primary = _role(_plans("2024-05-14T12", 0), "primary")
    lines = [{"type": t, "number": n, "levtype": "sfc", "param": p, "_offset": k, "_length": 1}
             for k, (t, n, p) in enumerate([("pf", "7", "st"), ("pf", "7", "swvl4"), ("pf", "8", "st"),
                                            ("cf", None, "st"), ("pf", "7", "2t"), ("pf", "7", "tp")])]
    text = "\n".join(json.dumps({k: v for k, v in r.items() if v is not None}) for r in lines)
    kept = src.select_index_records(text, primary.index_select)
    assert [(r["number"], r["param"]) for r in kept] == [("7", "st"), ("7", "swvl4"), ("7", "2t")]


@pytest.mark.parametrize("cycle,lead,own", [("2024-05-14T12", 198, "/20240514/12z/ifs/0p25/oper/"),
                                            ("2024-05-22T18", 6, "/20240522/18z/ifs/0p25/scda/")])
def test_a_cycle_without_z_takes_the_reference_orography(monkeypatch, tmp_path, cycle, lead, own):
    """The cycle's own step-0 index answers (oper at 12Z, scda at 18Z; the other path 404s) but holds no z: the donor
    reads the 2024-12-03 00Z z."""
    donor = _role(_plans(cycle, lead), "terrain_donor")
    held = "\n".join(json.dumps({"levtype": "sfc", "param": p, "_offset": 100 * k, "_length": 100})
                     for k, p in enumerate(["2t", "sp", "st", "swvl1"]))
    reference = json.dumps({"levtype": "sfc", "param": "z", "_offset": 0, "_length": 8})
    asked = []

    def fake_get(url, *, byte_range=None, attempts=10):
        asked.append(url)
        if url.endswith(".index") and own in url:
            return held.encode()
        if url.endswith(".index") and REFERENCE in url:
            return reference.encode()
        if REFERENCE in url and byte_range is not None:
            return b"GRIB\x00\x00\x00\x02"
        raise InitSourceRefusal(f"{url} answered HTTP 404")

    monkeypatch.setattr(src, "_http_get", fake_get)
    receipt = src._fetch_selected(donor, tmp_path / donor.filename, 1)
    assert REFERENCE in receipt["url"] and receipt["records"] == 1
    assert (tmp_path / donor.filename).read_bytes().startswith(b"GRIB")
    assert any("/scda/" in url for url in asked) and any("/oper/" in url and REFERENCE not in url for url in asked)


def test_a_cycle_with_z_reads_its_own_step_zero(monkeypatch, tmp_path):
    donor = _role(_plans("2026-10-23T12", 198), "terrain_donor")
    own = json.dumps({"levtype": "sfc", "param": "z", "_offset": 0, "_length": 8})
    asked = []

    def fake_get(url, *, byte_range=None, attempts=10):
        asked.append(url)
        if "/20261023/12z/ifs/0p25/oper/" in url:
            return own.encode() if url.endswith(".index") else b"GRIB\x00\x00\x00\x02"
        raise AssertionError(f"asked {url}")

    monkeypatch.setattr(src, "_http_get", fake_get)
    receipt = src._fetch_selected(donor, tmp_path / donor.filename, 1)
    assert "/20261023/12z/ifs/0p25/oper/20261023120000-0h-oper-fc.grib2" in receipt["url"]
    assert all(REFERENCE not in url and "/scda/" not in url for url in asked)
