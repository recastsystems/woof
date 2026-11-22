"""The AIFS start with the same cycle's IFS soil below 28 cm (row ``aifs-ifs-soil``).

THE BREAKAGE THIS PREVENTS.  AIFS open data publishes two soil layers (0-7 and 7-28 cm).  NOAA's RRFS start-state
program (case 7, hrrrv5 and the RRFS suites) fills RUC's nine nodes from the first guess only when it carries RUC's
nine nodes or four layers; with two it takes neither branch and every land cell starts at 0 K through all nine
levels, which the producer route refuses after case 7 (ed290e74).  The row adds IFS's layers 3 and 4 (28-100 and
100-289 cm) of the same cycle and lead, on the identical grid, so the first guess is the IFS row's four-layer column.
"""

from __future__ import annotations

import json

import pytest

from woof.hex import init_sources as src

TABLE = src.load_table()
ROW = src.resolve_source(TABLE, "aifs-ifs-soil")


def _index(records):
    return "\n".join(json.dumps(r) for r in records)


def test_the_row_validates_and_declares_four_soil_layers():
    src.validate_row(ROW)
    assert ROW["first_guess"]["nfgsoillevels"] == 4
    assert src.resolve_vtable(ROW).is_file()
    # the plain aifs row is unchanged: it keeps its two layers and its own posting time
    aifs = src.resolve_source(TABLE, "aifs")
    assert aifs["first_guess"]["nfgsoillevels"] == 2 and aifs["posting"]["lag_minutes"] == 326
    assert not any(p["role"] == "deep_soil_donor" for p in aifs["products"])


@pytest.mark.parametrize("lead", [0, 6, 144])
def test_every_lead_reads_aifs_and_only_ifs_soil_layers_3_and_4_of_the_same_cycle_and_lead(lead):
    cycle = src.Cycle.parse("2026-11-10T06")
    plans = src.plan_products(ROW, cycle, None, lead)
    donors = [p for p in plans if p.role == "deep_soil_donor"]
    primaries = [p for p in plans if p.role == "primary"]
    assert len(donors) == 1 and len(primaries) == 1
    donor = donors[0]
    assert f"/20261110/06z/ifs/0p25/oper/20261110060000-{lead}h-oper-fc.grib2" in donor.url
    assert f"/aifs-single/0p25/oper/20261110060000-{lead}h-oper-fc.grib2" in primaries[0].url
    ifs_index = _index([
        {"levtype": "sol", "param": "sot", "levelist": str(n), "_offset": 10 * n, "_length": 5} for n in (1, 2, 3, 4)
    ] + [
        {"levtype": "sol", "param": "vsw", "levelist": str(n), "_offset": 100 + 10 * n, "_length": 5} for n in (1, 2, 3, 4)
    ] + [{"levtype": "sfc", "param": "2t", "_offset": 500, "_length": 5}])
    kept = src.select_index_records(ifs_index, donor.index_select)
    # never IFS's layers 1-2 (AIFS's own are the start's), never an atmosphere or surface record
    assert sorted((r["param"], r["levelist"]) for r in kept) == [("sot", "3"), ("sot", "4"), ("vsw", "3"), ("vsw", "4")]


def test_the_ladder_stops_where_ifs_stops():
    # IFS's 06 and 18 UTC runs end at 144 h; AIFS goes on to 360 h on every cycle. A lead without IFS soil is refused
    # before anything is fetched, not after the AIFS bytes were paid for.
    src.plan_products(ROW, src.Cycle.parse("2026-11-10T12"), None, 360)
    with pytest.raises(src.InitSourceRefusal, match="outside the row's forecast cadence"):
        src.plan_products(ROW, src.Cycle.parse("2026-11-10T06"), None, 150)
    with pytest.raises(src.InitSourceRefusal, match="outside the row's forecast cadence"):
        src.plan_products(ROW, src.Cycle.parse("2026-11-10T12"), None, 3)


def test_the_donor_records_decode_through_the_rows_own_vtable_as_layers_3_and_4():
    # The ECMWF Vtable maps level pairs (2,3) and (3,4) on surface 151 to ST028100/ST100289 and SM028100/SM100289,
    # the IFS row's deep layers, so the donor needs no Vtable of its own.
    text = src.resolve_vtable(ROW).read_text()
    for name in ("ST028100", "ST100289", "SM028100", "SM100289"):
        assert name in text
    donor = next(p for p in ROW["products"] if p["role"] == "deep_soil_donor")
    assert set(donor["fields"]) == {"ST028100", "ST100289", "SM028100", "SM100289"}


def test_the_row_waits_for_the_later_of_the_two_postings():
    assert ROW["posting"]["lag_minutes"] == src.resolve_source(TABLE, "ifs")["posting"]["lag_minutes"]
    assert ROW["attribution"] == src.resolve_source(TABLE, "aifs")["attribution"]
