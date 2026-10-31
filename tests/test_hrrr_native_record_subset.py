"""hrrr-native moves only the records its preparation reads.

``woof fetch --source hrrr-native`` (2026-10-29 06Z, f00+f01) moved
2,050 MB: whole wrfnat objects of which the mapping reads 75 percent, a
wrfsfcf00 for one VEG record, and an hrrr-prs soil donor that took a
whole 381 MB wrfprsf00 for 29 MB of soil records plus a second copy of
the same 134 MB wrfsfcf00 the composition never reads.  The route's file
rows now select records by the .idx beside each object, and the donor
row's ``records`` cut the donor to its bound primary object.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from woof import download_budget, fetch, fetch_routes
from woof.record_index import MAX_PIECE, MIN_SPREAD_PIECE, _pieces, _spread, select_ncep

CYCLE = datetime(2026, 10, 29, 6)


def _index(lines):
    """An NCEP inventory whose records are 100 bytes apart, in order."""
    text = "".join(f"{n}:{(n - 1) * 100}:d=2026102906:{var}:{level}:anl:\n"
                   for n, (var, level) in enumerate(lines, 1))
    return text.encode(), len(lines) * 100


def _kept(lines, select):
    index, size = _index(lines)
    ranges, _ = select_ncep(index, select, size)
    return [lines[offset // 100] for offset, _length in ranges]


def _row(role):
    (row,) = [row for row in fetch_routes.route_for("hrrr-native").files if row.role == role]
    return row


def test_wrfnat_keeps_every_mapped_record_and_drops_what_the_mapping_never_reads():
    lines = [(var, f"{level} hybrid level")
             for level in (1, 50)
             for var in ("PRES", "HGT", "TMP", "SPFH", "UGRD", "VGRD", "VVEL", "CLMR",
                         "RWMR", "CIMIXR", "SNMR", "GRLE", "PMTF", "PMTC", "NCONCD",
                         "NCCICE", "SPNCR", "TKE", "MASSDEN", "FRACCC", "TCDC")]
    lines += [("PRES", "surface"), ("TMP", "surface"), ("LAND", "surface"),
              ("ICEC", "surface"), ("WEASD", "surface"), ("SNOD", "surface"),
              ("HPBL", "surface"), ("CAPE", "surface"), ("HGT", "cloud base"),
              ("TMP", "2 m above ground"), ("SPFH", "2 m above ground"),
              ("RH", "2 m above ground"), ("UGRD", "10 m above ground"),
              ("VGRD", "10 m above ground"), ("WIND", "10 m above ground"),
              ("UGRD", "80 m above ground"), ("REFD", "1000 m above ground"),
              ("var discipline=0 center=7 local_table=1 parmcat=7 parm=204", "surface")]
    kept = _kept(lines, _row("wrfnat").record_index["select"])
    dropped = set(lines) - set(kept)
    assert {var for var, _ in dropped if "hybrid" in _} == {"TKE", "MASSDEN", "FRACCC", "TCDC"}
    assert {(var, level) for var, level in dropped if "hybrid" not in level} == {
        ("HPBL", "surface"), ("CAPE", "surface"), ("HGT", "cloud base"),
        ("RH", "2 m above ground"), ("WIND", "10 m above ground"),
        ("UGRD", "80 m above ground"), ("REFD", "1000 m above ground"),
        ("var discipline=0 center=7 local_table=1 parmcat=7 parm=204", "surface")}
    assert kept == [line for line in lines if line not in dropped]   # file order


def test_a_renamed_index_line_moves_its_record_rather_than_dropping_it():
    """The 2021-12-10 18Z f01 index spells cloud water CLWMR; a single-level
    record under a name the selection does not know is kept the same way."""
    lines = [("CLWMR", "1 hybrid level"), ("TMP", "1 hybrid level"),
             ("LAND", "surface"), ("SNOWD", "surface"), ("SPFHX", "2 m above ground"),
             ("HPBL", "surface"), ("TMP", "2 m above ground"), ("UGRD", "10 m above ground")]
    kept = _kept(lines, _row("wrfnat").record_index["select"])
    assert kept == [("CLWMR", "1 hybrid level"), ("TMP", "1 hybrid level"),
                    ("LAND", "surface"), ("SNOWD", "surface"),
                    ("SPFHX", "2 m above ground"), ("TMP", "2 m above ground"),
                    ("UGRD", "10 m above ground")]


def test_the_vegetation_object_moves_its_veg_record_and_never_fails_without_one():
    select = _row("vegetation_surface").record_index["select"]
    lines = [("HGT", "surface"), ("VEG", "surface"), ("LAND", "surface"), ("TMP", "500 mb")]
    assert _kept(lines, select) == [("VEG", "surface"), ("LAND", "surface")]
    # HRRR before 2020-12-02 publishes no VEG: the start reads the GREENFRAC
    # climatology, as from the whole object, so the selection must not fail.
    pre_v4 = [("HGT", "surface"), ("LAND", "surface"), ("TMP", "500 mb")]
    assert _kept(pre_v4, select) == [("LAND", "surface")]


def test_the_soil_donor_moves_only_its_bound_primary_records(tmp_path):
    plan = fetch_routes.resolve_request("hrrr-native", cycle=CYCLE, hours=1)
    (donor,) = plan.donors
    lines = [("HGT", "50 mb"), ("HGT", "surface"), ("TMP", "surface")]
    lines += [(var, f"{depth}-{depth} m below ground")
              for depth in ("0", "0.01", "0.04", "0.1", "0.3", "0.6", "1", "1.6", "3")
              for var in ("TSOIL", "SOILW")]
    kept = _kept(lines, donor.records["select"])
    assert len(kept) == 19 and ("HGT", "50 mb") not in kept and ("TMP", "surface") not in kept

    args = SimpleNamespace(out=tmp_path, transport=None, force_refetch=False,
                           fetch_workers=None)
    donor_plan = fetch._donor_route_plan(donor, args)
    # Only the bound wrfprs object: not the donor route's own wrfsfcf00,
    # which this route already takes for its vegetation.
    assert [(obj.role, obj.lead) for obj in donor_plan.objects] == [("wrfprs", 0)]
    assert donor_plan.objects[0].record_index == donor.records
    assert donor_plan.extra_supplement_files == ()
    assert donor_plan.primary_files == (Path(donor_plan.objects[0].relpath),)


def test_the_donor_files_the_handoff_binds_are_the_cut_plan(tmp_path, monkeypatch):
    plan = fetch_routes.resolve_request("hrrr-native", cycle=CYCLE, hours=1, out=tmp_path)
    moved = []
    monkeypatch.setattr(fetch_routes, "run_plan",
                        lambda donor_plan, **kw: moved.append(donor_plan) or {})
    files = fetch._fetch_route_donors(plan, SimpleNamespace(
        out=tmp_path, transport=None, force_refetch=False, fetch_workers=None))
    (donor_plan,) = moved
    assert [obj.role for obj in donor_plan.objects] == ["wrfprs"]
    assert files["soil_surface_data"] == (tmp_path / "donor-hrrr-prs"
                                          / donor_plan.objects[0].relpath)


def test_every_hrrr_native_object_rides_its_index_and_is_priced_as_its_records():
    plan = fetch_routes.resolve_request("hrrr-native", cycle=CYCLE, hours=1)
    assert plan.objects and all(obj.record_index and obj.index_key.endswith(".grib2.idx")
                                for obj in plan.objects)
    assert fetch_routes.resolve_mode("hrrr-native", "idx-subset") == "idx-subset"
    estimate = download_budget.download_estimate(
        {"source": "hrrr-native", "cycle": "2026-10-29T06", "hours": 1})
    # Two wrfnat leads, one VEG record and the donor's 19 soil records:
    # about 1.2 GB where the whole objects were about 2.2 GB.
    assert 1.1e9 < estimate["bytes"] < 1.3e9, estimate
    assert "record file" in estimate["basis"]


def test_a_donor_route_that_composes_its_primary_cannot_be_cut():
    plan = fetch_routes.resolve_request("hrrr-native", cycle=CYCLE, hours=0)
    from dataclasses import replace
    composed = replace(plan, compose=(object(),))
    try:
        fetch_routes.donor_records_plan(composed, {"format": "ncep-idx", "select": [["X", "y"]]})
    except ValueError as error:
        assert "composes its primary" in str(error)
    else:
        raise AssertionError("a composed donor primary was cut by index")


def test_a_short_selection_is_spread_over_the_streams():
    """The soil donor's 19 records are one 27 MB run: one request on one
    connection took 12-27 s from the home nodes."""
    soil = [(247_000_000, 2_000_000), (300_000_000, 27_300_000)]
    pieces = _spread(_pieces(soil), 8)
    assert len(pieces) == 8
    assert all(length <= MIN_SPREAD_PIECE for _source, length, _dest in pieces)
    # The same bytes in the same places: contiguous in the file and the copy.
    assert sum(length for _s, length, _d in pieces) == 29_300_000
    for (s0, n0, d0), (s1, _n1, d1) in zip(pieces, pieces[1:]):
        assert d1 == d0 + n0 and (s1 == s0 + n0 or s1 == 300_000_000)
    # A selection already as many pieces as streams, one stream, or a small
    # selection stays as it was.
    big = _pieces([(0, 8 * MAX_PIECE)])
    assert _spread(big, 8) == big
    assert _spread(_pieces(soil), 1) == _pieces(soil)
    small = _pieces([(0, 1_500_000), (2_000_000, 50_000)])
    assert _spread(small, 8) == small
