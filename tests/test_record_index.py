"""Record selection by a publisher's index (fix/ecmwf-record-subset).

The defect: IFS open data moved whole objects (the 0.1-degree pair is
451 MB per lead, of which the mapping binds 165 MB) and each ECMWF-family
start moved the whole 513 MB GFS f000 for its one ICEC record.  These tests
pin the selection rows and the range transfer against a fake host.
"""

from __future__ import annotations

from datetime import datetime
import io
import json
from urllib.error import HTTPError

import pytest

from woof import fetch_routes, record_index


def _grib(body: bytes) -> bytes:
    length = 16 + len(body) + 4
    return b"GRIB" + b"\0\0\0\x02" + length.to_bytes(8, "big") + body + b"7777"


def _object(records):
    """A GRIB object, its ECMWF JSON-lines index and its NCEP .idx."""
    data = b""
    jsonl = []
    idx = []
    for number, (levtype, param, level) in enumerate(records, 1):
        message = _grib(f"{levtype}:{param}:{level}".encode() * 8)
        row = {"levtype": levtype, "param": param, "_offset": len(data),
               "_length": len(message)}
        if level:
            row["levelist"] = level
        jsonl.append(json.dumps(row))
        idx.append(f"{number}:{len(data)}:d=2026102812:{param.upper()}:{levtype}:anl:")
        data += message
    return data, "\n".join(jsonl).encode(), "\n".join(idx).encode()


class _Reply(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


def _host(data, index, *, etag='"v1"', index_status=200):
    asked = []

    def opener(request, timeout=None):
        url = request.full_url
        asked.append((request.get_method(), url, request.headers.get("Range")))
        if url.endswith((".index", ".idx")):
            if index_status != 200:
                raise HTTPError(url, index_status, "no", {}, io.BytesIO(b""))
            return _Reply(index, 200, {})
        if request.get_method() == "HEAD":
            return _Reply(b"", 200, {"Content-Length": str(len(data))})
        first, last = (int(v) for v in request.headers["Range"].split("=")[1].split("-"))
        return _Reply(data[first:last + 1], 206, {
            "Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": etag})

    return opener, asked


IFS_SELECT = {"format": "ecmwf-jsonl", "why": "test", "select": [
    {"levtype": "pl", "param": ["t", "u"]}, {"levtype": "sol", "param": ["sot"]}]}


def test_the_ecmwf_selection_moves_only_the_mapped_records(tmp_path):
    records = [("pl", "z", "500"), ("pl", "t", "500"), ("pl", "u", "500"),
               ("pl", "w", "500"), ("sfc", "msl", ""), ("sol", "sot", "1")]
    data, index, _ = _object(records)
    opener, asked = _host(data, index)
    dest = tmp_path / "pair.grib2"
    entry = record_index.download_records(
        "https://h/x.grib2", "https://h/x.index", dest,
        declaration=record_index.validate_declaration(IFS_SELECT, "t"), opener=opener)
    rows = [json.loads(line) for line in index.decode().splitlines()]
    want = b"".join(data[r["_offset"]:r["_offset"] + r["_length"]] for r in rows
                    if r["param"] in ("t", "u", "sot"))
    assert dest.read_bytes() == want
    assert entry["selected_records"] == 3 and entry["bytes"] == len(want)
    # t and u are adjacent: one range for them, one for sot.
    assert entry["requests"] == 2
    assert record_index.grib_messages(dest) == 3


def test_a_changed_object_between_ranges_is_refused(tmp_path):
    records = [("pl", "t", "500"), ("pl", "z", "500"), ("sol", "sot", "1")]
    data, index, _ = _object(records)
    opener, _ = _host(data, index)
    tags = iter(['"v1"', '"v2"'])

    def changing(request, timeout=None):
        reply = opener(request, timeout)
        if "Range" in request.headers:
            reply.headers = dict(reply.headers, ETag=next(tags))
        return reply

    with pytest.raises(ValueError, match="changed between range"):
        record_index.download_records(
            "https://h/x.grib2", "https://h/x.index", tmp_path / "o.grib2",
            declaration=record_index.validate_declaration(IFS_SELECT, "t"),
            opener=changing, streams=1)
    assert not (tmp_path / "o.grib2").exists()


def test_the_ncep_selection_takes_one_record_and_the_last_by_object_size(tmp_path):
    records = [("surface", "land", ""), ("surface", "icec", ""), ("surface", "icetmp", "")]
    data, _, idx = _object(records)
    opener, asked = _host(data, idx)
    for wanted, heads in (("ICEC", 0), ("ICETMP", 1)):
        dest = tmp_path / f"{wanted}.grib2"
        entry = record_index.download_records(
            "https://h/g.f000", "https://h/g.f000.idx", dest, opener=opener,
            declaration=record_index.validate_declaration(
                {"format": "ncep-idx", "select": [[wanted, "surface"]]}, "t"))
        assert entry["selected_records"] == 1
        assert wanted.lower().encode() in dest.read_bytes()
        assert sum(method == "HEAD" for method, *_ in asked) == heads
    with pytest.raises(ValueError, match="no record matching"):
        record_index.download_records(
            "https://h/g.f000", "https://h/g.f000.idx", tmp_path / "n.grib2", opener=opener,
            declaration=record_index.validate_declaration(
                {"format": "ncep-idx", "select": [["SNOD", "surface"]]}, "t"))


def test_an_absent_index_is_named_so_the_object_moves_whole(tmp_path):
    data, index, _ = _object([("pl", "t", "500")])
    opener, _ = _host(data, index, index_status=404)
    with pytest.raises(record_index.IndexAbsent):
        record_index.download_records(
            "https://h/x.grib2", "https://h/x.index", tmp_path / "o.grib2",
            declaration=record_index.validate_declaration(IFS_SELECT, "t"), opener=opener)


@pytest.mark.parametrize("bad", [
    {"format": "wgrib", "select": [{"param": "t"}]},
    {"format": "ecmwf-jsonl", "select": []},
    {"format": "ecmwf-jsonl", "select": [{"param": []}]},
    {"format": "ncep-idx", "select": [["ICEC"]]},
    {"format": "ecmwf-jsonl", "select": [{"param": "t"}], "level": 1},
])
def test_a_malformed_selection_is_refused_at_load(bad):
    with pytest.raises(ValueError):
        record_index.validate_declaration(bad, "t")


def test_the_ifs_routes_select_records_and_take_icec_alone():
    cycle = datetime(2026, 10, 28, 12)
    for source in ("ecmwf-open-data", "ecmwf-open-data-0p1"):
        plan = fetch_routes.resolve_request(source, cycle=cycle, hours=6)
        assert plan.objects and all(obj.record_index for obj in plan.objects), source
        assert all(obj.index_key.endswith(".index") and not obj.index_key.endswith(
            ".grib2.index") for obj in plan.objects)
        assert fetch_routes.resolve_mode(source, "idx-subset") == "idx-subset"
    for source in ("ecmwf-open-data", "ecmwf-open-data-0p1", "ecmwf-ens", "aifs"):
        (donor,) = fetch_routes.route_for(source).donors
        assert donor.records["select"] == [["ICEC", "surface"]], source
    # Every parameter the IFS mapping binds is in the selection (current and
    # 2024 names): 2t/2d/10u/10v, sp, skt, lsm, surface z, soil, and the
    # pressure-level height, temperature, humidity and wind.
    (row,) = fetch_routes.route_for("ecmwf-open-data").files
    clauses = {clause["levtype"]: set(clause["param"]) for clause in row.record_index["select"]}
    assert {"gh", "q", "t", "u", "v"} == clauses["pl"]
    assert {"10u", "10v", "2d", "2t", "lsm", "skt", "sp", "z", "st", "swvl1"} <= clauses["sfc"]
    assert {"sot", "vsw"} == clauses["sol"]


def test_aifs_takes_its_mapped_records_and_the_height_the_mapping_reads():
    """fix/aifs-record-subset: AIFS moved its oper-fc objects whole (61.9 of
    which the mapping reads 34.3 MB at the 2026-10-29 12Z f000).  Its cycles
    publish the pressure-level height as gh and z (from about 2026-05) or as
    z alone (2025-07 to 2026-04); the mapping reads gh first and z only where
    gh is absent, so the selection keeps exactly that record per level."""
    cycle = datetime(2026, 10, 29, 12)
    plan = fetch_routes.resolve_request("aifs", cycle=cycle, hours=6)
    assert plan.objects and all(obj.record_index and obj.index_key.endswith("-oper-fc.index")
                                for obj in plan.objects)
    assert fetch_routes.resolve_mode("aifs", "idx-subset") == "idx-subset"
    (row,) = fetch_routes.route_for("aifs").files
    select, prefer = row.record_index["select"], row.record_index["prefer"]
    both = [("pl", "z", "500"), ("pl", "gh", "500"), ("pl", "t", "500"), ("pl", "w", "500"),
            ("pl", "z", "850"), ("pl", "gh", "850"), ("sfc", "z", ""), ("sfc", "lsm", ""),
            ("sfc", "msl", ""), ("sol", "sot", "1")]
    data, index, _ = _object(both)
    rows = [json.loads(line) for line in index.splitlines()]
    kept = {(rows[n]["levtype"], rows[n]["param"], rows[n].get("levelist", ""))
            for n, span in enumerate((row["_offset"], row["_length"]) for row in rows)
            if span in record_index.select_ecmwf(index, select, prefer)}
    assert kept == {("pl", "gh", "500"), ("pl", "t", "500"), ("pl", "gh", "850"),
                    ("sfc", "z", ""), ("sfc", "lsm", ""), ("sol", "sot", "1")}
    # A cycle with no gh: the geopotential is the height the decode reads.
    _, z_only, _ = _object([("pl", "z", "500"), ("pl", "t", "500"), ("sfc", "z", "")])
    assert len(record_index.select_ecmwf(z_only, select, prefer)) == 3
    # Without the order both heights would move.
    assert len(record_index.select_ecmwf(index, select)) == 8


@pytest.mark.parametrize("prefer", [{"param": ["gh"]}, {"param": ["gh", "gh"]},
                                    {"param": ["gh", "z"], "levtype": ["pl", "sfc"]}, ["gh"]])
def test_a_malformed_prefer_order_is_refused_at_load(prefer):
    with pytest.raises(ValueError):
        record_index.validate_declaration(
            {"format": "ecmwf-jsonl", "select": [{"param": "gh"}], "prefer": prefer}, "t")
    with pytest.raises(ValueError):
        record_index.validate_declaration(
            {"format": "ncep-idx", "select": [["HGT", "500 mb"]],
             "prefer": {"param": ["gh", "z"]}}, "t")


def test_a_substituted_transport_still_moves_whole_objects(tmp_path):
    """An offline replay or standby store keeps its own whole-object transport."""
    plan = fetch_routes.resolve_request("ecmwf-open-data-0p1", cycle=datetime(2026, 10, 28, 12),
                                        hours=0, host="google")
    seen = []

    def download(url, dest, **kwargs):
        seen.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_grib(b"x" * 40))
        return {"name": dest.name, "bytes": dest.stat().st_size, "sha256": "x", "url": url}

    fetch_routes.run_plan(plan, out=tmp_path, progress=lambda *_: None,
                          downloader=download, probe=lambda *_: False)
    assert len(seen) == 2 and all(url.endswith(".grib2") for url in seen)


# --------------------------------------------------------------------------
# GEFS member pair (fix/gefs-record-subset)
# --------------------------------------------------------------------------
#
# The defect: a GEFS member moved both halves whole, 105-118 MB per lead,
# of which the packaged mapping binds 31-33 MB (pgrb2b, 92-102 MB, is read
# for 120 of its 473-505 records).  The route table now selects the bound
# records by the .idx beside each half.

#: wgrib2's names for the GRIB2 codes the GEFS mapping binds.
_NCEP_NAMES = {(0, 0, 0): "TMP", (0, 1, 0): "SPFH", (0, 2, 2): "UGRD", (0, 2, 3): "VGRD",
               (0, 3, 0): "PRES", (0, 3, 5): "HGT", (2, 0, 0): "LAND", (2, 0, 2): "TSOIL",
               (2, 0, 192): "SOILW", (10, 2, 0): "ICEC"}


def _gefs_bound_records():
    """(VAR, LEVEL) of every record the packaged GEFS mapping binds."""
    from importlib import resources

    mapping = json.loads(resources.files("woof.authorities").joinpath(
        "rw-wps-gefs-ensemble-grib2.mapping.json").read_text(encoding="utf-8"))
    levels = mapping["coordinates"]["vertical"]["levels"]
    bound = set()
    for field in mapping["fields"].values():
        for selector in field.get("selectors", ()):
            name = _NCEP_NAMES[(selector["discipline"], selector["category"],
                                selector["parameter"])]
            kind = selector["level_type"]
            if kind == 100:
                bound |= {(name, f"{pa / 100:g} mb") for pa in levels}
            elif kind == 1:
                bound.add((name, "surface"))
            elif kind == 103:
                bound.add((name, f"{selector['level_value']:g} m above ground"))
            elif kind == 106:
                bound.add((name, f"{selector['level_value']:g}-"
                                 f"{selector['second_level_value']:g} m below ground"))
            else:
                raise AssertionError(f"level type {kind} has no .idx name in this test")
    return bound


def test_the_gefs_halves_select_exactly_the_records_the_mapping_binds():
    bound = _gefs_bound_records()
    # 5 fields on 31 pressure levels, 5 surface, 2 at 2 m, 2 at 10 m, 2 x 4 soil.
    assert len(bound) == 155 + 5 + 2 + 2 + 8
    # Records each half also carries that the decode never reads.
    decoys = {("RH", "500 mb"), ("VVEL", "500 mb"), ("O3MR", "10 mb"), ("TMP", "0.4 mb"),
              ("TMP", "80 m above ground"), ("SPFH", "30-0 mb above ground"),
              ("PRES", "mean sea level"), ("PRMSL", "mean sea level"), ("SOILL", "0-0.1 m below ground"),
              ("HGT", "tropopause"), ("TMAX", "2 m above ground"), ("RH", "2 m above ground"),
              ("CAPE", "180-0 mb above ground"), ("UGRD", "planetary boundary layer")}
    rows = sorted(bound | decoys)
    idx = "\n".join(f"{n}:{100 * n}:d=2026102800:{var}:{level}:anl:ENS=+1"
                    for n, (var, level) in enumerate(rows, 1)).encode()
    route = fetch_routes.route_for("gefs")
    assert route.record_subset_supported
    assert [row.role for row in route.files] == ["pgrb2a", "pgrb2b"]
    # One selection for both halves: a record NCEP moves from one half to
    # the other (the surface height does, between step 0 and step 3) is
    # still taken.
    (select,) = {json.dumps(row.record_index["select"]) for row in route.files}
    ranges, _ = record_index.select_ncep(idx, json.loads(select), 100 * (len(rows) + 1))
    picked = {rows[offset // 100 - 1] for offset, _length in ranges}
    assert picked == bound


def test_a_gefs_member_plans_both_halves_by_their_idx():
    plan = fetch_routes.resolve_request("gefs", cycle=datetime(2026, 10, 28, 0), hours=6,
                                        member="p01")
    assert len(plan.objects) == 6
    assert all(obj.record_index and obj.index_key.endswith(".idx") for obj in plan.objects)
    assert fetch_routes.resolve_mode("gefs", "idx-subset") == "idx-subset"


@pytest.mark.parametrize("host,ranged", [
    ("https://noaa-gefs-pds.s3.amazonaws.com", True),
    ("https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod", False),
])
def test_the_paced_nomads_host_moves_a_selected_object_whole(tmp_path, monkeypatch, host,
                                                              ranged):
    """NOMADS spaces each request 2.5 s apart: 85 ranges per member-lead lose to the whole pair."""
    from woof.fetch_endpoints import Endpoint

    plan = fetch_routes.resolve_request("gefs", cycle=datetime(2026, 10, 28, 0), hours=0)
    obj = plan.objects[1]
    calls = []
    monkeypatch.setattr(record_index, "download_records",
                        lambda url, index_url, dest, **kw: calls.append(("ranges", url))
                        or {"name": dest.name})
    monkeypatch.setattr(fetch_routes, "_PACED_WHOLE_SAID", set())
    said = []

    def whole(url, dest, **kw):
        calls.append(("whole", url))
        return {"name": dest.name}

    endpoint = Endpoint(name="h", base=host, retention_hours=None, why="test")
    fetch_routes._download_along_ladder(
        plan, obj, tmp_path / obj.relpath, magic="GRIB", fetch=whole, opener=None,
        progress=said.append, ladder=(endpoint,))
    assert [kind for kind, _url in calls] == (["ranges"] if ranged else ["whole"])
    assert calls[0][1].startswith(host) and calls[0][1].endswith("pgrb2b.0p50.f000")
    assert any("paces each request" in line for line in said) is not ranged


def test_a_record_selection_does_not_move_the_reviewed_source_contract(monkeypatch):
    """A background reviewed before a route took its records by .idx is still the same contract."""
    import dataclasses

    from woof import background_contract

    selected = background_contract.capability_fingerprint("gefs")
    route = fetch_routes.route_for("gefs")
    whole = dataclasses.replace(route, files=tuple(
        dataclasses.replace(row, record_index=None) for row in route.files))
    real = fetch_routes.route_for
    monkeypatch.setattr(fetch_routes, "route_for",
                        lambda source, *a, **k: whole if source == "gefs" else real(source, *a, **k))
    assert background_contract.capability_fingerprint("gefs") == selected
def _ncep_idx(lines):
    """An NCEP .idx whose records are 100 bytes apart, from ``VAR:LEVEL`` pairs."""
    return "\n".join(f"{n}:{(n - 1) * 100}:d=2026102912:{line}:anl:"
                     for n, line in enumerate(lines, 1)).encode()


def test_the_rrfs_rows_take_every_soil_node_by_idx_line():
    """fix/rrfs-record-subset: RRFS prslev/2dfld moved whole (1,788 MB for
    f000+f003, of which the mapping reads 653 MB).  The old refusal: the nine
    zero-thickness RUC soil nodes collide on the ecCodes level key, so a
    level-keyed selection drops five.  Selecting .idx LINES keeps all nine,
    and the distractors (other levels of the same variables) stay out."""
    soil = ["0", "0.01", "0.04", "0.1", "0.3", "0.6", "1", "1.6", "3"]
    twodfld = (["UGRD:planetary boundary layer", "PRES:surface", "HGT:surface",
                "TMP:surface", "SPFH:surface"]
               + [f"{var}:{depth}-{depth} m below ground"
                  for depth in soil for var in ("TSOIL", "SOILW")]
               + ["WEASD:surface", "SNOD:surface", "TMP:2 m above ground",
                  "SPFH:2 m above ground", "UGRD:10 m above ground",
                  "VGRD:10 m above ground", "TMP:30 m above ground",
                  "LFTX:500-1000 mb", "HGT:cloud top", "LAND:surface", "ICEC:surface",
                  "PRES:top of atmosphere"])
    prslev = ([f"{var}:{level} mb" for level in (2, 5, 1000)
               for var in ("HGT", "TMP", "RH", "SPFH", "UGRD", "VGRD", "DZDT")]
              + ["TMP:30-0 mb above ground", "UGRD:30-0 mb above ground"])
    rows = {row.role: row for row in fetch_routes.route_for("rrfs").files}
    taken = {}
    for role, lines in (("2dfld", twodfld), ("prslev", prslev)):
        ranges, _ = record_index.select_ncep(
            _ncep_idx(lines), rows[role].record_index["select"], 100 * len(lines))
        taken[role] = [lines[offset // 100] for offset, _length in ranges]
    assert len([line for line in taken["2dfld"] if "below ground" in line]) == 18
    assert set(taken["2dfld"]) == {
        "PRES:surface", "HGT:surface", "TMP:surface", "WEASD:surface", "SNOD:surface",
        "LAND:surface", "ICEC:surface", "TMP:2 m above ground", "SPFH:2 m above ground",
        "UGRD:10 m above ground", "VGRD:10 m above ground"} | {
        f"{var}:{depth}-{depth} m below ground" for depth in soil for var in ("TSOIL", "SOILW")}
    assert taken["prslev"] == [f"{var}:{level} mb" for level in (2, 5, 1000)
                               for var in ("HGT", "TMP", "SPFH", "UGRD", "VGRD")]
    # The rrfs-ens companion is decoded with the deterministic mapping, so it
    # takes exactly the rrfs rows' records; every RRFS row selects by index.
    ens = {row.role: row for row in fetch_routes.route_for("rrfs-ens").files}
    for companion, role in (("companion_prslev", "prslev"), ("companion_2dfld", "2dfld")):
        assert ens[companion].record_index["select"] == rows[role].record_index["select"]
    for source in ("rrfs", "rrfs-ens"):
        assert all(row.record_index for row in fetch_routes.route_for(source).files)
        assert fetch_routes.resolve_mode(source, "idx-subset") == "idx-subset"
