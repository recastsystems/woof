"""Source identity, indexed range safety, and streaming member addressing."""

from datetime import datetime
import io
import json

import pytest

from woof import fetch_routes, source_adapters, source_readiness
from woof.member_grammar import load_member_grammar, MemberIdentityRefusal
from woof.member_index import select_member_ranges, download_indexed_member
from woof.source_authorities import packaged_member_grammar


INDEX = {"member_key": "number", "offset_key": "_offset", "length_key": "_length",
         "filters": {"type": "pf"}}


def _index(*rows):
    return "\n".join(json.dumps(row) for row in rows).encode()


def test_ensemble_links_and_separate_control_are_registry_facts():
    assert source_adapters.get_source_adapter("gfs").ensemble_source == "gefs"
    assert source_adapters.get_source_adapter("ifs").ensemble_source == "ecmwf-ens"
    assert source_adapters.get_source_adapter("aigfs").ensemble_source == "aigefs"
    assert source_adapters.get_source_adapter("ai-gefs").source_id == "aigefs"
    row = source_adapters.get_source_adapter("ecmwf-ens")
    assert row.ensemble_control_source == "ecmwf-open-data"
    grammar = load_member_grammar(packaged_member_grammar(row.member_set))
    assert [member.ordinal for member in grammar.members()] == list(range(1, 51))
    assert grammar.member_for_ordinal(0) is None
    with pytest.raises(MemberIdentityRefusal):
        grammar.member("p51")


def test_index_selection_never_includes_other_members_or_statistics():
    data = _index(
        {"number": "1", "type": "pf", "_offset": 100, "_length": 40},
        {"number": "2", "type": "pf", "_offset": 0, "_length": 40},
        {"number": "1", "type": "em", "_offset": 40, "_length": 40})
    assert select_member_ranges(data, INDEX, 1) == ((100, 40),)
    with pytest.raises(ValueError, match="no records"):
        select_member_ranges(data, INDEX, 3)


def test_public_regional_members_initialize_only_with_their_declared_companion(tmp_path):
    row = source_adapters.get_source_adapter("rrfs-ens")
    assert source_adapters.get_source_adapter("rrfs").ensemble_source == row.source_id
    # Runnable as table work: the member's atmosphere plus the declared
    # deterministic companion of the same cycle.  The row still names the
    # breakage a member without its companion has.
    assert row.runnable
    assert row.packaged_profile == "rrfs-ens-member-companion-grib2-v1"
    assert "companion" in row.composition_requirement
    assert "250 hPa" in row.composition_requirement
    assert "no soil state" in row.composition_requirement
    assert "model top sits above the data" in row.composition_requirement
    grammar = load_member_grammar(packaged_member_grammar(row.member_set))
    assert [m.member_id for m in grammar.members()] == [f"m{n:03d}" for n in range(1, 6)]
    assert grammar.member_for_ordinal(0) is None
    plan = fetch_routes.resolve_request("rrfsens", cycle=datetime(2026, 10, 1),
                                        hours=1, member="m005")
    member = [obj for obj in plan.objects if "/m005/" in obj.key]
    companion = [obj for obj in plan.objects if obj not in member]
    assert len(member) == 4
    assert sorted(obj.key.rsplit("/", 1)[-1] for obj in companion) == sorted(
        f"rrfs.t00z.{product}.3km.f{lead:03d}.conus.grib2"
        for product in ("prslev", "2dfld") for lead in (0, 1))
    assert all(obj.key.startswith("rrfs.20261001/00/") for obj in companion)
    assert all(obj.relpath == "upstream/" + obj.key for obj in plan.objects)
    assert plan.route.posting.streams
    # The companion is never a primary input: the member's own decode
    # stays one member and one generating process.
    # Object keys are POSIX relpaths; the plan holds platform paths (backslashes on Windows).
    assert {str(path).replace("\\", "/") for path in plan.primary_files} == {
        obj.relpath for obj in member}
    assert sorted(str(path).replace("\\", "/") for role, path in plan.extra_supplement_files
                  if role == "rrfs_ens_companion_data") == sorted(
        obj.relpath for obj in companion)
    for obj in plan.objects:
        path = tmp_path / obj.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"path fixture")
    fetch_routes.write_handoff(plan, tmp_path)
    from pathlib import Path
    handoff = json.loads((tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert all((Path(handoff["member_prep"]["inputs"]) / obj.key).is_file()
               for obj in member)
    tokens = [value for value in json.dumps(handoff).split('"')]
    # write_handoff spells paths for I/O (io_path: the extended-length form on Windows, unchanged on POSIX).
    from woof.filesystem_paths import io_path
    for obj in companion:
        binding = f"rrfs_ens_companion_data={(io_path(tmp_path) / obj.relpath).resolve()}"
        assert json.dumps(binding)[1:-1] in tokens


def test_member_profile_declares_its_companion_as_composition_rows():
    """The rrfs-ens profile is rows, not a code path.

    Every member field pins product definition template 1 on the member's
    own 14 levels; the composition's one binding declares the deterministic
    RRFS mapping as the member's companion, which stacks its levels above
    250 hPa onto the member's columns and lends the land state.
    """
    from woof.mapped_composition import load_composition
    from woof.source_authorities import (packaged_authorities,
                                          packaged_contributing_mappings)

    profile = source_adapters.get_source_adapter("rrfs-ens").packaged_profile
    authorities = packaged_authorities(profile)
    contract = load_composition(authorities["composition"], authorities["mapping"])
    assert contract["supplements"]["terrain_height"]["data_role"] == (
        "rrfs_ens_member_in_band_surface")
    (name, binding), = contract["field_sources"].items()
    assert binding["companion"] == {
        "vertical_extension": "above_primary_top",
        "fields": ["air_temperature", "specific_humidity", "eastward_wind",
                   "northward_wind", "geopotential_height"]}
    assert binding["time_alignment"] == "valid_time_exact"
    assert set(binding["fields"]) == {
        "skin_temperature", "land_fraction", "sea_ice_fraction",
        "snow_water_equivalent", "snow_depth", "soil_temperature",
        "volumetric_soil_moisture"}
    donor = packaged_contributing_mappings(profile)[binding["mapping_role"]]
    assert json.loads(donor.read_text())["name"] == binding["source_id"]
    mapping = json.loads(authorities["mapping"].read_text(encoding="utf-8"))
    assert mapping["coordinates"]["vertical"]["levels"] == [
        25000, 30000, 40000, 50000, 60000, 70000, 75000, 80000, 85000,
        90000, 92500, 95000, 97500, 100000]
    for field_name, field in mapping["fields"].items():
        if field.get("provider") == "composition_bound":
            assert field_name in binding["fields"]
        for selector in field.get("selectors", ()):
            assert selector["pdt"] == 1, field_name
    assert mapping["fields"]["specific_humidity_2m"]["derivation"] == (
        "humidity-2m-from-dewpoint")
    provenance = json.loads(authorities["provenance"].read_text(encoding="utf-8"))
    assert provenance["companion"]["binding"] == name


def test_a_member_without_its_companion_is_refused_in_the_rows_words(tmp_path):
    import argparse
    from woof import source_cli

    adapter = source_adapters.get_source_adapter("rrfs-ens")
    args = argparse.Namespace(
        mapping=None, composition=None, provenance=None, descriptor=None,
        author_mapping=None, contributing_mapping=None, source_format=None,
        supplement=[str(tmp_path / "member.2dfldnomads.grib2")])
    errors = source_cli._apply_packaged_profile(args, adapter, "woof prep")
    assert len(errors) == 1
    assert "rrfs_ens_companion_data" in errors[0]
    assert "no soil state" in errors[0]
    assert "model top sits above the data" in errors[0]
    args.supplement = [str(tmp_path / "member.2dfldnomads.grib2"),
                       f"rrfs_ens_companion_data={tmp_path / 'prslev.grib2'}"]
    args.mapping = args.composition = args.provenance = None
    args.contributing_mapping = None
    args.source_format = None
    assert source_cli._apply_packaged_profile(args, adapter, "woof prep") == []


@pytest.mark.parametrize("changed", [{"member": "2"}, {"pdt": "2"},
                                      {"pdt": "0"}, {"ensemble_size": "50"}])
def test_index_claim_cannot_override_actual_grib_identity(changed):
    from woof.member_prep import verify_member_rows
    row = source_adapters.get_source_adapter("ecmwf-ens")
    grammar = load_member_grammar(packaged_member_grammar(row.member_set))
    good = {"index": "0", "pdt": "1", "member": "1", "ensemble_type": "255",
            "ensemble_size": "51", "generating_process": "4",
            "forecast_generating_process_id": "161", "derived_forecast": "-"}
    with pytest.raises(MemberIdentityRefusal):
        verify_member_rows(grammar, grammar.member("p01"),
                           [good, {**good, "index": "1", **changed}],
                           source_label="indexed member")


@pytest.mark.parametrize("offset,length", [(-1, 40), (0, 0), (True, 40), (0, "40")])
def test_invalid_ranges_refuse_before_http(offset, length):
    with pytest.raises(ValueError, match="invalid"):
        select_member_ranges(_index({"number": "1", "type": "pf", "_offset": offset,
                                     "_length": length}), INDEX, 1)


def test_overlapping_ranges_cannot_duplicate_messages():
    with pytest.raises(ValueError, match="overlap"):
        select_member_ranges(_index(
            {"number": "1", "type": "pf", "_offset": 0, "_length": 40},
            {"number": "2", "type": "pf", "_offset": 30, "_length": 40}), INDEX, 1)


def test_indexed_route_preserves_member_paths_and_per_lead_streaming():
    plan = fetch_routes.resolve_request("ecmwf-ens", cycle=datetime(2026, 10, 1),
                                        hours=6, member="p50")
    assert plan.route.posting.streams
    assert [item.lead for item in plan.objects] == [0, 3, 6]
    assert all(item.member_ordinal == 50 for item in plan.objects)
    assert all(item.relpath.startswith("members/p50/") for item in plan.objects)
    assert all(item.idx_url.endswith("-enfo-ef.index") for item in plan.objects)
    assert all(not item.idx_url.endswith(".grib2.index") for item in plan.objects)
    assert plan.supplement_files == plan.primary_files


def test_posted_readiness_uses_replacement_index_suffix():
    plan = fetch_routes.resolve_request("ecmwf-ens", cycle=datetime(2026, 10, 1),
                                        hours=3, member="p01")
    from types import SimpleNamespace
    window = SimpleNamespace(source="ecmwf-ens", provider=None, plan=plan)
    urls = source_readiness.lead_urls(window, 0, "ecmwf", objects=(plan.objects[0],))
    assert len(urls) == 2
    assert urls[0].endswith("-0h-enfo-ef.grib2")
    assert urls[1].endswith("-0h-enfo-ef.index")


def test_archived_member_crosses_control_layout_change_without_date_refusal():
    with pytest.raises(ValueError, match="publication|2024-03-18|2024-03"):
        fetch_routes.resolve_request("ecmwf-ens", cycle=datetime(2024, 3, 18, 6),
                                     hours=3, member="p01")
    for cycle in (datetime(2024, 10, 29), datetime(2026, 5, 12),
                  datetime(2026, 5, 12, 6)):
        plan = fetch_routes.resolve_request("ecmwf-ens", cycle=cycle,
                                            hours=6, member="p01")
        assert [obj.lead for obj in plan.objects] == [0, 3, 6]
        assert all(obj.member_ordinal == 1 for obj in plan.objects)
        assert plan.objects[0].key.startswith(cycle.strftime("%Y%m%d/%Hz/"))
        assert plan.objects[0].idx_url.endswith("-enfo-ef.index")
    row = source_adapters.get_source_adapter("ecmwf-ens")
    archive = next(item for item in row.archive_windows if item.transport == "aws")
    assert archive.start == "2024-03-18T12"
    assert "CC-BY-4.0" in archive.note


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


def test_whole_ensemble_http_reply_is_refused_without_reading_body(tmp_path, monkeypatch):
    index = _index({"number": "1", "type": "pf", "_offset": 0, "_length": 40})
    whole = Response(b"must not be read", status=200, headers={"ETag": '"v1"'})
    responses = iter((Response(index), whole))
    monkeypatch.setattr("woof.member_index.paced_urlopen", lambda *a, **k: next(responses))
    reads = []
    whole.read = lambda *args: reads.append(args)
    dest = tmp_path / "member.grib2"
    with pytest.raises(ValueError, match="exact requested bytes"):
        download_indexed_member("https://example.invalid/data", "https://example.invalid/index",
                                dest, declaration=INDEX, member_ordinal=1,
                                source="ecmwf-ens", member="p01")
    assert not reads
    assert not dest.exists()
    assert not dest.with_suffix(".grib2.part").exists()


def test_changed_object_between_ranges_is_not_published(tmp_path, monkeypatch):
    index = _index({"number": "1", "type": "pf", "_offset": 0, "_length": 40},
                   {"number": "1", "type": "pf", "_offset": 80, "_length": 40})
    responses = iter((Response(index),
                      Response(b"x" * 40, 206, {"Content-Range": "bytes 0-39/120", "ETag": '"v1"'}),
                      Response(b"x" * 40, 206, {"Content-Range": "bytes 80-119/120", "ETag": '"v2"'})))
    requests = []
    def open_response(request, **kwargs):
        requests.append(request)
        return next(responses)
    monkeypatch.setattr("woof.member_index.paced_urlopen", open_response)
    dest = tmp_path / "member.grib2"
    with pytest.raises(ValueError, match="changed"):
        download_indexed_member("https://example.invalid/data", "https://example.invalid/index",
                                dest, declaration=INDEX, member_ordinal=1,
                                source="ecmwf-ens", member="p01")
    assert requests[-1].get_header("If-match") == '"v1"'
    assert not dest.exists()


@pytest.mark.parametrize("etag,content_range", [('W/"v1"', "bytes 0-39/40"),
                                                ("v1", "bytes 0-39/40"),
                                                ('"v1"', "bytes 0-39/20")])
def test_unpinned_or_impossible_range_is_rejected_before_body(tmp_path, monkeypatch,
                                                            etag, content_range):
    index = _index({"number": "1", "type": "pf", "_offset": 0, "_length": 40})
    invalid = Response(b"must not be read", 206,
                       {"Content-Range": content_range, "ETag": etag})
    reads = []
    invalid.read = lambda *args: reads.append(args)
    responses = iter((Response(index), invalid))
    monkeypatch.setattr("woof.member_index.paced_urlopen", lambda *a, **k: next(responses))
    with pytest.raises(ValueError, match="strong ETag|object size"):
        download_indexed_member("https://example.invalid/data", "https://example.invalid/index",
                                tmp_path / "member.grib2", declaration=INDEX,
                                member_ordinal=1, source="ecmwf-ens", member="p01")
    assert not reads


# fix/ecmwf-ens-record-subset: an ECMWF ENS member moved all 177 of its
# records per lead (80-91 MB) one range at a time (2.4 MB/s from a host that
# serves one object at 38.8 MB/s) while the mapping binds 85 of them
# (35.8-38.5 MB).  The row's record_index narrows the member and the ranges
# move in parallel, pinned to the first reply's ETag.

RECORDS = {"format": "ecmwf-jsonl", "why": "test", "select": [
    {"levtype": "pl", "param": ["t", "u"]}, {"levtype": "sol", "param": ["sot"]}]}


def test_a_record_selection_narrows_the_member_and_never_widens_it():
    data = _index(
        {"number": "1", "type": "pf", "levtype": "pl", "param": "t", "_offset": 0, "_length": 40},
        {"number": "2", "type": "pf", "levtype": "pl", "param": "t", "_offset": 40, "_length": 40},
        {"number": "1", "type": "pf", "levtype": "pl", "param": "w", "_offset": 80, "_length": 40},
        {"number": "1", "type": "pf", "levtype": "sol", "param": "sot", "_offset": 120, "_length": 40},
        {"number": "1", "type": "em", "levtype": "pl", "param": "u", "_offset": 160, "_length": 40})
    assert select_member_ranges(data, INDEX, 1) == ((0, 40), (80, 40), (120, 40))
    assert select_member_ranges(data, INDEX, 1, RECORDS) == ((0, 40), (120, 40))
    with pytest.raises(ValueError, match="keeps none"):
        select_member_ranges(data, INDEX, 1, {**RECORDS, "select": [{"param": ["msl"]}]})
    with pytest.raises(ValueError, match="no records"):
        select_member_ranges(data, INDEX, 3, RECORDS)


def test_the_ens_member_row_selects_its_mapped_records():
    plan = fetch_routes.resolve_request("ecmwf-ens", cycle=datetime(2026, 10, 28, 12),
                                        hours=6, member="p07")
    assert plan.objects and all(obj.member_index and obj.record_index for obj in plan.objects)
    assert fetch_routes.resolve_mode("ecmwf-ens", "idx-subset") == "idx-subset"
    (row,) = fetch_routes.route_for("ecmwf-ens").files
    (ifs,) = fetch_routes.route_for("ecmwf-open-data").files
    # The same parameters the IFS mapping binds, current and 2024 names.
    assert row.record_index["select"] == ifs.record_index["select"]


def test_a_record_selection_beside_a_lead_index_is_refused_at_load():
    with pytest.raises(ValueError, match="JSON-lines member index"):
        fetch_routes._record_index({"role": "state", "idx_sidecar": ".idx",
                                    "member_index": {"format": "wgrib2"},
                                    "record_index": RECORDS})
    with pytest.raises(ValueError, match="JSON-lines member index"):
        fetch_routes._record_index({"role": "state", "idx_sidecar": ".idx",
                                    "member_index": INDEX,
                                    "record_index": {"format": "ncep-idx",
                                                     "select": [["TMP", "500 mb"]]}})


def test_the_narrowed_member_moves_in_parallel_pinned_to_one_version(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    rows, data = [], b""
    for number, param in [(n, p) for p in ("t", "w", "u", "d") for n in ("1", "2")]:
        body = f"{number}{param}".encode() * 30
        rows.append({"number": number, "type": "pf", "levtype": "pl", "param": param,
                     "_offset": len(data), "_length": len(body)})
        data += body
    asked, lock = [], threading.Lock()

    def host(request, **kwargs):
        if request.full_url.endswith("/index"):
            return Response(_index(*rows))
        first, last = (int(v) for v in request.headers["Range"].split("=")[1].split("-"))
        with lock:
            asked.append((first, request.headers.get("If-match")))
        return Response(data[first:last + 1], 206,
                        {"Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": '"v1"'})

    monkeypatch.setattr("woof.member_index.paced_urlopen", host)
    monkeypatch.setattr("woof.member_prep.verify_member_file",
                        lambda grammar, member, path: SimpleNamespace(
                            messages=2, to_dict=lambda: {"member": member}))
    dest = tmp_path / "member.grib2"
    entry = download_indexed_member("https://example.invalid/data", "https://example.invalid/index",
                                    dest, declaration=INDEX, member_ordinal=1,
                                    source="ecmwf-ens", member="p01", records=RECORDS)
    want = [r for r in rows if r["number"] == "1" and r["param"] in ("t", "u")]
    assert dest.read_bytes() == b"".join(
        data[r["_offset"]:r["_offset"] + r["_length"]] for r in want)
    assert entry["selected_records"] == 2 and entry["requests"] == 2
    assert entry["bytes"] == sum(r["_length"] for r in want)
    # The first range learns the tag alone; every later one carries it.
    assert asked[0] == (0, None)
    assert all(tag == '"v1"' for _offset, tag in asked[1:])
    assert not dest.with_name(dest.name + ".part").exists()
