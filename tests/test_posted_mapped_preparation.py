"""A136 L3 (ii): the mapped engine prepared as its leads post, and A151.

The unit half runs everywhere: the merge of lead batches' alignment
receipts, the composition-inputs input plan an as-posted head binds, the
seal's row check against the lead markers, the posted metadata digest, the
route markers naming each object's lead and the composed files, and the
receipt identity without the run folder's paths (A151).

The real-bytes half decodes a registered composition from the model
gauntlet staging tree (``GPUWM_MODEL_GAUNTLET_STAGING``) whole and lead
batch by lead batch, and requires every frame and the whole receipt equal.
It skips where the tree is not staged.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from woof.ingest import boundary_stream
from woof.mapped_composition import (
    PostedCompositionRefusal,
    composition_receipt_binding_matches,
    composition_receipt_identity_sha256,
    merge_batch_alignment,
)
from _reserved_drives import gauntlet_staging_root


# ---------------------------------------------------------------------------
# merging the lead batches' alignment receipts
# ---------------------------------------------------------------------------

def _terrain(times, supplement_times):
    return {
        "schema": "subset", "status": "PASS", "field": "terrain_height",
        "terrain_full_sha256": "a" * 64, "terrain_subset_sha256": "b" * 64,
        "supplement_valid_times": list(supplement_times),
        "matched_primary_valid_times": list(times),
    }


def test_merged_alignment_lists_every_batch_time_once_and_keeps_shared_keys():
    first = _terrain(["2026-09-30T12:00:00"], ["2026-09-30T12:00:00"])
    second = _terrain(["2026-09-30T15:00:00", "2026-09-30T18:00:00"],
                      ["2026-09-30T15:00:00", "2026-09-30T18:00:00"])
    merged = merge_batch_alignment([first, second], label="terrain")
    assert merged["matched_primary_valid_times"] == [
        "2026-09-30T12:00:00", "2026-09-30T15:00:00", "2026-09-30T18:00:00"]
    # A supplement per lead (the source's own files) lists every batch's.
    assert merged["supplement_valid_times"] == merged[
        "matched_primary_valid_times"]
    assert merged["terrain_full_sha256"] == "a" * 64


def test_a_supplement_every_batch_reads_whole_is_listed_once():
    first = _terrain(["2026-09-30T12:00:00"], ["2026-09-30T12:00:00"])
    second = _terrain(["2026-09-30T18:00:00"], ["2026-09-30T12:00:00"])
    merged = merge_batch_alignment([first, second], label="terrain")
    assert merged["supplement_valid_times"] == ["2026-09-30T12:00:00"]


def test_the_first_valid_time_subset_digest_is_the_first_batch_s():
    first = {"field_subset_sha256": {"land_fraction": "1" * 64},
             "matched_primary_valid_times": ["2026-09-30T12:00:00"]}
    second = {"field_subset_sha256": {"land_fraction": "2" * 64},
              "matched_primary_valid_times": ["2026-09-30T18:00:00"]}
    merged = merge_batch_alignment([first, second], label="binding")
    assert merged["field_subset_sha256"] == {"land_fraction": "1" * 64}


def test_batches_that_disagree_on_a_whole_window_key_are_refused_by_name():
    first = _terrain(["2026-09-30T12:00:00"], ["2026-09-30T12:00:00"])
    second = dict(_terrain(["2026-09-30T18:00:00"], ["2026-09-30T18:00:00"]),
                  terrain_full_sha256="c" * 64)
    with pytest.raises(PostedCompositionRefusal,
                       match="disagree on 'terrain_full_sha256'"):
        merge_batch_alignment([first, second], label="terrain")


def test_overlapping_batch_times_are_refused():
    first = _terrain(["2026-09-30T12:00:00", "2026-09-30T15:00:00"], [])
    second = _terrain(["2026-09-30T15:00:00"], [])
    with pytest.raises(PostedCompositionRefusal, match="overlap"):
        merge_batch_alignment([first, second], label="terrain")


# ---------------------------------------------------------------------------
# the composition-inputs input plan and the seal's row check
# ---------------------------------------------------------------------------

def _row(path, digest="d" * 64, size=10):
    return {"path": path, "bytes": size, "sha256": digest}


def _manifest(lead_digest="e" * 64):
    return {
        "schema": "gpuwm-mapped-composition-inputs-v1",
        "mapping_sha256": "1" * 64, "composition_sha256": "2" * 64,
        "primary_files": [_row("f000.grib2"), _row("f003.grib2", lead_digest)],
        "supplements": {"surface": [_row("f000.grib2"),
                                    _row("f003.grib2", lead_digest)],
                        "donor": [_row("donor/gdas.f000", "9" * 64)]},
        "provenance": {"surface_provenance": _row("/pkg/prov.json", "3" * 64)},
        "decoders": {"engine": _row("/pkg/engine", "4" * 64)},
    }


def test_the_plan_blanks_every_lead_row_and_binds_the_fixed_inputs():
    plan = boundary_stream.input_plan(
        _manifest(), lead_role_prefix="", route_table_sha256="7" * 64,
        fixed_rows=["donor/gdas.f000"])
    rows = plan["manifest"]["primary_files"]
    assert all(row["sha256"] is None and row["bytes"] is None for row in rows)
    assert plan["manifest"]["supplements"]["donor"][0]["sha256"] == "9" * 64
    assert plan["manifest"]["provenance"]["surface_provenance"]["sha256"] \
        == "3" * 64
    # Another lead object's digest is the same plan; another donor is not.
    again = boundary_stream.input_plan(
        _manifest(lead_digest="f" * 64), lead_role_prefix="",
        route_table_sha256="7" * 64, fixed_rows=["donor/gdas.f000"])
    assert again == plan
    moved = _manifest()
    moved["supplements"]["donor"][0]["sha256"] = "8" * 64
    assert boundary_stream.input_plan(
        moved, lead_role_prefix="", route_table_sha256="7" * 64,
        fixed_rows=["donor/gdas.f000"]) != plan
    assert boundary_stream.input_plan(
        _manifest(), lead_role_prefix="", route_table_sha256="6" * 64,
        fixed_rows=["donor/gdas.f000"]) != plan


def _markers(lead_digest="e" * 64):
    return {
        "0": {"objects": [{"name": "f000.grib2", "bytes": 10,
                           "sha256": "d" * 64}]},
        "3": {"objects": [{"name": "f003.grib2", "bytes": 10,
                           "sha256": lead_digest}]},
    }


def test_the_seal_holds_every_lead_row_to_its_marker():
    boundary_stream.hold_composition_rows(
        _manifest(), _markers(), fixed_rows=["donor/gdas.f000"])
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="f003.grib2 .* not an object any lead"):
        boundary_stream.hold_composition_rows(
            _manifest(), _markers(lead_digest="0" * 64),
            fixed_rows=["donor/gdas.f000"])
    # A donor the head did not bind whole would have to be a posted lead.
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="donor/gdas.f000"):
        boundary_stream.hold_composition_rows(
            _manifest(), _markers(), fixed_rows=[])


def test_posted_user_metadata_takes_the_one_digest_the_identity_carries():
    identity = {"bridge_manifest_sha256": "1" * 64,
                "source_identity": {"composition_receipt_sha256": "2" * 64}}
    assert boundary_stream.posted_identity_leaf(
        identity, "composition_receipt_sha256") == "2" * 64
    twice = {**identity, "composition_receipt_sha256": "3" * 64}
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="carries 2 such values"):
        boundary_stream.posted_identity_leaf(
            twice, "composition_receipt_sha256")


def test_a_route_marker_names_each_object_s_lead_and_the_composed_file():
    from woof.fetch_as_posted import (composed_objects, marker_files,
                                       route_objects)

    objects = route_objects(
        [{"relpath": "a.f000", "role": "pgrb2a", "lead": 0, "bytes": 1,
          "sha256": "a" * 64},
         {"relpath": "invariant/HSURF", "role": "invariant", "lead": None,
          "bytes": 2, "sha256": "b" * 64}])
    composed = composed_objects(
        [{"name": "pairs/f000.grib2", "bytes": 3, "sha256": "c" * 64,
          "lead": 0, "parts": ["a.f000", "b.f000"]}])
    assert [item["lead"] for item in objects] == [0, None]
    assert composed == [{"name": "pairs/f000.grib2", "bytes": 3,
                         "sha256": "c" * 64, "lead": 0,
                         "parts": ["a.f000", "b.f000"]}]
    # A preparation reads both; the fetched objects alone are what moved.
    marker = {"objects": objects, "composed": composed}
    assert [item["name"] for item in marker_files(marker)] == [
        "a.f000", "invariant/HSURF", "pairs/f000.grib2"]


def test_the_seal_holds_a_composed_primary_to_its_marker_s_composed_file():
    manifest = _manifest()
    markers = _markers()
    # f003's primary is a composed pair, named under the marker's own
    # ``composed`` key, never among the objects it fetched.
    markers["3"] = {"objects": [{"name": "a.f003", "bytes": 4,
                                 "sha256": "a" * 64}],
                    "composed": [{"name": "f003.grib2", "bytes": 10,
                                  "sha256": "e" * 64, "lead": 3,
                                  "parts": ["a.f003"]}]}
    boundary_stream.hold_composition_rows(
        manifest, markers, fixed_rows=["donor/gdas.f000"])
    markers["3"]["composed"][0]["sha256"] = "0" * 64
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="f003.grib2 .* not an object any lead"):
        boundary_stream.hold_composition_rows(
            manifest, markers, fixed_rows=["donor/gdas.f000"])


def test_the_head_plan_and_the_seal_name_the_batches_contributing_mappings(
        monkeypatch, tmp_path):
    # A contributing mapping adds its donor's format to the decoder rows a
    # composition-inputs manifest seals.  Each batch and the one-shot path
    # name it, so the head's plan and the seal's manifest must as well, or
    # a donor of another format seals other decoder rows than one-shot.
    from types import SimpleNamespace

    from woof import mapped_authoring, mapped_composition, mapped_direct

    seen = {}

    def author(path, **kwargs):
        seen["seal"] = kwargs.get("contributing_mappings")
        return {"manifest": {"path": str(path), "sha256": "a" * 64}}

    def planned(path, **kwargs):
        seen["plan"] = kwargs.get("contributing_mappings")
        return _manifest()

    monkeypatch.setattr(mapped_authoring, "author_input_manifest", author)
    monkeypatch.setattr(mapped_authoring, "planned_input_manifest", planned)
    monkeypatch.setattr(mapped_composition, "posted_composition_bundle",
                        lambda *args, **kwargs: "bundle")
    lead = (tmp_path / "f000.grib2").resolve()
    donor = {"donor_mapping": str(tmp_path / "donor-mapping.json")}
    source = object.__new__(mapped_direct._PostedMappedSource)
    source.__dict__.update(
        input_manifest=(tmp_path / "inputs.json").resolve(),
        mapping=tmp_path / "mapping.json",
        composition=tmp_path / "composition.json",
        primary=(lead,), supplements={}, provenance={}, decoders={},
        contributing=dict(donor), source_format="grib2", leads=(0,),
        lead_of={lead: 0}, carried_any=False, batches=[], frames=None,
        planned={lead},
        posted=SimpleNamespace(route_table_sha256=lambda: "7" * 64),
        through=lambda position: None)
    mapped_direct._posted_input_plan(
        source, mapping=source.mapping, composition=source.composition,
        primary=source.primary, supplements={}, provenance={}, decoders={})
    assert source.finish()[2] == "bundle"
    assert seen == {"plan": donor, "seal": donor}


def _composed_marker(lead, roles):
    """A route marker of ``lead``: one object per role, concatenated into one file."""

    objects = [{"name": f"{role}_PT{0 if role == 'step0' else lead:03d}H.grib2",
                "role": role, "lead": 0 if role == "step0" else lead,
                "bytes": 1, "sha256": "a" * 64} for role in roles]
    return {"objects": objects,
            "composed": [{"name": f"composed/f{lead:03d}.grib2", "bytes": 1,
                          "sha256": "b" * 64, "lead": lead,
                          "parts": [item["name"] for item in objects]}]}


def test_a_later_batch_decodes_the_first_lead_again_where_its_file_alone_carries_statics(
        tmp_path):
    # A route that concatenates the cycle's step-0 objects into the first
    # lead's file only (the land and sea-ice fractions of a GDPS window):
    # a batch without that lead decoded its times without those
    # cycle-invariant fields and was refused, "mapped frame at 2026-10-01
    # 03:00:00 lacks required fields ['land_fraction']".
    from woof.mapped_direct import _PostedMappedSource

    root = tmp_path.resolve()
    primary = tuple(root / f"composed/f{lead:03d}.grib2" for lead in (0, 3, 6))

    def source(markers):
        posted = object.__new__(_PostedMappedSource)
        posted.__dict__.update(
            fetch_root=root, leads=(0, 3, 6), markers={}, lead_of={},
            expected={}, role_of={}, parts_of={}, carried_any=False,
            source_format="grib2",
            primary=primary, supplements={}, planned=set(primary))
        for lead, marker in markers.items():
            posted._read_marker(lead, marker)
        return posted

    posted = source({0: _composed_marker(0, ("state", "step0")),
                     3: _composed_marker(3, ("state",)),
                     6: _composed_marker(6, ("state",))})
    assert not posted._needs_first_lead({0})
    assert not posted._needs_first_lead({0, 3})
    assert posted._needs_first_lead({3})
    assert posted._needs_first_lead({3, 6})
    # What such a batch reads: the first lead's file, then its own.
    again, _ = posted._batch_inventory({0, 3})
    assert [path.name for path in again] == ["f000.grib2", "f003.grib2"]
    # A route whose every lead composes the same roles never needs it.
    pairs = source({lead: _composed_marker(lead, ("pgrb2a", "pgrb2b"))
                    for lead in (0, 3, 6)})
    assert not pairs._needs_first_lead({3})


def test_the_first_lead_decoded_again_is_listed_once_in_the_merged_receipt():
    from dataclasses import dataclass

    from woof.mapped_composition import without_earlier_batch_times

    @dataclass(frozen=True)
    class Batch:
        # The two receipt fields of a lead batch's bundle the trim reads.
        alignment_receipt: dict
        contributing_sources: tuple = ()

    t00, t03, t06 = ("2026-10-01T00:00:00", "2026-10-01T03:00:00",
                     "2026-10-01T06:00:00")
    first = Batch(_terrain([t00], [t00]))
    batches = [first,
               without_earlier_batch_times(Batch(_terrain([t00, t03], [t00])),
                                           first),
               without_earlier_batch_times(Batch(_terrain([t00, t06], [t00])),
                                           first)]
    merged = merge_batch_alignment(
        [batch.alignment_receipt for batch in batches], label="terrain")
    assert merged == _terrain([t00, t03, t06], [t00])


def test_a_supplement_of_one_primary_role_per_lead_follows_the_batch(tmp_path):
    # RRFS's surface is its own 2dfld file of each lead: read whole by
    # every batch, the first batch was refused while later leads' 2dfld
    # files had not posted.
    from woof.mapped_direct import _PostedMappedSource

    root = tmp_path.resolve()
    leads = (0, 1, 2)
    prslev = [root / f"prslev.f{lead:03d}.grib2" for lead in leads]
    surface = [root / f"2dfld.f{lead:03d}.grib2" for lead in leads]
    posted = object.__new__(_PostedMappedSource)
    posted.__dict__.update(
        fetch_root=root, leads=leads, markers={}, lead_of={}, expected={},
        role_of={}, parts_of={}, carried_any=False, source_format="grib2",
        primary=tuple(path for pair in zip(prslev, surface) for path in pair),
        supplements={"surface": tuple(surface)},
        planned={*prslev, *surface})
    posted._read_marker(0, {"objects": [
        {"name": path.name, "role": role, "lead": 0, "bytes": 1,
         "sha256": "a" * 64}
        for path, role in ((prslev[0], "prslev"), (surface[0], "2dfld"))]})
    primary, supplements = posted._batch_inventory({0})
    assert [path.name for path in primary] == [
        "prslev.f000.grib2", "2dfld.f000.grib2"]
    assert [path.name for path in supplements["surface"]] == [
        "2dfld.f000.grib2"]


@pytest.mark.parametrize("decoded", [1, 3])
def test_a_complete_posted_window_is_ready_for_independent_builds(tmp_path, decoded):
    from woof.mapped_direct import _PostedMappedSource

    source = object.__new__(_PostedMappedSource)
    # __init__ is bypassed, so give the source the lock its constructor makes
    # (1e39a57c05c serializes posted batch advancement across readers).
    source._decode_lock = threading.RLock()
    # ... and the decode format it records (1ff1dcd16e: a NetCDF source decodes one lead at a time).
    source.source_format = "grib2"
    source.leads = (0, 1, 2)
    source._next = decoded
    source.posted = boundary_stream.PostedLeads(
        tmp_path, source="test-source", cycle="2026-10-18T12")
    for lead in source.leads:
        (tmp_path / boundary_stream.posted_lead_marker_name(lead)).write_text(
            json.dumps({"schema": boundary_stream.POSTED_LEAD_SCHEMA,
                        "source": "test-source", "cycle": "2026-10-18T12",
                        "lead": lead}))
    advanced = []
    source._decode_batch = lambda start, end: advanced.append((start, end))
    source.markers = {0: {}}
    source._read_marker = lambda lead, marker: source.markers.update({lead: marker})
    source.posted.wait = lambda lead: source.posted.marker(lead) or pytest.fail(
        f"ready lead {lead} disappeared")
    assert source.prepare_independent_times()
    # Every marker is read before workers start; nothing is decoded yet, so
    # the first boundary time waits for one batch, not the whole window.
    assert set(source.leads[decoded:]) <= set(source.markers)
    assert advanced == []
    # Workers' frame lookups decode the rest without changing segment bindings.
    markers = dict(source.markers)
    source.through(2)
    source.through(1)
    assert source.markers == markers
    assert source.decoded_count == len(source.leads)
    assert advanced == ([(1, 2)] if decoded == 1 else [])


def _posted_source(tmp_path, leads, decoded):
    from woof.mapped_direct import _PostedMappedSource

    source = object.__new__(_PostedMappedSource)
    source._decode_lock = threading.RLock()
    source.source_format = "grib2"
    source.leads = tuple(leads)
    source._next = decoded
    source.posted = boundary_stream.PostedLeads(
        tmp_path, source="test-source", cycle="2026-10-18T12")
    for lead in source.leads:
        (tmp_path / boundary_stream.posted_lead_marker_name(lead)).write_text(
            json.dumps({"schema": boundary_stream.POSTED_LEAD_SCHEMA,
                        "source": "test-source", "cycle": "2026-10-18T12",
                        "lead": lead}))
    advanced = []
    source._decode_batch = lambda start, end: advanced.append((start, end))
    source.markers = {}
    source._read_marker = lambda lead, marker: source.markers.update({lead: marker})
    source.posted.wait = lambda lead: source.posted.marker(lead)
    return source, advanced


def test_a_long_posted_window_is_decoded_a_bounded_batch_at_a_time(tmp_path):
    from woof.mapped_direct import POSTED_BATCH_LEADS

    # A 48 h window whose every lead posted before the decode started (V59/V60
    # 2026-10-21): one lookup must not compose the whole window.
    source, advanced = _posted_source(tmp_path, range(49), decoded=2)
    source.through(2)
    assert advanced == [(2, 2 + POSTED_BATCH_LEADS - 1)]
    source.through(20)
    assert [end - start + 1 for start, end in advanced] == [POSTED_BATCH_LEADS] * 4
    assert advanced[-1][1] >= 20 and source.decoded_count == advanced[-1][1] + 1
    # The seal still reaches every lead, a batch at a time.
    source.through(48)
    assert source.decoded_count == 49 and advanced[-1][1] == 48


def test_the_first_batch_and_a_prefetch_decode_exactly_what_they_name(tmp_path):
    source, advanced = _posted_source(tmp_path, range(49), decoded=0)
    # The start time and the forcing time after it, for the first step.
    source.through(0, until=1)
    # The inland-water daily mean's 24 h window, in one batch.
    source.prefetch_through(23)
    assert advanced == [(0, 1), (2, 23)]


def test_a_missing_posted_lead_keeps_builds_ordered_without_waiting(tmp_path):
    from woof.mapped_direct import _PostedMappedSource

    source = object.__new__(_PostedMappedSource)
    source.leads, source._next = (0, 1, 2), 1
    source.posted = boundary_stream.PostedLeads(
        tmp_path, source="test-source", cycle="2026-10-18T12")
    source.through = lambda index: pytest.fail("pending window was decoded")
    source.posted.wait = lambda lead: pytest.fail("readiness check blocked")
    assert not source.prepare_independent_times()
    # Existing marker validation still refuses another cycle before admission.
    (tmp_path / boundary_stream.posted_lead_marker_name(1)).write_text(
        json.dumps({"schema": boundary_stream.POSTED_LEAD_SCHEMA,
                    "source": "test-source", "cycle": "2026-10-17T12",
                    "lead": 1}))
    with pytest.raises(boundary_stream.BoundaryProducerFailed, match="not test-source"):
        source.prepare_independent_times()


# ---------------------------------------------------------------------------
# A151: the receipt identity without the run folder's paths
# ---------------------------------------------------------------------------

def _receipt(root):
    body = {
        "schema": "gpuwm-mapped-composition-receipt-v1",
        "mapping": {"path": "/pkg/mapping.json", "sha256": "1" * 64},
        "input_manifest": {"path": f"{root}/inputs.json", "sha256": "2" * 64},
        "terrain_products": [{"path": f"{root}/f000.grib2",
                              "sha256": "3" * 64}],
        "terrain_provenance": {"provenance_path": f"{root}/prov.json",
                               "provenance_sha256": "4" * 64},
        "valid_times": ["2026-09-30T12:00:00"],
    }
    body["receipt_content_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def test_two_run_folders_of_the_same_bytes_bind_one_receipt_identity():
    one, other = _receipt("/runs/a"), _receipt("/runs/b")
    assert one["receipt_content_sha256"] != other["receipt_content_sha256"]
    assert composition_receipt_identity_sha256(one) \
        == composition_receipt_identity_sha256(other)
    changed = _receipt("/runs/a")
    changed["input_manifest"]["sha256"] = "5" * 64
    assert composition_receipt_identity_sha256(changed) \
        != composition_receipt_identity_sha256(one)


def test_a_cache_bound_before_a151_still_names_its_receipt():
    receipt = _receipt("/runs/a")
    assert composition_receipt_binding_matches(
        receipt["receipt_content_sha256"], receipt)
    assert composition_receipt_binding_matches(
        composition_receipt_identity_sha256(receipt), receipt)
    assert not composition_receipt_binding_matches("6" * 64, receipt)


# ---------------------------------------------------------------------------
# real bytes: a lead-batch decode equals the whole decode
# ---------------------------------------------------------------------------

#: The staging root, never one on or linked onto a reserved drive
#: (tests/_reserved_drives.py names the breakage).
STAGING = gauntlet_staging_root()


def _two_lead_windows():
    """Each source's staged window as ``(primary by lead, supplement rule)``.

    ``every`` is a supplement that is the primary inventory itself (it
    follows the batch); a path tuple is read whole by every batch.
    """

    icon = STAGING / "icon" / "eu-regular-latlon"
    icon_objects = sorted(icon.glob("*.grib2.bz2"))
    invariant = tuple(path for path in icon_objects
                      if "_time-invariant_" in path.name)
    return {
        "gefs": ((
            (STAGING / "gefs" / "gec00.t00z.pgrb2a.0p50.f000",
             STAGING / "gefs" / "gec00.t00z.pgrb2b.0p50.f000"),
            (STAGING / "gefs" / "gec00.t00z.pgrb2a.0p50.f003",
             STAGING / "gefs" / "gec00.t00z.pgrb2b.0p50.f003")),
            "every", ()),
        "aifs": ((
            (STAGING / "aifs" / "20260817000000-0h-oper-fc.grib2",),
            (STAGING / "aifs" / "20260817000000-6h-oper-fc.grib2",)),
            (STAGING / "aifs" / "20260817000000-0h-oper-fc.grib2",), ()),
        "aigfs": ((
            (STAGING / "aigfs" / "NOMADS.aigfs.t00z.pres.f000.grib2",
             STAGING / "aigfs" / "NOMADS.aigfs.t00z.sfc.f000.grib2"),
            (STAGING / "aigfs-hybrid" / "NOMADS.aigfs.t00z.pres.f006.grib2",
             STAGING / "aigfs" / "NOMADS.aigfs.t00z.sfc.f006.grib2")),
            (STAGING / "crosssource" / "gdas.t00z.pgrb2.0p25.f000",), ()),
        "icon-eu": ((
            tuple(path for path in icon_objects if "_000_" in path.name),
            tuple(path for path in icon_objects if "_001_" in path.name)),
            tuple(path for path in invariant if "_HSURF" in path.name),
            invariant),
    }


def _recipe(source):
    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import (packaged_authorities,
                                          packaged_contributing_mappings,
                                          packaged_profile)

    name = str(get_source_adapter(source).packaged_profile)
    profile = packaged_profile(name)
    authorities = packaged_authorities(name)
    return {
        "mapping": Path(authorities["mapping"]),
        "composition": Path(authorities["composition"]),
        "data_role": str(profile["data_role"]),
        "provenance": {str(profile["provenance_role"]):
                       Path(authorities["provenance"])},
        "contributing": {str(role): Path(path) for role, path
                         in packaged_contributing_mappings(name).items()},
        "format": str(profile["source_format"]),
    }


def _decode(recipe, primary, supplement, work, name, *, lead_batch=False):
    from woof.mapped_authoring import author_input_manifest
    from woof.mapped_composition import decode_composed_source

    manifest = work / f"{name}.json"
    authored = author_input_manifest(
        manifest, mapping_path=recipe["mapping"],
        composition_path=recipe["composition"], primary_files=primary,
        supplement_files={recipe["data_role"]: supplement},
        provenance_files=recipe["provenance"],
        contributing_mappings=recipe["contributing"] or None,
        expected_format=recipe["format"])
    return decode_composed_source(
        recipe["composition"], recipe["mapping"], primary,
        {recipe["data_role"]: supplement}, recipe["provenance"],
        input_manifest=manifest,
        input_manifest_sha256=authored["manifest"]["sha256"],
        contributing_mappings=recipe["contributing"] or None,
        scratch_destination=work / name / "out", lead_batch=lead_batch)


@pytest.mark.slow
@pytest.mark.parametrize("source", ["gefs", "aifs", "aigfs", "icon-eu"])
def test_a_lead_batch_decode_equals_the_whole_decode(source, tmp_path):
    from woof.mapped_composition import (
        PostedFrames, mapped_composition_receipt, posted_composition_bundle)

    leads, supplement_rule, common = _two_lead_windows()[source]
    files = [path for lead in leads for path in lead] + list(common)
    if not all(Path(path).is_file() for path in files) or not all(leads):
        pytest.skip(f"{source}'s two-lead window is not staged under {STAGING}")
    recipe = _recipe(source)
    common = tuple(common)
    whole_primary = common + tuple(path for lead in leads for path in lead)
    if supplement_rule == "every":
        whole_supplement = whole_primary
    else:
        whole_supplement = tuple(supplement_rule)
    whole = _decode(recipe, whole_primary, whole_supplement, tmp_path, "whole")
    batches = []
    for index, lead in enumerate(leads):
        primary = common + tuple(lead)
        supplement = (primary if supplement_rule == "every"
                      else tuple(supplement_rule))
        batches.append(_decode(recipe, primary, supplement, tmp_path,
                               f"batch-{index}", lead_batch=True))
    try:
        where = {}
        position = 0
        for batch in batches:
            for local in range(len(batch.frames)):
                where[position] = (batch.frames, local)
                position += 1
        frames = PostedFrames(whole.frames.valid_times, where.__getitem__)
        for batch in batches:
            frames.add_part(batch.frames)
        assert frames.valid_times == tuple(whole.frames.valid_times)
        for index in range(len(whole.frames)):
            assert frames.header(index).to_dict() \
                == whole.frames.header(index).to_dict()
            names = whole.frames.field_names(index)
            assert frames.field_names(index) == names
            for name in names:
                assert frames.field_digest(index, name) \
                    == whole.frames.field_digest(index, name)
            one = whole.frames[index]
            other = frames[index]
            for name in names:
                assert np.array_equal(
                    np.asarray(one.fields[name].values),
                    np.asarray(other.fields[name].values), equal_nan=True)
        seal = posted_composition_bundle(
            batches, frames, input_manifest_path=whole.input_manifest_path,
            input_manifest_sha256=whole.input_manifest_sha256,
            supplement_files={recipe["data_role"]: whole_supplement},
            shared_primary=bool(common))
        assert mapped_composition_receipt(seal) \
            == mapped_composition_receipt(whole)
    finally:
        whole.close()
        for batch in batches:
            batch.close()


def test_parallel_forcing_readers_decode_each_posted_batch_once():
    """Parallel root/forcing staging must not publish batch-N twice."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from types import SimpleNamespace
    from woof.mapped_direct import _PostedMappedSource
    source = object.__new__(_PostedMappedSource)
    source._decode_lock = threading.RLock()
    source._next = 0
    source.source_format = "grib2"
    source.leads = (0,)
    source.markers = {0: {}}
    source.posted = SimpleNamespace()
    calls = []
    entered, release = threading.Event(), threading.Event()
    def decode(start, end):
        calls.append((start, end))
        entered.set()
        assert release.wait(5)
    source._decode_batch = decode
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(source.through, 0)
        assert entered.wait(5)
        second = pool.submit(source.through, 0)
        time.sleep(0.05)
        release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert calls == [(0, 0)]
    assert source.decoded_count == 1
