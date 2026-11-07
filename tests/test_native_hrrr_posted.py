"""A136 L7c (b): native HRRR prepares as its leads post.

The native route's cache identity binds two digests of other documents:
``bridge_manifest_sha256`` (the decoded bridge's SHA256SUMS) and
``source_manifest_sha256`` (the fetch's SHA256SUMS).  Neither exists
before the last lead, so an as-posted head carries the input plan's
placeholder in both, its seal writes each document and its digest, and
every row of each document is held to a per-lead record bound as the lead
was read: the source rows to the leads' posted markers, the bridge rows to
the decoded files each segment was built from.  The decoder reads a lead
only once the preparation has held its two files to the lead's marker
(``PostedLeadAdmitter``).

CPU only; no device, no source data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from woof.ingest import boundary_stream
from woof.ingest.boundary_stream import (
    POSTED_LEAD_SCHEMA, BoundaryStreamError, PostedLeads, SourceBehind,
)

CYCLE = "2026-09-30T12"
ROUTE_TABLE = "7" * 64
LEADS = (6, 7, 8)
BRIDGE = "native/native-bridge/SHA256SUMS"
SOURCE = "native/posted-source/SHA256SUMS"
FIXED = ("./gate.txt", "./inventory.tsv")
DOCUMENT_KEYS = {"bridge_manifest_sha256": "bridge",
                 "source_manifest_sha256": "source_manifest"}


def _sha(text) -> str:
    data = text if isinstance(text, bytes) else str(text).encode()
    return hashlib.sha256(data).hexdigest()


def _objects(lead):
    return [{"role": "atmosphere",
             "name": f"hrrr.t12z.wrfnatf{lead:02d}.grib2",
             "url": "u", "endpoint": "s3", "bytes": len(f"nat{lead}"),
             "sha256": _sha(f"nat{lead}")},
            {"role": "soil", "name": f"hrrr.t12z.soilf{lead:02d}.grib2",
             "url": "u", "endpoint": "s3", "bytes": len(f"soil{lead}"),
             "sha256": _sha(f"soil{lead}")}]


def _marker(lead):
    return {"schema": POSTED_LEAD_SCHEMA, "source": "hrrr", "member": None,
            "cycle": CYCLE, "lead": lead,
            "valid_time": f"2026-09-30T{12 + lead:02d}:00:00Z",
            "objects": _objects(lead),
            "expected_at": "2026-09-30T13:00:00Z",
            "late_at": "2026-09-30T14:00:00Z",
            "first_seen_at": "2026-09-30T13:01:00Z",
            "fetched_at": "2026-09-30T13:01:30Z"}


def _decoded(lead, *, salt=""):
    return {f"./atmosphere-f{lead:02d}/TT.f32le": _sha(f"tt{lead}{salt}"),
            f"./soil-f{lead:02d}/SOILT.f32le": _sha(f"st{lead}")}


def _sums(rows) -> str:
    return "".join(f"{digest}  {name}\n"
                   for name, digest in sorted(dict(rows).items()))


def _plan_manifest():
    return {
        "schema": "gpuwm-hrrr-native-input-manifest-v1",
        "source": {"model": "HRRR", "product": "conus/wrfnat+wrfprs",
                   "cycle": "2026-09-30T12:00:00Z",
                   "forecast_hours": list(LEADS)},
        "files": {"bridge": {"name": "SHA256SUMS", "sha256": None},
                  "source_manifest": {"name": "SHA256SUMS", "sha256": None},
                  "namelist_input": {"name": "namelist.input",
                                     "sha256": "d" * 64}},
    }


def _native_posted(tmp_path, *, bridge_rows=None, source_rows=None,
                   seal_records=None, sealed_identity=None,
                   document_sha256=None):
    """A native bundle prepared as posted and sealed as the benchmark seals.

    ``bridge_rows`` and ``source_rows`` are the rows the seal writes into
    the two documents (default: the decoded records and the markers'
    objects); ``seal_records`` the decoded records the seal records.
    """

    from test_boundary_stream import _initial, _met, _snapshots, _times
    from woof.ingest.lateral_bc import StateBoundaryFrames

    snapshots = _snapshots(len(LEADS))
    times = _times(len(LEADS))
    plan = boundary_stream.input_plan(
        _plan_manifest(), lead_role_prefix="hrrr-f",
        route_table_sha256=ROUTE_TABLE,
        derived_roles=("bridge", "source_manifest"))
    placeholder = boundary_stream.as_posted_placeholder(
        boundary_stream.input_plan_sha256(plan))
    identity = {"bridge_manifest_sha256": placeholder,
                "source_manifest_sha256": placeholder,
                "static_cache_sha256": "e" * 64}
    root = tmp_path / "bundle"
    (root / "native").mkdir(parents=True)

    def in_place(staging, output):
        assert Path(staging) == Path(output)

    writer = boundary_stream.PreparedTreeWriter(
        staging=root, output_root=root, identity=identity, chained=True,
        cache_name="native/prepared-cache", publish=in_place)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    markers = {lead: _marker(lead) for lead in LEADS}
    records = {lead: _decoded(lead) for lead in LEADS}
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[seconds[k], seconds[k + 1]]
                         for k in range(len(times) - 1)],
            "fields": frames.inventory},
        proof_head={"schema": "test"}, forcing=frames,
        as_posted={
            "input_plan": plan,
            "start_markers": {lead: markers[lead] for lead in LEADS[:2]},
            "forcing_leads": list(LEADS),
            "seal_authored_proof_keys": ("decoder_sha256",
                                         "input_manifest_sha256",
                                         "source_inputs"),
            "manifest_path": "source-input-manifest.json",
            "lead_role_prefix": "hrrr-f",
            "derived_roles": ("bridge", "source_manifest"),
            "document_bound_identity_keys": DOCUMENT_KEYS,
            "documents": {
                "bridge": {"path": BRIDGE, "lead_rows": "decoded_leads",
                           "fixed_rows": FIXED},
                "source_manifest": {"path": SOURCE,
                                    "lead_rows": "posted_objects"}},
        })
    writer.bind_posted_leads(markers)
    writer.bind_decoded_leads(records)
    for index in range(1, len(LEADS)):
        frames.add_snapshot(snapshots[index], index=index)
        writer.write_segment(index - 1, frames.interval(index - 1, times))
        frames.release(index - 1)
    sealed_records = seal_records or records
    if bridge_rows is None:
        bridge_rows = {name: digest for record in sealed_records.values()
                       for name, digest in record.items()}
        bridge_rows.update({name: _sha(name) for name in FIXED})
    if source_rows is None:
        source_rows = {item["name"]: item["sha256"]
                       for lead in LEADS for item in _objects(lead)}
    # The seal writes its documents as LF bytes and binds the digest of the
    # file it wrote (seal_hrrr_posted_inputs, _seal_posted_bridge).  A
    # text-mode write here put CRLF on disk on Windows while the proof named
    # the LF text's digest, so every seal below was refused there.
    for relative, rows in ((BRIDGE, bridge_rows), (SOURCE, source_rows)):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(_sums(rows).encode("utf-8"))
    bridge_sha = _sha((root / BRIDGE).read_bytes())
    source_sha = _sha((root / SOURCE).read_bytes())
    manifest = _plan_manifest()
    manifest["files"]["bridge"]["sha256"] = bridge_sha
    manifest["files"]["source_manifest"]["sha256"] = source_sha
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (root / "source-input-manifest.json").write_bytes(text.encode("utf-8"))
    manifest_sha = _sha((root / "source-input-manifest.json").read_bytes())
    writer.write_posted_leads(markers, route_table_sha256=ROUTE_TABLE,
                              decoded=sealed_records)
    receipt = writer.seal_cache(
        identity=sealed_identity or {
            "bridge_manifest_sha256": bridge_sha,
            "source_manifest_sha256": source_sha,
            "static_cache_sha256": "e" * 64},
        manifest_sha256=manifest_sha,
        document_sha256=document_sha256 or {
            "bridge_manifest_sha256": bridge_sha,
            "source_manifest_sha256": source_sha})
    writer.publish({"schema": "test", "input_manifest_sha256": manifest_sha,
                    "decoder_sha256": bridge_sha,
                    "source_inputs": {"files": manifest["files"]},
                    "prepared_cache": {"content_sha256":
                                       receipt["content_sha256"]},
                    "posting": {"as_posted": True, "waits": [],
                                "leads_late": []},
                    "boundary_stream": writer.boundary_stream_proof()})
    return root, {"bridge": bridge_sha, "source": source_sha,
                  "manifest": manifest_sha}


# -- the identity rule ---------------------------------------------------------

def test_a_document_bound_key_changes_only_to_its_documents_digest():
    plan_sha = "1" * 64
    placeholder = boundary_stream.as_posted_placeholder(plan_sha)
    head = {"bridge_manifest_sha256": placeholder,
            "source_manifest_sha256": placeholder, "static": "s"}
    sealed = {"bridge_manifest_sha256": "b" * 64,
              "source_manifest_sha256": "c" * 64, "static": "s"}
    documents = {"bridge_manifest_sha256": "b" * 64,
                 "source_manifest_sha256": "c" * 64}
    changed = boundary_stream.check_as_posted_identity(
        head, sealed, plan_sha256=plan_sha, manifest_sha256="a" * 64,
        document_bound=documents)
    assert sorted(changed) == ["bridge_manifest_sha256",
                               "source_manifest_sha256"]
    with pytest.raises(BoundaryStreamError, match="not the document"):
        boundary_stream.check_as_posted_identity(
            head, sealed, plan_sha256=plan_sha, manifest_sha256="a" * 64,
            document_bound={**documents, "bridge_manifest_sha256": "f" * 64})
    # Without the rule the L3 ruling's keys are all that may change, so a
    # route that never named its documents is refused as before.
    with pytest.raises(BoundaryStreamError, match="only a manifest"):
        boundary_stream.check_as_posted_identity(
            head, sealed, plan_sha256=plan_sha, manifest_sha256="a" * 64)


def test_a_native_head_binds_the_plan_and_its_seal_writes_both_documents(
        tmp_path):
    root, digests = _native_posted(tmp_path)
    head = boundary_stream.read_head(root)
    posted = head["basis"]["as_posted"]
    placeholder = boundary_stream.as_posted_placeholder(
        posted["input_plan_sha256"])
    assert head["basis"]["input_manifest_sha256"] is None
    assert head["basis"]["cache"]["identity"]["bridge_manifest_sha256"] \
        == placeholder
    assert posted["document_bound_identity_keys"] == DOCUMENT_KEYS
    assert posted["decoded_rows"] is True
    # The head waited for the start needs, f(S) and f(S+1), and binds both.
    assert sorted(posted["start_marker_sha256"]) == ["6", "7"]
    sealed = boundary_stream.verify_seal(root, head=head)
    assert sealed["as_posted"]["input_manifest_sha256"] == digests["manifest"]
    assert sorted(sealed["as_posted"]["identity_changed"]) == [
        "bridge_manifest_sha256", "source_manifest_sha256"]
    for k in (0, 1):
        segment = json.loads(boundary_stream.segment_marker_path(
            root, k).read_text())
        assert segment["decoded_leads"] == {
            str(lead): boundary_stream.decoded_lead_record_sha256(
                _decoded(lead)) for lead in LEADS[k:k + 2]}
    # The source manifest is the fetch's SHA256SUMS of these objects.
    assert (root / SOURCE).read_text() == _sums(
        {item["name"]: item["sha256"] for lead in LEADS
         for item in _objects(lead)})


def test_a_head_without_documents_keeps_its_block(tmp_path):
    """A GFS or mapped head names no document, so its block and digest stay."""

    from test_posted_preparation import _as_posted_tree

    _writer, output, _digest = _as_posted_tree(tmp_path)
    posted = boundary_stream.read_head(output)["basis"]["as_posted"]
    assert not {"documents", "decoded_rows",
                "document_bound_identity_keys"} & set(posted)
    segment = json.loads(boundary_stream.segment_marker_path(
        output, 0).read_text())
    assert "decoded_leads" not in segment


def test_a_bridge_row_that_is_not_its_leads_decode_is_refused(tmp_path):
    rows = {name: digest for lead in LEADS
            for name, digest in _decoded(lead).items()}
    rows.update({name: _sha(name) for name in FIXED})
    rows["./atmosphere-f07/TT.f32le"] = "0" * 64
    root, _ = _native_posted(tmp_path, bridge_rows=rows)
    with pytest.raises(BoundaryStreamError,
                       match="files each lead's decode recorded"):
        boundary_stream.verify_seal(root, head=boundary_stream.read_head(root))


def test_a_bridge_without_its_fixed_rows_is_refused(tmp_path):
    rows = {name: digest for lead in LEADS
            for name, digest in _decoded(lead).items()}
    root, _ = _native_posted(tmp_path, bridge_rows=rows)
    with pytest.raises(BoundaryStreamError, match="fixed rows"):
        boundary_stream.verify_seal(root, head=boundary_stream.read_head(root))


def test_a_source_row_no_marker_named_is_refused(tmp_path):
    rows = {item["name"]: item["sha256"] for lead in LEADS
            for item in _objects(lead)}
    rows["hrrr.t12z.wrfnatf09.grib2"] = "9" * 64
    root, _ = _native_posted(tmp_path, source_rows=rows)
    with pytest.raises(BoundaryStreamError,
                       match="objects the leads' posted markers named"):
        boundary_stream.verify_seal(root, head=boundary_stream.read_head(root))


def test_a_segment_built_from_another_decode_is_refused(tmp_path):
    """The seal's record of a lead's decode is the one its segments bound."""

    records = {lead: _decoded(lead, salt="x" if lead == 8 else "")
               for lead in LEADS}
    root, _ = _native_posted(tmp_path, seal_records=records)
    with pytest.raises(BoundaryStreamError, match="built from decoded leads"):
        boundary_stream.verify_seal(root, head=boundary_stream.read_head(root))


def test_the_seal_writes_only_each_documents_own_digest(tmp_path):
    with pytest.raises(BoundaryStreamError, match="not the document"):
        _native_posted(tmp_path, document_sha256={
            "bridge_manifest_sha256": "f" * 64,
            "source_manifest_sha256": "c" * 64})
    with pytest.raises(RuntimeError, match="every document"):
        _native_posted(tmp_path / "other", document_sha256={
            "bridge_manifest_sha256": "f" * 64})


# -- the admission ---------------------------------------------------------------

def _fetch_folder(tmp_path, *, leads=LEADS, posted=LEADS):
    out = tmp_path / "data"
    folder = out / "posting"
    folder.mkdir(parents=True)
    (folder / "schedule.json").write_text(json.dumps({
        "schema": "gpuwm.posting-schedule.v1", "source": "hrrr",
        "cycle": CYCLE, "table_sha256": ROUTE_TABLE,
        "leads": [{"lead": lead, "valid_time": None, "expected_at": None,
                   "late_at": None, "state": "scheduled"}
                  for lead in leads]}), encoding="utf-8")
    for lead in leads:
        (out / f"hrrr.t12z.wrfnatf{lead:02d}.grib2").write_text(f"nat{lead}")
        (out / f"hrrr.t12z.soilf{lead:02d}.grib2").write_text(f"soil{lead}")
    for lead in posted:
        (folder / f"f{lead:03d}.json").write_text(
            json.dumps(_marker(lead)), encoding="utf-8")
    series = tmp_path / "series.tsv"
    series.write_text("".join(
        f"{lead}\t{out / f'hrrr.t12z.wrfnatf{lead:02d}.grib2'}\t"
        f"{out / f'hrrr.t12z.soilf{lead:02d}.grib2'}\n" for lead in leads),
        encoding="utf-8")
    return out, folder, series


def _admitter(tmp_path, folder, series):
    from tools.hrrr_pipeline import PostedLeadAdmitter

    posted = PostedLeads(folder, source="hrrr", cycle=CYCLE,
                         poll_seconds=0.01)
    return PostedLeadAdmitter(posted=posted, series=series,
                              admissions=tmp_path / "admitted")


def test_a_lead_is_admitted_for_its_series_files_once_held_to_its_marker(
        tmp_path):
    out, folder, series = _fetch_folder(tmp_path)
    admitter = _admitter(tmp_path, folder, series)
    admitter.admit(6)
    # Exactly what hrrr_grib2_bridge --series-workers-posted compares.
    assert (tmp_path / "admitted" / "f06.admitted").read_text() == (
        "forecast_hour\t6\n"
        f"atmosphere\t{out / 'hrrr.t12z.wrfnatf06.grib2'}\n"
        f"soil\t{out / 'hrrr.t12z.soilf06.grib2'}\n")
    assert admitter.markers[6] == _marker(6)
    assert not (tmp_path / "admitted" / "f07.admitted").exists()


def test_a_file_that_is_not_its_markers_object_is_never_admitted(tmp_path):
    out, folder, series = _fetch_folder(tmp_path)
    (out / "hrrr.t12z.soilf07.grib2").write_text("SOIL")
    admitter = _admitter(tmp_path, folder, series)
    with pytest.raises(ValueError, match="not the soil object"):
        admitter.admit(7)
    assert not (tmp_path / "admitted" / "f07.admitted").exists()


def test_the_admitter_runs_ahead_and_a_late_lead_ends_it_by_name(tmp_path):
    out, folder, series = _fetch_folder(tmp_path, posted=(6, 7))
    (folder / "failed.json").write_text(json.dumps({
        "code": "source_behind", "source": "hrrr", "cycle": CYCLE,
        "lead": 8, "expected_at": "2026-09-30T13:00:00Z",
        "late_at": "2026-09-30T14:00:00Z", "heard": True}))
    admitter = _admitter(tmp_path, folder, series)
    admitter.start()
    assert admitter.wait(6) == _marker(6)
    assert admitter.wait(7) == _marker(7)
    with pytest.raises(SourceBehind):
        admitter.wait(8)


def test_the_source_manifest_is_the_fetchs_sha256sums_of_the_window(tmp_path):
    out, folder, series = _fetch_folder(tmp_path)
    admitter = _admitter(tmp_path, folder, series)
    for lead in LEADS:
        admitter.admit(lead)
    text = admitter.source_manifest_text()
    # The legacy fetch's own format (fetch.py publish_manifest).
    assert text == "".join(
        f"{_sha(path.read_bytes())}  {path.name}\n"
        for path in sorted(out.glob("hrrr.*.grib2"), key=lambda p: p.name))
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(text)
    receipt = admitter.receipt(manifest)
    assert receipt["mode"] == "as_posted"
    assert receipt["forecast_hours"] == list(LEADS)
    assert receipt["payload_file_count"] == 2 * len(LEADS)


def test_a_donor_series_is_not_admitted_as_posted(tmp_path):
    out, folder, series = _fetch_folder(tmp_path)
    series.write_text(series.read_text().replace(
        "\n", f"\tPMSL={out / 'donor.grib2'}\n"))
    with pytest.raises(ValueError, match="nothing else"):
        _admitter(tmp_path, folder, series)


def test_the_producer_starts_the_posted_decoder_mode(tmp_path):
    from tools.hrrr_pipeline import HrrrPipelineProducer

    _out, _folder, series = _fetch_folder(tmp_path)
    producer = HrrrPipelineProducer(
        decoder=tmp_path / "decoder", series=series,
        output=tmp_path / "bridge", signals=tmp_path / "signals",
        cycle="2026-09-30 12:00:00", window=(0, 1, 0, 1), workers="2",
        log=tmp_path / "decoder.log", admissions=tmp_path / "admitted")
    assert producer.admissions == tmp_path / "admitted"
    # The command start() launches (decoder_argv): the posted mode, its
    # admission folder in the decoder's ADMIT_DIR slot.
    argv = producer.decoder_argv()
    assert argv[1] == "--series-workers-posted"
    assert argv[6] == str(tmp_path / "admitted")
    import inspect
    from tools import hrrr_pipeline

    source = inspect.getsource(hrrr_pipeline.HrrrPipelineProducer.start)
    assert "self.argv = self.decoder_argv()" in source


def test_a_decoder_that_failed_ends_the_wait_for_a_lead_not_posted(tmp_path):
    """The wait for a lead looks at the decoder, so its failure is said now.

    The decoder exits at once on a series it cannot read; before, the
    preparation sat out the lead's whole lateness budget before saying so.
    """

    from tools.hrrr_single_domain_benchmark import _await_admitted

    _out, folder, series = _fetch_folder(tmp_path, posted=(6,))
    admitter = _admitter(tmp_path, folder, series)
    admitter.start()

    class Decoder:
        failed = False
        checks = 0

        def check(self):
            self.checks += 1
            if self.failed:
                raise RuntimeError("pipeline producer exited 1")

    decoder = Decoder()
    # f007 never posts, so the admitter's own thread would wait for it for
    # the life of the process; stop() ends that wait and joins the thread,
    # whatever the assertions below find.
    try:
        assert _await_admitted(admitter, decoder, 6, poll_seconds=0.01)             == _marker(6)
        decoder.failed = True
        checks = decoder.checks
        # f007 has no marker and none is coming: only the decoder ends the wait.
        with pytest.raises(RuntimeError, match="exited 1"):
            _await_admitted(admitter, decoder, 7, poll_seconds=0.01)
        assert decoder.checks == checks + 1
    finally:
        admitter.stop()
    # Stopped means stopped: no thread is left reading the posting folder.
    assert not admitter._thread.is_alive()
    with pytest.raises(BoundaryStreamError, match="stopped by its owner"):
        admitter.wait(7)


def test_built_intervals_are_written_while_the_next_lead_is_awaited(tmp_path):
    """The build loop's collection runs between waits for a lead not posted.

    Collected only when a slot came round again, interval k waited for lead
    k + slots instead of k + 1 (test host 4, HRRR 2026-10-01T08).
    """

    from tools.hrrr_single_domain_benchmark import _await_admitted

    _out, folder, series = _fetch_folder(tmp_path, posted=(6, 7))
    admitter = _admitter(tmp_path, folder, series)
    admitter.start()

    class Decoder:
        def check(self):
            pass

    collected = []

    def between():
        collected.append(len(collected))
        if len(collected) == 3:
            # The lead posts while the loop is waiting for it.
            (folder / "f008.json").write_text(json.dumps(_marker(8)),
                                              encoding="utf-8")

    try:
        assert _await_admitted(admitter, Decoder(), 8, poll_seconds=0.01,
                               between=between) == _marker(8)
        assert len(collected) >= 3
    finally:
        admitter.stop()
    source = Path(__file__).resolve().parents[1] / "tools"         / "hrrr_single_domain_benchmark.py"
    text = source.read_text(encoding="utf-8")
    # The posted build loop awaits each lead with the collection between.
    assert "between=collect_finished)" in text


# -- the bundle ---------------------------------------------------------------

def test_an_as_posted_bundle_publishes_the_one_shot_manifest_and_proof(
        tmp_path):
    """The head leaves out what names the two documents; the seal writes them.

    The same synthetic preparation published one-shot and as posted gives a
    byte-equal source-input-manifest.json and an equal proof.
    """

    import shutil

    from woof import hrrr_prepared_bundle as bundle
    from test_hrrr_prepared_background import _hrrr_bundle

    one_shot = _hrrr_bundle(tmp_path / "one-shot")
    root = one_shot.root
    posted_root = tmp_path / "posted" / "prepared"
    shutil.copytree(root, posted_root)
    for name in ("source-input-manifest.json", "proof.json",
                 "namelist.wps", "namelist.input"):
        (posted_root / name).unlink()
    proof = json.loads((root / "proof.json").read_text())
    published = json.loads((root / "source-input-manifest.json").read_text())
    header = json.loads(
        (posted_root / "native" / "prepared-cache" / "header.json").read_text())
    source_manifest = tmp_path / "one-shot" / "SHA256SUMS"
    posted_manifest = tmp_path / "posted" / "SHA256SUMS"
    shutil.copyfile(source_manifest, posted_manifest)
    head = bundle.publish_hrrr_bundle_head(
        output_root=posted_root,
        prepared_cache=posted_root / "native" / "prepared-cache",
        static_cache=posted_root / "native-static.npz",
        static_receipt=posted_root / "native-static-receipt.json",
        geometry_receipt=posted_root / "native-geometry-receipt.json",
        bridge_manifest=posted_root / "native" / "native-bridge" / "SHA256SUMS",
        namelist_input=tmp_path / "one-shot" / "namelist.input",
        wps_namelist=tmp_path / "one-shot" / "namelist.wps",
        source_manifest=posted_manifest,
        experiment_config=posted_root / "experiment.toml",
        source_cycle=bundle_cycle(proof),
        source_forecast_hours=proof["source_forecast_hours"],
        model_forcing_hours=proof["model_forcing_hours"],
        preprocessing=proof["preprocessing"],
        source_identity=header["identity"]["source_identity"],
        physics_profile=one_shot.handoff["physics_profile"],
        cache_user_metadata=header["metadata"]["user"], as_posted=True)
    assert not (posted_root / "source-input-manifest.json").exists()
    assert not set(bundle.AS_POSTED_SEAL_KEYS) & set(head["proof_head"])
    assert head["manifest"]["files"]["bridge"]["sha256"] is None
    keys = bundle.seal_hrrr_posted_inputs(
        head, output_root=posted_root,
        bridge_manifest=posted_root / "native" / "native-bridge"
        / "SHA256SUMS", source_manifest=posted_manifest)
    assert (posted_root / "source-input-manifest.json").read_bytes() \
        == (root / "source-input-manifest.json").read_bytes()
    assert json.loads(json.dumps(keys)) == {
        key: proof[key] for key in bundle.AS_POSTED_SEAL_KEYS}
    sealed = bundle.seal_hrrr_bundle_proof(
        head, output_root=posted_root,
        prepared_cache=posted_root / "native" / "prepared-cache",
        static_cache=posted_root / "native-static.npz",
        geometry_receipt=posted_root / "native-geometry-receipt.json")
    sealed.update(keys)
    expected = dict(proof)
    # The header digest names the cache's own path-free bytes; both arms
    # restore the same cache, so the whole proof is equal.
    assert json.loads(json.dumps(sealed, sort_keys=True)) \
        == json.loads(json.dumps(expected, sort_keys=True))
    assert head["handoff"]["source_manifest_sha256"] \
        == _sha((root / "source-input-manifest.json").read_bytes())
    assert published["files"]["bridge"]["sha256"] == proof["decoder_sha256"]
    with pytest.raises(bundle.HrrrBundleError, match="as-posted"):
        bundle.seal_hrrr_posted_inputs(
            {**head, "as_posted": False}, output_root=posted_root,
            bridge_manifest=posted_manifest, source_manifest=posted_manifest)


def bundle_cycle(proof):
    from datetime import datetime

    return datetime.fromisoformat(proof["source_cycle"])


# -- the doors ----------------------------------------------------------------

def _wrapper_args(**extra):
    from tools.prepare_hrrr_wrf import _parser

    argv = ["--source-root", "data", "--static-cache", "s.npz",
            "--namelist-input", "namelist.input", "--output-root", "out",
            "--cycle", "2026-09-30_12:00:00"]
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return _parser().parse_args(argv)


def test_the_wrapper_reads_its_leads_as_posted_or_binds_a_manifest():
    from tools.prepare_hrrr_wrf import _as_posted_refusal

    assert _as_posted_refusal(_wrapper_args(as_posted="data/posting")) is None
    assert _as_posted_refusal(_wrapper_args(
        source_manifest="SHA256SUMS", source_manifest_sha256="a" * 64)) is None
    assert "required unless --as-posted" in _as_posted_refusal(
        _wrapper_args())
    assert "second, different manifest" in _as_posted_refusal(_wrapper_args(
        as_posted="data/posting", source_manifest="SHA256SUMS",
        source_manifest_sha256="a" * 64))
    assert "donor" in _as_posted_refusal(_wrapper_args(
        as_posted="data/posting", supplement="PMSL=donor.grib2"))
    assert "prefix-sealed" in _as_posted_refusal(_wrapper_args(
        as_posted="data/posting", sealed_prepared_cache=True))


def test_the_benchmark_refuses_a_digest_of_a_manifest_not_written_yet(
        tmp_path):
    from tools.hrrr_single_domain_benchmark import _parse_args

    base = ["--bridge", str(tmp_path / "bridge"), "--cycle",
            "2026-09-30_12:00:00", "--pipeline-series", "s.tsv",
            "--pipeline-decoder", "d", "--pipeline-signals",
            str(tmp_path / "signals"), "--source-root", str(tmp_path),
            "--static-cache", "s", "--static-receipt", "r",
            "--namelist-input", "n", "--prepared-cache",
            str(tmp_path / "cache"), "--prepare-only", "--run-seconds",
            "3600", "--history-interval-seconds", "3600", "--outdir",
            str(tmp_path / "report"), "--as-posted", str(tmp_path / "posting")]
    args = _parse_args(base + ["--source-manifest",
                               str(tmp_path / "new" / "SHA256SUMS")])
    assert args.as_posted == tmp_path / "posting"
    with pytest.raises(SystemExit):
        _parse_args(base + ["--source-manifest", str(tmp_path / "x"),
                            "--source-manifest-sha256", "a" * 64])
    existing = tmp_path / "SHA256SUMS"
    existing.write_text("")
    with pytest.raises(SystemExit):
        _parse_args(base + ["--source-manifest", str(existing)])


@pytest.mark.parametrize("domains", [1, 2, 4])
def test_the_door_fetches_beside_native_domains_unless_told_otherwise(
        capsys, monkeypatch, domains):
    from types import SimpleNamespace

    from woof.runplan import _native_fetch_beside

    plan = SimpleNamespace(run_options={})
    exp = SimpleNamespace(domains=tuple(object() for _ in range(domains)))
    observer = SimpleNamespace(enter_stage=lambda *a, **k: None)
    kwargs = dict(data_dir=Path("data"), run_dir=Path("run"),
                  observer=observer)
    hints = {"cycle": "2026-09-30T12", "source": "hrrr"}
    assert _native_fetch_beside(plan, {**hints, "as_posted": False}, exp,
                                prepare_only=False, **kwargs) is None
    assert _native_fetch_beside(plan, hints, exp, prepare_only=True,
                                **kwargs) is None
    assert capsys.readouterr().out == ""
    class Fetch:
        def __init__(self, arguments, **options):
            self.arguments = arguments
            self.posting = Path("data/posting")

        def await_schedule(self):
            return self.posting

    monkeypatch.setattr("woof.runplan._NativeFetchBeside", Fetch)
    beside = _native_fetch_beside(plan, hints, exp, prepare_only=False,
                                  **kwargs)
    assert beside.posting == Path("data/posting")
    assert "--whole-cycle" not in beside.arguments
    assert "fetch runs beside the preparation" in capsys.readouterr().out
    # The donor's document still has to bind it before any hour decodes.
    donors = SimpleNamespace(run_options={"supplement": ["PMSL=x.grib2"]})
    assert _native_fetch_beside(donors, hints, exp, prepare_only=False,
                                **kwargs) is None
    assert "PMSL donor" in capsys.readouterr().out
