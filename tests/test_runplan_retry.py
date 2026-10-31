"""Getting back to a working forecast without re-fetching the forcing.

A run that fails after its data is on disk -- at planning, at the
forecast stage, anywhere past the fetch -- used to have no way back
except a fresh run directory and the gigabytes that come with it.
Every stage between the bytes and the failure is create-only, so
re-running the same plan into the same ``output_root`` met the
preparer's "refusing existing output root" and the forecast runner's
"already holds a run's output".

These tests drive the real seams: :func:`woof.runplan._prepare_stage`,
:func:`woof.runplan._clear_forecast_output`, the real
:mod:`woof.stage_reuse` decision, and the real ``_hrrr_chain`` twice
over one run directory.  The fetch half is proven against real fetched
bytes in ``test_a_second_pass_over_a_real_fetched_directory_downloads_
nothing``, which reads the payload a live HRRR fetch left behind rather
than a fixture written to agree with the code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import requires_cupy
from _reserved_drives import refuse_reserved

from woof import stage_reuse
from woof.runplan import (EVENTS_FILENAME, EventStream, RunObserver,
                           _clear_forecast_output, _fetch_transfer_split,
                           _prepare_stage, read_events)


# ---------------------------------------------------------------------------
# What the fetch stage now says it did
# ---------------------------------------------------------------------------


def test_a_fetch_that_skipped_everything_says_so_in_its_receipt():
    """The line a retry has to be able to show a user.

    Both passes move the same total payload_bytes, so the size alone
    cannot tell them apart.  Each row says which it was.
    """

    payload = {
        "files": [
            {"name": "a.grib2", "bytes": 1000, "downloaded": False,
             "seconds": 0.5},
            {"name": "b.grib2", "bytes": 2000, "downloaded": False,
             "seconds": 0.7},
            # The receipts a fetch writes beside its payload.  Nobody
            # waited for these, so they are neither a download nor a
            # skip.
            {"name": "SHA256SUMS", "bytes": 90},
        ],
    }
    split = _fetch_transfer_split(payload)

    assert split["downloaded_files"] == 0
    assert split["downloaded_bytes"] == 0
    assert split["verified_files"] == 2
    assert split["verified_bytes"] == 3000
    assert split["receipt_files"] == 1
    assert "skipped 2 file(s)" in split["summary"]
    assert "3,000 B not re-downloaded" in split["summary"]


def test_a_first_pass_is_reported_as_a_download_not_as_a_skip():
    payload = {"files": [
        {"name": "a.grib2", "bytes": 1000, "downloaded": True,
         "seconds": 9.0}]}
    split = _fetch_transfer_split(payload)

    assert split["downloaded_files"] == 1
    assert split["verified_files"] == 0
    assert split["summary"].startswith("downloaded 1 file(s)")


def test_a_second_fetch_into_a_filled_directory_downloads_nothing(
        tmp_path, monkeypatch):
    """THE gap this closed, driven through the real fetch function.

    Every HRRR manifest row omitted ``downloaded``, so a fetch that
    skipped three gigabytes and a fetch that pulled three gigabytes
    published the identical receipt.  The transfer here is faked -- the
    alternative is a live download -- but the verify-skip is not: the
    second pass walks the GRIB envelope of the bytes on disk, counts
    their records and re-hashes them, exactly as it does against NOAA.
    """

    from datetime import datetime

    import woof.fetch as fetch
    from test_fetch import _fake_hrrr_product
    from tools import download_hrrr_native_subset as hrrr_transport

    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "data"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None,
                  out=out)
    first = json.loads(
        fetch.fetch_hrrr(**kwargs, progress=lambda line: None).read_text())

    payload = [entry for entry in first["files"]
               if entry.get("forecast_hour") is not None]
    assert payload and all(entry["downloaded"] is True for entry in payload)
    assert _fetch_transfer_split(first)["downloaded_files"] == len(payload)

    def refuse(request, *, workers, retries):
        raise AssertionError("re-downloaded a complete, verified product")

    monkeypatch.setattr(hrrr_transport, "_download_product", refuse)
    lines: list[str] = []
    second = json.loads(
        fetch.fetch_hrrr(**kwargs, progress=lines.append).read_text())

    assert all("verified -- skipped" in line for line in lines)
    payload = [entry for entry in second["files"]
               if entry.get("forecast_hour") is not None]
    assert all(entry["downloaded"] is False for entry in payload)
    # Same bytes claimed by both receipts; only the split tells them
    # apart, which is the whole reason the key exists.
    assert second["payload_bytes"] == first["payload_bytes"]
    split = _fetch_transfer_split(second)
    assert split["downloaded_files"] == 0
    assert split["verified_files"] == len(payload)
    assert split["verified_bytes"] == sum(
        entry["bytes"] for entry in payload)
    assert split["summary"].startswith(f"skipped {len(payload)} file(s)")

    # And the read-back API agrees, which it could not before: it counts
    # this exact key.
    assert fetch.fetch_throughput(out)["downloaded_files"] == 0
    assert fetch.fetch_throughput(out)["verified_files"] >= len(payload)


# ---------------------------------------------------------------------------
# The prepared bundle: reuse when it is this run's, rebuild when it is not
# ---------------------------------------------------------------------------


def _bundle(root: Path, identity: dict) -> Path:
    """A prepared bundle's published identity, in its real layout."""

    cache = root / "native" / "prepared-cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "header.json").write_text(json.dumps({
        "schema": "gpuwm-prepared-real-cache-v1",
        "status": "READY",
        "content_sha256": "0" * 64,
        "identity": identity,
    }), encoding="utf-8")
    (root / "payload.bin").write_bytes(b"x" * 4096)
    return root


def _identity(**overrides) -> dict:
    identity = {
        "source_manifest_sha256": "a" * 64,
        "namelist_sha256": "b" * 64,
        "forcing_hours": [0, 1, 2],
        "source_identity": dict(stage_reuse.engine_source_identity()),
    }
    identity.update(overrides)
    return identity


def _identity_for(command: list[str]) -> dict:
    """The identity a real preparer would publish for this argv.

    Built from the command's own values rather than from constants, so
    the fixture cannot drift into agreeing with the code by accident:
    the preparer records the source-manifest digest it was handed and
    the digest of the namelist it read, and this records the same two.
    """

    namelist = Path(command[command.index("--namelist-input") + 1])
    return _identity(
        source_manifest_sha256=command[
            command.index("--source-manifest-sha256") + 1],
        namelist_sha256=stage_reuse._sha256(namelist))


def test_an_untouched_run_directory_builds(tmp_path):
    calls = []
    decision = _prepare_stage(
        tmp_path / "prep", arguments=["--x", "1"], stated={},
        run=lambda: calls.append("ran"))

    assert decision["decision"] == stage_reuse.BUILD
    assert calls == ["ran"]
    assert decision["reason"] == "nothing is prepared here yet"


def test_a_bundle_this_run_would_rebuild_identically_is_reused(tmp_path):
    """The whole point: the second pass spends nothing.

    The first pass writes the binding; the second reads it, compares the
    arguments, the stated digests and this engine's own source identity,
    and never calls the preparer.
    """

    root = tmp_path / "prep"
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("digest  a.grib2\n", encoding="utf-8")
    arguments = ["--source-manifest", str(manifest), "--cycle",
                 "2024-05-03_12:00:00"]
    stated = {"source_manifest_sha256": "a" * 64}

    calls = []

    def prepare():
        _bundle(root, _identity())
        calls.append("ran")

    first = _prepare_stage(root, arguments=arguments, stated=stated,
                           run=prepare)
    assert first["decision"] == stage_reuse.BUILD
    assert calls == ["ran"]
    assert (root / stage_reuse.BINDING_NAME).is_file()

    second = _prepare_stage(root, arguments=arguments, stated=stated,
                            run=prepare)

    assert second["decision"] == stage_reuse.REUSE
    assert calls == ["ran"], "the preparer ran a second time"
    assert "same arguments" in second["reason"]
    assert "source_manifest_sha256" in second["compared"]


def test_a_bundle_built_from_different_bytes_is_rebuilt_and_named(tmp_path):
    root = tmp_path / "prep"
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("digest  a.grib2\n", encoding="utf-8")
    arguments = ["--source-manifest", str(manifest)]

    calls = []

    def prepare():
        _bundle(root, _identity())
        calls.append("ran")

    _prepare_stage(root, arguments=arguments,
                   stated={"source_manifest_sha256": "a" * 64}, run=prepare)
    # The forcing on disk changed underneath: a different SHA256SUMS is
    # a different fetch.
    manifest.write_text("other  a.grib2\n", encoding="utf-8")
    second = _prepare_stage(root, arguments=arguments,
                            stated={"source_manifest_sha256": "c" * 64},
                            run=prepare)

    assert second["decision"] == stage_reuse.REBUILD
    assert calls == ["ran", "ran"]
    fields = [item["field"] for item in second["differences"]]
    assert "source_manifest_sha256" in fields
    assert "arguments --source-manifest" in fields
    assert "source_manifest_sha256" in second["reason"]


def test_a_bundle_prepared_by_a_different_engine_is_rebuilt(tmp_path):
    """The engine moved, so the bundle is stale by definition.

    This is not pedantry about provenance.  The forecast runner
    re-derives the same identity and REFUSES a cache whose source
    identity has moved, so reusing one here would trade a rebuild that
    costs tens of seconds for a refusal that arrives after the
    preparation stage has already been reported as finished.
    """

    root = tmp_path / "prep"
    arguments = ["--cycle", "2024-05-03_12:00:00"]

    def prepare():
        _bundle(root, _identity(
            source_identity={"identity_source": "git",
                             "git_commit": "0" * 40,
                             "git_tree": "1" * 40,
                             "git_status_short": []}))

    _prepare_stage(root, arguments=arguments, stated={}, run=prepare)
    decision = _prepare_stage(root, arguments=arguments, stated={},
                              run=prepare)

    assert decision["decision"] == stage_reuse.REBUILD
    fields = [item["field"] for item in decision["differences"]]
    assert any(field.startswith("source_identity.") for field in fields)
    assert any("the engine has changed" in (item.get("note") or "")
               for item in decision["differences"])


def test_a_rebuild_moves_the_old_bundle_aside_and_deletes_nothing(tmp_path):
    root = tmp_path / "prep"
    arguments = ["--cycle", "2024-05-03_12:00:00"]

    def prepare():
        _bundle(root, _identity())

    _prepare_stage(root, arguments=arguments, stated={}, run=prepare)
    decision = _prepare_stage(
        root, arguments=["--cycle", "2024-05-04_12:00:00"], stated={},
        run=prepare)

    assert decision["decision"] == stage_reuse.REBUILD
    superseded = Path(decision["superseded"]["path"])
    assert superseded.is_dir()
    assert (superseded / "payload.bin").read_bytes() == b"x" * 4096
    assert decision["superseded"]["bytes"] >= 4096
    assert "nothing was deleted" in decision["superseded"]["note"]
    # And the rebuild landed where the chain expects it, not beside it.
    assert (root / "native" / "prepared-cache" / "header.json").is_file()


#: What a REAL preparation publishes as ``source_identity``: a
#: DATA-source block.  Trimmed from ``woof/gfs_direct.py:1497-1518``,
#: which is the only shipped adapter that records a git state at all --
#: and it nests it under ``git_source_identity``, so not one member of
#: ``SOURCE_IDENTITY_KEYS`` appears at the top level here.  ERA5 and the
#: mapped route carry no git block whatsoever.
GFS_SOURCE_IDENTITY = {
    "adapter": "gfs-pgrb2-0p25-direct-v1",
    "input_manifest_schema": "gpuwm-portable-input-manifest-v1",
    "input_manifest_sha256": "a" * 64,
    "initial_condition": {"cycle": "2024-05-03T12:00:00Z", "lead_hours": 0},
    "decoder": {"name": "gfs_grib2_bridge", "sha256": "b" * 64,
                "implementation": "gpuwm-all-rust-gfs-grib2-bridge"},
    "implementation_sha256": "c" * 64,
    "git_source_identity": {"available": True, "commit": "0" * 40,
                            "tree": "1" * 40},
    "preprocessing": {"backend": "cuda"},
}


def _binding(root: Path, arguments: list, *, engine) -> Path:
    """A ``stage-binding.json`` recording ``engine``, written by hand.

    By hand because the point of two of the tests below is a binding this
    engine did NOT write: one from a release that recorded no engine at
    all, and one from an engine that has since moved.
    """

    payload = {"schema": stage_reuse.BINDING_SCHEMA,
               "created": "2024-05-03T12:00:00Z",
               "arguments": stage_reuse.argument_binding(arguments),
               "stated": {}}
    if engine is not None:
        payload["engine"] = engine
    path = root / stage_reuse.BINDING_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def test_a_bundle_a_different_engine_prepared_is_not_reused_as_this_one(
        tmp_path):
    """REUSE says "by this same engine".  Nothing here said which engine.

    The reachable case, not an edge one: ``SOURCE_IDENTITY_KEYS`` is the
    shape of ``runtime_manifest.provenance`` and every adapter's
    ``source_identity`` is a DATA-source block, so a comparison that
    skipped each member it could not find skipped all six on every real
    bundle.  A bundle a DIFFERENT engine prepared then reused clean, and
    the forecast integrated new dynamics over initial and boundary
    conditions built by the old ingest code, under a receipt asserting
    the engine matched.
    """

    root = _bundle(tmp_path / "prep", _identity(
        source_identity=GFS_SOURCE_IDENTITY))
    _binding(root, ["--cycle", "2024-05-03_12:00:00"],
             engine={**stage_reuse.engine_source_identity(),
                     "git_commit": "9" * 40, "git_tree": "8" * 40})

    decision = stage_reuse.decide(
        root, stated={}, arguments=["--cycle", "2024-05-03_12:00:00"])

    assert decision["decision"] == stage_reuse.REBUILD
    assert "source_identity.git_commit" in [
        item["field"] for item in decision["differences"]]
    assert any("the engine has changed" in (item.get("note") or "")
               for item in decision["differences"])


def test_a_bundle_that_cannot_say_what_built_it_is_rebuilt(tmp_path):
    """The same rule ``STATEABLE`` already applies, on the other block.

    A binding written before the engine was recorded, beside a real
    adapter's identity, leaves the question "which code built this?"
    unanswerable.  That is the same class of ignorance as an install
    that cannot state its own identity, which this function has always
    turned into a rebuild -- the asymmetry was the defect.
    """

    root = _bundle(tmp_path / "prep", _identity(
        source_identity=GFS_SOURCE_IDENTITY))
    _binding(root, ["--cycle", "2024-05-03_12:00:00"], engine=None)

    decision = stage_reuse.decide(
        root, stated={}, arguments=["--cycle", "2024-05-03_12:00:00"])

    assert decision["decision"] == stage_reuse.REBUILD
    assert any(item.get("recorded") is None
               and "records no such member" in (item.get("note") or "")
               for item in decision["differences"])


def test_the_binding_records_the_engine_that_ran_the_stage(tmp_path):
    """And the whole answer rests on that, so it is asserted directly.

    Nothing a preparer publishes records the engine, so the binding is
    the only artifact that can; a bundle whose binding names this engine
    is still reused, which is what keeps the recovery seam a recovery
    seam rather than an unconditional rebuild.
    """

    root = tmp_path / "prep"
    arguments = ["--cycle", "2024-05-03_12:00:00"]
    calls = []

    def prepare():
        _bundle(root, _identity(source_identity=GFS_SOURCE_IDENTITY))
        calls.append("ran")

    _prepare_stage(root, arguments=arguments, stated={}, run=prepare)
    payload = json.loads(
        (root / stage_reuse.BINDING_NAME).read_text(encoding="utf-8"))
    assert payload["engine"] == stage_reuse.engine_source_identity()
    assert payload["engine"], "this install cannot state its own identity"

    second = _prepare_stage(root, arguments=arguments, stated={},
                            run=prepare)
    assert second["decision"] == stage_reuse.REUSE
    assert calls == ["ran"], "the preparer ran a second time"


def test_a_bundle_with_no_binding_is_rebuilt_rather_than_trusted(tmp_path):
    """A bundle prepared before this seam existed states no arguments.

    Rebuilding it is the cheap, correct answer -- the forcing beside it
    is still reused, which is where the minutes and the gigabytes are.
    """

    root = _bundle(tmp_path / "prep", _identity())
    calls = []
    decision = _prepare_stage(root, arguments=["--cycle", "x"], stated={},
                              run=lambda: calls.append("ran"))

    assert decision["decision"] == stage_reuse.REBUILD
    assert calls == ["ran"]
    assert stage_reuse.BINDING_NAME in decision["reason"]


def test_a_half_written_bundle_is_rebuilt_rather_than_trusted(tmp_path):
    root = tmp_path / "prep"
    cache = root / "native" / "prepared-cache"
    cache.mkdir(parents=True)
    (cache / "header.json").write_text(json.dumps({
        "schema": "gpuwm-prepared-real-cache-v1",
        "status": "WRITING", "identity": _identity()}), encoding="utf-8")
    stage_reuse.write_binding(root, arguments=["--cycle", "x"])

    decision = _prepare_stage(root, arguments=["--cycle", "x"], stated={},
                              run=lambda: None)

    assert decision["decision"] == stage_reuse.REBUILD
    assert "did not finish" in decision["reason"]


def test_a_domain_tree_is_rebuilt_because_one_identity_cannot_speak_for_all(
        tmp_path):
    root = tmp_path / "prep"
    for domain in ("d01", "d02"):
        cache = root / domain / "prepared-cache"
        cache.mkdir(parents=True)
        (cache / "header.json").write_text(json.dumps({
            "schema": "gpuwm-prepared-real-cache-v1", "status": "READY",
            "identity": _identity()}), encoding="utf-8")
    stage_reuse.write_binding(root, arguments=["--cycle", "x"])

    decision = _prepare_stage(root, arguments=["--cycle", "x"], stated={},
                              run=lambda: None)

    assert decision["decision"] == stage_reuse.REBUILD
    assert "2 prepared-cache headers" in decision["reason"]


def test_an_argument_binding_survives_a_moved_run_directory(tmp_path):
    """A path is bound by the digest of the file it names, not by itself.

    Otherwise a run directory that moved -- or a data directory the user
    relocated -- would rebuild for a reason that has nothing to do with
    what the preparation would produce.
    """

    first = tmp_path / "one" / "SHA256SUMS"
    second = tmp_path / "two" / "SHA256SUMS"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_text("digest  a.grib2\n", encoding="utf-8")

    assert (stage_reuse.argument_binding(["--source-manifest", str(first)])
            == stage_reuse.argument_binding(
                ["--source-manifest", str(second)]))

    second.write_text("different\n", encoding="utf-8")
    assert (stage_reuse.argument_binding(["--source-manifest", str(first)])
            != stage_reuse.argument_binding(
                ["--source-manifest", str(second)]))


def test_repeated_supersedes_never_collide_and_never_nest(tmp_path):
    root = tmp_path / "prep"
    moved = []
    for _ in range(3):
        root.mkdir(parents=True, exist_ok=True)
        (root / "payload.bin").write_bytes(b"z")
        moved.append(stage_reuse.supersede(root)["path"])

    assert len(set(moved)) == 3
    for path in moved:
        assert Path(path).is_dir()
        assert stage_reuse.SUPERSEDED_MARK not in Path(path).parent.name


# ---------------------------------------------------------------------------
# The forecast stage's output directory
# ---------------------------------------------------------------------------


def _observer(tmp_path) -> tuple[RunObserver, EventStream, Path]:
    events_path = tmp_path / EVENTS_FILENAME
    events = EventStream(events_path, mirror=None)
    events.__enter__()
    return RunObserver(events, heartbeat=None, root_domain=1), events, \
        events_path


def test_an_existing_empty_forecast_directory_gets_a_distinct_claim(tmp_path):
    forecast = tmp_path / "run"
    forecast.mkdir()
    observer, events, _ = _observer(tmp_path)
    try:
        claimed = _clear_forecast_output(forecast, observer=observer)
        assert claimed != forecast and claimed.is_dir()
    finally:
        events.__exit__(None, None, None)
    assert forecast.is_dir()


def test_an_earlier_attempts_output_stays_at_its_address(tmp_path):
    """The runner refuses to merge two runs into one receipt, correctly.

    So the retry gets a clean directory and the failed attempt's
    receipts stay readable beside it, rather than the retry dying on the
    refusal.
    """

    forecast = tmp_path / "run"
    forecast.mkdir()
    (forecast / "report.json").write_text("{}", encoding="utf-8")
    observer, events, events_path = _observer(tmp_path)
    try:
        claimed = _clear_forecast_output(forecast, observer=observer)
    finally:
        events.__exit__(None, None, None)

    assert claimed != forecast and claimed.is_dir()
    kept = forecast / "report.json"
    assert kept.is_file(), "the earlier attempt's receipt was not kept"
    assert not any(claimed.iterdir())

    warning = next(record for record in read_events(events_path)
                   if record.get("code") == "forecast_output_recovery")
    assert warning['previous_path'] == str(forecast)
    assert warning['path'] == str(claimed)


# ---------------------------------------------------------------------------
# The chain, driven twice over one run directory
# ---------------------------------------------------------------------------


def _record_fetch_request(arguments, out):
    from woof import fetch
    from woof.cli import build_parser

    args = build_parser().parse_args(["fetch", *arguments])
    payload = fetch._manifest_payload(
        source=args.source, cycle=fetch.parse_cycle(args.cycle, args.source),
        hours=(), area=fetch._resolve_area(args), files=[])
    (out / fetch.FETCH_MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")


# NEEDS CUPY INSTALLED, and opens no device: the chain reruns below go
# through the real `woof go` door, which refuses before the fetch stage
# without the array library (`this command needs cupy ...`), so no second
# pass, no prepare event and no second generation are ever produced.
# Measured on the Linux release node: red without cupy, green with it
# (proof/node-reds-276).
@requires_cupy
def test_the_chain_reruns_into_the_same_run_directory_without_refusing(
        tmp_path, monkeypatch):
    """The end-to-end shape of a Studio retry.

    Same plan, same ``output_root``: the fetch is asked for the same
    bytes it already verified, the preparation finds its own bundle and
    reuses it, and neither stage refuses.  Before this, the second pass
    died inside the preparer with "refusing existing output root".
    """

    import woof.go_cli as go_cli
    import woof.runplan as runplan_module
    from woof.runplan import EVENTS_FILENAME, execute_plan, load_plan

    from test_runplan import _hrrr_plan

    fetches: list[list[str]] = []
    prepares: list[list[str]] = []

    def fake_fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text("0" * 64 + "  a.grib2\n",
                                        encoding="utf-8")
        _record_fetch_request(arguments, out)
        fetches.append(list(arguments))
        return {"arguments": list(arguments)}

    def fake_stage(label, command, **_kwargs):
        if label != "prepare":
            return
        prepares.append(list(command))
        root = Path(command[command.index("--output-root") + 1])
        _bundle(root, _identity_for(command))

    monkeypatch.setattr(runplan_module, "_run_fetch", fake_fetch)
    monkeypatch.setattr(go_cli, "run_stage", fake_stage)
    geog = tmp_path / "GEOG"
    geog.mkdir()
    run_dir = tmp_path / "run"

    codes = []
    for _ in range(2):
        plan = load_plan(_hrrr_plan(tmp_path, run_dir,
                                    run_options={"geog_root": str(geog)}))
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        with EventStream(plan.run_dir / EVENTS_FILENAME,
                         mirror=None) as events:
            codes.append(execute_plan(plan, events=events))
        records = read_events(plan.run_dir / EVENTS_FILENAME)

    # Both passes get past fetch and preparation; both then stop at the
    # same place, because this fixture publishes no portable bundle for
    # the forecast stage to bind.
    assert len(fetches) == 2, "the second pass skipped the fetch stage"
    assert len(prepares) == 1, (
        "the preparer ran a second time over a bundle it had already built")
    failed = next(record for record in records
                  if record["event"] == "failed")
    assert "refusing existing output root" not in failed["message"]
    assert "portable bundle" in failed["message"]
    assert codes == [1, 1]


def _chain_that_fails_at_the_forecast(tmp_path, monkeypatch):
    """The chain that failed: fetch and prepare land, the forecast does not.

    The two stages under test are real -- the real ``_hrrr_chain``, the
    real ``_prepare_stage``, the real decision -- and only the three
    things that need a card or a network are replaced: the transfer, the
    preparer subprocess, and the forecast runner.  The forecast is made
    to fail because that is the case this work exists for; a run that
    succeeds never needs to come back.
    """

    import woof.go_cli as go_cli
    import woof.prepared_single_domain_forecast as forecast_runner
    import woof.runplan as runplan_module

    prepares: list[list[str]] = []
    fetches: list[list[str]] = []

    def fake_fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text("0" * 64 + "  a.grib2\n",
                                        encoding="utf-8")
        _record_fetch_request(arguments, out)
        fetches.append(list(arguments))
        return {"arguments": list(arguments),
                "transfers": {"downloaded_files": 0, "verified_files": 6,
                              "summary": "skipped 6 file(s)"}}

    def fake_stage(label, command, **_kwargs):
        if label != "prepare":
            return
        prepares.append(list(command))
        root = Path(command[command.index("--output-root") + 1])
        _bundle(root, _identity_for(command))
        (root / "public-wrapper-result.json").write_text(json.dumps({
            "portable_bundle": {
                "prepared_root": str(root),
                "proof_sha256": "1" * 64,
                "source_manifest_sha256": "2" * 64,
                "prepared_content_sha256": "3" * 64,
                "experiment_config": str(root / "experiment.toml"),
                "wps_namelist": str(root / "namelist.wps"),
            }}), encoding="utf-8")

    def fake_forecast(argv, **_kwargs):
        outdir = Path(argv[argv.index("--outdir") + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "report.json").write_text("{}", encoding="utf-8")
        return 1              # the card refused this plan

    monkeypatch.setattr(runplan_module, "_run_fetch", fake_fetch)
    monkeypatch.setattr(go_cli, "run_stage", fake_stage)
    monkeypatch.setattr(
        go_cli, "proof_digests",
        lambda root: {"proof": "1" * 64, "source_manifest": "2" * 64,
                      "prepared_content": "3" * 64})
    monkeypatch.setattr(forecast_runner, "main", fake_forecast)
    geog = tmp_path / "GEOG"
    geog.mkdir()
    return fetches, prepares


@requires_cupy
def test_the_prepare_event_says_which_of_the_two_happened(tmp_path,
                                                          monkeypatch):
    """A receipt a person reads, not a timing they have to interpret."""

    from woof.runplan import EVENTS_FILENAME, execute_plan, load_plan

    from test_runplan import _hrrr_plan

    fetches, prepares = _chain_that_fails_at_the_forecast(
        tmp_path, monkeypatch)
    run_dir = tmp_path / "run"

    decisions = []
    for _ in range(2):
        plan = load_plan(_hrrr_plan(
            tmp_path, run_dir,
            run_options={"geog_root": str(tmp_path / "GEOG")}))
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        with EventStream(plan.run_dir / EVENTS_FILENAME,
                         mirror=None) as events:
            assert execute_plan(plan, events=events) == 1
        records = read_events(plan.run_dir / EVENTS_FILENAME)
        finished = [record for record in records
                    if record["event"] == "stage_finished"
                    and record.get("stage") == "prepare"]
        decisions.append(finished[-1].get("prepared"))

    # The forcing was fetched once and verified once; the bundle was
    # built once and reused once.
    assert len(fetches) == 2
    assert len(prepares) == 1
    assert decisions[0]["decision"] == stage_reuse.BUILD
    assert decisions[1]["decision"] == stage_reuse.REUSE
    assert "reused" in decisions[1]["reason"]
    assert "source_manifest_sha256" in decisions[1]["compared"]


@requires_cupy
def test_the_second_attempts_forecast_gets_a_directory_of_its_own(
        tmp_path, monkeypatch):
    """The third create-only refusal on the retry path.

    The forecast runner will not merge into a directory that already
    holds output: the receipt it writes has to describe one run. The
    failed attempt stays at its original address and the retry receives
    another generation, preserving every prior receipt's frame paths.
    """

    from woof.runplan import EVENTS_FILENAME, execute_plan, load_plan

    from test_runplan import _hrrr_plan

    _chain_that_fails_at_the_forecast(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"

    for _ in range(2):
        plan = load_plan(_hrrr_plan(
            tmp_path, run_dir,
            run_options={"geog_root": str(tmp_path / "GEOG")}))
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        with EventStream(plan.run_dir / EVENTS_FILENAME,
                         mirror=None) as events:
            execute_plan(plan, events=events)

    forecast = run_dir / "chain" / "run"
    kept = sorted((run_dir / "chain").glob("run-attempt-*/run"))
    assert len(kept) == 1, "the second attempt did not get a new generation"
    assert (kept[0] / "report.json").is_file()
    assert (forecast / "report.json").is_file()


# ---------------------------------------------------------------------------
# Real bytes
# ---------------------------------------------------------------------------


#: A live fetch's data directory (``fetch-manifest.json`` inside), named by
#: the environment.  It once defaulted to a path on F:, a drive no test may
#: read (tests/_reserved_drives.py names the breakage).
REAL_FETCH = refuse_reserved(os.environ.get(
    "GPUWM_TEST_REAL_FETCH", "gpuwm-fixture-unset/real-fetch"),
    "GPUWM_TEST_REAL_FETCH")


@pytest.mark.skipif(not (REAL_FETCH / "fetch-manifest.json").is_file(),
                    reason="no real fetched payload named by "
                           "GPUWM_TEST_REAL_FETCH")
def test_a_real_fetch_manifest_reports_its_transfers(tmp_path):
    """Read off a live HRRR fetch's own receipt, not a fixture.

    A manifest written before the rows carried ``downloaded`` says
    nothing either way, and this asserts that it is reported as saying
    nothing rather than as a fetch that downloaded nothing -- the exact
    confusion the key was added to end.
    """

    payload = json.loads(
        (REAL_FETCH / "fetch-manifest.json").read_text(encoding="utf-8"))
    split = _fetch_transfer_split(payload)

    assert split["downloaded_files"] + split["verified_files"] \
        + split["receipt_files"] == len(payload["files"])
    if all("downloaded" not in entry for entry in payload["files"]):
        assert split["summary"] == (
            "this fetch recorded no per-file transfer state")
