"""Fail-closed contracts for the public HRRR hierarchy orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import hrrr_hierarchy_direct
from woof.hrrr_forecast import hrrr_forcing_end_hour, hrrr_source_window
from woof.hrrr_hierarchy_direct import (
    _atomic_staging_sibling,
    _compare_stock_experiment,
    _expected_root_cache_identity,
    _require_raw_stock_delta,
    _require_raw_wps_contract,
    _supported_hierarchy_slice,
    _validated_root_preparation_binding,
    verified_root_forcing_inventory,
)
from woof.ingest.hrrr_target import HrrrTargetDomain
from woof.native_wrf_contract import CERTIFIED_ETA_LEVELS
from woof.wrf_physics_inventory import EXPORT_USE_THETA_M
from woof.core.devices import DeviceOptions, DEVICES_OFF
from woof.simulated_radar import SimulatedRadarOptions, OFF as RADAR_OFF


def test_atomic_staging_sibling_keeps_deep_windows_publication_short(tmp_path):
    """The d06 transaction must not repeat a long public output basename."""

    parent = tmp_path
    while len(str(parent)) < 125:
        parent /= "deep-path-budget-segment"
    # exist_ok: a temporary folder already 125 characters deep takes no
    # segment, and then ``parent`` is tmp_path itself.
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / ("hrrr-six-domain-z80-" + "x" * 72)

    staging = _atomic_staging_sibling(output)

    assert staging.parent == output.parent
    assert staging.name.startswith(".d-")
    assert len(staging.name) == len(".d-") + 10
    assert output.name not in staging.name
    # Reproduce the nested hierarchy/domain/prepared-cache publication depth
    # that raised WinError 206 in the genuine six-domain run.
    payload = (
        staging / ".d-0123456789" / "domains" / ".d-0123456789"
        / ".p-012345abcd" / "header.json"
    )
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"path-budget-pass")
    assert payload.read_bytes() == b"path-budget-pass"


@pytest.mark.parametrize("cache_kind", ["foreign", "local", "unwritable"])
def test_join_highres_cache_survives_publication_and_reuses_tiles(
        tmp_path, monkeypatch, cache_kind):
    from io import BytesIO
    import os
    from types import SimpleNamespace
    from woof import stage_reuse
    from woof.static import highres_production as owner
    from woof.static.highres_fetch import fetch_file
    from test_hrrr_native_static import _fixture

    join = hrrr_hierarchy_direct
    preparation = tmp_path / "sealed"
    preparation.mkdir()
    target, cache, static_receipt = _fixture(preparation)
    local_cache = tmp_path / "local-cache"
    if cache_kind == "foreign":
        recorded_cache = ("/preparation/source-cache" if os.name == "nt"
                          else r"C:\preparation\source-cache")
    else:
        recorded_cache = str(local_cache)
    if cache_kind == "unwritable":
        local_cache.write_bytes(b"occupied by a file")
    config = owner.HighresStaticConfig(enabled=True, cache_root=Path(recorded_cache))
    day = datetime(2026, 9, 5)
    receipt = json.loads(static_receipt.read_text(encoding="utf-8"))
    receipt["highres"] = {"status": "APPLIED", "config": config.echo(),
        "case_date": day.date().isoformat(), "grid": owner._grid_identity(target.grid(), 1)}
    receipt["highres"]["config"]["cache_root"] = recorded_cache
    static_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    sealed_files = {str(p.relative_to(preparation)): p.read_bytes() if p.is_file() else None
                    for p in preparation.rglob("*")}
    authority = tmp_path / "authority"
    authority.write_text("fixture", encoding="utf-8")
    digest = join.sha256_file(authority)
    identity = {"source_identity": {"static_highres": owner.static_highres_identity(config)},
                "source_manifest_sha256": digest, "bridge_manifest_sha256": digest,
                "static_cache_sha256": join.sha256_file(cache), "namelist_sha256": digest,
                "forcing_hours": [0, 1]}
    identity["source_identity"]["static_highres"]["cache_root"] = recorded_cache
    header = {"schema": "gpuwm-prepared-real-cache-v1", "status": "READY",
              "identity": identity, "content_sha256": digest,
              "metadata": {"user": {"initial_valid_time": day.isoformat()}}}
    report = {"status": "PASS", "source_identity": identity["source_identity"],
              "prepared_cache": {"content_sha256": digest}}
    monkeypatch.setattr(join, "_root_paths", lambda root: {
        "static_cache": cache, "static_receipt": static_receipt,
        "prepared_cache": preparation, "bridge": preparation,
        "bridge_manifest": authority, "preparation_report": authority})
    monkeypatch.setattr(join, "_json", lambda path:
                        header if path.name == "header.json" else report)
    monkeypatch.setattr(join, "resolve_cpu_bridge", lambda path: authority)
    monkeypatch.setattr(join, "_require_raw_stock_delta", lambda *a:
                        {"certified_native_runtime": {"domains.sfcp_to_sfcp": [True]}})
    run = SimpleNamespace(sf_surface_physics=2, num_soil_layers=4, mp_physics=6)
    root = SimpleNamespace(run=run, grid_id=1)
    exp = SimpleNamespace(root=root, domains=[root, root], start_time=day, run_seconds=60.)
    monkeypatch.setattr(join, "_native_experiment", lambda *a, **k: (exp, "fixture", _Run()))
    monkeypatch.setattr(join, "publish_hierarchy_forecast_companions", lambda *a, **k: {})
    monkeypatch.setattr(join, "load_hrrr_target_domain", lambda path: target)
    monkeypatch.setattr(join, "_supported_hierarchy_slice", lambda *a, **k: None)
    monkeypatch.setattr(join, "validated_corridor_selection", lambda *a: None)
    monkeypatch.setattr(join, "_require_raw_wps_contract", lambda *a: {})
    monkeypatch.setattr(join, "_expected_root_cache_identity", lambda *a, **k: identity)
    monkeypatch.setattr(join, "PreparedCacheReader", lambda *a, **k:
                        SimpleNamespace(verify_all=lambda: None))
    restored = SimpleNamespace(initial_result=SimpleNamespace(state=None),
                               met=None, boundaries=None, receipt={"content_sha256": digest})
    def restore(*args, **kwargs):
        # The stage builds its children on the CPU, so its read of the
        # root must not need CUDA: a CuPy restore here stopped every
        # HRRR tree preparation on a CPU-only install.
        import numpy

        assert kwargs.get("array_module") is numpy
        return restored

    monkeypatch.setattr(join, "restore_prepared_cache", restore)
    monkeypatch.setattr(join, "_surface_state", lambda *a, **k: None)
    monkeypatch.setattr(join, "sealed_source_leads", lambda *a: (0, 1))
    monkeypatch.setattr(join, "load_hrrr_native_series", lambda *a, **k: (object(),))
    handed = []

    def static_catalog(*args, **kwargs):
        # A170: the carrier builds land cover (fields "auto"), so each
        # domain's GEOG selection reads it and the catalog is handed it.
        handed.append(kwargs.get("static_highres"))
        return SimpleNamespace(files=()), {"selections": {"d01": None}}

    monkeypatch.setattr(join, "verified_static_catalog", static_catalog)
    monkeypatch.setattr(join, "grids_from_projection_config", lambda exp: (target.grid(),))
    monkeypatch.setattr(join, "ParentInitView", lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(join, "NestedInputCatalog", lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(join, "_source_identity", lambda *a: {})
    checked = []
    original_require = owner.require_prepared_highres

    def require(*args, **kwargs):
        original_require(*args, **kwargs)
        checked.append(True)

    monkeypatch.setattr(owner, "require_prepared_highres", require)
    observed = []
    receipts = []
    downloads = []

    def opener(url, offset):
        assert offset == 0
        downloads.append(url)
        return BytesIO(b"stand-in source tile")

    def highres_receipt(config, role):
        # Use the real fetch sidecars and receipt writer; only the source
        # bytes and the expensive hierarchy/warp operations are stand-ins.
        fetched = [fetch_file(
            f"https://source.invalid/{name}", config.cache_root / name,
            urlopen=opener).receipt() for name in (f"{role}-tile", "landcover")]
        receipt = {"status": "APPLIED", "config": config.echo(),
                   "grid": owner._grid_identity(target.grid(), 2),
                   "case_date": day.date().isoformat(), "files": fetched}
        owner._write_receipt(config, receipt)
        receipts.append(receipt)
        return receipt

    def initialize(**kwargs):
        config = kwargs["catalog"].static_highres
        assert checked == [True] * (len(observed) + 1)
        receipt = highres_receipt(config, "child")
        path = kwargs["artifact_output"] / "domains" / "d02" / "geometry-receipt.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"highres": receipt}), encoding="utf-8")
        observed.append(config.cache_root)
        return SimpleNamespace(timings_seconds={}, artifacts=SimpleNamespace(receipt={}),
                               wrf_manifest={})

    corridor_roots = []

    def corridor(**kwargs):
        # Sealed before the children since plan call 5 (2.9.0); the child
        # initialization below must resolve the same cache.
        config = kwargs["static_highres"]
        corridor_roots.append(config.cache_root)
        receipt = highres_receipt(config, "corridor")
        path = kwargs["directory"] / "receipt.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"highres": receipt}), encoding="utf-8")
        return {"highres": receipt}

    monkeypatch.setattr(join, "initialize_and_export_native_hierarchy", initialize)
    monkeypatch.setattr(join, "emit_statics_corridor_set", corridor)
    # The stand-in set carries no corridor fields, so no child is born on one.
    monkeypatch.setattr(join, "corridor_birth_statics", lambda *a: {})
    output = tmp_path / "joined"
    arguments = dict(
        root_preparation=preparation, root_domain_spec=authority,
        wps_namelist=authority, namelist_input=authority,
        stock_wrf_namelist_input=authority, geog_root=preparation,
        source_manifest=authority, source_manifest_sha256=digest,
        valid_time=day, output_root=output, workers=1, statics_corridor="all")
    resolved = (local_cache if cache_kind == "local"
                else join._highres_cache_sibling(output))
    for attempt in range(2):
        result = join.prepare_hrrr_hierarchy(**arguments)
        artifacts = output / "hierarchy-artifacts"
        transient = [str(path.relative_to(output)) for path in artifacts.rglob("*")
                     if path.is_file() and b".d-" in path.read_bytes()]
        missing = [str(path) for receipt in receipts for path in (
            Path(receipt["config"]["cache_root"]), Path(receipt["receipt_path"]))
            if not path.exists()]
        assert not transient and not missing, {"staging_references": transient,
                                              "missing_receipt_paths": missing}
        assert observed[-1] == resolved == corridor_roots[-1]
        assert handed[-1] is not None and handed[-1].cache_root == resolved
        assert not resolved.is_relative_to(output)
        for name in ("child-tile", "corridor-tile", "landcover"):
            assert (resolved / name).read_bytes() == b"stand-in source tile"
        assert result["provenance"]["static_highres_cache"] == {
            "recorded_cache_root": recorded_cache, "cache_root": str(resolved),
            "substituted": cache_kind != "local"}
        # Reuse hashes the published artifacts, without traversing fetched tiles.
        hashed = []
        original_sha256 = stage_reuse._sha256
        with monkeypatch.context() as tracking:
            def sha256(path):
                hashed.append(path)
                return original_sha256(path)

            tracking.setattr(stage_reuse, "_sha256", sha256)
            stage_reuse._published_files(output, ())
        assert hashed
        assert all(not path.is_relative_to(resolved) for path in hashed)
        assert len(downloads) == 3
        if attempt == 0:
            assert [file["cache_hit"] for receipt in receipts
                    for file in receipt["files"]] == [False, False, False, True]
            stage_reuse.supersede(output)
            assert resolved.is_dir()
        else:
            assert all(file["cache_hit"] for receipt in receipts[2:]
                       for file in receipt["files"])
    assert {str(p.relative_to(preparation)): p.read_bytes() if p.is_file() else None
            for p in preparation.rglob("*")} == sealed_files
    assert not list(tmp_path.glob(".d-*"))


def test_highres_cache_sibling_is_short_stable_and_beside_the_output(
        tmp_path, monkeypatch):
    sibling_of = hrrr_hierarchy_direct._highres_cache_sibling
    long_name = "hrrr-six-domain-z80-" + "x" * 72

    sibling = sibling_of(tmp_path / long_name)

    digest = hashlib.sha256(long_name.encode("utf-8")).hexdigest()[:6]
    assert sibling == tmp_path / f"hrrr-six-d-{digest}.highres"
    assert len(sibling.name) == 25
    short = sibling_of(tmp_path / "joined")
    assert short.name == (
        "joined-" + hashlib.sha256(b"joined").hexdigest()[:6] + ".highres")
    # A rebuild names the same output root however it spells it, and finds
    # the same folder; the recorded path is absolute either way.
    monkeypatch.chdir(tmp_path)
    assert sibling_of(Path(long_name)) == sibling
    assert sibling_of(Path(long_name)).is_absolute()
    assert sibling_of(tmp_path / long_name.upper()).name.casefold() == (
        sibling.name.casefold())
    # Two outputs that share the first ten characters keep separate folders.
    assert sibling_of(tmp_path / (long_name[:-1] + "y")) != sibling


def test_highres_fetch_receipts_stay_under_the_windows_path_limit_at_depth(
        tmp_path, monkeypatch):
    """A long output name under a deep folder keeps every receipt writable.

    The shape of the staging test above: a 125-character parent and a
    92-character output name.  Repeating that name in the fetch folder put
    the receipts at 295 characters and more, past the 259 Windows accepts.
    """
    import os
    from test_hrrr_native_static import _target
    from woof.static import highres_production as owner

    join = hrrr_hierarchy_direct
    if len(str(tmp_path)) >= 124:
        pytest.skip("temporary root already exceeds the 125-character parent")
    parent = tmp_path / ("p" * (125 - len(str(tmp_path)) - 1))
    parent.mkdir()
    assert len(str(parent)) == 125
    output = parent / ("hrrr-six-domain-z80-" + "x" * 72)
    recorded = ("/preparation/source-cache" if os.name == "nt"
                else r"C:\preparation\source-cache")
    config = owner.HighresStaticConfig(enabled=True, cache_root=Path(recorded))
    from woof import fetch_guard
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: None)

    resolved, record = join._local_highres_cache(
        config, recorded_cache_root=recorded, output_root=output)

    replaced = []
    original_replace = os.replace

    def replace_spy(source, target):
        if Path(target).parent == Path(resolved.cache_root) / "receipts":
            replaced.extend((str(source), str(target)))
        return original_replace(source, target)

    grid = _target().grid()
    receipts = []
    with monkeypatch.context() as spying:
        spying.setattr(owner, "os", SimpleNamespace(
            replace=replace_spy, getpid=os.getpid))
        # Every source a configuration can name enters the receipt name.
        for terrain in owner._TERRAIN_SOURCE_CHOICES:
            for landcover in owner._LANDCOVER_SOURCE_CHOICES:
                asked = replace(resolved, terrain_source=terrain,
                                landcover_source=landcover)
                receipt = {"status": "APPLIED", "config": asked.echo(),
                           "grid": owner._grid_identity(grid, 6),
                           "case_date": "2026-09-05"}
                owner._write_receipt(asked, receipt)
                receipts.append(receipt["receipt_path"])

    written = [str(path) for path in Path(resolved.cache_root).rglob("*")]
    assert len(receipts) == len(set(receipts)) == (
        len(owner._TERRAIN_SOURCE_CHOICES)
        * len(owner._LANDCOVER_SOURCE_CHOICES))
    over = {path: len(path) for path in (*receipts, *replaced, *written)
            if len(path) > 259}
    assert not over, over
    assert record == {"recorded_cache_root": recorded,
                      "cache_root": str(resolved.cache_root),
                      "substituted": True}
    assert resolved.cache_root == join._highres_cache_sibling(output)
    again, _record = join._local_highres_cache(
        config, recorded_cache_root=recorded, output_root=output)
    assert again.cache_root == resolved.cache_root


def _prepare_without_inputs(output, tmp_path, monkeypatch):
    """Run the stage on absent inputs: it refuses or reaches the input check."""
    from woof import hrrr_hierarchy_direct as join

    monkeypatch.setattr(join, "resolve_cpu_bridge", lambda path: tmp_path)
    missing = tmp_path / "absent-input"
    return join.prepare_hrrr_hierarchy(
        root_preparation=missing, root_domain_spec=missing,
        wps_namelist=missing, namelist_input=missing,
        stock_wrf_namelist_input=missing, geog_root=missing,
        source_manifest=missing, source_manifest_sha256="0" * 64,
        valid_time=datetime(2026, 9, 5), output_root=output)


def _deepest_published(output):
    return (output / "hierarchy-artifacts" / "domains" / "d01"
            / "prepared-cache" / "header.json")


def test_a_deep_output_root_is_refused_before_any_work_where_windows_limits_paths(
        tmp_path, monkeypatch):
    """The deep shape publishes a bundle the forecast cannot open.

    A 125-character folder and a 92-character output name: the tree is
    written under short staging names and published by one rename, so
    the preparation ran to the end and the forecast then found its own
    header at 277 characters, past the 259 Windows opens.  The stage now
    refuses before reading an input, naming that path.
    """
    from woof import fetch_guard

    if len(str(tmp_path)) >= 124:
        pytest.skip("temporary root already exceeds the 125-character parent")
    parent = tmp_path / ("p" * (125 - len(str(tmp_path)) - 1))
    output = parent / ("hrrr-six-domain-z80-" + "x" * 72)
    deepest = _deepest_published(output)
    assert len(str(deepest)) == 277
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)

    with pytest.raises(ValueError) as caught:
        _prepare_without_inputs(output, tmp_path, monkeypatch)

    message = str(caught.value)
    assert "277 characters" in message
    assert "259" in message
    assert "18 characters shorter" in message
    assert "--output-root" in message
    assert "LongPathsEnabled" in message
    assert message.endswith(f"Longest path: {deepest}")
    assert not parent.exists()


def test_the_path_limit_refusal_binds_only_where_windows_limits_paths(
        tmp_path, monkeypatch):
    """Off Windows, or with long paths on, the deep shape prepares."""
    from woof import fetch_guard

    parent = tmp_path / ("p" * 125)
    output = parent / ("hrrr-six-domain-z80-" + "x" * 72)
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: None)

    with pytest.raises(FileNotFoundError):
        _prepare_without_inputs(output, tmp_path, monkeypatch)


@pytest.mark.parametrize("length,refused", [(259, False), (260, True)])
def test_the_path_limit_refusal_starts_one_character_past_the_limit(
        tmp_path, monkeypatch, length, refused):
    from woof import fetch_guard
    from woof.native_domain_artifacts import published_path_refusal

    tail = len(str(_deepest_published(Path("x")))) - 1
    width = length - tail - len(str(tmp_path)) - 1
    if width < 1:
        pytest.skip("temporary root too deep for this boundary")
    output = tmp_path / ("b" * width)
    assert len(str(_deepest_published(output))) == length
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)

    assert (published_path_refusal(output) is not None) is refused
    if refused:
        with pytest.raises(ValueError, match="1 character shorter"):
            _prepare_without_inputs(output, tmp_path, monkeypatch)
    else:
        with pytest.raises(FileNotFoundError):
            _prepare_without_inputs(output, tmp_path, monkeypatch)


_EXTENDED = "\\\\?\\"


def _windows_spelling(monkeypatch):
    """``is_extended`` as it answers on Windows, where the spelling exists."""
    import os

    from woof import filesystem_paths

    monkeypatch.setattr(filesystem_paths, "is_extended",
                        lambda path: os.fspath(path).startswith(_EXTENDED))


def test_the_extended_spelling_of_a_deep_root_prepares_and_the_plain_one_is_refused(
        tmp_path, monkeypatch):
    """``woof go`` and ``woof run-plan`` hand a deep run folder over in
    the extended spelling, which Windows opens at any length.  The refusal
    measured that spelling like a plain one and stopped every run folder
    of 176 characters or more; it now lets it through and still refuses
    the plain spelling of the same folder."""
    from woof import fetch_guard

    if len(str(tmp_path)) >= 124:
        pytest.skip("temporary root already exceeds the 125-character parent")
    parent = tmp_path / ("p" * (125 - len(str(tmp_path)) - 1))
    output = parent / ("hrrr-six-domain-z80-" + "x" * 72)
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)
    _windows_spelling(monkeypatch)

    with pytest.raises(FileNotFoundError):
        _prepare_without_inputs(
            Path(_EXTENDED + str(output)), tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="277 characters"):
        _prepare_without_inputs(output, tmp_path, monkeypatch)
    assert not parent.exists()


def test_a_short_output_name_is_refused_on_the_export_it_stages(
        tmp_path, monkeypatch):
    """The staged export, not the published tree, is deepest under a short name.

    ``wrf-native-input.tmp-<pid>/.root-export.tmp-<pid>/manifest.json``
    sits in the stage's staging sibling, 88 characters below the folder
    whatever the output is called, so a one-character output name in a
    176-character folder published a tree at 237 characters and failed
    writing its export at 264.
    """
    from woof import fetch_guard

    if len(str(tmp_path)) >= 170:
        pytest.skip("temporary root already exceeds the 176-character folder")
    folder = tmp_path / ("p" * (176 - len(str(tmp_path)) - 1))
    output = folder / "o"
    assert len(str(_deepest_published(output))) == 237
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)

    with pytest.raises(ValueError) as caught:
        _prepare_without_inputs(output, tmp_path, monkeypatch)

    message = str(caught.value)
    assert "264 characters" in message
    assert "in a folder at least 5 characters shorter" in message
    assert message.replace("\\", "/").endswith(
        ".root-export.tmp-4294967295/manifest.json")
    assert not folder.exists()


def _highres_config(recorded, terrain_source="auto",
                    landcover_source="auto"):
    from woof.static import highres_production as owner

    return owner.HighresStaticConfig(
        enabled=True, cache_root=Path(recorded),
        terrain_source=terrain_source, landcover_source=landcover_source)


def _widest_source_pair(cache_root):
    """The terrain and land-cover sources whose receipt name is longest."""
    from woof.static import highres_production as owner

    return max(
        ((terrain, landcover)
         for terrain in owner._TERRAIN_SOURCE_CHOICES
         for landcover in owner._LANDCOVER_SOURCE_CHOICES),
        key=lambda pair: len(str(owner.deepest_receipt_path(
            cache_root, *pair))))


def test_the_substituted_highres_receipts_are_measured_before_the_fetch_folder_exists(
        tmp_path, monkeypatch):
    """A join that fetches beside the output root writes its receipts there.

    In a folder where the tree and its staging fit, the widest receipt's
    partial name in ``<name[:10]>-<hash6>.highres/receipts/`` reaches 262
    characters, so the join failed writing it after the restore and
    decode.  It is now refused before that folder or any staging exists,
    naming the receipt path.  The join is configured with the widest
    source pair, whose receipt is the one measured.
    """
    from woof import fetch_guard
    from woof.static.highres_production import deepest_receipt_path

    join = hrrr_hierarchy_direct
    output = tmp_path / "joined"
    sibling = join._highres_cache_sibling(output)
    terrain, landcover = _widest_source_pair(sibling)
    extra = len(str(deepest_receipt_path(
        sibling, terrain, landcover))) - len(str(tmp_path))
    width = 259 - extra + 3 - len(str(tmp_path)) - 1
    if width < 1:
        pytest.skip("temporary root too deep for this shape")
    folder = tmp_path / ("q" * width)
    output = folder / "joined"
    sibling = join._highres_cache_sibling(output)
    receipt = deepest_receipt_path(sibling, terrain, landcover)
    assert len(str(receipt)) == 262
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)
    from woof.native_domain_artifacts import published_path_refusal
    assert published_path_refusal(output) is None

    recorded = "relative/source-cache"
    with pytest.raises(ValueError) as caught:
        join._local_highres_cache(
            _highres_config(recorded, terrain, landcover),
            recorded_cache_root=recorded, output_root=output)

    message = str(caught.value)
    assert "262 characters" in message
    assert "in a folder at least 3 characters shorter" in message
    assert message.endswith(f"Longest path: {receipt}")
    assert not folder.exists()


def test_a_recorded_fetch_folder_is_not_measured_as_the_substitute(
        tmp_path, monkeypatch):
    """The sibling's receipts count only when the join fetches into it."""
    from woof import fetch_guard
    from woof.static.highres_production import deepest_receipt_path

    join = hrrr_hierarchy_direct
    output = tmp_path / "joined"
    extra = len(str(deepest_receipt_path(
        join._highres_cache_sibling(output)))) - len(str(tmp_path))
    width = 259 - extra + 3 - len(str(tmp_path)) - 1
    if width < 1:
        pytest.skip("temporary root too deep for this shape")
    output = tmp_path / ("q" * width) / "joined"
    recorded = tmp_path / "recorded-cache"
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)

    resolved, record = join._local_highres_cache(
        _highres_config(recorded), recorded_cache_root=str(recorded),
        output_root=output)

    assert record["substituted"] is False
    assert resolved.cache_root == recorded
    assert not join._highres_cache_sibling(output).exists()


def test_deepest_receipt_path_is_the_longest_partial_a_receipt_writes(
        tmp_path, monkeypatch):
    """The measure is the real writer's: each source pair as that pair's
    receipt spells it, the widest pid.  Only the 16-character identity,
    a hash of the payload, differs from the measured name."""
    import os
    import re

    from test_hrrr_native_static import _target
    from woof.fetch_guard import WINDOWS_WIDEST_PID
    from woof.static import highres_production as owner

    config = _highres_config(tmp_path / "cache")
    grid = _target().grid()
    written = []
    original_replace = os.replace

    def replace_spy(source, target):
        if Path(target).parent == config.cache_root / "receipts":
            written.append(str(source))
        return original_replace(source, target)

    with monkeypatch.context() as spying:
        spying.setattr(owner, "os", SimpleNamespace(
            replace=replace_spy, getpid=lambda: WINDOWS_WIDEST_PID))
        for terrain in owner._TERRAIN_SOURCE_CHOICES:
            for landcover in owner._LANDCOVER_SOURCE_CHOICES:
                asked = replace(config, terrain_source=terrain,
                                landcover_source=landcover)
                del written[:]
                owner._write_receipt(asked, {
                    "status": "APPLIED", "config": asked.echo(),
                    "grid": owner._grid_identity(grid, 21),
                    "case_date": "2026-09-05"})
                assert len(written) == 1
                measured = str(owner.deepest_receipt_path(
                    config.cache_root, terrain, landcover))
                assert re.sub(
                    r"static_highres_[0-9a-f]{16}_",
                    "static_highres_" + "f" * 16 + "_",
                    written[0]) == measured, (terrain, landcover)


def test_a_default_config_join_in_a_125_character_folder_is_not_refused(
        tmp_path, monkeypatch):
    """The join measures the receipts its configuration writes.

    Measuring the widest source pair refused a default-configuration
    join, whose ``auto``/``auto`` receipts fit, in run folders of about
    119 to 149 characters.  In a 125-character folder with a short
    output name of ten characters, which already gives the fetch folder
    its widest name, the widest pair's receipt would pass the limit, and
    the default join fetches beside the output root all the same.
    """
    from woof import fetch_guard
    from woof.static.highres_production import deepest_receipt_path

    join = hrrr_hierarchy_direct
    if len(str(tmp_path)) >= 124:
        pytest.skip("temporary root already exceeds the 125-character folder")
    folder = tmp_path / ("p" * (125 - len(str(tmp_path)) - 1))
    output = folder / "short-name"
    sibling = join._highres_cache_sibling(output)
    assert len(str(folder)) == 125
    assert len(str(deepest_receipt_path(
        sibling, *_widest_source_pair(sibling)))) > 259
    assert len(str(deepest_receipt_path(sibling))) <= 259
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)

    recorded = "relative/source-cache"
    resolved, record = join._local_highres_cache(
        _highres_config(recorded), recorded_cache_root=recorded,
        output_root=output)

    assert record["substituted"] is True
    assert resolved.cache_root == sibling
    assert sibling.is_dir()


@dataclass(frozen=True)
class _Run:
    grid_id: int = 1
    mp_physics: int = 6
    bl_pbl_physics: int = 1
    sf_sfclay_physics: int = 91
    sf_surface_physics: int = 2
    ra_physics: int = 0
    ra_lw_physics: int = 0
    ra_sw_physics: int = 1
    cu_physics: int = 0
    clos_choice: int = 0
    ishallow: int = 0
    radt: float = 1.0
    radt_minutes: float = 1.0
    cudt_minutes: float = 0.0
    nx: int = 199
    ny: int = 199
    nz: int = 49
    dx: float = 2999.4213047435587
    dy: float = 2999.4213047435587
    dt: float = 15.0
    specified: bool = True
    nested: bool = False
    spec_zone: int = 1
    relax_zone: int = 4
    moist: bool = True
    moist_cq: bool = True
    nest_microphysics_transition: str = "same-scheme-only"
    output_interval_s: float = 3600.0
    restart_interval_s: float = 0.0
    nwp_diagnostics: int = 0
    inflow_perturbation: bool = False
    inflow_perturbation_seed: int = 0
    inflow_perturbation_amplitude_scale: float = 1.0
    inflow_perturbation_faces: str = "inflow"


@dataclass(frozen=True)
class _Domain:
    grid_id: int
    run: _Run = _Run()
    parent_id: int = 0
    i_parent_start: int = 1
    j_parent_start: int = 1
    parent_grid_ratio: int = 1
    parent_time_step_ratio: int = 1
    history_interval_s: float = 300.0


@dataclass(frozen=True)
class _Projection:
    map_proj: str = "lambert"
    ref_lat: float = 35.5028506728143
    ref_lon: float = -98.0021669285660
    truelat1: float = 38.5
    truelat2: float = 38.5
    stand_lon: float = -97.5


@dataclass(frozen=True)
class _Vertical:
    eta_levels: tuple[float, ...] = CERTIFIED_ETA_LEVELS
    p_top: float = 10_000.0
    hybrid_opt: int = 2
    etac: float = 0.2


@dataclass(frozen=True)
class _Experiment:
    name: str
    domains: tuple[_Domain, ...]
    feedback: int = 0
    smooth_option: int = 0
    run_seconds: float = 43_200.0
    projection: _Projection = _Projection()
    vertical: _Vertical = _Vertical()
    spec_bdy_width: int = 5
    devices: DeviceOptions = DEVICES_OFF
    simulated_radar: SimulatedRadarOptions = RADAR_OFF
    # 18fb332a0, lane/recover-physics-params: the recovered optional set
    # belongs to ExperimentConfig. No set stays absent from its document;
    # this stock-comparison fixture must carry the same default explicitly.
    physics_params: object | None = None


def _native() -> _Experiment:
    child_run = replace(
        _Run(), grid_id=2, nx=300, ny=300, dx=999.8071015811862,
        dy=999.8071015811862, dt=5.0, specified=False, nested=True,
    )
    return _Experiment(
        name="native",
        domains=(
            _Domain(1),
            _Domain(
                2, child_run, parent_id=1, i_parent_start=50,
                j_parent_start=50, parent_grid_ratio=3,
                parent_time_step_ratio=3),
        ),
    )


def _target(**changes) -> HrrrTargetDomain:
    values = {
        "name": "hrrr_test",
        "map_proj": "lambert",
        "nx": 199,
        "ny": 199,
        "nz": 49,
        "dx_m": 2999.4213047435587,
        "dy_m": 2999.4213047435587,
        "ref_lat": 35.5028506728143,
        "ref_lon": -98.0021669285660,
        "truelat1": 38.5,
        "truelat2": 38.5,
        "stand_lon": -97.5,
        "time_step_seconds": 15,
        "spec_bdy_width": 5,
        "spec_zone": 1,
        "relax_zone": 4,
    }
    values.update(changes)
    return HrrrTargetDomain(**values)


def _raw_runtime_namelist(
        max_dom, *, longwave, theta_m, ghg_input=None, run_hours=12,
        do_radar_ref=None, land_surface=2, soil_layers=4):
    """The certified raw runtime shape.  ``ghg_input`` and
    ``do_radar_ref`` are the two STOCK-ONLY keys: passing either marks
    this text as the stock half of the pair, and the native half must
    omit both."""

    def repeated(value):
        return ", ".join(str(value) for _ in range(max_dom))
    ghg = "" if ghg_input is None else f" ghg_input = {ghg_input},\n"
    radar = ("" if do_radar_ref is None
             else f" do_radar_ref = {do_radar_ref},\n")
    return (
        "&time_control\n"
        f" run_hours = {run_hours},\n"
        " interval_seconds = 3600,\n"
        f" frames_per_outfile = {repeated(1)},\n"
        " restart = .false.,\n"
        " io_form_history = 2,\n io_form_restart = 2,\n"
        " io_form_input = 2,\n io_form_boundary = 2,\n/\n"
        "&domains\n"
        f" max_dom = {max_dom},\n"
        " num_metgrid_levels = 51,\n"
        " num_metgrid_soil_levels = 9,\n"
        " sfcp_to_sfcp = .true.,\n/\n"
        "&physics\n"
        f" ra_lw_physics = {repeated(longwave)},\n"
        f" ra_sw_physics = {repeated(1)},\n"
        " isfflx = 1,\n ifsnow = 1,\n icloud = 1,\n"
        f" sf_surface_physics = {repeated(land_surface)},\n"
        f" surface_input_source = 1,\n num_soil_layers = {soil_layers},\n"
        f" sf_urban_physics = {repeated(0)},\n"
        " sst_update = 0,\n"
        f"{ghg}{radar}/\n"
        f"&dynamics\n use_theta_m = {theta_m},\n/\n"
    )


def _sealed_forcing(exp) -> tuple[int, ...]:
    """The forcing inventory a root prepared for EXP would carry.

    The duration ceiling is a property of the sealed root preparation, so
    every gate call needs one.  Tests about geometry and physics say "the
    root that was prepared for this experiment" with this; the tests that
    are ABOUT the ceiling pass their own inventory explicitly.  Derived
    from the preparer's own endpoint arithmetic so this helper seals what
    ``tools.prepare_hrrr_wrf`` actually seals, sub-hour runs included.
    """

    return tuple(range(hrrr_forcing_end_hour(exp.run_seconds) + 1))


def _slice(exp, target) -> None:
    _supported_hierarchy_slice(
        exp, target, forcing_hours=_sealed_forcing(exp))


def test_public_gate_accepts_generic_parent_ordered_easy_physics_slice():
    native = _native()
    target = _target()
    _slice(native, target)
    _slice(replace(native, domains=(_Domain(1),)), target)

    child3 = replace(
        native.domains[1], grid_id=3, parent_id=2,
        i_parent_start=100, j_parent_start=100,
        run=replace(
            native.domains[1].run, grid_id=3, nx=240, ny=240,
            dx=333.2690338603954, dy=333.2690338603954,
            dt=5.0 / 3.0))
    sibling4 = replace(
        native.domains[1], grid_id=4, parent_id=1,
        i_parent_start=70, j_parent_start=60,
        run=replace(
            native.domains[1].run, grid_id=4, nx=180, ny=180))
    generic = replace(
        native, domains=(*native.domains, child3, sibling4))
    _slice(generic, target)

    with pytest.raises(ValueError, match="parent-before-child"):
        _slice(replace(
            generic, domains=(generic.domains[0], generic.domains[2],
                              generic.domains[1], generic.domains[3])), target)
    # Two-way trees are admitted: feedback is a runtime coupling the tree
    # executor runs, not a property of anything this stage prepares.
    _slice(replace(_native(), feedback=1), target)
    _slice(replace(_native(), feedback=1, smooth_option=2), target)
    # A mixed WSM6 -> Morrison edge that names no transition policy.
    #
    # This used to refuse as "unsupported mixed" (before v1.3.1 the only
    # admitted mixed edge was Thompson -> NSSL), then as "the pair needs
    # its policy named".  Since 2.7.3 the unset key resolves to the
    # closure the pair takes and the coupler receipt records the requested
    # and the effective policy, so the tree is admitted with the moist/CQ
    # contract the transition requires on both domains ...
    unnamed = replace(
        _native(), domains=(
            replace(_native().domains[0], run=replace(
                _native().domains[0].run, moist=True, moist_cq=True)),
            replace(_native().domains[1], run=replace(
                _native().domains[1].run, mp_physics=10,
                moist=True, moist_cq=True))))
    _slice(unnamed, target)

    # ... while naming the closure of ANOTHER edge for this pair is a
    # contradiction and still fails closed by name.
    contradicted = replace(
        unnamed, domains=(unnamed.domains[0], replace(
            unnamed.domains[1], run=replace(
                unnamed.domains[1].run,
                nest_microphysics_transition="mp8-to-mp18-mass-diagnosed-v1"))))
    with pytest.raises(ValueError, match="closure of another edge"):
        _slice(contradicted, target)

    # Naming the pair's own closure admits the same tree.
    admitted = replace(
        _native(), domains=(
            replace(_native().domains[0], run=replace(
                _native().domains[0].run, moist=True, moist_cq=True)),
            replace(_native().domains[1], run=replace(
                _native().domains[1].run, mp_physics=10,
                moist=True, moist_cq=True,
                nest_microphysics_transition="mp-edge-mass-diagnosed-v1"))))
    _slice(admitted, target)


def test_public_gate_admits_explicit_thompson_to_nssl_tree_without_consent():
    base = _native()
    root_run = replace(base.domains[0].run, mp_physics=8, moist_cq=True)
    d02_run = replace(
        base.domains[1].run, mp_physics=8, moist_cq=True)
    d03_run = replace(
        d02_run, grid_id=3, nx=180, ny=180,
        dx=d02_run.dx / 3.0, dy=d02_run.dy / 3.0,
        dt=d02_run.dt / 3.0, mp_physics=18,
        nest_microphysics_transition=(
            "mp8-to-mp18-mass-diagnosed-v1"))
    d04_run = replace(
        d03_run, grid_id=4, nx=120, ny=120,
        dx=d03_run.dx / 2.0, dy=d03_run.dy / 2.0,
        dt=d03_run.dt / 2.0,
        nest_microphysics_transition="same-scheme-only")
    mixed = replace(base, domains=(
        replace(base.domains[0], run=root_run),
        replace(base.domains[1], run=d02_run),
        _Domain(
            3, d03_run, parent_id=2, i_parent_start=20,
            j_parent_start=20, parent_grid_ratio=3,
            parent_time_step_ratio=3),
        _Domain(
            4, d04_run, parent_id=3, i_parent_start=20,
            j_parent_start=20, parent_grid_ratio=2,
            parent_time_step_ratio=2),
    ))

    _slice(mixed, _target())

    # The same tree with the key left at its default on the mixed edge
    # resolves to the Thompson -> NSSL closure and is admitted.
    unnamed_policy = replace(
        mixed, domains=(*mixed.domains[:2], replace(
            mixed.domains[2], run=replace(
                mixed.domains[2].run,
                nest_microphysics_transition="same-scheme-only")),
            mixed.domains[3]))
    _slice(unnamed_policy, _target())


def test_public_gate_admits_a_same_scheme_p3_hierarchy():
    """mp=50 rides the same plan leg every other admitted scheme rides.

    The scheme's former exclusion from _SUPPORTED_MICROPHYSICS is
    retired (see the constant's comment); this is the reachability proof
    for the pure-P3 tree -- the shape a stranger's public config takes.
    """
    base = _native()
    pure = replace(base, domains=(
        replace(base.domains[0], run=replace(
            base.domains[0].run, mp_physics=50)),
        replace(base.domains[1], run=replace(
            base.domains[1].run, mp_physics=50)),
    ))
    _slice(pure, _target())


def test_public_gate_admits_mixed_p3_edges_under_the_named_policy():
    """Both directions of the ratified rime-pair closure, on this route.

    A Thompson root carrying a P3 child (entry closure: qi/qs/qg merge,
    rime pair diagnosed) and a P3 root carrying a Thompson child (exit
    closure: split by rime state), each admitted with the edge policy
    named and again with the key left at its default, which resolves to
    the same closure.
    """
    base = _native()
    for root_mp, child_mp in ((8, 50), (50, 8)):
        root_run = replace(
            base.domains[0].run, mp_physics=root_mp,
            moist=True, moist_cq=True)
        child_run = replace(
            base.domains[1].run, mp_physics=child_mp,
            moist=True, moist_cq=True,
            nest_microphysics_transition="mp-edge-mass-diagnosed-v1")
        mixed = replace(base, domains=(
            replace(base.domains[0], run=root_run),
            replace(base.domains[1], run=child_run)))
        _slice(mixed, _target())

        unnamed = replace(mixed, domains=(mixed.domains[0], replace(
            mixed.domains[1], run=replace(
                mixed.domains[1].run,
                nest_microphysics_transition="same-scheme-only"))))
        _slice(unnamed, _target())


def test_public_gate_accepts_short_ohio_z80_sealed_root():
    target = _target(
        name="hrrr_ohio_z80", nx=192, ny=160, nz=80,
        dx_m=3000.0, dy_m=3000.0, ref_lat=40.0, ref_lon=-83.0,
        truelat1=30.0, truelat2=60.0, stand_lon=-84.0)
    root_run = replace(
        _Run(), nx=192, ny=160, nz=80, dx=3000.0, dy=3000.0)
    child_run = replace(
        root_run, grid_id=2, nx=90, ny=90, dx=1000.0, dy=1000.0,
        dt=5.0, specified=False, nested=True)
    vertical = _Vertical(
        eta_levels=tuple(1.0 - i / 80.0 for i in range(81)))
    short = _Experiment(
        name="ohio-z80", run_seconds=3600.0,
        projection=_Projection(
            ref_lat=40.0, ref_lon=-83.0, truelat1=30.0,
            truelat2=60.0, stand_lon=-84.0),
        vertical=vertical,
        domains=(
            _Domain(1, root_run),
            _Domain(
                2, child_run, parent_id=1, i_parent_start=30,
                j_parent_start=30, parent_grid_ratio=3,
                parent_time_step_ratio=3),
        ),
    )
    _slice(short, target)


def test_public_gate_runs_the_whole_sealed_forcing_window_not_twelve_hours():
    """A retained f00..f24 hierarchy, and the ceiling that replaced 43200.

    The gate carried ``run_seconds <= 43_200`` with the sentence "the
    native f00..f12 forcing horizon", while the root wrapper's
    ``hrrr_source_window``, the fetch stage and the forcing-hour equality
    below it all handle f00..f24.  A user who fetched and prepared a
    24 h root was refused at the last gate before publication, by a
    constant describing neither the data nor the code.  The ceiling is the
    sealed preparation's own inventory now, so this asserts BOTH ends: a
    24 h window runs 24 h, and a 12 h window still refuses 15 h.
    """

    target = _target()
    long_run = replace(_native(), run_seconds=86_400.0)

    _supported_hierarchy_slice(
        long_run, target, forcing_hours=tuple(range(25)))
    # The intermediate case the field report used: > 12 h, < 24 h.
    _supported_hierarchy_slice(
        replace(long_run, run_seconds=54_000.0), target,
        forcing_hours=tuple(range(16)))

    with pytest.raises(ValueError) as refusal:
        _supported_hierarchy_slice(
            replace(long_run, run_seconds=54_000.0), target,
            forcing_hours=tuple(range(13)))
    message = str(refusal.value)
    assert "12 h forcing horizon" in message
    assert "f00..f12" in message
    assert "15 h" in message
    # A zero-length or negative request is still refused, and an
    # inventory that is not a contiguous run of hourly leads from zero is
    # not an authority at all.
    with pytest.raises(ValueError, match="must be positive"):
        _supported_hierarchy_slice(
            replace(long_run, run_seconds=0.0), target,
            forcing_hours=tuple(range(25)))
    with pytest.raises(ValueError, match="contiguous hourly leads"):
        _supported_hierarchy_slice(
            long_run, target, forcing_hours=(0, 1, 3))
    with pytest.raises(ValueError, match="contiguous hourly leads"):
        _supported_hierarchy_slice(long_run, target, forcing_hours=())


def test_sub_hour_run_expects_the_series_the_preparer_actually_seals():
    """The 900 s field failure: a floor endpoint against a ceiling seal.

    With ``run_seconds = 900`` tools.prepare_hrrr_wrf sizes its window
    with ``hrrr_source_window`` -- a ceiling, because the endpoint at
    0.25 h lies BETWEEN forcing hours and boundary temporal
    interpolation needs the bracketing frame above it -- and seals model
    forcing hours (0, 1).  This stage recomputed the endpoint with a
    floor, expected ``(0,)``, and refused every sub-hour root the
    preparer can build ("expected (0,), got (0, 1)"); ``(0,)`` is also a
    series the tree runner downstream rejects outright, because a single
    frame brackets nothing.  Both stages derive the endpoint from
    :func:`woof.hrrr_forecast.hrrr_forcing_end_hour` now.
    """

    # What the preparer actually seals for a 900 s run at lead 0.
    window = hrrr_source_window(
        cycle=datetime(2026, 7, 28, 18), start_hour=0, run_seconds=900.0)
    sealed = tuple(range(len(window)))
    assert sealed == (0, 1)

    # The direct hierarchy accepts exactly that series for the same run,
    # and its duration gate admits the run against it.
    assert verified_root_forcing_inventory(
        sealed, run_seconds=900.0) == sealed
    _supported_hierarchy_slice(
        replace(_native(), run_seconds=900.0), _target(),
        forcing_hours=sealed)

    # Whole-hour runs are unchanged: floor and ceiling agree there.
    assert verified_root_forcing_inventory(
        (0, 1), run_seconds=3600.0) == (0, 1)
    assert verified_root_forcing_inventory(
        tuple(range(13)), run_seconds=43_200.0) == tuple(range(13))

    # Genuinely wrong inventories still refuse with the same sentence:
    # truncated, over-long, and gapped.
    with pytest.raises(ValueError, match="consecutive hourly HRRR forcing"):
        verified_root_forcing_inventory((0,), run_seconds=900.0)
    with pytest.raises(ValueError, match="consecutive hourly HRRR forcing"):
        verified_root_forcing_inventory((0, 1, 2), run_seconds=3600.0)
    with pytest.raises(ValueError, match="consecutive hourly HRRR forcing"):
        verified_root_forcing_inventory((0, 2), run_seconds=7200.0)
    # And in the orchestration a truncated root never even reaches that
    # check: the sealed-horizon gate refuses it first.
    with pytest.raises(ValueError, match="forcing horizon"):
        _supported_hierarchy_slice(
            replace(_native(), run_seconds=900.0), _target(),
            forcing_hours=(0,))


def test_public_gate_accepts_a_per_domain_history_cadence():
    """d01 hourly beside d02 every 15 minutes -- the ladder's whole point.

    Every other stage already supported it: the experiment loader gives
    each domain its own ``history_interval_s`` and divisibility check, the
    clock registers a per-domain history alarm, and the tree runner writes
    per-domain wrfouts on it.  Only this drift check disagreed, and it did
    so after the expensive preparation, with "output_interval_s (900.0,
    3600.0)".  Preparation writes no history frame at all.
    """

    native = _native()
    laddered = replace(native, domains=(
        native.domains[0],
        replace(native.domains[1], history_interval_s=900.0,
                run=replace(native.domains[1].run,
                            output_interval_s=900.0)),
    ))

    _slice(laddered, _target())

    # A control that must still refuse: a genuinely trajectory-bearing
    # per-domain difference is not an output cadence.
    drifted = replace(laddered, domains=(
        laddered.domains[0],
        replace(laddered.domains[1], run=replace(
            laddered.domains[1].run, sf_surface_physics=3)),
    ))
    with pytest.raises(ValueError, match="trajectory controls differ"):
        _slice(drifted, _target())


def test_public_gate_accepts_a_child_only_inflow_perturbation():
    """The P3 inflow keys on a child, refused after expensive preparation.

    Same finding as the history cadence above, for the third time: the
    generator acts at runtime FORCE on the child-owned rolling NEST
    boundary tables, and preparation never computes those.  The ruling
    that these fields are preparation-inert is already committed as
    ``woof.ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS``; this
    gate was the second preparation-side comparison and had not been
    given it, so an LES tree with the generator ON died here holding a
    prepared root that was in fact exactly the right prepared root.
    """

    native = _native()
    seeded = replace(native, domains=(
        native.domains[0],
        replace(native.domains[1], run=replace(
            native.domains[1].run,
            inflow_perturbation=True,
            inflow_perturbation_seed=20160524,
            inflow_perturbation_amplitude_scale=1.0,
            inflow_perturbation_faces="inflow")),
    ))

    _slice(seeded, _target())

    # The control: a field preparation DOES read still refuses, so this
    # is an admission of four named keys and not a hole in the check.
    drifted = replace(seeded, domains=(
        seeded.domains[0],
        replace(seeded.domains[1], run=replace(
            seeded.domains[1].run, sf_sfclay_physics=1)),
    ))
    with pytest.raises(ValueError, match="trajectory controls differ"):
        _slice(drifted, _target())


def test_public_gate_accepts_a_grell_freitas_root_over_a_cumulus_off_child():
    """The Grell-family keys ride with cu_physics.

    The importer writes clos_choice and ishallow only on a domain whose
    cu_physics is 3, so a Grell-Freitas root set to clos_choice = 1 has a
    cumulus-off child holding 0 for both, and this drift check refused
    the tree after the fetch and the root preparation with
    "{'clos_choice': (0, 1)}".  Preparation reads neither key.
    """

    native = _native()
    root = replace(native.domains[0], run=replace(
        native.domains[0].run, cu_physics=3, clos_choice=1, ishallow=1))
    tree = replace(native, domains=(root, native.domains[1]))

    _slice(tree, _target())

    # The control: a field preparation does read still refuses.
    drifted = replace(tree, domains=(
        root, replace(tree.domains[1], run=replace(
            tree.domains[1].run, sf_surface_physics=3))))
    with pytest.raises(ValueError, match="trajectory controls differ"):
        _slice(drifted, _target())


def test_root_binding_ignores_the_grell_selectors_a_namelist_cannot_carry():
    """A root prepared from the TOML binds the namelist's d01.

    A namelist written without clos_choice and ishallow imports them at
    0 while the sealed root was prepared with the TOML's values, and the
    hierarchy stage stopped with "native namelist d01 trajectory controls
    differ from the sealed root preparation: run.clos_choice".
    """

    native = _native()
    grell = replace(native.domains[0], run=replace(
        native.domains[0].run, cu_physics=3))
    sealed = asdict(grell)
    sealed["run"]["clos_choice"] = 1
    sealed["run"]["ishallow"] = 1
    identity = {"domain_config": sealed, "namelist_sha256": "c" * 64}

    digest, prepared = _validated_root_preparation_binding(identity, grell)
    assert digest == "c" * 64
    assert prepared == sealed

    drifted = replace(grell, run=replace(grell.run, cu_physics=1))
    with pytest.raises(ValueError, match="run.cu_physics"):
        _validated_root_preparation_binding(identity, drifted)


def test_root_binding_ignores_write_cadence_and_inert_diagnostics():
    """The three switches that made root and hierarchy irreconcilable.

    A sealed root is built from a shipped physics profile, which writes
    ``restart_interval_s = 0`` and no diagnostics; a hierarchy is imported
    from the user's namelist, which may carry ``restart_interval = 60``
    and whose cumulus-off domains inherit RunConfig's live ``cudt_minutes
    = 5.0``.  None of the three changes one model step -- woof/io/
    restart.py already lets all three differ across a checkpoint boundary
    -- so none of them may decide whether a prepared root can be reused.
    """

    native = _native()
    sealed = asdict(native.domains[0])
    sealed["run"]["restart_interval_s"] = 0.0
    sealed["run"]["nwp_diagnostics"] = 0
    sealed["run"]["cudt_minutes"] = 5.0
    identity = {"domain_config": sealed, "namelist_sha256": "b" * 64}

    live = replace(native.domains[0], run=replace(
        native.domains[0].run, restart_interval_s=3600.0,
        nwp_diagnostics=1, cudt_minutes=0.0))
    digest, prepared = _validated_root_preparation_binding(identity, live)

    assert digest == "b" * 64
    # The document RETURNED is still the sealed one, byte for byte: the
    # normalization decides the comparison, never what gets re-hashed.
    assert prepared == sealed

    # And a real trajectory difference in the same document still refuses.
    drifted = replace(live, run=replace(live.run, sf_surface_physics=3))
    with pytest.raises(ValueError, match="run.sf_surface_physics"):
        _validated_root_preparation_binding(identity, drifted)


def test_root_cache_reuse_binds_effective_d01_not_child_topology():
    native = _native()
    identity = {
        "domain_config": asdict(native.domains[0]),
        "namelist_sha256": "a" * 64,
    }
    digest, prepared = _validated_root_preparation_binding(
        identity, native.domains[0])
    assert digest == "a" * 64
    assert prepared == identity["domain_config"]

    deeper = replace(native, domains=(*native.domains, replace(
        native.domains[1], grid_id=3, parent_id=2,
        run=replace(native.domains[1].run, grid_id=3))))
    assert _validated_root_preparation_binding(
        identity, deeper.domains[0])[0] == "a" * 64

    drifted_root = replace(
        native.domains[0], run=replace(native.domains[0].run, mp_physics=10))
    with pytest.raises(ValueError, match="d01 trajectory controls differ"):
        _validated_root_preparation_binding(identity, drifted_root)
    with pytest.raises(ValueError, match="invalid namelist_sha256"):
        _validated_root_preparation_binding(
            {**identity, "namelist_sha256": "editable"}, native.domains[0])


def test_hierarchy_expected_identity_preserves_validated_namelist_invariant():
    native = _native()
    identity = {
        "domain_config": asdict(native.domains[0]),
        "namelist_sha256": "a" * 64,
        "source_identity": {"source": "fixture"},
        "namelist_extension_invariant": {
            "schema": "gpuwm-namelist-extension-invariant-v1",
            "sha256": "e" * 64,
        },
    }
    expected = _expected_root_cache_identity(
        identity, root_domain=native.domains[0],
        bridge_manifest_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
        static_cache_sha256="d" * 64,
        forcing_hours=(0, 1))
    assert expected["namelist_extension_invariant"] == identity[
        "namelist_extension_invariant"]

    identity["namelist_extension_invariant"]["unchecked"] = True
    with pytest.raises(ValueError, match="valid extension invariant"):
        _expected_root_cache_identity(
            identity, root_domain=native.domains[0],
            bridge_manifest_sha256="b" * 64,
            source_manifest_sha256="c" * 64,
            static_cache_sha256="d" * 64,
            forcing_hours=(0, 1))


def test_sealed_legacy_variant_reaches_the_hierarchy_import(tmp_path):
    """The 4/4 IMPLEMENTATION resolves from the sealed root, end to end.

    A WRF namelist spells radiation as selector integers, so the battery's
    legacy-RRTMG profile cannot survive a namelist round trip on its own:
    the importer's default maps 4/4 to the RTE+RRTMGP substitution.  The
    route therefore reads the sealed root preparation's recorded
    ``ra_rrtmg_variant`` and imports under it.  Three legs, one run:
    the profile pins the variant, the unplumbed default is REFUSED by the
    d01 binding (the seam, kept as the negative control), and the plumbed
    import resolves the legacy engine and binds.
    """

    from woof.experiment import load_experiment
    from woof.hrrr_hierarchy_direct import (_native_experiment,
                                             _sealed_root_rrtmg_variant)
    from woof.ingest.prepared_cache import prepared_domain_config_identity
    from woof.physics_compat import (RRTMG_VARIANT_LEGACY,
                                      THOMPSON_LEGACY_RRTMG_PROFILE_ID,
                                      WRF_RRTMG_LEGACY,
                                      single_domain_runtime_switches)
    from tools import battery_wrf_node_plan as node_plan

    # Leg 1: the profile the root preparer materializes pins the variant.
    switches = single_domain_runtime_switches(
        THOMPSON_LEGACY_RRTMG_PROFILE_ID)
    assert switches["ra_rrtmg_variant"] == RRTMG_VARIANT_LEGACY
    assert switches["wrf_rrtmg_compatibility"] == WRF_RRTMG_LEGACY

    repo = Path(__file__).resolve().parents[1]
    config = (repo / "configs" / "battery"
              / "shape_3km_thompson_rrtmg_legacy.toml")
    outdir = tmp_path / "node"
    node_plan.build(config, outdir, ranks=24, repository_root=repo)
    exp = load_experiment(config)
    sealed = {
        "domain_config": prepared_domain_config_identity(exp.domains[0]),
        "namelist_sha256": "a" * 64,
    }
    assert _sealed_root_rrtmg_variant(sealed) == RRTMG_VARIANT_LEGACY

    # Leg 2 (the seam): the importer default resolves the substitution
    # engine, and the d01 binding refuses the mismatch by name.
    default_exp, _resolved, _report = _native_experiment(
        outdir / "namelist.wps", outdir / "namelist.native.input")
    assert default_exp.domains[0].run.ra_rrtmg_variant == "rte-rrtmgp"
    with pytest.raises(ValueError, match="ra_rrtmg_variant"):
        _validated_root_preparation_binding(sealed, default_exp.domains[0])

    # Leg 3 (the plumb): importing under the sealed variant resolves the
    # legacy engine and its compatibility token, and the binding accepts.
    plumbed_exp, _resolved, _report = _native_experiment(
        outdir / "namelist.wps", outdir / "namelist.native.input",
        rrtmg_variant=RRTMG_VARIANT_LEGACY)
    run = plumbed_exp.domains[0].run
    assert run.ra_rrtmg_variant == RRTMG_VARIANT_LEGACY
    assert run.wrf_rrtmg_compatibility == WRF_RRTMG_LEGACY
    _validated_root_preparation_binding(sealed, plumbed_exp.domains[0])

    # Headers written before the field existed fail open to the importer
    # default; the binding above is what still refuses real mismatches.
    assert _sealed_root_rrtmg_variant({"namelist_sha256": "a" * 64}) is None
    assert _sealed_root_rrtmg_variant(
        {"domain_config": {"run": []}}) is None


def test_coupled_legacy_import_is_admitted_by_the_radiation_slice(tmp_path):
    """The B-04 structural refusal, closed at the canonicalization point.

    The WRF namelist importer emits a coupled 4/4 request as the
    historical aggregate spelling -- ra_physics=4 with the split fields
    at their -1 defaults -- so comparing RAW fields made the slice's
    (4, 4) admission unreachable for every namelist-imported experiment:
    the check demanded ra_physics=0 AND an explicit pair at once, and its
    error text advertised a case no import could satisfy.  The slice now
    compares the RESOLVED pair through woof.config.radiation_scheme_ids,
    the production resolver; emission is untouched (sealed bytes stay
    sealed).

    The resolver admits the aggregate restated on both streams for the
    same reason: it is one selection written twice, and only a
    disagreement between the spellings is refused.
    """

    from woof.config import radiation_scheme_ids
    from woof.experiment import load_experiment
    from woof.hrrr_hierarchy_direct import _native_experiment
    from woof.hrrr_route_inputs import target_domain
    from tools import battery_wrf_node_plan as node_plan

    repo = Path(__file__).resolve().parents[1]
    config = (repo / "configs" / "battery"
              / "shape_3km_thompson_rrtmg_legacy.toml")
    outdir = tmp_path / "node"
    node_plan.build(config, outdir, ranks=24, repository_root=repo)
    exp, _resolved, _report = _native_experiment(
        outdir / "namelist.wps", outdir / "namelist.native.input",
        rrtmg_variant="rrtmg_legacy")
    run = exp.domains[0].run
    # The aggregate spelling, exactly as imported.
    assert (run.ra_physics, run.ra_lw_physics, run.ra_sw_physics) \
        == (4, -1, -1)
    target = target_domain(load_experiment(config))

    _supported_hierarchy_slice(exp, target, forcing_hours=tuple(range(25)))

    # CONTROL 1: the aggregate spelling of radiation OFF resolves to
    # (0, 0), preserving the declared inactive radiation operation.
    radiation_off = replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, ra_physics=0))
        for domain in exp.domains))
    _supported_hierarchy_slice(radiation_off, target,
                               forcing_hours=tuple(range(25)))

    # CONTROL 2: the imported aggregate RESTATED on both streams is the
    # same selection written twice, so it resolves to (4, 4) and the
    # slice admits it.  Refusing this was refusing a caller for agreeing
    # with the importer.
    restated = replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, ra_lw_physics=4,
                                    ra_sw_physics=4))
        for domain in exp.domains))
    assert all(radiation_scheme_ids(domain.run) == (4, 4)
               for domain in restated.domains)
    _supported_hierarchy_slice(restated, target,
                               forcing_hours=tuple(range(25)))

    # CONTROL 3: a spelling that CONTRADICTS itself (the aggregate names
    # one engine, the explicit pair another) is refused by the resolver
    # itself, by name.
    with pytest.raises(ValueError, match="contradict each other"):
        incoherent = replace(exp, domains=tuple(
            replace(domain, run=replace(domain.run, ra_lw_physics=1,
                                        ra_sw_physics=1))
            for domain in exp.domains))
        _supported_hierarchy_slice(incoherent, target,
                                   forcing_hours=tuple(range(25)))


def test_stock_comparison_allows_only_explicit_longwave_runtime_change():
    native = _native()
    stock = replace(native, name="stock", domains=tuple(
        replace(domain, run=replace(domain.run, ra_lw_physics=1))
        for domain in native.domains))
    _compare_stock_experiment(native, stock)

    physics_drift = replace(stock, domains=(
        stock.domains[0],
        replace(stock.domains[1], run=replace(
            stock.domains[1].run, sf_surface_physics=0)),
    ))
    with pytest.raises(ValueError, match="beyond the allowed"):
        _compare_stock_experiment(native, physics_drift)

    with pytest.raises(ValueError, match="must select ra_lw_physics=1"):
        _compare_stock_experiment(native, replace(stock, domains=(
            stock.domains[0], native.domains[1])))


def test_stock_comparison_rejects_forecast_duration_drift():
    native = _native()
    stock = replace(native, name="stock", run_seconds=15.0, domains=tuple(
        replace(domain, run=replace(domain.run, ra_lw_physics=1))
        for domain in native.domains))
    with pytest.raises(ValueError, match="beyond the allowed"):
        _compare_stock_experiment(native, stock)


@pytest.mark.parametrize("max_dom", [1, 2, 21])
def test_raw_namelist_gate_allows_only_explicit_runtime_deltas(
        tmp_path, max_dom):
    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native.write_text(
        _raw_runtime_namelist(max_dom, longwave=0, theta_m=0),
        encoding="ascii")
    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=0, ghg_input=0, do_radar_ref=1),
        encoding="ascii")
    receipt = _require_raw_stock_delta(native, stock)
    assert set(receipt["allowed_deltas"]) == {
        "physics.ra_lw_physics",
        "physics.ghg_input", "physics.do_radar_ref"}
    # Not a delta: both halves declare the export's dry theta.
    assert receipt["shared_dry_theta"] == {
        "dynamics.use_theta_m": [EXPORT_USE_THETA_M]}
    assert receipt["schema"] == "gpuwm-native-to-stock-namelist-delta-v5"
    assert receipt["max_dom"] == max_dom

    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=0, ghg_input=0, do_radar_ref=1,
            run_hours=6),
        encoding="ascii")
    with pytest.raises(ValueError, match="time_control/run_hours"):
        _require_raw_stock_delta(native, stock)

    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=0, ghg_input=1, do_radar_ref=1),
        encoding="ascii")
    with pytest.raises(ValueError, match="stock-only ghg_input=0"):
        _require_raw_stock_delta(native, stock)

    # do_radar_ref is MANDATORY in the stock half, at 1.  Omitted, the
    # stock arm writes history frames with no REFL_10CM and every
    # reflectivity score on it is unanswerable; at 0 the switch is
    # present and says the wrong thing.  Both refuse.
    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=0, ghg_input=0),
        encoding="ascii")
    with pytest.raises(ValueError, match="do_radar_ref"):
        _require_raw_stock_delta(native, stock)

    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=0, ghg_input=0, do_radar_ref=0),
        encoding="ascii")
    with pytest.raises(ValueError, match="stock-only do_radar_ref=1"):
        _require_raw_stock_delta(native, stock)

    # And it stays FORBIDDEN in the native half: woof does not read it,
    # so a native namelist claiming to control REFL_10CM is claiming
    # something the arm that reads the file ignores.
    native.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=0, theta_m=0, do_radar_ref=1),
        encoding="ascii")
    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=0, ghg_input=0, do_radar_ref=1),
        encoding="ascii")
    with pytest.raises(ValueError, match="do_radar_ref must be omitted"):
        _require_raw_stock_delta(native, stock)


@pytest.mark.parametrize("native_theta, stock_theta", [
    (0, 1), (1, 1), (1, 0)])
def test_raw_namelist_gate_refuses_a_moist_theta_half(
        tmp_path, native_theta, stock_theta):
    """Both halves declare dry theta, and the refusal says what breaks.

    The stock arm's files are the direct export, which holds dry theta
    and says so in its header.  A stock namelist that says 1 (every pair
    generated before the export wrote dry theta) stops wrf.exe at its
    input gate, so the route names that before it prepares anything.
    """

    assert EXPORT_USE_THETA_M == 0
    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native.write_text(
        _raw_runtime_namelist(2, longwave=0, theta_m=native_theta),
        encoding="ascii")
    stock.write_text(
        _raw_runtime_namelist(
            2, longwave=1, theta_m=stock_theta, ghg_input=0,
            do_radar_ref=1),
        encoding="ascii")
    with pytest.raises(ValueError) as refusal:
        _require_raw_stock_delta(native, stock)
    text = str(refusal.value)
    assert "use_theta_m = 0 (dry theta)" in text
    assert f"native [{native_theta}], stock [{stock_theta}]" in text
    assert "use_theta_m values must be consistent" in text
    assert "regenerate the pair" in text


@pytest.mark.parametrize(
    "old, new, key",
    [
        ("interval_seconds = 3600", "interval_seconds = 1800",
         "interval_seconds"),
        ("io_form_input = 2", "io_form_input = 3", "io_form_input"),
        ("isfflx = 1", "isfflx = 0", "isfflx"),
        ("num_soil_layers = 4", "num_soil_layers = 9",
         "num_soil_layers"),
        ("interval_seconds = 3600", "interval_seconds = 3600.0",
         "interval_seconds"),
        ("restart = .false.", "restart = 0", "restart"),
        ("sfcp_to_sfcp = .true.", "sfcp_to_sfcp = 1",
         "sfcp_to_sfcp"),
        ("io_form_input = 2", "io_form_input = 2.0", "io_form_input"),
        ("isfflx = 1", "isfflx = 1.0", "isfflx"),
    ],
)
def test_raw_namelist_gate_rejects_dropped_runtime_drift(
        tmp_path, old, new, key):
    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native_text = _raw_runtime_namelist(4, longwave=0, theta_m=0)
    stock_text = _raw_runtime_namelist(
        4, longwave=1, theta_m=0, ghg_input=0, do_radar_ref=1)
    assert old in native_text and old in stock_text
    native.write_text(native_text.replace(old, new), encoding="ascii")
    stock.write_text(stock_text.replace(old, new), encoding="ascii")
    with pytest.raises(ValueError, match=key):
        _require_raw_stock_delta(native, stock)


@pytest.mark.parametrize("land_surface,soil_layers,refused", [
    (2, 4, False), (2, 9, True), (3, 9, False), (3, 4, True), (4, 4, False), (4, 9, True)])
def test_the_soil_pin_follows_the_land_surface(tmp_path, land_surface, soil_layers, refused):
    """Four layers for Noah and Noah-MP, nine for RUC, one column count for the tree.

    The contract pinned four whatever the land surface, so a RUC tree (the sub-km default on a nested
    --source hrrr domain) was refused after the download and the root preparation.
    """

    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native.write_text(_raw_runtime_namelist(
        3, longwave=0, theta_m=0, land_surface=land_surface, soil_layers=soil_layers), encoding="ascii")
    stock.write_text(_raw_runtime_namelist(
        3, longwave=1, theta_m=0, ghg_input=0, do_radar_ref=1, land_surface=land_surface,
        soil_layers=soil_layers), encoding="ascii")
    if refused:
        with pytest.raises(ValueError, match="num_soil_layers"):
            _require_raw_stock_delta(native, stock)
        return
    receipt = _require_raw_stock_delta(native, stock)
    assert receipt["certified_native_runtime"]["physics.num_soil_layers"] == [soil_layers]


def test_the_soil_pin_refuses_a_tree_without_one_land_surface(tmp_path):
    """The pinned column count is the land surface's, so a tree must name one scheme on every domain."""

    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    good_native = _raw_runtime_namelist(2, longwave=0, theta_m=0)
    good_stock = _raw_runtime_namelist(2, longwave=1, theta_m=0, ghg_input=0, do_radar_ref=1)
    for old, new in ((" sf_surface_physics = 2, 2,\n", " sf_surface_physics = 2, 3,\n"),
                     (" sf_surface_physics = 2, 2,\n", "")):
        assert old in good_native and old in good_stock
        native.write_text(good_native.replace(old, new), encoding="ascii")
        stock.write_text(good_stock.replace(old, new), encoding="ascii")
        # The refusal names what breaks without one land surface, not only what it needs.
        with pytest.raises(ValueError, match="sf_surface_physics.*no single soil column to pin"):
            _require_raw_stock_delta(native, stock)


def test_the_route_asks_for_an_optional_stock_wrf_export():
    """A missing ORACLE file must not destroy a prepared hierarchy.

    ``native_hierarchy`` offers required/optional/off and defaults to
    ``required``; ``gfs_direct`` asks for ``optional``.  This route
    passed nothing and inherited ``required``, so it was the only one
    that discarded a complete, verified GPU hierarchy when the
    unchanged-WRF file set could not represent the state -- which a
    single sub-freezing soil node anywhere in the domain is enough to
    cause.  Read from the call site itself because the whole
    orchestration below it needs a sealed root preparation to run; the
    behaviour of each mode is pinned in tests/test_native_hierarchy.py.
    """
    import ast
    import inspect
    from woof import hrrr_hierarchy_direct

    tree = ast.parse(inspect.getsource(hrrr_hierarchy_direct))
    modes = [
        keyword.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "initialize_and_export_native_hierarchy"
        for keyword in node.keywords
        if keyword.arg == "stock_wrf_export"
    ]
    assert modes == ["stock_wrf_export"]
    assert inspect.signature(
        hrrr_hierarchy_direct.prepare_hrrr_hierarchy
    ).parameters["stock_wrf_export"].default == "optional"


def test_raw_wps_gate_binds_hourly_hierarchy_contract(tmp_path):
    wps = tmp_path / "namelist.wps"
    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 3600,\n/\n",
        encoding="ascii")
    assert _require_raw_wps_contract(wps, 4)["status"] == "PASS"

    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 1800,\n/\n",
        encoding="ascii")
    with pytest.raises(ValueError, match="interval_seconds"):
        _require_raw_wps_contract(wps, 4)

    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 3600.0,\n/\n",
        encoding="ascii")
    with pytest.raises(ValueError, match="interval_seconds"):
        _require_raw_wps_contract(wps, 4)


def test_the_hierarchy_prints_the_digest_its_own_chain_promises(
        tmp_path, monkeypatch, capsys):
    """The one placeholder in the emitted chain no stage used to fill.

    `woof domain --source hrrr` closes a multi-domain emission with
    ``--preparation-receipt-sha256 <printed by the hierarchy>``, and the
    forecast runner checks that digest against ``receipt.json``.  Walking
    that printed route end to end found the hierarchy never printed it,
    so the chain could not be completed without hashing a file by hand.
    """

    output_root = tmp_path / "hierarchy"

    def _fake(*, output_root: Path, **_ignored):
        output_root.mkdir(parents=True)
        (output_root / "receipt.json").write_text(
            '{"status": "PASS"}\n', encoding="utf-8")
        return {
            "status": "PASS",
            "workers": 8,
            "timing_seconds": {"total": 1.0},
            "wrf_manifest": {"files": {}},
        }

    monkeypatch.setattr(
        hrrr_hierarchy_direct, "prepare_hrrr_hierarchy", _fake)
    assert hrrr_hierarchy_direct.main([
        "--root-preparation", str(tmp_path / "root"),
        "--root-domain-spec", str(tmp_path / "d01-target.json"),
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--namelist-input", str(tmp_path / "namelist.input"),
        "--stock-wrf-namelist-input", str(tmp_path / "stock.namelist.input"),
        "--geog-root", str(tmp_path / "geog"),
        "--source-manifest", str(tmp_path / "SHA256SUMS"),
        "--source-manifest-sha256", "0" * 64,
        "--valid-time", "2020-01-01_00:00:00",
        "--output-root", str(output_root),
    ]) == 0

    printed = json.loads(capsys.readouterr().out)
    expected = hashlib.sha256(
        (output_root / "receipt.json").read_bytes()).hexdigest()
    assert printed["preparation_receipt_sha256"] == expected


@pytest.mark.parametrize("perturbed", [False, True],
                         ids=["no-block", "deferred-block"])
def test_the_hierarchy_receipt_relays_the_root_s_perturbation_deferral(
        tmp_path, monkeypatch, perturbed):
    """The receipt the tree runner binds records a deferred bubble.

    This stage builds the children from namelists and never reads the
    configuration, so the root preparation, which does, seals the
    [perturbation] deferral into its identity and this stage relays it
    into receipt.json, beside the identity every domain's cache carries
    under ``root_preparation``.  Without a block the receipt is the one
    it always was.
    """
    from types import SimpleNamespace
    from woof.experiment import (
        BubbleConfig, DEFERRED_PERTURBATION_SCHEMA, PerturbationConfig)
    from test_hrrr_native_static import _fixture

    @dataclass(frozen=True)
    class _Tree:
        root: object
        domains: list
        start_time: datetime
        run_seconds: float
        perturbation: object = None

    join = hrrr_hierarchy_direct
    preparation = tmp_path / "sealed"
    preparation.mkdir()
    target, cache, static_receipt = _fixture(preparation)
    authority = tmp_path / "authority"
    authority.write_text("fixture", encoding="utf-8")
    digest = join.sha256_file(authority)
    day = datetime(2026, 8, 25, 18)
    bubbles = PerturbationConfig(bubbles=(BubbleConfig(
        center_lat=39.5, center_lon=-98.5, center_height_m=1500.0,
        radius_km=10.0, depth_m=1500.0, amplitude_k=0.01),))
    deferral = {"schema": DEFERRED_PERTURBATION_SCHEMA,
                "status": "DEFERRED_TO_FORECAST_INITIALIZATION",
                "config": bubbles.receipt()}
    source_identity = {"initial_perturbation": deferral} if perturbed else {}
    identity = {"source_identity": source_identity,
                "source_manifest_sha256": digest, "bridge_manifest_sha256": digest,
                "static_cache_sha256": join.sha256_file(cache), "namelist_sha256": digest,
                "forcing_hours": [0, 1]}
    header = {"schema": "gpuwm-prepared-real-cache-v1", "status": "READY",
              "identity": identity, "content_sha256": digest,
              "metadata": {"user": {"initial_valid_time": day.isoformat()}}}
    report = {"status": "PASS", "source_identity": source_identity,
              "prepared_cache": {"content_sha256": digest}}
    monkeypatch.setattr(join, "_root_paths", lambda root: {
        "static_cache": cache, "static_receipt": static_receipt,
        "prepared_cache": preparation, "bridge": preparation,
        "bridge_manifest": authority, "preparation_report": authority})
    monkeypatch.setattr(join, "_json", lambda path:
                        header if path.name == "header.json" else report)
    monkeypatch.setattr(join, "resolve_cpu_bridge", lambda path: authority)
    monkeypatch.setattr(join, "_require_raw_stock_delta", lambda *a:
                        {"certified_native_runtime": {"domains.sfcp_to_sfcp": [True]}})
    run = SimpleNamespace(sf_surface_physics=2, num_soil_layers=4, mp_physics=6)
    root = SimpleNamespace(run=run, grid_id=1)
    exp = _Tree(root=root, domains=[root, root], start_time=day,
                run_seconds=60.)
    monkeypatch.setattr(join, "_native_experiment",
                        lambda *a, **k: (exp, "fixture", _Run()))
    monkeypatch.setattr(join, "publish_hierarchy_forecast_companions", lambda *a, **k: {})
    monkeypatch.setattr(join, "load_hrrr_target_domain", lambda path: target)
    monkeypatch.setattr(join, "_supported_hierarchy_slice", lambda *a, **k: None)
    monkeypatch.setattr(join, "validated_corridor_selection", lambda *a: None)
    monkeypatch.setattr(join, "_require_raw_wps_contract", lambda *a: {})
    monkeypatch.setattr(join, "_expected_root_cache_identity",
                        lambda *a, **k: identity)
    monkeypatch.setattr(join, "PreparedCacheReader", lambda *a, **k:
                        SimpleNamespace(verify_all=lambda: None))
    monkeypatch.setattr(join, "restore_prepared_cache", lambda *a, **k: SimpleNamespace(
        initial_result=SimpleNamespace(state=None), met=None, boundaries=None,
        receipt={"content_sha256": digest}))
    monkeypatch.setattr(join, "_surface_state", lambda *a, **k: None)
    monkeypatch.setattr(join, "sealed_source_leads", lambda *a: (0, 1))
    monkeypatch.setattr(join, "load_hrrr_native_series",
                        lambda *a, **k: (object(),))
    monkeypatch.setattr(join, "verified_static_catalog", lambda *a:
                        (SimpleNamespace(files=()), {"selections": {"d01": None}}))
    monkeypatch.setattr(join, "grids_from_projection_config",
                        lambda exp: (target.grid(),))
    monkeypatch.setattr(join, "ParentInitView", lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(join, "NestedInputCatalog",
                        lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(join, "_source_identity", lambda *a: {})
    built = []

    def initialize(**kwargs):
        built.append((kwargs["source_identity"], kwargs["exp"]))
        return SimpleNamespace(timings_seconds={},
                               artifacts=SimpleNamespace(receipt={}),
                               wrf_manifest={})

    monkeypatch.setattr(join, "initialize_and_export_native_hierarchy", initialize)
    monkeypatch.setattr(join, "emit_statics_corridor_set", lambda **k: None)
    output = tmp_path / "joined"
    result = join.prepare_hrrr_hierarchy(
        root_preparation=preparation, root_domain_spec=authority,
        wps_namelist=authority, namelist_input=authority,
        stock_wrf_namelist_input=authority, geog_root=preparation,
        source_manifest=authority, source_manifest_sha256=digest,
        valid_time=day, output_root=output, workers=1)

    published = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    ((hierarchy_identity, exported),) = built
    assert hierarchy_identity["root_preparation"] == source_identity
    if perturbed:
        assert published["initial_perturbation"] == deferral
        assert result["initial_perturbation"] == deferral
        # The export sees the tree's bubbles, so the companion WRF set
        # refuses the state it cannot hold, as on every other source.
        assert exported.perturbation == bubbles
    else:
        assert "initial_perturbation" not in published
        assert "initial_perturbation" not in result
        assert exported is exp
