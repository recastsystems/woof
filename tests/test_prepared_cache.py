"""CPU-only integrity tests for the prepared real-data cache container."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import woof.ingest.prepared_cache as prepared_cache_module
from woof.core.grid import BaseState, make_vertical_coord
from woof.ingest.lateral_bc import (
    BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
)
from woof.ingest.prepared_cache import (
    _prepared_cache_staging_path,
    _restore_lbc_mode,
    PreparedCacheCorruptError, PreparedCacheMismatchError,
    PreparedCacheReader, extend_prepared_cache,
    prepared_cache_identity,
    select_prepared_met_fields, write_prepared_cache,
)
from woof.io.restart import STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS


def _fixture():
    coord = make_vertical_coord(2, hybrid_opt=0)
    base = BaseState(
        mub=np.full((2, 2), 90_000.0), p_top=10_000.0,
        pb=np.full((2, 2, 2), 50_000.0),
        alb=np.full((2, 2, 2), 0.8),
        thb=np.full((2, 2, 2), 290.0),
        phb=np.zeros((3, 2, 2)), terrain_z=np.zeros((2, 2)))
    side = SideBoundary(
        np.arange(4, dtype=np.float64).reshape(1, 2, 2),
        np.full((1, 2, 2), 0.25, dtype=np.float64))
    boundaries = LateralBoundaries((BoundaryInterval(
        0.0, 3600.0, {"u": FieldBoundary(side, side, side, side)}),),
        5, 1, 4)
    state = SimpleNamespace(u=np.arange(12, dtype=np.float32).reshape(3, 2, 2))
    for index, name in enumerate(STATE_SETUP_ARRAYS):
        setattr(state, name, np.array([index], dtype=np.float32))
    scalar_values = {
        "mub": None, "p_top": 10_000.0,
        "cf1": 1.0, "cf2": 2.0, "cf3": 3.0,
        "cfn": 4.0, "cfn1": 5.0,
        "has_msf": True, "rotational": True,
    }
    assert set(scalar_values) == set(STATE_SETUP_SCALARS)
    for name, value in scalar_values.items():
        setattr(state, name, value)
    state.lateral_boundaries = boundaries
    initial = SimpleNamespace(
        state=state, coord=coord, base=base,
        surface_pressure=np.full((2, 2), 99_000.0),
        surface_qv=np.full((2, 2), 0.01))
    surface = np.ones((2, 2), dtype=np.float32)
    met = SimpleNamespace(fields={
        "LANDSEA": surface, "SKINTEMP": 280.0 * surface,
        "SOILT": np.ones((9, 2, 2), dtype=np.float32),
        "SOILW": np.full((9, 2, 2), 0.2, dtype=np.float32),
        "T2": 279.0 * surface,
        "U10": np.ones((2, 3), dtype=np.float32),
        "V10": np.ones((3, 2), dtype=np.float32),
    })
    return initial, met, boundaries


def _extension_identity(*, source_hours, model_start, domain_start,
                        bridge, source_manifest):
    model_hours = list(range(len(source_hours)))
    namelist_sha256 = "d" * 64 if max(source_hours) == 1 else "e" * 64
    return {
        "bridge_manifest_sha256": bridge,
        "source_manifest_sha256": source_manifest,
        "static_cache_sha256": "c" * 64,
        "namelist_sha256": namelist_sha256,
        "namelist_extension_invariant": {
            "schema": "gpuwm-namelist-extension-invariant-v1",
            "sha256": "1" * 64,
        },
        "domain_config": {
            "start_time": domain_start.isoformat(),
            "run": {
                "run_seconds": float(len(model_hours) - 1) * 3600.0,
                "specified": True,
                "nested": False,
            },
        },
        "forcing_hours": model_hours,
        "source_identity": {
            "adapter": "fixture",
            "source_cycle": datetime(2026, 7, 20).isoformat(),
            "model_start_time": model_start.isoformat(),
            "source_forecast_hours": list(source_hours),
            "model_forcing_hours": model_hours,
        },
    }


def _suffix_fixture(*, seam_delta=0.0):
    initial, met, old = _fixture()
    old_side = old.intervals[0].fields["u"].west
    endpoint = old_side.value + 3600.0 * old_side.tendency + seam_delta
    suffix_side = SideBoundary(
        endpoint, np.full_like(endpoint, 0.5, dtype=np.float64))
    suffix = LateralBoundaries((BoundaryInterval(
        0.0, 3600.0,
        {"u": FieldBoundary(
            suffix_side, suffix_side, suffix_side, suffix_side)}),),
        5, 1, 4)
    initial.state.lateral_boundaries = suffix
    return initial, met, suffix


def _manifest_extension(prior_identity, new_identity):
    return {
        "schema": "gpuwm-source-manifest-prefix-extension-v1",
        "predecessor_sha256": prior_identity["source_manifest_sha256"],
        "extended_sha256": new_identity["source_manifest_sha256"],
        "old_source_forecast_hours": [0, 1],
        "new_source_forecast_hours": [0, 1, 2],
        "suffix_source_forecast_hours": [1, 2],
        "retained_entries": 4,
        "added_entries": ["atmosphere-f02", "surface-f02"],
    }


def _bridge_extension(prior_identity, suffix_identity, new_identity):
    return {
        "schema": "gpuwm-bridge-manifest-prefix-extension-v1",
        "predecessor_sha256": prior_identity["bridge_manifest_sha256"],
        "suffix_sha256": suffix_identity["bridge_manifest_sha256"],
        "extended_sha256": new_identity["bridge_manifest_sha256"],
        "old_source_forecast_hours": [0, 1],
        "new_source_forecast_hours": [0, 1, 2],
        "suffix_source_forecast_hours": [1, 2],
        "retained_entries": 48,
        "added_entries": ["atmosphere-f02", "soil-f02"],
    }


def test_prepared_identity_serializes_per_domain_start_time():
    @dataclass(frozen=True)
    class Domain:
        start_time: datetime

    identity = prepared_cache_identity(
        bridge_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        static_cache_sha256="c" * 64,
        namelist_sha256="d" * 64,
        domain_config=Domain(datetime(2026, 7, 20, 0, 5)),
        forcing_offsets_seconds=(0, 300),
        source_identity={"adapter": "fixture"})

    assert identity["domain_config"]["start_time"] == \
        "2026-07-20T00:05:00"
    assert identity["forcing_offsets_seconds"] == [0, 300]


def test_nested_lbc_restore_opt_in_is_identity_bound_and_root_safe():
    child_identity = {
        "domain_config": {
            "parent_id": 1,
            "run": {"nested": True, "specified": False},
        },
    }
    root_identity = {
        "domain_config": {
            "parent_id": 0,
            "run": {"nested": False, "specified": True},
        },
    }

    assert _restore_lbc_mode(
        lbc_metadata=None, identity=child_identity,
        allow_nested_without_lbc=True) == "nested-parent-forced"
    assert _restore_lbc_mode(
        lbc_metadata={}, identity=root_identity,
        allow_nested_without_lbc=False) == "external"
    with pytest.raises(PreparedCacheMismatchError, match="standalone"):
        _restore_lbc_mode(
            lbc_metadata=None, identity=child_identity,
            allow_nested_without_lbc=False)
    with pytest.raises(PreparedCacheMismatchError, match="standalone"):
        _restore_lbc_mode(
            lbc_metadata=None, identity=root_identity,
            allow_nested_without_lbc=True)
    with pytest.raises(TypeError, match="must be bool"):
        _restore_lbc_mode(
            lbc_metadata={}, identity=root_identity,
            allow_nested_without_lbc=1)


def test_prepared_cache_staging_path_is_compact_and_target_independent(
        tmp_path):
    short = _prepared_cache_staging_path(
        tmp_path / "prepared", nonce="012345abcd")
    long = _prepared_cache_staging_path(
        tmp_path / ("prepared-" + "x" * 120), nonce="012345abcd")

    assert short == tmp_path / ".p-012345abcd"
    assert long == short
    assert len(long.name) == 13


@pytest.mark.parametrize(
    "nonce", ("short", "012345ABCD", "012345abcg", 1234))
def test_prepared_cache_staging_path_rejects_unsafe_explicit_nonce(
        tmp_path, nonce):
    with pytest.raises(ValueError, match="10 lowercase hex"):
        _prepared_cache_staging_path(tmp_path / "prepared", nonce=nonce)


def test_prepared_cache_paths_fit_failed_windows_parent_budget():
    parent = PureWindowsPath("C:\\" + "x" * 225)
    target = parent / "prepared-cache"
    staging = _prepared_cache_staging_path(
        target, nonce="012345abcd")

    assert len(str(parent)) == 228
    assert len(str(target)) == 243
    assert len(str(staging)) == 242
    assert len(str(staging / "a00000.npy")) == 253
    assert len(str(staging / "header.json")) == 254
    assert len(str(target / "a00000.npy")) == 254
    assert len(str(target / "header.json")) == 255


def test_a_native_hrrr_soil_rides_the_cache_as_the_canonical_surface(
        tmp_path):
    """The REAL writer, fed the REAL native soil derivation, satisfies
    the runner's exact surface demand.

    Field 2026-08-06, first portable HRRR case: the runner's preflight
    refused with 'prepared cache lacks the exact source-neutral Noah
    surface inventory' because the native preparation wrote its cache
    with no ``surface=`` at all -- while the suite that guards the
    reader had INVENTED the canonical inventory in its own fixture
    header instead of exercising this writer.  This test derives the
    surface the way the preparation now does
    (``preprocess_land_surface_soil`` -> ``canonical_noah_surface``),
    writes a real cache, and compares the header against the runner's
    own frozenset -- the very comparison at the preflight gate -- plus
    the writer's promised drop of the legacy native soil pair.
    """
    import json

    from woof.ingest.ruc_soil import preprocess_land_surface_soil
    from woof.native_wrf_contract import canonical_noah_surface
    from woof.prepared_single_domain_forecast import (
        _CANONICAL_SURFACE_FIELDS)

    initial, met, boundaries = _fixture()
    shape = met.fields["LANDSEA"].shape
    fields = dict(met.fields)
    # Physically plausible native soil profiles: the derivation is the
    # certified one and it validates its inputs' physical ranges.
    nodes = fields["SOILT"].shape[0]
    fields["SOILT"] = np.stack(
        [np.full(shape, 290.0 - 2.0 * k, dtype=np.float32)
         for k in range(nodes)])
    fields["SOILW"] = np.stack(
        [np.full(shape, 0.10 + 0.02 * k, dtype=np.float32)
         for k in range(nodes)])
    fields.setdefault("XICE", np.zeros(shape, dtype=np.float32))
    fields.setdefault("SNOW", np.zeros(shape, dtype=np.float32))
    soil = preprocess_land_surface_soil(
        fields, sf_surface_physics=2,
        soil_type=np.full(shape, 6.0),
        deep_soil_temperature=np.full(shape, 281.0))
    surface = canonical_noah_surface(soil)
    assert set(surface) == _CANONICAL_SURFACE_FIELDS

    receipt = write_prepared_cache(
        tmp_path / "cache", identity={"source": "hrrr-native"},
        initial_result=initial,
        met=SimpleNamespace(fields=MappingProxyType(fields)),
        boundaries=boundaries, surface=surface)
    assert receipt["status"] == "BUILT"
    header = json.loads(
        (tmp_path / "cache" / "header.json").read_text(encoding="utf-8"))
    metadata = header["metadata"]
    # The preflight gate's own comparison, verbatim.
    assert set(metadata["surface_fields"]) == _CANONICAL_SURFACE_FIELDS
    # The legacy native soil pair no longer rides in met: the canonical
    # surface IS the soil statement of a portable cache.
    assert set(metadata["met_fields"]).isdisjoint({"SOILT", "SOILW"})
    # Every canonical array is really in the payload, restorable by name.
    for name in sorted(_CANONICAL_SURFACE_FIELDS):
        assert f"surface/{name}" in header["arrays"]


def test_default_prepared_cache_bytes_ignore_disabled_seal_option(tmp_path):
    initial, met, boundaries = _fixture()
    first = tmp_path / "implicit-default"
    second = tmp_path / "explicit-default"

    one = write_prepared_cache(
        first, identity={"source": "unchanged"}, initial_result=initial,
        met=met, boundaries=boundaries)
    two = write_prepared_cache(
        second, identity={"source": "unchanged"}, initial_result=initial,
        met=met, boundaries=boundaries, sealed_forcing_extension=False)

    assert one["content_sha256"] == two["content_sha256"]
    first_header = PreparedCacheReader(
        first, expected_identity={"source": "unchanged"}).header
    second_header = PreparedCacheReader(
        second, expected_identity={"source": "unchanged"}).header
    for key in ("schema", "status", "identity", "metadata", "arrays",
                "content_sha256", "payload_bytes"):
        assert first_header[key] == second_header[key]
    assert "forcing_extension_mode" not in first_header["metadata"]
    for key, spec in first_header["arrays"].items():
        other = second_header["arrays"][key]
        assert (first / spec["file"]).read_bytes() == \
            (second / other["file"]).read_bytes()


def _extension_identities():
    start = datetime(2026, 7, 20)
    prior_identity = _extension_identity(
        source_hours=[0, 1], model_start=start, domain_start=start,
        bridge="a" * 64, source_manifest="b" * 64)
    suffix_identity = _extension_identity(
        source_hours=[1, 2], model_start=start + timedelta(hours=1),
        domain_start=start + timedelta(hours=1), bridge="e" * 64,
        source_manifest="f" * 64)
    extended_identity = _extension_identity(
        source_hours=[0, 1, 2], model_start=start, domain_start=start,
        bridge="9" * 64, source_manifest="f" * 64)
    return prior_identity, suffix_identity, extended_identity


def test_prepared_cache_extension_reuses_prefix_and_appends_nonzero_hour(
        tmp_path):
    prior_identity, suffix_identity, extended_identity = _extension_identities()
    composite = "9" * 64
    initial, met, boundaries = _fixture()
    suffix_initial, suffix_met, suffix_boundaries = _suffix_fixture()
    prior = tmp_path / "prior"
    suffix = tmp_path / "suffix"
    output = tmp_path / "extended"
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=boundaries, sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_boundaries)

    receipt = extend_prepared_cache(
        output, predecessor=prior, suffix=suffix,
        identity=extended_identity, metadata={"forcing_hours": [0, 1, 2]},
        source_manifest_extension=_manifest_extension(
            prior_identity, extended_identity),
        bridge_manifest_extension=_bridge_extension(
            prior_identity, suffix_identity, extended_identity))

    reader = PreparedCacheReader(output, expected_identity=extended_identity)
    assert reader.verify_all()["status"] == "PASS"
    assert len(reader.header["metadata"]["lbc"]["intervals"]) == 2
    assert receipt["appended_interval"] == [3600.0, 7200.0]
    assert receipt["bridge_manifest_sha256"] == composite
    assert receipt["predecessor_payloads"] == "linked"
    prior_reader = PreparedCacheReader(prior, expected_identity=prior_identity)
    for key, old_spec in prior_reader.arrays.items():
        new_spec = reader.arrays[key]
        assert Path(prior / old_spec["file"]).samefile(
            output / new_spec["file"])
    np.testing.assert_array_equal(
        reader.read_array("lbc/0/u/west/value"),
        prior_reader.read_array("lbc/0/u/west/value"))
    np.testing.assert_array_equal(
        reader.read_array("lbc/1/u/west/value"),
        PreparedCacheReader(
            suffix, expected_identity=suffix_identity).read_array(
                "lbc/0/u/west/value"))


def test_prepared_cache_extension_refuses_changed_seam_without_output(
        tmp_path):
    prior, prior_identity, arguments = _extension_inputs(
        tmp_path, seam_delta=1.0)
    output = tmp_path / "extended"

    with pytest.raises(PreparedCacheMismatchError, match="shared endpoint"):
        extend_prepared_cache(output, **arguments)
    assert not output.exists()


def test_prepared_cache_extension_refuses_changed_namelist_invariant(
        tmp_path):
    prior, prior_identity, arguments = _extension_inputs(tmp_path)
    arguments["identity"]["namelist_extension_invariant"]["sha256"] = "2" * 64
    output = tmp_path / "extended"

    with pytest.raises(
            PreparedCacheMismatchError, match="immutable namelist fields"):
        extend_prepared_cache(output, **arguments)
    assert not output.exists()


def test_prepared_cache_extension_refuses_predecessor_mutation(tmp_path):
    prior, prior_identity, arguments = _extension_inputs(tmp_path)
    output = tmp_path / "extended"
    prior_reader = PreparedCacheReader(prior, expected_identity=prior_identity)
    payload = prior / prior_reader.arrays["state/u"]["file"]
    array = np.load(payload, allow_pickle=False)
    array.flat[0] += 1.0
    with payload.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)

    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        extend_prepared_cache(output, **arguments)
    assert not output.exists()


def test_prepared_cache_extension_rechecks_hardlinked_stage_for_toctou(
        tmp_path, monkeypatch):
    prior, prior_identity, arguments = _extension_inputs(tmp_path)
    output = tmp_path / "extended"
    original = prepared_cache_module._BundleWriter.link_verified
    injected = False

    def mutate_after_link(writer, key, reader):
        nonlocal injected
        original(writer, key, reader)
        if key == "state/u" and not injected:
            injected = True
            payload = reader.path / reader.arrays[key]["file"]
            array = np.load(payload, allow_pickle=False)
            array.flat[-1] += 1.0
            with payload.open("wb") as stream_handle:
                np.save(stream_handle, array, allow_pickle=False)

    monkeypatch.setattr(
        prepared_cache_module._BundleWriter, "link_verified",
        mutate_after_link)

    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        extend_prepared_cache(output, **arguments)
    assert injected
    assert not output.exists()


def _extension_inputs(tmp_path, *, seam_delta=0.0):
    prior_identity, suffix_identity, extended_identity = _extension_identities()
    initial, met, boundaries = _fixture()
    suffix_initial, suffix_met, suffix_boundaries = _suffix_fixture(
        seam_delta=seam_delta)
    prior, suffix = tmp_path / "prior", tmp_path / "suffix"
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=boundaries, sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_boundaries)
    arguments = dict(
        predecessor=prior, suffix=suffix, identity=extended_identity,
        metadata={},
        source_manifest_extension=_manifest_extension(
            prior_identity, extended_identity),
        bridge_manifest_extension=_bridge_extension(
            prior_identity, suffix_identity, extended_identity))
    return prior, prior_identity, arguments


def _drive_without_hard_links(monkeypatch):
    """os.link answering as exFAT does (Windows winerror 1, Linux EPERM)."""
    import errno
    import os

    calls = []

    def link(source, destination, *args, **kwargs):
        calls.append(Path(destination))
        if os.name == "nt":
            raise OSError(errno.EINVAL, "Incorrect function", str(source), 1,
                          str(destination))
        raise OSError(errno.EPERM, "Operation not permitted", str(source),
                      None, str(destination))

    monkeypatch.setattr(os, "link", link)
    return calls


def test_prepared_cache_extension_on_a_drive_without_hard_links_copies(
        tmp_path, monkeypatch, capsys):
    """exFAT has no hard links.  The extension copies the earlier hours
    instead, proves each copy byte-identical, verifies the whole staged
    cache against its manifest, and says in one line that the copy costs
    disk.  It used to refuse, citing only that cost."""
    from woof import explain

    monkeypatch.setattr(explain, "_PRINTED_ONCE", set())
    prior, prior_identity, arguments = _extension_inputs(tmp_path)
    before = {path.name: path.read_bytes() for path in prior.iterdir()}
    calls = _drive_without_hard_links(monkeypatch)
    output = tmp_path / "extended"

    receipt = extend_prepared_cache(output, **arguments)

    assert receipt["status"] == "BUILT"
    assert receipt["predecessor_payloads"] == "copied"
    assert receipt["appended_interval"] == [3600.0, 7200.0]
    # One link attempt: the first refusal switches the rest to copying.
    assert len(calls) == 1
    reader = PreparedCacheReader(output, expected_identity=arguments["identity"])
    assert reader.verify_all()["status"] == "PASS"
    prior_reader = PreparedCacheReader(prior, expected_identity=prior_identity)
    for key, old_spec in prior_reader.arrays.items():
        copied = output / reader.arrays[key]["file"]
        assert copied.read_bytes() == (prior / old_spec["file"]).read_bytes()
        assert not copied.samefile(prior / old_spec["file"])
    assert {path.name: path.read_bytes() for path in prior.iterdir()} == before
    note = capsys.readouterr().err
    assert note.count("copied rather than linked") == 1
    assert "disk space" in note
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "extended", "prior", "suffix"]


def test_prepared_cache_extension_refuses_a_copy_the_disk_cannot_hold(
        tmp_path, monkeypatch):
    """The copy is priced before its first byte: the whole earlier cache
    against the free space, refused by name when it does not fit."""
    from woof import filesystem_paths

    _prior, _identity, arguments = _extension_inputs(tmp_path)
    _drive_without_hard_links(monkeypatch)
    monkeypatch.setattr(filesystem_paths, "_free_bytes", lambda folder: 1024)
    output = tmp_path / "extended"

    with pytest.raises(filesystem_paths.CopyWouldNotFitError) as refused:
        extend_prepared_cache(output, **arguments)

    message = str(refused.value)
    assert "prepared cache" in message and "1.0 KiB free" in message
    assert not output.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prior", "suffix"]


def test_prepared_cache_extension_refuses_a_gap(tmp_path):
    prior, prior_identity, arguments = _extension_inputs(tmp_path)
    arguments["identity"]["forcing_hours"] = [0, 1, 3]
    output = tmp_path / "extended"

    with pytest.raises(PreparedCacheMismatchError, match="exactly one"):
        extend_prepared_cache(output, **arguments)
    assert not output.exists()


def _header(path):
    import json

    return json.loads((path / "header.json").read_text(encoding="utf-8"))


def _cleared_out_extension(tmp_path, *, predecessor_built=True,
                           seam_delta=0.0):
    """A predecessor hour whose ``u`` table clears out at its end, and a
    suffix hour from that frame (moved by ``seam_delta``), each interval
    from the builder, which records the frame its tendency was built
    toward.  ``predecessor_built=False`` drops that record, as a cache
    written before 2.8.1 carries none."""

    from dataclasses import replace

    from woof.ingest.lateral_bc import build_lateral_interval_from_sides

    prior_identity, suffix_identity, extended_identity = _extension_identities()
    rng = np.random.default_rng(5)

    def frame(values):
        values = np.asarray(values, dtype=np.float32).astype(np.float64)
        return {side: {"u": values}
                for side in ("west", "east", "south", "north")}

    first = frame(rng.uniform(1e-5, 1e-3, (1, 5, 12)))
    cleared = frame(np.zeros((1, 5, 12)))
    last = frame(rng.uniform(1e-5, 1e-3, (1, 5, 12)))
    old = build_lateral_interval_from_sides(
        first, cleared, start_seconds=0.0, end_seconds=3600.0)
    if not predecessor_built:
        old = replace(old)
    new = build_lateral_interval_from_sides(
        frame(np.full((1, 5, 12), seam_delta)), last,
        start_seconds=0.0, end_seconds=3600.0)
    initial, met, _ = _fixture()
    initial.state.lateral_boundaries = LateralBoundaries((old,), 5, 1, 4)
    suffix_initial, suffix_met, _ = _fixture()
    suffix_initial.state.lateral_boundaries = LateralBoundaries(
        (new,), 5, 1, 4)
    prior, suffix, output = (
        tmp_path / "prior", tmp_path / "suffix", tmp_path / "extended")
    write_prepared_cache(
        prior, identity=prior_identity, initial_result=initial, met=met,
        boundaries=initial.state.lateral_boundaries,
        sealed_forcing_extension=True)
    write_prepared_cache(
        suffix, identity=suffix_identity, initial_result=suffix_initial,
        met=suffix_met, boundaries=suffix_initial.state.lateral_boundaries)
    arguments = dict(
        predecessor=prior, suffix=suffix, identity=extended_identity,
        metadata={"forcing_hours": [0, 1, 2]},
        source_manifest_extension=_manifest_extension(
            prior_identity, extended_identity),
        bridge_manifest_extension=_bridge_extension(
            prior_identity, suffix_identity, extended_identity))
    return output, arguments, extended_identity


def test_prepared_cache_extension_joins_a_frame_that_clears_out(tmp_path):
    """A140b: the predecessor's last row records the frame its tendency
    was built toward, which is the suffix's start frame, so an hour that
    clears a hydrometeor out at the join extends; the rebuilt identity read
    the same join as a changed endpoint frame."""

    from woof.state_serialization_contract import (
        BUILT_END_FRAME_PREFIX_SCHEMA)

    output, arguments, identity = _cleared_out_extension(tmp_path)
    prior = _header(arguments["predecessor"])
    prior_rows = prior["metadata"]["lbc"]["intervals"]
    assert set(prior_rows[0]) == {
        "start_seconds", "end_seconds", "fields", "end_frame_sha256"}
    assert prior["metadata"]["lateral_boundary_prefix"]["schema"] \
        == BUILT_END_FRAME_PREFIX_SCHEMA

    extend_prepared_cache(output, **arguments)

    reader = PreparedCacheReader(output, expected_identity=identity)
    assert reader.verify_all()["status"] == "PASS"
    stored = reader.header["metadata"]["lateral_boundary_prefix"]
    assert stored["schema"] == BUILT_END_FRAME_PREFIX_SCHEMA
    boundaries = prepared_cache_module._reader_boundaries(reader)
    assert prepared_cache_module._forcing_prefix(boundaries) == stored
    rows = stored["intervals"]
    assert rows[0]["end_frame_sha256"] == rows[1]["start_frame_sha256"]
    assert [row.get("end_frame_sha256")
            for row in reader.header["metadata"]["lbc"]["intervals"]] == [
        row["end_frame_sha256"] for row in rows]
    from woof.state_serialization_contract import (
        lateral_boundary_prefix_identity)
    rebuilt = lateral_boundary_prefix_identity(
        SimpleNamespace(lateral_boundaries=boundaries),
        rebuilt_end_frames=True)["intervals"]
    assert rebuilt[0]["end_frame_sha256"] != rebuilt[1]["start_frame_sha256"]


def test_prepared_cache_extension_refuses_a_real_discontinuity_at_a_clear_out(
        tmp_path):
    output, arguments, _ = _cleared_out_extension(tmp_path, seam_delta=1e-4)
    with pytest.raises(PreparedCacheMismatchError,
                       match="shared endpoint frame$"):
        extend_prepared_cache(output, **arguments)
    assert not output.exists()


def test_a_cache_written_before_end_frames_extends_in_the_rebuilt_identity(
        tmp_path):
    """A predecessor that records no end frame (every cache written before
    2.8.1) still reads and extends, hashed in the rebuilt identity, as the
    existing extension tests show.  A clear-out at its join is refused by
    name, because that identity cannot tell it from another frame."""

    output, arguments, _ = _cleared_out_extension(
        tmp_path, predecessor_built=False)
    prior = _header(arguments["predecessor"])
    assert set(prior["metadata"]["lbc"]["intervals"][0]) == {
        "start_seconds", "end_seconds", "fields"}
    with pytest.raises(PreparedCacheMismatchError,
                       match="rebuilt end-frame identity"):
        extend_prepared_cache(output, **arguments)
    assert not output.exists()


def test_prepared_cache_staging_collision_preserves_foreign_tree(
        tmp_path, monkeypatch):
    initial, met, boundaries = _fixture()
    target = tmp_path / "prepared"
    staging = tmp_path / ".p-012345abcd"
    staging.mkdir()
    sentinel = staging / "foreign.txt"
    sentinel.write_text("owned elsewhere", encoding="utf-8")
    monkeypatch.setattr(
        prepared_cache_module, "_prepared_cache_staging_path",
        lambda _path: staging)

    with pytest.raises(FileExistsError):
        write_prepared_cache(
            target, identity={"source": "abc"}, initial_result=initial,
            met=met, boundaries=boundaries)

    assert sentinel.read_text(encoding="utf-8") == "owned elsewhere"
    assert not target.exists()


def test_prepared_cache_mid_write_failure_removes_only_owned_staging(
        tmp_path, monkeypatch):
    initial, met, boundaries = _fixture()
    target = tmp_path / "prepared"
    staging = tmp_path / ".p-012345abcd"
    monkeypatch.setattr(
        prepared_cache_module, "_prepared_cache_staging_path",
        lambda _path: staging)

    def injected_write_failure(_writer, _key, _value):
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(
        prepared_cache_module._BundleWriter, "add",
        injected_write_failure)

    with pytest.raises(RuntimeError, match="injected write failure"):
        write_prepared_cache(
            target, identity={"source": "abc"}, initial_result=initial,
            met=met, boundaries=boundaries)

    assert not staging.exists()
    assert not target.exists()


def test_prepared_cache_publication_race_preserves_competing_target(
        tmp_path, monkeypatch):
    initial, met, boundaries = _fixture()
    target = tmp_path / "prepared"
    staging = tmp_path / ".p-012345abcd"
    monkeypatch.setattr(
        prepared_cache_module, "_prepared_cache_staging_path",
        lambda _path: staging)

    def competing_publish(_source, destination):
        destination = Path(destination)
        destination.mkdir()
        (destination / "foreign.txt").write_text(
            "competing publisher", encoding="utf-8")
        raise FileExistsError("injected publication race")

    monkeypatch.setattr(
        prepared_cache_module.os, "replace", competing_publish)

    with pytest.raises(FileExistsError, match="injected publication race"):
        write_prepared_cache(
            target, identity={"source": "abc"}, initial_result=initial,
            met=met, boundaries=boundaries)

    assert not staging.exists()
    assert (target / "foreign.txt").read_text(encoding="utf-8") == (
        "competing publisher")


def test_prepared_cache_round_trip_verifies_every_array(tmp_path):
    initial, met, boundaries = _fixture()
    identity = {"source": "abc", "config": {"nx": 2}}
    path = tmp_path / "prepared"
    receipt = write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=boundaries, metadata={"forcing_hours": [0, 1]})

    reader = PreparedCacheReader(path, expected_identity=identity)
    verified = reader.verify_all()
    assert receipt["status"] == "BUILT"
    assert verified["status"] == "PASS"
    assert verified["content_sha256"] == receipt["content_sha256"]
    assert verified["array_count"] == receipt["array_count"]
    assert verified["payload_bytes"] == receipt["payload_bytes"]
    assert all(
        spec["file"].startswith("a")
        and spec["file"].endswith(".npy")
        and len(spec["file"]) == 10
        for spec in reader.arrays.values())


def test_prepared_cache_refuses_identity_drift_and_payload_corruption(tmp_path):
    initial, met, boundaries = _fixture()
    identity = {"source": "abc"}
    path = tmp_path / "prepared"
    write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=boundaries)

    # The refusal names the release that wrote the cache and the exact
    # fields, rather than asserting that something differs: the old
    # wording sent a test host 7 pilot to their experiment TOML after a
    # package upgrade had changed the identity document.
    with pytest.raises(PreparedCacheMismatchError,
                       match="these identity fields differ: source"):
        PreparedCacheReader(path, expected_identity={"source": "changed"})

    reader = PreparedCacheReader(path, expected_identity=identity)
    payload = path / reader.arrays["state/u"]["file"]
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 0x01
    payload.write_bytes(raw)
    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        PreparedCacheReader(
            path, expected_identity=identity).read_array("state/u")


def test_prepared_cache_never_overwrites_valid_bundle(tmp_path):
    initial, met, boundaries = _fixture()
    path = tmp_path / "prepared"
    write_prepared_cache(
        path, identity={"source": "abc"}, initial_result=initial, met=met,
        boundaries=boundaries)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_prepared_cache(
            path, identity={"source": "abc"}, initial_result=initial,
            met=met, boundaries=boundaries)


def test_nested_child_cache_may_omit_external_lbc_for_wrfinput_export(tmp_path):
    initial, met, _boundaries = _fixture()
    identity = {
        "domain_config": {
            "grid_id": 2,
            "parent_id": 1,
            "run": {"nested": True, "specified": False},
        },
    }
    path = tmp_path / "prepared-child"

    receipt = write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=None)
    reader = PreparedCacheReader(path, expected_identity=identity)

    assert receipt["status"] == "BUILT"
    assert reader.header["metadata"]["lbc"] is None
    assert not any(name.startswith("lbc/") for name in reader.arrays)
    assert reader.verify_all()["status"] == "PASS"


def test_child_cache_writes_a_restored_mappingproxy_hydrometeor_record(
        tmp_path):
    """The exact v1.3.1 crash, reproduced through the type that caused it.

    ``restore_prepared_cache`` hands back
    ``CachedInitialResult.hydrometeor_initialization`` as a
    ``MappingProxyType`` on purpose -- a restored document must not be
    mutable.  ``write_prepared_cache`` then copies that record into the
    child cache it derives, through ``_json_copy -> _canonical ->
    json.dumps``, which serializes ``dict`` and nothing that merely
    behaves like one: "TypeError: Object of type mappingproxy is not JSON
    serializable", from inside a nested hierarchy publication, naming no
    field.  Nesting one proxy inside another is deliberate: a fix that
    only unwrapped the top level would still crash on the real payload.
    """

    initial, met, boundaries = _fixture()
    record = MappingProxyType({
        "schema": "gpuwm-hrrr-microphysics-initialization-v3",
        "state_source_absent_fields": MappingProxyType({
            "qnr": MappingProxyType({"expected_float32": 0.0}),
        }),
        "source_mass_fields": ("QC", "QR"),
    })
    initial = SimpleNamespace(
        **vars(initial), hydrometeor_initialization=record)
    identity = {
        "domain_config": {
            "grid_id": 2,
            "parent_id": 1,
            "run": {"nested": True, "specified": False},
        },
    }
    path = tmp_path / "prepared-child-mappingproxy"

    receipt = write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=None)
    reader = PreparedCacheReader(path, expected_identity=identity)

    assert receipt["status"] == "BUILT"
    stored = reader.header["metadata"]["hydrometeor_initialization"]
    assert stored == {
        "schema": "gpuwm-hrrr-microphysics-initialization-v3",
        "state_source_absent_fields": {"qnr": {"expected_float32": 0.0}},
        # Tuples normalize to lists exactly as they always did.
        "source_mass_fields": ["QC", "QR"],
    }
    # A proxy and its underlying mapping must hash the same, or the same
    # prepared state would carry two content digests depending on which
    # object the caller happened to hold.
    assert prepared_cache_module._canonical(record) \
        == prepared_cache_module._canonical(dict(stored))
    # An unordered container still has no canonical serialization.
    with pytest.raises(TypeError, match="cannot contain a set"):
        prepared_cache_module._canonical({"species": {"QC", "QR"}})


@pytest.mark.parametrize("identity", (
    {"source": "root"},
    {"domain_config": {"parent_id": 0,
                       "run": {"nested": False, "specified": True}}},
    {"domain_config": {"parent_id": 1,
                       "run": {"nested": True, "specified": True}}},
))
def test_prepared_cache_refuses_lbc_omission_without_nested_identity(
        tmp_path, identity):
    initial, met, _boundaries = _fixture()
    path = tmp_path / "invalid-child"
    with pytest.raises(ValueError, match="identity-bound nested"):
        write_prepared_cache(
            path, identity=identity, initial_result=initial, met=met,
            boundaries=None)
    assert not path.exists()


def test_prepared_cache_accepts_source_neutral_canonical_surface(tmp_path):
    initial, met, boundaries = _fixture()
    fields = dict(met.fields)
    fields.pop("SOILT")
    fields.pop("SOILW")
    met = SimpleNamespace(fields=fields)
    plane = np.ones((2, 2), dtype=np.float32)
    surface = {
        "TSK": 280.0 * plane,
        "TSLB": np.full((4, 2, 2), 279.0, dtype=np.float32),
        "SMOIS": np.full((4, 2, 2), 0.2, dtype=np.float32),
        "SH2O": np.full((4, 2, 2), 0.2, dtype=np.float32),
        "TMN": 278.0 * plane,
        "SEAICE": np.zeros((2, 2), dtype=np.float32),
        "XLAND": plane,
        "LANDMASK": plane,
        "SNOW": np.zeros((2, 2), dtype=np.float32),
        "SNOWH": np.zeros((2, 2), dtype=np.float32),
    }
    path = tmp_path / "prepared"
    identity = {"source": "era5"}
    write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=boundaries, surface=surface)
    reader = PreparedCacheReader(path, expected_identity=identity)
    assert set(reader.header["metadata"]["surface_fields"]) == set(surface)
    np.testing.assert_array_equal(
        reader.read_array("surface/TSLB"), surface["TSLB"])


def test_select_prepared_met_fields_detaches_exact_physics_contract():
    _, met, _ = _fixture()
    fields = dict(met.fields)
    fields.update({
        "SST": np.full((2, 2), 281.0, dtype=np.float32),
        "PRES": np.full((2, 2, 2), 80_000.0, dtype=np.float32),
    })
    met = SimpleNamespace(fields=fields)

    selected = select_prepared_met_fields(met)

    assert set(selected.fields) == {
        "LANDSEA", "SKINTEMP", "SOILT", "SOILW", "SST", "T2", "U10",
        "V10",
    }
    assert "PRES" not in selected.fields
    for name, value in selected.fields.items():
        assert value.flags.c_contiguous
        assert not np.shares_memory(value, fields[name])
        np.testing.assert_array_equal(value, fields[name])
    with pytest.raises(TypeError):
        selected.fields["PRES"] = fields["PRES"]

    fields["T2"][...] = -1.0
    np.testing.assert_array_equal(
        selected.fields["T2"], np.full((2, 2), 279.0, dtype=np.float32))


def test_select_prepared_met_fields_keeps_legacy_soil_only_when_required():
    _, met, _ = _fixture()
    canonical_surface = {"placeholder": object()}

    selected = select_prepared_met_fields(
        met, surface=canonical_surface)

    assert "SOILT" not in selected.fields
    assert "SOILW" not in selected.fields


# ---------------------------------------------------------------------------
# V-12: a package upgrade must not make every existing prepared tree
# unrunnable, and must never blame the user's experiment file for it
# ---------------------------------------------------------------------------
# v1.1.0 gave every domain an optional per-domain `start_time` for
# staggered nest starts.  The prepared-cache identity is compared by
# strict equality, and a v1.0.1 header was serialized before the field
# existed, so after upgrading the wheel EVERY prepared tree in the field
# refused with "d01 cache domain config differs from experiment" -- a
# sentence naming the experiment TOML, which was innocent.  A test host 7
# validation run diffed the two documents on a real preserved 1.0.1
# tree: eleven cached top-level keys against twelve live ones, exactly
# one added key, and zero value differences among the eleven shared keys
# or the ~110 `run` fields.  These tests mirror that shape.


def _live_experiment():
    from woof.experiment import load_experiment

    return load_experiment(
        Path(__file__).parents[1] / "configs"
        / "gfs_wrf_hierarchy_proof.toml")


def _live_domain_identity():
    """The live 12-key identity, from the shipped two-domain config."""

    from woof.ingest.prepared_cache import prepared_domain_config_identity

    return prepared_domain_config_identity(_live_experiment().root)


def _undelayed():
    """What "no delayed start" looks like for this experiment."""

    from woof.ingest.prepared_cache import undelayed_identity_defaults

    return undelayed_identity_defaults(_live_experiment())


def test_a_v101_shape_header_still_binds_after_the_upgrade():
    from woof.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    # 11 v1.0.1 keys + start_time (v1.1.0) + spawn (v1.8, dormant
    # spawn-triggered nests) + tiles (df5cf42d0) + output (8d8855a8c) +
    # retire/rearm/follow (8e663a751/21254c056) + WRF's history window,
    # history_begin_s/history_end_s (cb8af5ccb).  Every post-v1.0.1 field
    # must appear in BOTH the live document and the tolerance table, or
    # this pin moves.  The lifecycle five landed WITHOUT tolerance entries
    # and re-opened the V-12 hole; the entries were added when this pin
    # caught up with them (2026-08-30).  The window pair repeated it: the
    # single-domain runner and the stock-WRF export refused every bundle
    # prepared before it, 2.8.0's included, until its entries were added
    # (A168).
    post_v101 = ("start_time", "spawn", "tiles", "output",
                 "retire", "rearm", "follow",
                 "history_begin_s", "history_end_s")
    assert len(live) == 20
    for field in post_v101:
        assert field in live
    cached = {key: value for key, value in live.items()
              if key not in post_v101}
    assert len(cached) == 11

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert differing == []
    assert sorted(tolerated) == sorted(post_v101)


def test_a_field_absent_from_the_header_but_IN_USE_still_refuses():
    """The narrowness is what makes tolerating the absence accurate."""

    from woof.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    live["start_time"] = "2026-07-20T03:00:00"
    cached = {key: value for key, value in live.items() if key != "start_time"}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert tolerated == []
    assert differing == ["start_time"]


def test_a_header_from_before_the_history_window_binds():
    """A168: the window pair is tolerated absent at its default on the raw
    walk (the stock-WRF export's comparison), and dropped at any value
    once both documents are normalised, because preparation reads
    neither key."""

    from woof.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config)

    window = ("history_begin_s", "history_end_s")
    live = _live_domain_identity()
    assert (live["history_begin_s"], live["history_end_s"]) == (0.0, None)
    cached = {key: value for key, value in live.items()
              if key not in window}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert sorted(tolerated) == sorted(window)
    assert differing == []

    windowed = {**live, "history_begin_s": 3600.0}
    _, differing = compare_prepared_domain_config(
        cached, windowed, not_in_use=_undelayed())
    assert differing == ["history_begin_s"]
    _, differing = compare_prepared_domain_config(
        effective_prepared_domain_config(cached),
        effective_prepared_domain_config(windowed), not_in_use=_undelayed())
    assert differing == []


def test_a_real_configuration_change_is_still_refused():
    """Tolerance is about absent fields, never about differing values."""

    from woof.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    cached = {key: value for key, value in live.items() if key != "start_time"}
    cached["run"] = {**cached["run"], "nx": int(cached["run"]["nx"]) + 1}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert tolerated == ["start_time"]
    assert differing == ["run.nx"]


def test_a_header_from_a_NEWER_gpuwm_is_refused_not_tolerated():
    """Absence in the live document is skew in the other direction."""

    from woof.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    cached = {**live, "a_field_this_build_does_not_have": 1}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert tolerated == []
    assert differing == ["a_field_this_build_does_not_have"]


def test_acoustic_off_centering_reuses_prepared_arrays_without_ignoring_grid_changes():
    from copy import deepcopy
    from woof.ingest.prepared_cache import compare_prepared_domain_config
    cached = _live_domain_identity()
    configured = deepcopy(cached)
    cached["run"]["epssm"] = 0.1
    configured["run"]["epssm"] = 0.2
    assert compare_prepared_domain_config(cached, configured) == ([], [])
    configured["run"]["dx"] += 1.0
    assert "run.dx" in compare_prepared_domain_config(cached, configured)[1]


def test_prior_atmosphere_header_can_enable_fire_without_dropping_moisture_or_grid_identity():
    from copy import deepcopy
    from dataclasses import fields
    from woof.sfire_config import FireRunFields
    from woof.ingest.prepared_cache import (
        SFIRE_PREPARATION_INERT_RUN_FIELDS, compare_prepared_domain_config)
    assert SFIRE_PREPARATION_INERT_RUN_FIELDS == frozenset(f"run.{field.name}" for field in fields(FireRunFields))
    cached = _live_domain_identity()
    configured = deepcopy(cached)
    for key in SFIRE_PREPARATION_INERT_RUN_FIELDS:
        cached["run"].pop(key.removeprefix("run."), None)
    configured["run"].update(ifire=2, sr_x=5, sr_y=5, fire_static="fire-static.npz",
        fire_smoke=True, fire_is_real_perim=True, fmoist_run=True)
    assert compare_prepared_domain_config(cached, configured) == ([], [])
    cached["run"]["fire_moisture_namelist"] = ""
    assert compare_prepared_domain_config(cached, configured) == ([], [])
    cached["run"]["fire_moisture_namelist"] = "unavailable-moisture-state"
    assert "run.fire_moisture_namelist" in compare_prepared_domain_config(cached, configured)[1]
    cached["run"].pop("fire_moisture_namelist")
    configured["run"]["moist"] = not cached["run"]["moist"]
    assert "run.moist" in compare_prepared_domain_config(cached, configured)[1]


def test_only_the_domain_config_is_default_tolerant():
    """Every other identity member is a hash of bytes and stays strict."""

    from woof.ingest.prepared_cache import compare_prepared_identity

    live = _live_domain_identity()
    expected = {
        "domain_config": live,
        "namelist_sha256": "a" * 64,
        "static_cache_sha256": "b" * 64,
    }
    cached = {
        "domain_config": {k: v for k, v in live.items() if k != "start_time"},
        "namelist_sha256": "a" * 64,
        "static_cache_sha256": "b" * 64,
    }
    assert compare_prepared_identity(
        cached, expected, not_in_use=_undelayed()) == (["start_time"], [])

    # A missing digest is not schema growth; it is a different cache.
    del cached["static_cache_sha256"]
    tolerated, differing = compare_prepared_identity(
        cached, expected, not_in_use=_undelayed())
    assert differing == ["static_cache_sha256"]


def test_the_refusal_names_the_versions_and_the_fields():
    """Never again a message that blames the experiment file."""

    from woof import __version__
    from woof.ingest.prepared_cache import (
        CACHE_WRITER_KEY, UNSTAMPED_WRITER, prepared_identity_refusal,
    )

    unstamped = prepared_identity_refusal(
        subject="d01 prepared cache", header={},
        differing=["run.nx", "start_time"])
    assert "d01 prepared cache" in unstamped
    assert UNSTAMPED_WRITER in unstamped
    assert f"woof {__version__}" in unstamped
    assert "run.nx, start_time" in unstamped
    # It never points at the experiment file for a package difference.
    assert "differs from experiment" not in unstamped

    stamped = prepared_identity_refusal(
        subject="d01 prepared cache",
        header={CACHE_WRITER_KEY: {"gpuwm_version": "1.0.1"}},
        differing=["run.nx"], re_prepare="rw-wps --source gfs ...")
    assert "prepared by 1.0.1" in stamped
    assert "Re-prepare it with: rw-wps --source gfs ..." in stamped


def test_a_cache_written_now_stamps_the_release_that_wrote_it():
    """So the next schema change can say which release wrote the bundle."""

    from woof import __version__
    from woof.ingest.prepared_cache import (
        CACHE_WRITER_KEY, cache_writer_version,
    )

    assert cache_writer_version({}) != __version__
    assert cache_writer_version(
        {CACHE_WRITER_KEY: {"gpuwm_version": __version__}}) == __version__


# ---------------------------------------------------------------------------
# The identity-table guard.  Four RunConfig fields in a row
# (mp28_aerosol_source, wif_climatology_path, p3_backend,
# ntiedtke_tiedtke_closure) joined RunConfig without joining
# DEFAULT_TOLERANT_IDENTITY_FIELDS, and each time every prepared tree
# written before the field was refused with "these identity fields differ:
# run.<name>".  The table's docstring asked the field's author to add the
# entry; four misses say the instruction is not where the enforcement
# belongs.  These tests pin today's RunConfig against the field set at the
# identity header's introduction, so a new field fails the suite until its
# author classifies it.

#: RunConfig's 70 fields at 1c6290410 (2026-07-19), the commit that
#: introduced the prepared-cache identity header binding
#: asdict(DomainConfig) -- and with it every run.* field -- into every
#: cache.  Every prepared tree ever written carries these, so they need no
#: classification.  Taken with ``git show 1c6290410:woof/config.py``.
_RUN_FIELDS_AT_IDENTITY_HEADER_INTRODUCTION = frozenset({
    "nx", "ny", "nz", "dx", "dy", "ztop", "dt", "run_seconds", "clock_dt",
    "p_surf", "time_step_sound", "epssm", "smdiv", "khdif", "kvdif",
    "damp_opt", "zdamp", "dampcoef", "output_interval_s", "case",
    "hybrid_opt", "etac", "moist", "mp_physics", "moist_adv_opt",
    "no_mp_heating", "mp_tend_lim", "diff_6th_opt", "diff_6th_factor",
    "diff_6th_slopeopt", "diff_6th_thresh", "km_opt", "c_s", "w_damping",
    "open_x", "open_y", "base_temp", "specified", "spec_bdy_width",
    "spec_zone", "relax_zone", "spec_exp", "emdiv", "h_sca_adv_order",
    "terrain_opt", "hill_height", "hill_halfwidth", "map_proj",
    "sf_sfclay_physics", "sf_surface_physics", "bl_pbl_physics",
    "ysu_topdown_pblmix", "ra_physics", "cu_physics", "radt_minutes",
    "cudt_minutes", "radt", "bldt", "hypsometric_opt", "restart_interval_s",
    "nested", "grid_id", "top_lid", "moist_cq", "morr_rimed_ice",
    "wsm6_hail_opt", "ra_lw_physics", "ra_sw_physics", "icloud",
    "swrad_scat",
})
_IDENTITY_HEADER_INTRODUCTION = "1c6290410"


def _run_config_field_names() -> frozenset[str]:
    from dataclasses import fields

    from woof.config import RunConfig

    return frozenset(field.name for field in fields(RunConfig))


def _run_paths(table) -> frozenset[str]:
    return frozenset(path for path in table if path.startswith("run."))


def _identity_tables():
    """Every table the comparison consults, as ``{name: run.* paths}``."""

    from woof.ingest import prepared_cache as module

    return {
        "DEFAULT_TOLERANT_IDENTITY_FIELDS":
            _run_paths(module.DEFAULT_TOLERANT_IDENTITY_FIELDS),
        "NON_TRAJECTORY_IDENTITY_FIELDS":
            _run_paths(module.NON_TRAJECTORY_IDENTITY_FIELDS),
        "INERT_DIAGNOSTIC_IDENTITY_FIELDS":
            _run_paths(module.INERT_DIAGNOSTIC_IDENTITY_FIELDS),
        "PREPARATION_INERT_RUN_FIELDS":
            _run_paths(module.PREPARATION_INERT_RUN_FIELDS),
        "STRICT_IDENTITY_FIELDS":
            _run_paths(module.STRICT_IDENTITY_FIELDS),
    }


def test_every_run_field_added_since_the_identity_header_is_classified():
    """A new RunConfig field fails here until its author rules on it."""

    from woof.ingest.prepared_cache import STRICT_IDENTITY_FIELDS

    current = _run_config_field_names()
    baseline = _RUN_FIELDS_AT_IDENTITY_HEADER_INTRODUCTION
    tables = _identity_tables()
    strict = tables["STRICT_IDENTITY_FIELDS"]
    # A dropped field (the three partitions the comparison never looks
    # at) tolerates absence a fortiori; a default-tolerant field
    # tolerates it at the default.  Either is a ruling.
    tolerant = frozenset().union(*(
        paths for name, paths in tables.items()
        if name != "STRICT_IDENTITY_FIELDS"))

    removed = sorted(baseline - current)
    assert not removed, (
        f"RunConfig lost {removed}, which every prepared tree ever written "
        f"carries in its header: the comparison will now refuse ALL of them "
        f"as 'a field the cache carries and this build does not'.  Removing "
        f"a header-era field needs its own ruling on older trees, not a "
        f"baseline edit.")

    for name in sorted(current - baseline):
        path = f"run.{name}"
        assert path in tolerant or path in strict, (
            f"RunConfig.{name} joined RunConfig after the prepared-cache "
            f"identity baseline ({_IDENTITY_HEADER_INTRODUCTION}) and is in "
            f"neither DEFAULT_TOLERANT_IDENTITY_FIELDS nor "
            f"STRICT_IDENTITY_FIELDS in woof/ingest/prepared_cache.py, so "
            f"every prepared tree written before it will be refused with "
            f"'these identity fields differ: {path}'.  Add {path!r} to "
            f"DEFAULT_TOLERANT_IDENTITY_FIELDS with the scoping argument "
            f"(its not-in-use default reproduces the pre-field prepared "
            f"state), or to STRICT_IDENTITY_FIELDS with the reason an older "
            f"tree really must be refused for it (it changes the prepared "
            f"initial state or the boundary tables).")

    # One ruling per field: a path in two tables has two contradictory
    # reasons, and the walk would honour whichever it consulted first.
    names = list(tables)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            both = sorted(tables[first] & tables[second])
            assert not both, f"{both} listed in both {first} and {second}"

    # A strict entry is a written reason, or it is nothing.
    for path, reason in STRICT_IDENTITY_FIELDS.items():
        assert path.startswith("run."), path
        assert isinstance(reason, str) and reason.strip(), (
            f"STRICT_IDENTITY_FIELDS[{path!r}] must say what the field "
            f"changes about the prepared state")

    # A table entry naming a field RunConfig no longer has is a stale
    # ruling: the field was renamed or removed and the table was not.
    for table, paths in tables.items():
        stale = sorted(path for path in paths if path[4:] not in current)
        assert not stale, (
            f"{table} names {stale}, but RunConfig has no such field: "
            f"renamed or removed, and the table was not updated with it.")


def test_tolerant_run_entries_carry_their_defaults_into_the_not_in_use_map():
    """An entry in the table that the not-in-use map lacks is dead.

    The walk tolerates a path only when it is BOTH in the table and in
    the caller's ``not_in_use`` map, so a run.* entry added to the table
    alone would refuse exactly as if it were absent.  Every run.* entry
    must therefore reach ``undelayed_identity_defaults`` with the
    dataclass default in the identity document's own JSON spelling.
    """

    from dataclasses import fields

    from woof.config import RunConfig
    from woof.ingest.prepared_cache import (
        DEFAULT_TOLERANT_IDENTITY_FIELDS, _json_copy,
    )

    defaults = {field.name: field.default for field in fields(RunConfig)}
    not_in_use = _undelayed()
    for path in sorted(_run_paths(DEFAULT_TOLERANT_IDENTITY_FIELDS)):
        name = path[4:]
        assert path in not_in_use, (
            f"{path} is in DEFAULT_TOLERANT_IDENTITY_FIELDS but not in "
            f"undelayed_identity_defaults(), so the walk never tolerates it")
        assert not_in_use[path] == _json_copy(defaults[name]), path


def _live_tolerant_run_paths_at_default(live) -> list[str]:
    """The run.* tolerant paths the shipped two-domain config leaves at
    their not-in-use value; stripping these from a header is exactly a
    tree prepared before they existed."""

    from woof.ingest.prepared_cache import DEFAULT_TOLERANT_IDENTITY_FIELDS

    not_in_use = _undelayed()
    return sorted(
        path for path in _run_paths(DEFAULT_TOLERANT_IDENTITY_FIELDS)
        if live["run"].get(path[4:]) == not_in_use[path])


def test_the_walk_tolerates_every_nested_run_entry_absent_from_a_header():
    """The table says its walk reaches run.*; this is the measurement.

    A header with NONE of the tolerant run.* fields is what a tree
    prepared at the identity header's introduction looks like, and it
    must bind against today's document with zero differing paths.
    """

    from woof.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    stripped = _live_tolerant_run_paths_at_default(live)
    # The shipped config leaves the overwhelming majority at default;
    # a config that set most of them would make this test vacuous.
    assert len(stripped) >= 75, stripped
    cached = {**live, "run": {
        key: value for key, value in live["run"].items()
        if f"run.{key}" not in stripped}}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert differing == []
    assert sorted(tolerated) == stripped


def test_a_nested_run_field_absent_from_the_header_but_IN_USE_still_refuses():
    """Narrowness, at the nested level: a non-default value is a change."""

    from woof.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    # Not in the shipped config's non-default set, and its default is
    # 1; 0 is a real selection the older tree was not prepared under.
    assert live["run"]["bl_mynn_edmf"] == 1
    live["run"] = {**live["run"], "bl_mynn_edmf": 0}
    cached = {**live, "run": {
        key: value for key, value in live["run"].items()
        if key != "bl_mynn_edmf"}}

    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert tolerated == []
    assert differing == ["run.bl_mynn_edmf"]


@pytest.mark.parametrize("name,default,selected", [
    ("afwa_buoy_opt", 0, 1),
    ("afwa_severe_opt", 0, 1),
    ("diag_print", 0, 1),
    ("waves", "off", "spectral"),
    ("waves_options", "", '{"wind_input": true}'),
    ("wtddt", 30.0, 15.0),
    ("snicar_snowshape_opt", 3, 1),
    ("snicar_snowbc_intmix", True, False),
    ("snicar_snowdust_intmix", False, True),
    ("snicar_bc_ppb", 0.0, 1.0),
    ("snicar_dust_ppm", 0.0, 1.0),
    ("snicar_ice_density", 900.0, 910.0),
    ("snicar_bubble_radius_um", 1000.0, 900.0),
])
@pytest.mark.parametrize("at_default", [True, False])
def test_pre_diagnostic_header_accepts_only_the_inactive_default(
        name, default, selected, at_default):
    """Older headers can omit a runtime control only at its prior default.

    A newly selected option still needs an explicit identity rather than being
    silently treated as the older header's inactive selection.
    """
    from woof.ingest.prepared_cache import compare_prepared_domain_config

    value = default if at_default else selected
    tolerated, differing = compare_prepared_domain_config(
        {"run": {}}, {"run": {name: value}}, not_in_use=_undelayed())
    path = f"run.{name}"
    assert tolerated == ([path] if at_default else [])
    assert differing == ([] if at_default else [path])


def _diagnostic_switch_values(name):
    """The not-in-use value of an output-only switch, and a set one."""

    from woof.config import RunConfig

    default = RunConfig.__dataclass_fields__[name].default
    return default, (True if isinstance(default, bool) else 1)


@pytest.mark.parametrize("name", ["tke_budget", "sase_flux_diag",
                                  "hmix_k_diag", "nwp_diagnostics"])
@pytest.mark.parametrize("switched_on", [True, False])
def test_an_output_only_switch_reuses_the_unchanged_cache(
        tmp_path, name, switched_on):
    """Switching a diagnostic on or off reads the cache it was prepared as.

    These switches choose which buffers a forecast writes; preparation
    reads none of them.  The cache used to refuse all but nwp_diagnostics
    by name ("these identity fields differ: run.hmix_k_diag") and demand a
    second preparation of identical arrays.
    """

    from woof.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config)

    default, chosen = _diagnostic_switch_values(name)
    before, after = (default, chosen) if switched_on else (chosen, default)
    written = _live_domain_identity()
    written["run"] = {**written["run"], name: before}
    live = {**written, "run": {**written["run"], name: after}}
    initial, met, boundaries = _fixture()
    path = tmp_path / "prepared"
    write_prepared_cache(
        path, identity={"source": "abc", "domain_config": written},
        initial_result=initial, met=met, boundaries=boundaries)
    original = PreparedCacheReader(
        path, expected_identity={"source": "abc", "domain_config": written})

    reader = PreparedCacheReader(
        path, expected_identity={"source": "abc", "domain_config": live})

    assert reader.verify_all()["status"] == "PASS"
    assert reader.header["identity"]["domain_config"]["run"][name] == before
    assert reader.tolerated_identity_fields == ()
    for key in original.arrays:
        np.testing.assert_array_equal(
            reader.read_array(key), original.read_array(key))
    # The tree runner's gate normalizes both sides first; same answer.
    assert compare_prepared_domain_config(
        effective_prepared_domain_config(written),
        effective_prepared_domain_config(live),
        not_in_use=_undelayed()) == ([], [])
    # A cache written before the switch existed lacks the key entirely.
    older = {**written, "run": {
        key: value for key, value in written["run"].items() if key != name}}
    assert compare_prepared_domain_config(
        older, live, not_in_use=_undelayed()) == ([], [])


@pytest.mark.parametrize("change", ["run.nz", "source"])
def test_an_output_only_switch_does_not_hide_a_preparation_change(
        tmp_path, change):
    written = _live_domain_identity()
    live = {**written, "run": {**written["run"], "hmix_k_diag": True,
                               "tke_budget": 1, "sase_flux_diag": True}}
    expected = {"source": "abc", "domain_config": live}
    if change == "run.nz":
        live["run"]["nz"] = written["run"]["nz"] + 1
    else:
        expected["source"] = "changed"
    initial, met, boundaries = _fixture()
    path = tmp_path / "prepared"
    write_prepared_cache(
        path, identity={"source": "abc", "domain_config": written},
        initial_result=initial, met=met, boundaries=boundaries)

    with pytest.raises(PreparedCacheMismatchError) as refused:
        PreparedCacheReader(path, expected_identity=expected)
    message = str(refused.value)
    assert f"these identity fields differ: {change}" in message
    assert "hmix_k_diag" not in message and "tke_budget" not in message


def test_the_inert_diagnostic_set_is_the_restart_table():
    """One table of output-only switches, read by restart and the cache."""

    from woof.checkpoint_identity import CONFIG_DIAGNOSTIC_FIELDS
    from woof.ingest.prepared_cache import INERT_DIAGNOSTIC_IDENTITY_FIELDS
    from woof.io import restart

    assert restart.CONFIG_DIAGNOSTIC_FIELDS is CONFIG_DIAGNOSTIC_FIELDS
    assert INERT_DIAGNOSTIC_IDENTITY_FIELDS == {
        f"run.{name}" for name in CONFIG_DIAGNOSTIC_FIELDS}


def test_rational_forcing_cache_preserves_time_law_and_setup_identity(tmp_path):
    from dataclasses import replace
    from woof.ingest.lateral_bc import RationalTimeLaw, evaluate_boundary_side
    initial, met, old = _fixture()
    source = old.intervals[0].fields['u'].west
    nonlinear = replace(source, time_law=RationalTimeLaw(
        np.full(source.value.shape, 0.001), np.full(source.value.shape, 0.0001)))
    fields = {'u':FieldBoundary(nonlinear, nonlinear, nonlinear, nonlinear)}
    boundaries = replace(old, intervals=(replace(old.intervals[0], fields=fields),))
    initial.state.lateral_boundaries = boundaries
    path = tmp_path/'rational'
    write_prepared_cache(path, identity={'source':'generic-time-law-test'},
        initial_result=initial, met=met, boundaries=boundaries)
    reader = PreparedCacheReader(path, expected_identity={'source':'generic-time-law-test'})
    reader.verify_all()
    restored = prepared_cache_module._reader_boundaries(reader)
    assert prepared_cache_module._forcing_prefix(restored) == prepared_cache_module._forcing_prefix(boundaries)
    actual = restored.intervals[0].fields['u'].west
    for t in (0., 900., 1800., 3600.):
        expected = evaluate_boundary_side(nonlinear, t)
        observed = evaluate_boundary_side(actual, t)
        np.testing.assert_array_equal(observed[0], expected[0])
        np.testing.assert_array_equal(observed[1], expected[1])


#: Run in a child interpreter whose import system refuses CuPy, so the
#: test states the CPU-only install whether or not this box has CuPy.
_HOST_RESTORE_WITHOUT_CUPY = r'''
import sys
from importlib.abc import MetaPathFinder


class RejectCupy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cupy" or fullname.startswith("cupy."):
            raise ModuleNotFoundError(f"blocked for this test: {fullname}")
        return None


sys.meta_path.insert(0, RejectCupy())

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from woof.config import RunConfig
from woof.core.grid import BaseState, make_vertical_coord
from woof.core.state import DomainState
from woof.ingest.lateral_bc import (
    attach_lateral_boundaries, build_state_lateral_boundaries)
from woof.ingest.prepared_cache import (
    restore_prepared_cache, write_prepared_cache)
from woof.state_serialization_contract import (
    STATE_DERIVED_SETUP_ARRAYS, STATE_SERIALIZED_ATTRS, STATE_SETUP_ARRAYS)

nz, ny, nx = 3, 12, 14
cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, ztop=15000.0,
                dt=10.0, run_seconds=3600.0, moist=True, mp_physics=6,
                terrain_opt=1)
coord = make_vertical_coord(nz, hybrid_opt=0)
base = BaseState(
    mub=np.full((ny, nx), 90_000.0), p_top=10_000.0,
    pb=np.linspace(95_000.0, 20_000.0, nz)[:, None, None]
    * np.ones((nz, ny, nx)),
    alb=np.full((nz, ny, nx), 0.9), thb=np.full((nz, ny, nx), 300.0),
    phb=np.linspace(0.0, 1.4e5, nz + 1)[:, None, None]
    * np.ones((nz + 1, ny, nx)),
    terrain_z=np.zeros((ny, nx)))
static = {
    "MAPFAC_M": np.full((ny, nx), 1.01), "MAPFAC_U": np.full((ny, nx + 1), 1.01),
    "MAPFAC_V": np.full((ny + 1, nx), 1.01), "F": np.full((ny, nx), 1.0e-4),
    "E": np.full((ny, nx), 5.0e-5), "SINALPHA": np.full((ny, nx), 0.1),
    "COSALPHA": np.full((ny, nx), np.sqrt(0.99))}

setup_only = set(STATE_SETUP_ARRAYS) | set(STATE_DERIVED_SETUP_ARRAYS)
rng = np.random.default_rng(20260928)


def prepared(offset):
    state = DomainState(cfg, array_module=np)
    state.load_base(coord, base)
    state.set_map_coriolis(
        static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
        static["F"], static["E"], sina=static["SINALPHA"],
        cosa=static["COSALPHA"])
    for name in STATE_SERIALIZED_ATTRS:
        array = getattr(state, name, None)
        if array is None or name in setup_only:
            continue
        array[...] = (offset + rng.standard_normal(array.shape)).astype(
            array.dtype)
    return state


state, later = prepared(0.0), prepared(1.0)
boundaries = build_state_lateral_boundaries([state, later], [0.0, 3600.0])
attach_lateral_boundaries(state, boundaries)
initial = SimpleNamespace(
    state=state, coord=coord, base=base,
    surface_pressure=np.full((ny, nx), 99_000.0),
    surface_qv=np.full((ny, nx), 0.01))
surface = np.ones((ny, nx), dtype=np.float32)
met = SimpleNamespace(fields={
    "LANDSEA": surface, "SKINTEMP": 280.0 * surface,
    "SOILT": np.full((9, ny, nx), 281.0, dtype=np.float32),
    "SOILW": np.full((9, ny, nx), 0.2, dtype=np.float32),
    "T2": 279.0 * surface, "U10": surface, "V10": surface})
identity = {"source": "host-restore-without-cupy"}
path = Path(sys.argv[1]) / "prepared-cache"
write_prepared_cache(path, identity=identity, initial_result=initial,
                     met=met, boundaries=boundaries)

restored = restore_prepared_cache(
    path, expected_identity=identity, cfg=cfg, static=static,
    array_module=np)
back = restored.initial_result.state
compared = 0
for name in STATE_SERIALIZED_ATTRS:
    written = getattr(state, name, None)
    if written is None:
        continue
    value = getattr(back, name)
    assert type(value) is np.ndarray, (name, type(value))
    assert value.dtype == written.dtype, name
    assert np.array_equal(value, written), name
    compared += 1
assert compared >= 10, compared
assert [(i.start_seconds, i.end_seconds)
        for i in restored.boundaries.intervals] == [(0.0, 3600.0)]
assert back.lateral_boundaries is restored.boundaries
assert restored.receipt["status"] == "RESTORED"
assert "cupy" not in sys.modules, "the host restore imported CuPy"
print("host-restore", compared)
'''


def test_a_prepared_root_restores_to_host_arrays_without_cupy(tmp_path):
    """The HRRR domain tree's root read works on a machine with no CuPy.

    The hierarchy stage builds its children on the CPU from the sealed
    root, and the restore it reads that root through imported CuPy
    unconditionally, so every HRRR tree preparation on a CPU-only
    install died with ``No module named 'cupy'`` before any child was
    made.  Written by the shipped writer from a real host state, read
    back by the shipped reader with CuPy unimportable: every serialized
    array comes back as the same NumPy bytes, the boundaries attach on
    the host, and the setup fingerprint check passes.
    """

    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", _HOST_RESTORE_WITHOUT_CUPY, str(tmp_path)],
        cwd=root, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.startswith("host-restore"), completed.stdout


@pytest.mark.parametrize("change", ["none", "map", "independent-inverse"])
def test_pre_inverse_prepared_cache_binds_old_words_and_derived_inverse(
        tmp_path, change):
    """A genuine old digest restores; different setup still refuses."""
    import subprocess
    import sys

    script = _HOST_RESTORE_WITHOUT_CUPY.replace(
        "write_prepared_cache(path, identity=identity, initial_result=initial,",
        """import woof.state_serialization_contract as contract
current_arrays = contract.STATE_SETUP_ARRAYS
contract.STATE_SETUP_ARRAYS = tuple(n for n in current_arrays if n != 'msfv_inv')
old_digest = contract.setup_fingerprint(state)
write_prepared_cache(path, identity=identity, initial_result=initial,""").replace(
        "restored = restore_prepared_cache(",
        """contract.STATE_SETUP_ARRAYS = current_arrays
assert contract.setup_fingerprint(state) != old_digest
if sys.argv[2] == 'map':
    static['F'] = static['F'] + 1.e-6
elif sys.argv[2] == 'independent-inverse':
    static['MF_VX_INV'] = np.nextafter(
        np.float32(1.) / np.asarray(static['MAPFAC_V'], dtype=np.float32),
        np.float32(0.))
restored = restore_prepared_cache(""")
    if change == "none":
        script += """
assert restored.receipt['setup_fingerprint_compatibility'] == 'derived-map-inverse-v1'
assert restored.receipt['setup_fingerprint'] == contract.setup_fingerprint(back)
assert restored.receipt['setup_fingerprint'] != old_digest
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), change],
        cwd=Path(__file__).resolve().parents[1], capture_output=True,
        text=True, check=False)
    if change == "none":
        assert completed.returncode == 0, completed.stderr[-4000:]
    else:
        assert completed.returncode != 0
        assert "different setup fingerprint" in completed.stderr[-4000:]


def test_legacy_prepared_digest_refuses_independent_inverse_words():
    from woof.state_serialization_contract import (
        legacy_derived_map_setup_fingerprint, setup_fingerprint)

    initial, _, _ = _fixture()
    state = initial.state
    state.msfv = np.full((3, 2), np.float32(1.03))
    state.msfv_inv = np.float32(1.) / state.msfv
    old = legacy_derived_map_setup_fingerprint(state)
    current = setup_fingerprint(state)
    assert old is not None and old != current
    state.msfv_inv = np.nextafter(state.msfv_inv, np.float32(0.))
    assert legacy_derived_map_setup_fingerprint(state) is None
    assert setup_fingerprint(state) != current


def test_strict_seal_host_read_preserves_published_theta_words(tmp_path):
    import os
    import subprocess
    import sys

    script = _HOST_RESTORE_WITHOUT_CUPY.replace(
        'sys.meta_path.insert(0, RejectCupy())', '').replace(
        'assert "cupy" not in sys.modules, "the host restore imported CuPy"', '').replace(
        'thb=np.full((nz, ny, nx), 300.0)',
        'thb=np.full((nz, ny, nx), 290.0)').replace(
        'terrain_opt=1)', 'terrain_opt=1, diff_opt=1)').replace(
        'array_module=np)\nback =',
        'array_module=np, adopt_dynamics_carrier=False)\nback =')
    assert 'adopt_dynamics_carrier=False' in script
    script += '''
live = restore_prepared_cache(path, expected_identity=identity, cfg=cfg,
    static=static, array_module=np).initial_result.state
assert (live.thb == np.float32(300)).all()
assert not np.array_equal(live.thp, state.thp)
reference = live.existing_scratch('diff1_theta_initial')
assert reference is not None, 'reference must precede initial feedback and FORCE'
expected = live.thp + live.thb
expected -= np.float32(300)
np.testing.assert_array_equal(reference.view(np.uint32), expected.view(np.uint32))
before = reference.copy()
live.thp += np.float32(.125)
from woof.core.streaming import prime_coordinate_reference
prime_coordinate_reference(live, cfg)
np.testing.assert_array_equal(reference.view(np.uint32), before.view(np.uint32))
'''
    completed = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        env=dict(os.environ, GPUWM_WRF_EXACT='1'))
    assert completed.returncode == 0, completed.stderr[-4000:]


def test_a_restore_array_module_other_than_numpy_or_the_default_is_refused(
        tmp_path):
    """Only the two supported targets exist; anything else is a caller
    bug that would otherwise surface as a half-built state."""

    initial, met, boundaries = _fixture()
    identity = {"source": "array-module-refusal"}
    path = tmp_path / "cache"
    write_prepared_cache(path, identity=identity, initial_result=initial,
                         met=met, boundaries=boundaries)
    with pytest.raises(TypeError,
                       match=r"array_module must be None \(CUDA\) or numpy"):
        prepared_cache_module.restore_prepared_cache(
            path, expected_identity=identity, cfg=None, static={},
            array_module=SimpleNamespace())


@pytest.mark.parametrize("changed", [None, ("polar", True), ("fft_filter_lat", 60.0)])
def test_old_regional_header_accepts_only_default_polar_controls(changed):
    from woof.ingest.prepared_cache import compare_prepared_domain_config

    live = _live_domain_identity()
    cached = {**live, "run": {key: value for key, value in live["run"].items()
                               if key not in ("polar", "fft_filter_lat")}}
    if changed is not None:
        live["run"][changed[0]] = changed[1]
    tolerated, differing = compare_prepared_domain_config(
        cached, live, not_in_use=_undelayed())
    assert differing == ([] if changed is None else ["run." + changed[0]])
    assert sorted(tolerated) == sorted("run." + key for key in ("polar", "fft_filter_lat")
                                      if changed is None or key != changed[0])
