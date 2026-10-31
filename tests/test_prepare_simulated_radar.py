"""Worker-shaped radar options survive every regional preparation family.

Source decoding and state arrays use the existing small CPU fixtures. The
configuration loaders, prepare orchestration, publication, manifests and
downstream configuration bindings are production code.
"""
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest
from woof.source_adapters import source_adapters


# The exact TOML shape emitted by workers/gpu/simulated_radar.py:with_config,
# including the inline site mappings that differ from the usual string sites.
WORKER_RADAR_TOML = '''
[simulated_radar]
enabled = true
sites = [{ id = "TEST", lat = 35.3, lon = -97.5, height_m = 350.0 }]
scan_strategy = "custom"
elevations_deg = [0.5, 1.5]
formats = ["cfradial1"]
fields = ["reflectivity", "velocity"]
timing = "history"
range_km = 20.0
gate_spacing_m = 1000.0
azimuth_step_deg = 10.0
volume_duration_s = 60.0
'''

_MAPPED_SOURCE_IDS = tuple(row.source_id for row in source_adapters()
                          if row.runnable and row.runner == "mapped_composition_v1")


def _worker_config(path):
    path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n" +
                    WORKER_RADAR_TOML, encoding="utf-8")
    from woof.experiment import load_experiment
    from woof.simulated_radar_config import SimulatedRadarOptions
    exp = load_experiment(path)
    assert exp.simulated_radar.enabled
    assert exp.simulated_radar == SimulatedRadarOptions.from_mapping(
        tomllib.loads(WORKER_RADAR_TOML)["simulated_radar"])
    return exp


def _assert_radar_authority(path, expected):
    from woof.experiment import load_experiment
    observed = load_experiment(path)
    assert observed.simulated_radar == expected.simulated_radar
    assert tomllib.loads(path.read_text(encoding="utf-8"))["simulated_radar"] \
        == tomllib.loads(WORKER_RADAR_TOML)["simulated_radar"]


def test_native_prepare_publishes_the_worker_table_before_array_work(tmp_path, monkeypatch):
    from test_hrrr_configured_physics import _case
    from tools import hrrr_single_domain_benchmark as benchmark

    baseline, target, config, namelist, wps = _case(tmp_path)
    exp = _worker_config(config)
    before = config.read_bytes()
    domain = tmp_path / "domain.json"
    domain.write_text(json.dumps(target.to_payload()), encoding="utf-8")
    cache = tmp_path / "prepared-cache"
    cache.mkdir()
    published = tmp_path / "published" / "experiment.toml"
    args = benchmark._parse_args([
        "--bridge", str(tmp_path / "bridge"),
        "--cycle", baseline.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "--manifest-sha256", "a" * 64,
        "--source-manifest-sha256", "b" * 64,
        "--static-cache", str(tmp_path / "static.npz"),
        "--static-receipt", str(tmp_path / "static.json"),
        "--namelist-input", str(namelist), "--wps-namelist", str(wps),
        "--domain-spec", str(domain), "--experiment-config", str(config),
        "--publish-experiment-config", str(published),
        "--prepared-cache", str(cache), "--prepare-only",
        "--run-seconds", str(exp.run_seconds), "--preprocess-backend", "cpu",
        "--preprocess-workers", "1", "--outdir", str(tmp_path / "run"),
    ])
    monkeypatch.setattr(benchmark, "_validated_namelist_extension_identity",
                        lambda *a, **k: {})
    monkeypatch.setattr(benchmark, "_configured_soil_mesh", lambda *a: None)

    class ReachedArrays(Exception):
        pass

    def at_arrays(*args, **kwargs):
        _assert_radar_authority(published, exp)
        raise ReachedArrays

    monkeypatch.setattr(benchmark, "_budgeted_preprocess_backend", at_arrays)
    with pytest.raises(ReachedArrays):
        benchmark.run(args)
    assert config.read_bytes() == before
    _assert_radar_authority(published, exp)


def test_native_publication_refuses_a_lost_radar_option(tmp_path):
    from test_hrrr_configured_physics import _case
    from woof.experiment_document import ExperimentDocumentError, publish_experiment_document
    from woof.hrrr_configuration import resolve_root_experiment

    baseline, target, config, namelist, wps = _case(tmp_path)
    exp = _worker_config(config)
    actual, raw = resolve_root_experiment(target=target, vertical=baseline.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    raw["simulated_radar"]["azimuth_step_deg"] = 5.0
    published = tmp_path / "mismatch.toml"
    with pytest.raises(ExperimentDocumentError, match="simulated radar"):
        publish_experiment_document(published, raw, actual)
    assert not published.exists()


def test_documents_without_radar_keep_their_emitted_bytes():
    from woof.experiment_document import render_experiment_document
    raw = {"experiment": {"name": "plain"}, "shared": {"nz": 12},
           "domain": [{"grid_id": 1}]}
    assert render_experiment_document(raw) == (
        "# woof generated experiment authority -- do not hand-edit.\n"
        "# Rendered from the tables the preparation itself built, and\n"
        "# verified to reload to the same per-domain identity.\n\n"
        '[experiment]\nname = "plain"\n\n'
        "[shared]\nnz = 12\n\n[[domain]]\ngrid_id = 1\n")


def test_native_publication_roundtrips_mixed_site_ids_and_coordinates(tmp_path):
    from test_hrrr_configured_physics import _case
    from woof.experiment_document import publish_experiment_document
    from woof.experiment import load_experiment
    from woof.hrrr_configuration import resolve_root_experiment

    baseline, target, config, namelist, wps = _case(tmp_path)
    text = WORKER_RADAR_TOML.replace('sites = [{', 'sites = ["KTLX", {')
    config.write_text(config.read_text(encoding="utf-8") + text, encoding="utf-8")
    exp = load_experiment(config)
    before = config.read_bytes()
    actual, raw = resolve_root_experiment(target=target, vertical=baseline.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    published = publish_experiment_document(tmp_path / "mixed.toml", raw, actual)
    assert load_experiment(published).simulated_radar == exp.simulated_radar
    assert tomllib.loads(published.read_text(encoding="utf-8"))["simulated_radar"] \
        == tomllib.loads(text)["simulated_radar"]
    assert config.read_bytes() == before


def test_generic_emitter_roundtrips_mixed_scalar_and_mapping_arrays():
    from woof.toml_document import emit_experiment_toml
    raw = {"future_table": {"values": ["plain", {"quoted.key": 3,
        "nested": {"flags": [True, False]}, "another": [1, {"x": 2.0}]}]}}
    assert tomllib.loads(emit_experiment_toml(raw)) == raw


def test_gfs_prepare_preserves_worker_options_and_forecast_binding(tmp_path, monkeypatch):
    from woof import stage_cli
    from woof.experiment import load_experiment
    import test_gfs_initial_perturbation as inputs
    from test_posted_preparation import _gfs_as_posted_route

    author = inputs._config
    authored = []

    def config_with_radar(*args, **kwargs):
        path = author(*args, **kwargs)
        exp = _worker_config(path)
        authored.append((path, path.read_bytes(), exp))
        return path

    monkeypatch.setattr(inputs, "_config", config_with_radar)
    def post_next(replay):
        replay.publish(1)
        replay.publish(2)
        replay.publish(3)
    proof, root, manifest, _replay, _seen = _gfs_as_posted_route(
        tmp_path, monkeypatch, actions=[post_next])
    path, before, exp = authored[0]
    assert path.read_bytes() == before
    _assert_radar_authority(path, exp)
    assert json.loads(manifest.read_text())["files"]["experiment_config"]["sha256"] \
        == hashlib.sha256(before).hexdigest()
    command = stage_cli.sim_command(stage_cli.resolve_bundle(root),
        experiment_config=path, wps_namelist=tmp_path / "experiment.namelist.wps",
        outdir=tmp_path / "forecast")
    bound = Path(command[command.index("--experiment-config") + 1])
    assert load_experiment(bound).simulated_radar == exp.simulated_radar
    assert proof["status"].startswith("READY")


@pytest.mark.parametrize("domain_count", [1, 2])
@pytest.mark.parametrize("source", _MAPPED_SOURCE_IDS)
def test_mapped_prepare_preserves_worker_options_for_all_mapped_sources(tmp_path, monkeypatch, domain_count, source):
    from woof import mapped_direct, stage_cli
    from woof.experiment import VerticalConfig, load_experiment
    from woof.ingest.hrrr_target import HrrrTargetDomain
    from woof.physics_compat import WSM6_PROFILE_ID
    from woof.source_adapters import get_source_adapter
    from tools.hrrr_single_domain_benchmark import _experiment_tables
    from test_mapped_direct import _install_prepare_fakes, _START

    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=domain_count, backend="cpu")
    # This existing array fixture does not fingerprint state bytes. Supply its
    # cache's identity field so the real forecast command can bind the result.
    import woof.ingest.prepared_cache as prepared_cache
    seal = prepared_cache.PreparedCacheStream.seal
    monkeypatch.setattr(prepared_cache.PreparedCacheStream, "seal",
                        lambda self: {**seal(self), "content_sha256": "c" * 64})
    target = replace(HrrrTargetDomain.legacy_500x500(), nx=50, ny=50, nz=49)
    vertical = VerticalConfig(eta_levels=tuple(float(v) for v in np.linspace(1., 0., 50)),
                              p_top=10000., hybrid_opt=2, etac=.2)
    raw, _ = _experiment_tables(vertical, run_seconds=3600., target=target,
        start_time=_START, physics_profile=WSM6_PROFILE_ID)
    adapter = get_source_adapter(source)
    if adapter.upstream_model_id is not None:
        raw["fetch"] = {"source": source}
    if domain_count == 2:
        raw["domain"].append({"grid_id": 2, "parent_id": 1,
            "i_parent_start": 18, "j_parent_start": 18,
            "parent_grid_ratio": 3, "parent_time_step_ratio": 3,
            "nx": 30, "ny": 30, "specified": False, "nested": True,
            "history_interval_s": 3600.})
    config = args["experiment_config"]
    from woof.toml_document import emit_experiment_toml
    config.write_text(emit_experiment_toml(raw), encoding="utf-8")
    exp = _worker_config(config)
    before = config.read_bytes()
    for grid, domain in zip(expected.grids, exp.domains):
        grid.dx = domain.run.dx
        grid.dy = domain.run.dy
    # Restore the real loader. Only the meteorological arrays stay substituted.
    monkeypatch.setattr(mapped_direct, "load_experiment", load_experiment)
    args["stock_wrf_export"] = "off"
    args["preprocess_workers"] = 1
    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert config.read_bytes() == before
    document = tomllib.loads(config.read_text(encoding="utf-8"))
    if adapter.upstream_model_id is not None:
        assert document["fetch"]["source"] == source
    else:
        assert "fetch" not in document
    _assert_radar_authority(config, exp)
    assert calls["initialize"] == len(expected.snapshots)
    command = stage_cli.sim_command(stage_cli.resolve_bundle(args["output_root"]),
        experiment_config=config, wps_namelist=args["wps_namelist"],
        outdir=tmp_path / "forecast")
    bound = Path(command[command.index("--experiment-config") + 1])
    assert load_experiment(bound).simulated_radar == exp.simulated_radar
    assert proof["schema"] == (mapped_direct.PROOF_SCHEMA if domain_count == 1
                               else mapped_direct.HIERARCHY_PROOF_SCHEMA)


def test_era5_prepare_accepts_worker_table_after_manifest_verification(tmp_path, monkeypatch):
    from woof import era5_direct
    from test_hrrr_configured_physics import _case
    from test_era5_direct import _era5_inputs

    _baseline, _target, config, _namelist, wps = _case(tmp_path)
    exp = _worker_config(config)
    loader = era5_direct.load_era5_adapter_config
    args = _era5_inputs(tmp_path, monkeypatch, domains=1)
    args.update(experiment_config=config, wps_namelist=wps)
    roles = {role: args[role] for role in
             ("grib", "vtable", "bridge", "wps_namelist", "experiment_config")}
    manifest = args["input_manifest"]
    manifest.write_text(json.dumps({"schema": era5_direct.INPUT_MANIFEST_SCHEMA,
        "files": {role: {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for role, path in roles.items()}}), encoding="utf-8")
    args["input_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(era5_direct, "load_era5_adapter_config", loader)
    before = config.read_bytes()

    class ReachedGeometry(Exception):
        pass

    def geometry(actual, *args, **kwargs):
        assert actual.simulated_radar == exp.simulated_radar
        raise ReachedGeometry

    monkeypatch.setattr(era5_direct, "validate_native_lambert_contract", geometry)
    with pytest.raises(ReachedGeometry):
        era5_direct.prepare_era5_wrf(**args)
    assert config.read_bytes() == before
    _assert_radar_authority(config, exp)


def test_era5_prepare_publishes_worker_request_and_forecast_binding(tmp_path, monkeypatch):
    from woof import era5_direct, gfs_direct, stage_cli
    from woof.experiment import load_experiment
    from woof.ingest import boundary_stream, prepared_cache
    from woof.core import grid as core_grid
    from woof.static import highres_production
    from test_hrrr_configured_physics import _case
    from test_era5_direct import _era5_inputs
    from test_gfs_initial_perturbation import _cpu_preparation

    _baseline, _target, config, _namelist, wps = _case(tmp_path)
    exp = _worker_config(config)
    before = config.read_bytes()
    loader = era5_direct.load_era5_adapter_config
    args = _era5_inputs(tmp_path, monkeypatch, domains=1)
    args.update(experiment_config=config, wps_namelist=wps,
                preprocess_backend="cpu", preprocess_workers=1)
    roles = {role: args[role] for role in
             ("grib", "vtable", "bridge", "wps_namelist", "experiment_config")}
    args["input_manifest"].write_text(json.dumps({
        "schema": era5_direct.INPUT_MANIFEST_SCHEMA,
        "files": {role: {"name": path.name,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for role, path in roles.items()}}), encoding="utf-8")
    args["input_manifest_sha256"] = hashlib.sha256(
        args["input_manifest"].read_bytes()).hexdigest()
    monkeypatch.setattr(era5_direct, "load_era5_adapter_config", loader)

    # Share the existing immutable CPU array fixtures. Keep ERA5 loading,
    # input verification, head/segment/seal publication and command binding
    # real; only decoding, array construction and native array transport stop
    # at their small fixture boundaries.
    _cpu_preparation(monkeypatch, exp)
    for name in ("resolve_preprocess_backend", "release_backend_memory",
                 "_survey_static_catalog", "adapt_experiment_for_statics",
                 "_static_from_geog", "StateBoundaryFrames",
                 "attach_lateral_boundaries", "initialize_real",
                 "_write_static_cache", "_write_geometry_receipt",
                 "interpolate_era5_to_lambert", "soil_mesh_plan_from_case",
                 "preprocess_land_surface_soil"):
        monkeypatch.setattr(era5_direct, name, getattr(gfs_direct, name))
    grid = gfs_direct._validate_grid_and_vertical_contract()
    grid.dx, grid.dy = exp.root.run.dx, exp.root.run.dy
    monkeypatch.setattr(era5_direct, "validate_native_lambert_contract",
                        lambda *a, **kw: grid)
    snapshots = tuple(SimpleNamespace(
        valid_time=exp.start_time + timedelta(hours=hour),
        levels_hpa=np.asarray([1000., 50.]),
        fields={"SKINTEMP": np.ones((3, 3)), "SOILGEO": np.zeros((3, 3))})
        for hour in (0, 3))
    monkeypatch.setattr(era5_direct, "cached_era5_snapshots", lambda *a, **kw: snapshots)
    monkeypatch.setattr(era5_direct, "parse_vtable", lambda *a: {})
    monkeypatch.setattr(era5_direct, "canonical_units", lambda *a: {})
    monkeypatch.setattr(era5_direct, "_soil_source_orography", lambda *a: None)
    monkeypatch.setattr(era5_direct, "_canonical_surface", lambda *a: {})
    monkeypatch.setattr(highres_production, "apply_prepared_highres",
        lambda fields, _grid, **kw: (fields, kw["baseline_receipt"]))
    monkeypatch.setattr(era5_direct, "RootTerrainBlend",
        lambda *a, **kw: SimpleNamespace(before_initialize=lambda *a: None))
    monkeypatch.setattr(core_grid, "make_vertical_coord", lambda *a, **kw: object())
    monkeypatch.setattr(boundary_stream, "_host_available", lambda: 64 * 1024 ** 3)
    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")

    class ArrayTransport:
        def __init__(self, directory, *, identity, **kwargs):
            self.directory = Path(directory)
            self.identity = identity
            self.metadata = {}

        def move(self, directory):
            self.directory = Path(directory)

        def write_head(self, **kwargs):
            self.directory.mkdir(parents=True)
            self.metadata = dict(kwargs["metadata"] or {})
            return {"identity": self.identity, "metadata": self.metadata,
                    "arrays": {}, "payload_bytes": 0, "lbc": kwargs["lbc"],
                    "setup_core_fingerprint": "0" * 64}

        def write_segment(self, index, interval):
            return {"index": index,
                    "start_seconds": float(interval.start_seconds),
                    "end_seconds": float(interval.end_seconds),
                    "fields": sorted(interval.fields), "arrays": {},
                    "payload_bytes": 0, "prefix": {}}

        def seal(self, *, identity=None):
            if identity is not None:
                self.identity = identity
            basis = json.loads(json.dumps({
                "schema": prepared_cache.PREPARED_CACHE_SCHEMA,
                "identity": self.identity, "metadata": self.metadata,
                "arrays": {}, "payload_bytes": 0}, default=str))
            content = hashlib.sha256(prepared_cache._canonical(basis).encode()).hexdigest()
            (self.directory / "header.json").write_text(json.dumps(
                {**basis, "content_sha256": content}), encoding="utf-8")
            return {"schema": "gpuwm-prepared-cache-v1", "status": "BUILT",
                    "content_sha256": content, "array_count": 0, "payload_bytes": 0}

    monkeypatch.setattr(prepared_cache, "PreparedCacheStream", ArrayTransport)
    monkeypatch.setattr(era5_direct, "export_prepared_wrf",
                        lambda *a, **kw: {"status": "fixture", "files": {}})
    proof = era5_direct.prepare_era5_wrf(**args)
    published = json.loads((args["output_root"] / "proof.json").read_text())
    assert published == proof
    assert proof["status"].startswith("READY")
    assert proof["source_inputs"]["files"]["experiment_config"]["sha256"] \
        == hashlib.sha256(before).hexdigest()
    assert config.read_bytes() == before
    _assert_radar_authority(config, exp)
    command = stage_cli.sim_command(stage_cli.resolve_bundle(args["output_root"]),
        experiment_config=config, wps_namelist=wps, outdir=tmp_path / "forecast")
    bound = Path(command[command.index("--experiment-config") + 1])
    assert load_experiment(bound).simulated_radar == exp.simulated_radar
    assert not (tmp_path / "forecast").exists()


def test_metgrid_prepare_keeps_worker_table_in_immutable_authority(tmp_path, monkeypatch):
    from woof import metem_door, metem_forecast
    from woof.ingest import preprocess_backend
    from woof.static import projection
    from woof.experiment import build_experiment
    from test_prepared_document_recovery import _source

    text, report, _baseline, namelist, controls = _source(tmp_path, monkeypatch)
    text += WORKER_RADAR_TOML
    exp = build_experiment(tomllib.loads(text), source="metgrid radar fixture")
    met = tmp_path / "met_em.d01.2026-05-17_18_00_00.nc"
    met.write_bytes(b"input")
    run = SimpleNamespace(toml_text=text, experiment=exp, namelist_input=namelist,
        paths={1: (met,)}, interval_seconds=3600., controls=controls,
        coverage_seconds=exp.run_seconds, substitution_report=report)
    monkeypatch.setattr(metem_forecast, "resolve_metem_vertical", lambda run, text, **kw: (text, "explicit", None))
    monkeypatch.setattr(metem_door, "metgrid_memory_admission", lambda *a, **kw: {})
    monkeypatch.setattr(preprocess_backend, "resolve_preprocess_backend",
                        lambda *a, **kw: SimpleNamespace(receipt=lambda: {"backend": "test"}))

    class ReachedGeometry(Exception):
        pass

    monkeypatch.setattr(projection, "grids_from_projection_config",
        lambda *a: (_ for _ in ()).throw(ReachedGeometry()))
    prepared = tmp_path / "prepared"
    with pytest.raises(ReachedGeometry):
        metem_forecast.prepare_metem_run(run, prepared, preprocess_backend="cpu")
    assert (prepared / "experiment.toml").read_text(encoding="utf-8") == text
    _assert_radar_authority(prepared / "experiment.toml", exp)


def test_wrfinput_prepare_keeps_worker_table_in_immutable_authority(tmp_path, monkeypatch):
    from woof import wrfinput_forecast
    from woof.ingest import wrfinput
    from woof.core import landuse
    from woof.static import projection
    from woof.experiment import build_experiment
    from test_prepared_document_recovery import _source

    text, report, _baseline, namelist, _ = _source(tmp_path, monkeypatch)
    text += WORKER_RADAR_TOML
    exp = build_experiment(tomllib.loads(text), source="WRF radar fixture")
    boundary = tmp_path / "wrfbdy_d01"
    boundary.write_bytes(b"boundary input")
    inputs = {}
    for domain in exp.domains:
        path = tmp_path / f"wrfinput_d{domain.grid_id:02d}"
        path.write_bytes(b"state input")
        inputs[domain.grid_id] = path
    run = SimpleNamespace(toml_text=text, namelist_input=namelist,
        wrfbdy_path=boundary, wrfinput_paths=inputs, substitution_report=report,
        coverage=SimpleNamespace(coverage_seconds=exp.run_seconds,
            times=(exp.start_time,), end=exp.start_time + timedelta(seconds=exp.run_seconds),
            forcing_interval_seconds=3600.))
    attrs = dict(MMINLU="MODIFIED_IGBP_MODIS_NOAH", NUM_LAND_CAT=21,
        ISWATER=17, ISLAKE=21, ISICE=15, ISURBAN=13, ISOILWATER=14,
        CEN_LAT=35., USE_THETA_M=0)
    raw = {name: np.zeros((2, 2), np.float32) for name in
        ("LU_INDEX", "ISLTYP", "LANDMASK", "SNOW", "XICE", "TSLB", "SST",
         "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "HGT")}
    monkeypatch.setattr(wrfinput, "read_wrfinput", lambda path, **kw: SimpleNamespace(
        path=path, global_attributes=attrs, raw=raw,
        surface_input_dispositions={}, soil_unit_conversions={}))
    monkeypatch.setattr(wrfinput, "read_wrfbdy", lambda *a, **kw: object())
    monkeypatch.setattr(landuse, "initialize_landuse", lambda *a, **kw: object())
    monkeypatch.setattr(projection, "grids_from_projection_config", lambda exp: [object() for _ in exp.domains])
    from woof.ingest import wrfinput_nest_initial
    # The WRF-file nest start (med_nest_initial, c49be92105) blends the nest against its parent's real grids;
    # this fixture's 2 x 2 stand-ins have none, and the worker-table authority under test does not read it.
    monkeypatch.setattr(wrfinput_nest_initial, "nest_input_start",
                        lambda nest, parent, **kw: SimpleNamespace(restored=nest, receipt={}))
    monkeypatch.setattr(wrfinput_nest_initial, "feed_back_file_fields", lambda parent, nest, **kw: parent)
    prepared = wrfinput_forecast.prepare_wrf_run(run, tmp_path / "prepared")
    assert prepared.experiment.simulated_radar == exp.simulated_radar
    assert prepared.experiment_config.read_text(encoding="utf-8") == text
    _assert_radar_authority(prepared.experiment_config, exp)


def test_all_runnable_source_rows_use_exercised_prepare_families():
    from woof.source_adapters import get_source_adapter, source_adapters
    assert get_source_adapter("hrrr-prs").runner == "mapped_composition_v1"
    assert {row.runner for row in source_adapters() if row.runnable} <= {
        "hrrr_f00_f12_v1", "gfs_pgrb2_0p25_v1", "era5_combined_grib1_v1",
        "mapped_composition_v1", "twentycrv3_member_grib2_v1"}


def test_member_archive_prepare_forwards_worker_config_to_generic_prepare(tmp_path, monkeypatch):
    from woof import mapped_authoring, mapped_engine_bridge, twentycrv3_wrf
    from woof.experiment import load_experiment
    from test_hrrr_configured_physics import _case

    _baseline, _target, config, _namelist, wps = _case(tmp_path)
    exp = _worker_config(config)
    before = config.read_bytes()
    paths = {}
    for name in ("mapping", "composition", "provenance", "manifest", "p0", "p1", "s0"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    document = {"files": [{"role": "sfc", "path": str(paths["s0"])}],
        "cadence_seconds": 3600, "member": "001", "member_identity": "a" * 64}
    monkeypatch.setattr(twentycrv3_wrf, "_manifest", lambda *a: (document, (paths["p0"], paths["p1"])))
    monkeypatch.setattr(twentycrv3_wrf, "load_composition", lambda *a: {
        "supplements": {"terrain_height": {"data_role": "terrain", "provenance_role": "terrain"}}})
    monkeypatch.setattr(twentycrv3_wrf, "load_mapping", lambda *a: {
        "format": "grib2", "target": {"boundary_interval_seconds": 3600},
        "coordinates": {"vertical": {"levels": [1000., 500.]}}})
    monkeypatch.setattr(twentycrv3_wrf, "_mapped_engine_choice", lambda **kw: mapped_engine_bridge.ENGINE_RUST)
    monkeypatch.setattr(mapped_engine_bridge, "engine_record_inventory", lambda primary: {path: [] for path in primary})
    monkeypatch.setattr(twentycrv3_wrf, "_verify_archive_inventory", lambda *a: None)
    monkeypatch.setattr(mapped_authoring, "author_input_manifest",
                        lambda path, **kw: path.write_text("{}", encoding="utf-8"))
    seen = []
    def prepare(**kwargs):
        assert Path(kwargs["experiment_config"]).read_bytes() == before
        assert load_experiment(kwargs["experiment_config"]).simulated_radar == exp.simulated_radar
        seen.append(kwargs)
        return {"forwarded": True}
    monkeypatch.setattr(twentycrv3_wrf, "prepare_mapped_wrf", prepare)
    assert twentycrv3_wrf.prepare_20crv3_wrf(mapping=paths["mapping"],
        composition=paths["composition"], provenance=paths["provenance"],
        manifest=paths["manifest"], manifest_sha256="b" * 64,
        wps_namelist=wps, geog_root=tmp_path, experiment_config=config,
        output_root=tmp_path / "prepared", preprocess_backend="cpu") == {"forwarded": True}
    assert len(seen) == 1
    assert config.read_bytes() == before
