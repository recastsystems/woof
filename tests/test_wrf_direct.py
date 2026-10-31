from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import woof.wrf_direct as wrf_direct

from woof.wrf_direct import (
    _direct_export_soil_geometry,
    _global_updates,
    _domain_global_attributes,
    _dimensions,
    _array_sha256,
    _canonical,
    _interp_nodes,
    _lbc_to_wrf,
    _load_contract,
    _load_static_geometry_receipt,
    _prepare_output_target,
    _publish_staging,
    _contract_payload_sha256,
    _physics_contract_bundle,
    _prepared_vertical_contract,
    _resolved_prototype_value,
    _date_text,
    _forcing_interval_indices,
    _hierarchy_global_updates,
    _validated_hierarchy,
    _surface_fields,
    _validate_file,
    _write_timestamp_at,
    _write_wrfinput,
    PreparedCache,
    PreparedDomainArtifacts,
    StockWrfExportUnsupported,
    export_prepared_wrf,
    export_prepared_wrf_hierarchy,
    load_domain_artifacts_manifest,
    write_domain_artifacts_manifest,
    _wrf_noah_landuse,
    _wrf_soil_category,
)
from woof.physics_compat import (
    MYNN_PROFILE_ID,
    validate_single_domain_physics_profile,
)
from woof.native_wrf_contract import native_geometry_contract
from woof.wrf_physics_inventory import EXPORT_USE_THETA_M
from woof.static.lambert import LambertGrid
from woof.vertical_contract import expected_coordinate_shapes
from woof.config import RunConfig
from woof.core.microphysics_transition import MP8_TO_MP18_POLICY


class _Cache:
    def __init__(self, value):
        self.value = value

    def array(self, name):
        return self.value


def test_stock_wrf_export_rejects_gpuwm_mixed_microphysics_before_output(
        tmp_path):
    parent = RunConfig(
        nx=16, ny=16, nz=4, dx=3000.0, dy=3000.0, ztop=10000.0,
        dt=6.0, run_seconds=60.0, moist=True, moist_cq=True,
        mp_physics=8)
    child = RunConfig(
        nx=24, ny=24, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
        dt=2.0, run_seconds=60.0, moist=True, moist_cq=True,
        mp_physics=18, nested=True,
        nest_microphysics_transition=MP8_TO_MP18_POLICY)
    exp = SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, parent_id=0, run=parent),
        SimpleNamespace(grid_id=2, parent_id=1, run=child),
    ))
    output = tmp_path / "stock-wrf"
    with pytest.raises(StockWrfExportUnsupported,
                       match="GPUWM extension.*not stock-WRF") as refusal:
        export_prepared_wrf_hierarchy(exp, (), output)
    # Typed, so a caller preparing a FORECAST can tell this apart from a
    # corrupt cache; and it names the deltas as data, not only as prose.
    assert isinstance(refusal.value, ValueError)
    assert refusal.value.unsupported == {"mp_physics": (18, 8)}
    assert not output.exists()
    assert not output.with_name(output.name + f".tmp-{os.getpid()}").exists()


def _minimal_prepared_cache(root: Path, run: dict) -> Path:
    """The smallest cache header the export gate reads before it refuses."""

    root.mkdir(parents=True)
    header = {
        "schema": "gpuwm-prepared-real-cache-v1",
        "status": "READY",
        "identity": {"domain_config": {"run": run}},
        "metadata": {"user": {}},
        "arrays": {},
        "payload_bytes": 0,
    }
    header["content_sha256"] = hashlib.sha256(
        _canonical({key: header[key] for key in (
            "schema", "identity", "metadata", "arrays", "payload_bytes")}
        ).encode("utf-8")).hexdigest()
    (root / "header.json").write_text(
        json.dumps(header, indent=2, sort_keys=True), encoding="utf-8")
    return root


def test_the_profile_free_export_gate_keeps_its_exact_v2_stock_slice(tmp_path):
    """F2 fixes the LAYERING; this pins the gate's CONTENT, unchanged.

    The compatibility branch -- the one the hierarchy root export takes,
    because it supplies no profile -- still demands bl_pbl_physics=1,
    sf_sfclay_physics=91 and sf_surface_physics=2, and still writes
    nothing when it refuses.  The message is the one the probes quoted
    verbatim.
    """

    cache = _minimal_prepared_cache(tmp_path / "prepared", {
        "mp_physics": 6, "hybrid_opt": 2, "hypsometric_opt": 2,
        "specified": True, "nested": False, "spec_bdy_width": 5,
        # The MYNN tree the probes drove: registry-reachable, and refused.
        "bl_pbl_physics": 5, "sf_sfclay_physics": 5, "sf_surface_physics": 2,
    })
    output = tmp_path / "wrf-native-input"

    with pytest.raises(StockWrfExportUnsupported) as refusal:
        export_prepared_wrf(
            cache, tmp_path / "static.npz", tmp_path / "geometry.json",
            output, valid_time=datetime(2026, 7, 29, 6))

    assert str(refusal.value) == (
        "unsupported direct-export configuration: "
        "{'bl_pbl_physics': (5, 1), 'sf_sfclay_physics': (5, 91)}")
    assert refusal.value.unsupported == {
        "bl_pbl_physics": (5, 1), "sf_sfclay_physics": (5, 91)}
    assert not output.exists()
    assert not list(tmp_path.glob("wrf-native-input.tmp-*"))


def test_experiment_config_suite_export_is_not_pinned_to_the_v2_slice(
        tmp_path, monkeypatch):
    """The profileless PREPARATION contract (owner ruling 2026-07-31).

    The exact cache the pinned test above shows the legacy no-profile
    branch refusing -- a registry-reachable MYNN tree -- must pass the
    physics gate under ``experiment_config_suite=True``, because the
    hash-bound experiment config is the authority now.  The v2 slice
    pin stays where it is: on callers that ask for neither contract.
    A marker planted at the first post-gate step proves the gate was
    crossed rather than the whole export succeeding by accident.
    """

    cache = _minimal_prepared_cache(tmp_path / "prepared", {
        "mp_physics": 6, "hybrid_opt": 2, "hypsometric_opt": 2,
        "specified": True, "nested": False, "spec_bdy_width": 5,
        "bl_pbl_physics": 5, "sf_sfclay_physics": 5, "sf_surface_physics": 2,
        "cu_physics": 0, "ra_lw_physics": 0, "ra_sw_physics": 1,
        # A real cache identity is asdict(RunConfig), so it always
        # carries num_soil_layers; the v2-slice fixture above can omit
        # it only because that branch never resolves the tuple.
        "num_soil_layers": 4,
    })

    def marker(*args, **kwargs):
        raise RuntimeError("reached-post-physics-gate")

    monkeypatch.setattr(wrf_direct, "_forcing_interval_indices", marker)
    with pytest.raises(RuntimeError, match="reached-post-physics-gate"):
        export_prepared_wrf(
            cache, tmp_path / "static.npz", tmp_path / "geometry.json",
            tmp_path / "wrf-native-input",
            valid_time=datetime(2026, 7, 29, 6),
            experiment_config_suite=True)


def test_experiment_config_suite_export_keeps_the_expert_acknowledgement(
        tmp_path, monkeypatch, capsys):
    """Suite freedom is not consent silence.

    The exporter's profileless recompute APPLIES the registry-owned
    tuple governance -- that is the contract
    ``experiment_config_suite=True`` carries, and it is what this test
    guards.  Its severity is warn-not-block's: a tuple outside what the
    registry declares reachable is still implemented component by
    component, so it runs and says so in one line naming both delivery
    spellings, and delivering the acknowledgement silences that line.

    The tuple used to be the Noah-MP profile's, under the
    noahmp-host-column-throughput-v1 expert acknowledgement.  d97690a4c
    admitted noah-mp as a per-domain land-surface option and made that
    acknowledgement an optional advisory, so the Noah-MP tuple is
    registry-reachable on this route and exports with no line; that is
    pinned first.  The same suite with the analytic clear-sky radiation
    (ra_lw_physics = ra_sw_physics = 90), an implemented option no
    declared route reaches, carries the contract now.
    """

    from woof.physics_compat import (
        NOAHMP_PROFILE_ID,
        single_domain_runtime_switches,
    )
    from woof.physics_registry import physics_registry

    geometry = {"hybrid_opt": 2, "hypsometric_opt": 2,
                "specified": True, "nested": False, "spec_bdy_width": 5}
    noahmp = {**single_domain_runtime_switches(NOAHMP_PROFILE_ID), **geometry}

    def marker(*args, **kwargs):
        raise RuntimeError("reached-post-physics-gate")

    monkeypatch.setattr(wrf_direct, "_forcing_interval_indices", marker)

    def export(cache, acknowledgements=()):
        with pytest.raises(RuntimeError, match="reached-post-physics-gate"):
            export_prepared_wrf(
                cache, tmp_path / "static.npz", tmp_path / "geometry.json",
                tmp_path / "wrf-native-input",
                valid_time=datetime(2026, 7, 29, 6),
                experiment_config_suite=True,
                expert_acknowledgements=acknowledgements)
        return [line for line in capsys.readouterr().err.splitlines()
                if line.startswith("warning:")
                and "registry reachability" in line]

    assert export(_minimal_prepared_cache(
        tmp_path / "prepared-noahmp", noahmp)) == []

    acknowledgement = physics_registry()["authority"][
        "unnamed_tree_outside_reachability_acknowledgement_id"]
    analytic = _minimal_prepared_cache(
        tmp_path / "prepared-analytic",
        {**noahmp, "ra_lw_physics": 90, "ra_sw_physics": 90})
    unacked = export(analytic)
    assert len(unacked) == 1, unacked
    assert "'outside-registry-declared-reachability'" in unacked[0]
    assert f"--ack {acknowledgement}" in unacked[0]
    assert f'acknowledgements = ["{acknowledgement}"]' in unacked[0]

    assert export(analytic, (acknowledgement,)) == []


def test_single_domain_cli_forwards_physics_profile_and_acknowledgement(
        tmp_path, monkeypatch, capsys):
    observed = {}

    def export(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"status": "READY"}

    monkeypatch.setattr(wrf_direct, "export_prepared_wrf", export)
    monkeypatch.setattr(sys, "argv", [
        "woof.wrf_direct",
        "--prepared-cache", str(tmp_path / "prepared"),
        "--static-cache", str(tmp_path / "static.npz"),
        "--geometry-receipt", str(tmp_path / "geometry.json"),
        "--output", str(tmp_path / "output"),
        "--valid-time", "2026-07-18_00:00:00",
        "--physics-profile", MYNN_PROFILE_ID,
        "--ack", "consent-v1",
    ])

    wrf_direct.main()

    assert observed["kwargs"]["physics_profile"] == MYNN_PROFILE_ID
    assert observed["kwargs"]["expert_acknowledgements"] == ("consent-v1",)
    assert json.loads(capsys.readouterr().out)["status"] == "READY"


def test_hierarchy_cli_does_not_forward_single_domain_physics_options(
        tmp_path, monkeypatch, capsys):
    observed = {}

    def export(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"status": "READY"}

    monkeypatch.setattr(
        wrf_direct, "export_prepared_wrf_namelists", export)
    monkeypatch.setattr(sys, "argv", [
        "woof.wrf_direct",
        "--domain-artifacts", str(tmp_path / "domains.json"),
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--namelist-input", str(tmp_path / "namelist.input"),
        "--output", str(tmp_path / "output"),
    ])

    wrf_direct.main()

    assert "physics_profile" not in observed["kwargs"]
    assert "expert_acknowledgements" not in observed["kwargs"]
    assert json.loads(capsys.readouterr().out)["status"] == "READY"


def test_contract_inventory_is_frozen():
    contract = _load_contract()
    assert len(contract["wrfinput"]["variables"]) == 183
    assert len(contract["wrfbdy"]["variables"]) == 115


@pytest.mark.parametrize(
    ("mp_physics", "number_names"),
    [
        (6, ()),
        (8, ("QNICE", "QNRAIN")),
        (10, ("QNICE", "QNSNOW", "QNRAIN", "QNGRAUPEL")),
        (
            18,
            (
                "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
                "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
            ),
        ),
    ],
)
def test_registry_physics_contract_adds_exact_input_and_boundary_scalars(
        mp_physics, number_names):
    frozen = _load_contract()
    resolved = _physics_contract_bundle(frozen, mp_physics)
    input_names = {item["name"] for item in resolved["wrfinput"]["variables"]}
    bdy_names = {item["name"] for item in resolved["wrfbdy"]["variables"]}
    assert set(number_names) <= input_names
    for name in number_names:
        for suffix in (
                "_BXS", "_BXE", "_BYS", "_BYE",
                "_BTXS", "_BTXE", "_BTYS", "_BTYE"):
            assert name + suffix in bdy_names
    # Resolution is non-mutating: the provenance-frozen WSM6 template stays
    # byte/variable-count stable.
    assert len(frozen["wrfinput"]["variables"]) == 183
    assert len(frozen["wrfbdy"]["variables"]) == 115


def test_wsm6_resolved_contract_digest_remains_byte_semantically_stable():
    assert _contract_payload_sha256(
        _physics_contract_bundle(_load_contract(), 6)
    ) == "f486395516ca9746d8b259f06ccb5913a23210515f3cd0328348f09695cb1366"


def test_nssl2_resolved_contract_uses_exact_registry_metadata():
    resolved = _physics_contract_bundle(_load_contract(), 18)
    specs = {
        item["name"]: item for item in resolved["wrfinput"]["variables"]
    }

    assert specs["QHAIL"]["attributes"] == {
        "FieldType": 104,
        "MemoryOrder": "XYZ",
        "description": "Hail mixing ratio",
        "units": "kg kg-1",
        "stagger": "",
        "coordinates": "XLONG XLAT XTIME",
    }
    assert specs["QNDROP"]["attributes"]["units"] == "# kg-1"
    assert specs["QNCCN"]["attributes"]["description"] == (
        "CCN Number concentration"
    )
    assert specs["QVGRAUPEL"]["attributes"]["units"] == "m(3) kg(-1)"
    assert specs["QVHAIL"]["attributes"]["units"] == "m(3) kg(-1)"


@pytest.mark.parametrize("mass_levels", [35, 49, 80])
@pytest.mark.parametrize("mp_physics", [8, 10, 18])
def test_number_moment_wrfinput_contract_writes_arbitrary_vertical_shape(
        tmp_path, mass_levels, mp_physics):
    contract = _physics_contract_bundle(
        _load_contract(), mp_physics)["wrfinput"]
    dimensions = _dimensions(contract, nx=3, ny=2, nz=mass_levels)
    geometry = {
        "center_lat": 35.5, "center_lon": -98.0,
        "ref_lat": 35.5, "ref_lon": -98.0,
        "truelat1": 30.0, "truelat2": 60.0,
        "stand_lon": -97.0,
    }
    updates = _global_updates(
        valid_time=datetime(2026, 7, 18), nx=3, ny=2, nz=mass_levels,
        dx=1000.0, dy=1000.0, dt=5.0, geometry=geometry)
    path = tmp_path / f"wrfinput-mp{mp_physics}-{mass_levels}"
    # Live export supplies all coordinate arrays from the prepared cache. Use
    # shape-correct stand-ins here so this unit gate isolates contract sizing
    # instead of asking the frozen 49-level prototype to invent a new grid.
    fields = {}
    for spec in contract["variables"]:
        if "prototype_value" not in spec:
            continue
        if spec["dimensions"] == ["Time", "bottom_top"]:
            fields[spec["name"]] = np.zeros(mass_levels, dtype=np.float32)
        elif spec["dimensions"] == ["Time", "bottom_top_stag"]:
            fields[spec["name"]] = np.zeros(mass_levels + 1, dtype=np.float32)
    _write_wrfinput(
        path, contract, dimensions, updates, fields,
        "2026-07-18_00:00:00")
    expected = {
        8: ("QNICE", "QNRAIN"),
        10: ("QNICE", "QNSNOW", "QNRAIN", "QNGRAUPEL"),
        18: (
            "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
            "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
        ),
    }[mp_physics]
    with netCDF4.Dataset(path) as dataset:
        for name in expected:
            variable = dataset.variables[name]
            assert variable.dtype == np.dtype("float32")
            assert variable.shape == (1, mass_levels, 2, 3)
            assert not np.any(variable[:])


def _export_a_real_prepared_cache(tmp_path, mp_physics, numbers=None,
                                  *, boundary_scalars=None,
                                  replace_start=None, intervals=1, surface_override=None,
                                  zones=(5, 1, 4), cells=12,
                                  **run_overrides):
    """A real CPU cold start through the prepared cache and the export.

    Initialization, prepared cache and a profile-free
    ``export_prepared_wrf``, with no physics selection handed to the
    exporter.  Returns ``(output, cfg, seeded)``: the export folder, the
    configuration the cache records, and the state's own values of the
    ``numbers`` named (WRF name to state name), read before the export.

    ``boundary_scalars`` is the scalar inventory the boundary is built
    with (an aerosol-forced mp=28 run adds nwfa and nifa to qv, as its
    initialization does when the aerosols come from input), and
    ``replace_start`` edits the start state in place AFTER the boundary
    was built from it, which is what a separate initial analysis
    (``--initial-inputs``) does to a prepared cache.  ``intervals`` is
    the number of hourly boundary intervals.  ``zones`` is
    ``(spec_bdy_width, spec_zone, relax_zone)`` and ``cells`` the square
    domain's mass points per side.
    """
    import dataclasses

    from test_arbitrary_vertical_gpu import _sha256, _static
    from test_real_init import _analyzed_hrrr_real_init, _host_array
    from woof.gfs_direct import _geometry_contract
    from woof.ingest.lateral_bc import (
        attach_lateral_boundaries, build_state_lateral_boundaries)
    from woof.ingest.prepared_cache import write_prepared_cache

    ny, nx = cells, cells
    spec_bdy_width, spec_zone, relax_zone = zones
    valid_time = datetime(2026, 7, 20, 6)
    # A specified mp=28 domain otherwise refuses without WRF's WIF
    # climatology file; the aerosol source is not what is under test.
    aerosol = {"mp28_aerosol_source": "synthetic"} if mp_physics == 28 else {}
    run = dict(bl_pbl_physics=1, sf_sfclay_physics=91, sf_surface_physics=2)
    run.update(run_overrides)
    result, cfg = _analyzed_hrrr_real_init(
        mp_physics, shape=(ny, nx), map_proj=1, hypsometric_opt=2,
        specified=True, nested=False, spec_bdy_width=spec_bdy_width,
        spec_zone=spec_zone, relax_zone=relax_zone, **run, **aerosol)
    state = result.state
    seeded = {}
    for wrf_name, state_name in (numbers or {}).items():
        value = np.array(_host_array(getattr(state, state_name)),
                         dtype=np.float32)
        assert value.max() > 0.0, state_name
        seeded[wrf_name] = value

    grid = LambertGrid(39.0, -84.0, 30.0, 60.0, -84.0,
                       cfg.dx, cfg.dy, nx + 1, ny + 1)
    static = _static(grid, ny, nx)
    state.set_map_coriolis(
        static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
        static["F"], static["E"],
        sina=static["SINALPHA"], cosa=static["COSALPHA"])
    if boundary_scalars is not None:
        state._external_scalar_boundary_fields = tuple(boundary_scalars)
    boundaries = build_state_lateral_boundaries(
        [state] * (intervals + 1),
        tuple(valid_time + timedelta(hours=hour)
              for hour in range(intervals + 1)),
        spec_bdy_width=spec_bdy_width, spec_zone=spec_zone,
        relax_zone=relax_zone)
    attach_lateral_boundaries(state, boundaries)
    if replace_start is not None:
        replace_start(state)

    static_path = tmp_path / "native-static.npz"
    np.savez(static_path, **static)
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps({
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "cache": {"path": static_path.name,
                  "bytes": static_path.stat().st_size,
                  "sha256": _sha256(static_path)},
        "geometry": _geometry_contract(grid, cfg),
    }), encoding="utf-8")
    plane = np.ones((ny, nx), dtype=np.float32)
    met = SimpleNamespace(fields={
        "LANDSEA": plane, "SKINTEMP": 289.0 * plane, "T2": 289.0 * plane,
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
    })
    surface = {
        "TSK": 289.0 * plane,
        "TSLB": np.full((4, ny, nx), 288.0, dtype=np.float32),
        "SMOIS": np.full((4, ny, nx), 0.25, dtype=np.float32),
        "SH2O": np.full((4, ny, nx), 0.25, dtype=np.float32),
        "TMN": 287.0 * plane,
        "SEAICE": np.zeros((ny, nx), dtype=np.float32),
        "XLAND": plane, "LANDMASK": plane,
        "SNOW": np.zeros((ny, nx), dtype=np.float32),
        "SNOWH": np.zeros((ny, nx), dtype=np.float32),
    }
    if surface_override is not None:
        surface_override(surface, static)
        np.savez(static_path, **static)
        geometry_doc = json.loads(geometry_path.read_text())
        geometry_doc["cache"]["sha256"] = _sha256(static_path)
        geometry_doc["cache"]["bytes"] = static_path.stat().st_size
        geometry_path.write_text(json.dumps(geometry_doc))
    cache_path = tmp_path / "prepared-cache"
    write_prepared_cache(
        cache_path,
        identity={"domain_config": {"run": dataclasses.asdict(cfg)},
                  "forcing_hours": list(range(intervals + 1)),
                  "static_cache_sha256": _sha256(static_path)},
        initial_result=result, met=met, surface=surface,
        boundaries=boundaries,
        metadata={"initial_valid_time": valid_time.isoformat()})
    output = tmp_path / "wrf-native-input"
    export_prepared_wrf(
        cache_path, static_path, geometry_path, output,
        valid_time=valid_time, boundary_interval_seconds=3600)
    return output, cfg, seeded


@pytest.mark.parametrize("mp_physics, numbers", [
    (8, {"QNRAIN": "nr", "QNICE": "ni"}),
    (28, {"QNRAIN": "nr", "QNICE": "ni", "QNCLOUD": "nc"}),
])
def test_the_stock_export_writes_the_cold_start_seeded_number_moments(
        tmp_path, mp_physics, numbers):
    """A Thompson cold start's seeded numbers reach the exported wrfinput.

    The cold start seeds rain and ice numbers (and mp=28's droplet
    number) where the analysed mass has none, as real.exe does with
    make_RainNumber, make_IceNumber and make_DropletNumber.  The inventory
    rows named QNRAIN and QNICE by their Registry names qnr and qni, the
    prepared cache holds them as nr and ni, and the export found no such
    array and wrote both as zero beside a nonzero QRAIN and QICE.  The
    whole route runs here on the CPU: initialization, prepared cache,
    export, and the file read back bit for bit.
    """
    output, _cfg, seeded = _export_a_real_prepared_cache(
        tmp_path, mp_physics, numbers)

    with netCDF4.Dataset(output / "wrfinput_d01") as dataset:
        for wrf_name, value in seeded.items():
            written = np.asarray(dataset.variables[wrf_name][0],
                                 dtype=np.float32)
            np.testing.assert_array_equal(
                written.view(np.uint32), value.view(np.uint32),
                err_msg=wrf_name)


@pytest.mark.parametrize("mp_physics", [8, 10, 28])
def test_a_profile_free_export_states_the_physics_its_cache_was_prepared_with(
        tmp_path, mp_physics):
    """No physics selection, and the file still says what the arrays are.

    ``python -m woof.wrf_direct`` without ``--physics-profile``, the
    ERA5 door and the tree export hand the exporter no selection, and on
    2.8.0 it stamped none: the frozen contract's MP_PHYSICS = 6,
    RA_LW_PHYSICS = 0, RA_SW_PHYSICS = 1 and CU_PHYSICS = 0 went out
    beside Thompson or Morrison arrays and number moments, and stock WRF
    reads its schemes from those attributes.  An RTE+RRTMGP 4/4, cumulus
    on cache is used so every one of the seven that can differ from the
    frozen file does.
    """
    output, cfg, _seeded = _export_a_real_prepared_cache(
        tmp_path, mp_physics, ra_physics=0, ra_lw_physics=4,
        ra_sw_physics=4, cu_physics=1)
    assert (cfg.mp_physics, cfg.ra_lw_physics, cfg.ra_sw_physics,
            cfg.cu_physics) == (mp_physics, 4, 4, 1)
    expected = {
        "MP_PHYSICS": mp_physics, "RA_LW_PHYSICS": 4, "RA_SW_PHYSICS": 4,
        "SF_SFCLAY_PHYSICS": 91, "SF_SURFACE_PHYSICS": 2,
        "BL_PBL_PHYSICS": 1, "CU_PHYSICS": 1,
    }
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        with netCDF4.Dataset(output / name) as dataset:
            stated = {key: int(dataset.getncattr(key)) for key in expected}
        assert stated == expected, name


@pytest.mark.parametrize("categories, islake", [(61, 21), (20, -1), (21, 21)])
def test_the_export_writes_the_land_categories_its_static_has(
        tmp_path, categories, islake):
    """LANDUSEF, land_cat_stag and NUM_LAND_CAT follow the static.

    An urban canopy run prepares WRF's MODIS legend with the built types
    51-61, 61 planes; the 15 arc-second MODIS set has 20 and no lake
    class.  The export wrote into the frozen file's 21 and stopped after
    the forecast had run ("LANDUSEF source shape (61, 170, 170) !=
    (21, 170, 170)", the SF template with sf_urban_physics = 1 and Annual
    NLCD on the HRRR route).  real.exe writes num_land_cat planes and
    states it in both files, as the forecast's own history does.  The
    manifest's resolved physics contract digest stays the one the
    preparation recomputes from mp_physics alone: a first version of the
    fix put the resized contract's digest there and the run stopped one
    check later ("proof export source hashes differ from preparation").
    """
    from woof.prepared_single_domain_forecast import (
        _resolved_wrf_direct_contract_sha256)

    expected = {}

    def land_use(_surface, static):
        ny, nx = static["LU_INDEX"].shape
        planes = np.zeros((categories, ny, nx), dtype=np.float64)
        planes[9] = 0.4                    # grassland
        planes[categories - 1] = 0.6       # the last plane: LCZ 11 / lake
        static["LANDUSEF"] = planes
        expected["LANDUSEF"] = planes.astype(np.float32)

    output, _cfg, _seeded = _export_a_real_prepared_cache(
        tmp_path, 8, surface_override=land_use)
    with netCDF4.Dataset(output / "wrfinput_d01") as dataset:
        assert len(dataset.dimensions["land_cat_stag"]) == categories
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["LANDUSEF"][0], dtype=np.float32),
            expected["LANDUSEF"])
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        with netCDF4.Dataset(output / name) as dataset:
            assert int(dataset.getncattr("NUM_LAND_CAT")) == categories, name
            assert int(dataset.getncattr("ISLAKE")) == islake, name
            assert int(dataset.getncattr("ISURBAN")) == 13, name
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["resolved_physics_contract_sha256"] == \
        _resolved_wrf_direct_contract_sha256(8)


@pytest.mark.parametrize("gwd_opt", [1, 3])
def test_a_terrain_drag_export_records_the_digest_the_preflight_recomputes(
        tmp_path, gwd_opt):
    """A gwd_opt run's export names the digest its forecast checks.

    The export writes the statistics the drag scheme reads (gwd_opt = 3:
    the twenty GSL large- and small-scale fields), and it once also took
    the digest of that drag-extended contract.  The forecast's preflight
    recomputes the digest from mp_physics alone, so every gwd_opt != 0
    run -- the hrrr_v4_gsd41 recipe on hrrr-native among them -- finished
    its forecast and then exited rc=1 at the seal's preflight with
    "proof export source hashes differ from preparation" (2.9.10).  The
    files keep the statistics; the digest is the package's.
    """
    from woof.prepared_single_domain_forecast import (
        _resolved_wrf_direct_contract_sha256)
    from woof.static.orographic import required_static_fields

    names = required_static_fields(0, gwd_opt)
    written = {}

    def drag_statistics(_surface, static):
        ny, nx = static["HGT_M"].shape
        for index, name in enumerate(names):
            value = np.full((ny, nx), 1.0 + 0.25 * index, dtype=np.float32)
            static[name] = value
            written[name] = value

    output, cfg, _seeded = _export_a_real_prepared_cache(
        tmp_path, 28, surface_override=drag_statistics, gwd_opt=gwd_opt)
    assert int(cfg.gwd_opt) == gwd_opt
    with netCDF4.Dataset(output / "wrfinput_d01") as dataset:
        for name in names:
            np.testing.assert_array_equal(
                np.asarray(dataset.variables[name][0], dtype=np.float32),
                written[name])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["resolved_physics_contract_sha256"] == \
        _resolved_wrf_direct_contract_sha256(28)


def test_a_profile_free_export_refuses_a_cache_that_records_no_physics():
    """Filling a missing selector from the frozen file is what was wrong."""

    from woof.wrf_direct import recorded_physics_selectors

    recorded = {"mp_physics": 8, "ra_physics": 0, "ra_lw_physics": 4,
                "ra_sw_physics": 4, "sf_sfclay_physics": 91,
                "sf_surface_physics": 2, "bl_pbl_physics": 1,
                "cu_physics": 0}
    assert recorded_physics_selectors(recorded) == {
        key: recorded[key] for key in (
            "mp_physics", "ra_lw_physics", "ra_sw_physics",
            "sf_sfclay_physics", "sf_surface_physics", "bl_pbl_physics",
            "cu_physics")}
    # The combined spelling states the pair it runs.
    combined = {**recorded, "ra_physics": 4, "ra_lw_physics": -1,
                "ra_sw_physics": -1}
    assert recorded_physics_selectors(combined)["ra_lw_physics"] == 4
    assert recorded_physics_selectors(combined)["ra_sw_physics"] == 4
    without = {key: value for key, value in recorded.items()
               if key != "cu_physics"}
    with pytest.raises(ValueError, match=r"records no \['cu_physics'\]"):
        recorded_physics_selectors(without, label="d02 direct-export")


def _wrf_side_strips(field, width):
    """The four ``*_B??`` layouts of one mass-level field, WRF order.

    West and east are ``(width, z, y)``, south and north ``(width, z,
    x)``, east and north outermost first (share/module_bc.F).
    """

    return {
        "XS": field[:, :, :width].transpose(2, 0, 1),
        "XE": field[:, :, -width:][:, :, ::-1].transpose(2, 0, 1),
        "YS": field[:, :width, :].transpose(1, 0, 2),
        "YE": field[:, -width:, :][:, ::-1, :].transpose(1, 0, 2),
    }


def test_the_export_holds_one_theta_representation_and_says_which(tmp_path):
    """Dry theta in the header, the prognostic record and the boundary.

    WRF couples its PROGNOSTIC into ``T_B*`` (main/real_em.F:872) and
    reads the prognostic from ``THM``; ``USE_THETA_M`` says which theta
    that is, and wrf.exe's input gate compares it with the namelist.  The
    exporter used to write ``USE_THETA_M = 1`` and a moist ``THM`` over
    boundary rows that were the engine's own dry-coupled tables, so a
    ``use_theta_m = 1`` wrf.exe read dry rows as moist theta: measured
    with stock WRF V4.6.1 on this very case, the specified boundary row
    dropped 2.887 K at the first step (theta * a*qv / (1 + a*qv), a =
    Rv/Rd, predicted 2.8874 K), and held to 3e-5 K once the pair was dry.
    This runs the real route on the CPU (initialization, prepared cache,
    export) and reads the bytes back.
    """

    output, _cfg, _ = _export_a_real_prepared_cache(tmp_path, 6)
    rvovrd = 461.6 / 287.0
    with netCDF4.Dataset(output / "wrfinput_d01") as initial:
        with netCDF4.Dataset(output / "wrfbdy_d01") as boundary:
            assert int(initial.getncattr("USE_THETA_M")) == EXPORT_USE_THETA_M
            assert int(boundary.getncattr("USE_THETA_M")) == EXPORT_USE_THETA_M
            assert EXPORT_USE_THETA_M == 0
            dry_t = np.asarray(initial.variables["T"][0], dtype=np.float32)
            thm = np.asarray(initial.variables["THM"][0], dtype=np.float32)
            # What real.exe writes under use_theta_m = 0: THM is T, bit
            # for bit.
            np.testing.assert_array_equal(
                thm.view(np.uint32), dry_t.view(np.uint32))
            qv = np.asarray(initial.variables["QVAPOR"][0], dtype=np.float64)
            # Moist enough that the two representations cannot be
            # confused.
            assert qv.max() > 1.0e-3
            mu = (np.asarray(initial.variables["MU"][0], dtype=np.float64)
                  + np.asarray(initial.variables["MUB"][0], dtype=np.float64))
            c1h = np.asarray(initial.variables["C1H"][0], dtype=np.float64)
            c2h = np.asarray(initial.variables["C2H"][0], dtype=np.float64)
            mass = c1h[:, None, None] * mu[None] + c2h[:, None, None]
            theta = dry_t.astype(np.float64) + 300.0
            dry = mass * (theta - 300.0)
            moist = mass * (theta * (1.0 + rvovrd * qv) - 300.0)
            width = boundary.dimensions["bdy_width"].size
            # The tables are the engine's own coupling of its float32
            # state, not real.exe's product of the file's T: they sit
            # within a few float32 steps of the full theta, carried
            # through the mass.
            bound = (4.0 * float(np.spacing(np.float32(theta.max())))
                     * float(mass.max()))
            dry_strips = _wrf_side_strips(dry, width)
            moist_strips = _wrf_side_strips(moist, width)
            for suffix in ("XS", "XE", "YS", "YE"):
                table = np.asarray(
                    boundary.variables[f"T_B{suffix}"][0], dtype=np.float64)
                assert table.shape == dry_strips[suffix].shape, suffix
                assert (np.abs(table - dry_strips[suffix]).max()
                        <= bound), suffix
                # ... and nowhere near the moist coupling the old header
                # claimed: thousands of times the bound.
                assert (np.abs(table - moist_strips[suffix]).max()
                        > 1.0e3 * bound), suffix
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["use_theta_m"] == EXPORT_USE_THETA_M


def test_an_hrrr_width_boundary_exports_its_own_ten_row_wrfbdy(tmp_path):
    """U-E6: spec_bdy_width = 10 (the HRRR recipe's 1 + 9) exports.

    The exporter required spec_bdy_width = 5, the frozen v4.6.1 file's
    bdy_width, so every HRRR-recipe preparation had its companion
    wrfinput/wrfbdy refused ("unsupported direct-export configuration:
    {'spec_bdy_width': (10, 5)}").  The real CPU route here (cold start,
    prepared cache, export) writes a 10-row wrfbdy whose strips are the
    prepared tables' and which the real WRF-input reader takes back.
    """
    from woof.ingest.wrfinput import read_wrfinput, read_wrfbdy

    output, cfg, _ = _export_a_real_prepared_cache(
        tmp_path, 6, zones=(10, 1, 9), cells=28)
    assert cfg.spec_bdy_width == 10
    with netCDF4.Dataset(output / "wrfbdy_d01") as boundary:
        assert boundary.dimensions["bdy_width"].size == 10
        for suffix in ("XS", "XE", "YS", "YE"):
            assert boundary.variables[f"T_B{suffix}"].shape[1] == 10
            assert np.isfinite(np.asarray(
                boundary.variables[f"T_B{suffix}"][0])).all()
    dims = dict(west_east=cfg.nx, west_east_stag=cfg.nx + 1,
                south_north=cfg.ny, south_north_stag=cfg.ny + 1,
                bottom_top=cfg.nz, bottom_top_stag=cfg.nz + 1,
                soil_layers_stag=4)
    restored = read_wrfinput(output / "wrfinput_d01", cfg=cfg,
                             expected_dimensions=dims, require_complete=False)
    read_back = read_wrfbdy(output / "wrfbdy_d01", restored=restored,
                            run_seconds=3600, forcing_interval_seconds=3600,
                            spec_bdy_width=cfg.spec_bdy_width, cfg=cfg)
    assert len(read_back.intervals) == 1


def test_the_export_refuses_a_boundary_narrower_than_its_zones(tmp_path):
    """The width is free, the zones still have to fit in it."""

    cache = _minimal_prepared_cache(tmp_path / "prepared", {
        "mp_physics": 6, "hybrid_opt": 2, "hypsometric_opt": 2,
        "specified": True, "nested": False, "spec_bdy_width": 8,
        "spec_zone": 1, "relax_zone": 9,
        "bl_pbl_physics": 1, "sf_sfclay_physics": 91, "sf_surface_physics": 2,
    })
    with pytest.raises(StockWrfExportUnsupported) as refusal:
        export_prepared_wrf(
            cache, tmp_path / "static.npz", tmp_path / "geometry.json",
            tmp_path / "wrf-native-input",
            valid_time=datetime(2026, 7, 29, 6))
    assert set(refusal.value.unsupported) == {"spec_bdy_width"}


def test_dry_export_roundtrips_through_the_real_wrf_input_reader(tmp_path):
    """The initial dry theta must match the boundary's FP32 coupling.

    Summing thb + thp in float64 before writing T retained bits that the
    model's float32 total_theta discarded before building its boundary
    tables.  Under a dry declaration the reader correctly rejects that
    difference, especially where perturbation theta is close to zero.
    """
    from woof.ingest.wrfinput import read_wrfinput, read_wrfbdy

    output, cfg, _ = _export_a_real_prepared_cache(tmp_path, 6)
    dims = dict(west_east=cfg.nx, west_east_stag=cfg.nx + 1,
                south_north=cfg.ny, south_north_stag=cfg.ny + 1,
                bottom_top=cfg.nz, bottom_top_stag=cfg.nz + 1,
                soil_layers_stag=4)
    restored = read_wrfinput(output / "wrfinput_d01", cfg=cfg,
                             expected_dimensions=dims, require_complete=False)
    boundary = read_wrfbdy(output / "wrfbdy_d01", restored=restored,
                           run_seconds=3600, forcing_interval_seconds=3600,
                           cfg=cfg)
    assert len(boundary.intervals) == 1


def _replace_the_start(state):
    """A separate initial analysis: every boundary field of the start state
    differs from the atmosphere the boundary head was built from, as the
    mapped route's ``--initial-inputs`` leaves a prepared cache."""
    state.u[...] += np.float32(1.5)
    state.v[...] -= np.float32(0.75)
    state.thp[...] += np.float32(0.4)
    state.php[...] += np.float32(30.0)
    state.mup[...] += np.float32(40.0)
    state.qv[...] *= np.float32(1.03)
    state.nwfa[...] *= np.float32(1.2)
    state.nifa[...] *= np.float32(0.8)


def _door_read(output, cfg):
    from woof.ingest.wrfinput import read_wrfinput, read_wrfbdy

    dims = dict(west_east=cfg.nx, west_east_stag=cfg.nx + 1,
                south_north=cfg.ny, south_north_stag=cfg.ny + 1,
                bottom_top=cfg.nz, bottom_top_stag=cfg.nz + 1,
                soil_layers_stag=4)
    restored = read_wrfinput(output / "wrfinput_d01", cfg=cfg,
                             expected_dimensions=dims, require_complete=False)
    boundary = read_wrfbdy(output / "wrfbdy_d01", restored=restored,
                           run_seconds=7200, forcing_interval_seconds=3600,
                           cfg=cfg)
    return restored, boundary


def _aerosol_forced_export(tmp_path):
    """mp=28 whose boundary forces the aerosol numbers, start replaced after
    the boundary was built, two hourly intervals; the door reads it under
    ``use_aero_icbc = .true.`` (aer_init_opt = 1), so QNWFA/QNIFA are
    checked too."""
    import dataclasses

    output, cfg, _ = _export_a_real_prepared_cache(
        tmp_path, 28, boundary_scalars=("qv", "nwfa", "nifa"),
        replace_start=_replace_the_start, intervals=2)
    return output, dataclasses.replace(cfg, aer_init_opt=1)


def test_a_replaced_start_state_exports_a_pair_the_wrf_door_accepts(tmp_path):
    """--stock-wrf-export, then the WRF-input door, on WOOF's own files.

    The mapped route builds its boundary head from the boundary source and
    then replaces the start state with the separate initial analysis.  The
    export copied that head into wrfbdy record 0, so the first record
    described another atmosphere than wrfinput_d01, and the door refused
    WOOF's own pair ("wrfbdy U west does not match initial wrfinput_d01:
    100,250 of 100,250 points differ" on a 400 x 400 RAP crop).  Its
    aerosol tables were zeros beside a nonzero QNWFA, the second refusal
    behind the first.  real.exe's first record is always the coupling of
    its own wrfinput; WRFDA's da_update_bc is WRF's rule for a replaced
    start: the first record becomes the new start's coupling and its
    tendency keeps the interval's end.  The door accepts the pair, restores
    exactly the prepared start state, and its forcing reaches the prepared
    boundary at the end of the first interval and from then on.
    """
    from woof.ingest.lateral_bc import evaluate_boundary_side

    output, cfg = _aerosol_forced_export(tmp_path)
    restored, boundary = _door_read(output, cfg)
    assert len(boundary.intervals) == 2
    assert {"u", "v", "theta", "phi", "mu", "qv", "nwfa", "nifa"} <= set(
        boundary.intervals[0].fields)

    cache = PreparedCache(tmp_path / "prepared-cache")
    # The door restores the prepared START state, bit for bit.
    for wrf_name, key in (("U", "state/u"), ("V", "state/v"),
                          ("PH", "state/php"), ("MU", "state/mup"),
                          ("QVAPOR", "state/qv"), ("QNWFA", "state/nwfa"),
                          ("QNIFA", "state/nifa")):
        np.testing.assert_array_equal(
            np.asarray(restored.raw[wrf_name], np.float32).view(np.uint32),
            np.asarray(cache.array(key), np.float32).view(np.uint32),
            err_msg=wrf_name)
    theta = (np.asarray(cache.array("base/thb"), np.float32)
             + np.asarray(cache.array("state/thp"), np.float32)
             - np.float32(300.0))
    np.testing.assert_array_equal(
        np.asarray(restored.raw["T"], np.float32).view(np.uint32),
        theta.view(np.uint32))

    # The first interval ends on the prepared boundary, and the second
    # starts there: the replaced start changes the first record only.
    for name in ("u", "v", "theta", "phi", "mu", "qv", "nwfa", "nifa"):
        for side in ("west", "east", "south", "north"):
            prefix = f"lbc/0/{name}/{side}"
            prepared_end = (
                np.asarray(cache.array(f"{prefix}/value"), np.float64)
                + 3600.0 * np.asarray(cache.array(f"{prefix}/tendency"),
                                      np.float64))
            door_end, _ = evaluate_boundary_side(
                getattr(boundary.intervals[0].fields[name], side), 3600.0)
            second, _ = evaluate_boundary_side(
                getattr(boundary.intervals[1].fields[name], side), 0.0)
            prepared_second = np.asarray(
                cache.array(f"lbc/1/{name}/{side}/value"), np.float64)
            scale = np.abs(prepared_end).max() + 1e-30
            assert np.abs(door_end - prepared_end).max() <= 1e-5 * scale, (
                name, side)
            assert np.abs(second - prepared_second).max() <= 1e-5 * scale, (
                name, side)
    with netCDF4.Dataset(output / "wrfbdy_d01") as dataset:
        for suffix, side in (("XS", "west"), ("YE", "north")):
            for wrf_name, name in (("U", "u"), ("QNWFA", "nwfa")):
                for kind, part in (("B", "value"), ("BT", "tendency")):
                    written = np.asarray(
                        dataset[f"{wrf_name}_{kind}{suffix}"][1], np.float32)
                    np.testing.assert_array_equal(
                        written, np.asarray(
                            _lbc_to_wrf(cache, 1, name, side, part),
                            np.float32))


def test_the_door_still_refuses_a_first_record_that_is_not_its_wrfinput(
        tmp_path):
    """The check stays: the prepared head under the replaced start is what
    the export used to write, and it is a different atmosphere."""
    from woof.ingest.wrfinput import read_wrfinput, read_wrfbdy

    output, cfg = _aerosol_forced_export(tmp_path)
    cache = PreparedCache(tmp_path / "prepared-cache")
    with netCDF4.Dataset(output / "wrfbdy_d01", "a") as dataset:
        for suffix, side in (("XS", "west"), ("XE", "east"),
                             ("YS", "south"), ("YE", "north")):
            dataset[f"U_B{suffix}"][0] = _lbc_to_wrf(
                cache, 0, "u", side, "value")
    with pytest.raises(ValueError, match="wrfbdy U west does not match "
                       "initial wrfinput_d01"):
        _door_read(output, cfg)


def test_global_updates_keep_stock_wrf_v4_gate_and_geometry():
    geometry = {
        "center_lat": 35.5,
        "center_lon": -98.0,
        "ref_lat": 35.5,
        "truelat1": 38.5,
        "truelat2": 38.5,
        "stand_lon": -97.5,
    }
    updates = _global_updates(
        valid_time=datetime(2026, 7, 18), nx=500, ny=400, nz=49,
        dx=1000.0, dy=1000.0, dt=5.0, geometry=geometry)
    assert "GPUWM" in updates["TITLE"]
    assert "V4.6.1" in updates["TITLE"]
    # The frozen contract holds real.exe's captured USE_THETA_M = 1; the
    # export overrides it with the representation it actually writes.
    assert updates["USE_THETA_M"] == EXPORT_USE_THETA_M == 0
    assert _load_contract()["wrfinput"]["global_attributes"][
        "USE_THETA_M"] == 1
    assert "USE_THETA_M" in _domain_global_attributes(updates)
    assert updates["WEST-EAST_GRID_DIMENSION"] == 501
    assert updates["SOUTH-NORTH_GRID_DIMENSION"] == 401
    assert updates["GHG_INPUT"] == 0
    assert updates["GRID_ID"] == 1
    assert updates["PARENT_ID"] == 0


def test_front_door_global_updates_carry_selected_physics():
    geometry = {
        "center_lat": 35.5,
        "center_lon": -98.0,
        "ref_lat": 35.5,
        "truelat1": 38.5,
        "truelat2": 38.5,
        "stand_lon": -97.5,
    }
    selection = validate_single_domain_physics_profile(MYNN_PROFILE_ID)

    updates = _global_updates(
        valid_time=datetime(2026, 7, 18), nx=20, ny=16, nz=8,
        dx=1000.0, dy=1000.0, dt=5.0, geometry=geometry,
        physics_selection=selection)

    assert updates["MP_PHYSICS"] == 6
    assert updates["SF_SFCLAY_PHYSICS"] == 5
    assert updates["BL_PBL_PHYSICS"] == 5
    assert updates["SF_SURFACE_PHYSICS"] == 2
    assert updates["RA_LW_PHYSICS"] == 0
    assert updates["RA_SW_PHYSICS"] == 1
    assert updates["CU_PHYSICS"] == 0


@pytest.mark.parametrize(
    ("surface", "layers"),
    ((2, 4), (3, 9), (4, 4)),
)
def test_direct_export_soil_geometry_follows_selected_surface_scheme(
        surface, layers):
    depths, thicknesses = _direct_export_soil_geometry(surface, layers)

    assert depths.shape == (layers,)
    assert thicknesses.shape == (layers,)
    assert np.all(np.diff(depths) > 0.0)
    assert np.all(thicknesses > 0.0)


def test_direct_export_dimensions_follow_ruc_nine_layer_state():
    dimensions = _dimensions(
        _load_contract()["wrfinput"], nx=31, ny=23, nz=17,
        num_soil_layers=9)

    assert dimensions["soil_layers_stag"] == 9


@pytest.mark.parametrize("nz", [4, 17, 49, 80, 113])
def test_direct_export_dimensions_are_derived_for_any_level_count(nz):
    contract = _load_contract()
    for role in ("wrfinput", "wrfbdy"):
        dimensions = _dimensions(contract[role], nx=31, ny=23, nz=nz)
        assert dimensions["bottom_top"] == nz
        assert dimensions["bottom_top_stag"] == nz + 1


def _synthetic_vertical_cache(nz, *, nx=3, ny=2, p_top=12_345.0):
    coord_shapes = expected_coordinate_shapes(nz)
    shapes = {f"coord/{name}": shape
              for name, shape in coord_shapes.items()}
    shapes.update({
        "state/u": (nz, ny, nx + 1),
        "state/v": (nz, ny + 1, nx),
        "state/w": (nz + 1, ny, nx),
        "state/php": (nz + 1, ny, nx),
        "state/thp": (nz, ny, nx),
        "state/qv": (nz, ny, nx),
        "state/qc": (nz, ny, nx),
        "state/qr": (nz, ny, nx),
        "state/qi": (nz, ny, nx),
        "state/qs": (nz, ny, nx),
        "state/qg": (nz, ny, nx),
        "state/mup": (ny, nx),
        "base/mub": (ny, nx),
        "base/pb": (nz, ny, nx),
        "base/alb": (nz, ny, nx),
        "base/thb": (nz, ny, nx),
        "base/phb": (nz + 1, ny, nx),
        "base/terrain_z": (ny, nx),
    })

    class Cache:
        _arrays = {name: {"shape": list(shape)}
                   for name, shape in shapes.items()}
        header = {"metadata": {
            "base_scalars": {"p_top": p_top},
            "coord_scalars": {"p_top": p_top},
        }}

        @staticmethod
        def array(name):
            assert name == "coord/znw"
            return np.linspace(1.0, 0.0, nz + 1)

    return Cache()


@pytest.mark.parametrize("nz", [4, 17, 49, 80, 113])
def test_prepared_export_vertical_contract_is_count_agnostic(nz):
    cache = _synthetic_vertical_cache(nz)
    assert _prepared_vertical_contract(
        cache, nx=3, ny=2, nz=nz) == 12_345.0


def test_prepared_export_vertical_contract_rejects_dimension_drift():
    cache = _synthetic_vertical_cache(80)
    cache._arrays["state/w"] = {"shape": [80, 2, 3]}
    with pytest.raises(ValueError, match="shape drift"):
        _prepared_vertical_contract(cache, nx=3, ny=2, nz=80)


def test_zero_prototype_scaffold_derives_live_vertical_shape(tmp_path):
    path = tmp_path / "prototype.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("bottom_top", 80)
        variable = dataset.createVariable(
            "T_BASE", "f4", ("Time", "bottom_top"))
        resolved = _resolved_prototype_value(variable, [[0.0] * 49])
        assert resolved.shape == (80,)
        assert resolved.dtype == np.float32
        np.testing.assert_array_equal(resolved, 0.0)


def test_nonzero_mismatched_prototype_remains_fail_closed(tmp_path):
    path = tmp_path / "prototype.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("bottom_top", 80)
        variable = dataset.createVariable(
            "PROFILE", "f4", ("Time", "bottom_top"))
        with pytest.raises(ValueError, match="nonzero frozen prototype"):
            _resolved_prototype_value(variable, [[1.0] * 49])


def test_noah_landuse_remaps_lakes_but_preserves_mask():
    raw = np.array([[1, 17, 21], [13, 21, 20]], dtype=np.int32)
    mapped, lakes = _wrf_noah_landuse(raw)
    np.testing.assert_array_equal(
        mapped, np.array([[1, 17, 17], [13, 17, 20]], dtype=np.int32))
    np.testing.assert_array_equal(
        lakes, np.array([[False, False, True], [False, True, False]]))


def test_stock_export_writes_the_soil_category_real_exe_writes():
    """Land over water soil carries silty clay loam; water carries water."""
    raw_lu = np.array([[12, 17, 21, 10, 13]], dtype=np.int32)
    sct_dom = np.array([[14.0, 14.0, 14.0, 6.0, 14.0]], dtype=np.float32)
    lu, _ = _wrf_noah_landuse(raw_lu)
    np.testing.assert_array_equal(
        _wrf_soil_category(sct_dom, lu), [[8, 14, 14, 6, 8]])
    np.testing.assert_array_equal(lu, [[12, 17, 17, 10, 13]])


def test_lbc_orientation_matches_wrf_side_layout():
    west = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    actual_west = _lbc_to_wrf(
        _Cache(west), 0, "u", "west", "value")
    np.testing.assert_array_equal(actual_west, np.transpose(west, (2, 0, 1)))

    south = np.arange(2 * 5 * 4).reshape(2, 5, 4)
    actual_south = _lbc_to_wrf(
        _Cache(south), 0, "u", "south", "value")
    np.testing.assert_array_equal(actual_south, np.transpose(south, (1, 0, 2)))

    mu_west = np.arange(1 * 3 * 5).reshape(1, 3, 5)
    actual_mu = _lbc_to_wrf(
        _Cache(mu_west), 0, "mu", "west", "value")
    np.testing.assert_array_equal(actual_mu, mu_west[0].T)


def test_hourly_forcing_inventory_supports_arbitrary_horizon():
    class Cache:
        header = {
            "identity": {"forcing_hours": list(range(25))},
            "arrays": {f"lbc/{index}/u/west/value": {}
                       for index in range(24)},
        }

    assert _forcing_interval_indices(Cache(), 3600) == list(range(24))


def test_forcing_inventory_supports_six_hour_cadence():
    class Cache:
        header = {
            "identity": {"forcing_hours": [0, 6, 12]},
            "arrays": {f"lbc/{index}/u/west/value": {}
                       for index in range(2)},
        }

    assert _forcing_interval_indices(Cache(), 21_600) == [0, 1]


def test_forcing_inventory_supports_five_minute_offsets():
    class Cache:
        header = {
            "identity": {"forcing_offsets_seconds": [0, 300, 600]},
            "arrays": {f"lbc/{index}/u/west/value": {}
                       for index in range(2)},
        }

    assert _forcing_interval_indices(Cache(), 300) == [0, 1]


def test_forcing_inventory_fails_closed_on_cadence_mismatch():
    class Cache:
        header = {
            "identity": {"forcing_hours": [0, 1, 3]},
            "arrays": {
                "lbc/0/u/west/value": {},
                "lbc/1/u/west/value": {},
            },
        }

    try:
        _forcing_interval_indices(Cache(), 3600)
    except ValueError as exc:
        assert "cadence" in str(exc)
    else:
        raise AssertionError("gapped forcing hours were accepted")


def test_canonical_surface_bypasses_source_specific_soil_nodes():
    names = {
        "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
        "LANDMASK", "SNOW", "SNOWH",
    }
    values = {name: np.ones((2, 2), dtype=np.float64) for name in names}
    for name in ("TSLB", "SMOIS", "SH2O"):
        values[name] = np.ones((4, 2, 2), dtype=np.float64)
    values["TSK"] *= 280.0
    values["TMN"] *= 279.0

    class Cache:
        _arrays = {f"surface/{name}": {} for name in names}

        def array(self, name):
            return values[name.split("/", 1)[1]]

    static = {
        "GREENFRAC": np.full((12, 2, 2), 0.5),
        "ALBEDO12M": np.full((12, 2, 2), 20.0),
        "LAI12M": np.full((12, 2, 2), 2.0),
        "SNOALB": np.full((2, 2), 60.0),
        "LANDMASK": np.ones((2, 2)),
    }
    actual = _surface_fields(Cache(), static, datetime(2024, 4, 15))
    np.testing.assert_array_equal(actual["TSLB"], values["TSLB"])
    np.testing.assert_array_equal(actual["SH2O"], values["SH2O"])
    np.testing.assert_allclose(actual["VEGFRA"], 50.0)


def _frozen_soil_case(coldest_kelvin: float):
    """A met-node surface case whose coldest soil node is COLDEST_KELVIN."""

    nodes = np.full((9, 2, 2), 285.0)
    # The HRRR source nodes bracketing Noah's deepest midpoint (1.50 m),
    # in one column of four: the measured shape of the field failure.
    nodes[6:8, 1, 1] = coldest_kelvin
    values = {
        "met/SKINTEMP": np.full((2, 2), 288.0),
        "met/XICE": np.zeros((2, 2)),
        "met/SOILT": nodes,
        "met/SOILW": np.full((9, 2, 2), 0.25),
    }

    class Cache:
        _arrays: dict = {}

        def array(self, name):
            return values[name]

    static = {
        "LANDMASK": np.ones((2, 2)),
        "TMN": np.full((2, 2), 283.0),
        "GREENFRAC": np.full((12, 2, 2), 0.5),
        "ALBEDO12M": np.full((12, 2, 2), 20.0),
        "LAI12M": np.full((12, 2, 2), 2.0),
        "SNOALB": np.full((2, 2), 60.0),
    }
    return Cache(), static


def test_frozen_soil_refusal_is_typed_export_representability():
    """The refusal a caller PREPARING a forecast has to be able to catch.

    woof's Noah surface layer integrates frozen soil perfectly well;
    only the unchanged-WRF wrfinput file set has no table-driven
    SH2O partition to write here.  That is what
    ``StockWrfExportUnsupported`` is for, and as a bare ``ValueError``
    this one was uncatchable -- a nested HRRR preparation over the
    mountainous west finished every expensive stage and then discarded
    a complete, verified GPU hierarchy because one soil node in one
    column, at the deepest level, was below freezing.
    """
    cache, static = _frozen_soil_case(272.0)
    with pytest.raises(StockWrfExportUnsupported, match="frozen-soil SH2O"):
        _surface_fields(cache, static, datetime(2024, 4, 15))


def test_a_warm_column_still_exports_so_the_refusal_is_not_free():
    """Negative control: the same case above freezing writes SH2O."""
    cache, static = _frozen_soil_case(285.0)
    fields = _surface_fields(cache, static, datetime(2024, 4, 15))
    np.testing.assert_array_equal(fields["SH2O"], fields["SMOIS"])


def test_timestamp_writer_appends_multiple_boundary_records(tmp_path):
    path = tmp_path / "times.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("DateStrLen", 19)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        _write_timestamp_at(times, "2026-07-18_00:00:00", 0)
        _write_timestamp_at(times, "2026-07-18_01:00:00", 1)
    with netCDF4.Dataset(path) as dataset:
        decoded = [b"".join(row).decode("ascii")
                   for row in dataset.variables["Times"][:]]
    assert decoded == ["2026-07-18_00:00:00", "2026-07-18_01:00:00"]


def test_hrrr_nodes_interpolate_to_noah_midpoints():
    depths = np.array([0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0])
    nodes = depths[:, None, None] * 10.0 + 280.0
    output = _interp_nodes(nodes)
    np.testing.assert_allclose(
        output[:, 0, 0], 280.0 + 10.0 * np.array([0.05, 0.25, 0.70, 1.50]))


def test_valid_time_is_normalized_to_utc():
    value = datetime(2026, 7, 17, 17, tzinfo=timezone(-timedelta(hours=7)))
    assert _date_text(value) == "2026-07-18_00:00:00"


def test_prepared_array_digest_is_verified(tmp_path):
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "array.npy"
    np.save(path, value)
    digest = _array_sha256(value)
    header = {
        "schema": "gpuwm-prepared-real-cache-v1",
        "status": "READY",
        "identity": {},
        "metadata": {},
        "payload_bytes": int(value.nbytes),
        "arrays": {
            "state/test": {
                "file": path.name,
                "shape": [2, 3],
                "dtype": "float32",
                "sha256": digest,
            },
        },
    }
    basis = {name: header[name] for name in (
        "schema", "identity", "metadata", "arrays", "payload_bytes")}
    header["content_sha256"] = hashlib.sha256(
        _canonical(basis).encode("utf-8")).hexdigest()
    (tmp_path / "header.json").write_text(json.dumps(header), encoding="utf-8")
    np.testing.assert_array_equal(PreparedCache(tmp_path).array("state/test"), value)

    mutated = np.load(path, mmap_mode="r+")
    mutated[0, 0] = 99.0
    mutated.flush()
    del mutated
    try:
        PreparedCache(tmp_path).array("state/test")
    except ValueError as exc:
        assert "digest drift" in str(exc)
    else:
        raise AssertionError("modified cache payload was accepted")


def _domain(grid_id, parent_id, *, specified, nested):
    return SimpleNamespace(
        grid_id=grid_id,
        parent_id=parent_id,
        i_parent_start=1 if grid_id == 1 else 20,
        j_parent_start=1 if grid_id == 1 else 15,
        parent_grid_ratio=1 if grid_id == 1 else 3,
        parent_time_step_ratio=1 if grid_id == 1 else 3,
        run=SimpleNamespace(specified=specified, nested=nested),
    )


def _artifact(grid_id):
    return PreparedDomainArtifacts(
        grid_id, Path(f"prepared-d{grid_id:02d}"),
        Path(f"static-d{grid_id:02d}.npz"),
        Path(f"geometry-d{grid_id:02d}.json"))


def test_hierarchy_pairs_artifacts_by_contiguous_namelist_grid_id():
    d01 = _domain(1, 0, specified=True, nested=False)
    d02 = _domain(2, 1, specified=False, nested=True)
    exp = SimpleNamespace(
        domains=(d01, d02),
        projection=SimpleNamespace(map_proj="lambert"))
    pairs = _validated_hierarchy(exp, (_artifact(2), _artifact(1)))
    assert [(domain.grid_id, artifact.grid_id)
            for domain, artifact in pairs] == [(1, 1), (2, 2)]


def test_hierarchy_rejects_missing_domain_artifact():
    d01 = _domain(1, 0, specified=True, nested=False)
    d02 = _domain(2, 1, specified=False, nested=True)
    exp = SimpleNamespace(
        domains=(d01, d02),
        projection=SimpleNamespace(map_proj="lambert"))
    with pytest.raises(ValueError, match="exactly cover"):
        _validated_hierarchy(exp, (_artifact(1),))


def test_hierarchy_rejects_artifact_misbinding_and_parent_cycle():
    d01 = _domain(1, 0, specified=True, nested=False)
    d02 = _domain(2, 1, specified=False, nested=True)
    exp = SimpleNamespace(
        domains=(d01, d02),
        projection=SimpleNamespace(map_proj="lambert"))

    with pytest.raises(ValueError, match="exactly cover"):
        _validated_hierarchy(exp, (_artifact(1), _artifact(3)))
    with pytest.raises(ValueError, match="duplicate artifacts"):
        _validated_hierarchy(exp, (_artifact(1), _artifact(2), _artifact(2)))

    d02.parent_id = 2
    with pytest.raises(ValueError, match="earlier parent"):
        _validated_hierarchy(exp, (_artifact(1), _artifact(2)))


def test_child_global_updates_emit_wrf_nest_identity():
    domain = _domain(2, 1, specified=False, nested=True)
    cfg = {"nx": 60, "ny": 45, "nz": 8,
           "dx": 4000.0, "dy": 4000.0, "dt": 20.0, **_WSM6_STOCK_PHYSICS}
    geometry = {
        "center_lat": 36.0, "center_lon": -97.0,
        "ref_lat": 39.7, "ref_lon": -83.9,
        "truelat1": 30.0, "truelat2": 60.0,
        "stand_lon": -83.9,
    }
    updates = _hierarchy_global_updates(
        valid_time=datetime(1999, 5, 3, 12), cfg=cfg,
        geometry=geometry, domain=domain)
    assert updates["GRID_ID"] == 2
    assert updates["PARENT_ID"] == 1
    assert updates["I_PARENT_START"] == 20
    assert updates["J_PARENT_START"] == 15
    assert updates["PARENT_GRID_RATIO"] == 3
    assert updates["DT"] == 20.0


def test_domain_artifact_manifest_is_relocatable_and_strict(tmp_path):
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(json.dumps({
        "schema": "gpuwm-native-domain-artifacts-v1",
        "domains": [{
            "grid_id": 1,
            "prepared_cache": "cache/d01",
            "static_cache": "static/d01.npz",
            "geometry_receipt": "geometry/d01.json",
        }],
    }), encoding="utf-8")
    (artifact,) = load_domain_artifacts_manifest(manifest)
    assert artifact.prepared_cache == tmp_path / "cache" / "d01"
    assert artifact.static_cache == tmp_path / "static" / "d01.npz"

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["domains"][0]["unbound"] = "forbidden"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys must be exactly"):
        load_domain_artifacts_manifest(manifest)


def test_domain_artifact_manifest_rejects_absolute_and_parent_escape(tmp_path):
    manifest = tmp_path / "artifacts.json"
    base = {
        "schema": "gpuwm-native-domain-artifacts-v1",
        "domains": [{
            "grid_id": 1,
            "prepared_cache": "cache/d01",
            "static_cache": "static/d01.npz",
            "geometry_receipt": "geometry/d01.json",
        }],
    }
    base["domains"][0]["prepared_cache"] = str(tmp_path.resolve())
    manifest.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="relative path"):
        load_domain_artifacts_manifest(manifest)

    base["domains"][0]["prepared_cache"] = "../escape"
    manifest.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_domain_artifacts_manifest(manifest)


def test_domain_artifact_manifest_writer_round_trips_relative_inventory(
        tmp_path):
    artifacts = []
    for grid_id in (1, 2):
        root = tmp_path / f"d{grid_id:02d}"
        prepared = root / "prepared-cache"
        prepared.mkdir(parents=True)
        (prepared / "header.json").write_text("{}", encoding="utf-8")
        static = root / "native-static.npz"
        static.write_bytes(b"static")
        geometry = root / "geometry.json"
        geometry.write_text("{}", encoding="utf-8")
        artifacts.append(PreparedDomainArtifacts(
            grid_id, prepared, static, geometry))

    manifest = tmp_path / "domain-artifacts.json"
    payload = write_domain_artifacts_manifest(
        manifest, tuple(reversed(artifacts)))
    loaded = load_domain_artifacts_manifest(manifest)

    assert [item["grid_id"] for item in payload["domains"]] == [1, 2]
    assert payload["domains"][0]["prepared_cache"] \
        == "d01/prepared-cache"
    assert [item.grid_id for item in loaded] == [1, 2]
    assert loaded[1].static_cache == artifacts[1].static_cache.resolve()
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_file_validation_checks_real_netcdf_domain_global_attributes(tmp_path):
    contract = _load_contract()["wrfinput"]
    dimensions = _dimensions(contract, nx=3, ny=2, nz=49)
    geometry = {
        "center_lat": 35.5, "center_lon": -98.0,
        "ref_lat": 35.5, "ref_lon": -98.0,
        "truelat1": 30.0, "truelat2": 60.0,
        "stand_lon": -97.0,
    }
    updates = _global_updates(
        valid_time=datetime(2026, 7, 18), nx=3, ny=2, nz=49,
        dx=1000.0, dy=1000.0, dt=5.0, geometry=geometry)
    path = tmp_path / "wrfinput_d01"
    _write_wrfinput(
        path, contract, dimensions, updates, {}, "2026-07-18_00:00:00")
    expected = _domain_global_attributes(updates)
    result = _validate_file(
        path, contract, nx=3, ny=2, nz=49,
        expected_global_attributes=expected)
    assert result["bytes"] > 0

    with netCDF4.Dataset(path, "r+") as dataset:
        dataset.setncattr("GRID_ID", np.int32(2))
    with pytest.raises(ValueError, match="global attribute drift"):
        _validate_file(
            path, contract, nx=3, ny=2, nz=49,
            expected_global_attributes=expected)

    child_cfg = {
        "nx": 3, "ny": 2, "nz": 49,
        "dx": 1000.0 / 3.0, "dy": 1000.0 / 3.0, "dt": 5.0 / 3.0,
        **_WSM6_STOCK_PHYSICS,
    }
    child = _domain(2, 1, specified=False, nested=True)
    child_updates = _hierarchy_global_updates(
        valid_time=datetime(2026, 7, 18), cfg=child_cfg,
        geometry=geometry, domain=child)
    child_path = tmp_path / "wrfinput_d02"
    _write_wrfinput(
        child_path, contract, dimensions, child_updates, {},
        "2026-07-18_00:00:00")
    _validate_file(
        child_path, contract, nx=3, ny=2, nz=49,
        expected_global_attributes=_domain_global_attributes(child_updates))
    with netCDF4.Dataset(child_path) as dataset:
        assert dataset.getncattr("GRID_ID") == 2
        assert dataset.getncattr("PARENT_ID") == 1
        assert dataset.getncattr("I_PARENT_START") == 20
        assert dataset.getncattr("J_PARENT_START") == 15
        assert dataset.getncattr("PARENT_GRID_RATIO") == 3
        assert dataset.getncattr("DX") == np.float32(1000.0 / 3.0)
        assert dataset.getncattr("DT") == np.float32(5.0 / 3.0)


def test_static_geometry_receipt_is_cache_and_namelist_bound(tmp_path):
    static = tmp_path / "native-static.npz"
    static.write_bytes(b"content-addressed-static")
    geometry = {
        "mass_shape": [2, 3], "nz": 4,
        "dx_m": 1000.0, "dy_m": 1000.0,
        "ref_lat": 35.5, "ref_lon": -98.0,
        "truelat1": 30.0, "truelat2": 60.0,
        "stand_lon": -97.0, "center_lat": 35.5,
        "center_lon": -98.0, "lat_range": [35.0, 36.0],
        "lon_range": [-99.0, -97.0],
    }
    receipt = tmp_path / "geometry.json"
    payload = {
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "cache": {
            "path": static.name, "bytes": static.stat().st_size,
            "sha256": hashlib.sha256(static.read_bytes()).hexdigest(),
        },
        "geometry": geometry,
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    observed, _digest = _load_static_geometry_receipt(
        receipt, static, expected_geometry=geometry)
    assert observed == geometry

    payload["geometry"] = {**geometry, "center_lon": -90.0}
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from namelist"):
        _load_static_geometry_receipt(
            receipt, static, expected_geometry=geometry)


def test_failed_overwrite_restores_previous_valid_tree(tmp_path, monkeypatch):
    output = tmp_path / "wrf-ready"
    output.mkdir()
    (output / "manifest.json").write_text("previous", encoding="utf-8")
    staging = tmp_path / "wrf-ready.tmp-123"
    staging.mkdir()
    (staging / "manifest.json").write_text("candidate", encoding="utf-8")
    backup = output.with_name(output.name + ".previous-valid")
    real_replace = wrf_direct.os.replace

    def fail_candidate_publish(source, destination):
        if Path(source) == staging and Path(destination) == output:
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(wrf_direct.os, "replace", fail_candidate_publish)
    with pytest.raises(OSError, match="injected"):
        _publish_staging(staging, output, backup)
    assert (output / "manifest.json").read_text(encoding="utf-8") == "previous"
    assert staging.exists()
    assert not backup.exists()


def test_interrupted_backup_is_recovered_before_next_overwrite(tmp_path):
    output = tmp_path / "wrf-ready"
    backup = output.with_name(output.name + ".previous-valid")
    backup.mkdir()
    (backup / "manifest.json").write_text("previous", encoding="utf-8")
    assert _prepare_output_target(output, overwrite=True) == backup
    assert (output / "manifest.json").read_text(encoding="utf-8") == "previous"
    assert not backup.exists()


#: The selectors a WSM6 + YSU + MM5 + Noah + Dudhia cache records, the
#: frozen contract's own physics.
_WSM6_STOCK_PHYSICS = {
    "mp_physics": 6, "ra_physics": 0, "ra_lw_physics": 0,
    "ra_sw_physics": 1, "sf_sfclay_physics": 91, "sf_surface_physics": 2,
    "bl_pbl_physics": 1, "cu_physics": 0,
}


def _hierarchy_context(domain, physics, static_hash):
    cfg = {name: getattr(domain.run, name)
           for name in ("nx", "ny", "nz", "dx", "dy", "dt", *physics)}
    geometry = {"center_lat": 36.0, "center_lon": -97.0,
                "ref_lat": 39.7, "ref_lon": -83.9, "truelat1": 30.0,
                "truelat2": 60.0, "stand_lon": -83.9}
    return _HierarchyCache(domain.grid_id), cfg, geometry, 5000.0, static_hash


def _hierarchy_experiment(domains):
    return SimpleNamespace(
        domains=tuple(domains), projection=SimpleNamespace(map_proj="lambert"),
        start_time=datetime(1999, 5, 3, 12),
        vertical=SimpleNamespace(
            eta_levels=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0),
            p_top=5000.0, hybrid_opt=2))


class _HierarchyCache:
    def __init__(self, grid_id):
        self.header_path = Path(f"header-d{grid_id:02d}.json")
        self.header = {"content_sha256": f"content-d{grid_id:02d}"}


@pytest.mark.parametrize("max_dom", range(1, 22))
def test_hierarchy_export_publishes_every_input_and_root_boundary_atomically(
        tmp_path, monkeypatch, max_dom):
    def run(nx, ny, dx, dt, *, specified, nested):
        # The physics a real cache records (asdict(RunConfig) carries
        # every selector), which each wrfinput states.
        return SimpleNamespace(
            nx=nx, ny=ny, nz=8, dx=dx, dy=dx, dt=dt,
            specified=specified, nested=nested, **_WSM6_STOCK_PHYSICS)

    d01 = _domain(1, 0, specified=True, nested=False)
    d01.run = run(100, 80, 12000.0, 60.0,
                  specified=True, nested=False)
    domains = [d01]
    for grid_id in range(2, max_dom + 1):
        # d01-d04 is a deep chain; d05 is its root sibling and d06 starts
        # a second branch.  Larger synthetic cases remain root siblings so
        # the 1..21 exporter cardinality test does not require huge grids.
        parent_id = ({2: 1, 3: 2, 4: 3, 5: 1, 6: 2}.get(grid_id, 1))
        domain = _domain(
            grid_id, parent_id, specified=False, nested=True)
        ratio_depth = 1 if parent_id == 1 else grid_id - 1
        domain.run = run(
            60, 60, 12000.0 / (3 ** ratio_depth),
            60.0 / (3 ** ratio_depth), specified=False, nested=True)
        domains.append(domain)
    exp = _hierarchy_experiment(tuple(domains))
    artifacts = tuple(_artifact(grid_id) for grid_id in range(1, max_dom + 1))

    def fake_root_export(_prepared, _static, _geometry, output, **_kwargs):
        output.mkdir()
        (output / "wrfinput_d01").write_bytes(b"root-input")
        (output / "wrfbdy_d01").write_bytes(b"root-boundary")
        (output / "manifest.json").write_text("{}", encoding="utf-8")
        return {
            "files": {
                "wrfinput_d01": {"bytes": 10, "sha256": "root-input"},
                "wrfbdy_d01": {"bytes": 13, "sha256": "root-boundary"},
            },
            "boundary_record_count": 1,
            "boundary_times": ["1999-05-03_12:00:00"],
            "next_boundary_times": ["1999-05-03_18:00:00"],
            "forcing_hours": [0, 6],
        }

    def fake_context(artifact, domain, _exp, _expected_grid, _valid_time):
        return _hierarchy_context(
            domain, _WSM6_STOCK_PHYSICS, f"static-d{domain.grid_id:02d}")

    written_updates = {}

    def fake_write(path, _contract, _dimensions, updates, _fields, _stamp,
                   engine=None):
        path.write_bytes(b"child-input")
        written_updates[path.name] = updates
        # One writer decision governs the whole hierarchy, so every child
        # export is handed the engine the root resolved.
        assert engine == "rust"

    monkeypatch.setattr(wrf_direct, "export_prepared_wrf", fake_root_export)
    monkeypatch.setattr(
        wrf_direct, "grids_from_projection_config",
        lambda _exp: tuple(object() for _ in range(max_dom)))
    monkeypatch.setattr(wrf_direct, "_prepared_domain_context", fake_context)
    monkeypatch.setattr(wrf_direct.np, "load", lambda *_a, **_k: nullcontext({}))
    monkeypatch.setattr(wrf_direct, "_wrfinput_fields", lambda *_a, **_k: {})
    monkeypatch.setattr(wrf_direct, "_write_wrfinput", fake_write)
    monkeypatch.setattr(
        wrf_direct, "_validate_file",
        lambda path, *_a, **_k: {"bytes": path.stat().st_size,
                                 "sha256": path.name})
    monkeypatch.setattr(
        wrf_direct, "_sha256", lambda path: f"sha-{Path(path).name}")

    output = tmp_path / "wrf-ready"
    manifest = export_prepared_wrf_hierarchy(
        exp, artifacts, output, boundary_interval_seconds=21600)
    expected_files = [
        "manifest.json", "wrfbdy_d01",
        *(f"wrfinput_d{grid_id:02d}"
          for grid_id in range(1, max_dom + 1)),
    ]
    assert sorted(path.name for path in output.iterdir()) == sorted(
        expected_files)
    assert manifest["status"] == "READY"
    assert [item["grid_id"] for item in manifest["hierarchy"]] == list(
        range(1, max_dom + 1))
    assert set(name for name in manifest["files"] if name.startswith(
        "wrfbdy")) == {"wrfbdy_d01"}
    if max_dom > 1:
        assert written_updates["wrfinput_d02"]["GRID_ID"] == 2
        assert written_updates["wrfinput_d02"]["PARENT_GRID_RATIO"] == 3
    if max_dom >= 6:
        assert written_updates["wrfinput_d04"]["PARENT_ID"] == 3
        assert written_updates["wrfinput_d05"]["PARENT_ID"] == 1
        assert written_updates["wrfinput_d06"]["PARENT_ID"] == 2
    assert not list(tmp_path.glob("wrf-ready.tmp-*"))


class _UVOnlyCache:
    """The only two arrays ``_wrfinput_fields`` reads before it rebuilds
    the projection, so the geometry gate is reached on a cache that
    carries nothing else."""

    def __init__(self, nz: int, ny: int, nx: int):
        self._arrays = {
            "state/u": np.zeros((nz, ny, nx + 1)),
            "state/v": np.zeros((nz, ny + 1, nx)),
        }

    def array(self, name):
        return self._arrays[name]


def test_wrfinput_fields_refuse_a_receipt_that_lost_the_nest_anchor():
    """Negative control on the derived-vs-inherited geometry gate.

    A geometry receipt written before ``known_x``/``known_y`` were
    carried is exactly a nest receipt with the anchor removed: the
    exporter rebuilt the projection on ``ProjectedGrid``'s centred WPS
    default and emitted XLAT/XLONG for a footprint ~300 km south-west of
    the nest, beside MAPFAC/F/SINALPHA read from the static cache built
    on the real grid.  Feed the exporter that receipt and require a
    refusal that names both footprints -- the receipt's own
    ``lat_range``/``lon_range``, which had no consumer at all, are what
    it is measured against.
    """
    root = LambertGrid(
        ref_lat=35.5, ref_lon=-98.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.5, dx=9_000.0, dy=9_000.0, e_we=201, e_sn=201)
    nest = root.nest(50, 50, 3, 201, 201)
    cfg = SimpleNamespace(nx=200, ny=200, nz=4, dx=3_000.0, dy=3_000.0)
    geometry = native_geometry_contract(nest, cfg)
    cache = _UVOnlyCache(cfg.nz, cfg.ny, cfg.nx)
    valid_time = datetime(2026, 7, 18)

    stripped = {name: value for name, value in geometry.items()
                if name not in ("known_x", "known_y")}
    footprint_refusal = "does not reproduce the footprint"
    with pytest.raises(ValueError, match=footprint_refusal) as refusal:
        wrf_direct._wrfinput_fields(
            cache, {}, stripped, valid_time, p_top=5000.0)
    message = str(refusal.value)
    assert "lat_range" in message
    # Both footprints, by value, so the reader can see the displacement.
    assert repr(geometry["lat_range"][0]) in message

    # A receipt with no recorded footprint at all cannot be checked, and
    # is refused rather than exported unmeasured.
    unmeasured = {name: value for name, value in geometry.items()
                  if name != "lat_range"}
    with pytest.raises(ValueError, match="carries no lat_range"):
        wrf_direct._wrfinput_fields(
            cache, {}, unmeasured, valid_time, p_top=5000.0)

    # The whole receipt rebuilds the nest where the nest actually is, so
    # the gate passes and the export continues into the prepared arrays.
    with pytest.raises(KeyError, match="state/qv"):
        wrf_direct._wrfinput_fields(
            cache, {}, geometry, valid_time, p_top=5000.0)


def test_the_root_export_stages_where_a_door_measures_it(tmp_path, monkeypatch):
    """A door refuses a deep root on the unchanged-WRF export's staging.

    That path is ``wrf-native-input.tmp-<pid>/.root-export.tmp-<pid>/
    manifest.json`` inside the door's own staging, and it is only right
    if the hierarchy exporter really hands d01's exporter a folder inside
    its staging and d01's exporter stages beside it: the export is run
    here, from a door's staging and at the widest process id, up to that
    hand-off.
    """
    from woof.fetch_guard import WINDOWS_WIDEST_PID
    from woof.native_domain_artifacts import (
        _atomic_staging_sibling, hierarchy_bundle_write_paths)

    class HandedOver(Exception):
        pass

    handed = []

    def root_export(_prepared, _static, _geometry, output, **_kwargs):
        handed.append(Path(output))
        raise HandedOver()

    def run(nx, ny, dx, dt, *, specified, nested):
        return SimpleNamespace(nx=nx, ny=ny, nz=8, dx=dx, dy=dx, dt=dt,
                               specified=specified, nested=nested,
                               mp_physics=6)

    d01 = _domain(1, 0, specified=True, nested=False)
    d01.run = run(100, 80, 12000.0, 60.0, specified=True, nested=False)
    d02 = _domain(2, 1, specified=False, nested=True)
    d02.run = run(60, 60, 4000.0, 20.0, specified=False, nested=True)
    exp = _hierarchy_experiment((d01, d02))
    monkeypatch.setattr(wrf_direct, "export_prepared_wrf", root_export)
    monkeypatch.setattr(wrf_direct, "grids_from_projection_config",
                        lambda _exp: (object(), object()))
    monkeypatch.setattr(os, "getpid", lambda: WINDOWS_WIDEST_PID)
    output_root = tmp_path / "prepared"
    door_staging = _atomic_staging_sibling(output_root, nonce="f" * 10)

    with pytest.raises(HandedOver):
        export_prepared_wrf_hierarchy(
            exp, (_artifact(1), _artifact(2)),
            door_staging / "wrf-native-input",
            boundary_interval_seconds=21600)

    staged = (wrf_direct.export_staging_path(handed[0]) / "manifest.json")
    assert staged in hierarchy_bundle_write_paths(output_root)
    assert len(str(staged)) == max(
        len(str(path)) for path in hierarchy_bundle_write_paths(output_root))


def test_the_tree_export_states_each_domains_recorded_physics(
        tmp_path, monkeypatch):
    """Every wrfinput of a tree states its OWN domain's recorded physics.

    The tree export handed its nests no selection, so each child wrfinput
    stated the frozen contract's MP_PHYSICS = 6, RA_LW_PHYSICS = 0,
    RA_SW_PHYSICS = 1 and CU_PHYSICS = 0 beside Thompson arrays, and its
    root went through the profile-free single-domain export, which did
    the same.  Here the root records cumulus on and the nest records it
    off, so a nest stating its parent's physics would show it too.
    """

    thompson = {**_WSM6_STOCK_PHYSICS, "mp_physics": 8, "ra_physics": 0,
                "ra_lw_physics": 4, "ra_sw_physics": 4}

    def run(nx, ny, dx, dt, *, specified, nested, cu_physics):
        return SimpleNamespace(
            nx=nx, ny=ny, nz=8, dx=dx, dy=dx, dt=dt, specified=specified,
            nested=nested, **{**thompson, "cu_physics": cu_physics})

    d01 = _domain(1, 0, specified=True, nested=False)
    d01.run = run(100, 80, 12000.0, 60.0, specified=True, nested=False,
                  cu_physics=1)
    d02 = _domain(2, 1, specified=False, nested=True)
    d02.run = run(60, 60, 4000.0, 20.0, specified=False, nested=True,
                  cu_physics=0)
    exp = _hierarchy_experiment((d01, d02))

    root_kwargs = {}

    def fake_root_export(_prepared, _static, _geometry, output, **kwargs):
        # The root takes the profile-free export, whose own stamping
        # test_a_profile_free_export_states_the_physics_its_cache_was_
        # prepared_with measures on a real cache.
        root_kwargs.update(kwargs)
        output.mkdir()
        for name in ("wrfinput_d01", "wrfbdy_d01"):
            (output / name).write_bytes(name.encode())
        return {"files": {name: {"bytes": 1, "sha256": name}
                          for name in ("wrfinput_d01", "wrfbdy_d01")},
                "boundary_record_count": 1,
                "boundary_times": ["1999-05-03_12:00:00"],
                "next_boundary_times": ["1999-05-03_18:00:00"],
                "forcing_hours": [0, 6]}

    def fake_context(artifact, domain, _exp, _expected_grid, _valid_time):
        return _hierarchy_context(domain, thompson, "static")

    written = {}
    validated = {}

    def fake_write(path, _contract, _dimensions, updates, _fields, _stamp,
                   engine=None):
        path.write_bytes(b"child-input")
        written[path.name] = updates

    def fake_validate(path, *_args, expected_global_attributes=None, **_kw):
        validated[path.name] = expected_global_attributes
        return {"bytes": path.stat().st_size, "sha256": path.name}

    monkeypatch.setattr(wrf_direct, "export_prepared_wrf", fake_root_export)
    monkeypatch.setattr(wrf_direct, "grids_from_projection_config",
                        lambda _exp: (object(), object()))
    monkeypatch.setattr(wrf_direct, "_prepared_domain_context", fake_context)
    monkeypatch.setattr(wrf_direct.np, "load", lambda *_a, **_k: nullcontext({}))
    monkeypatch.setattr(wrf_direct, "_wrfinput_fields", lambda *_a, **_k: {})
    monkeypatch.setattr(wrf_direct, "_write_wrfinput", fake_write)
    monkeypatch.setattr(wrf_direct, "_validate_file", fake_validate)
    monkeypatch.setattr(
        wrf_direct, "_sha256", lambda path: f"sha-{Path(path).name}")

    export_prepared_wrf_hierarchy(
        exp, (_artifact(1), _artifact(2)), tmp_path / "wrf-ready",
        boundary_interval_seconds=21600)

    # The root is exported with no selection, so its file states the
    # cache's own recorded physics (the real-cache test above).
    assert "physics_profile" not in root_kwargs
    assert "experiment_config_suite" not in root_kwargs
    expected = {"MP_PHYSICS": 8, "RA_LW_PHYSICS": 4, "RA_SW_PHYSICS": 4,
                "SF_SFCLAY_PHYSICS": 91, "SF_SURFACE_PHYSICS": 2,
                "BL_PBL_PHYSICS": 1, "CU_PHYSICS": 0}
    stated = {key: written["wrfinput_d02"].get(key) for key in expected}
    assert stated == expected
    # ...and the file check holds the written file to the same values.
    assert {key: validated["wrfinput_d02"].get(key)
            for key in expected} == expected


class _VegetationCache:
    """The two PreparedCache members the start-vegetation export reads."""

    def __init__(self, arrays):
        self._arrays = {name: None for name in arrays}
        self._values = arrays

    def array(self, name):
        return self._values[name]


def _vegetation_static(ny=6, nx=7):
    rng = np.random.default_rng(20261007)
    landmask = np.ones((ny, nx))
    landmask[0, :] = 0.0
    return {
        "GREENFRAC": rng.uniform(0.05, 0.95, (12, ny, nx)),
        "LAI12M": rng.uniform(0.1, 6.0, (12, ny, nx)),
        "ALBEDO12M": rng.uniform(8.0, 25.0, (12, ny, nx)),
        "LANDMASK": landmask,
    }


def test_the_export_writes_the_vegetation_the_run_starts_with():
    """VEGFRA, LAI and ALBBCK are the forecast's own start values.

    The export wrote the calendar month's raw GREENFRAC, LAI12M and
    ALBEDO12M while the prepared forecast starts from the analyzed VEGFRA
    (or GREENFRAC on the date) and date-interpolated LAI and albedo, so
    stock WRF ran another land surface: 12 % RMS VEGFRA and 89 W m-2 RMS
    latent heat after ten minutes on a HRRR-start crop.
    """
    from woof.core.landuse import monthly_background_albedo, surface_leaf_area
    from woof.static.build import monthly_interp_to_date

    static = _vegetation_static()
    when = datetime(2024, 5, 21, 18)
    ruc = {"sf_surface_physics": 3, "rdlai2d": True}
    analyzed = np.full(static["LANDMASK"].shape, 42.5)
    got = wrf_direct._run_start_vegetation(
        _VegetationCache({"met/VEGFRA": analyzed}), static, when, ruc)
    np.testing.assert_array_equal(got["VEGFRA"], analyzed)
    np.testing.assert_array_equal(
        got["LAI"], surface_leaf_area(SimpleNamespace(**ruc),
                                      static["LAI12M"], when))
    np.testing.assert_array_equal(
        got["ALBBCK"], monthly_background_albedo(
            static["ALBEDO12M"], static["LANDMASK"], when))
    assert np.all(got["ALBBCK"][0] == np.float32(0.08))

    climatology = wrf_direct._run_start_vegetation(
        _VegetationCache({}), static, when, ruc)
    np.testing.assert_allclose(
        climatology["VEGFRA"],
        100.0 * monthly_interp_to_date(static["GREENFRAC"], when))
    # Not the calendar month's raw value the export used to write.
    assert not np.allclose(climatology["VEGFRA"],
                           100.0 * static["GREENFRAC"][4])
    assert not np.allclose(climatology["LAI"], static["LAI12M"][4])


def test_stock_export_snow_ice_th2_and_attributes_are_read_back(tmp_path):
    def surfaces(surface, static):
        surface["SNOW"][0, :4] = [0.0, 9.999, 10.0, 100.0]
        surface["SEAICE"][1, :2] = [0.0, 1.0]
        surface["LANDMASK"][1, :2] = [0.0, 1.0]
        surface["XLAND"][1, :2] = [2.0, 1.0]
        static["LU_INDEX"][1, :2] = 21
        static["SCT_DOM"][1, :2] = 14
    output, cfg, _ = _export_a_real_prepared_cache(tmp_path, 8,
        surface_override=surfaces, diff_opt=1, diff_6th_slopeopt=0)
    with netCDF4.Dataset(output / "wrfinput_d01") as ds:
        np.testing.assert_array_equal(ds["SNOWC"][0,0,:4], [0,0,1,1])
        assert ds["IVGTYP"][0,1,1] == 15
        assert ds["ISLTYP"][0,1,1] == 16
        assert ds["SNOALB"][0,1,1] == 0.75
        assert ds["IVGTYP"][0,1,0] == 17
        pressure = ds["P"][0,0] + ds["PB"][0,0]
        from woof.wrf_direct import _RCP, _P0
        expected = ds["T2"][0] * (_P0 / pressure) ** _RCP
        np.testing.assert_allclose(ds["TH2"][0], expected, rtol=2e-7)
        assert ds.SURFACE_INPUT_SOURCE == 3
        assert ds.DIFF_OPT == 1
        assert ds.DIFF_6TH_SLOPEOPT == 0
        assert ds.GHG_INPUT == 0


@pytest.mark.parametrize('topo,gwd', [(0,0),(1,0),(2,1),(0,3),(1,3)])
def test_export_preserves_selected_terrain_drag_statistics(tmp_path, topo, gwd):
    from woof.static.orographic import required_static_fields
    cfg = {'topo_wind': topo, 'gwd_opt': gwd}
    names = required_static_fields(topo,gwd)
    static = {name: np.arange(24,dtype=np.float32).reshape(4,6)+i*.125
              for i,name in enumerate(names)}
    fields = wrf_direct._terrain_drag_export_fields(static,cfg,(4,6))
    contract = _physics_contract_bundle(_load_contract(),6,cfg)['wrfinput']
    path = tmp_path/'wrfinput'
    _write_wrfinput(path,contract,_dimensions(contract,nx=6,ny=4,nz=49),{},
                    fields,'2026-10-28_03:00:00',engine='python')
    with netCDF4.Dataset(path) as ds:
        for name in names:
            assert ds[name].dimensions == ('Time','south_north','west_east')
            np.testing.assert_array_equal(ds[name][0].data.view('u4'), static[name].view('u4'))
        if 'VAR_SSO' in names:
            assert ds['GOT_VAR_SSO'][0] == 1


def test_export_refuses_incomplete_terrain_drag_statistics():
    cfg={'gwd_opt':3}
    with pytest.raises(ValueError,match='missing required terrain-drag statistic VARLS'):
        wrf_direct._terrain_drag_export_fields({},cfg,(4,6))
    with pytest.raises(ValueError,match='invalid terrain-drag statistic VARLS'):
        wrf_direct._terrain_drag_export_fields({'VARLS':np.ones((6,4))},cfg,(4,6))


def test_export_preserves_lake_bathymetry_and_presence(tmp_path):
    depth = np.arange(24, dtype=np.float32).reshape(4,6) + 10
    cfg = {"sf_lake_physics": 1, "use_lakedepth": 1}
    mask = np.ones((4,6), dtype=np.float32)
    fields = wrf_direct._lake_export_fields({"LAKE_DEPTH": depth}, cfg, mask)
    contract = _physics_contract_bundle(_load_contract(), 6, cfg)["wrfinput"]
    path = tmp_path / "wrfinput"
    _write_wrfinput(path, contract, _dimensions(contract,nx=6,ny=4,nz=49), {},
                   fields, "2026-10-28_03:00:00", engine="python")
    with netCDF4.Dataset(path) as ds:
        np.testing.assert_array_equal(ds["LAKE_DEPTH"][0].data.view("u4"), depth.view("u4"))
        assert ds["LAKE_DEPTH_FLAG"][0] == 1
    with pytest.raises(ValueError, match="missing required lake bathymetry"):
        wrf_direct._lake_export_fields({}, cfg, mask)
    assert wrf_direct._lake_export_fields({}, cfg, np.zeros_like(mask)) == {}
