from __future__ import annotations

from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import woof.mapped_direct as mapped_direct
from woof.ingest.lateral_bc import start_last_forcing_order
from woof.ingest.source_coverage import PREPARATION_REFUSAL_EXIT_CODE
from woof.moisture_floor_receipt import (
    MOISTURE_FLOOR_BY_DOMAIN_KEY,
    MOISTURE_FLOOR_KEY,
    MOISTURE_FLOOR_SCHEMA,
    moisture_floor_proof_entries,
)


_START = datetime(2026, 7, 20)
_DIGEST = "a" * 64


def test_provenance_evidence_name_is_short_stable_and_role_bound():
    role = "twentycrv3_in_band_surface_provenance"
    name = mapped_direct._provenance_evidence_name(role, ".json")

    assert name == "provenance-b3611ca2660a0367.json"
    assert len(name) == 32
    assert mapped_direct._provenance_evidence_name(role, ".json") == name
    assert mapped_direct._provenance_evidence_name(role + "-other", ".json") \
        != name


def _target_mapping(**updates):
    target = {
        "max_dom": 4,
        "target_vertical_levels": 49,
        "require_lateral_boundaries": True,
        "boundary_interval_seconds": 3600,
    }
    target.update(updates)
    return {"format": "netcdf", "target": target}


def _experiment(domain_count: int, *, nz: int = 49, run_seconds: int = 3600,
                eta: bool = True):
    domains = []
    for index in range(domain_count):
        run = SimpleNamespace(
            nz=nz,
            # Mass-grid extent of the _Grid fakes below.  A real domain run
            # config carries these; the prebuilt-static loader is shape
            # checked against them.
            nx=2,
            ny=2,
            hybrid_opt=2,
            etac=0.2,
            # The analytic base state's t00: the vertical-coordinate
            # survey turns a domain's terrain into ITS base surface
            # pressure, and that depends on the domain's own base_temp.
            base_temp=290.0,
            spec_bdy_width=5,
            spec_zone=1,
            relax_zone=4,
            # The land-surface selector the soil seam routes on.  Noah here,
            # which is the geometry this adapter's declarative soil contract
            # has a target for; the seam refuses the others by name rather
            # than defaulting.
            sf_surface_physics=2,
            # soil_layer_count(cfg) reads the requested count and checks it
            # against the scheme's defined geometries; Noah defines exactly
            # four.  The parametric-soil landing made the count an explicit
            # config read and this fake never carried it.
            num_soil_layers=4,
        )
        domains.append(SimpleNamespace(
            grid_id=index + 1, parent_id=0 if index == 0 else index,
            start_time=None, run=run))
    exp = SimpleNamespace(
        domains=tuple(domains),
        root=domains[0],
        start_time=_START,
        run_seconds=run_seconds,
        vertical=SimpleNamespace(
            # ``eta=False`` is the imported-WRF-config shape: a level
            # COUNT with no explicit ladder, because stock WRF derives
            # the ladder in real.exe and import-namelist has nothing to
            # translate.
            eta_levels=(
                tuple(np.linspace(1.0, 0.0, nz + 1)) if eta else ()),
            p_top=10_000.0,
            # The two hybrid selectors a real VerticalConfig always
            # carries.  The route reads them before it builds any
            # coordinate, to derive the one this run's terrain can order.
            hybrid_opt=2,
            etac=0.2,
        ),
    )
    exp.dt_exact = lambda grid_id: Fraction(60, 3 ** (grid_id - 1))
    exp.domain_start_offset_exact = lambda _grid_id: Fraction(0)
    exp.domain_start_time = lambda _grid_id: exp.start_time
    return exp


@pytest.mark.parametrize(
    ("mapping", "exp", "interval", "hierarchy", "message"),
    [
        (
            _target_mapping(max_dom=1),
            _experiment(2),
            3600,
            True,
            "max_dom",
        ),
        (
            # A count with NO explicit ladder is what an imported WRF
            # config carries; matched or not, a ladder cannot be
            # derived from it, and the refusal names both counts and
            # both reconciling doors.
            _target_mapping(target_vertical_levels=48),
            _experiment(1, eta=False),
            3600,
            False,
            "vertical ladder is missing",
        ),
        (
            _target_mapping(boundary_interval_seconds=10_800),
            _experiment(1),
            3600,
            False,
            "boundary.*interval|cadence",
        ),
    ],
)
def test_mapped_target_contract_fails_closed(
        mapping, exp, interval, hierarchy, message):
    with pytest.raises(ValueError, match=message):
        mapped_direct._validate_target_contract(
            mapping, exp, interval, hierarchy=hierarchy,
        )


def test_mapped_target_contract_takes_whole_multiples_only_when_declared():
    exp = _experiment(1)
    with pytest.raises(ValueError, match="differs from target contract 3600"):
        mapped_direct._validate_target_contract(
            _target_mapping(), exp, 10_800, hierarchy=False)
    multiples = _target_mapping(accept_boundary_interval_multiples=True)
    receipt = mapped_direct._validate_target_contract(
        multiples, exp, 10_800, hierarchy=False)
    assert receipt["boundary_interval_seconds"] == 10_800
    with pytest.raises(ValueError, match="not a whole multiple of the 3600"):
        mapped_direct._validate_target_contract(
            multiples, exp, 5400, hierarchy=False)


def test_mapped_target_contract_returns_bound_receipt():
    receipt = mapped_direct._validate_target_contract(
        _target_mapping(), _experiment(2), 3600, hierarchy=True,
    )
    assert receipt["domain_count"] == 2
    assert receipt["mapping_max_dom"] == 4
    assert receipt["target_vertical_levels"] == 49
    assert receipt["mapping_reference_vertical_levels"] == 49
    assert receipt["vertical_levels_adopted_from"] \
        == "mapping-reference-count"
    assert receipt["boundary_interval_seconds"] == 3600
    assert receipt["require_lateral_boundaries"] is True


def test_mapped_target_contract_adopts_an_explicit_experiment_ladder():
    """A config CARRYING a valid ladder runs at its own level count.

    This is the circle-breaker of UX finding N6: the count-equality
    refusal named no breakage the ladder validation does not already
    prevent -- every downstream array is allocated from the
    experiment, and the interpolation target IS the experiment's
    ladder -- so a valid explicit ladder is adopted and the receipt
    records the adoption beside the mapping's reference count.
    """

    receipt = mapped_direct._validate_target_contract(
        _target_mapping(), _experiment(1, nz=44), 3600, hierarchy=False,
    )
    assert receipt["target_vertical_levels"] == 44
    assert receipt["mapping_reference_vertical_levels"] == 49
    assert receipt["vertical_levels_adopted_from"] == "experiment-eta-ladder"


def test_mapped_target_contract_names_both_counts_and_doors_without_ladder():
    """No ladder: one refusal, both counts, both reconciling doors."""

    from woof.ingest.source_coverage import VerticalLadderRefusal

    with pytest.raises(VerticalLadderRefusal) as caught:
        mapped_direct._validate_target_contract(
            _target_mapping(), _experiment(2, nz=44, eta=False), 3600,
            hierarchy=True, experiment_config=Path("mycase.toml"),
        )
    message = str(caught.value)
    assert "nz=44" in message and "e_vert=45" in message
    assert "49" in message
    assert "mycase.toml" in message
    remedy = caught.value.remedy
    assert "eta_levels" in remedy
    assert "woof domain" in remedy
    # 45 interfaces for the user's own 44 levels: the exact edit that
    # terminates, which is what the old shape-(0,) circle never named.
    assert "45 interfaces" in remedy


def test_mapped_target_contract_refuses_a_malformed_ladder_as_a_refusal():
    from woof.ingest.source_coverage import VerticalLadderRefusal

    exp = _experiment(1, nz=44)
    exp.vertical.eta_levels = exp.vertical.eta_levels[:-1] + (0.5,)
    with pytest.raises(VerticalLadderRefusal, match="1.0.*0.0|decreasing"):
        mapped_direct._validate_target_contract(
            _target_mapping(), exp, 3600, hierarchy=False,
        )


def test_mapped_target_contract_accepts_five_minute_hierarchy_and_names_real_refusal():
    receipt = mapped_direct._validate_target_contract(
        _target_mapping(boundary_interval_seconds=300),
        _experiment(2), 300, hierarchy=True)
    assert receipt["boundary_interval_seconds"] == 300

    with pytest.raises(
            ValueError,
            match=r"310 s.*whole number of root-domain steps.*31/6"):
        mapped_direct._validate_target_contract(
            _target_mapping(boundary_interval_seconds=310),
            _experiment(2), 310, hierarchy=True)


class _Grid:
    def __init__(self, domain_id: int):
        self.domain_id = domain_id

    def mapfac_m(self):
        return np.ones((2, 2), dtype=np.float64)

    def mapfac_u(self):
        return np.ones((2, 3), dtype=np.float64)

    def mapfac_v(self):
        return np.ones((3, 2), dtype=np.float64)

    def coriolis_m(self):
        plane = np.ones((2, 2), dtype=np.float64)
        return plane, plane

    def rotation_m(self):
        plane = np.ones((2, 2), dtype=np.float64)
        return np.zeros_like(plane), plane


class _State:
    def __init__(self):
        self.lateral_boundaries = None

    def set_map_coriolis(self, *_args, **_kwargs):
        pass


class _Selection:
    resolution_tokens = ("default",)

    def __init__(self, geog_root: Path):
        self._geog_root = geog_root

    @property
    def root(self):
        # A real GeogSelection carries the tree it resolved against, and
        # every builder passes it back in beside the selection.
        return self._geog_root

    def path(self, field):
        return self._geog_root / f"{field}.bin"

    def landuse_global_attrs(self):
        # The route reads ISLAKE from here to tell an inland body from the
        # ocean before assembling water temperature; a real GeogSelection
        # reads it out of the GEOG index.
        return {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}


def _write(path: Path, contents: bytes = b"test") -> Path:
    path.write_bytes(contents)
    return path


def _install_prepare_fakes(
        monkeypatch, tmp_path, *, domain_count: int, backend: str,
        cadence: int = 3600, run_seconds: int | None = None,
        mapping_updates=None, nz: int = 49, eta: bool = True,
        water_policy="era5_class_coherent"):
    run_seconds = cadence if run_seconds is None else run_seconds
    if domain_count > 1:
        # These doubles stand in for the one-shot hierarchy call, which is
        # the tree route with chaining off (and a CUDA tree's route); the
        # chained tree is covered by tests/test_stream_tree_producer.py, by
        # test_a_two_domain_mapped_tree_chains_through_prepare_mapped_wrf
        # below and by the real-data byte-identity proof.
        monkeypatch.setenv("WOOF_CHAINED_PREP", "0")
    exp = _experiment(domain_count, nz=nz, run_seconds=run_seconds, eta=eta)
    # Preparation validates actual configured physics before decoding.
    from dataclasses import asdict
    from woof.config import RunConfig

    defaults = asdict(RunConfig(
        nx=2, ny=2, nz=nz, dx=3000, dy=3000, ztop=20000,
        dt=60, run_seconds=run_seconds))
    for domain in exp.domains:
        domain.run = SimpleNamespace(**(defaults | vars(domain.run)))
    mapping = _target_mapping(**(mapping_updates or {}))
    grids = tuple(_Grid(index + 1) for index in range(domain_count))
    snapshots = tuple(
        SimpleNamespace(
            valid_time=_START + timedelta(seconds=index * cadence),
            levels_hpa=np.array([1000.0, 500.0, 100.0]),
            fields={
                "SOURCE_OROGRAPHY": np.zeros((2, 2), dtype=np.float64),
                "PRES": np.ones((3, 2, 2), dtype=np.float64),
            },
        )
        for index in range(2)
    )
    bundle = SimpleNamespace(
        regular_snapshots=lambda: snapshots,
        # The real bundle owns the engine's compose scratch while the
        # route streams valid times out of it, and the route releases
        # it when the tree is published; a stub without `close` would
        # let that release quietly stop happening.
        close=lambda: None,
        mapping_sha256="1" * 64,
        composition_sha256="2" * 64,
        input_manifest_sha256="3" * 64,
        # Filled in below from the engine the route will actually
        # resolve.  Since the compose port this route decodes in
        # process, so its decoder inventory is ONE row -- the engine --
        # and `prepare_mapped_wrf` refuses a bundle whose decoders are
        # not the ones it asked for.  A fixture that reported an empty
        # inventory would be asserting the pre-port route.
        decoder_paths={},
        decoder_sha256={},
        soil_layer_contract={"fixture": "declarative-soil-contract"},
        contributing_sources=(),
    )
    composition_receipt = {"receipt_content_sha256": "4" * 64}
    states = tuple(_State() for _ in snapshots)
    results = tuple(SimpleNamespace(state=state) for state in states)
    mets = tuple(
        SimpleNamespace(
            fields={
                "frame": index,
                "SOURCE_OROGRAPHY": np.full((2, 2), 123.0 + index),
            },
            # The finished field the assembly hands back on this route.
            # The soil assertion below proves THIS array is what the
            # router consumed, which is what stops a receipt from
            # describing a field nobody used.
            water_temperature=np.full((2, 2), 284.0 + index),
        )
        for index in range(len(snapshots))
    )
    boundaries = object()
    soil = object()
    calls = {
        "build_static": 0,
        "interpolate": 0,
        "initialize": 0,
        "hierarchy": [],
        "single_export": [],
    }

    geog_root = tmp_path / "geog"
    geog_root.mkdir()
    files = {
        name: _write(tmp_path / name)
        for name in (
            "composition.json", "mapping.json", "primary.bin",
            "supplement.bin", "provenance.md", "manifest.json",
            "namelist.wps", "experiment.toml",
        )
    }
    files["mapping.json"].write_text(json.dumps(mapping), encoding="utf-8")
    files["composition.json"].write_text(json.dumps({
        "supplements": {"terrain_height": {"provenance_role": "terrain"}},
    }), encoding="utf-8")
    bundle.mapping_sha256 = hashlib.sha256(
        files["mapping.json"].read_bytes()
    ).hexdigest()
    bundle.composition_sha256 = hashlib.sha256(
        files["composition.json"].read_bytes()
    ).hexdigest()
    bundle.input_manifest_sha256 = hashlib.sha256(
        files["manifest.json"].read_bytes()
    ).hexdigest()
    bundle.mapping_path = files["mapping.json"].resolve()
    bundle.composition_path = files["composition.json"].resolve()
    bundle.input_manifest_path = files["manifest.json"].resolve()
    bundle.terrain_data_paths = (files["supplement.bin"].resolve(),)
    bundle.terrain_provenance_path = files["provenance.md"].resolve()
    bundle.terrain_provenance_sha256 = hashlib.sha256(
        files["provenance.md"].read_bytes()
    ).hexdigest()

    # The staged engine this route resolves, faked at the ladder rather
    # than by editing the route: `prepare_mapped_wrf` resolves it before
    # any output directory exists, and the bundle it then accepts must
    # name that exact binary.
    from woof import mapped_engine_bridge

    engine_binary = _write(tmp_path / "gpuwm_mapped_engine.exe").resolve()
    monkeypatch.setattr(
        mapped_engine_bridge, "require_engine", lambda: engine_binary)
    bundle.decoder_paths = {mapped_engine_bridge.ENGINE_NAME: engine_binary}
    bundle.decoder_sha256 = {
        mapped_engine_bridge.ENGINE_NAME: hashlib.sha256(
            engine_binary.read_bytes()).hexdigest(),
    }

    monkeypatch.setattr(
        mapped_direct,
        "load_mapping",
        lambda _path, **_kwargs: mapping,
    )
    files["experiment.toml"].write_text("[experiment]\n", encoding="utf-8")
    monkeypatch.setattr(mapped_direct, "load_experiment", lambda _path: exp)
    monkeypatch.setattr(
        mapped_direct, "validate_native_lambert_contracts",
        lambda *_args, **_kwargs: grids,
    )
    if hasattr(mapped_direct, "validate_native_lambert_contract"):
        monkeypatch.setattr(
            mapped_direct, "validate_native_lambert_contract",
            lambda *_args, **_kwargs: grids[0],
        )
    monkeypatch.setattr(
        mapped_direct, "decode_composed_source",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        mapped_direct, "mapped_composition_receipt",
        lambda _bundle: composition_receipt,
    )
    monkeypatch.setattr(
        mapped_direct, "validate_explicit_eta_grid",
        lambda *_args, **_kwargs: None,
    )

    class FakeGeogSelection:
        @staticmethod
        def from_case_data(_case, _domain_id):
            return _Selection(geog_root)

    monkeypatch.setattr(mapped_direct, "GeogSelection", FakeGeogSelection)

    static = {
        "HGT_M": np.zeros((2, 2), dtype=np.float64),
        "LU_INDEX": np.ones((2, 2), dtype=np.int32),
        "SCT_DOM": np.ones((2, 2), dtype=np.int32),
        "TMN": np.full((2, 2), 280.0, dtype=np.float64),
        "LANDMASK": np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
    }

    def build_static(*_args, **_kwargs):
        calls["build_static"] += 1
        return static

    monkeypatch.setattr(mapped_direct, "build_static", build_static)
    # The vertical-coordinate survey reads the terrain of every domain
    # this run can touch before the first coordinate is built, through
    # the geography ladder the route itself uses.  This fixture's
    # geography is synthetic, so the survey is pointed at the same
    # synthetic terrain the rest of the route gets.
    import woof.hrrr_native_static as _native_static
    import woof.static.build as _static_build

    monkeypatch.setattr(
        _native_static, "verified_static_catalog",
        lambda *_args, **_kwargs: (SimpleNamespace(), {}))
    monkeypatch.setattr(
        _static_build, "geog_selection_from_catalog",
        lambda *_args, **_kwargs: _Selection(geog_root))
    monkeypatch.setattr(
        _static_build, "build_terrain",
        lambda *_args, **_kwargs: static["HGT_M"])
    # With a [static.highres] overlay the survey reads the complete field
    # set instead, because the overlay replaces HGT_M out of it; both the
    # build and the overlay are answered with this fixture's statics.
    import woof.static.highres_production as _highres_production

    monkeypatch.setattr(
        _static_build, "build_static_for_domain",
        lambda *_args, **_kwargs: static)
    monkeypatch.setattr(
        _highres_production, "apply_highres_statics",
        lambda fields, *_args, **_kwargs: (fields, {}))
    preprocess = SimpleNamespace(receipt=lambda: {"backend": backend})
    monkeypatch.setattr(
        mapped_direct, "resolve_preprocess_backend",
        lambda *_args, **_kwargs: preprocess,
    )

    def interpolate(source, _grid, **kwargs):
        calls["interpolate"] += 1
        # Metgrid classifies masked-field target cells by the model
        # landmask; the mapped lane must declare it like every other lane.
        np.testing.assert_array_equal(
            kwargs["target_landmask"],
            np.asarray(static["LANDMASK"]) >= 0.5,
            err_msg="mapped lane must pass the static landmask as the "
                    "masked-field target classification")
        # ... and it must name its water statics, or the assembly runs
        # with no lake class and a lake joined to the sea by a coarse
        # coastline can share the ocean's provider.  This route reaches
        # every rw-wps composition, 20CRv3 included.
        statics = kwargs["water_temperature_statics"]
        assert statics is not None
        assert statics.route == mapped_direct._WATER_ROUTE
        assert statics.lake_category == 21
        np.testing.assert_array_equal(
            statics.lake,
            np.asarray(static["LU_INDEX"]) == 21,
            err_msg="mapped lane must name lakes from the land-use "
                    "table's own ISLAKE")
        calls.setdefault("water_statics", []).append(statics)
        return mets[snapshots.index(source)]

    def initialize(met, *_args, **_kwargs):
        calls["initialize"] += 1
        calls.setdefault("initialize_operands", []).append(_kwargs)
        return results[mets.index(met)]

    monkeypatch.setattr(
        mapped_direct, "interpolate_era5_to_lambert", interpolate,
    )
    monkeypatch.setattr(mapped_direct, "initialize_real", initialize)
    # The mapped lane no longer hands every state to one builder: it
    # streams, adding each state's perimeter frames as that state is
    # built so the state itself can be released, and each state names its
    # own POSITION.  A single domain builds the START time FIRST, writes
    # it into the prepared head and releases it, then writes each interval
    # as soon as its two times exist; a domain tree still builds the start
    # time LAST and keeps only that one.  This double records the adds,
    # the intervals and the releases, so the tests prove the builder saw
    # every state, in which order, and that arrival order and position
    # were kept distinct.
    frame_calls = {"added": [], "arrival": [], "built": [],
                   "intervals": [], "released": [], "replaced": []}

    class RecordingFrames:
        inventory = ("u", "v", "theta", "phi", "mu")
        # What one written interval holds in host RAM, which a chained
        # head prices the forecast's boundary series from.
        interval_host_bytes = 1 << 20

        def __init__(self, **kwargs):
            frame_calls["kwargs"] = kwargs

        def add_state(self, state, *, index=None):
            frame_calls["added"].append(state)
            frame_calls["arrival"].append(index)

        def replace_start_state(self, state, *, index=0):
            # Recorded in call order beside the intervals, so a test can
            # see the start frame was replaced before interval 0 was built.
            frame_calls["replaced"].append(
                (index, state, len(frame_calls["intervals"])))

        def build(self, actual_times):
            frame_calls["built"].append(tuple(actual_times))
            return boundaries

        def interval(self, index, actual_times):
            frame_calls["intervals"].append(index)
            seconds = [(value - actual_times[0]).total_seconds()
                       for value in actual_times]
            return SimpleNamespace(
                start_seconds=seconds[index],
                end_seconds=seconds[index + 1], fields={})

        def release(self, index):
            frame_calls["released"].append(index)

    monkeypatch.setattr(mapped_direct, "StateBoundaryFrames", RecordingFrames)
    calls["frames"] = frame_calls

    def attach(state, value):
        state.lateral_boundaries = value

    monkeypatch.setattr(mapped_direct, "attach_lateral_boundaries", attach)
    def lake_skin(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError(
            "mapped lane must not run the retired lake skin override")

    if hasattr(mapped_direct, "interpolate_lake_skin_temperature"):
        monkeypatch.setattr(
            mapped_direct, "interpolate_lake_skin_temperature", lake_skin,
        )

    def preprocess_soil(*_args, **kwargs):
        assert kwargs.get("lake_mask") is None
        assert kwargs.get("lake_skin_temperature") is None
        np.testing.assert_array_equal(
            kwargs["landmask"], static["LANDMASK"],
            err_msg="mapped soil must classify by the static landmask")
        assert kwargs["terrain"] is static["HGT_M"]
        np.testing.assert_array_equal(
            kwargs["source_orography"],
            mets[0].fields["SOURCE_OROGRAPHY"],
            err_msg="mapped soil must receive the composition source "
                    "orography for the elevation lapse")
        # RECEIPT IMPLIES CONSUMPTION.  The route pays for an assembly
        # and prints a policy receipt; the field the router integrates
        # has to be that assembly and not the per-cell fuse behind it.
        assert kwargs["water_temperature"] is mets[0].water_temperature
        assert kwargs["route"] == mapped_direct._WATER_ROUTE
        assert kwargs["water_temperature_policy"] == water_policy
        return soil

    monkeypatch.setattr(
        mapped_direct, "preprocess_land_surface_soil", preprocess_soil,
    )
    monkeypatch.setattr(
        mapped_direct, "native_static_export_fields",
        lambda actual, _grid: actual,
    )
    monkeypatch.setattr(
        mapped_direct, "canonical_noah_surface", lambda value: {"soil": value},
    )

    def write_static(path, *_args, **_kwargs):
        path.write_bytes(b"static")
        return {"sha256": hashlib.sha256(b"static").hexdigest()}

    def write_geometry(path, *_args, **_kwargs):
        path.write_text("{}", encoding="utf-8")
        return {"status": "PASS"}

    monkeypatch.setattr(mapped_direct, "write_native_static_cache", write_static)
    monkeypatch.setattr(
        mapped_direct, "write_native_geometry_receipt", write_geometry,
    )
    monkeypatch.setattr(
        mapped_direct, "prepared_cache_identity",
        lambda **kwargs: {"identity": kwargs},
    )

    import woof.ingest.prepared_cache as prepared_cache_module

    class RecordingCacheStream:
        """The prepared-cache writer, recorded: head, segments, seal."""

        def __init__(self, directory, **kwargs):
            self.directory = Path(directory)
            calls["cache_stream"] = {"kwargs": kwargs, "segments": []}

        def move(self, directory):
            self.directory = Path(directory)

        def write_head(self, **kwargs):
            self.directory.mkdir(parents=True)
            calls["cache_stream"]["head"] = kwargs
            return {"identity": {}, "metadata": {}, "arrays": {},
                    "payload_bytes": 0, "lbc": kwargs["lbc"],
                    "setup_core_fingerprint": "0" * 64}

        def write_segment(self, index, interval):
            calls["cache_stream"]["segments"].append(index)
            return {"index": index,
                    "start_seconds": float(interval.start_seconds),
                    "end_seconds": float(interval.end_seconds),
                    "fields": [], "arrays": {}, "payload_bytes": 0,
                    "prefix": {}}

        def seal(self):
            return {"status": "PASS"}

    monkeypatch.setattr(
        prepared_cache_module, "PreparedCacheStream", RecordingCacheStream)

    def single_export(*args, **kwargs):
        calls["single_export"].append((args, kwargs))
        from woof.physics_compat import single_domain_physics_selection

        return {"schema": "gpuwm-native-direct-wrf-export-v3",
                "physics": single_domain_physics_selection(
                    exp.root.run,
                    expert_acknowledgements=kwargs["expert_acknowledgements"],
                    acknowledgement_provenance=kwargs["acknowledgement_provenance"])}

    monkeypatch.setattr(mapped_direct, "export_prepared_wrf", single_export)

    # One stand-in initialization result per domain, carrying the floor
    # field the real RealInitResult carries.  A test names a fired floor
    # by writing this dict's receipt before it drives the preparation.
    floor_results = {
        f"d{int(domain.grid_id):02d}": SimpleNamespace(
            surface_moisture_floor={})
        for domain in exp.domains}
    hierarchy_result = SimpleNamespace(
        static_catalog_receipt={"status": "PASS"},
        source_coverage_receipt={"status": "PASS"},
        # The real RegularSourceHierarchyResult declares this with a
        # None default; a double that omits it is not the contract the
        # route consumes.  None here is the no-corridor preparation, so
        # the proof this test compares stays byte-for-byte what it was.
        statics_corridor_receipt=None,
        forcing_times=tuple(snapshot.valid_time for snapshot in snapshots),
        boundary_interval_seconds=cadence,
        hierarchy=SimpleNamespace(
            artifacts=SimpleNamespace(receipt={"status": "PASS"}),
            wrf_manifest={"schema": "gpuwm-native-direct-wrf-hierarchy-export-v1"},
            timings_seconds={"initialize_children": 0.1},
            # The per-domain initialization moisture-floor receipt the real
            # NativeHierarchyExportResult carries, on the same rule as the
            # corridor above: a double that omits what the proof writer
            # reads is not the contract the route consumes.  Built by the
            # SHIPPED builder from per-domain stand-in results rather than
            # written out here, so a test that asks for a fired child floor
            # gets the block shape the route really writes.
            moisture_floor_receipts={},
        ),
    )

    def hierarchy(**kwargs):
        calls["hierarchy"].append(kwargs)
        # Rebuilt at CALL time: a test sets a floor on one domain's
        # stand-in result before it drives the preparation, the way the
        # real children only get their floors during the export.
        hierarchy_result.hierarchy.moisture_floor_receipts = (
            moisture_floor_proof_entries(
                tuple(floor_results.items()),
                when_unrecorded=(
                    "the mapped prepare double holds no ingest receipt")))
        return hierarchy_result

    monkeypatch.setattr(
        mapped_direct, "initialize_and_export_regular_source_hierarchy",
        hierarchy,
    )
    args = {
        "composition": files["composition.json"],
        "mapping": files["mapping.json"],
        "primary_files": [files["primary.bin"]],
        "supplement_files": {"terrain": files["supplement.bin"]},
        "provenance_files": {"terrain": files["provenance.md"]},
        "input_manifest": files["manifest.json"],
        "input_manifest_sha256": _DIGEST,
        "wps_namelist": files["namelist.wps"],
        "geog_root": geog_root,
        "experiment_config": files["experiment.toml"],
        "output_root": tmp_path / "output",
        "preprocess_backend": backend,
    }
    expected = SimpleNamespace(
        exp=exp,
        mapping=mapping,
        grids=grids,
        snapshots=snapshots,
        results=results,
        mets=mets,
        boundaries=boundaries,
        soil=soil,
        bundle=bundle,
        static=static,
        hierarchy_floor_results=floor_results,
    )
    return args, calls, expected


def test_single_domain_preserves_configured_physics_in_direct_export(monkeypatch, tmp_path):
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert proof["schema"] == mapped_direct.PROOF_SCHEMA
    assert len(calls["single_export"]) == 1
    assert not calls["hierarchy"]
    assert calls["build_static"] == 1
    assert calls["interpolate"] == len(expected.snapshots)
    assert args["output_root"].is_dir()
    # Streaming contract: every forcing time contributed its perimeter
    # frames, one state at a time, and the intervals were assembled once
    # from those frames.  The old builder took all the states at once,
    # which is what made preprocessing hold them all.
    frames = calls["frames"]
    count = len(expected.snapshots)
    assert len(frames["added"]) == count
    assert frames["kwargs"]["spec_bdy_width"] == (
        expected.exp.root.run.spec_bdy_width)
    # ORDERING CONTRACT (chained preparation): the start time is built
    # FIRST, written into the head and released, and each later time is
    # built after it with one forcing time resident.  Each interval is
    # written as soon as its two times exist and its older frame is then
    # released, so no whole-set boundary build happens at all.
    assert frames["arrival"] == list(range(count))
    assert frames["added"] == [result.state for result in expected.results]
    assert frames["built"] == []
    assert frames["intervals"] == list(range(count - 1))
    assert frames["released"] == list(range(count - 1))
    stream = calls["cache_stream"]
    assert stream["segments"] == list(range(count - 1))
    assert stream["head"]["initial_result"] is expected.results[0]
    assert len(stream["head"]["lbc"]["schedule"]) == count - 1
    # The start state carries no boundaries: the cache's fingerprint is
    # built from the segments, not from an attachment.
    assert all(result.state.lateral_boundaries is None
               for result in expected.results)
    assert proof["boundary_stream"]["head_sha256"]
    assert (args["output_root"] / "boundary-stream" / "head.json").is_file()


def test_prepare_threads_the_output_root_into_the_compose_scratch(
    monkeypatch,
    tmp_path,
):
    """The prep route names its destination to the compose scratch.

    Named breakage: the engine stages the whole composed frame stream
    -- tens of GB for the biggest registered sources -- in a scratch
    directory, and staged in the SYSTEM temp it filled a RAM-backed
    tmpfs /tmp and killed the bare default prep with a disk-quota
    error.  ``decode_composed_source`` places the scratch beside the
    destination it is told about; this pins that the prep front door
    actually tells it, because a caller that omits the destination
    silently re-selects the system temp and resurrects the failure.
    """

    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    recorded: dict[str, object] = {}

    def recording_decode(*_args, **kwargs):
        recorded.update(kwargs)
        return expected.bundle

    monkeypatch.setattr(
        mapped_direct, "decode_composed_source", recording_decode)

    mapped_direct.prepare_mapped_wrf(**args)

    assert recorded["scratch_destination"] == (
        Path(args["output_root"]).resolve())


def test_separate_analysis_starts_the_boundary_from_itself_and_uses_donor_aerosols(monkeypatch, tmp_path):
    # The first boundary record is the start state's own coupling and
    # interval 0 runs from it to the boundary product's next time, WRFDA
    # da_update_bc's rule (operational HRRR runs it after GSI).  It used to
    # stay the boundary product's f00, so an HRRR start under RAP boundaries
    # began its specified and relaxation zones from the RAP atmosphere.
    from contextlib import contextmanager
    import woof.initial_source as initial_source

    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    expected.exp.root.run.use_rap_aero_icbc = True
    expected.mets[0].fields.update(QNWFA=None, QNIFA=None)
    args["initial_inputs"] = tmp_path / "initial.json"
    monkeypatch.setattr(initial_source, "read_initial_inputs", lambda _: {})
    donor_result = SimpleNamespace(state=_State())
    calls_to_initializer = []
    original = mapped_direct.initialize_real

    def initialize(met, *positional, **kwargs):
        calls_to_initializer.append(kwargs)
        if "aerosol_snapshot" in kwargs:
            assert _is_donor_cut_of(kwargs["aerosol_snapshot"], expected.mets[0])
            return donor_result
        return original(met, *positional, **kwargs)

    monkeypatch.setattr(mapped_direct, "initialize_real", initialize)

    @contextmanager
    def analysis(*_, **kwargs):
        assert kwargs["valid_time"] == expected.exp.start_time
        yield initial_source.InitialAnalysis(
            expected.snapshots[0], expected.mapping,
            expected.bundle.soil_layer_contract,
            {"source": "distinct-analysis"}, {"request.json": b"bound donor"})

    monkeypatch.setattr(initial_source, "decode_initial_analysis", analysis)
    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert calls["frames"]["added"] == [result.state for result in expected.results]
    # Replaced once, frame 0, by the analysis, before any interval exists.
    assert calls["frames"]["replaced"] == [(0, donor_result.state, 0)]
    assert calls["cache_stream"]["head"]["initial_result"] is donor_result
    assert proof["initial_source"]["source"] == "distinct-analysis"
    assert proof["initial_source"]["aerosol_source"] == "boundary-analysis"
    assert _is_donor_cut_of(calls_to_initializer[1]["aerosol_snapshot"], expected.mets[0])
    assert (args["output_root"] / "source-evidence" / "initial" / "request.json").read_bytes() == b"bound donor"


def _is_donor_cut_of(donor, boundary_analysis):
    """The donor is the boundary analysis cut to the fields a donor is read for.

    The rest of the boundary analysis is released before the initial
    analysis is built (woof.ingest.real.aerosol_donor_snapshot); the kept
    fields are the boundary analysis' own arrays, not copies.
    """
    from woof.ingest.real import AEROSOL_DONOR_COORDINATE_FIELDS
    keep = {*AEROSOL_DONOR_COORDINATE_FIELDS, "QNWFA", "QNIFA", "QNWFA_SFC", "QNIFA_SFC"}
    return (donor is not boundary_analysis
            and set(donor.fields) == keep & set(boundary_analysis.fields)
            and all(donor.fields[name] is boundary_analysis.fields[name]
                    for name in donor.fields))


def test_the_chained_admission_is_priced_with_the_mapping_it_prepares_from(
    monkeypatch,
    tmp_path,
):
    """The forecast beside this preparation carries the mapping's masses.

    A user's own mapping runs as ``mapped``, a name whose registry row
    publishes no hydrometeor, while the boundary this route writes
    carries every mass the mapping declares.  The chained admission
    weighs the forecast beside its producer on one card, so it is priced
    with the mapping document itself
    (woof.boundary_fields.source_boundary_species).  Red with the
    source left out of the route's ``writer.admit``: the admission sees
    ``None`` and prices water vapour alone.
    """

    from woof.ingest import boundary_stream

    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    seen = []
    admit = boundary_stream.PreparedTreeWriter.admit

    def recording_admit(self, **kwargs):
        seen.append(kwargs.get("source"))
        return admit(self, **kwargs)

    monkeypatch.setattr(
        boundary_stream.PreparedTreeWriter, "admit", recording_admit)

    mapped_direct.prepare_mapped_wrf(**args)

    assert len(seen) == 1 and seen[0] is expected.mapping


def test_long_provenance_role_publishes_short_hash_named_evidence(
    monkeypatch,
    tmp_path,
):
    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    role = "twentycrv3_in_band_surface_provenance"
    args["provenance_files"] = {
        role: expected.bundle.terrain_provenance_path,
    }
    _bind_test_composition(args, expected, {
        "supplements": {"terrain_height": {"provenance_role": role}},
    })

    mapped_direct.prepare_mapped_wrf(**args)

    evidence = args["output_root"] / "source-evidence" / (
        mapped_direct._provenance_evidence_name(role, ".md")
    )
    assert evidence.read_bytes() == b"test"
    assert role not in evidence.name


def _bind_test_composition(args, expected, composition):
    args["composition"].write_text(json.dumps(composition), encoding="utf-8")
    expected.bundle.composition_sha256 = hashlib.sha256(
        args["composition"].read_bytes()).hexdigest()


def _install_donor_publication_fakes(
    monkeypatch, tmp_path, *, domain_count=1, terrain_donor=False,
):
    """Fake heavy weather work; exercise actual sealed evidence publication."""
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=domain_count, backend="cpu",
    )
    composition = {
        "supplements": {"terrain_height": {"provenance_role": "terrain"}},
        "field_sources": {},
    }
    records = []
    for name, fields in (
        ("surface", ["land_fraction"]),
        ("soil", ["soil_temperature", "volumetric_soil_moisture"]),
    ):
        role = f"{name}_authority"
        # Same basename, distinct paths and bytes: neither basename nor the
        # terrain file can stand in for a donor's declared identity.
        path = tmp_path / name / "provenance.md"
        path.parent.mkdir()
        path.write_text(f"sealed {name} donor evidence\n", encoding="utf-8")
        args["provenance_files"][role] = path
        composition["field_sources"][name] = {
            "provenance_role": role, "fields": fields,
        }
        records.append({
            "binding": name,
            "provenance": {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        })
    if terrain_donor:
        composition["supplements"] = {}
        composition["field_sources"]["surface"]["fields"].append("terrain_height")
        del args["provenance_files"]["terrain"]
        terrain = records[0]["provenance"]
        expected.bundle.terrain_provenance_path = Path(terrain["path"])
        expected.bundle.terrain_provenance_sha256 = terrain["sha256"]
    expected.bundle.contributing_sources = tuple(records)
    _bind_test_composition(args, expected, composition)
    return args, calls, expected


@pytest.mark.parametrize("domain_count", [1, 2])
@pytest.mark.parametrize("terrain_donor", [False, True])
def test_distinct_donor_provenance_publishes_for_single_and_nested_domains(
    monkeypatch, tmp_path, domain_count, terrain_donor,
):
    args, calls, _expected = _install_donor_publication_fakes(
        monkeypatch, tmp_path, domain_count=domain_count,
        terrain_donor=terrain_donor,
    )
    expected_bytes = {
        role: path.read_bytes() for role, path in args["provenance_files"].items()
    }

    mapped_direct.prepare_mapped_wrf(**args)

    evidence = args["output_root"] / "source-evidence"
    actual = {
        role: (evidence / mapped_direct._provenance_evidence_name(
            role, ".md")).read_bytes()
        for role in expected_bytes
    }
    assert actual == expected_bytes
    assert len(list(evidence.glob("provenance-*"))) == len(expected_bytes)
    assert bool(calls["hierarchy"]) == (domain_count == 2)


@pytest.mark.parametrize("when", ["after_decode", "during_copy"])
def test_donor_provenance_tamper_never_publishes_output(
    monkeypatch, tmp_path, when,
):
    args, _calls, _expected = _install_donor_publication_fakes(monkeypatch, tmp_path)
    donor = args["provenance_files"]["soil_authority"]
    if when == "after_decode":
        donor.write_bytes(b"changed donor evidence")
    else:
        original_copy = mapped_direct.shutil.copy2

        def corrupt_copy(source, destination, *copy_args, **copy_kwargs):
            result = original_copy(source, destination, *copy_args, **copy_kwargs)
            if Path(source).resolve() == donor.resolve():
                Path(destination).write_bytes(b"changed during evidence copy")
            return result

        monkeypatch.setattr(mapped_direct.shutil, "copy2", corrupt_copy)

    with pytest.raises(ValueError, match="mapped evidence changed before publication"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert not args["output_root"].exists()


def test_donor_provenance_path_swap_is_not_hidden_by_equal_contents(monkeypatch, tmp_path):
    args, _calls, _expected = _install_donor_publication_fakes(monkeypatch, tmp_path)
    original = args["provenance_files"]["soil_authority"]
    replacement = tmp_path / "replacement.md"
    replacement.write_bytes(original.read_bytes())
    args["provenance_files"]["soil_authority"] = replacement

    with pytest.raises(ValueError, match="decoded provenance path differs.*soil_authority"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert not args["output_root"].exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "undeclared"])
def test_inconsistent_decoded_donor_bindings_do_not_publish(monkeypatch, tmp_path, change):
    args, _calls, expected = _install_donor_publication_fakes(monkeypatch, tmp_path)
    records = list(expected.bundle.contributing_sources)
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[1] = records[0]
    else:
        records[1] = dict(records[1], binding="not_declared")
    expected.bundle.contributing_sources = tuple(records)

    with pytest.raises(ValueError, match="decoded contributing source inventory"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert not args["output_root"].exists()


def test_declared_provenance_roles_cannot_be_relabelled_after_decode(monkeypatch, tmp_path):
    args, _calls, _expected = _install_donor_publication_fakes(monkeypatch, tmp_path)
    args["provenance_files"]["unbound_role"] = args["provenance_files"].pop("soil_authority")

    with pytest.raises(ValueError, match="decoded provenance role inventory"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert not args["output_root"].exists()


def test_source_manifest_override_publishes_the_routes_own_authority(
    monkeypatch,
    tmp_path,
):
    """A named-source route's user-facing manifest survives the bridge.

    This row's ancestor asserted the ``_predecoded_bundle`` bypass --
    the one branch that skipped ``decode_composed_source``, and the last
    entry in ``ENGINE_GAPS``.  The seam that replaced it: the route
    verifies its OWN sealed manifest, bridges it into the generic
    composition-inputs document, and hands both here.  The decode runs
    against the bridged twin, but the prepared tree's evidence copy and
    identity chain stay bound to the route's own document -- it is what
    the forecast leg re-verifies with the route's own reader.
    """

    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cuda",
    )
    source_manifest = tmp_path / "member-manifest.json"
    source_manifest.write_text(
        '{"fixture": "route-manifest"}\n', encoding="utf-8")
    digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    args["_source_manifest"] = source_manifest
    args["_source_manifest_sha256"] = digest
    args["_source_adapter"] = "rw-wps-fixture-member-v1"

    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert len(calls["hierarchy"]) == 1
    routed = calls["hierarchy"][0]
    assert routed["source_identity"]["adapter"] \
        == "rw-wps-fixture-member-v1"
    assert routed["source_identity"]["input_manifest_sha256"] == digest
    assert routed["source_manifest_sha256"] == digest
    assert routed["bridge_manifest_sha256"] == digest
    assert routed["input_provenance"]["input_manifest_sha256"] == digest
    evidence = (
        args["output_root"] / "source-evidence" / "input-manifest.json")
    assert evidence.read_bytes() == source_manifest.read_bytes()
    # The bridged manifest is not erased -- it stays sealed inside the
    # composition receipt, which the proof carries whole.
    assert proof["source_composition"]["receipt_content_sha256"] == "4" * 64


def test_a_source_manifest_with_a_stale_sha_is_refused(
    monkeypatch,
    tmp_path,
):
    """The override is verified, never trusted: bytes must hash as declared."""

    args, _calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    source_manifest = tmp_path / "member-manifest.json"
    source_manifest.write_text("{}", encoding="utf-8")
    args["_source_manifest"] = source_manifest
    args["_source_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source manifest bytes differ"):
        mapped_direct.prepare_mapped_wrf(**args)
    args["_source_manifest_sha256"] = None
    with pytest.raises(ValueError, match="atomic pair"):
        mapped_direct.prepare_mapped_wrf(**args)


def test_prepare_rejects_mapping_change_between_target_and_decode(
    monkeypatch,
    tmp_path,
):
    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )

    def decode(*_args, **_kwargs):
        args["mapping"].write_text(
            json.dumps(_target_mapping(max_dom=3)),
            encoding="utf-8",
        )
        expected.bundle.mapping_sha256 = hashlib.sha256(
            args["mapping"].read_bytes()
        ).hexdigest()
        return expected.bundle

    monkeypatch.setattr(mapped_direct, "decode_composed_source", decode)
    with pytest.raises(ValueError, match="target validation and decode"):
        mapped_direct.prepare_mapped_wrf(**args)


def test_prepare_rejects_changed_evidence_bytes_before_publication(
    monkeypatch,
    tmp_path,
):
    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )

    def receipt(_bundle):
        args["composition"].write_bytes(b"changed composition")
        return {"receipt_content_sha256": "4" * 64}

    monkeypatch.setattr(mapped_direct, "mapped_composition_receipt", receipt)
    with pytest.raises(ValueError, match="evidence changed"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert not args["output_root"].exists()


@pytest.mark.parametrize(("domain_count", "backend", "expected_workers"), [
    (2, "cpu", 8),
    (2, "cuda", 1),
    (4, "cpu", 8),
])
def test_mapped_hierarchy_routes_complete_root_inputs(
        monkeypatch, tmp_path, domain_count, backend, expected_workers):
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=domain_count, backend=backend,
    )
    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert not calls["single_export"]
    assert len(calls["hierarchy"]) == 1
    routed = calls["hierarchy"][0]
    assert routed["exp"] is expected.exp
    assert routed["grids"] == expected.grids
    assert routed["snapshots"] == expected.snapshots
    assert routed["root_initial_result"] is expected.results[0]
    assert routed["root_met"] is expected.mets[0]
    assert routed["root_soil"] is expected.soil
    assert routed["root_static_fields"] is expected.static
    assert routed["root_boundaries"] is expected.boundaries
    assert routed["bridge_manifest_sha256"] == expected.bundle.input_manifest_sha256
    assert routed["source_manifest_sha256"] == expected.bundle.input_manifest_sha256
    assert routed["source_identity"]["mapping_sha256"] == (
        expected.bundle.mapping_sha256
    )
    assert routed["source_identity"]["composition_sha256"] == (
        expected.bundle.composition_sha256
    )
    assert routed["source_identity"]["adapter"] == \
        "rw-wps-mapped-composition-v2"
    assert set(routed["source_inventory"]) == set(expected.snapshots[0].fields)
    assert routed["workers"] == expected_workers
    assert routed["preprocess_backend"] == backend
    assert routed["soil_layer_contract"] is expected.bundle.soil_layer_contract
    assert routed["artifact_manifest_reference"] == (
        "../hierarchy-artifacts/domain-artifacts.json"
    )
    assert proof["domain_count"] == domain_count
    assert proof["hierarchy_workers"] == expected_workers


def _noise_bubble():
    """A 0.01 K bubble, the noise-twin shape the pair door authors."""
    from woof.experiment import BubbleConfig, PerturbationConfig

    return PerturbationConfig(bubbles=(BubbleConfig(
        center_lat=35.5, center_lon=-97.5, center_height_m=1500.0,
        radius_km=10.0, depth_m=1500.0, amplitude_k=0.01),))


def test_a_mapped_tree_defers_the_perturbation_block_the_gfs_tree_defers(
        monkeypatch, tmp_path, capsys):
    """The HRRR pressure-level tree prepares a [perturbation] block.

    The tree runner applies the bubbles to the restored states whatever
    source prepared the tree, so this route records them as deferred in
    the same receipt the GFS route writes and keeps its arrays
    unperturbed.  It refused the block outright before.
    """
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    expected.exp.perturbation = _noise_bubble()

    proof = mapped_direct.prepare_mapped_wrf(**args)

    from woof.experiment import deferred_initial_perturbation

    assert proof["initial_perturbation"] == deferred_initial_perturbation(
        expected.exp, "GFS-direct prepared-cache", announce=False)
    assert proof["initial_perturbation"]["config"] == (
        expected.exp.perturbation.receipt())
    published = json.loads(
        (args["output_root"] / "proof.json").read_text(encoding="utf-8"))
    assert published["initial_perturbation"] == proof["initial_perturbation"]
    (routed,) = calls["hierarchy"]
    assert routed["exp"].perturbation is expected.exp.perturbation
    # The prepared arrays stay the source's: no initialization was handed
    # an applier, and the cache identity is the unperturbed one.
    assert all(operands.get("initial_perturbation") is None
               for operands in calls["initialize_operands"])
    assert "initial_perturbation" not in routed["source_identity"]
    assert "deferred to prepared forecast initialization" in (
        capsys.readouterr().err)


def test_a_mapped_tree_without_the_block_writes_the_proof_it_always_did(
        monkeypatch, tmp_path):
    args, _calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert "initial_perturbation" not in proof


def test_a_single_mapped_domain_defers_the_perturbation_block(
        monkeypatch, tmp_path):
    """The prepared single-domain runner applies the bubbles to its restored
    state (fix/single-domain-perturbation), so a single-domain preparation
    is no longer refused: it prepares unperturbed arrays, and the runner
    writes the treatment receipt when it applies them."""
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    expected.exp.perturbation = _noise_bubble()

    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert proof is not None
    assert args["output_root"].is_dir()


def test_mapped_hierarchy_preserves_five_minute_offsets_end_to_end(
        monkeypatch, tmp_path):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu",
        cadence=300, mapping_updates={"boundary_interval_seconds": 300})

    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert proof["forcing_offsets_seconds"] == [0, 300]
    hierarchy_call = calls["hierarchy"][0]
    assert hierarchy_call["forcing_offsets_seconds"] == (0, 300)
    assert "forcing_hours" not in hierarchy_call
    assert args["output_root"].is_dir()


@pytest.mark.parametrize(
    ("domain_count", "cadence", "nz", "mapping_updates", "message"),
    [
        (2, 3600, 49, {"max_dom": 1}, "max_dom"),
        # nz differing from the mapping's reference count is ADOPTED
        # when the experiment carries an explicit ladder, so the
        # ladder-less imported-config shape is what still fails closed
        # here (the call below passes eta=False for the off-reference
        # count).
        (1, 3600, 48, {}, "vertical ladder is missing"),
        (
            1,
            3600,
            49,
            {"boundary_interval_seconds": 10_800},
            "boundary.*interval|cadence",
        ),
        (
            2,
            310,
            49,
            {"boundary_interval_seconds": 310},
            "whole number of root-domain steps",
        ),
    ],
)
def test_invalid_target_contract_stops_before_static_or_preprocessing(
        monkeypatch, tmp_path, domain_count, cadence, nz, mapping_updates,
        message):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch,
        tmp_path,
        domain_count=domain_count,
        backend="cpu",
        cadence=cadence,
        nz=nz,
        eta=(nz == 49),
        mapping_updates=mapping_updates,
    )
    with pytest.raises(ValueError, match=message):
        mapped_direct.prepare_mapped_wrf(**args)

    assert calls["build_static"] == 0
    assert calls["interpolate"] == 0
    assert calls["initialize"] == 0
    assert not calls["hierarchy"]
    assert not calls["single_export"]
    assert not args["output_root"].exists()


def test_analysis_capable_mapping_prepares_a_complete_forcing_window(monkeypatch, tmp_path):
    """A decode profile accepting one analysis can also supply a forecast window."""
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
        mapping_updates={"require_lateral_boundaries": False})
    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert calls["initialize"] == 2
    contract = mapped_direct._validate_target_contract(
        _expected.mapping, _expected.exp, 3600, hierarchy=False)
    assert contract["require_lateral_boundaries"] is True
    assert proof["boundary_interval_seconds"] == 3600


def test_analysis_capable_mapping_refuses_one_time_as_forecast_forcing(monkeypatch, tmp_path):
    """One analysis supplies no second boundary state, regardless of decoder policy."""
    from woof.ingest.source_coverage import ForcingSeriesRefusal
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
        mapping_updates={"require_lateral_boundaries": False})
    expected.bundle.regular_snapshots = lambda: expected.snapshots[:1]
    with pytest.raises(ForcingSeriesRefusal, match="requires at least two forcing times"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert calls["build_static"] == calls["interpolate"] == calls["initialize"] == 0
    assert not args["output_root"].exists()


def _icon_global_target_updates():
    """The packaged icon-global target's spacing keys: the 1 h DWD posts,
    taken at whole multiples (A173), while its route defaults to 3 h."""

    from woof.source_authorities import (
        BOUNDARY_MULTIPLES_KEY, packaged_mapping_target)

    target = packaged_mapping_target("icon-global-grib2-v1")
    assert target["boundary_interval_seconds"] == 3600
    assert target[BOUNDARY_MULTIPLES_KEY] is True
    return {"boundary_interval_seconds": 3600, BOUNDARY_MULTIPLES_KEY: True}


def _counting_decode(monkeypatch):
    decodes = []
    decode = mapped_direct.decode_composed_source

    def counted(*args, **kwargs):
        decodes.append(1)
        return decode(*args, **kwargs)

    monkeypatch.setattr(mapped_direct, "decode_composed_source", counted)
    return decodes


def _write_share_interval(args, seconds):
    args["wps_namelist"].write_text(
        f"&share\n interval_seconds = {seconds},\n/\n", encoding="utf-8")


@pytest.mark.parametrize("cadence_h", [3, 1])
def test_plan_review_holds_a_run_to_its_own_spacing_not_the_publishers(
        monkeypatch, tmp_path, cadence_h):
    """A173 review: icon-global's mapping declares the 1 h DWD posts and
    takes whole multiples, so its default 3 h run is held at plan review
    to the 3 h it is prepared at, the namelist's interval_seconds.  At
    dt = 54 s (3 h = 200 steps, 1 h = 66 2/3) the 3 h run prepares, and the
    same config fetched at --cadence 1 is refused before any decode, naming
    the 3600 s and the namelist it was read from."""

    seconds = cadence_h * 3600
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
        cadence=seconds, run_seconds=10800,
        mapping_updates=_icon_global_target_updates())
    expected.exp.dt_exact = lambda grid_id: Fraction(54, 3 ** (grid_id - 1))
    expected.exp.root.run.dt = 54
    _write_share_interval(args, seconds)
    decodes = _counting_decode(monkeypatch)

    if cadence_h == 3:
        proof = mapped_direct.prepare_mapped_wrf(**args)
        assert proof["schema"] == mapped_direct.PROOF_SCHEMA
        assert decodes == [1]
        assert len(calls["single_export"]) == 1
        return
    with pytest.raises(
            ValueError,
            match=r"boundary_interval_seconds = 3600 s for mapped target at "
                  r"&share/interval_seconds of .*namelist\.wps is not a "
                  r"whole number of root-domain steps: d01 dt = 54 s "
                  r"exactly, cadence/dt = 200/3"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert decodes == []
    assert calls["build_static"] == 0
    assert not args["output_root"].exists()


def test_plan_review_refuses_a_nest_start_off_the_runs_own_seams(
        monkeypatch, tmp_path):
    """The seam half: a child starting at +1 h on a 3 h icon-global run
    lands on no forcing time.  Plan review holds it to the run's 3 h and
    refuses it before the decode, as it did before A173 moved the
    mapping's spacing to the publisher's hour."""

    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu",
        cadence=10800, mapping_updates=_icon_global_target_updates())
    offsets = {1: Fraction(0), 2: Fraction(3600)}
    expected.exp.domain_start_offset_exact = lambda grid_id: offsets[grid_id]
    _write_share_interval(args, 10800)
    decodes = _counting_decode(monkeypatch)

    with pytest.raises(
            ValueError,
            match=r"d02 .*not aligned to the boundary-forcing cadence: "
                  r"boundary_interval_seconds = 10800 s"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert decodes == []
    assert not calls["hierarchy"]


def test_plan_review_without_a_namelist_spacing_waits_for_the_series(
        monkeypatch, tmp_path):
    """A namelist that declares no interval_seconds leaves a whole-multiple
    target's spacing unknown until the decode: plan review checks every
    other limit, and the decoded 3 h series is held to the timing law at
    its own spacing, so dt = 54 s is not refused for the 1 h it never
    uses."""

    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
        cadence=10800, mapping_updates=_icon_global_target_updates())
    expected.exp.dt_exact = lambda grid_id: Fraction(54, 3 ** (grid_id - 1))
    expected.exp.root.run.dt = 54
    assert mapped_direct._plan_review_spacing(
        expected.mapping["target"], args["wps_namelist"]) == (None, None)

    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert proof["schema"] == mapped_direct.PROOF_SCHEMA
    assert len(calls["single_export"]) == 1


def test_plan_review_of_a_single_spacing_target_reads_no_namelist(tmp_path):
    """A target without whole multiples takes one spacing, so plan review
    holds the run to it whatever the namelist says."""

    namelist = tmp_path / "namelist.wps"
    namelist.write_text("&share\n interval_seconds = 10800,\n/\n",
                        encoding="utf-8")
    assert mapped_direct._plan_review_spacing(
        _target_mapping()["target"], namelist) == (3600, None)
    multiples = _target_mapping(accept_boundary_interval_multiples=True)
    assert mapped_direct._plan_review_spacing(
        multiples["target"], namelist) == (
            10800, f"&share/interval_seconds of {namelist}")
    namelist.write_text("&share\n interval_seconds = 5400,\n/\n",
                        encoding="utf-8")
    assert mapped_direct._plan_review_spacing(
        multiples["target"], namelist) == (None, None)


def _mapped_cli_args(source_format: str, *decoder_args: str) -> list[str]:
    return [
        "--source-format", source_format,
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input", "/source/forcing-000",
        "--supplement", "terrain=/source/terrain-000",
        "--provenance", "terrain_provenance=/case/terrain.md",
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        *decoder_args,
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
    ]


def test_mapped_cli_rejects_source_format_mapping_mismatch(
        monkeypatch, capsys):
    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda _path: {"format": "netcdf"},
    )

    with pytest.raises(SystemExit) as error:
        mapped_direct.main(_mapped_cli_args(
            "grib1", "--grib1-bridge", "/bin/grib1_bridge",
        ))

    assert error.value.code == 2
    assert "differs from mapping format netcdf" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("source_format", "decoder_args", "expected"),
    [
        # A GRIB1 run that pins ANY decoder tool leaves the engine route
        # and owes the subprocess contract, so naming the wrong tool
        # still fails closed on the right one.  Omitting every flag no
        # longer belongs here: `mapped-engine` decodes GRIB1 records in
        # process and the door must stop asking for a bridge nothing
        # launches -- pinned below by its own test.
        ("grib1", ("--grib2-dump", "/bin/grib2_dump"), "grib1_bridge"),
        (
            "grib2",
            ("--grib2-inventory", "/bin/grib2_inventory"),
            "grib2_dump",
        ),
        (
            "netcdf",
            ("--grib1-bridge", "/bin/grib1_bridge"),
            "grib1_bridge",
        ),
    ],
)
def test_mapped_cli_decoder_inventory_fails_closed(
        monkeypatch, capsys, source_format, decoder_args, expected):
    """Still closed, now delivered as the refusal it always was.

    It used to be an argparse usage error -- ``error: grib2 requires
    decoder flags [...]`` and exit 2 -- which is the staged-tool
    papercut's original shape: a demand for a flag pointing at a file
    the reader may not have, with nothing about getting one.  The gate
    is unchanged and still names the missing role; it now exits with
    the preparation-refusal status and carries a remedy.
    """

    monkeypatch.setattr(
        mapped_direct, "load_mapping",
        lambda _path: {"format": source_format},
    )

    status = mapped_direct.main(_mapped_cli_args(source_format, *decoder_args))

    assert status == PREPARATION_REFUSAL_EXIT_CODE
    message = capsys.readouterr().err
    assert expected in message
    assert "decoder inventory differs from the contract" in message
    assert "remedy:" in message


def test_an_undeclared_compose_asks_for_the_bridge_its_route_will_launch(
        monkeypatch, capsys):
    """The bridge is required exactly while Python composes GRIB1.

    Named breakage: the staged-tool papercut -- a refusal that asks the
    caller for the path to an executable nothing is going to launch --
    and its mirror image, waving a run through that then dies because
    the tool it needed was never resolved.

    The condition that decides which one is right is the ROUTE, not the
    format: ``prepare_mapped_wrf`` COMPOSES on every call, so a GRIB1
    prep needs ``grib1_bridge`` exactly while ``compose`` is undeclared
    for GRIB1.  The engine composes it today, so the demand is measured
    against a table where it does not -- the state every format arrives
    in before its port is measured, and the state this door was written
    for.

    Kept because the near-miss is the expensive one: asking the
    capability table about ``decode`` instead of ``compose`` makes this
    door wave a bare run through as "Rust decodes in process", and the
    route then hands the work to the Python engine with no tools -- the
    refusal arrives nineteen frames deep in the decoder contract instead
    of at the door.  With the shipped table now agreeing about both
    subcommands, only this direction can still catch that.
    """

    from woof import mapped_engine_bridge

    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    monkeypatch.setattr(
        mapped_engine_bridge, "ENGINE_CAPABILITIES",
        dict(mapped_engine_bridge.ENGINE_CAPABILITIES,
             compose=frozenset(
                 mapped_engine_bridge.ENGINE_CAPABILITIES["compose"] or ())
             - {"grib1"}))
    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda _path: {"format": "grib1"})
    monkeypatch.setattr(
        mapped_direct, "prepare_mapped_wrf",
        lambda **kwargs: pytest.fail("the route ran with no decoder tools"))

    assert mapped_direct.main(_mapped_cli_args("grib1")) \
        == PREPARATION_REFUSAL_EXIT_CODE

    message = capsys.readouterr().err
    assert "grib1 decoder inventory differs from the contract" in message
    assert "grib1_bridge" in message
    assert "remedy:" in message


def test_a_compose_capable_engine_drops_the_grib1_bridge_demand(
        monkeypatch, capsys):
    """The shipped contract for GRIB1: nothing to launch, nothing asked.

    This was pinned against a clearly-labelled FAKE capability entry
    before the compose port landed; the entry is now the real one, and
    the assertion is unchanged.  A bare GRIB1 prep reaches
    ``prepare_mapped_wrf`` with every tool argument ``None`` because the
    engine composes those records in process -- the staged-tool papercut
    closing on its own evidence rather than on a hand-maintained list of
    formats.
    """

    observed = {}
    monkeypatch.delenv("GPUWM_MAPPED_ENGINE", raising=False)
    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda _path: {"format": "grib1"})

    def prepare(**kwargs):
        observed["prepare"] = kwargs
        return {"schema": "proof", "status": "PASS"}

    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", prepare)

    assert mapped_direct.main(_mapped_cli_args("grib1")) == 0

    assert observed["prepare"]["grib1_bridge"] is None
    assert observed["prepare"]["grib2_inventory"] is None
    assert observed["prepare"]["grib2_dump"] is None
    assert '"status": "PASS"' in capsys.readouterr().out


def test_role_bindings_preserve_aliases_paths_and_supplement_order():
    bindings = mapped_direct._role_bindings([
        "terrain-height.v1=/source/terrain=analysis-a.grib2",
        "terrain-height.v1=/source/terrain-b.grib2",
        "land_mask=/source/land.nc",
    ], multiple=True)

    assert bindings == {
        "terrain-height.v1": (
            Path("/source/terrain=analysis-a.grib2"),
            Path("/source/terrain-b.grib2"),
        ),
        "land_mask": (Path("/source/land.nc"),),
    }


def test_role_bindings_reject_duplicate_singleton_and_malformed_roles():
    with pytest.raises(ValueError, match="duplicate binding.*terrain"):
        mapped_direct._role_bindings([
            "terrain=/source/a", "terrain=/source/b",
        ], multiple=False)
    for binding in ("terrain", "=/source/a", "bad role=/source/a", "terrain="):
        with pytest.raises(ValueError, match="ROLE=PATH"):
            mapped_direct._role_bindings([binding], multiple=False)


def _corridor_forwarded(monkeypatch, corridor_argv):
    """What `main` hands `prepare_mapped_wrf` for one corridor spelling."""

    observed = {}
    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda path: {"format": "grib2"})

    def prepare(**kwargs):
        observed.update(kwargs)
        return {"schema": "proof", "status": "PASS"}

    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", prepare)
    assert mapped_direct.main([
        "--source-format", "grib2",
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input", "/source/f000.grib2",
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        "--grib2-inventory", "/bin/grib2_inventory",
        "--grib2-dump", "/bin/grib2_dump",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
        *corridor_argv,
    ]) == 0
    return observed["statics_corridor"]


def test_mapped_cli_forwards_the_corridor_selection_in_both_spellings(
        monkeypatch):
    # The bare flag stays the STRING "all" rather than being expanded
    # here: `validated_corridor_selection` owns the resolution, and two
    # readers expanding it independently is how a priced corridor set
    # and a written one come apart.
    assert _corridor_forwarded(monkeypatch, ["--statics-corridor"]) == "all"
    assert _corridor_forwarded(
        monkeypatch, ["--statics-corridor", "2,3"]) == (2, 3)
    assert _corridor_forwarded(monkeypatch, []) is None


@pytest.mark.parametrize(
    "value, message",
    [("2,x", "comma-separated list of child grid ids"),
     (",", "empty grid-id list")],
)
def test_mapped_cli_refuses_a_malformed_corridor_at_the_door(
        monkeypatch, capsys, value, message):
    # At the DOOR: the mapping must not have to load before a typo in an
    # argv value is reported, or the reader gets a FileNotFoundError for
    # an unrelated path and never sees the real complaint.
    def refuse_load(path):  # pragma: no cover - must never run
        raise AssertionError("the mapping loaded before argv was checked")

    monkeypatch.setattr(mapped_direct, "load_mapping", refuse_load)
    with pytest.raises(SystemExit):
        mapped_direct.main([
            "--source-format", "grib2",
            "--composition", "/case/composition.json",
            "--mapping", "/case/mapping.json",
            "--input", "/source/f000.grib2",
            "--input-manifest", "/case/input-manifest.json",
            "--input-manifest-sha256", _DIGEST,
            "--wps-namelist", "/case/namelist.wps",
            "--geog-root", "/static/WPS_GEOG",
            "--experiment-config", "/case/experiment.toml",
            "--output-root", "/output/mapped",
            "--statics-corridor", value,
        ])
    assert message in capsys.readouterr().err


def test_mapped_cli_forwards_exact_composed_hierarchy_arguments(
        monkeypatch, capsys):
    observed = {}

    def load_mapping(path):
        observed["loaded_mapping"] = path
        return {"format": "grib2"}

    def prepare(**kwargs):
        observed["prepare"] = kwargs
        return {"schema": "proof", "status": "PASS"}

    monkeypatch.setattr(mapped_direct, "load_mapping", load_mapping)
    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", prepare)
    argv = [
        "--source-format", "grib2",
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input", "/source/f000.grib2",
        "--input", "/source/f003.grib2",
        "--supplement", "terrain=/source/terrain-f000.grib2",
        "--supplement", "terrain=/source/terrain-f003.grib2",
        "--provenance", "terrain_provenance=/case/terrain.md",
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        "--grib2-inventory", "/bin/grib2_inventory",
        "--grib2-dump", "/bin/grib2_dump",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "7",
        "--cpu-preprocess-bridge", "/bin/libgpuwm_preprocess_cpu.so",
        "--hierarchy-workers", "6",
    ]

    assert mapped_direct.main(argv) == 0

    assert observed["loaded_mapping"] == Path("/case/mapping.json")
    assert observed["prepare"] == {
        "composition": Path("/case/composition.json"),
        "mapping": Path("/case/mapping.json"),
        "primary_files": [
            Path("/source/f000.grib2"),
            Path("/source/f003.grib2"),
        ],
        "supplement_files": {
            "terrain": (
                Path("/source/terrain-f000.grib2"),
                Path("/source/terrain-f003.grib2"),
            ),
        },
        "provenance_files": {
            "terrain_provenance": Path("/case/terrain.md"),
        },
        # No --contributing-mapping flags: a single-source preparation
        # forwards an empty cross-source inventory.
        "contributing_mappings": {},
        "input_manifest": Path("/case/input-manifest.json"),
        "input_manifest_sha256": _DIGEST,
        "grib1_bridge": None,
        "grib2_inventory": Path("/bin/grib2_inventory"),
        "grib2_dump": Path("/bin/grib2_dump"),
        "wps_namelist": Path("/case/namelist.wps"),
        "geog_root": Path("/static/WPS_GEOG"),
        "static_input": None,
        "static_receipt": None,
        "experiment_config": Path("/case/experiment.toml"),
        "output_root": Path("/output/mapped"),
        "preprocess_backend": "cpu",
        "preprocess_workers": 7,
        "cpu_preprocess_bridge": Path("/bin/libgpuwm_preprocess_cpu.so"),
        "hierarchy_workers": 6,
        "stock_wrf_export": "optional",
        # This argv names no --statics-corridor, and the absence is
        # forwarded as itself: None is "seal nothing", which is what
        # every preparation that does not move a nest wants.
        "statics_corridor": None,
    }
    assert '"status": "PASS"' in capsys.readouterr().out


def test_mapped_cli_input_list_resolves_the_named_files_in_order(
        monkeypatch, capsys, tmp_path):
    observed = {}

    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda path: {"format": "grib2"})

    def prepare(**kwargs):
        observed["prepare"] = kwargs
        return {"schema": "proof", "status": "PASS"}

    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", prepare)
    input_list = tmp_path / "inputs.list"
    # CRLF line endings and a blank line: what a Windows-authored list
    # looks like.  The paths come through verbatim, in file order.
    input_list.write_bytes(b"/source/f000.grib2\r\n\r\n/source/f003.grib2\n")
    argv = [
        "--source-format", "grib2",
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input-list", str(input_list),
        "--supplement", "terrain=/source/terrain-f000.grib2",
        "--provenance", "terrain_provenance=/case/terrain.md",
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        "--grib2-inventory", "/bin/grib2_inventory",
        "--grib2-dump", "/bin/grib2_dump",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
    ]

    assert mapped_direct.main(argv) == 0

    assert observed["prepare"]["primary_files"] == [
        Path("/source/f000.grib2"),
        Path("/source/f003.grib2"),
    ]
    assert '"status": "PASS"' in capsys.readouterr().out


def test_mapped_cli_refuses_both_input_spellings(tmp_path, capsys):
    input_list = tmp_path / "inputs.list"
    input_list.write_bytes(b"/source/f000.grib2\n")
    with pytest.raises(SystemExit) as stop:
        mapped_direct.main([
            "--source-format", "grib2",
            "--input", "/source/f000.grib2",
            "--input-list", str(input_list),
        ])
    assert stop.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_mapped_cli_refuses_an_empty_or_missing_input_list(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda path: {"format": "grib2"})
    input_list = tmp_path / "inputs.list"
    input_list.write_bytes(b"\r\n\n")
    argv = [
        "--source-format", "grib2",
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input-list", str(input_list),
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        "--grib2-inventory", "/bin/grib2_inventory",
        "--grib2-dump", "/bin/grib2_dump",
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
    ]
    with pytest.raises(SystemExit) as stop:
        mapped_direct.main(argv)
    assert stop.value.code == 2
    assert "names no input files" in capsys.readouterr().err

    input_list.unlink()
    with pytest.raises(SystemExit) as stop:
        mapped_direct.main(argv)
    assert stop.value.code == 2
    assert "--input-list" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Prebuilt hash-bound static cache (the bypass the other adapters already have)
# ---------------------------------------------------------------------------

def _install_prebuilt_static(monkeypatch, tmp_path, args, *, fields):
    """Point the run at a prebuilt native-static cache and record the calls.

    verify_native_static_receipt / load_native_static_cache are the already
    certified native_wrf_contract primitives, covered in
    tests/test_native_wrf_contract.py.  What is under test here is that
    prepare_mapped_wrf routes to them at all, with the right arguments,
    instead of rebuilding geography from WPS_GEOG.
    """

    static_npz = tmp_path / "prior-native-static.npz"
    static_npz.write_bytes(b"prebuilt-static-npz")
    receipt_path = tmp_path / "prior-geometry-receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    receipt = {"schema": "gpuwm-native-static-direct-v1", "status": "PASS"}
    seen = {"verify": [], "load": []}

    def verify(actual_receipt, actual_static, grid, cfg):
        seen["verify"].append((actual_receipt, actual_static, grid, cfg))
        return receipt

    def load(path, grid, ny, nx):
        seen["load"].append((path, grid, ny, nx))
        return dict(fields)

    monkeypatch.setattr(mapped_direct, "verify_native_static_receipt", verify)
    monkeypatch.setattr(mapped_direct, "load_native_static_cache", load)
    args["static_input"] = static_npz
    args["static_receipt"] = receipt_path
    return SimpleNamespace(
        npz=static_npz, receipt_path=receipt_path, receipt=receipt, seen=seen)


def test_prebuilt_static_cache_replaces_the_wps_geog_rebuild(
        monkeypatch, tmp_path):
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    prebuilt = _install_prebuilt_static(
        monkeypatch, tmp_path, args, fields=expected.static)

    proof = mapped_direct.prepare_mapped_wrf(**args)

    # The WPS_GEOG rebuild did not happen at all.
    assert calls["build_static"] == 0
    assert len(prebuilt.seen["verify"]) == 1
    assert len(prebuilt.seen["load"]) == 1
    # The receipt is verified against the same resolved cache that is loaded,
    # and against the target grid -- not merely read.
    verify_receipt, verify_static, verify_grid, _cfg = prebuilt.seen["verify"][0]
    assert verify_receipt == prebuilt.receipt_path.resolve()
    assert verify_static == prebuilt.npz.resolve()
    assert verify_grid is expected.grids[0]
    load_path, load_grid, _ny, _nx = prebuilt.seen["load"][0]
    assert load_path == prebuilt.npz.resolve()
    assert load_grid is expected.grids[0]

    execution = proof["execution_inputs"]
    assert execution["root_static_provider"] == "prebuilt-hash-bound-cache"
    assert execution["root_static_receipt"] == prebuilt.receipt
    # geog_root is still bound: the proof names the resolved datasets, and a
    # child domain would still need the tree.
    assert execution["geog_root"] == str(args["geog_root"].resolve())
    assert execution["geog_datasets"]
    # The run still publishes its own cache, so the next cycle can reuse it.
    assert (args["output_root"] / "native-static.npz").is_file()
    assert (args["output_root"] / "geometry-receipt.json").is_file()


def test_default_path_still_builds_from_geog_and_names_the_provider(
        monkeypatch, tmp_path):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )

    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert calls["build_static"] == 1
    execution = proof["execution_inputs"]
    assert execution["root_static_provider"] == "native-wps-geog"
    assert execution["root_static_receipt"] is None


def test_prebuilt_static_cache_is_recorded_in_the_hierarchy_proof(
        monkeypatch, tmp_path):
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu",
    )
    prebuilt = _install_prebuilt_static(
        monkeypatch, tmp_path, args, fields=expected.static)

    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert proof["schema"] == mapped_direct.HIERARCHY_PROOF_SCHEMA
    assert calls["build_static"] == 0
    execution = proof["execution_inputs"]
    assert execution["root_static_provider"] == "prebuilt-hash-bound-cache"
    assert execution["root_static_receipt"] == prebuilt.receipt
    # The loaded root static is what the children are seeded from.
    assert calls["hierarchy"][0]["root_static_fields"] == expected.static


def test_static_input_and_receipt_must_be_supplied_together(
        monkeypatch, tmp_path):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    args["static_input"] = tmp_path / "prior-native-static.npz"

    with pytest.raises(ValueError, match="must be supplied together"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert calls["build_static"] == 0

    del args["static_input"]
    args["static_receipt"] = tmp_path / "prior-geometry-receipt.json"
    with pytest.raises(ValueError, match="must be supplied together"):
        mapped_direct.prepare_mapped_wrf(**args)


def test_absent_prebuilt_static_fails_closed_before_any_work(
        monkeypatch, tmp_path):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu",
    )
    args["static_input"] = tmp_path / "does-not-exist.npz"
    args["static_receipt"] = tmp_path / "also-missing.json"

    from woof.ingest.source_coverage import RunInputRefusal

    with pytest.raises(RunInputRefusal, match=r"--static-input"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert calls["build_static"] == 0
    assert not args["output_root"].exists()


def test_mapped_cli_rejects_a_half_supplied_static_cache(monkeypatch, capsys):
    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda _path: {"format": "netcdf"})
    monkeypatch.setattr(
        mapped_direct, "prepare_mapped_wrf",
        lambda **_kwargs: pytest.fail("must not prepare"))
    argv = [
        "--source-format", "netcdf",
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input", "/source/f000.nc",
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
        "--static-input", "/prior/native-static.npz",
    ]

    with pytest.raises(SystemExit):
        mapped_direct.main(argv)
    assert "--static-input and --static-receipt" in capsys.readouterr().err


def test_mapped_cli_forwards_the_prebuilt_static_pair(monkeypatch, capsys):
    observed = {}
    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda _path: {"format": "netcdf"})
    monkeypatch.setattr(
        mapped_direct, "prepare_mapped_wrf",
        lambda **kwargs: observed.update(kwargs) or {"status": "PASS"})
    argv = [
        "--source-format", "netcdf",
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input", "/source/f000.nc",
        "--input-manifest", "/case/input-manifest.json",
        "--input-manifest-sha256", _DIGEST,
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
        "--static-input", "/prior/native-static.npz",
        "--static-receipt", "/prior/geometry-receipt.json",
    ]

    assert mapped_direct.main(argv) == 0
    assert observed["static_input"] == Path("/prior/native-static.npz")
    assert observed["static_receipt"] == Path("/prior/geometry-receipt.json")
    # --geog-root is still mandatory on this route.
    assert observed["geog_root"] == Path("/static/WPS_GEOG")
    assert '"status": "PASS"' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The next command.  Every other native front door that reaches a prepared
# bundle finishes by printing the forecast line with its digests filled in
# -- GFS, ERA5 and 20CRv3 all do.  The mapped route, which is the one every
# packaged source (icon-eu, rap, gefs, rrfs, ...) runs on, printed nothing,
# so a user who had just prepared a cycle had to hand-extract three SHA-256
# values out of a 42 KB proof document, and the one they reached for first
# -- the `proof_content_sha256` field inside it -- is the one value that
# can never be right.
# ---------------------------------------------------------------------------

def _staged_mapped_proof(tmp_path, monkeypatch):
    """A prepare double that writes a real proof.json, as the route does."""

    output_root = tmp_path / "prepared"
    # The real single-domain mapped proof's shape, taken off one: there
    # is NO top-level `input_manifest_sha256` on this route -- the
    # published manifest's digest lives in the composition receipt, and
    # `source-evidence/input-manifest.json` is written from those bytes
    # and verified against it.  A fixture that invented the top-level
    # key would have passed while the real document routed the reader to
    # the multi-domain runner.
    proof = {
        "schema": mapped_direct.PROOF_SCHEMA,
        "status": "PASS",
        "source_composition": {"input_manifest": {"sha256": _DIGEST}},
        "prepared_cache": {"content_sha256": "b" * 64},
        "physics": {"profile": "gpuwm-wsm6-ysu-noah-rrtmg-v1"},
    }

    def prepare(**kwargs):
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "proof.json").write_text(
            json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")
        return proof

    monkeypatch.setattr(
        mapped_direct, "load_mapping", lambda path: {"format": "grib2"})
    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", prepare)
    return output_root


def _mapped_argv(output_root, tmp_path, *extra):
    return [
        "--source-format", "grib2",
        "--composition", str(tmp_path / "composition.json"),
        "--mapping", str(tmp_path / "mapping.json"),
        "--input", str(tmp_path / "f000.grib2"),
        "--input-manifest", str(tmp_path / "input-manifest.json"),
        "--input-manifest-sha256", _DIGEST,
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--geog-root", str(tmp_path / "WPS_GEOG"),
        "--experiment-config", str(tmp_path / "experiment.toml"),
        "--output-root", str(output_root),
        *extra,
    ]


def test_every_packaged_source_is_a_source_the_runner_accepts():
    """The seam that lets this door answer without the forecast import.

    `_next_command_lines` decides whether to print a command by asking
    `packaged_profile_sources()`, because the RW-WPS standalone wheel
    stages this preprocessing module and excludes the forecast executor
    -- importing the runner here breaks that distribution.  What makes
    the substitution sound is that the runner builds its own mapped
    table from this one, so anything this door prints a command for is
    an id `--source` takes.  Held here rather than assumed.
    """

    from woof.prepared_single_domain_forecast import (
        SUPPORTED_SOURCES, _MAPPED_PACKAGED_PROFILE,
    )
    from woof.source_adapters import packaged_profile_sources

    packaged = set(packaged_profile_sources())
    assert packaged, "no packaged sources at all -- instrument is blind"
    assert packaged <= SUPPORTED_SOURCES, sorted(packaged - SUPPORTED_SOURCES)
    assert packaged == set(_MAPPED_PACKAGED_PROFILE), sorted(
        packaged ^ set(_MAPPED_PACKAGED_PROFILE))
    # ...and the id a user's own mapping prepares under is NOT one of
    # them, which is the arm that prints prose instead.
    assert "mapped" not in packaged


def test_the_mapped_route_prints_its_ready_to_run_forecast_command(
        monkeypatch, capsys, tmp_path):
    output_root = _staged_mapped_proof(tmp_path, monkeypatch)

    assert mapped_direct.main(_mapped_argv(
        output_root, tmp_path,
        "--prepared-forecast-source", "icon-eu")) == 0

    captured = capsys.readouterr()
    # stdout stays the machine-readable proof document, exactly as before.
    assert json.loads(captured.out)["status"] == "PASS"
    proof_digest = hashlib.sha256(
        (output_root / "proof.json").read_bytes()).hexdigest()
    for expected in (
            "python -m woof.prepared_single_domain_forecast",
            "--source icon-eu",
            f"--proof-sha256 {proof_digest}",
            f"--source-manifest-sha256 {_DIGEST}",
            f"--prepared-content-sha256 {'b' * 64}",
            "--physics-profile gpuwm-wsm6-ysu-noah-rrtmg-v1",
            "--io-mode history --outdir",
    ):
        assert expected in captured.err, expected


def test_a_mapping_of_your_own_says_why_no_command_can_be_printed(
        monkeypatch, capsys, tmp_path):
    """`--source mapped` prepares, and no prepared runner accepts it.

    Prose, never a command: the prepared-forecast runners take only the
    packaged source ids, so printing a line with `--source mapped` in it
    would be printing a command that exits 2 when pasted.
    """

    output_root = _staged_mapped_proof(tmp_path, monkeypatch)

    assert mapped_direct.main(_mapped_argv(
        output_root, tmp_path,
        "--prepared-forecast-source", "mapped")) == 0

    captured = capsys.readouterr()
    assert "python -m woof.prepared_single_domain_forecast \\" \
        not in captured.err
    assert "mapped" in captured.err
    assert "no forecast command" in captured.err


def test_an_unnamed_prepared_source_still_says_the_preparation_is_done(
        monkeypatch, capsys, tmp_path):
    """The hand-written `python -m woof.mapped_direct` call."""

    output_root = _staged_mapped_proof(tmp_path, monkeypatch)

    assert mapped_direct.main(_mapped_argv(output_root, tmp_path)) == 0

    captured = capsys.readouterr()
    assert "preparation complete" in captured.err
    assert "--prepared-forecast-source" in captured.err


def test_mapped_preparation_exports_mynn_from_its_own_config(monkeypatch, tmp_path):
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    expected.exp.root.run.bl_pbl_physics = 5
    expected.exp.root.run.sf_sfclay_physics = 5
    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert calls["single_export"][0][1]["experiment_config_suite"] is True
    selectors = proof["export"]["physics"]["domains"]["1"]["selectors"]
    assert selectors["bl_pbl_physics"] == 5
    assert selectors["sf_sfclay_physics"] == 5


def test_mapped_preparation_refuses_export_physics_drift(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    args, _calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    monkeypatch.setattr(mapped_direct, "export_prepared_wrf",
                        lambda *a, **k: {"schema": "gpuwm-native-direct-wrf-export-v3",
                                          "physics": {}})
    with pytest.raises(RuntimeError, match="export physics differs"):
        mapped_direct.prepare_mapped_wrf(**args)
    _assert_failed_unsealed(args["output_root"], "export physics differs")


def _assert_failed_unsealed(root, reason):
    """A failure after the head leaves an unsealed tree marked failed.

    The head is published early so a forecast can start beside the
    preparation; a later failure keeps it, with ``failed.json`` naming the
    reason, so the waiting forecast ends with that reason and the next
    preparation of the same output root rebuilds it.
    """
    from woof.ingest.boundary_stream import (
        prepared_tree_complete, unfinished_tree_reason,
    )

    assert not prepared_tree_complete(root)
    failed = json.loads(
        (root / "boundary-stream" / "failed.json").read_text(encoding="utf-8"))
    assert reason in failed["reason"]
    assert "producer failed" in unfinished_tree_reason(root)


def test_an_unchained_failure_publishes_nothing(monkeypatch, tmp_path):
    """WOOF_CHAINED_PREP=0: the head waits for the seal, as before."""
    monkeypatch.setenv("WOOF_CHAINED_PREP", "0")
    args, _calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    monkeypatch.setattr(mapped_direct, "export_prepared_wrf",
                        lambda *a, **k: {"schema": "gpuwm-native-direct-wrf-export-v3",
                                          "physics": {}})
    with pytest.raises(RuntimeError, match="export physics differs"):
        mapped_direct.prepare_mapped_wrf(**args)
    assert not args["output_root"].exists()


def test_a_two_domain_mapped_tree_chains_through_prepare_mapped_wrf(
        monkeypatch, tmp_path):
    """A CPU tree with chaining on: head, one segment per root interval, seal.

    Driven through ``prepare_mapped_wrf`` itself, with the hierarchy's
    head and seal and the artifact writers stood in (their own tests and
    the real-data identity proof cover them) and the real stream writer
    between them, so this holds the route's wiring: the head carries the
    tree and binds the child's receipt, every root interval becomes a
    segment, the seal is handed the whole boundary series, the one-shot
    hierarchy call is not made, and the forecast doors bind the head as a
    tree.  The start states between head and seal go through
    ``TreeStartStates`` (A136 L7a): released at the head with each child's
    head cache digest, re-read at the seal (the root's boundary set from
    the sealed cache, which here is the recording cache stream: every
    segment it was handed), and the sealed tree held to the head.
    """
    import woof.ingest.prepared_cache as prepared_cache_module
    from woof import stage_cli
    from woof.ingest.boundary_stream import (
        LAYOUT_DOMAIN_TREE, read_head, segment_marker_path)

    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    # The seal's receipt carries the sealed cache's content digest, which
    # the start states are re-read against and the sealed tree held to.
    monkeypatch.setattr(
        prepared_cache_module.PreparedCacheStream, "seal",
        lambda self: {"status": "PASS", "content_sha256": "f" * 64})
    receipt = {"grid_id": 2, "artifacts": {
        "prepared_cache": {"payload_bytes": 64,
                           "content_sha256": "e" * 64}}}
    receipt_bytes = json.dumps(receipt).encode("utf-8")
    seen = {}
    tree_head = SimpleNamespace(
        child_results=("d02 start",),
        forcing_identity={"forcing_hours": [0, 1]},
        static_receipt={"status": "PASS"},
        source_coverage_receipt={"status": "PASS"},
        statics_corridor_receipt=None,
        bound_source_identity=lambda identity: dict(identity))

    def head(**kwargs):
        seen["head"] = kwargs
        return tree_head

    def static_files(directory, *, domain, grid, static_fields):
        (Path(directory) / "native-static.npz").write_bytes(b"static")
        (Path(directory) / "geometry-receipt.json").write_text(
            "{}", encoding="utf-8")
        return {"sha256": "c" * 64}, {}

    def children(domain_root, *, exp, child_results, **kwargs):
        seen["children"] = child_results
        folder = Path(domain_root) / "d02"
        folder.mkdir(parents=True)
        (folder / "receipt.json").write_bytes(receipt_bytes)
        return [SimpleNamespace(receipt=receipt)]

    def binding(**kwargs):
        seen["binding"] = kwargs
        return SimpleNamespace(identity={"source": "chained-tree-test"},
                               metadata={"user": {}})

    def seal(sealed_head, **kwargs):
        seen["seal"] = (sealed_head, kwargs)
        Path(kwargs["artifact_output"]).mkdir(parents=True)
        return SimpleNamespace(
            hierarchy=SimpleNamespace(
                moisture_floor_receipts={},
                artifacts=SimpleNamespace(receipt={"status": "PASS"}),
                wrf_manifest={"status": "NOT_REQUESTED"},
                timings_seconds={"initialize_children": 0.1}),
            statics_corridor_receipt=None)

    class StartStates:
        @classmethod
        def release(cls, *, root_result, root_met, child_results,
                    child_content_sha256):
            seen["release"] = {"root_result": root_result,
                               "child_results": child_results,
                               "children": dict(child_content_sha256)}
            return cls()

        def reread(self, root, *, exp, grids, root_identity,
                   root_content_sha256):
            seen["reread"] = root_content_sha256
            return (seen["release"]["root_result"], "root met",
                    SimpleNamespace(intervals=tuple(
                        calls["cache_stream"]["segments"])),
                    seen["release"]["child_results"])

        def require_sealed_is_head(self, artifact_receipt, *,
                                   root_content_sha256):
            seen["sealed_is_head"] = root_content_sha256

    monkeypatch.setattr(mapped_direct, "TreeStartStates", StartStates)
    monkeypatch.setattr(
        mapped_direct, "prepare_regular_source_hierarchy_head", head)
    monkeypatch.setattr(mapped_direct, "write_domain_static_files",
                        static_files)
    monkeypatch.setattr(mapped_direct, "write_child_domain_artifacts",
                        children)
    monkeypatch.setattr(mapped_direct, "root_domain_artifact_binding",
                        binding)
    monkeypatch.setattr(mapped_direct, "seal_regular_source_hierarchy", seal)
    monkeypatch.setattr(mapped_direct, "hierarchy_moisture_floor_receipts",
                        lambda *operands: {})

    proof = mapped_direct.prepare_mapped_wrf(**args)

    root = args["output_root"]
    published = read_head(root)
    assert published["decision"]["chained"] is True
    assert published["layout"] == LAYOUT_DOMAIN_TREE
    assert published["domains"] == ["d01", "d02"]
    assert published["basis"]["tree"]["children_receipts"] == {
        "d02": hashlib.sha256(receipt_bytes).hexdigest()}
    assert published["basis"]["cache"]["directory"] == (
        "hierarchy-head/domains/d01/prepared-cache")
    # The children came from the start time, into the head.
    assert seen["children"] == ("d02 start",)
    assert seen["head"]["root_initial_result"] is expected.results[0]
    # Released at the head with the child's head cache digest, and the
    # sealed tree held to the head.
    assert seen["release"]["root_result"] is expected.results[0]
    assert seen["release"]["children"] == {"d02": "e" * 64}
    assert seen["reread"] == seen["sealed_is_head"] == "f" * 64
    # One segment per root interval, and the seal holds the whole series.
    intervals = len(expected.snapshots) - 1
    assert all(segment_marker_path(root, k).is_file()
               for k in range(intervals))
    sealed_head, sealed = seen["seal"]
    assert sealed_head is tree_head
    assert len(sealed["root_boundaries"].intervals) == intervals
    assert sealed["artifact_output"] == root / "hierarchy-artifacts"
    # The one-shot hierarchy call is the unchained route's.
    assert not calls["hierarchy"]
    assert proof["boundary_stream"]["head_sha256"] == published["head_sha256"]
    assert (root / "proof.json").is_file()
    # The forecast doors bind this head as a tree.
    bundle = stage_cli.resolve_head_bundle(root, published["head_sha256"])
    assert bundle["layout"] == "tree" and bundle["domains"] == 2


@pytest.mark.parametrize("mode", ["optional", "required", "off"])
@pytest.mark.parametrize("surface", [1, 5])
def test_mapped_export_intent_preserves_requested_native_physics(
        monkeypatch, tmp_path, mode, surface):
    from woof.config import RunConfig

    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    changes = {"hybrid_opt": 2, "hypsometric_opt": 2, "specified": True,
               "nested": False, "mp_physics": 6, "sf_sfclay_physics": surface,
               "bl_pbl_physics": 1 if surface == 1 else 5,
               "sf_surface_physics": 2 if surface == 1 else 3,
               "num_soil_layers": 4 if surface == 1 else 6}
    expected.exp.root.run = RunConfig(**(vars(expected.exp.root.run) | changes))
    proof = mapped_direct.prepare_mapped_wrf(**args, stock_wrf_export=mode)
    assert proof["stock_wrf_export"] == mode
    assert calls["initialize"] == 2
    assert len(calls["single_export"]) == (0 if mode == "off" else 1)
    assert args["output_root"].is_dir()
    if mode == "off":
        assert proof["export"]["status"] == "NOT_REQUESTED"
    else:
        selectors = proof["export"]["physics"]["domains"]["1"]["selectors"]
        for name in ("sf_sfclay_physics", "bl_pbl_physics", "sf_surface_physics"):
            assert selectors[name] == changes[name]


# The compatibility export represents sf_sfclay_physics 1/5 and MYNN since
# 3f8833f0643; what it still cannot write is RUC's soil (it demands Noah, 2),
# so the MYNN/RUC tree is the one that must refuse before decoding.
@pytest.mark.parametrize("surface", [5])
def test_required_hierarchy_export_refuses_before_decode_or_output(
        monkeypatch, tmp_path, surface):
    from woof.config import RunConfig
    from woof.wrf_direct import StockWrfExportUnsupported

    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    for domain in expected.exp.domains:
        domain.run = RunConfig(**(vars(domain.run) | {
            "mp_physics": 6, "hybrid_opt": 2, "hypsometric_opt": 2,
            "specified": domain.grid_id == 1, "nested": domain.grid_id != 1,
            "sf_sfclay_physics": surface,
            "bl_pbl_physics": 1 if surface == 1 else 5,
            "sf_surface_physics": 2 if surface == 1 else 3,
            "num_soil_layers": 4 if surface == 1 else 6}))
    def forbidden(*args, **kwargs):
        raise AssertionError("required export must refuse before decode")
    monkeypatch.setattr(mapped_direct, "decode_composed_source", forbidden)
    with pytest.raises(StockWrfExportUnsupported) as caught:
        mapped_direct.prepare_mapped_wrf(**args, stock_wrf_export="required")
    assert caught.value.unsupported == {"sf_surface_physics": (3, 2)}
    assert calls["build_static"] == calls["initialize"] == 0
    assert not args["output_root"].exists()


@pytest.mark.parametrize("mode", ["optional", "off"])
def test_native_hierarchy_receives_export_intent(monkeypatch, tmp_path, mode):
    args, calls, _ = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    proof = mapped_direct.prepare_mapped_wrf(**args, stock_wrf_export=mode)
    assert calls["hierarchy"][0]["stock_wrf_export"] == mode
    assert proof["stock_wrf_export"] == mode


def test_optional_export_refusal_keeps_prepared_artifacts(monkeypatch, tmp_path):
    from woof.wrf_direct import StockWrfExportUnsupported

    args, calls, _ = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    def unsupported(*args, **kwargs):
        raise StockWrfExportUnsupported("frozen soil cannot be represented")
    monkeypatch.setattr(mapped_direct, "export_prepared_wrf", unsupported)
    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert proof["export"]["status"] == "REFUSED"
    assert proof["export"]["schema"] == "gpuwm-native-direct-wrf-export-v3"
    assert "frozen soil" in proof["export"]["reason"]
    assert (args["output_root"] / "prepared-cache").is_dir()
    assert (args["output_root"] / "proof.json").is_file()
    assert calls["initialize"] == 2


@pytest.mark.parametrize("error", [OSError("write failed"), ValueError("corrupt cache")])
def test_optional_export_does_not_hide_io_or_corruption(monkeypatch, tmp_path, error):
    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    args, _, _ = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    def failed(*args, **kwargs):
        raise error
    monkeypatch.setattr(mapped_direct, "export_prepared_wrf", failed)
    with pytest.raises(type(error), match=str(error)):
        mapped_direct.prepare_mapped_wrf(**args)
    _assert_failed_unsealed(args["output_root"], str(error))
    assert not list(tmp_path.glob("output.tmp-*"))


@pytest.mark.parametrize("domains", [1, 2])
@pytest.mark.parametrize("pressure", [True, False])
def test_declared_preparation_policy_reaches_root_and_children(monkeypatch, tmp_path, domains, pressure):
    args, calls, expected = _install_prepare_fakes(monkeypatch, tmp_path,
        domain_count=domains, backend="cpu", water_policy="wrf_compat")
    # The normal entry reads actual caller-authored bytes through CaseData.
    args["experiment_config"].write_text(
        '[experiment]\n[case_data]\nforcing="not-fetched.grib"\nvtable="Vtable"\n'
        'wps_namelist="namelist.wps"\ngeog_root="geog"\noutput_title="caller"\n'
        'water_temperature_policy="wrf_compat"\nsfcp_to_sfcp=' + str(pressure).lower() + '\n',
        encoding="utf-8")
    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert all(row["sfcp_to_sfcp"] is pressure for row in calls["initialize_operands"])
    assert all(row.policy == "wrf_compat" for row in calls["water_statics"])
    if domains == 2:
        routed = calls["hierarchy"][0]
        assert routed["sfcp_to_sfcp"] is pressure
        assert routed["water_temperature_policy"] == "wrf_compat"
        policy = routed["source_identity"]["preparation_case_policy"]
        assert policy["sfcp_to_sfcp"] is pressure
        assert policy["water_temperature_policy"] == "wrf_compat"


# ---------------------------------------------------------------------
# the prepared DOCUMENT states whether the initialization floored vapour
# ---------------------------------------------------------------------


#: A fired surface floor, in the ingest's own receipt shape: the count
#: and the magnitude, which are what a reader has to have before the
#: word "floored" is something anyone can act on.
_FIRED_SURFACE_FLOOR = {
    "policy": "flag-sh-surface-qv-floored-to-wrf-qv-min-value",
    "floored_cells": 4,
    "min_value": -3.1e-07,
    "floor_value": 1e-06,
}


def _written_proof(args):
    """The proof as a READER of the bundle gets it: off disk, through
    json, not the mapping the route happened to return.

    The defect this item closes is a receipt that reached memory and
    stderr and stopped there, so asserting on the returned dict would
    reproduce the defect inside the test.
    """

    returned = mapped_direct.prepare_mapped_wrf(**args)
    written = json.loads(
        (args["output_root"] / "proof.json").read_text(encoding="utf-8"))
    assert written["schema"] == returned["schema"]
    return written


@pytest.mark.parametrize("floor, fired", [({}, False),
                                          (_FIRED_SURFACE_FLOOR, True)])
def test_a_single_domain_proof_states_the_floor_that_did_and_did_not_fire(
        monkeypatch, tmp_path, floor, fired):
    """BOTH ANSWERS ARE IN THE DOCUMENT, which is the whole point.

    A fired floor carries its magnitudes so the modification can be
    judged.  An unfired one is STATED rather than left out: an absent key
    reads as "prepared before the receipt existed", which is a claim
    about the release and not about this forecast, and no reader of the
    bundle could tell the two apart.
    """

    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    expected.results[0].surface_moisture_floor = floor

    proof = _written_proof(args)

    block = proof[MOISTURE_FLOOR_KEY]
    assert block["schema"] == MOISTURE_FLOOR_SCHEMA
    assert block["recorded"] is True
    assert block["fired"] is fired
    entry = block["floors"]["surface_moisture_floor"]
    assert entry["fired"] is fired
    assert entry.get("receipt", {}) == floor


def test_a_hierarchy_proof_answers_for_the_parent_and_for_each_child(
        monkeypatch, tmp_path):
    """A NEST IS NOT ONE ANSWER.  Each domain runs its own
    ``initialize_real``, so a root whose analyzed surface needed no floor
    and a child whose blended terrain produced one are two facts about
    one forecast; a tree-wide verdict would lose both.
    """

    args, _calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    expected.hierarchy_floor_results["d02"].surface_moisture_floor = (
        _FIRED_SURFACE_FLOOR)

    proof = _written_proof(args)

    blocks = proof[MOISTURE_FLOOR_BY_DOMAIN_KEY]
    assert set(blocks) == {"d01", "d02"}
    assert blocks["d01"]["fired"] is False
    assert blocks["d01"]["floors"]["surface_moisture_floor"] == {
        "fired": False}
    assert blocks["d02"]["fired"] is True
    assert blocks["d02"]["floors"]["surface_moisture_floor"] == {
        "fired": True, "receipt": _FIRED_SURFACE_FLOOR}
    # The single-domain key is NOT on a tree document: same name, two
    # shapes is how a consumer starts throwing on half a bundle library.
    assert MOISTURE_FLOOR_KEY not in proof


def test_only_the_start_time_builds_the_receipts_its_result_carries(
        monkeypatch, tmp_path):
    """Every later forcing time contributes its state to the boundaries only.

    The route keeps the start time's result and drops the others', so each
    later time is initialized with ``boundary_only`` and skips the receipts
    nobody reads; the start time, built first for the prepared head
    (woof/ingest/boundary_stream.py), builds them.
    """
    args, calls, _ = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    mapped_direct.prepare_mapped_wrf(**args, stock_wrf_export="off")
    assert calls["initialize"] == 2
    assert [row["boundary_only"] for row in calls["initialize_operands"]] == [
        False, True]


def _deep_mapped_root(tmp_path):
    """A 92-character output name in a 125-character folder: 277 deep."""
    if len(str(tmp_path)) >= 124:
        pytest.skip("temporary root already exceeds the 125-character parent")
    parent = tmp_path / ("p" * (125 - len(str(tmp_path)) - 1))
    return parent / ("mapped-tree-domain-z" + "x" * 72)


def test_a_mapped_domain_tree_too_deep_for_windows_is_refused_before_decode(
        monkeypatch, tmp_path):
    """The mapped door publishes the same domain tree the HRRR stage does;
    under a root that puts its header at 277 characters it is refused
    before any source is decoded or any static field built."""
    from woof import fetch_guard

    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    args["output_root"] = _deep_mapped_root(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("decoded under a root the forecast cannot read")

    monkeypatch.setattr(mapped_direct, "decode_composed_source", forbidden)
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)

    with pytest.raises(ValueError) as caught:
        mapped_direct.prepare_mapped_wrf(**args)

    message = str(caught.value)
    assert message.startswith("refusing output root ")
    assert "277 characters" in message
    assert calls["build_static"] == calls["initialize"] == 0
    assert not calls["hierarchy"]
    assert not args["output_root"].parent.exists()


def test_a_single_mapped_domain_prepares_where_a_tree_is_refused(
        monkeypatch, tmp_path):
    """A single domain publishes no tree, so the same root prepares."""
    from woof import fetch_guard

    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    args["output_root"] = _deep_mapped_root(tmp_path)
    monkeypatch.setattr(fetch_guard, "windows_path_limit", lambda: 259)

    mapped_direct.prepare_mapped_wrf(**args)

    assert calls["build_static"] == 1
    assert args["output_root"].is_dir()


# D-02 (2.8.8 acceptance, row I1): a RUC preparation whose layer-source
# soil moisture floor fired (90ae5edc9) wrote ``soil_moisture_floor`` into
# proof.json but not into the cache's user metadata, and the front door,
# which binds every CONDITIONAL_PREPARATION_RECEIPTS key from the proof and
# compares the user metadata EXACTLY, refused its own preparation with
# "prepared cache user metadata differs from source/proof/experiment".
# Every writer of the root's user metadata -- the single-domain head, the
# one-shot tree, the chained tree's head binding and its seal -- carries
# the identical receipt.
_RUC_FLOOR_RECEIPT = {
    "policy": "wrf-ruc-layer-source-smois-max-0.005",
    "floored_values": 660, "floored_columns": 425,
    "per_level": {"SMOIS_L1": 425, "SMOIS_L2": 235},
    "min_pre_floor": -0.029,
}


def _fire_the_ruc_layer_floor(monkeypatch):
    routed = mapped_direct.preprocess_land_surface_soil

    def fired(*args, **kwargs):
        routed(*args, **kwargs)
        return SimpleNamespace(soil_moisture_floor=dict(_RUC_FLOOR_RECEIPT))

    monkeypatch.setattr(mapped_direct, "preprocess_land_surface_soil", fired)


def _assert_user_binds_the_proofs_receipts(user, proof):
    from woof.ingest.prepared_cache import CONDITIONAL_PREPARATION_RECEIPTS

    assert proof["soil_moisture_floor"] == _RUC_FLOOR_RECEIPT
    for key in CONDITIONAL_PREPARATION_RECEIPTS:
        assert (key in user) == (key in proof), key
        assert user.get(key) == proof.get(key), key


def test_a_fired_ruc_floor_is_bound_in_the_single_domain_cache_header(
        monkeypatch, tmp_path):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    _fire_the_ruc_layer_floor(monkeypatch)

    proof = mapped_direct.prepare_mapped_wrf(**args)

    _assert_user_binds_the_proofs_receipts(
        calls["cache_stream"]["head"]["metadata"], proof)


def test_a_healthy_single_domain_cache_header_is_unchanged(
        monkeypatch, tmp_path):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")

    proof = mapped_direct.prepare_mapped_wrf(**args)

    assert "soil_moisture_floor" not in proof
    assert "soil_moisture_floor" not in calls["cache_stream"]["head"]["metadata"]


def test_a_fired_ruc_floor_is_bound_in_the_one_shot_tree_root_metadata(
        monkeypatch, tmp_path):
    args, calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    _fire_the_ruc_layer_floor(monkeypatch)

    proof = mapped_direct.prepare_mapped_wrf(**args)

    (routed,) = calls["hierarchy"]
    _assert_user_binds_the_proofs_receipts(routed["root_metadata"], proof)


def test_a_fired_ruc_floor_is_bound_in_the_chained_tree_root_metadata(
        monkeypatch, tmp_path):
    """The chained tree binds d01's user metadata at the head, before the
    proof exists, so the binding is held to the receipt the proof head
    will record (the same soil state's)."""
    args, _calls, _expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=2, backend="cpu")
    monkeypatch.setenv("WOOF_CHAINED_PREP", "1")
    _fire_the_ruc_layer_floor(monkeypatch)
    seen = {}

    class Bound(Exception):
        pass

    def static_files(directory, *, domain, grid, static_fields):
        return {"sha256": "c" * 64}, {}

    def binding(**kwargs):
        seen["root_metadata"] = kwargs["root_metadata"]
        raise Bound

    monkeypatch.setattr(
        mapped_direct, "prepare_regular_source_hierarchy_head",
        lambda **_kwargs: SimpleNamespace(
            child_results=("d02 start",),
            forcing_identity={"forcing_hours": [0, 1]},
            bound_source_identity=lambda identity: dict(identity)))
    monkeypatch.setattr(mapped_direct, "write_domain_static_files",
                        static_files)
    monkeypatch.setattr(mapped_direct, "write_child_domain_artifacts",
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mapped_direct, "root_domain_artifact_binding",
                        binding)

    with pytest.raises(Bound):
        mapped_direct.prepare_mapped_wrf(**args)

    _assert_user_binds_the_proofs_receipts(
        seen["root_metadata"], {"soil_moisture_floor": _RUC_FLOOR_RECEIPT})
