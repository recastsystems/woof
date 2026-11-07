"""WRF's topo_wind and gwd_opt as settings: import, validation, identity.

The kernels are graded against WRF v4.7.1 in
tests/test_terrain_drag_wrf471_parity.py; this file holds the plumbing that
decides whether they run at all.  Both options are off by default and a
domain without them must resolve, fingerprint and checkpoint exactly as it
did before the two fields existed.
"""
from __future__ import annotations

import dataclasses
import tomllib
from types import SimpleNamespace

import numpy as np

import pytest

from woof.config import (GWD_OPT_VALUES, TOPO_WIND_VALUES, RunConfig,
                          terrain_drag_refusal, validate_run_config)
from woof.experiment import build_experiment
from test_namelist_gaps import _import, _with


class _NumpyCuPy:
    """NumPy standing in for CuPy, with the one runtime attribute the driver reads.

    The AMD fold (c896f599e11) made _couple_pbl_slot ask
    ``cp.cuda.runtime.is_hip``; bare NumPy has no ``cuda``.
    """

    cuda = SimpleNamespace(runtime=SimpleNamespace(is_hip=False))

    def __getattr__(self, name):
        return getattr(np, name)


_NP_CUPY = _NumpyCuPy()


def _cfg(**over) -> RunConfig:
    base = dict(nx=10, ny=10, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=12.0, run_seconds=3600.0)
    base.update(over)
    return RunConfig(**base)


def _ysu(text):
    return text.replace(" bl_pbl_physics = 11, 11,\n",
                        " bl_pbl_physics = 1, 1,\n")


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

def test_the_fields_are_appended_last_at_wrfs_defaults():
    names = [f.name for f in dataclasses.fields(RunConfig)]
    # The drag pair was the tail until development branch (37a99fcd4,
    # merged at dd5ded73f) appended zadvect_implicit_variant after it,
    # default "wrf_471"; the pair keeps its positional place and WRF's 0.
    # 2.8.5 appends further fields after zadvect_implicit_variant
    # (tests/test_config_freeze.py pins that tail).
    at = names.index("topo_wind")
    assert names[at:at + 3] == ["topo_wind", "gwd_opt", "zadvect_implicit_variant"]
    assert RunConfig.__dataclass_fields__["topo_wind"].default == 0
    assert RunConfig.__dataclass_fields__["gwd_opt"].default == 0
    assert TOPO_WIND_VALUES == (0, 1, 2) and GWD_OPT_VALUES == (0, 1, 3)


@pytest.mark.parametrize("kwargs, needle", [
    (dict(topo_wind=3, gwd_opt=0, bl_pbl_physics=1), "topo_wind = 3"),
    (dict(topo_wind=0, gwd_opt=2, bl_pbl_physics=1), "gwd_opt = 2"),
    (dict(topo_wind=True, gwd_opt=0, bl_pbl_physics=1), "topo_wind = True"),
    (dict(topo_wind=1, gwd_opt=0, bl_pbl_physics=5), "is neither"),
    (dict(topo_wind=2, gwd_opt=0, bl_pbl_physics=1, sf_urban_physics=2),
     "BEP"),
    (dict(topo_wind=0, gwd_opt=3, bl_pbl_physics=0), "PBL driver"),
])
def test_each_refusal_names_what_would_break(kwargs, needle):
    why = terrain_drag_refusal(**kwargs)
    assert why is not None and needle in why


@pytest.mark.parametrize("kwargs", [
    dict(topo_wind=0, gwd_opt=0, bl_pbl_physics=0),
    dict(topo_wind=1, gwd_opt=0, bl_pbl_physics=1),
    dict(topo_wind=2, gwd_opt=3, bl_pbl_physics=1),
    dict(topo_wind=0, gwd_opt=1, bl_pbl_physics=5),
    dict(topo_wind=0, gwd_opt=3, bl_pbl_physics=11),
    # WRF v4.6.1's Shin-Hong reads CTOPO/CTOPO2 (module_bl_shinhong.F:1387,
    # :1472-1473); refused here until development branch.
    dict(topo_wind=1, gwd_opt=0, bl_pbl_physics=11),
    dict(topo_wind=2, gwd_opt=1, bl_pbl_physics=11),
])
def test_the_ported_combinations_pass(kwargs):
    assert terrain_drag_refusal(**kwargs) is None


def test_validate_run_config_refuses_topo_wind_without_ysu():
    cfg = _cfg(bl_pbl_physics=5, sf_sfclay_physics=5, topo_wind=1)
    with pytest.raises(ValueError, match="is neither"):
        validate_run_config(cfg)


def test_direct_physics_initialization_has_the_same_terrain_refusal():
    from woof.core.physics import initialize_physics

    cfg = _cfg(bl_pbl_physics=5, sf_sfclay_physics=5, topo_wind=1)
    with pytest.raises(ValueError, match="is neither"):
        initialize_physics(SimpleNamespace(), cfg)


@pytest.mark.parametrize("sase, option", [(False, 3), (True, 1), (True, 3)])
def test_drag_uses_the_pbl_boundary_inputs_without_changing_scalars(
        monkeypatch, sase, option):
    import woof.core.physics as physics
    import woof.core.terrain_drag as terrain_drag

    shape = (4, 2, 3)
    pblh = np.full(shape[1:], 100.0, np.float32)
    kpbl = np.zeros(shape[1:], np.int32)
    diagnosed = np.full(shape[1:], 325.0, np.float32)
    diagnosed_top = np.full(shape[1:], 4, np.int32)
    fields = {"pblh": pblh, "kpbl": kpbl,
              "xland": np.ones(shape[1:], np.float32),
              "br": np.full(shape[1:], 0.2, np.float32)}
    atmosphere = {name: np.full(shape, value, np.float32)
                  for name, value in (("u", 10.0), ("v", 3.0),
                                      ("theta", 300.0), ("dz", 100.0))}
    rates = {name: np.full(shape, value, np.float32)
             for name, value in (("du", 1.0), ("dv", 2.0),
                                 ("dtheta", 3.0), ("dqv", 4.0))}
    called = {}

    def diagnose(u, v, theta, *, dz_col):
        assert (u is atmosphere["u"] and v is atmosphere["v"]
                and theta is atmosphere["theta"]
                and dz_col is atmosphere["dz"])
        called["diagnose"] = True
        return diagnosed

    def top(dz, height):
        assert dz is atmosphere["dz"] and height is diagnosed
        return diagnosed_top

    def apply(atmosphere_arg, du, dv, **kwargs):
        assert atmosphere_arg is atmosphere
        called.update(kwargs)
        du[...] += 5.0
        dv[...] -= 6.0

    monkeypatch.setattr(physics, "cp", _NP_CUPY)
    monkeypatch.setattr(physics, "launch_bulk_richardson_zi", diagnose)
    monkeypatch.setattr(terrain_drag, "pbl_top_from_height", top)
    driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(sina=np.zeros(shape[1:], np.float32),
                                   cosa=np.ones(shape[1:], np.float32))
    driver.fields = fields
    driver.sase_active = sase
    driver.bldt_seconds = 24.0
    driver.terrain_drag = SimpleNamespace(gwd_opt=option, apply_gwd=apply)
    driver.gf_rthblten = np.zeros(shape, np.float32)
    driver.gf_rqvblten = np.zeros(shape, np.float32)
    driver.pbl_raw_rates = {name: np.zeros(shape, np.float32)
                            for name in ("du", "dv")}
    sentinel = object()

    def couple(state, cfg, raw):
        # The drag was deposited before mass coupling, and both retained
        # momentum lanes and cumulus forcing see the same PBL slot.
        assert np.all(raw["du"] == 6.0) and np.all(raw["dv"] == -4.0)
        return sentinel

    monkeypatch.setattr(physics, "couple_ysu_tendencies", couple)
    result = driver._couple_pbl_slot(_cfg(), rates, atmosphere=atmosphere)
    assert result is sentinel
    live = sase and option == 3
    assert called["pblh"] is (diagnosed if live else pblh)
    assert called["kpbl"] is (diagnosed_top if live else kpbl)
    assert ("diagnose" in called) is live
    assert called["dt"] == 24.0 and called["dx"] == 3000.0
    assert np.all(fields["pblh"] == 100.0) and np.all(fields["kpbl"] == 0)
    assert np.all(rates["dtheta"] == 3.0) and np.all(rates["dqv"] == 4.0)
    assert np.all(driver.gf_rthblten == 3.0)
    assert np.all(driver.gf_rqvblten == 4.0)
    assert np.all(driver.pbl_raw_rates["du"] == 6.0)
    assert np.all(driver.pbl_raw_rates["dv"] == -4.0)


def test_disabled_drag_keeps_the_shared_pbl_slot_unchanged(monkeypatch):
    import woof.core.physics as physics

    shape = (4, 2, 3)
    rates = {name: np.full(shape, value, np.float32)
             for name, value in (("du", 1.0), ("dv", 2.0),
                                 ("dtheta", 3.0), ("dqv", 4.0))}
    original = {name: value.copy() for name, value in rates.items()}
    driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    driver.terrain_drag = None
    driver.state = object()
    driver.gf_rthblten = driver.gf_rqvblten = None
    driver.pbl_raw_rates = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled drag must not run or allocate diagnostics")

    monkeypatch.setattr(driver, "_apply_gwd", forbidden)
    monkeypatch.setattr(physics, "launch_bulk_richardson_zi", forbidden)
    monkeypatch.setattr(physics, "couple_ysu_tendencies",
                        lambda state, cfg, raw: raw)
    assert driver._couple_pbl_slot(_cfg(), rates) is rates
    for name in original:
        assert np.array_equal(rates[name], original[name])


# ---------------------------------------------------------------------------
# The namelist importer
# ---------------------------------------------------------------------------

def test_topo_wind_and_gwd_opt_translate_per_domain(tmp_path):
    text, _ = _import(tmp_path, _ysu(_with(
        physics=" topo_wind = 1, 0,\n", dynamics=" gwd_opt = 3, 0,\n")))
    raw = tomllib.loads(text)
    assert (raw["shared"]["topo_wind"], raw["shared"]["gwd_opt"]) == (1, 3)
    child = raw["domain"][1]
    assert (child["topo_wind"], child["gwd_opt"]) == (0, 0)
    exp = build_experiment(raw, source="test")
    assert [d.run.topo_wind for d in exp.domains] == [1, 0]
    assert [d.run.gwd_opt for d in exp.domains] == [3, 0]


def test_the_v3_physics_placement_of_gwd_opt_imports(tmp_path):
    text, _ = _import(tmp_path, _with(physics=" gwd_opt = 1, 0,\n"))
    assert tomllib.loads(text)["shared"]["gwd_opt"] == 1


def test_the_two_placements_disagreeing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="gwd_opt"):
        _import(tmp_path, _with(physics=" gwd_opt = 1, 0,\n",
                                dynamics=" gwd_opt = 3, 0,\n"))


def test_topo_wind_under_shin_hong_is_refused_not_dropped(tmp_path):
    # WRF's Shin-Hong reads ctopo too; woof carries the arm in YSU only.
    with pytest.raises(ValueError, match="Shin-Hong"):
        _import(tmp_path, _with(physics=" topo_wind = 1, 1,\n"))


def test_out_of_range_values_are_refused(tmp_path):
    with pytest.raises(ValueError, match="topo_wind"):
        _import(tmp_path, _ysu(_with(physics=" topo_wind = 3, 0,\n")))
    with pytest.raises(ValueError, match="gwd_opt"):
        _import(tmp_path, _with(dynamics=" gwd_opt = 2, 0,\n"))


def test_a_child_domain_with_terrain_drag_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="grid_id = 2.*gwd_opt = 3"):
        _import(tmp_path, _with(dynamics=" gwd_opt = 3, 3,\n"))


def test_an_omitted_key_imports_byte_identically(tmp_path):
    from test_namelist_import import INPUT_TEXT

    text, _ = _import(tmp_path, INPUT_TEXT)
    zero, _ = _import(tmp_path, _with(dynamics=" gwd_opt = 0, 0,\n"))
    assert zero == text
    assert "topo_wind" not in text and "gwd_opt" not in text


# ---------------------------------------------------------------------------
# Identity: off is invisible, on binds
# ---------------------------------------------------------------------------

def test_the_defaults_leave_the_restart_identity_alone(tmp_path):
    from woof.core.model import restart_identity_payload
    from test_namelist_import import INPUT_TEXT

    text, _ = _import(tmp_path, INPUT_TEXT)
    payload = restart_identity_payload(
        build_experiment(tomllib.loads(text), source="test"))
    for domain in payload["domains"]:
        assert "topo_wind" not in domain["run"]
        assert "gwd_opt" not in domain["run"]
    text, _ = _import(tmp_path, _ysu(_with(physics=" topo_wind = 2, 0,\n")))
    payload = restart_identity_payload(
        build_experiment(tomllib.loads(text), source="test"))
    assert payload["domains"][0]["run"]["topo_wind"] == 2
    assert "topo_wind" not in payload["domains"][1]["run"]


def test_the_checkpoint_echo_drops_them_at_their_defaults():
    from woof.io.restart import (TERRAIN_DRAG_RUN_DEFAULTS,
                                  _configuration_digest_values)

    values = dataclasses.asdict(_cfg())
    digest = _configuration_digest_values(values)
    for name in TERRAIN_DRAG_RUN_DEFAULTS:
        assert name not in digest
    values["gwd_opt"] = 3
    assert _configuration_digest_values(values)["gwd_opt"] == 3


def test_an_older_prepared_header_is_tolerated_only_at_the_default():
    from woof.ingest.prepared_cache import DEFAULT_TOLERANT_IDENTITY_FIELDS

    assert {"run.topo_wind", "run.gwd_opt"} <= DEFAULT_TOLERANT_IDENTITY_FIELDS


def test_the_registry_lists_both_as_implemented():
    import json
    from pathlib import Path

    registry = json.loads((Path(__file__).resolve().parents[1] / "woof"
                           / "physics_registry_v2.json").read_text(
                               encoding="utf-8"))
    rows = registry["parameters"]
    assert rows["topo_wind"]["enum"] == [0, 1, 2]
    assert rows["gwd_opt"]["enum"] == [0, 1, 3]
    assert "unimplemented_reason" not in rows["topo_wind"]


# ---------------------------------------------------------------------------
# The static request
# ---------------------------------------------------------------------------

def test_each_option_asks_the_static_build_for_its_own_fields():
    from woof.core.terrain_drag import required_static_fields
    from woof.static.orographic import OROGRAPHIC_ROWS, orographic_request

    assert required_static_fields(0, 0) == ()
    assert required_static_fields(1, 0) == ("VAR_SSO",)
    assert required_static_fields(2, 0) == ("VAR",)
    one = required_static_fields(0, 1)
    assert set(one) == {"VAR", "CON", "OA1", "OA2", "OA3", "OA4", "OL1",
                        "OL2", "OL3", "OL4"}
    three = required_static_fields(0, 3)
    assert len(three) == 20 and all(n.endswith(("LS", "SS")) for n in three)
    for names in (one, three, ("VAR_SSO",)):
        assert set(orographic_request(names)) == set(names)
        assert set(names) <= set(OROGRAPHIC_ROWS)


def test_the_rows_are_geogrid_tbl_arws_default_rows():
    from woof.static.orographic import OROGRAPHIC_ROWS

    row = OROGRAPHIC_ROWS["VAR_SSO"]
    assert (row.directory(("default",)), row.gcell, row.masked_water) == (
        "varsso_10m", True, False)
    assert row.directory(("5m",)) == "varsso_5m"
    for name, leaf in (("CON", "con"), ("OA3", "oa3"), ("OL4", "ol4")):
        row = OROGRAPHIC_ROWS[name]
        assert row.directory(("default",)) == f"orogwd_10m/{leaf}"
        assert (row.gcell, row.masked_water) == (False, True)
    for name, leaf in (("VARLS", "varls"), ("OA2SS", "oa2ss")):
        assert OROGRAPHIC_ROWS[name].directory(("default",)) == \
            f"orogwd3_10m/{leaf}"


def test_a_missing_dataset_is_refused_with_where_to_get_it(tmp_path):
    from woof.static.orographic import build_orographic_fields

    with pytest.raises(FileNotFoundError, match="varsso_10m"):
        build_orographic_fields(object(), tmp_path, ("VAR_SSO",),
                                landuse_path=tmp_path / "landuse")


def test_orographic_arithmetic_never_falls_back_to_the_python_sampler(
        tmp_path, monkeypatch):
    from woof.static import orographic, rust_bridge

    monkeypatch.setattr(orographic, "_refuse_missing", lambda *args: None)
    monkeypatch.setattr(rust_bridge, "route", lambda *args: None)
    with pytest.raises(RuntimeError, match="changes WPS coordinate"):
        orographic.build_orographic_fields(
            object(), tmp_path, ("VAR_SSO",),
            landuse_path=tmp_path / "landuse")


def test_the_selection_request_is_empty_by_default(tmp_path):
    from woof.runtime import with_terrain_drag_statics
    from woof.static.build import GeogSelection

    selection = GeogSelection.fallback(tmp_path)
    assert selection.orographic == ()
    assert with_terrain_drag_statics(selection, _cfg()) is selection
    cfg = _cfg(bl_pbl_physics=1, topo_wind=1, gwd_opt=1)
    assert set(with_terrain_drag_statics(selection, cfg).orographic) == (
        {"VAR_SSO", "VAR", "CON", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2",
         "OL3", "OL4"})


def test_standalone_static_command_builds_the_requested_drag_fields(
        tmp_path, monkeypatch):
    import woof.runtime as runtime
    from woof.static.build import GeogSelection

    cfg = _cfg(bl_pbl_physics=1, topo_wind=1, gwd_opt=3)
    domain = SimpleNamespace(grid_id=1, run=cfg)
    data = SimpleNamespace(geog_root=tmp_path, static_highres=None)
    selection = GeogSelection.fallback(tmp_path)
    grid = object()
    monkeypatch.setattr(runtime, "single_domain", lambda exp: domain)
    monkeypatch.setattr(runtime, "experiment_grid", lambda exp, data: grid)
    monkeypatch.setattr(GeogSelection, "from_case_data",
                        lambda data, domain_id: selection)

    def build(g, root, *, selection):
        assert g is grid and root is tmp_path
        assert set(selection.orographic) == set(
            runtime.with_terrain_drag_statics(
                GeogSelection.fallback(tmp_path), cfg).orographic)
        assert len(selection.orographic) == 21
        return {name: np.zeros((2, 3), np.float32)
                for name in selection.orographic}

    monkeypatch.setattr(runtime, "build_static", build)
    path = runtime.write_static(object(), data, tmp_path / "static.npz")
    with np.load(path) as built:
        assert len(built.files) == 21 and "VAR_SSO" in built.files


@pytest.mark.parametrize("topo, gwd, resident_planes, directional_planes", [
    (0, 0, 0, 0), (1, 0, 2, 0), (2, 0, 2, 0),
    (0, 1, 10, 8), (0, 3, 20, 16), (1, 3, 22, 16),
])
def test_terrain_drag_has_resident_workspace_and_kernel_memory_prices(
        topo, gwd, resident_planes, directional_planes):
    from woof.core import preflight as pf
    from woof.core.physics_inventory import (
        terrain_drag_array_shapes, terrain_drag_transient_shapes)
    from woof.experiment import experiment_from_run_config
    from datetime import datetime, timezone

    cfg = _cfg(bl_pbl_physics=1, sf_sfclay_physics=1,
               topo_wind=topo, gwd_opt=gwd)
    resident = terrain_drag_array_shapes(cfg)
    assert len(resident) == resident_planes
    assert all(shape == (cfg.ny, cfg.nx) for shape in resident.values())
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1,
                                                  tzinfo=timezone.utc))
    dc = exp.domains[0]
    items = {item.name: item for item in pf.estimate_domain(dc).items}
    for name, shape in resident.items():
        assert items[name].category == "physics" and items[name].shape == shape
    transients = terrain_drag_transient_shapes(cfg)
    for name, shape in transients.items():
        assert items[name].category == "transient" and items[name].shape == shape
    if gwd:
        assert transients["terrain_drag/directional_statistics"] == (
            directional_planes, cfg.ny, cfg.nx)
        assert transients["terrain_drag/column_heights"] == (
            2, cfg.nz, cfg.ny, cfg.nx)
    modules = pf.domain_kernel_modules(dc, prices_refl=False)
    assert ("terrain_drag_composed" in modules) is bool(topo or gwd)
    assert pf.CHAINED_TRANSLATION_UNIT_FRAMES[
        "terrain_drag_composed"].max_local_size_bytes == 1184
    if not topo and not gwd:
        assert not transients
        assert not any(name.startswith("terrain_drag/") for name in items)


def test_sase_gsl_prices_its_live_boundary_inputs():
    from woof.config import SASE_PBL_SCHEME
    from woof.core.physics_inventory import terrain_drag_transient_shapes

    cfg = _cfg(bl_pbl_physics=SASE_PBL_SCHEME, gwd_opt=3)
    shapes = terrain_drag_transient_shapes(cfg)
    assert shapes["terrain_drag/sase_boundary"] == (2, cfg.ny, cfg.nx)
    assert "terrain_drag/sase_boundary" not in terrain_drag_transient_shapes(
        dataclasses.replace(cfg, gwd_opt=1))


@pytest.mark.parametrize("option, spp_pbl", [(3, 1), (3, 0), (1, 1)])
def test_gsl_drag_receives_the_spp_pbl_pattern_wrf_hands_it(
        monkeypatch, option, spp_pbl):
    """module_pbl_driver.F passes spp_pbl and pattern_spp_pbl to gwdo_gsl,
    which perturbs VAR, VARSS and both variance caps by the pattern's lowest
    level (rstoch1D = pattern_spp_pbl(i,kts,j)).  The port launched the GSL
    drag with rstoch = 0 under spp_pbl = 1 until development branch, which
    the WRF v4.6.1 oracle's spp cases measured (A/dx3000_spp,
    A/dx5000_spp, A/dx13000_spp, B/dx9000_spp; tests/
    test_gsl_drag_wrf461_column_oracle.py).  gwd_opt = 1 takes no pattern.
    """
    import woof.core.physics as physics

    monkeypatch.setattr(physics, "cp", np)
    shape = (4, 2, 3)
    plane = np.ones(shape[1:], np.float32)
    pattern = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(sina=plane, cosa=plane)
    driver.fields = {"pblh": plane, "kpbl": np.ones(shape[1:], np.int32),
                     "xland": plane, "br": plane}
    driver.sase_active = False
    driver.bldt_seconds = 24.0
    driver.spp_patterns = {"pbl": pattern} if spp_pbl else {}
    called = {}

    def apply(atmosphere, du, dv, **kwargs):
        called.update(kwargs)

    drag = SimpleNamespace(gwd_opt=option, apply_gwd=apply)
    rates = {name: np.zeros(shape, np.float32) for name in ("du", "dv")}
    driver._apply_gwd(drag, _cfg(spp_pbl=spp_pbl), rates, {"u": rates["du"]})
    if option == 3 and spp_pbl:
        assert called["rstoch"] is not None
        np.testing.assert_array_equal(called["rstoch"], pattern[0])
    else:
        assert called["rstoch"] is None
