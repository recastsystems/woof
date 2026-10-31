"""A downscaled child on its own static geography (engine item E3).

Two halves.  The science: the port of WRF v4.7.1 ndown's two steps
(:func:`woof.offline_child_geography.rebalance_to_child_terrain`, WRF's
``rebalance``, and :func:`blend_child_terrain`, WRF's ``blend_terrain``)
graded against WRF's own Fortran on the column cases of
``tools/ndown_wrf471_oracle`` (byte-unmodified source, gfortran 15.2 -O0,
glibc 2.43; fixture ``tests/data/ndown_wrf471.npz``).  The door: a child
builds its own geography by default, ``--parent-terrain`` keeps the route
every child ran before, and a child whose geography cannot be built is
refused before anything is reserved or run, naming that flag.

The port computes in float64 where WRF computes in REAL (float32), the
engine's preparation arithmetic throughout (the analytic base state of
:func:`woof.ingest.real._make_real_base_serial` that every real-data run
is built on).  So the grade is WRF's own float32 rounding: each field is
held to a few float32 epsilons of the total it is part of (the pressure
perturbation of the total pressure, the geopotential perturbation of the
column-top geopotential), bounds that sit just above the largest distance
measured on every case and orders of magnitude below what any single
mutation of the method produces (pinned below as well).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from conftest import requires_netcdf_bridge
from woof.cli import main as cli_main
from woof.config import load_config
from woof.core.grid import finalize_vertical_coord, make_vertical_coord
from woof.offline_child_geography import (
    CHILD_TERRAIN_OWN, CHILD_TERRAIN_PARENT, ChildGeographyError,
    ChildStaticPolicy, blend_child_terrain, load_child_static_policy,
    parent_projected_grid, parse_child_static_table,
    rebalance_to_child_terrain, static_table_text)
from test_downscale_cli import (  # noqa: F401  (the autouse fixture applies here)
    _SURFACE_PARENT_CONFIG, _a_box_that_can_draw, _restart_evidence)
from test_offline_child import _history

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/ndown_wrf471.npz"
#: sha256 of the fixture of record (tools/ndown_wrf471_oracle: make_cases.py
#: write, build.sh, make_cases.py collect; a development machine, 2026-10-01).  A
#: regenerated fixture changes this on purpose.
FIXTURE_SHA256 = "27a7a0187a90068b022189fca8e3b2cc7460c77d12c3a5e4c0cd2fa2255fc28d"
EPS32 = float(np.finfo(np.float32).eps)

#: Field -> bound, in float32 epsilons of the field's scale (see
#: :func:`_distances`).  Largest measured on the fixture: pb 4.2, t_init 1.5,
#: alb 4.1, phb 4.5, mub 3.9, t_2 2.1, p 0.03, alt 4.7, ph_2 4.6, psfc 4.5
#: (geopotential within 1.1 cm, the pressure perturbation within 0.0003 Pa).
#: The four mutations below land 250 to 57,000 times outside these bounds.
BOUNDS = {"pb": 8.0, "t_init": 4.0, "alb": 8.0, "phb": 8.0, "mub": 8.0,
          "t_2": 4.0, "p": 8.0, "alt": 8.0, "ph_2": 8.0, "psfc": 8.0}


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE.is_file():
        pytest.skip(f"{FIXTURE} is absent")
    return np.load(FIXTURE)


def _stems(data, prefix):
    return sorted({key.split("/")[0] for key in data.files
                   if key.startswith(prefix)})


def _case(data, stem):
    return (lambda key: data[f"{stem}/in/{key}"],
            lambda key: data[f"{stem}/wrf/{key}"])


def _jki(array):
    """numpy (j, k, i) <-> the port's (k, j, i)."""
    return np.transpose(np.asarray(array), (1, 0, 2))


def _rebalance(case_in, **override):
    """The port on one oracle case, with WRF's float32 coefficients."""
    znw = case_in("znw")
    coord = make_vertical_coord(
        znw.size - 1, hybrid_opt=int(case_in("hybrid_opt")),
        etac=float(case_in("etac")), eta_levels=znw)
    finalize_vertical_coord(coord, float(case_in("p_top")))
    for key in ("c1f", "c2f", "c3f", "c4f", "c1h", "c2h", "c3h", "c4h",
                "dnw", "rdnw", "rdn"):
        setattr(coord, key, np.asarray(case_in(key), dtype=np.float64))
    arguments = dict(
        theta=_jki(case_in("t_2")).astype(np.float64),
        mu=case_in("mu_2").astype(np.float64),
        qv=_jki(case_in("qv")).astype(np.float64), coord=coord,
        p_top=float(case_in("p_top")),
        terrain_interpolated=case_in("ht").astype(np.float64),
        terrain_child=case_in("ht_fine").astype(np.float64),
        base_temp=float(case_in("t00")),
        hypsometric_opt=int(case_in("hypsometric_opt")))
    arguments.update(override)
    return rebalance_to_child_terrain(**arguments)


def _distances(columns, wrf):
    """Per field, max |port - WRF| in float32 epsilons of its scale."""
    base = columns.base
    top = np.abs(base.phb + columns.ph)[-1][None]
    pairs = {
        "pb": (base.pb, wrf("pb"), base.pb),
        "t_init": (base.thb - 300.0, wrf("t_init"), base.thb),
        "alb": (base.alb, wrf("alb"), base.alb),
        "phb": (base.phb, wrf("phb"), top),
        "mub": (base.mub, wrf("mub"), base.mub),
        "t_2": (columns.theta, wrf("t_2"), columns.theta + 300.0),
        "p": (columns.p, wrf("p"), columns.p + base.pb),
        "alt": (columns.alt, wrf("alt"), columns.alt),
        "ph_2": (columns.ph, wrf("ph_2"), top),
        "psfc": (columns.psfc, wrf("psfc"), columns.psfc),
    }
    out = {}
    for name, (port, oracle, scale) in pairs.items():
        oracle = np.asarray(oracle, dtype=np.float64)
        if oracle.ndim == 3:
            oracle = _jki(oracle)
        out[name] = float(np.max(np.abs(port - oracle)
                                 / (EPS32 * np.abs(scale))))
    return out


def test_the_fixture_is_the_recorded_words():
    if not FIXTURE.is_file():
        pytest.skip(f"{FIXTURE} is absent")
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_the_fixture_covers_every_regime(fixture):
    rebalance = _stems(fixture, "rebalance-")
    assert len(rebalance) == 12 and len(_stems(fixture, "blend-")) == 3
    for stem in rebalance:
        case_in, _ = _case(fixture, stem)
        change = case_in("ht_fine") - case_in("ht")
        assert np.isfinite(change).all()
    # sea on both terrains, the parent's sea under the child's land, and a
    # child's ridges and canyons hundreds of metres above and below its parent
    coast = _case(fixture, "rebalance-h59-hyps2-coast")[0]
    assert coast("ht")[0, 0] == 0 and coast("ht_fine")[0, 0] == 0
    assert coast("ht")[0, 1] == 0 and coast("ht_fine")[0, 1] > 0
    canyon = _case(fixture, "rebalance-h59-hyps2-canyon")[0]
    change = canyon("ht_fine") - canyon("ht")
    assert change.max() > 400 and change.min() < -250
    hyps = {int(_case(fixture, s)[0]("hypsometric_opt")) for s in rebalance}
    hybrid = {int(_case(fixture, s)[0]("hybrid_opt")) for s in rebalance}
    assert hyps == {1, 2} and hybrid == {0, 2}


def test_rebalance_is_wrfs_on_every_case(fixture):
    worst = {}
    for stem in _stems(fixture, "rebalance-"):
        case_in, wrf = _case(fixture, stem)
        distances = _distances(_rebalance(case_in), wrf)
        for name, value in distances.items():
            assert value <= BOUNDS[name], (stem, name, value)
            worst[name] = max(worst.get(name, 0.0), value)
    assert set(worst) == set(BOUNDS)


@pytest.mark.parametrize("mutation", [
    "terrains-swapped", "dry-column", "other-hypsometric-form",
    "no-theta-shift"])
def test_the_gate_fires_on_a_rebalance_that_is_not_wrfs(fixture, mutation):
    """Each single change of the method lands far outside the bounds."""
    stem = "rebalance-h59-hyps2-canyon"
    case_in, wrf = _case(fixture, stem)
    if mutation == "terrains-swapped":
        columns = _rebalance(
            case_in, terrain_interpolated=case_in("ht_fine").astype(float),
            terrain_child=case_in("ht").astype(float))
    elif mutation == "dry-column":
        columns = _rebalance(case_in, qv=np.zeros_like(
            _jki(case_in("qv")), dtype=np.float64))
    elif mutation == "other-hypsometric-form":
        columns = _rebalance(case_in, hypsometric_opt=1)
    else:
        # the base state on the child's terrain without ndown's
        # t_init - t_init_int shift: interpolated theta kept as it was
        reference = _rebalance(case_in)
        columns = _rebalance(
            case_in, theta=_jki(case_in("t_2")).astype(float)
            - reference.theta_shift)
    distances = _distances(columns, wrf)
    assert max(distances[name] / BOUNDS[name] for name in BOUNDS) > 100.0


def test_rebalance_on_its_own_terrain_changes_nothing(fixture):
    """ndown's step is a projection: a state already on the child's
    terrain, rebalanced onto that same terrain, comes back as it was."""
    case_in, _ = _case(fixture, "rebalance-s40-hyps2-high")
    first = _rebalance(case_in)
    again = _rebalance(
        case_in, theta=first.theta,
        terrain_interpolated=case_in("ht_fine").astype(float))
    assert np.array_equal(again.theta_shift, np.zeros_like(first.theta))
    assert np.array_equal(again.theta, first.theta)
    assert np.array_equal(again.p, first.p)
    assert np.array_equal(again.ph, first.ph)
    assert np.array_equal(again.psfc, first.psfc)


def test_blend_is_wrfs(fixture):
    """Outside the blend band the result is a copy, bit for bit; inside it
    WRF's float32 weighted sum is within its own rounding of the port's."""
    for stem in _stems(fixture, "blend-"):
        case_in, wrf = _case(fixture, stem)
        coarse = case_in("ter_interpolated")
        fine = case_in("ter_input")
        sbw = int(case_in("spec_bdy_width"))
        width = int(case_in("blend_width"))
        port = blend_child_terrain(coarse, fine, spec_bdy_width=sbw,
                                   blend_width=width)
        oracle = wrf("ter_input")
        ny, nx = fine.shape
        i = np.arange(1, nx + 1)[None, :]
        j = np.arange(1, ny + 1)[:, None]
        edge = np.minimum(np.minimum(i, j),
                          np.minimum(nx + 1 - i, ny + 1 - j))
        band = (edge > sbw) & (edge <= sbw + width)
        copied = ~band
        assert np.array_equal(port[copied].astype(np.float32),
                              oracle[copied]), stem
        assert np.array_equal(port[edge <= sbw], coarse[edge <= sbw])
        assert np.array_equal(port[edge > sbw + width],
                              fine[edge > sbw + width])
        scale = np.maximum(np.abs(coarse), np.abs(fine))[band]
        distance = np.abs(port[band] - oracle[band].astype(np.float64))
        assert np.all(distance <= 4.0 * EPS32 * scale), stem


def test_a_child_too_small_for_its_blend_is_refused():
    terrain = np.zeros((20, 20))
    with pytest.raises(ChildGeographyError, match="--parent-terrain"):
        blend_child_terrain(terrain, terrain, spec_bdy_width=7,
                            blend_width=5)


# ---------------------------------------------------------------------
# The [static] table and the policy
# ---------------------------------------------------------------------

def test_the_static_table_drops_no_key():
    with pytest.raises(ValueError, match="terain"):
        parse_child_static_table({"terain": "own"}, source="child.toml")
    with pytest.raises(ValueError, match="'own' builds"):
        parse_child_static_table({"terrain": "mine"}, source="child.toml")
    assert parse_child_static_table(None, source="child.toml") == {}
    assert parse_child_static_table(
        {"terrain": "parent"}, source="child.toml") == {"terrain": "parent"}


def _child_toml(tmp_path, static=""):
    from woof.downscale import _derive_child_run_config, _render_child_toml

    merged = _derive_child_run_config(
        _SURFACE_PARENT_CONFIG,
        parent={"nx": 20, "ny": 18, "dx": 3000.0, "dy": 3000.0}, ratio=3,
        child_nx=24, child_ny=24, run_seconds=600.0, output_interval_s=300.0)
    path = tmp_path / "child.toml"
    path.write_text(_render_child_toml(merged) + static, encoding="utf-8")
    return path


def test_own_geography_is_the_default_and_a_flag_wins(tmp_path):
    from woof.offline_child import resolve_child_run_config

    plain = _child_toml(tmp_path)
    cfg = resolve_child_run_config(plain)
    policy = load_child_static_policy(plain, cfg, geog_root=tmp_path)
    assert policy.terrain == CHILD_TERRAIN_OWN and policy.own
    assert policy.source == "engine default"
    # a 1 km child takes the engine's high-resolution terrain default
    assert policy.highres is not None
    assert policy.highres.terrain_source == "copernicus-dem-glo30"
    flagged = load_child_static_policy(plain, cfg, terrain="parent")
    assert flagged.terrain == CHILD_TERRAIN_PARENT
    assert flagged.source == "door flag"
    assert flagged.receipt()["terrain_policy"] == "sint-parent-inherited"


def _subkm_child_toml(tmp_path, static=""):
    from woof.downscale import _derive_child_run_config, _render_child_toml

    merged = _derive_child_run_config(
        _SURFACE_PARENT_CONFIG,
        parent={"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}, ratio=3,
        child_nx=24, child_ny=24, run_seconds=600.0, output_interval_s=300.0)
    path = tmp_path / "child-333m.toml"
    path.write_text(_render_child_toml(merged) + static, encoding="utf-8")
    return path


def test_a_subkm_child_takes_land_cover_and_soil_not_the_30s_baseline(tmp_path):
    """WOOF run_373d64f91dce7cdb: a 333 m nest-down built its land use, soil,
    albedo and emissivity from 30-arc-second data (2-3 child cells per source
    cell), so its 2 m temperature was made of ~1 km squares."""
    from woof.offline_child import resolve_child_run_config
    from woof.offline_child_geography import CHILD_LAND_COVER_MAX_DX_M

    path = _subkm_child_toml(tmp_path)
    cfg = resolve_child_run_config(path)
    assert float(cfg.dx) == pytest.approx(1000.0 / 3)
    assert float(cfg.dx) <= CHILD_LAND_COVER_MAX_DX_M
    policy = load_child_static_policy(path, cfg, geog_root=tmp_path)
    assert policy.highres is not None and policy.highres.enabled
    assert policy.highres.fields == "auto"
    assert policy.highres.landcover_source == "auto"
    assert policy.highres.soil_source == "soilgrids"
    assert policy.highres.terrain_source == "copernicus-dem-glo30"
    assert policy.highres.applies_to(type("G", (), {"dx": float(cfg.dx)})())
    assert policy.receipt()["highres"]["fields"] == "auto"
    # the 1 km child keeps the terrain-only default it always had
    one_km = _child_toml(tmp_path)
    one_cfg = resolve_child_run_config(one_km)
    assert load_child_static_policy(one_km, one_cfg).highres.fields == "terrain"


def test_a_declared_block_is_kept_and_the_cache_flag_moves_the_cache(tmp_path):
    from woof.offline_child import resolve_child_run_config

    cache = tmp_path / "box-cache"
    path = _subkm_child_toml(tmp_path)
    cfg = resolve_child_run_config(path)
    flagged = load_child_static_policy(path, cfg, highres_cache=cache)
    assert flagged.highres.cache_root == cache.resolve()
    assert flagged.highres.fields == "auto"
    mine = json.dumps(str(tmp_path / "mine"))
    declared = _subkm_child_toml(
        tmp_path, '\n[static]\n[static.highres]\nenabled = true\n'
        f'cache_root = {mine}\nfields = "terrain"\n')
    dcfg = resolve_child_run_config(declared)
    kept = load_child_static_policy(declared, dcfg)
    assert kept.highres.fields == "terrain"
    assert kept.highres.cache_root == tmp_path / "mine"
    off = _subkm_child_toml(
        tmp_path, '\n[static]\n[static.highres]\nenabled = false\n'
        f'cache_root = {mine}\n')
    ocfg = resolve_child_run_config(off)
    assert not load_child_static_policy(off, ocfg).highres.enabled


def test_the_file_decides_when_no_flag_is_given(tmp_path):
    from woof.offline_child import resolve_child_run_config

    path = _child_toml(tmp_path, static='\n[static]\nterrain = "parent"\n'
                       'smooth_option = "1-2-1"\nsmooth_passes = 3\n')
    cfg = resolve_child_run_config(path)
    policy = load_child_static_policy(path, cfg)
    assert policy.terrain == CHILD_TERRAIN_PARENT
    assert policy.source == "child config [static]"
    assert policy.receipt()["terrain_smoothing"]["smooth_passes"] == 3
    assert load_child_static_policy(path, cfg, terrain="own").own


def test_a_derived_config_says_which_terrain_it_was_built_on(tmp_path):
    text = static_table_text(ChildStaticPolicy(terrain="parent"))
    assert text == '\n[static]\nterrain = "parent"\n'
    text = static_table_text(ChildStaticPolicy(
        terrain="own", geog_root=tmp_path, source="door flag"))
    assert f'geog_root = {json.dumps(str(tmp_path))}' in text


@pytest.mark.parametrize("static", [
    'smooth_option = "none"\n',
    'smooth_option = "1-2-1"\nsmooth_passes = 3\n',
    'smooth_precision = "wps-float32"\n',
])
def test_the_high_resolution_overlay_keeps_the_childs_smoother(tmp_path, static):
    from woof.offline_child import resolve_child_run_config

    path = _child_toml(tmp_path, '\n[static]\n' + static)
    cfg = resolve_child_run_config(path)
    policy = load_child_static_policy(path, cfg)
    assert policy.highres is not None and policy.highres.enabled
    assert policy.highres.smoothing_for(cfg.grid_id) == policy.smoothing


def test_relative_geog_flags_use_the_command_directory(tmp_path, monkeypatch):
    from woof.offline_child import resolve_child_run_config

    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = _child_toml(config_dir, '\n[static]\ngeog_root = "file-geog"\n')
    cfg = resolve_child_run_config(path)
    file_policy = load_child_static_policy(path, cfg)
    flag_policy = load_child_static_policy(path, cfg, geog_root=Path("flag-geog"))
    assert file_policy.geog_root == config_dir / "file-geog"
    assert flag_policy.geog_root == tmp_path / "flag-geog"


def test_only_the_child_route_reads_a_static_table(tmp_path):
    """Any other RunConfig route would drop it, so it refuses it."""
    path = _child_toml(tmp_path, static='\n[static]\nterrain = "own"\n')
    with pytest.raises(ValueError, match="downscaled child"):
        load_config(path)
    assert load_config(path, child_static=True).dx == 1000.0
    bad = _child_toml(tmp_path, static='\n[static]\nsmooth_pases = 3\n')
    with pytest.raises(ValueError, match="smooth_pases"):
        load_config(bad, child_static=True)


# ---------------------------------------------------------------------
# The parent's projection, and the door
# ---------------------------------------------------------------------
#
# Each case below hands woof a parent wrfout and lets it READ the file,
# which woof decodes through the Rust rw_netcdf binary and nothing else
# (NetcdfBridgeMissing otherwise).  The windows-2025 CPU job builds no
# native tools, and all six died there on that refusal (2.8.2 public CI),
# a red for a missing tool rather than a defect.  So each is gated on the
# conftest capability probe, which names how to stage the bridge; the
# rest of this deck writes or reads nothing through it and keeps running
# everywhere, as test_downscale_cli.py gates its own deck per test.

def _lambert_history(path, valid_time, *, ny=18, nx=20):
    """A fixture frame whose XLAT/XLONG ARE its projection's."""
    from woof.static.projection import projection_class

    _history(path, valid_time, ny=ny, nx=nx)
    grid = projection_class("lambert")(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=nx + 1, e_sn=ny + 1)
    lat, lon = grid.latlon_mass()
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.variables["XLAT"][0] = lat.astype(np.float32)
        dataset.variables["XLONG"][0] = lon.astype(np.float32)


@requires_netcdf_bridge
def test_a_parent_whose_coordinates_are_its_projection_is_placed(tmp_path):
    frame = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    _lambert_history(frame, datetime(1974, 4, 3, 12))
    grid, evidence = parent_projected_grid(frame)
    assert grid.e_we == 21 and grid.e_sn == 19
    assert evidence["anchor"].startswith("CEN_LAT/CEN_LON")
    assert max(evidence["lat_error_deg"], evidence["lon_error_deg"]) < 1e-5


@requires_netcdf_bridge
def test_a_downscaled_parent_is_placed_on_its_own_first_point(tmp_path):
    """A child's history carries its parent's CEN_LAT/CEN_LON, as WRF's
    ndown writes them (main/ndown_em.F:446,773); a grandchild still finds
    the grid its coordinates are on."""
    frame = tmp_path / "wrfout_d02_1974-04-03_12_00_00"
    _lambert_history(frame, datetime(1974, 4, 3, 12))
    with netCDF4.Dataset(frame, "a") as dataset:
        dataset.CEN_LAT = np.float32(35.4)
        dataset.CEN_LON = np.float32(-96.7)
    grid, evidence = parent_projected_grid(frame)
    assert evidence["anchor"].startswith("the frame's own first mass point")
    assert max(evidence["lat_error_deg"], evidence["lon_error_deg"]) < 1e-5


@requires_netcdf_bridge
def test_a_parent_that_contradicts_its_projection_is_refused(tmp_path):
    frame = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    _lambert_history(frame, datetime(1974, 4, 3, 12))
    with netCDF4.Dataset(frame, "a") as dataset:
        dataset.variables["XLAT"][0, 4, 5] += 0.01
    with pytest.raises(ChildGeographyError,
                       match="disagrees with its own XLAT/XLONG"):
        parent_projected_grid(frame)


def _door(tmp_path, *extra):
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _lambert_history(
            tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00",
            start + timedelta(hours=index))
    restart = _restart_evidence(
        tmp_path / "gpuwmrst_d01_1974-04-03_12_00_00.npz",
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    return [
        "downscale", str(tmp_path), "--parent-restart", str(restart),
        "--point", "35.0,-97.0", "--ratio", "1", "--child-size", "12,10", "--physics-profile", "parent",
        "--child-levels", "4,2.5", "--hours", "0.25",
        "--output-interval-seconds", "900",
        "--out", str(tmp_path / "child-run"), *extra]


def _plan(captured):
    return json.loads(captured.out[captured.out.index("{"):])


@requires_netcdf_bridge
def test_the_door_builds_the_childs_own_geography_by_default(
        tmp_path, capsys):
    missing = tmp_path / "no-geog"
    assert cli_main(_door(tmp_path, "--geog-root", str(missing),
                          "--dry-run")) == 0
    captured = capsys.readouterr()
    plan = _plan(captured)
    assert plan["child_terrain"]["terrain"] == "own"
    assert plan["child_terrain"]["terrain_policy"] == \
        "child-own-static-geography"
    assert "is not a directory" in plan["child_terrain_problem"]
    assert "--parent-terrain" in plan["child_terrain_problem"]
    assert "child terrain: its own static geography" in captured.out
    config = (tmp_path / "child-run.child.toml").read_text(encoding="utf-8")
    assert '\n[static]\nterrain = "own"\n' in config


@requires_netcdf_bridge
def test_a_child_whose_geography_cannot_be_built_is_refused_first(
        tmp_path, capsys):
    rc = cli_main(_door(tmp_path, "--geog-root", str(tmp_path / "no-geog")))
    captured = capsys.readouterr()
    assert rc == 2
    assert "is not a directory" in captured.err
    assert "--parent-terrain" in captured.err
    # refused before anything was reserved
    assert not (tmp_path / "child-run").exists()


@requires_netcdf_bridge
def test_parent_terrain_keeps_the_route_every_child_ran_before(
        tmp_path, capsys):
    assert cli_main(_door(tmp_path, "--parent-terrain", "--dry-run")) == 0
    captured = capsys.readouterr()
    plan = _plan(captured)
    assert plan["child_terrain"]["terrain"] == "parent"
    assert plan["child_terrain_problem"] is None
    assert "child terrain: the parent's, interpolated" in captured.out
    config = (tmp_path / "child-run.child.toml").read_text(encoding="utf-8")
    assert '\n[static]\nterrain = "parent"\n' in config
