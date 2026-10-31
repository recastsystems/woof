"""Urban legend contracts using synthetic data and CPU table readers."""
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.static.highres import (
    landcover_legend, expand_landuse_baseline,
    NLCD_TO_MODIS21_INLAND, CGLC_MODIS_LCZ_TO_MODIS21,
)
from woof.static.highres_production import (
    HighresStaticConfig, resolve_static_highres, static_highres_identity,
    prepared_highres_settings_match,
)
from woof.config import RunConfig
from woof.core.landuse import initialize_landuse
from woof.core.urban_tables import urban_category_set


def test_legend_off_bytes_and_count():
    # 21 (Developed, Open Space) is MODIS grassland without an urban canopy
    # since fix/nlcd-open-space 4b98c266b7 (with one it stays urban type 51).
    expected_nlcd = {11:21,12:15,21:10,22:13,23:13,24:13,31:16,
                     41:4,42:1,43:5,52:7,71:10,81:10,82:12,90:11,95:11}
    expected_cglc = {**{i:i for i in range(1,22)},
                     **{i:13 for i in range(51,62)}}
    for source, expected in (("annual-nlcd", expected_nlcd),
                             ("cglc-modis-lcz", expected_cglc)):
        mapping, count = landcover_legend(source, use_wudapt_lcz=1)
        assert count == 21
        assert json.dumps(mapping).encode() == json.dumps(expected).encode()
        values = np.array([mapping[k] for k in sorted(mapping)], dtype=np.int16)
        assert values.tobytes() == np.array(
            [expected[k] for k in sorted(expected)], dtype=np.int16).tobytes()


def test_urban_crosswalks_and_count():
    cglc, count = landcover_legend("cglc-modis-lcz", sf_urban_physics=1,
                                  use_wudapt_lcz=1)
    assert count == 61
    assert [cglc[n] for n in range(51,62)] == list(range(51,62))
    nlcd, count = landcover_legend("annual-nlcd", sf_urban_physics=1)
    assert count == 61
    assert [nlcd[n] for n in (21,22,23,24)] == [51,51,52,53]
    # Keys come from the table reader, including a changed table inventory.
    assert [nlcd[n] for n in (21,23,24)] == list(urban_category_set(isurban=13).lcz[:3])


def test_mismatch_refusals():
    with pytest.raises(ValueError, match="USING 10 WUDAPT LCZ WITHOUT URBPARM_LCZ.TBL"):
        landcover_legend("cglc-modis-lcz", sf_urban_physics=1)
    with pytest.raises(ValueError, match="different urban parameters"):
        landcover_legend("annual-nlcd", sf_urban_physics=1, use_wudapt_lcz=1)


def test_landuse_guard():
    def run(values, legend):
        values = np.array([values], np.int32)
        return initialize_landuse(values, soil_type=np.full(values.shape, 6),
            landmask=np.ones(values.shape), snow=0., xice=0.,
            valid_time=datetime(2026,7,1), cen_lat=30.,
            mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21,
            isice=15, urban_legend=legend)
    with pytest.raises(ValueError, match="exceeds VEGPARM.TBL category count 20"):
        run([51], False)
    np.testing.assert_array_equal(run(list(range(51,62)), True).ivgtyp,
                                  [list(range(51,62))])
    with pytest.raises(ValueError, match="exceeds VEGPARM.TBL"):
        run([50], True)


def test_identity_and_config_threading():
    off = HighresStaticConfig(True, Path("cache"))
    on = replace(off, sf_urban_physics=1, use_wudapt_lcz=1)
    expected = {"enabled":True,"cache_root":"cache","on_refuse":"error",
                "terrain_source":"auto","fields":"auto","landcover_source":"auto"}
    assert static_highres_identity(off) == expected
    digest = lambda cfg: hashlib.sha256(json.dumps(
        static_highres_identity(cfg), sort_keys=True).encode()).hexdigest()
    assert digest(off) != digest(on)
    assert not prepared_highres_settings_match(static_highres_identity(off), on)
    resolved = resolve_static_highres({"static":{"highres":{"enabled":True,"cache_root":"cache"}}},
        source="synthetic", base_dir=Path('.'),
        run_config=SimpleNamespace(sf_urban_physics=1,use_wudapt_lcz=1))
    assert resolved.sf_urban_physics == 1
    assert resolved.use_wudapt_lcz == 1


def test_baseline_fraction_padding():
    original = np.arange(21*4, dtype=np.float64).reshape(21,2,2)
    baseline = {"LANDUSEF":original,"LU_INDEX":np.ones((2,2))}
    assert expand_landuse_baseline(baseline, 21) is baseline
    expanded = expand_landuse_baseline(baseline, 61)
    assert expanded["LANDUSEF"].shape == (61,2,2)
    assert expanded["LANDUSEF"][:21].tobytes() == original.tobytes()
    assert not expanded["LANDUSEF"][21:].any()


def test_wrfout_category_count():
    from woof.io.wrfout import wrf_global_attrs
    from woof.static.lambert import LambertGrid
    grid = LambertGrid(ref_lat=30.,ref_lon=-100.,truelat1=30.,truelat2=60.,
                       stand_lon=-100.,dx=1000.,dy=1000.,e_we=3,e_sn=3)
    off = wrf_global_attrs(grid, datetime(2026,7,1))
    on = wrf_global_attrs(grid, datetime(2026,7,1),
                         run=RunConfig(nx=2,ny=2,nz=2,dx=1000.,dy=1000.,ztop=1000.,
                                       dt=1.,run_seconds=1.,sf_urban_physics=1))
    assert off["NUM_LAND_CAT"] == 21
    assert on["NUM_LAND_CAT"] == 61


def test_wrfinput_fraction_inventory():
    from woof.ingest import wrfinput as wi
    assert "FRC_URB2D" not in wi.IGNORED_WRFINPUT
    assert "FRC_URB2D" in wi.EXPLICIT_AUXILIARY_WRFINPUT
    assert wi.WRFINPUT_DIMENSIONS["FRC_URB2D"] == ("south_north","west_east")


def test_synthetic_urban_rasters(tmp_path):
    from test_highres_geog import _grid, _write_raster, _bound
    from woof.static.highres import resample_mapped_categories
    for source, raw, target, lcz in (("cglc-modis-lcz",61,61,1),
                                    ("annual-nlcd",24,53,0)):
        path = tmp_path / (source + ".tif")
        _write_raster(path, np.full((16,16),raw,dtype=np.uint8))
        mapping, count = landcover_legend(source, sf_urban_physics=1,
                                          use_wudapt_lcz=lcz)
        fractions = resample_mapped_categories(_bound(path), _grid(), mapping,
                                               category_count=count)
        assert fractions.shape == (61,8,8)
        np.testing.assert_allclose(fractions[target-1], 1., atol=1e-6)
        assert not fractions[12].any()


def test_wrfinput_fraction_returned(tmp_path):
    import netCDF4
    from test_analyzed_scalar_boundaries import _cfg, _input, _read
    cfg = _cfg()
    path = _input(tmp_path / "input.nc", cfg)
    absent = _read(path, cfg)
    assert "FRC_URB2D" not in absent.raw
    with netCDF4.Dataset(path, "a") as ds:
        ds.createVariable("FRC_URB2D", "f4", ("Time","south_north","west_east"))[:] = .375
    present = _read(path, cfg)
    np.testing.assert_array_equal(present.raw["FRC_URB2D"],
                                  np.full((cfg.ny,cfg.nx),.375))
    for name in absent.raw:
        np.testing.assert_array_equal(present.raw[name], absent.raw[name])


def test_production_urban_receipt(tmp_path, monkeypatch):
    from datetime import date
    from test_static_highres_landcover_sources import (
        _stub_categories, _stub_fetch, _noah_baseline, _grid, N, MODIS21_ATTRS,
    )
    from woof.static.highres_production import apply_highres_statics
    from woof.static import highres
    _stub_categories(monkeypatch)
    _stub_fetch(monkeypatch, {}, pixels={"61": 144})
    def categories(source, grid, mapping, *, category_count):
        fractions = np.zeros((category_count,grid.e_sn-1,grid.e_we-1))
        fractions[mapping[61]-1] = 1.
        return fractions
    monkeypatch.setattr(highres, "resample_mapped_categories", categories)
    grid = _grid(40.,-100.,dx=1000.,n=N+1)
    config = HighresStaticConfig(True,tmp_path,sf_urban_physics=1,use_wudapt_lcz=1)
    fields, receipt = apply_highres_statics(_noah_baseline(),grid,config=config,
        domain_id=1,case_date=date(2026,7,1),landuse_attrs=MODIS21_ATTRS)
    assert fields["LANDUSEF"].shape == (61,N,N)
    np.testing.assert_array_equal(fields["LU_INDEX"], np.full((N,N),61.))
    assert receipt["config"]["urban_legend"] == "urban"
    assert receipt["landcover"]["urban_collapse"]["legend"] == "urban"
    assert receipt["landcover"]["urban_collapse"]["crosswalk"]["61"] == 61


def test_the_preparation_doors_read_the_urban_legend_from_the_experiment(tmp_path):
    """Every preparation route (HRRR, GFS, ERA5, the tool doors) reads the
    overlay through load_static_highres with no run configuration.  Before
    the loader read the experiment beside it, an urban run was prepared with
    every LCZ class folded into category 13 -- found on a 750 m HRRR run whose
    history files carried no class 51-61 at all."""
    from woof.static.highres_production import load_static_highres

    root = Path(__file__).resolve().parents[1]
    source = (root / "configs" / "probe_320x256.toml").read_text(encoding="utf-8")
    static = '\n[static.highres]\nenabled = true\ncache_root = "cache"\n'
    assert source.count("[shared]\n") == 1 and "[static" not in source
    # The probe's first mass level is about 9 m up, inside every urban
    # class's canopy, and the loader refuses a UCM there (WRF's own stop,
    # module_sf_urban.F:825; woof.experiment.ucm_first_level_refusal).
    # Lifting its lowest layer to about 80 m (three interfaces moved, the
    # count kept) is what lets this test reach the legend it is about.
    lowest = "1.0, 0.9978, 0.99519, 0.99212, 0.98849,"
    assert source.count(lowest) == 1
    source = source.replace(lowest, "1.0, 0.98900, 0.98880, 0.98860, 0.98849,")

    def load(text):
        path = tmp_path / f"exp{len(list(tmp_path.iterdir()))}.toml"
        path.write_text(text, encoding="utf-8")
        return load_static_highres(path)

    urban = load(source.replace(
        "[shared]\n", "[shared]\nsf_urban_physics = 1\nuse_wudapt_lcz = 1\n")
        + static)
    assert (urban.sf_urban_physics, urban.use_wudapt_lcz) == (1, 1)
    assert static_highres_identity(urban)["urban_legend"] == "urban"
    plain = load(source + static)
    explicit_off = load(source.replace(
        "[shared]\n", "[shared]\nsf_urban_physics = 0\n") + static)
    for cfg in (plain, explicit_off):
        assert (cfg.sf_urban_physics, cfg.use_wudapt_lcz) == (0, 0)
        assert static_highres_identity(cfg) == static_highres_identity(
            HighresStaticConfig(True, cfg.cache_root))


def test_every_land_use_initialisation_says_whether_the_urban_legend_is_on():
    """initialize_landuse admits the LCZ categories only when told the urban
    legend is on.  Four routes (the prepared HRRR tree's physics, the DA
    nested child, the offline child and the wrfinput door) did not say, and
    the first LCZ run on the HRRR tree stopped with "active land IVGTYP 53
    exceeds VEGPARM.TBL category count 20".  Every call site now passes it."""
    import ast

    root = Path(__file__).resolve().parents[1] / "woof"
    silent = []
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).parts[0] == "hex":
            # the hex Noah port's own initialize_landuse (woof/hex/mpasphys), another routine with another signature
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None))
                    == "initialize_landuse"
                    and "urban_legend" not in {k.arg for k in node.keywords}):
                silent.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    assert not silent, silent


# --- A169: the sealed reader reads the urban legend back ------------------

def _sealed(cfg):
    """The echo a seal records, through the JSON a header holds."""
    return json.loads(json.dumps(static_highres_identity(cfg)))


@pytest.mark.parametrize("option,lcz,landcover", [
    (1, 0, "annual-nlcd"), (1, 1, "cglc-modis-lcz"), (3, 1, "cglc-modis-lcz")])
def test_sealed_urban_carrier_reads_back_to_the_carrier_it_sealed(
        tmp_path, option, lcz, landcover):
    """Since the urban merge the echo writes urban_legend and use_wudapt_lcz
    when an urban model runs, and the one sealed-echo reader read neither:
    every sealed HRRR prepared-cache restore and native hierarchy rebuild of
    an urban run with a carrier was refused ("does not have a key
    urban_legend, use_wudapt_lcz")."""
    from woof.static.highres_production import (
        parse_sealed_static_highres, parse_static_table)
    cfg = HighresStaticConfig(True, tmp_path / "cache", fields="all",
                              landcover_source=landcover,
                              sf_urban_physics=option, use_wudapt_lcz=lcz,
                              terrain_smoothing=((1, "1-2-1", 3),))
    echo = _sealed(cfg)
    assert (echo["urban_legend"], echo["use_wudapt_lcz"]) == ("urban", lcz)
    # The table reader alone still refuses them: they are no
    # [static.highres] key.
    # (The refusal lists every unknown key; with lane/sf-landsurface the
    # SLUCM echo also records urban_fraction, listed first.)
    with pytest.raises(ValueError, match="does not have a key .*'urban_legend'"):
        parse_static_table({"highres": {k: v for k, v in echo.items()
                                        if k != "terrain_smoothing"}},
                           source="sealed", base_dir=tmp_path)
    run = SimpleNamespace(sf_urban_physics=option, use_wudapt_lcz=lcz)
    back = parse_sealed_static_highres(echo, source="sealed",
                                       base_dir=tmp_path, run_config=run)
    assert back == cfg
    assert static_highres_identity(back) == static_highres_identity(cfg)
    assert prepared_highres_settings_match(echo, back)
    # Without the run, the legend still reads back.  The identity follows
    # it only for SLUCM (option 1): since ae393b7c3 (lane/sf-landsurface)
    # a SLUCM carrier records its urban_fraction algorithm, so its content
    # names its model, and a carrier is identified by its urban model as
    # well as its legend (coordinator ruling, 2026-10-07).  A BEP/BEM
    # carrier's echo names no model, so a run-less read reads it as SLUCM
    # and its identity differs by exactly that record; restores and
    # rebuilds pass run_config, as above.
    alone = parse_sealed_static_highres(echo, source="sealed", base_dir=tmp_path)
    assert (alone.sf_urban_physics, alone.use_wudapt_lcz) == (1, lcz)
    if option == 1:
        assert static_highres_identity(alone) == static_highres_identity(cfg)
    else:
        differs = {key for key in set(static_highres_identity(alone))
                   | set(static_highres_identity(cfg))
                   if static_highres_identity(alone).get(key)
                   != static_highres_identity(cfg).get(key)}
        assert differs == {"urban_fraction"}, differs
    assert landcover_legend(landcover, sf_urban_physics=alone.sf_urban_physics,
                            use_wudapt_lcz=alone.use_wudapt_lcz) \
        == landcover_legend(landcover, sf_urban_physics=option,
                            use_wudapt_lcz=lcz)


def test_sealed_carrier_without_urban_model_reads_back_unchanged(tmp_path):
    from woof.static.highres_production import parse_sealed_static_highres
    cfg = HighresStaticConfig(True, tmp_path / "cache")
    run = SimpleNamespace(sf_urban_physics=0, use_wudapt_lcz=0)
    for kwargs in ({}, {"run_config": run}):
        assert parse_sealed_static_highres(
            _sealed(cfg), source="sealed", base_dir=tmp_path, **kwargs) == cfg


@pytest.mark.parametrize("sealed,requested,words", [
    ((1, 1), (0, 0), ("Local Climate Zones", "without an urban model")),
    ((0, 0), (1, 1), ("without an urban model", "Local Climate Zones")),
    ((1, 1), (1, 0), ("Local Climate Zones", "URBPARM.TBL")),
    ((2, 0), (3, 1), ("URBPARM.TBL", "Local Climate Zones")),
])
def test_a_restore_under_another_urban_legend_is_refused_by_name(
        tmp_path, sealed, requested, words):
    """The legend decides which land-cover classes a prepared static keeps;
    restoring or rebuilding under another one reads land use the run did not
    ask for, and the whole-carrier comparison refused it without saying
    which setting moved."""
    from woof.static.highres_production import (
        parse_sealed_static_highres, require_sealed_urban_legend)
    cfg = HighresStaticConfig(True, tmp_path / "cache",
                              sf_urban_physics=sealed[0],
                              use_wudapt_lcz=sealed[1])
    run = SimpleNamespace(sf_urban_physics=requested[0],
                          use_wudapt_lcz=requested[1])
    for call in (
            lambda: parse_sealed_static_highres(
                _sealed(cfg), source="the sealed root", base_dir=tmp_path,
                run_config=run),
            lambda: require_sealed_urban_legend(
                _sealed(cfg), run, source="the sealed root")):
        with pytest.raises(ValueError, match="urban legend mismatch") as error:
            call()
        message = str(error.value)
        assert message.index(words[0]) < message.index(words[1])
        assert f"sf_urban_physics = {requested[0]}" in message


def test_a_slucm_carrier_restores_only_under_slucm_on_the_same_legend(tmp_path):
    """A carrier whose content depends on the urban model is identified by
    the model as well as the legend (coordinator ruling, 2026-10-07).

    Since ae393b7c3 (lane/sf-landsurface) a SLUCM carrier (option 1) seals
    its urban_fraction algorithm: the built fractions SLUCM reads.  Reused
    under BEP or BEP+BEM on the same legend those fractions would be read
    wrong, so the reader refuses; under SLUCM it restores.  Carriers of the
    two models whose content does not depend on the model still move
    between them on the same legend.
    """
    from woof.static.highres_production import parse_sealed_static_highres
    slucm = HighresStaticConfig(True, tmp_path / "cache", sf_urban_physics=1,
                                use_wudapt_lcz=1)
    with pytest.raises(ValueError, match="urban fraction algorithm"):
        parse_sealed_static_highres(
            _sealed(slucm), source="sealed", base_dir=tmp_path,
            run_config=SimpleNamespace(sf_urban_physics=3, use_wudapt_lcz=1))
    same = parse_sealed_static_highres(
        _sealed(slucm), source="sealed", base_dir=tmp_path,
        run_config=SimpleNamespace(sf_urban_physics=1, use_wudapt_lcz=1))
    assert (same.sf_urban_physics, same.use_wudapt_lcz) == (1, 1)
    assert static_highres_identity(same) == static_highres_identity(slucm)
    bep = HighresStaticConfig(True, tmp_path / "cache", sf_urban_physics=3,
                              use_wudapt_lcz=1)
    back = parse_sealed_static_highres(
        _sealed(bep), source="sealed", base_dir=tmp_path,
        run_config=SimpleNamespace(sf_urban_physics=2, use_wudapt_lcz=1))
    assert (back.sf_urban_physics, back.use_wudapt_lcz) == (2, 1)
    assert static_highres_identity(back) == static_highres_identity(bep)


@pytest.mark.parametrize("keys", [
    {"urban_legend": "urban"},
    {"use_wudapt_lcz": 1},
    {"urban_legend": "collapsed", "use_wudapt_lcz": 0},
    {"urban_legend": "urban", "use_wudapt_lcz": 2},
    {"urban_legend": "urban", "use_wudapt_lcz": True},
    {"urban_legend": "urban", "use_wudapt_lcz": "1"},
])
def test_sealed_urban_keys_the_carrier_never_writes_are_refused(tmp_path, keys):
    from woof.static.highres_production import parse_sealed_static_highres
    echo = {**_sealed(HighresStaticConfig(True, tmp_path / "cache")), **keys}
    with pytest.raises(ValueError, match="never records"):
        parse_sealed_static_highres(echo, source="sealed", base_dir=tmp_path)


def test_the_restore_and_rebuild_hold_the_sealed_legend_to_the_run():
    """Both sealed readers hand over the run they restore or rebuild: the
    native hierarchy rebuild its tree's root run, the prepared-cache
    preflight the experiment's root run, before the whole-carrier
    comparison."""
    import inspect
    from woof import hrrr_hierarchy_direct, prepared_single_domain_forecast
    rebuild = inspect.getsource(hrrr_hierarchy_direct.prepare_hrrr_hierarchy)
    assert "run_config=native_exp.root.run" in rebuild
    preflight = inspect.getsource(
        prepared_single_domain_forecast.preflight_prepared_forecast)
    named = preflight.index("require_sealed_urban_legend(")
    assert named < preflight.index("prepared_highres_settings_match(\n")


def test_the_native_hrrr_preparation_and_its_static_builder_pick_one_legend(tmp_path):
    """The native HRRR benchmark resolved the carrier without the run, so an
    urban run's static receipt (built through load_static_highres, which
    reads the experiment) named the urban legend and the benchmark's carrier
    did not: the static was refused and the seal recorded no legend."""
    import inspect
    from tools import hrrr_single_domain_benchmark as benchmark
    from tools import prepare_hrrr_wrf as prepare
    assert "run_config=exp.root.run" in inspect.getsource(benchmark)
    resolved = prepare._resolved_static_highres(
        {"static": {"highres": {"enabled": True, "cache_root": "cache"}}},
        tmp_path / "namelist.input",
        SimpleNamespace(sf_urban_physics=1, use_wudapt_lcz=1))
    assert static_highres_identity(resolved)["urban_legend"] == "urban"


def test_the_native_static_builder_admits_the_urban_legend_and_nothing_else():
    """The builder's own check stopped every urban static at category 21
    ("LU_INDEX is outside the MODIS-Noah categories"); with 61 categories
    the urban classes pass, and a 21-category static still tops out at 21."""
    from tools.hrrr_build_native_static import validate_static
    from woof.ingest.hrrr_target import HrrrTargetDomain
    target = HrrrTargetDomain.legacy_500x500()
    shape = (target.ny, target.nx)
    fields = {name: np.ones(shape) for name in (
        "HGT_M", "LANDMASK", "SCT_DOM", "SOILTEMP", "SNOALB", "MAPFAC_M",
        "F", "E", "COSALPHA")}
    fields.update(SINALPHA=np.zeros(shape), GREENFRAC=np.ones((12,) + shape),
                  LAI12M=np.ones((12,) + shape),
                  MAPFAC_U=np.ones((target.ny, target.nx + 1)),
                  MAPFAC_V=np.ones((target.ny + 1, target.nx)))
    urban = {**fields, "LU_INDEX": np.full(shape, 56.0),
             "LANDUSEF": np.zeros((61,) + shape)}
    assert validate_static(urban, target)["field_count"] == len(urban)
    collapsed = {**urban, "LANDUSEF": np.zeros((21,) + shape)}
    with pytest.raises(ValueError, match="categories 1..21"):
        validate_static(collapsed, target)


def test_an_urban_run_whose_carrier_only_smooths_restores_its_seal(tmp_path):
    """resolve_static_highres gives the run's urban selectors only to a
    carrier the configuration declared or defaulted, so an urban run with a
    d01 terrain smoothing and no [static.highres] seals a disabled carrier
    that records no legend (load_static_highres on the GFS, ERA5 and mapped
    routes, the run's own run on native hrrr).  Holding that seal to the
    run refused every prepared forecast of such a run ("urban legend
    mismatch"), and preparing again wrote the same seal.  A disabled
    carrier replaced no land cover, so no legend shaped its statics."""
    from woof.experiment import build_experiment
    from woof.static.highres_production import (
        load_static_highres, parse_sealed_static_highres,
        require_sealed_urban_legend)
    import tomllib

    root = Path(__file__).resolve().parents[1]
    source = (root / "configs" / "probe_320x256.toml").read_text(encoding="utf-8")
    # The UCM needs the lowest layer out of the canopy (see
    # test_the_preparation_doors_read_the_urban_legend_from_the_experiment).
    lowest = "1.0, 0.9978, 0.99519, 0.99212, 0.98849,"
    assert source.count(lowest) == 1 and "[static" not in source
    source = source.replace(lowest, "1.0, 0.98900, 0.98880, 0.98860, 0.98849,")
    domain = "[[domain]]\ngrid_id = 1\n"
    assert source.count(domain) == 1
    source = source.replace("[shared]\n", "[shared]\nsf_urban_physics = 1\n").replace(
        domain, domain + 'static = { smooth_option = "1-2-1", smooth_passes = 3 }\n')
    path = tmp_path / "urban-smoothed.toml"
    path.write_text(source, encoding="utf-8")
    raw = tomllib.loads(source)
    run = build_experiment(
        {k: v for k, v in raw.items()
         if k not in ("case_data", "fetch", "ingest", "static")},
        source=str(path)).root.run
    assert (run.sf_urban_physics, run.use_wudapt_lcz) == (1, 0)

    for carrier in (load_static_highres(path),
                    resolve_static_highres(raw, source=str(path),
                                           base_dir=tmp_path, run_config=run)):
        assert carrier is not None and not carrier.enabled
        assert carrier.terrain_smoothing == ((1, "1-2-1", 3),)
        sealed = _sealed(carrier)
        assert "urban_legend" not in sealed
        # The prepared-cache preflight: the named check, then the
        # whole-carrier comparison against the experiment it restores.
        require_sealed_urban_legend(sealed, run, source="the prepared cache")
        assert prepared_highres_settings_match(sealed, load_static_highres(path))
        # The native hierarchy rebuild's reader comes back with the carrier
        # it sealed, so a rebuild records the same identity.
        back = parse_sealed_static_highres(
            sealed, source="sealed root preparation", base_dir=tmp_path,
            run_config=run)
        assert back == carrier
        assert static_highres_identity(back) == sealed

    # An enabled carrier is still held to the run's legend.
    enabled = {**sealed, "enabled": True}
    with pytest.raises(ValueError, match="urban legend mismatch"):
        require_sealed_urban_legend(enabled, run, source="the prepared cache")
    with pytest.raises(ValueError, match="urban legend mismatch"):
        parse_sealed_static_highres(enabled, source="sealed", base_dir=tmp_path,
                                    run_config=run)
    # And malformed legend keys are refused whether or not it is enabled.
    with pytest.raises(ValueError, match="never records"):
        require_sealed_urban_legend({**sealed, "urban_legend": "urban"}, run,
                                    source="the prepared cache")


def test_the_sealed_reader_takes_a_run_without_urban_selectors(tmp_path):
    """The native hierarchy rebuild hands parse_sealed_static_highres its
    tree's run, and a run object need not carry the urban selectors (the
    hierarchy tests' stub run does not): reading them as attributes raised
    AttributeError where require_sealed_urban_legend reads them as a run
    with no urban model."""
    from woof.static.highres_production import parse_sealed_static_highres
    cfg = HighresStaticConfig(True, tmp_path / "cache",
                              terrain_smoothing=((2, "1-2-1", 3),))
    back = parse_sealed_static_highres(
        _sealed(cfg), source="sealed root preparation", base_dir=tmp_path,
        run_config=SimpleNamespace())
    assert back == cfg
    urban = replace(cfg, sf_urban_physics=1)
    with pytest.raises(ValueError, match="urban legend mismatch"):
        parse_sealed_static_highres(
            _sealed(urban), source="sealed root preparation",
            base_dir=tmp_path, run_config=SimpleNamespace())
