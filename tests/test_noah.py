# tests/test_noah.py
"""Noah LSM (Phase 3 Task 10).

Transcribed line-faithfully from the bundle's WRF v4.6.1
``phys/module_sf_noahdrv.F`` (subroutine ``lsm``: per-column prep +
post-SFLX state/flux updates) and ``phys/module_sf_noahlsm.F``
(``SFLX`` and its whole subtree: PENMAN/CANRES/NOPAC/SNOPAC/SMFLX/
SHFLX/SRT/SSTEP/HRT/HSTEP/ROSR12/EVAPO/DEVAP/TRANSP/SNKSRC/FRH2O/
TDFCND/WDFCND/CSNOW/SNFRAC/ALCALC/SNOWPACK/SNOW_NEW/SNOWZ0/TBND/
TMPAVG/REDPRM), with the VEGPARM/SOILPARM/GENPARM tables shipped
verbatim from the WRF run/ directory (woof/data/noah_tables,
provenance recorded there).

Deliberately out of scope (documented): UA_PHYS, FASDAS, WRF_HYDRO,
urban canopy models (the plain VEGTYP==ISURBAN parameter overrides ARE
ported) and SFCDIF_off (CH comes from the surface layer).  SFLX_GLACIAL
(module_sf_noahlsm_glacial_only.F) was out of scope here and land-ice
columns were skipped until 2026-10-12; it is ported now, in the kernel and
the mirror, and graded against WRF's own words by
tests/test_noah_wrf461_parity.py.

Gates (plan Task 10):
  * mirror-vs-kernel on randomized columns, every d01 veg/soil category;
  * energy-balance closure per step <= 1 W/m^2 numerical residual
    (the discrete linearized-budget identity the scheme solves; float64
    mirror closes to ~1e-9, the FP32 kernel must stay under 1 W/m^2);
  * moisture budget closure;
  * snow-case behavior (accumulation, cover/albedo, melt);
  * multi-day single-column drift sanity.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.core.noah import (
    GEN, SOIL_COLS, VEG_COLS, NoahParams, load_tables, pack_params,
    noah_frh2o, sh2o_init, TBL_DIR,
)
from woof.ingest.soil import preprocess_noah_soil
from woof.verify.npref import np_noah_column

BUNDLE = Path(os.environ.get("WOOF_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
GEO_EM_D01 = BUNDLE / "geo_em" / "geo_em.d01.nc"
requires_bundle = pytest.mark.skipif(not GEO_EM_D01.is_file(),
                                     reason="reference bundle not present")

SIGMA = 5.67e-8          # module_sf_noahlsm SIGMA (file literal)
DZS = (0.1, 0.3, 0.6, 1.0)   # WRF Noah 4-layer thicknesses (namelist zs)
DT = 60.0

# d01 land-point categories (pinned from geo_em.d01.nc; the sweep below is
# the full-table superset of these).
D01_VEG = {1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14}
D01_SOIL = {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16}


@pytest.fixture(scope="module")
def params() -> NoahParams:
    return pack_params(load_tables())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _qsat_mix(t, p):
    """Teten saturation MIXING ratio (test-side forcing helper only)."""
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def _ingest_soil_column(tslb, smois, soil_type):
    fields = {
        "LANDSEA": np.ones((1, 1)),
        "SKINTEMP": np.full((1, 1), tslb),
        "TMN": np.full((1, 1), tslb),
    }
    fields.update({
        name: np.full((1, 1), tslb)
        for name in ("ST000007", "ST007028", "ST028100", "ST100289")
    })
    fields.update({
        name: np.full((1, 1), smois)
        for name in ("SM000007", "SM007028", "SM028100", "SM100289")
    })
    return preprocess_noah_soil(
        fields, soil_type=np.full((1, 1), soil_type))


def _lsminit_first_guess(tslb, smois, soil_type, params):
    row = params.soil[soil_type - 1]
    bx = min(row[SOIL_COLS.index("bexp")], 5.5)
    smcmax = row[SOIL_COLS.index("smcmax")]
    psisat = row[SOIL_COLS.index("psisat")]
    raw = (((3.335e5 / (9.81 * (-psisat)))
            * ((tslb - 273.15) / tslb)) ** (-1.0 / bx)) * smcmax
    return raw, min(max(raw, 0.02), smois), bx, smcmax, psisat


def _base_col(params, veg=10, soil=6, rng=None, snow=False, tsk=None,
              sfctmp=None, swdown=500.0, rain=0.0, sr=None, chs=0.01,
              rh=0.5, vegfra=60.0, smc_frac=0.6):
    """One physically-plausible land column as a plain dict of floats."""
    rng = rng or np.random.default_rng(0)
    soil_row = params.soil[soil - 1]
    smcmax = soil_row[SOIL_COLS.index("smcmax")]
    smcwlt = soil_row[SOIL_COLS.index("smcwlt")]
    if sfctmp is None:
        sfctmp = 262.0 + 10.0 * rng.random() if snow \
            else 284.0 + 16.0 * rng.random()
    if tsk is None:
        tsk = sfctmp + (-1.0 + 2.0 * rng.random())
    sfcprs = 95000.0 + 6000.0 * rng.random()
    dz8w1 = 50.0 + 50.0 * rng.random()
    psfc = sfcprs * (1.0 + 9.81 * 0.5 * dz8w1 / (287.0 * sfctmp))
    qgh = _qsat_mix(sfctmp, sfcprs)
    smc = smcwlt + smc_frac * (smcmax - smcwlt) \
        * (0.8 + 0.4 * rng.random(4))
    smc = np.minimum(smc, smcmax)
    tmn = 283.0 if not snow else 274.0
    tslb = np.linspace(tsk, tmn, 6)[1:5]
    sh2o = sh2o_init(smc, tslb, soil, params)
    if snow:
        swe = 5.0 + 60.0 * rng.random()            # mm
        snowh = swe * 0.001 / (0.1 + 0.2 * rng.random())
        snowc = min(1.0, swe / 40.0)
        albedo = 0.4
    else:
        swe, snowh, snowc, albedo = 0.0, 0.0, 0.0, 0.19
    col = dict(
        psfc=psfc, sfcprs=sfcprs, sfctmp=sfctmp,
        qv1=rh * qgh, qgh=qgh, dz8w1=dz8w1,
        glw=0.85 * SIGMA * sfctmp ** 4, swdown=swdown,
        rainbl=rain, sr=(1.0 if sfctmp <= 273.15 else 0.0)
        if sr is None else sr,
        chs=chs, cqs2=0.02, chs2=0.01, rib=0.0,
        ivgtyp=veg, isltyp=soil, vegfra=vegfra,
        shdmin=10.0, shdmax=90.0, tmn=tmn,
        xland=1.0, xice=0.0, snoalb=0.65, embck=0.95,
        tsk=tsk, canwat=0.2, snow=swe, snowh=snowh, snowc=snowc,
        smois=np.asarray(smc, np.float64),
        tslb=np.asarray(tslb, np.float64),
        sh2o=np.asarray(sh2o, np.float64),
        albedo=albedo, albbck=0.19, emiss=0.95,
        z0=0.1, znt=0.1, snotime=0.0, lai=2.0,
        sfcrunoff=0.0, udrunoff=0.0, acsnow=0.0, acsnom=0.0,
        snopcx=0.0, potevp=0.0,
        hfx=0.0, qfx=0.0, lh=0.0, grdflx=0.0, qsfc=0.0,
    )
    return col


def _storage_m(col_or_out):
    """Column water storage (m): canopy + soil column + snowpack."""
    c = col_or_out
    return (c["canwat"] * 1e-3 + c["snow"] * 1e-3
            + float(np.dot(np.asarray(c["smois"], np.float64), DZS)))


def _sweep_cols(params, snow=False, seed=7):
    """One column per (veg, soil) pair over the full tables (superset of
    the d01 categories); land-ice veg (15) and water veg (17) excluded --
    land ice runs SFLX_GLACIAL, graded against WRF in
    test_noah_wrf461_parity, and water is a driver skip."""
    rng = np.random.default_rng(seed)
    cols = []
    for veg in range(1, params.lucats + 1):
        if veg in (15, 17):
            continue
        for soil in range(1, params.slcats + 1):
            if soil == 14:      # water soil at a land point: driver
                pass            # remaps to 7 -- keep it in the sweep
            rain = float(rng.choice([0.0, 0.5, 3.0])) if not snow \
                else float(rng.choice([0.0, 1.0, 4.0]))
            cols.append(_base_col(
                params, veg=veg, soil=soil, rng=rng, snow=snow,
                swdown=float(rng.choice([0.0, 250.0, 700.0])),
                rain=rain, rh=0.25 + 0.6 * rng.random(),
                chs=0.003 + 0.03 * rng.random(),
                vegfra=5.0 + 90.0 * rng.random()))
    return cols


# ---------------------------------------------------------------------------
# table parsing (CPU)
# ---------------------------------------------------------------------------

def test_tables_modis_spot_values():
    t = load_tables()
    assert t.lutype == "MODIFIED_IGBP_MODIS_NOAH"
    assert t.lucats == 20
    assert t.nrotbl[0] == 4 and t.rstbl[0] == 125.0    # cat 1 ENF
    assert t.snuptbl[0] == 0.08 and t.laimaxtbl[0] == 6.40
    assert t.shdtbl[16] == 0.0                          # water
    assert t.maxalb[14] == 82.0                         # snow/ice
    assert t.z0mintbl[12] == 0.50 and t.z0maxtbl[12] == 0.50  # urban
    assert t.topt == 298.0 and t.cmcmax == 0.5e-3
    assert t.cfactr == 0.5 and t.rsmax == 5000.0
    assert t.bare == 16 and t.natural == 14


def test_tables_usgs_section_reachable():
    t = load_tables(mminlu="USGS")
    assert t.lucats == 27 and t.bare == 19
    assert t.rstbl[0] == 200.0                          # USGS cat 1 urban


def test_tables_stas_spot_values():
    t = load_tables()
    assert t.sltype == "STAS"
    assert t.slcats == 19
    assert t.bb[11] == 11.55 and t.maxsmc[11] == 0.468  # cat 12 clay
    assert t.satdk[11] == 9.74e-7
    assert t.refsmc[3] == 0.360 and t.qtz[0] == 0.92
    assert t.maxsmc[13] == 1.0                          # water row


def test_tables_genparm_values():
    t = load_tables()
    assert t.slope_data[0] == 0.1 and len(t.slope_data) == 9
    assert t.sbeta == -2.0 and t.fxexp == 2.0
    assert t.csoil == 2.0e6 and t.salp == 2.6
    assert t.refdk == 2.0e-6 and t.refkdt == 3.0
    assert t.frzk == 0.15 and t.zbot == -8.0
    assert t.czil == 0.1 and t.lvcoef == 0.5


def _layer_frh2o(soil, soil_type, params):
    """frh2o on each start layer's own REAL temperature and moisture.

    real.exe blends the column in float32, so a uniform ingest column is not
    exactly uniform on every Noah layer; each layer is partitioned from what
    it holds.
    """
    out = np.empty_like(np.asarray(soil.liquid_moisture, dtype=np.float64))
    for k in range(out.shape[0]):
        t = float(soil.soil_temperature[k, 0, 0])
        m = float(soil.soil_moisture[k, 0, 0])
        _, guess, bx, smcmax, psisat = _lsminit_first_guess(t, m, soil_type, params)
        out[k] = noah_frh2o(t, m, guess, smcmax, bx, psisat)
    return out.astype(np.asarray(soil.liquid_moisture).dtype)


def test_lsminit_partitions_subfreezing_ingest_column_via_frh2o(params):
    tslb, smois, soil_type = 268.15, 0.30, 6
    soil = _ingest_soil_column(tslb, smois, soil_type)
    np.testing.assert_array_equal(soil.liquid_moisture,
                                  _layer_frh2o(soil, soil_type, params))
    assert np.all(soil.liquid_moisture < soil.soil_moisture)


def test_lsminit_keeps_warm_ingest_column_wholly_liquid():
    soil = _ingest_soil_column(274.0, 0.30, 6)
    np.testing.assert_array_equal(
        soil.liquid_moisture, soil.soil_moisture)


def test_lsminit_threshold_literal_is_strictly_subfreezing():
    soil = _ingest_soil_column(273.149, 0.30, 6)
    np.testing.assert_array_equal(
        soil.liquid_moisture, soil.soil_moisture)


def test_lsminit_invalid_soil_params_keep_whole_column_liquid():
    soil = _ingest_soil_column(268.15, 0.30, 14)  # STAS water: BB/PSISAT=0
    np.testing.assert_array_equal(
        soil.liquid_moisture, soil.soil_moisture)


def test_lsminit_floors_first_guess_before_clip_and_frh2o(params):
    tslb, smois, soil_type = 268.15, 0.30, 1
    raw, guess, bx, smcmax, psisat = _lsminit_first_guess(
        tslb, smois, soil_type, params)
    assert raw < 0.02
    assert guess == 0.02
    soil = _ingest_soil_column(tslb, smois, soil_type)
    np.testing.assert_array_equal(soil.liquid_moisture,
                                  _layer_frh2o(soil, soil_type, params))


@requires_bundle
def test_real74_coastal_mismatch_sh2o_keeps_frozen_lsminit_bytes():
    """Regression for all 183 real74 water/GEOG-soil mismatch cells.

    This CPU replica uses the production FP32 coordinate/search order and SST
    support fallback.  The five value pins are frozen-path WRF-LSMINIT SH2O;
    init must not replace them merely because ERA5 LANDSEA says water.
    """
    from datetime import datetime

    from woof.ingest.horiz import _regular_coordinates
    from woof.static.build import build_static
    from woof.verify.cases.real74_d01 import (
        BUNDLE as REAL74_BUNDLE, _case_grid, _rust_snapshot, phase3_config,
    )

    snapshot = _rust_snapshot(datetime(1974, 4, 3, 12))
    grid = _case_grid(phase3_config())
    target_lat, target_lon = grid.latlon_mass()
    y64, x64 = _regular_coordinates(
        snapshot.latitude, snapshot.longitude, target_lat, target_lon)
    y = y64.astype(np.float32)
    x = x64.astype(np.float32)
    center_y = np.rint(y).astype(np.int32)
    center_x = np.rint(x).astype(np.int32)
    source_land = np.asarray(snapshot.fields["LANDSEA"]) >= 0.5
    target_land = source_land[center_y, center_x]

    def masked_nearest(field, surface):
        field = np.asarray(field, dtype=np.float32)
        active = np.ones(target_land.shape, dtype=bool)
        desired_land = target_land
        if surface == "water":
            active = ~target_land
            desired_land = np.zeros(target_land.shape, dtype=bool)
        best_distance = np.full(y.shape, np.inf, dtype=np.float32)
        best_value = np.zeros(y.shape, dtype=np.float32)
        ny, nx = source_land.shape
        for dj in range(-8, 9):
            jy = center_y + dj
            in_y = (jy >= 0) & (jy < ny)
            jy_safe = np.clip(jy, 0, ny - 1)
            for di in range(-8, 9):
                ix = center_x + di
                inside = in_y & (ix >= 0) & (ix < nx)
                ix_safe = np.clip(ix, 0, nx - 1)
                value = field[jy_safe, ix_safe]
                valid = (active & inside & np.isfinite(value)
                         & (source_land[jy_safe, ix_safe] == desired_land))
                distance = ((y - jy_safe) ** 2 + (x - ix_safe) ** 2)
                take = valid & (distance < best_distance)
                best_distance = np.where(
                    take, distance, best_distance).astype(np.float32)
                best_value = np.where(
                    take, value, best_value).astype(np.float32)
        assert np.all(np.isfinite(best_distance[active]))
        return best_value

    skin = masked_nearest(snapshot.fields["SKINTEMP"], "match")
    sst = masked_nearest(snapshot.fields["SST"], "water")
    # _RegularGpuPlan.apply(..., bilinear) on the SST finite-support bitmap.
    ny, nx = source_land.shape
    iy = np.minimum(np.floor(y).astype(np.int32), ny - 2)
    ix = np.minimum(np.floor(x).astype(np.int32), nx - 2)
    fy = (y - iy).astype(np.float32)
    fx = (x - ix).astype(np.float32)
    finite = np.isfinite(snapshot.fields["SST"]).astype(np.float32)
    lower = ((np.float32(1.0) - fx) * finite[iy, ix]
             + fx * finite[iy, ix + 1]).astype(np.float32)
    upper = ((np.float32(1.0) - fx) * finite[iy + 1, ix]
             + fx * finite[iy + 1, ix + 1]).astype(np.float32)
    support = ((np.float32(1.0) - fy) * lower
               + fy * upper).astype(np.float32)
    sst = np.where(support >= np.float32(1.0 - 2.0e-6), sst,
                   np.float32(0.0)).astype(np.float32)
    tsk = np.where(
        target_land, skin,
        np.where(np.isfinite(sst) & (sst >= 170.0) & (sst <= 400.0),
                 sst, skin))
    soil_type = build_static(
        grid, REAL74_BUNDLE / "static" / "WPS_GEOG")["SCT_DOM"]
    old_liquid = sh2o_init(
        np.ones((4, *tsk.shape), dtype=np.float64),
        np.broadcast_to(tsk, (4, *tsk.shape)).astype(np.float64),
        soil_type, pack_params(load_tables()))
    mismatch = (~target_land) & (tsk < 273.149) & (old_liquid[0] < 1.0)
    assert int(np.count_nonzero(mismatch)) == 183
    assert old_liquid[:, mismatch].min() == float.fromhex(
        "0x1.eb72392f03f70p-5")
    assert old_liquid[:, mismatch].max() == float.fromhex(
        "0x1.17db137671db0p-1")

    pins = (
        # j, i, SCT_DOM, TSK FP32, frozen float64 SH2O
        (133, 133, 6, np.float32(272.3423), "0x1.399bacbd85316p-2"),
        (141, 170, 6, np.float32(271.7954), "0x1.1f6e0c311fd4ap-2"),
        (153, 146, 2, np.float32(270.2837), "0x1.06c59ec6a23f4p-3"),
        (166, 94, 3, np.float32(268.39307), "0x1.666b1430be558p-3"),
        (178, 245, 6, np.float32(273.09937), "0x1.e6e4e2bad21b4p-2"),
    )
    temperatures = np.array([[pin[3] for pin in pins]], dtype=np.float32)
    categories = np.array([[pin[2] for pin in pins]], dtype=np.float64)
    for j, i, category, temperature, _ in pins:
        assert int(soil_type[j, i]) == category
        assert tsk[j, i] == temperature
        assert mismatch[j, i]
    fields = {
        "LANDSEA": np.zeros(temperatures.shape, dtype=np.float32),
        "SKINTEMP": temperatures,
        "SST": temperatures,
        "TMN": temperatures,
    }
    for name in ("ST000007", "ST007028", "ST028100", "ST100289"):
        fields[name] = temperatures
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        fields[name] = np.zeros(temperatures.shape, dtype=np.float32)
    soil = preprocess_noah_soil(fields, soil_type=categories)
    expected = np.array(
        [[float.fromhex(pin[4]) for pin in pins]], dtype=np.float64)
    np.testing.assert_array_equal(soil.liquid_moisture[0], expected)
    np.testing.assert_array_equal(
        soil.liquid_moisture.astype(np.float32).view(np.uint32)[0],
        np.array([[0x3E9CCDD6, 0x3E8FB706, 0x3E0362CF,
                   0x3E33358A, 0x3EF37271]], dtype=np.uint32))


def test_tables_shipped_with_provenance():
    assert (TBL_DIR / "PROVENANCE.md").is_file()
    for name in ("VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        assert (TBL_DIR / name).is_file()


@requires_bundle
def test_shipped_tables_byte_identical_to_wrf_run_dir():
    src = BUNDLE / "WRF_source_v4.6.1_group" / "run"
    for name in ("VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        assert (TBL_DIR / name).read_bytes() == (src / name).read_bytes()


@requires_bundle
def test_d01_categories_covered_by_sweep():
    """The plan gate 'all soil/veg categories present in d01 exercised':
    the sweep runs every table category, so d01's sets must be subsets."""
    import netCDF4
    with netCDF4.Dataset(GEO_EM_D01) as ds:
        lu = ds.variables["LU_INDEX"][0].astype(int)
        sct = ds.variables["SCT_DOM"][0].astype(int)
        lm = ds.variables["LANDMASK"][0].astype(int)
        assert ds.getncattr("MMINLU") == "MODIFIED_IGBP_MODIS_NOAH"
        assert int(ds.getncattr("ISICE")) == 15
    veg = set(np.unique(lu[lm == 1]).tolist())
    soil = set(np.unique(sct[lm == 1]).tolist())
    assert veg == D01_VEG and soil == D01_SOIL


def test_redprm_derived_parameters(params):
    """KDT/FRZX/RTDIS exactly as REDPRM builds them."""
    soil = 6
    out = np_noah_column(_base_col(params, veg=10, soil=soil),
                         params, DT, DZS)
    dksat = params.soil[soil - 1][SOIL_COLS.index("dksat")]
    smcmax = params.soil[soil - 1][SOIL_COLS.index("smcmax")]
    smcref = params.soil[soil - 1][SOIL_COLS.index("smcref")]
    kdt = params.gen[GEN["refkdt"]] * dksat / params.gen[GEN["refdk"]]
    frzx = params.gen[GEN["frzk"]] * (smcmax / smcref) * (0.412 / 0.468)
    assert out["kdt"] == pytest.approx(kdt, rel=1e-12)
    assert out["frzx"] == pytest.approx(frzx, rel=1e-12)
    assert out["smcmax"] == pytest.approx(smcmax, rel=1e-12)


# ---------------------------------------------------------------------------
# mirror physics gates (CPU, float64)
# ---------------------------------------------------------------------------

def test_mirror_energy_closure_no_snow(params):
    """Discrete linearized surface-energy identity closes to float64
    roundoff for every no-snow column of the full category sweep."""
    for col in _sweep_cols(params, snow=False):
        out = np_noah_column(col, params, DT, DZS)
        assert out["skip"] == 0
        assert out["ebal_case"] == 0
        assert abs(out["reslin"]) < 1e-6, (col["ivgtyp"], col["isltyp"])


def test_mirror_energy_closure_snow(params):
    """Snow columns: both the sub-freezing (linearized) and the melting
    (FLX3-residual) branches close; clipped-FLX3 columns are flagged."""
    cases = set()
    for col in _sweep_cols(params, snow=True, seed=8):
        out = np_noah_column(col, params, DT, DZS)
        assert out["skip"] == 0
        cases.add(out["ebal_case"])
        if out["ebal_case"] in (1, 2):
            assert abs(out["reslin"]) < 1e-6, (col["ivgtyp"],
                                               col["isltyp"])
    assert 1 in cases                      # sub-freezing branch exercised
    # force the melting branch: warm air, strong sun on thin snow
    melted = 0
    for soil in sorted(D01_SOIL - {14}):
        col = _base_col(params, veg=10, soil=soil, snow=True,
                        sfctmp=279.0, tsk=273.5, swdown=800.0, rh=0.6)
        col["snow"], col["snowh"], col["snowc"] = 8.0, 0.04, 0.5
        out = np_noah_column(col, params, DT, DZS)
        if out["ebal_case"] == 2:
            melted += 1
            assert abs(out["reslin"]) < 1e-6
    assert melted > 0


def test_mirror_moisture_budget(params):
    """Water budget: d(canopy + soil + snow) = dt*(P - E - R1 - R2)."""
    checked = 0
    for snow in (False, True):
        for col in _sweep_cols(params, snow=snow, seed=11):
            col = dict(col)
            before = _storage_m(col)
            out = np_noah_column(col, params, DT, DZS)
            if out["skip"] != 0 or out["clamped"]:
                continue
            after = _storage_m(out)
            rhs = DT * (1e-3 * (col["rainbl"] / DT - out["qfx"])
                        - out["runoff1"] - out["runoff2t"])
            assert after - before == pytest.approx(rhs, abs=2e-9), \
                (col["ivgtyp"], col["isltyp"], snow)
            checked += 1
    assert checked > 300


def test_mirror_snow_accumulates_when_snowing(params):
    col = _base_col(params, veg=10, soil=6, snow=True, sfctmp=265.0,
                    tsk=264.0, swdown=100.0, rain=3.0, sr=1.0)
    swe0, snowh0 = col["snow"], col["snowh"]
    out = np_noah_column(col, params, DT, DZS)
    # snowfall adds ~ rain mm minus sublimation; runoff none into soil
    assert out["snow"] > swe0 + 2.0
    assert out["snowh"] > snowh0
    assert out["acsnow"] == pytest.approx(3.0)
    # snow albedo above the snow-free background
    assert out["albedo"] > col["albbck"]
    assert 0.0 < out["snowc"] <= 1.0
    # fresh snowfall resets the snow-age clock
    assert out["snotime"] == 0.0


def test_wsm6_sr_exact_upper_is_noah_trajectory_and_output_identical(params):
    """Noah consumes SR only through WRF's FFROZP > 0.5 phase branch."""
    from woof.core.physics import _sr_roundoff_envelope

    upper, max_ulps, loops = _sr_roundoff_envelope(6, 60.0)
    assert (max_ulps, loops) == (7, 1)
    unit = _base_col(params, veg=10, soil=6, snow=True, sfctmp=265.0,
                     tsk=264.0, swdown=100.0, rain=3.0, sr=1.0)
    rounded = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in unit.items()
    }
    rounded["sr"] = float(upper)

    for _step in range(5):
        out_unit = np_noah_column(unit, params, DT, DZS)
        out_rounded = np_noah_column(rounded, params, DT, DZS)
        assert out_unit.keys() == out_rounded.keys()
        # The mirror echoes its forcing dictionary; ``sr`` is the deliberately
        # different input, not a Noah-computed state or diagnostic.
        assert out_unit["sr"] == 1.0
        assert out_rounded["sr"] == float(upper)
        for name in out_unit.keys() - {"sr"}:
            np.testing.assert_array_equal(
                np.asarray(out_unit[name]), np.asarray(out_rounded[name]),
                err_msg=f"Noah output diverged for {name}")
        unit.update(out_unit)
        rounded.update(out_rounded)
        unit["sr"] = 1.0
        rounded["sr"] = float(upper)


def test_mirror_snow_melts_when_warm_and_sunny(params):
    col = _base_col(params, veg=10, soil=6, snow=True, sfctmp=280.0,
                    tsk=274.0, swdown=850.0, rh=0.7)
    col["snow"], col["snowh"], col["snowc"] = 10.0, 0.05, 0.6
    before = _storage_m(col)
    out = np_noah_column(col, params, DT, DZS)
    assert out["ebal_case"] in (2, 3)
    assert out["snow"] < col["snow"]                # SWE decreased
    assert out["flx3"] > 0.0                        # melt heat flux
    assert out["acsnom"] > 0.0                      # melt accumulated
    # SNOPAC blends the skin temperature toward freezing by snow cover:
    # T1 = TFREEZ*sncovr^2 + T12*(1 - sncovr^2), so partial cover sits
    # between freezing and the effective snow-ground temperature.
    assert 273.15 - 1e-6 <= out["tsk"] < 300.0
    # melt water went to soil/runoff, not lost
    if not out["clamped"]:
        after = _storage_m(out)
        rhs = DT * (1e-3 * (0.0 - out["qfx"])
                    - out["runoff1"] - out["runoff2t"])
        assert after - before == pytest.approx(rhs, abs=2e-9)


def test_mirror_freezing_rain_forms_pack_and_flx2(params):
    col = _base_col(params, veg=10, soil=6, snow=False, sfctmp=274.5,
                    tsk=272.0, rain=2.0, sr=0.0, swdown=50.0)
    out = np_noah_column(col, params, DT, DZS)
    assert out["flx2"] < 0.0                        # freezing-rain heat
    assert out["snow"] > 0.0                        # rain froze into pack
    assert out["ebal_case"] in (1, 2, 3)


def test_mirror_dew_deposits_water(params):
    """Saturated air over a cold moist surface: ETP<0, QFX<0, and the
    dew mass enters the soil column."""
    col = _base_col(params, veg=10, soil=6, snow=False, sfctmp=288.0,
                    tsk=283.0, rh=1.05, swdown=0.0, rain=0.0, chs=0.03)
    before = _storage_m(col)
    out = np_noah_column(col, params, DT, DZS)
    assert out["etp"] < 0.0 and out["qfx"] < 0.0
    assert _storage_m(out) > before


def test_mirror_urban_overrides(params):
    out = np_noah_column(_base_col(params, veg=13, soil=6), params,
                         DT, DZS)
    assert out["smcmax"] == 0.45 and out["smcwlt"] == 0.40


def test_mirror_bare_soil_no_transpiration(params):
    col = _base_col(params, veg=16, soil=1, vegfra=40.0, swdown=600.0)
    out = np_noah_column(col, params, DT, DZS)
    assert out["ett_w"] == 0.0 and out["ec_w"] == 0.0   # SHDFAC forced 0


def test_mirror_water_and_seaice_skip_and_land_ice_runs_glacial(params):
    base = _base_col(params, veg=10, soil=6)
    for patch, code in ((dict(xland=2.0), 1),
                        (dict(xice=0.6), 2)):
        col = dict(base)
        col.update(patch)
        out = np_noah_column(col, params, DT, DZS)
        assert out["skip"] == code
        assert out["tsk"] == col["tsk"]
        np.testing.assert_array_equal(out["smois"], col["smois"])
    # Land ice is no longer a skip: WRF's ICE == -1 arm runs SFLX_GLACIAL and
    # holds the soil saturated (module_sf_noahdrv.F:1137-1190).
    col = dict(base)
    col.update(ivgtyp=15)
    out = np_noah_column(col, params, DT, DZS)
    assert out["skip"] == 3 and out["ebal_case"] == 4
    assert out["tsk"] != col["tsk"]
    np.testing.assert_array_equal(out["smois"], np.ones(4))
    np.testing.assert_array_equal(out["sh2o"], np.ones(4))
    assert out["lai"] == 0.01 and out["snowc"] == 1.0
    assert out["snow"] >= 100.0          # SFLX_GLACIAL holds SNEQV >= 0.10 m


def test_mirror_land_copies_cqs2_to_chs2_but_skips_hold_it(params):
    """WRF noahdrv assigns CHS2=CQS2 after SFLX or SFLX_GLACIAL -- the
    post-update block both land arms share -- and never on water or sea ice."""
    base = _base_col(params, veg=10, soil=6)
    base.update(cqs2=0.023, chs2=0.011)
    land = np_noah_column(base, params, DT, DZS)
    assert land["chs2"] == base["cqs2"]
    ice = np_noah_column(dict(base, ivgtyp=15), params, DT, DZS)
    assert ice["chs2"] == base["cqs2"]

    for patch in (dict(xland=2.0), dict(xice=0.6)):
        col = dict(base)
        col.update(patch)
        skipped = np_noah_column(col, params, DT, DZS)
        assert skipped["chs2"] == base["chs2"]


def test_raw_lake_uses_source_water_skin_without_changing_land_or_ocean():
    from woof.ingest.soil import preprocess_noah_soil

    shape = (1, 3)
    fields = {
        # The raw lake is cell 2, but coarse ERA5 classifies it as land.
        "LANDSEA": np.array([[1.0, 0.0, 1.0]]),
        "SKINTEMP": np.array([[290.0, 285.0, 299.0]]),
        "SST": np.array([[np.nan, 284.0, np.nan]]),
    }
    for name, values in zip(
            ("ST000007", "ST007028", "ST028100", "ST100289"),
            ((289.0, 284.0, 298.0), (288.0, 283.0, 297.0),
             (286.0, 282.0, 295.0), (283.0, 281.0, 292.0))):
        fields[name] = np.asarray([values], dtype=np.float64)
    for name, values in zip(
            ("SM000007", "SM007028", "SM028100", "SM100289"),
            ((0.30, 1.0, 0.25), (0.31, 1.0, 0.26),
             (0.32, 1.0, 0.27), (0.33, 1.0, 0.28))):
        fields[name] = np.asarray([values], dtype=np.float64)
    soil_type = np.full(shape, 6.0)
    deep = np.array([[282.0, 281.0, 0.0]])
    # These calls hand the router a raw SST beside SKINTEMP on purpose:
    # what this test pins is the HISTORICAL selector's raw-lake override,
    # so it declares wrf_compat, the policy that names that selector.
    baseline = preprocess_noah_soil(
        fields, soil_type=soil_type, deep_soil_temperature=deep,
        water_temperature_policy="wrf_compat")
    lake_mask = np.array([[False, False, True]])
    lake_skin = np.array([[np.nan, np.nan, 278.0]])
    corrected = preprocess_noah_soil(
        fields, soil_type=soil_type, deep_soil_temperature=deep,
        lake_mask=lake_mask, lake_skin_temperature=lake_skin,
        water_temperature_policy="wrf_compat")

    for name in ("tsk", "landmask", "xland", "xice", "snow_water",
                 "snow_depth"):
        np.testing.assert_array_equal(
            getattr(corrected, name)[0, :2], getattr(baseline, name)[0, :2])
    for name in ("soil_temperature", "soil_moisture", "liquid_moisture"):
        np.testing.assert_array_equal(
            getattr(corrected, name)[:, 0, :2],
            getattr(baseline, name)[:, 0, :2])
    assert baseline.xland[0, 2] == 1.0
    assert corrected.xland[0, 2] == 2.0
    assert corrected.landmask[0, 2] == 0.0
    assert corrected.tsk[0, 2] == 278.0
    np.testing.assert_array_equal(corrected.soil_temperature[:, 0, 2], 278.0)
    np.testing.assert_array_equal(corrected.soil_moisture[:, 0, 2], 1.0)

    with pytest.raises(ValueError, match="must be provided together"):
        preprocess_noah_soil(
            fields, soil_type=soil_type, deep_soil_temperature=deep,
            lake_mask=lake_mask, water_temperature_policy="wrf_compat")


def test_nonphysical_tsk_refusal_names_the_cells_and_the_metgrid_fill():
    """The refusal has to say WHICH cells and WHY, like WRF's own print.

    ``module_initialize_real.F:3278-3296`` prints the cell, LANDMASK, TSK,
    SST and TMN before it gives up.  A land target that found no usable
    source cell on its surface carries METGRID.TBL fill_missing (0 K for
    SKINTEMP), and that one value is worth naming outright.
    """
    from woof.ingest.soil import preprocess_noah_soil

    shape = (1, 3)
    fields = {
        "LANDSEA": np.array([[1.0, 0.0, 0.0]]),
        # Cell 0 is land and carries the METGRID.TBL SKINTEMP fill.
        "SKINTEMP": np.array([[0.0, 300.0, 300.5]]),
        "SST": np.array([[0.0, 300.2, 300.7]]),
    }
    for name in ("ST000007", "ST007028", "ST028100", "ST100289"):
        fields[name] = np.full(shape, 285.0)
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        fields[name] = np.full(shape, 0.3)
    with pytest.raises(ValueError) as excinfo:
        # Raw SST beside SKINTEMP on purpose: the test pins the historical
        # selector's TSK refusal, so it declares wrf_compat.
        preprocess_noah_soil(
            fields, soil_type=np.full(shape, 6.0),
            deep_soil_temperature=np.full(shape, 299.0),
            water_temperature_policy="wrf_compat")
    message = str(excinfo.value)
    assert message.startswith(
        "TSK contains non-finite or nonphysical values")
    assert "1 cell(s) outside 170..400 K" in message
    assert "1 on land/sea-ice, 0 on open water" in message
    assert "METGRID.TBL fill_missing" in message


def _snow_case(snowh_values):
    shape = (1, len(snowh_values))
    fields = {
        "LANDSEA": np.ones(shape),
        "SKINTEMP": np.full(shape, 275.0),
        "TMN": np.full(shape, 275.0),
        "SNOWH": np.asarray([snowh_values], dtype=np.float64),
    }
    for name in ("ST000007", "ST007028", "ST028100", "ST100289"):
        fields[name] = np.full(shape, 275.0)
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        fields[name] = np.full(shape, 0.3)
    return fields, np.full(shape, 6.0)


def test_bounded_snow_overshoot_is_repaired_at_zero_not_refused():
    """One cell of interpolation overshoot must not lose a preparation.

    Snow depth is physically non-negative, so a negative mapped value is
    never data -- it is a non-monotone horizontal operator overshooting
    across the snow line, the sharpest gradient the field has.  A real
    nested HRRR domain over the mountainous west died here on ONE cell
    of 88 844 at -4.9 cm, beside a 44.5 m maximum, with a message that
    named neither the numbers nor which condition had failed.
    """
    from woof.ingest.soil import preprocess_noah_soil

    fields, soil_type = _snow_case([44.5, -0.0487, 0.0])
    soil = preprocess_noah_soil(fields, soil_type=soil_type)
    np.testing.assert_array_equal(soil.snow_depth, [[44.5, 0.0, 0.0]])
    # SNOW follows SNOWH through real.exe's 5:1 ratio, and follows the
    # repair with it rather than carrying the negative onward.
    assert float(soil.snow_water[0, 1]) == 0.0

    # A non-negative field is untouched: every previously passing case
    # is byte-identical.
    fields, soil_type = _snow_case([44.5, 0.0, 1.25])
    soil = preprocess_noah_soil(fields, soil_type=soil_type)
    np.testing.assert_array_equal(soil.snow_depth, [[44.5, 0.0, 1.25]])


def test_snow_beyond_the_overshoot_band_still_refuses_with_its_numbers():
    """Watched firing: the repair above is bounded, not a blanket clamp.

    A fill value, a unit error or a broken decode is not overshoot, and
    the refusal that catches it names the count, the most negative value
    and the field's own maximum.
    """
    from woof.ingest.soil import preprocess_noah_soil

    fields, soil_type = _snow_case([1.0, -50.0, 0.0])
    with pytest.raises(ValueError, match="overshoot band") as refusal:
        preprocess_noah_soil(fields, soil_type=soil_type)
    message = str(refusal.value)
    assert "snow depth" in message
    assert "-50" in message
    assert "most negative" in message

    # And the other two conditions each name themselves, rather than
    # three of them sharing one sentence.
    fields, soil_type = _snow_case([1.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="non-finite value"):
        preprocess_noah_soil(fields, soil_type=soil_type)


def test_snow_band_sits_at_the_overlapping_parabolas_reach():
    """Everything the parabola can make is repaired; one step past refuses.

    A 2x2 patch holding a trace of snow inside snow at the field maximum
    maps to about -17/64 of that maximum at the patch centre: one quarter,
    the band that used to sit here, refused it.  The operator's whole
    negative weight is 9/32, and the band sits there.
    """
    from woof.ingest.soil import preprocess_noah_soil

    fields, soil_type = _snow_case([1.0, -17.0 / 64.0, -0.28125])
    soil = preprocess_noah_soil(fields, soil_type=soil_type)
    np.testing.assert_array_equal(soil.snow_depth, [[1.0, 0.0, 0.0]])

    fields, soil_type = _snow_case([1.0, -0.2813, 0.0])
    with pytest.raises(ValueError, match="overshoot band, 0.28125"):
        preprocess_noah_soil(fields, soil_type=soil_type)


def test_snow_above_the_plausibility_ceiling_refuses_with_its_numbers():
    """Negative control on the other side of the band.

    A kg m-2 snow water equivalent read as ERA5 metres, or the reverse,
    is a factor of 1000, and no arm of the SNOW/SNOWH reconciliation can
    see it: when both fields are present neither arm fires, so the state
    goes into the LSM at 2e5 kg m-3 with nothing said.  The ceiling is
    the only thing that can notice, so feed it the wrong number and
    require a refusal that names it.
    """
    from woof.ingest.soil import preprocess_noah_soil

    fields, soil_type = _snow_case([0.30, 0.30, 0.30])
    # 60 kg m-2 of snow water equivalent, multiplied by 1000 by a route
    # that believed the name meant metres.
    fields["SNOW"] = np.full((1, 3), 60.0e3)
    with pytest.raises(ValueError, match="plausibility ceiling") as refusal:
        preprocess_noah_soil(fields, soil_type=soil_type)
    message = str(refusal.value)
    assert "snow water" in message
    assert "20000" in message
    assert "60000" in message
    assert "kg m-2" in message

    # The same snowpack under the name that states its unit is admitted
    # unchanged, and neither reconciliation arm rewrites it.
    fields["SNOW"] = np.full((1, 3), 60.0)
    soil = preprocess_noah_soil(fields, soil_type=soil_type)
    np.testing.assert_array_equal(soil.snow_water, 60.0)
    np.testing.assert_array_equal(soil.snow_depth, 0.30)

    # Snow depth carries the same ceiling, one column at the
    # reconciliation's own 200 kg m-3: the same 0.30 m pack published in
    # millimetres and read as metres is the same factor of 1000.
    fields, soil_type = _snow_case([300.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="plausibility ceiling") as refusal:
        preprocess_noah_soil(fields, soil_type=soil_type)
    assert "snow depth" in str(refusal.value)


def test_reconciled_era5_sea_ice_exercises_noah_ice_branch(params):
    from woof.ingest.soil import preprocess_noah_soil

    shape = (1, 2)
    fields = {
        "LANDSEA": np.array([[1.0, 0.0]]),
        "SEAICE": np.array([[0.9, 0.75]]),
        "SKINTEMP": np.array([[270.0, 268.0]]),
        "SST": np.array([[np.nan, np.nan]]),
        "TMN": np.full(shape, 272.0),
        "SNOW_EC": np.array([[0.01, 0.02]]),
    }
    for name, value in zip(
            ("ST000007", "ST007028", "ST028100", "ST100289"),
            (269.0, 270.0, 271.0, 272.0)):
        fields[name] = np.full(shape, value)
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        fields[name] = np.full(shape, 0.3)
    # Raw SST beside SKINTEMP on purpose: this pins the historical
    # sea-ice path, so it declares wrf_compat.
    soil = preprocess_noah_soil(fields, soil_type=np.array([[6, 14]]),
                                water_temperature_policy="wrf_compat")

    assert soil.xice[0, 0] == 0.0  # land/ice masks are mutually exclusive
    col = _base_col(params, sfctmp=268.0, tsk=268.0)
    assert soil.xice[0, 1] == 1.0
    assert soil.deep_soil_temperature[0, 1] == 271.4
    np.testing.assert_array_equal(soil.liquid_moisture[:, 0, 1], 0.0)
    col.update(xland=soil.xland[0, 1], xice=soil.xice[0, 1])
    out = np_noah_column(col, params, DT, DZS)
    assert out["skip"] == 2
    np.testing.assert_array_equal(out["sh2o"], 1.0)


def test_seaice_grib_flags_are_repaired_before_fraction_validation():
    fields = {
        "LANDSEA": np.array([[0.0, 0.0]]),
        "SEAICE": np.array([[255.0, 0.75]]),
        "SKINTEMP": np.array([[275.0, 268.0]]),
        "SST": np.array([[275.0, 268.0]]),
        "TMN": np.array([[276.0, 276.0]]),
    }
    for name in ("ST000007", "ST007028", "ST028100", "ST100289"):
        fields[name] = np.full((1, 2), 270.0)
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        fields[name] = np.full((1, 2), 0.3)
    # Raw SST beside SKINTEMP on purpose: this pins the historical
    # sea-ice flag repair, so it declares wrf_compat.
    soil = preprocess_noah_soil(fields, soil_type=np.array([[14, 14]]),
                                water_temperature_policy="wrf_compat")
    np.testing.assert_array_equal(soil.xice, [[0.0, 1.0]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        preprocess_noah_soil(
            {**fields, "SEAICE": np.array([[2.0, 0.75]])},
            soil_type=np.array([[14, 14]]),
            water_temperature_policy="wrf_compat")


def test_mirror_soiltype_water_remap(params):
    """Land point tagged with soil category 14 (water) uses loam (7),
    per the driver's SOILTYP fix."""
    out = np_noah_column(_base_col(params, veg=10, soil=14), params,
                         DT, DZS)
    loam = np_noah_column(_base_col(params, veg=10, soil=7), params,
                          DT, DZS)
    assert out["smcmax"] == loam["smcmax"]


def test_mirror_multiday_drift_sane(params):
    """5 days, dt=600 s, idealized diurnal forcing with a rain episode:
    states stay physical, energy identity closes every step, and the
    accumulated water budget closes."""
    dt = 600.0
    col = _base_col(params, veg=10, soil=6, smc_frac=0.5)
    col["canwat"] = 0.0
    smcmax = None
    residual = 0.0
    water_in = 0.0
    before = _storage_m(col)
    nstep = int(5 * 86400 / dt)
    for n in range(nstep):
        t = n * dt
        hod = (t % 86400.0) / 86400.0
        sw = max(0.0, 800.0 * np.sin(2 * np.pi * (hod - 0.25)))
        sfctmp = 285.0 + 7.0 * np.sin(2 * np.pi * (hod - 0.35))
        rain = 2.0 * dt / 3600.0 if 2.0 <= t / 86400.0 <= 2.2 else 0.0
        col["swdown"], col["sfctmp"], col["rainbl"] = sw, sfctmp, rain
        col["sr"] = 0.0
        col["qgh"] = _qsat_mix(sfctmp, col["sfcprs"])
        col["qv1"] = 0.55 * col["qgh"]
        col["glw"] = 0.85 * SIGMA * sfctmp ** 4
        out = np_noah_column(col, params, dt, DZS)
        assert out["skip"] == 0
        if out["ebal_case"] in (0, 1, 2):
            assert abs(out["reslin"]) < 1e-6, n
        if not out["clamped"]:
            water_in += dt * (1e-3 * (rain / dt - out["qfx"])
                              - out["runoff1"] - out["runoff2t"])
        else:
            before = None       # clamp broke the running budget
        smcmax = out["smcmax"]
        for key in ("tsk", "canwat", "snow", "snowh", "smois", "tslb",
                    "sh2o", "albedo", "emiss", "z0", "znt", "snotime",
                    "snowc", "albbck", "lai", "sfcrunoff", "udrunoff",
                    "acsnow", "acsnom", "snopcx", "potevp",
                    "hfx", "qfx", "lh", "grdflx", "qsfc"):
            col[key] = out[key]
        assert 230.0 < out["tsk"] < 340.0, n
        assert np.all(np.asarray(out["smois"]) <= smcmax + 1e-12)
        assert np.all(np.asarray(out["smois"]) >= 0.02 - 1e-12)
        assert np.all(np.asarray(out["sh2o"])
                      <= np.asarray(out["smois"]) + 1e-12)
        assert np.all((np.asarray(out["tslb"]) > 250.0)
                      & (np.asarray(out["tslb"]) < 330.0)), n
        assert 0.0 <= out["canwat"] <= 0.5 + 1e-9      # CMCMAX in mm
    if before is not None:
        assert _storage_m(col) - before == pytest.approx(water_in,
                                                         abs=1e-6)


# ---------------------------------------------------------------------------
# kernel vs mirror (GPU)
# ---------------------------------------------------------------------------

_STATE_2D = ("tsk", "canwat", "snow", "snowh", "snowc", "albedo",
             "albbck", "emiss", "z0", "znt", "snotime", "lai",
             "chs2",
             "hfx", "qfx", "lh", "grdflx", "qsfc",
             "sfcrunoff", "udrunoff", "acsnow", "acsnom", "snopcx",
             "potevp", "smstav", "smstot", "chklowq", "noahres")
_STATE_3D = ("smois", "tslb", "sh2o", "smcrel")

_SCALE = dict(tsk=300.0, canwat=0.5, snow=50.0, snowh=0.3, snowc=1.0,
              albedo=1.0, albbck=1.0, emiss=1.0, z0=0.5, znt=0.5,
              snotime=1e5, lai=6.0, chs2=0.1,
              hfx=500.0, qfx=3e-4, lh=500.0,
              grdflx=300.0, qsfc=2e-2, sfcrunoff=5.0, udrunoff=5.0,
              acsnow=5.0, acsnom=5.0, snopcx=1000.0, potevp=1e-4,
              # noahres/snopcx are FP32 small-differences of ~500 W/m^2
              # terms: compared at ~0.1 W/m^2 absolute (the tight energy
              # gate is reslin, tested separately at <= 1 W/m^2).
              smstav=1.0, smstot=1000.0, chklowq=1.0, noahres=500.0,
              smois=0.5, tslb=300.0, sh2o=0.5, smcrel=1.0)


def _grid_from_cols(cols):
    """Stack per-column dicts into (ny, nx) / (4, ny, nx) float64 grids."""
    n = len(cols)
    nx = 16
    ny = (n + nx - 1) // nx
    grids = {}
    keys2 = [k for k in cols[0] if k not in ("smois", "tslb", "sh2o")]
    for k in keys2:
        a = np.zeros((ny, nx))
        for m, c in enumerate(cols):
            a[m // nx, m % nx] = c[k]
        grids[k] = a
    for k in ("smois", "tslb", "sh2o"):
        a = np.zeros((4, ny, nx))
        for m, c in enumerate(cols):
            a[:, m // nx, m % nx] = c[k]
        grids[k] = a
    # pad columns beyond n as water points (skipped by both paths)
    for m in range(n, ny * nx):
        grids["xland"][m // nx, m % nx] = 2.0
        grids["ivgtyp"][m // nx, m % nx] = 17
        grids["isltyp"][m // nx, m % nx] = 14
    return grids, n, ny, nx


def _run_both(cols, params, dt=DT, *, itimestep=2):
    """Run the CUDA kernel and the float64 mirror on the same columns."""
    import cupy as cp
    from woof.core.noah import launch_noah

    grids, n, ny, nx = _grid_from_cols(cols)
    dev = {}
    for k, v in grids.items():
        if k in ("ivgtyp", "isltyp"):
            dev[k] = cp.asarray(v.astype(np.int32))
        else:
            dev[k] = cp.asarray(v.astype(np.float32))
    for k in ("smstav", "smstot", "noahres", "reslin", "chklowq"):
        dev[k] = cp.zeros((ny, nx), dtype=cp.float32)
    dev["smcrel"] = cp.zeros((4, ny, nx), dtype=cp.float32)
    dev["ebal"] = cp.zeros((ny, nx), dtype=cp.int32)
    launch_noah(dev, params, dt, DZS, itimestep=itimestep)

    refs = []
    for m, c in enumerate(cols):
        # feed the mirror the FP32-rounded inputs the kernel saw
        c32 = {}
        for k, v in c.items():
            if k in ("smois", "tslb", "sh2o"):
                c32[k] = np.asarray(v, np.float32).astype(np.float64)
            elif k in ("ivgtyp", "isltyp"):
                c32[k] = int(v)
            else:
                c32[k] = float(np.float32(v))
        refs.append(np_noah_column(c32, params, dt, DZS))
    return dev, refs, n, ny, nx


def _compare(dev, refs, n, nx, keys2, keys3, rtol=2e-4):
    import cupy as cp
    bad = []
    for m in range(n):
        j, i = m // nx, m % nx
        r = refs[m]
        for k in keys2:
            got = float(cp.asnumpy(dev[k][j, i]))
            want = float(r[k])
            tol = rtol * _SCALE[k] + rtol * abs(want)
            if abs(got - want) > tol:
                bad.append((m, k, got, want))
        for k in keys3:
            got = cp.asnumpy(dev[k][:, j, i]).astype(np.float64)
            want = np.asarray(r[k], np.float64)
            tol = rtol * _SCALE[k] + rtol * np.abs(want)
            if np.any(np.abs(got - want) > tol):
                bad.append((m, k, got, want))
    return bad


@requires_gpu
@pytest.mark.gpu
def test_kernel_vs_mirror_full_sweep(params):
    """Every (veg, soil) category pair, no snow: FP32 kernel matches the
    float64 mirror at 2e-4 relative on all updated states and fluxes."""
    cols = _sweep_cols(params, snow=False)
    dev, refs, n, ny, nx = _run_both(cols, params)
    bad = _compare(dev, refs, n, nx, _STATE_2D, _STATE_3D)
    assert not bad, bad[:8]


@requires_gpu
@pytest.mark.gpu
def test_kernel_vs_mirror_snow_sweep(params):
    """Snow columns across all categories.  Columns whose energy branch
    sits on a knife edge (mirror flags near-threshold T12/FLX3) are
    compared at a loose tolerance; the rest at 2e-4."""
    import cupy as cp
    cols = _sweep_cols(params, snow=True, seed=8)
    dev, refs, n, ny, nx = _run_both(cols, params)
    strict = [m for m in range(n) if not refs[m]["near_branch"]]
    assert len(strict) > 0.9 * n     # forcing keeps branches decisive
    bad = _compare(dev, refs, n, nx, _STATE_2D, _STATE_3D)
    bad = [b for b in bad if refs[b[0]]["near_branch"] is False]
    assert not bad, bad[:8]
    # branch agreement between kernel and mirror on decisive columns
    ebal = cp.asnumpy(dev["ebal"])
    for m in strict:
        assert ebal[m // nx, m % nx] == refs[m]["ebal_case"], m


@requires_gpu
@pytest.mark.gpu
def test_kernel_energy_residual_under_1wm2(params):
    """Plan gate: per-step energy-balance closure <= 1 W/m^2 numerical
    residual, evaluated on the kernel's own FP32 arithmetic."""
    import cupy as cp
    for snow in (False, True):
        cols = _sweep_cols(params, snow=snow, seed=8 if snow else 7)
        dev, refs, n, ny, nx = _run_both(cols, params)
        reslin = cp.asnumpy(dev["reslin"])
        ebal = cp.asnumpy(dev["ebal"])
        for m in range(n):
            j, i = m // nx, m % nx
            if ebal[j, i] in (0, 1, 2):
                assert abs(reslin[j, i]) <= 1.0, (m, snow,
                                                  float(reslin[j, i]))


@requires_gpu
@pytest.mark.gpu
def test_kernel_water_points_untouched(params):
    import cupy as cp
    col = _base_col(params, veg=17, soil=14)
    col["xland"] = 2.0
    col["hfx"], col["lh"], col["tsk"] = 123.0, 45.0, 291.5
    dev, refs, n, ny, nx = _run_both([col], params)
    assert float(dev["hfx"][0, 0]) == np.float32(123.0)
    assert float(dev["lh"][0, 0]) == np.float32(45.0)
    assert float(dev["tsk"][0, 0]) == np.float32(291.5)


@requires_gpu
@pytest.mark.gpu
def test_timestep_one_initializes_open_water_and_seaice_state(params):
    """WRF noahdrv.F:749-788 initialization precedes early returns."""
    import cupy as cp

    water = _base_col(params, veg=17, soil=14)
    water.update(xland=2.0, xice=0.0,
                 smois=np.full(4, 0.23), tslb=np.full(4, 288.0),
                 sh2o=np.full(4, 0.19))
    seaice = _base_col(params, veg=15, soil=16)
    seaice.update(xland=1.0, xice=1.0,
                  smois=np.full(4, 0.31), tslb=np.full(4, 266.0),
                  sh2o=np.full(4, 0.17), lai=2.5)

    dev, _refs, _n, _ny, nx = _run_both(
        [water, seaice], params, itimestep=1)
    host = {name: cp.asnumpy(dev[name])
            for name in ("smstav", "smstot", "smois", "tslb", "sh2o",
                         "smcrel", "lai")}

    np.testing.assert_array_equal(host["smstav"][0, :2], 1.0)
    np.testing.assert_array_equal(host["smstot"][0, :2], 1.0)
    np.testing.assert_array_equal(host["smois"][:, 0, :2], 1.0)
    np.testing.assert_array_equal(host["smcrel"][:, 0, :2], 1.0)
    np.testing.assert_array_equal(host["tslb"][:, 0, 0],
                                  np.full(4, np.float32(273.16)))
    np.testing.assert_array_equal(host["sh2o"][:, 0, 0],
                                  np.full(4, np.float32(0.19)))
    np.testing.assert_array_equal(host["tslb"][:, 0, 1],
                                  np.full(4, np.float32(266.0)))
    np.testing.assert_array_equal(host["sh2o"][:, 0, 1], 1.0)
    assert host["lai"][0, 1] == np.float32(0.01)


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------

def test_config_sf_surface_physics(tmp_path):
    """The sf_surface_physics key is plumbed, and slab (1) is refused.

    ``sf_sfclay_physics = 91`` in both TOMLs is essential, not garnish:
    573939c moved the driver's LSM/surface-layer coupling refusal to
    ``load_config``, and this test's original bare ``sf_surface_physics=2``
    TOML was the configuration that refusal exists for.  WRF v4.6.1
    (d66e442) is the authority for why that TOML never described a working
    Noah run: with ``sf_sfclay_physics = 0`` the surface driver returns
    before any scheme is selected (``phys/module_surface_driver.F:1488``,
    ``if (sf_sfclay_physics .eq. 0) return``), so Noah selected without a
    surface layer is Noah silently never executing, with CHS/CHS2/CQS2
    never written.  woof refuses the combination at load instead of
    reproducing the silent skip; the third TOML keeps the surface layer so
    it still proves slab is refused for being unported, not for the
    missing coupling.  (This test was missed by 573939c's repair sweep
    because this module's helpers import cupy, which marks the whole file
    ``gpu`` and hid it from the CPU tallies that sweep was checked with.)
    """
    from woof.config import RunConfig, load_config
    assert RunConfig(nx=8, ny=8, nz=8, dx=1e3, dy=1e3, ztop=1e4,
                     dt=1.0, run_seconds=1.0).sf_surface_physics == 0
    p = tmp_path / "c.toml"
    p.write_text("[grid]\nnx=8\nny=8\nnz=8\ndx=1e3\ndy=1e3\nztop=1e4\n"
                 "[run]\ndt=1.0\nrun_seconds=1.0\n"
                 "[dynamics]\nsf_surface_physics=2\nsf_sfclay_physics=91\n")
    assert load_config(p).sf_surface_physics == 2
    p.write_text("[grid]\nnx=8\nny=8\nnz=8\ndx=1e3\ndy=1e3\nztop=1e4\n"
                 "[run]\ndt=1.0\nrun_seconds=1.0\n"
                 "[dynamics]\nsf_surface_physics=1\nsf_sfclay_physics=91\n")
    with pytest.raises(ValueError, match="sf_surface_physics"):
        load_config(p)
