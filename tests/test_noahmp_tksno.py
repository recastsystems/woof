"""opt_tksno: snow thermal conductivity choices, written from published formulas.

Re-implemented from published descriptions, not bitwise to upstream; see
docs/noahmp-opt-tksno.md. The hand values below are worked from each paper's
formula at rho = 250 kg m-3 (a settled seasonal snowpack).
"""
import math

import numpy as np
import pytest
from conftest import requires_gpu

from woof.core.noahmp_tksno import OPT_TKSNO_VALUES, snow_thermal_conductivity, validate

RHO = 250.0
#: Worked by hand from each paper's formula, rho = 250 kg m-3.
HAND = {
    1: 3.2217e-6 * 250.0 ** 2,             # Yen 1965:      0.20135625
    2: 0.02 + 2.5e-6 * 250.0 ** 2,         # Anderson 1976: 0.17625
    3: 0.35,                               # constant
    4: 2.576e-6 * 250.0 ** 2 + 0.074,      # Verseghy 1991: 0.235
    5: 2.22 * 0.25 ** 1.88,                # Yen 1981:      0.1638...
}


def test_hand_computed_values_from_the_papers():
    assert HAND[1] == pytest.approx(0.20135625, abs=1e-12)
    assert HAND[2] == pytest.approx(0.17625, abs=1e-12)
    assert HAND[4] == pytest.approx(0.235, abs=1e-12)
    assert HAND[5] == pytest.approx(2.22 * math.exp(1.88 * math.log(0.25)), abs=1e-12)
    for opt, want in HAND.items():
        got = snow_thermal_conductivity(np.float32(RHO), opt)
        assert got.dtype == np.float32
        assert float(got) == pytest.approx(want, rel=2e-7), opt


def test_option_one_is_the_v4_line_bit_for_bit():
    rho = np.linspace(50.0, 550.0, 101, dtype=np.float32)
    want = (np.float32(3.2217e-6) * np.power(rho, np.float32(2.0))).astype(np.float32)
    assert snow_thermal_conductivity(rho, 1).tobytes() == want.tobytes()


def test_the_formulas_order_where_the_literature_puts_them():
    # Dense wind slab (400 kg m-3): Yen 1965 is the most conductive, then
    # Verseghy, Anderson, Yen 1981, and the constant lowest; at fresh snow
    # (100 kg m-3) the constant is far above every density curve.
    dense = {o: float(snow_thermal_conductivity(np.float32(400.0), o)) for o in OPT_TKSNO_VALUES}
    fresh = {o: float(snow_thermal_conductivity(np.float32(100.0), o)) for o in OPT_TKSNO_VALUES}
    assert dense[1] > dense[4] > dense[2] > dense[5] > dense[3]
    assert all(fresh[3] > fresh[o] for o in (1, 2, 4, 5))


@pytest.mark.parametrize("bad", [0, 6, 1.0, "2", None])
def test_values_without_a_formula_are_refused(bad):
    with pytest.raises(ValueError):
        validate(bad)


def _thermoprop_columns():
    """Four columns: three, two, one and zero snow layers; one urban, one IST=2."""
    n = 6
    dzsnso = np.zeros((n, 7), np.float32)
    snice = np.zeros((n, 3), np.float32)
    snliq = np.zeros((n, 3), np.float32)
    isnow = np.array([-3, -2, -1, 0, -2, -1], np.int32)
    layers_dz = np.array([0.05, 0.10, 0.20], np.float32)
    layers_ice = np.array([9.0, 26.0, 62.0], np.float32)
    layers_liq = np.array([0.0, 1.0, 3.0], np.float32)
    for c, s in enumerate(isnow):
        for k in range(int(s) + 1, 1):
            j = k + 2
            dzsnso[c, j] = layers_dz[j]
            snice[c, j] = layers_ice[j]
            snliq[c, j] = layers_liq[j]
    dzsnso[:, 3:] = np.array([0.1, 0.3, 0.6, 1.0], np.float32)
    snowh = dzsnso[:, :3].sum(axis=1).astype(np.float32)
    snowh[3] = np.float32(0.02)
    stc = np.full((n, 7), 268.0, np.float32)
    stc[5, 3:] = np.float32(272.0)
    return dict(
        dzsnso=dzsnso, snice=snice, snliq=snliq,
        smc=np.full((n, 4), 0.31, np.float32), sh2o=np.full((n, 4), 0.18, np.float32),
        stc=stc, snowh=snowh, dt=np.full(n, 300.0, np.float32),
        smcmax=np.full((n, 4), 0.439, np.float32), csoil=np.full(n, 2.0e6, np.float32),
        quartz=np.full((n, 4), 0.40, np.float32), isnow=isnow,
        ist=np.array([1, 1, 1, 1, 1, 2], np.int32),
        urban_flag=np.array([0, 0, 0, 0, 1, 0], np.int32))


def _host_thermoprop(cols, c, opt):
    from woof.core import noahmp_leaves as leaves
    real_csnow = leaves.csnow

    def csnow(isnow, snice, snliq, dzsnso, **kw):
        tksno, cvsno, snicev, snliqv, epore = real_csnow(isnow, snice, snliq, dzsnso, **kw)
        for k in range(isnow + 1, 1):
            j = k + 2
            rho = np.float32(np.float32(snice[j] + snliq[j]) / np.float32(dzsnso[j]))
            tksno[j] = snow_thermal_conductivity(rho, opt)
        return tksno, cvsno, snicev, snliqv, epore

    leaves.csnow = csnow
    try:
        return leaves.thermoprop(
            int(cols["isnow"][c]), int(cols["ist"][c]), cols["dzsnso"][c], cols["dt"][c],
            cols["snowh"][c], cols["snice"][c], cols["snliq"][c], cols["smc"][c],
            cols["sh2o"][c], cols["stc"][c], cols["smcmax"][c], cols["csoil"][c],
            cols["quartz"][c], bool(cols["urban_flag"][c]))
    finally:
        leaves.csnow = real_csnow


@requires_gpu
@pytest.mark.parametrize("opt", [2, 3, 4, 5])
def test_device_thermoprop_matches_the_paper_formula_and_touches_only_conductivity(opt):
    import cupy as cp
    from woof.core.noahmp_thermal_slab import evaluate_thermoprop_slab

    cols = _thermoprop_columns()
    n = cols["isnow"].shape[0]
    dev = {k: cp.asarray(v) for k, v in cols.items()}
    base = {k: v.get() for k, v in evaluate_thermoprop_slab(dev, n).items()}
    got = {k: v.get() for k, v in evaluate_thermoprop_slab(dev, n, opt_tksno=opt).items()}
    # Only DF moves; heat capacity, fractions and FACT are byte-identical.
    for name in base:
        if name != "df":
            assert got[name].tobytes() == base[name].tobytes(), name
    for c in range(n):
        s = int(cols["isnow"][c])
        df_host = _host_thermoprop(cols, c, opt)[0]
        np.testing.assert_allclose(got["df"][c], df_host, rtol=2e-6, atol=0.0)
        if s == 0:
            assert got["df"][c].tobytes() == base["df"][c].tobytes()
        else:
            # deep soil (layers 2-4) never sees snow conductivity
            assert got["df"][c, 4:].tobytes() == base["df"][c, 4:].tobytes()
            assert not np.array_equal(got["df"][c, s + 3:3], base["df"][c, s + 3:3])


@requires_gpu
@pytest.mark.parametrize("vegetation", ["land", "glacier"])
def test_a_snow_covered_step_runs_every_option_and_option_one_is_the_default(vegetation):
    import cupy as cp
    from test_noahmp_runtime import _build, _GRASSLAND, _ICE, _LOAM
    from woof.core.physics import _prepare_atmosphere

    veg = _GRASSLAND if vegetation == "land" else _ICE
    out = {}
    for opt in (None, 1, 3, 5):
        kw = {} if opt is None else {"opt_tksno": opt}
        state, cfg, driver = _build(nx=6, ny=4, vegtyp=veg, soiltyp=16 if veg == _ICE else _LOAM,
                                    snow_mm=120.0, snow_depth_m=0.45, **kw)
        atmosphere = _prepare_atmosphere(state)
        for _ in range(3):
            driver._run_noahmp(atmosphere, cfg, 3)
        cp.cuda.get_current_stream().synchronize()
        out[opt] = {k: driver.fields[k].get() for k in ("tsnoxy", "tslb", "tsk", "grdflx", "snow", "snowh", "snliqxy")}
    for k in out[None]:
        assert out[1][k].tobytes() == out[None][k].tobytes(), k
        for opt in (3, 5):
            assert np.isfinite(out[opt][k]).all(), (opt, k)
    # The test column is a melting pack (303 K surface), so the snow layers sit
    # at freezing; the conductivity shows up in the soil and in the melt.
    for opt in (3, 5):
        assert any(not np.array_equal(out[opt][k], out[1][k]) for k in out[1]), opt
