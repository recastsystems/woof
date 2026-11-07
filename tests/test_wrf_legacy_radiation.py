"""CPU gates for the WRF RRTM-LW / Dudhia-SW port lane."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
import shutil
import sys

import numpy as np
import pytest

from woof.config import (RunConfig, radiation_enabled,
                          radiation_scheme_ids, validate_run_config)
from woof.core.dudhia import (DudhiaShortwaveRadiation,
                               dudhia_shortwave_columns,
                               wrf_solar_geometry)
from woof.core.rrtm import (RRTM_DATA_SHA256, load_rrtm_raw_tables,
                             rrtm_default_trace_gases,
                             rrtm_upper_layer_count)


def _cfg(**updates):
    values = dict(nx=2, ny=1, nz=4, dx=1000.0, dy=1000.0,
                  ztop=8000.0, dt=10.0, run_seconds=60.0)
    values.update(updates)
    return RunConfig(**values)


def test_split_radiation_config_preserves_legacy_and_validates_pairs():
    assert radiation_scheme_ids(_cfg(ra_physics=4)) == (4, 4)
    # The split representation still resolves the 1/1 pair exactly -- that is
    # this test's subject and it is unchanged.
    split = _cfg(ra_lw_physics=1, ra_sw_physics=1, icloud=1, swrad_scat=0.7)
    assert radiation_scheme_ids(split) == (1, 1)
    assert radiation_enabled(split)
    # Every independently implemented SW spectrum can retain RRTM LW.
    for shortwave in (0, 1, 4, 90):
        validated = validate_run_config(_cfg(ra_lw_physics=1, ra_sw_physics=shortwave))
        assert radiation_scheme_ids(validated) == (1, shortwave)
    with pytest.raises(ValueError, match="both be explicit"):
        validate_run_config(_cfg(ra_lw_physics=1))
    # The aggregate restated on both streams is ONE selection written
    # twice, so it resolves and validates rather than being refused.
    consistent = validate_run_config(_cfg(
        ra_physics=4, ra_lw_physics=4, ra_sw_physics=4))
    assert radiation_scheme_ids(consistent) == (4, 4)
    # A genuine contradiction still refuses, naming both spellings.
    with pytest.raises(ValueError, match="contradict each other"):
        validate_run_config(_cfg(
            ra_physics=4, ra_lw_physics=1, ra_sw_physics=1))
    assert radiation_scheme_ids(validate_run_config(_cfg(
        ra_lw_physics=4, ra_sw_physics=1))) == (4, 1)


def test_the_aggregate_restated_on_both_streams_is_admitted():
    """One selection written twice is not two selections.

    ra_physics=N means "N on both streams", so restating it as
    ra_lw_physics=N/ra_sw_physics=N adds no information and contradicts
    nothing.  It is the shape the WRF namelist importer's aggregate
    output takes the moment a caller also writes the split fields, and
    every run-path consumer reads the pair through radiation_scheme_ids,
    so the resolved pair and the raw echo in the receipts agree.
    """

    for aggregate in (4, 90):
        cfg = validate_run_config(_cfg(ra_physics=aggregate,
                                       ra_lw_physics=aggregate,
                                       ra_sw_physics=aggregate))
        assert radiation_scheme_ids(cfg) == (aggregate, aggregate)
        assert radiation_enabled(cfg)
    off = validate_run_config(_cfg(ra_physics=0, ra_lw_physics=0,
                                   ra_sw_physics=0))
    assert radiation_scheme_ids(off) == (0, 0)
    assert not radiation_enabled(off)


def test_the_two_radiation_spellings_disagreeing_is_refused():
    """A disagreement has no defined answer, so it is the refusal.

    Each case below writes an aggregate that names one engine beside a
    split pair that names another; the run would take the split pair
    while the receipts echoed the aggregate.  The message says which two
    values disagree and gives the way out: set one spelling.
    """

    for aggregate, lw, sw in ((4, 1, 1), (4, 4, 1), (4, 0, 4), (90, 4, 4)):
        with pytest.raises(ValueError, match="contradict each other") as raised:
            radiation_scheme_ids(_cfg(ra_physics=aggregate,
                                      ra_lw_physics=lw, ra_sw_physics=sw))
        message = str(raised.value)
        assert f"ra_physics={aggregate}" in message
        assert f"ra_lw_physics={lw}/ra_sw_physics={sw}" in message
        assert "ra_physics=0" in message


def test_rrtm_data_exact_record_inventory_and_known_coefficients():
    tables = load_rrtm_raw_tables()
    assert tables.sha256 == RRTM_DATA_SHA256
    assert len(tables.arrays) == 42
    assert len(tables.record_bytes) == 16
    assert sum(tables.record_bytes) == 749120
    assert tables.nbytes == 749120
    assert tables.arrays["abscoefL1"].shape == (5, 13, 16)
    assert tables.arrays["abscoefH8"].shape == (5, 53, 16)
    assert tables.arrays["abscoefL16"].shape == (9, 5, 13, 16)
    # First payload bytes after the first big-endian record marker.
    assert tables.arrays["abscoefL1"][0, 0, 0] == np.float32(0.2972800)
    assert all(not value.flags.writeable for value in tables.arrays.values())


def test_rrtm_data_rejects_corrupt_fortran_record_marker(tmp_path):
    source = load_rrtm_raw_tables().source_path
    damaged = tmp_path / "RRTM_DATA"
    shutil.copyfile(source, damaged)
    with damaged.open("r+b") as stream:
        stream.seek(19844)  # record-one trailing marker
        stream.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="marker mismatch"):
        load_rrtm_raw_tables(damaged)


def test_rrtm_setup_policies_match_wrf_v461():
    assert rrtm_upper_layer_count(10000.0) == 25
    assert rrtm_upper_layer_count(5000.0) == 13  # NINT(12.5) -> 13
    gases = rrtm_default_trace_gases(1974)
    # The REAL(8) values RRTMLWRAD holds: float32 arithmetic widened
    # (tests/test_rrtm_lw_wrf461_parity.py grades them against the Fortran).
    assert gases == pytest.approx({
        "co2": 333.5068493173175e-6,
        "n2o": 319.0e-9,
        "ch4": 1774.0e-9,
    }, rel=1.0e-7)
    assert gases["n2o"] == float(np.float32(319.0e-9))
    assert gases["ch4"] == float(np.float32(1774.0e-9))
    assert gases["co2"] == float(np.float32(gases["co2"]))


def test_rrtm_asset_is_bound_by_active_restart_identity():
    from woof.io import restart

    cfg = _cfg(ra_lw_physics=1, ra_sw_physics=1)
    driver = SimpleNamespace(radiation_callable=None, cumulus_callable=None)
    assets = restart._active_asset_identity(cfg, driver)
    assert assets["wrf_rrtm_data"]["bytes"] == 749248
    assert assets["wrf_rrtm_data"]["sha256"] == RRTM_DATA_SHA256


def test_wrf_solar_geometry_pins_midpoint_hour_angle_and_solcon():
    # The pinned words are WRF 4.6.1's sun (the host float64 mirror); the
    # WRF 4.8 default is computed on the device and graded there
    # (tests/test_solar_generation_gpu.py).
    mu, solcon = wrf_solar_geometry(
        datetime(1974, 4, 3, 18, 12), np.asarray([39.0]),
        np.asarray([-87.0]), hour_offset_seconds=360.0,
        generation="wrf_461")
    assert float(mu[0]) == pytest.approx(0.8238347270070797, abs=1.0e-14)
    assert solcon == pytest.approx(1369.697368038525, abs=1.0e-12)


def _scalar_swpara(t, p, qv, qc, qr, qi, qs, qg, dz, mu, *, solcon,
                   swrad_scat=1.0):
    """Independent scalar mirror of module_ra_sw.F:SWPARA."""
    albtab = np.asarray([
        [0, 0, 0, 0], [69, 58, 40, 15], [90, 80, 70, 60],
        [94, 90, 82, 78], [96, 92, 85, 80]], np.float64).T
    abstab = np.asarray([
        [0, 0, 0, 0], [0, 2.5, 4, 5], [0, 2.6, 7, 10],
        [0, 3.3, 10, 14], [0, 3.7, 10, 15]], np.float64).T
    rho = p / (287.0 * t)
    xwvp = rho * qv * dz * 1000.0
    xatp = rho * dz
    xlwp = rho * 1000.0 * dz * (
        qc + 0.1 * qi + 0.05 * qr + 0.02 * qs + 0.05 * qg)
    sdown = [solcon * mu]
    heat = []
    ww = uv = oldalb = oldabc = totabs = 0.0
    dsca = dabs = dscld = dabsa = 0.0
    xmuval = (0.0, 0.2, 0.5, 1.0)
    for k in range(len(t)):
        ww += xlwp[k]
        uv += xwvp[k]
        wgm = ww / mu
        ugcm = uv * 0.0001 / mu
        oldabs = totabs
        totabs = 2.9 * ugcm / ((1.0 + 141.5 * ugcm) ** 0.635
                              + 5.925 * ugcm)
        xsca = swrad_scat * 1.0e-5 * xatp[k] / mu
        xabs = ((totabs - oldabs) * (sdown[0] - dscld - dsca - dabsa)
                / sdown[k])
        xabs = max(xabs, 0.0)
        alw = min(np.log10(wgm + 1.0), 3.999)
        iil = max(i for i in range(3) if mu > xmuval[i])
        wx = (mu - xmuval[iil]) / (xmuval[iil + 1] - xmuval[iil])
        jjl = int(alw)
        wy = alw - jjl

        def interp(table):
            return ((1.0 - wx) * (1.0 - wy) * table[iil, jjl]
                    + wx * (1.0 - wy) * table[iil + 1, jjl]
                    + (1.0 - wx) * wy * table[iil, jjl + 1]
                    + wx * wy * table[iil + 1, jjl + 1])

        alba, absc = interp(albtab), interp(abstab)
        xalb = max((alba - oldalb) * (sdown[0] - dsca - dabs)
                   / sdown[k], 0.0)
        xabsc = max((absc - oldabc) * (sdown[0] - dsca - dabs)
                    / sdown[k], 0.0)
        dscld += (xalb + xabsc) * sdown[k] * 0.01
        dsca += xsca * sdown[k]
        dabs += xabs * sdown[k]
        oldalb, oldabc = alba, absc
        trans = 100.0 - xalb - xabsc - 100.0 * xabs - 100.0 * xsca
        if trans < 1.0:
            ff = 99.0 / (xalb + xabsc + 100.0 * xabs + 100.0 * xsca)
            xabsc *= ff
            xabs *= ff
            trans = 1.0
        heat.append(sdown[k] * (xabsc + 100.0 * xabs) * 0.01
                    / (rho[k] * 1004.5 * dz[k]))
        sdown.append(max(1.0e-9, sdown[k] * trans * 0.01))
    return np.asarray(heat), sdown[-1]


def test_dudhia_columns_match_independent_wrf_scalar_mirror():
    t = np.asarray([[225.0, 245.0, 270.0, 292.0]], np.float64)
    p = np.asarray([[18000.0, 35000.0, 65000.0, 92000.0]], np.float64)
    qv = np.asarray([[0.0002, 0.001, 0.004, 0.009]], np.float64)
    qc = np.asarray([[0.0, 2.0e-5, 4.0e-4, 1.0e-4]], np.float64)
    qr = np.asarray([[0.0, 0.0, 1.0e-4, 3.0e-4]], np.float64)
    qi = np.asarray([[2.0e-5, 8.0e-5, 0.0, 0.0]], np.float64)
    qs = np.asarray([[1.0e-5, 5.0e-5, 0.0, 0.0]], np.float64)
    qg = np.asarray([[0.0, 1.0e-5, 3.0e-5, 0.0]], np.float64)
    dz = np.asarray([[1500.0, 1400.0, 1200.0, 600.0]], np.float64)
    mu = np.asarray([0.63], np.float64)
    heating, swdown, gsw = dudhia_shortwave_columns(
        t, p, qv, qc, qr, qi, qs, qg, dz, mu, np.asarray([0.18]),
        solcon=1369.7)
    expected_h, expected_sw = _scalar_swpara(
        *[value[0] for value in (t, p, qv, qc, qr, qi, qs, qg, dz)],
        mu[0], solcon=1369.7)
    np.testing.assert_allclose(heating[0], expected_h, rtol=2.0e-15,
                               atol=1.0e-18)
    assert swdown[0] == pytest.approx(expected_sw, rel=2.0e-15)
    assert gsw[0] == pytest.approx(0.82 * expected_sw, rel=2.0e-15)


def test_dudhia_night_is_exact_zero():
    layer = np.ones((2, 3), np.float32)
    heating, swdown, gsw = dudhia_shortwave_columns(
        270.0 * layer, 70000.0 * layer, 0.005 * layer,
        *(np.zeros_like(layer) for _ in range(5)), 1000.0 * layer,
        np.asarray([-0.1, 0.0], np.float32),
        np.asarray([0.2, 0.5], np.float32), solcon=1370.0)
    assert np.array_equal(heating, np.zeros_like(heating))
    assert np.array_equal(swdown, np.zeros_like(swdown))
    assert np.array_equal(gsw, np.zeros_like(gsw))


class _NumpyCupy:
    float32 = np.float32
    int32 = np.int32
    asarray = staticmethod(np.asarray)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    zeros_like = staticmethod(np.zeros_like)


def test_dudhia_adapter_enters_production_driver_radiation_seam(
        monkeypatch):
    """Atmosphere -> stock adapter -> RadiationResult -> held driver seam."""
    import woof.core.physics as physics

    monkeypatch.setitem(sys.modules, "cupy", _NumpyCupy)
    monkeypatch.setattr(physics, "cp", np)
    adapter = DudhiaShortwaveRadiation(
        datetime(1974, 4, 3, 18), np.asarray([[39.0, 39.1]], np.float32),
        np.asarray([[-87.0, -86.9]], np.float32))
    shape = (4, 1, 2)
    atmosphere = {
        "temperature": np.full(shape, 275.0, np.float32),
        "pressure": np.asarray([25000, 45000, 70000, 92000], np.float32)
                    [:, None, None] * np.ones(shape, np.float32),
        "qv": np.full(shape, 0.004, np.float32),
        "qc": np.zeros(shape, np.float32),
        "qi": np.zeros(shape, np.float32),
        "dz": np.full(shape, 1000.0, np.float32),
        "exner": np.ones(shape, np.float32),
    }
    state = SimpleNamespace(
        elapsed_seconds=0.0, p=np.zeros(shape, np.float32))
    # A NumPy-shim seam test: it selects the host-computable WRF 4.6.1 sun
    # (the WRF 4.8 default runs on the device only).
    cfg = _cfg(ra_lw_physics=0, ra_sw_physics=1, radt_minutes=12.0,
               solar_generation="wrf_461")
    driver = object.__new__(physics.PhysicsDriver)
    # No cumulus scheme, so no cumulus radiation feedback (_run_radiation reads it).
    driver.cumulus_callable = None
    driver._wrf_rthraten = None
    driver.rthraten = np.zeros(shape, np.float32)
    driver.swint = None
    driver.radiation_callable = adapter
    # Dudhia is shortwave-only, so it never declares publishes_olr and a
    # real driver leaves the OLR slot empty.  Stated explicitly because a
    # hand-built driver has to declare every slot the seam reads.
    driver.olr = None
    # Dudhia computes no cloud fraction, so no CLDFRA buffer (RULINGS 1).
    driver.cldfra = None
    # No aer_opt = 1 or 2 is selected, so no aerosol diagnostic slots
    # (golden/ports-rad: the seam reads them like the other optional slots).
    driver.aod5503d = None
    driver.aerosol2d = None
    driver.aodtot = None
    # The fixture leaves swint_opt disabled, as the real driver's optional slot.
    driver.swint = None
    driver.fields = {
        "albedo": np.full((1, 2), 0.2, np.float32),
        "glw": np.full((1, 2), 300.0, np.float32),
        "swdown": np.zeros((1, 2), np.float32),
    }
    # The carrier contract is a slot the radiation seam writes, so a
    # hand-built driver declares it like every other one.  Seeded the way
    # initialize_physics seeds a shortwave-only run: the 300 W m-2 here is
    # the CALLER'S declared constant, and the assertion below that it
    # survives the call is the same statement in numbers.
    from woof.core import radiation_carriers
    driver.carriers = radiation_carriers.CarrierContract()
    driver.carriers.declare(
        "glw", source=radiation_carriers.CARRIER_SOURCE_DECLARED_CONSTANT)
    driver.surface_moisture_ledger = None
    sentinel = object()
    captured = {}

    def couple(_state, _cfg, *, rtheta, **_kwargs):
        captured["rtheta"] = rtheta.copy()
        return sentinel

    monkeypatch.setattr(physics, "couple_column_tendencies", couple)
    driver._run_radiation(atmosphere, state, cfg)
    assert driver.radiation_tendencies is sentinel
    assert np.array_equal(driver.rthratenlw, np.zeros(shape, np.float32))
    assert np.any(driver.rthratensw > 0.0)
    assert np.array_equal(captured["rtheta"], driver.rthratensw)
    assert np.all(driver.fields["swdown"] > 0.0)
    assert np.array_equal(driver.fields["glw"],
                          np.full((1, 2), 300.0, np.float32))
    assert adapter.update_count == 1
    # The shortwave half went through a scheme; the declared longwave did
    # not, and the record says so rather than crediting Dudhia with a
    # longwave it never computed.
    assert (driver.carriers.source("swdown")
            == radiation_carriers.CARRIER_SOURCE_RADIATION_SCHEME)
    assert (driver.carriers.source("glw")
            == radiation_carriers.CARRIER_SOURCE_DECLARED_CONSTANT)
    from woof.io import restart
    identity = restart._radiation_setup_identity(driver, cfg)
    assert identity["scheme_ids"] == {"lw": 0, "sw": 1}
    assert identity["algorithms"]["sw"] == \
        "wrf-v4.6.1-dudhia-shortwave-v1"
    assert identity["dudhia"] == {
        "swrad_scat": 1.0,
        "icloud": 1,
        "supported_path": "no-chem,no-eclipse,no-slope",
        "oracle": "WRF-v4.6.1 phys/module_ra_sw.F:SWRAD/SWPARA",
    }


def test_rrtm_pair_resolves_to_the_composed_adapter(monkeypatch):
    """The 1/1 pair binds to RRTM+Dudhia, and to nothing adjacent.

    This asserted a NotImplementedError while the longwave solver was
    unfinished.  The port landed, so the subject changes from "refuses"
    to "resolves to exactly the right thing": the resolver must reach
    RRTMDudhiaRadiation and must not fall through to the RTE+RRTMGP or
    analytic adapters, which is the substitution the old refusal existed
    to prevent.
    """
    import woof.core.physics as physics
    from woof.core.rrtm_lw import RRTMDudhiaRadiation

    cfg = _cfg(ra_lw_physics=1, ra_sw_physics=1)
    monkeypatch.setattr(physics, "physics_driver_required", lambda _cfg: True)
    built = {}

    class _Recorded(RRTMDudhiaRadiation):
        def __post_init__(self):
            built["adapter"] = self

    monkeypatch.setattr("woof.core.rrtm_lw.RRTMDudhiaRadiation", _Recorded)
    # The resolver runs before any allocation the driver needs, so a bare
    # namespace state gets us past the binding and no further.
    with pytest.raises(AttributeError):
        physics.initialize_physics(
            SimpleNamespace(), cfg,
            radiation_start_time=datetime(1974, 4, 3, 12),
            radiation_latitude=np.zeros((1, 2), np.float32),
            radiation_longitude=np.zeros((1, 2), np.float32))
    adapter = built.get("adapter")
    assert isinstance(adapter, RRTMDudhiaRadiation), built
    assert adapter.icloud == cfg.icloud
    assert adapter.swrad_scat == cfg.swrad_scat


def _dudhia_parity_inputs(ncol=19, nlay=23):
    """Deterministic, non-degenerate FP32 columns for the remote GPU gate."""
    rng = np.random.default_rng(461)
    sigma = np.linspace(0.08, 0.98, nlay, dtype=np.float32)[None, :]
    column_scale = np.linspace(0.97, 1.03, ncol, dtype=np.float32)[:, None]
    temperature = ((214.0 + 79.0 * sigma) * column_scale).astype(np.float32)
    pressure = ((9000.0 + 87000.0 * sigma ** np.float32(1.35))
                * column_scale).astype(np.float32)
    qv = ((1.0e-4 + 0.012 * sigma ** np.float32(2.2))
          * column_scale).astype(np.float32)
    cloud = np.exp(-((sigma - 0.64) / 0.15) ** 2).astype(np.float32)
    ice = np.exp(-((sigma - 0.31) / 0.12) ** 2).astype(np.float32)
    jitter = rng.uniform(0.85, 1.15, size=(ncol, nlay)).astype(np.float32)
    qc = (3.5e-4 * cloud * jitter).astype(np.float32)
    qr = (8.0e-5 * cloud * jitter[:, ::-1]).astype(np.float32)
    qi = (1.4e-4 * ice * jitter).astype(np.float32)
    qs = (5.0e-5 * ice * jitter[:, ::-1]).astype(np.float32)
    qg = (2.5e-5 * cloud * ice * jitter).astype(np.float32)
    dz = np.broadcast_to(
        np.linspace(900.0, 350.0, nlay, dtype=np.float32),
        (ncol, nlay)).copy()
    mu = np.linspace(-0.08, 0.94, ncol, dtype=np.float32)
    albedo = np.linspace(0.06, 0.72, ncol, dtype=np.float32)
    exner = (pressure / np.float32(100000.0)) ** np.float32(287.0 / 1004.5)
    return (temperature, pressure, qv, qc, qr, qi, qs, qg, dz,
            mu, albedo, exner.astype(np.float32))


@pytest.mark.gpu
def test_dudhia_cuda_matches_numpy_fp32_columns():
    """Remote gate: CuPy and NumPy execute the same WRF column recurrence."""
    import cupy as cp

    host = _dudhia_parity_inputs()
    expected = dudhia_shortwave_columns(
        *host[:9], host[9], host[10], solcon=1369.7, exner=host[11],
        icloud=1, swrad_scat=0.83)
    device = tuple(cp.asarray(value) for value in host)
    actual_device = dudhia_shortwave_columns(
        *device[:9], device[9], device[10], solcon=1369.7,
        exner=device[11], icloud=1, swrad_scat=0.83)
    cp.cuda.Stream.null.synchronize()

    assert all(isinstance(value, cp.ndarray) for value in actual_device)
    assert actual_device[0].shape == host[0].shape
    assert actual_device[1].shape == host[9].shape
    assert all(value.dtype == cp.float32 for value in actual_device)
    actual = tuple(cp.asnumpy(value) for value in actual_device)
    np.testing.assert_allclose(
        actual[0], expected[0], rtol=3.0e-4, atol=2.0e-8)
    np.testing.assert_allclose(
        actual[1], expected[1], rtol=5.0e-5, atol=2.0e-3)
    np.testing.assert_allclose(
        actual[2], expected[2], rtol=5.0e-5, atol=2.0e-3)
    # The first two columns are night/twilight and must stay exactly dark.
    assert np.array_equal(actual[0][:2], np.zeros_like(actual[0][:2]))
    assert np.array_equal(actual[1][:2], np.zeros_like(actual[1][:2]))


@pytest.mark.gpu
def test_dudhia_cuda_production_adapter_smoke():
    """Remote gate: real packing, solar geometry, and GPU result contract."""
    import cupy as cp
    from woof.core.physics import RadiationResult

    nz, ny, nx = 18, 4, 7
    bottom_to_top = np.arange(nz, dtype=np.float32)[:, None, None]
    frac = bottom_to_top / np.float32(nz - 1)
    tile = np.ones((nz, ny, nx), np.float32)
    temperature = cp.asarray((292.0 - 73.0 * frac) * tile)
    pressure = cp.asarray(
        (95000.0 * np.exp(-2.0 * frac)).astype(np.float32) * tile)
    qv = cp.asarray((0.011 * np.exp(-4.2 * frac)).astype(np.float32) * tile)
    qc = cp.asarray(
        (3.0e-4 * np.exp(-((frac - 0.42) / 0.16) ** 2)).astype(
            np.float32) * tile)
    qi = cp.asarray(
        (1.2e-4 * np.exp(-((frac - 0.72) / 0.13) ** 2)).astype(
            np.float32) * tile)
    zero = cp.zeros((nz, ny, nx), cp.float32)
    atmosphere = {
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "qc": qc,
        "qi": qi,
        "dz": cp.full((nz, ny, nx), 650.0, cp.float32),
        "exner": (pressure / cp.float32(100000.0))
                 ** cp.float32(287.0 / 1004.5),
    }
    state = SimpleNamespace(
        elapsed_seconds=0.0, qc=qc, qr=zero, qi=qi, qs=zero, qg=zero)
    fields = {
        "albedo": cp.full((ny, nx), 0.19, cp.float32),
        "glw": cp.full((ny, nx), 311.0, cp.float32),
    }
    adapter = DudhiaShortwaveRadiation(
        datetime(1974, 4, 3, 18, 0),
        np.full((ny, nx), 39.0, np.float32),
        np.full((ny, nx), -87.0, np.float32),
        swrad_scat=1.0, icloud=1)
    result = adapter(
        atmosphere=atmosphere, fields=fields, state=state,
        cfg=_cfg(nx=nx, ny=ny, nz=nz, ra_lw_physics=0,
                 ra_sw_physics=1, radt_minutes=12.0))
    cp.cuda.Stream.null.synchronize()

    assert isinstance(result, RadiationResult)
    assert result.rthratenlw.shape == (nz, ny, nx)
    assert result.rthratensw.shape == (nz, ny, nx)
    assert result.swdown.shape == (ny, nx)
    assert result.glw is fields["glw"]
    assert result.gsw.shape == (ny, nx)
    assert result.coszen.shape == (ny, nx)
    assert bool(cp.all(cp.isfinite(result.rthratensw)))
    assert bool(cp.all(cp.isfinite(result.swdown)))
    assert bool(cp.all(cp.isfinite(result.gsw)))
    expected_coszen, _ = wrf_solar_geometry(
        datetime(1974, 4, 3, 18, 0),
        np.full((ny, nx), 39.0, np.float32),
        np.full((ny, nx), -87.0, np.float32),
        hour_offset_seconds=0.5 * 12.0 * 60.0)
    cp.testing.assert_array_equal(
        result.coszen, cp.asarray(expected_coszen, cp.float32))
    assert bool(cp.all(result.rthratenlw == 0.0))
    assert bool(cp.all(result.glw == cp.float32(311.0)))
    assert float(cp.max(result.rthratensw).get()) > 0.0
    assert float(cp.min(result.swdown).get()) > 0.0
    assert adapter.update_count == 1
