"""Radar latent heating, lane 6 (design E3): the WOOF options around NOAA's
builder.

* four 15-minute windows per leg hour, each from its own volume;
* per-member dt_cond and threshold draws, deterministic in (seed, member);
* the suppression rule, checked on a built slot, and the strict option
  that makes it hold exactly;
* the PBL extension;
* latent heat nudging, the comparison arm.

Every option is off by default, so a default config is still NOAA's
product (graded bit for bit in ``test_radar_tten_builder.py``).
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.da import radar_tten

FIXTURE = Path(__file__).resolve().parent / "data" / \
    "radar_tten_oracle_fixture.npz"


# ---------------------------------------------------------------- host rules

def test_four_windows_of_an_hour_are_noaas_slots():
    assert radar_tten.window_end_minutes(60.0, 4) == (15.0, 30.0, 45.0, 60.0)
    assert radar_tten.window_end_minutes(30.0, 2) == (15.0, 30.0)
    with pytest.raises(radar_tten.RadarTtenError):
        radar_tten.window_end_minutes(60.0, 0)
    with pytest.raises(radar_tten.RadarTtenError):
        radar_tten.window_end_minutes(0.0, 4)


def test_the_default_config_is_noaas_and_every_option_leaves_it():
    assert radar_tten.RadarTtenConfig().is_noaa()
    for change in ({"pbl_extension": True}, {"strict_suppression": True},
                   {"latent_heat_period_min": 30.0}):
        assert not dataclasses.replace(radar_tten.RadarTtenConfig(),
                                       **change).is_noaa()


def test_member_draws_are_deterministic_bounded_and_distinct():
    base = radar_tten.RadarTtenConfig()
    a, rec_a = radar_tten.member_config(base, 3, 20261001)
    b, rec_b = radar_tten.member_config(base, 3, 20261001)
    assert a == b and rec_a == rec_b
    draws = [radar_tten.member_config(base, m, 20261001)[0]
             for m in range(32)]
    periods = [c.latent_heat_period_min for c in draws]
    offsets = [c.convection_refl_threshold_dbz - 28.0 for c in draws]
    assert min(periods) >= 15.0 and max(periods) <= 30.0
    assert min(offsets) >= -3.0 and max(offsets) <= 3.0
    assert len(set(periods)) == 32
    for c in draws:
        assert c.warm_min_dbz == c.convection_refl_threshold_dbz
        assert not c.is_noaa()
    control, record = radar_tten.member_config(base, None, 20261001)
    assert control is base and record["perturbed"] is False
    other = radar_tten.member_config(base, 3, 20261002)[0]
    assert other != a


def test_window_paths_are_found_by_valid_time_and_a_gap_is_refused(tmp_path):
    start = datetime(2026, 10, 1, 19, 0)
    for stamp in ("20261001T1915Z", "20261001T1930Z", "20261001T1945Z"):
        (tmp_path / stamp).mkdir()
        (tmp_path / stamp / "ref.f32").write_bytes(b"")
    with pytest.raises(radar_tten.RadarTtenError, match="1 of 4"):
        radar_tten.window_paths(tmp_path, start, 60.0, 4)
    (tmp_path / "20261001T2000Z").mkdir()
    (tmp_path / "20261001T2000Z" / "ref.f32").write_bytes(b"")
    paths = radar_tten.window_paths(tmp_path, start, 60.0, 4)
    assert [p.parent.name for p in paths] == [
        "20261001T1915Z", "20261001T1930Z", "20261001T1945Z",
        "20261001T2000Z"]


# ---------------------------------------------------------------- on the card

def _fixture(cp):
    data = np.load(FIXTURE)
    return data, [cp.asarray(data[f"in_{name}"]) for name in
                  ("ref", "t", "p", "q", "h")]


@pytest.mark.gpu
@requires_gpu
def test_noaas_smoothing_heats_observed_clear_air_and_strict_stops_it():
    import cupy as cp

    data, (ref, theta, p, q, h) = _fixture(cp)
    keep = {}
    noaa, receipt = radar_tten.build_tendency(ref, theta, p, q, h,
                                              intermediates=keep)
    sup = receipt["suppression"]
    assert receipt["config"]["noaa_product"] is True
    assert sup["observed_clear_points"] > 0
    # the receipt agrees with the slot itself
    again = radar_tten.suppression_receipt(noaa, keep["ref_cone"])
    assert again == sup
    strict_cfg = radar_tten.RadarTtenConfig(strict_suppression=True)
    strict, strict_receipt = radar_tten.build_tendency(
        ref, theta, p, q, h, strict_cfg)
    assert strict_receipt["suppression"]["holds"] is True
    assert strict_receipt["suppression"]["observed_clear_heated"] == 0
    assert strict_receipt["suppression_before_strict"] == sup
    # strict changes observed-clear points only
    refc = cp.asnumpy(keep["ref_cone"])[:-1]
    clear = (refc > -100.0) & (refc < 0.001)
    a, b = cp.asnumpy(noaa), cp.asnumpy(strict)
    np.testing.assert_array_equal(a[:-1][~clear], b[:-1][~clear])
    np.testing.assert_array_equal(a[-1], b[-1])
    assert np.all(b[:-1][clear] == 0.0)


@pytest.mark.gpu
@requires_gpu
def test_the_pbl_extension_only_lowers_heating_in_deeply_observed_columns():
    import cupy as cp

    data, (ref, theta, p, q, h) = _fixture(cp)
    keep = {}
    noaa, _ = radar_tten.build_tendency(ref, theta, p, q, h,
                                        intermediates=keep)
    cfg = radar_tten.RadarTtenConfig(pbl_extension=True)
    ext, receipt = radar_tten.build_tendency(ref, theta, p, q, h, cfg)
    # an unreachable depth flags no column: NOAA's slot, bit for bit
    never = radar_tten.RadarTtenConfig(pbl_extension=True,
                                       pbl_extension_depth_hpa=1.0e9)
    same, never_receipt = radar_tten.build_tendency(ref, theta, p, q, h,
                                                    never)
    assert never_receipt["pbl_extended_columns"] == 0
    np.testing.assert_array_equal(cp.asnumpy(same).view(np.int32),
                                  cp.asnumpy(noaa).view(np.int32))
    # a reachable one flags columns; a column whose PBL top is below the
    # level-7 floor can then be heated below level 7, never above its top
    assert receipt["pbl_extended_columns"] > 0
    a, b = cp.asnumpy(noaa), cp.asnumpy(ext)
    below_floor = slice(0, 6)                     # Fortran levels 1..6
    # NOAA never heats below level 7 (smoothing is horizontal only)
    assert not ((a[below_floor] > 0) & (a[below_floor] <= 1)).any()
    # the extension heats there only above the column's own PBL top,
    # where the cone fill left coverage
    refc = cp.asnumpy(keep["ref_cone"])[below_floor]
    heated_b = (b[below_floor] > 0) & (b[below_floor] <= 1)
    assert not (heated_b & (refc <= -200.0)).any()


@pytest.mark.gpu
@requires_gpu
def test_windows_build_one_slot_each_and_the_clock_walks_them():
    import cupy as cp
    from test_radar_tten_forcing import _moist_state

    state, cfg = _moist_state(nz=30)
    shape = state.thp.shape
    refs = []
    for n in range(4):
        r = np.full(shape, -99999.0, dtype=np.float32)
        r[10:20, 2:6, 2 + n:4 + n] = 45.0
        r[10:20, 2:6, 7] = -99.0
        refs.append(cp.asarray(r))
    times = radar_tten.window_end_minutes(60.0, 4)
    forcing = radar_tten.build_forcing(state, refs, times)
    assert len(forcing.slots) == 4
    assert forcing.slot_minutes == times
    for receipt in forcing.receipts:
        assert receipt["suppression"]["observed_clear_points"] > 0
    with pytest.raises(radar_tten.RadarTtenError, match="each window"):
        radar_tten.build_forcing(state, refs[:3], times)


@pytest.mark.gpu
@requires_gpu
def test_latent_heat_nudging_scales_the_schemes_own_heating():
    """One microphysics call, the scheme's part played by hand: a known
    heating increment and a known model rain rate, so alpha is known."""
    import cupy as cp
    from test_radar_tten_forcing import _moist_state

    state, cfg = _moist_state(nz=30)
    nz, ny, nx = state.thp.shape
    dt = float(cfg.dt)
    # radar: Z = 200 R^1.6 rain over columns 0-3, observed dry over 4-6,
    # no coverage over 7-9
    ref = np.full((nz, ny, nx), -99999.0, dtype=np.float32)
    rain_mm_h = 8.0
    ref[:, :, 0:4] = np.float32(10.0 * np.log10(200.0 * rain_mm_h ** 1.6))
    ref[:, :, 4:7] = -99.0
    nudging = radar_tten.build_nudging(state, [cp.asarray(ref)], (60.0,))
    rec = nudging.receipts[0]
    assert rec["columns_observed_rain"] == 4 * ny
    assert rec["columns_observed_dry"] == 3 * ny
    assert rec["columns_no_information"] == 3 * ny
    np.testing.assert_allclose(rec["max_rate_mm_h"], rain_mm_h, rtol=1e-4)
    radar_tten.attach(state, nudging, cfg)
    nudging.before_microphysics(state, cfg, dt)
    lh = np.zeros((nz, ny, nx), np.float32)
    lh[5:15] = 0.02                                   # K this step
    state.thp += cp.asarray(lh)
    model_rate = 2.0                                  # mm/h everywhere
    state.scratch((ny, nx), "mp_rainncv")[...] = model_rate * dt / 3600.0
    thp_mp = cp.asnumpy(state.thp).copy()
    qv_mp = cp.asnumpy(state.qv).copy()
    nudging.after_microphysics(state, cfg, dt)
    radar_tten.detach(state)
    inc = cp.asnumpy(state.thp) - thp_mp
    qv_ratio = cp.asnumpy(state.qv) / qv_mp
    heat = lh > 0
    # observed 8 against model 2: alpha capped at 2, increment = lh
    np.testing.assert_allclose(inc[:, :, 0:4][heat[:, :, 0:4]], 0.02,
                               rtol=1e-3)
    # observed dry under model rain: alpha 0.5, increment = -lh / 2
    np.testing.assert_allclose(inc[:, :, 4:7][heat[:, :, 4:7]], -0.01,
                               rtol=1e-3)
    # no coverage and no heating: untouched
    assert not inc[:, :, 7:].any()
    assert not inc[~heat].any()
    # relative humidity kept: warmer air holds more vapour, cooler less
    assert (qv_ratio[:, :, 0:4][heat[:, :, 0:4]] > 1.0).all()
    assert (qv_ratio[:, :, 4:7][heat[:, :, 4:7]] < 1.0).all()
    receipt = nudging.receipt()
    assert receipt["calls_by_slot"] == [1]
    assert receipt["point_steps_nudged_up"] == int(heat[:, :, 0:4].sum())
    assert receipt["point_steps_nudged_down"] == int(heat[:, :, 4:7].sum())


@pytest.mark.gpu
@requires_gpu
def test_a_forcing_stops_at_its_active_minutes_and_releases_the_clamp():
    """A pre-forecast: forced for its first minutes, free after, in one
    integration.  After the end the call is a pass-through: the scheme's
    own increment stands and the case's clamp comes back."""
    import cupy as cp
    from test_radar_tten_forcing import _moist_state

    state, cfg = _moist_state(nz=30, dt=30.0)
    slot = np.full(state.thp.shape, 0.001, dtype=np.float32)
    slot[-1] = 1.0
    forcing = radar_tten.RadarTtenForcing([cp.asarray(slot)], [60.0],
                                          active_minutes=1.0)
    radar_tten.attach(state, forcing, cfg)
    seen = []
    for _ in range(3):
        forcing.before_microphysics(state, cfg, cfg.dt)
        seen.append(forcing.scheme_config(cfg).mp_tend_lim)
        before = cp.asnumpy(state.thp).copy()
        state.thp += np.float32(0.25)              # the scheme's increment
        forcing.after_microphysics(state, cfg, cfg.dt)
        after = cp.asnumpy(state.thp)
        last_increment = after - before
    radar_tten.detach(state)
    assert forcing.calls_by_slot == [2]
    assert forcing.calls_after_active == 1
    assert seen[:2] == [radar_tten.HRRR_MP_TEND_LIM] * 2
    assert seen[2] == cfg.mp_tend_lim
    np.testing.assert_allclose(last_increment, 0.25, rtol=1e-5)
    receipt = forcing.receipt()
    assert receipt["active_minutes"] == 1.0
    assert receipt["calls_after_active"] == 1
    with pytest.raises(radar_tten.RadarTtenError):
        radar_tten.RadarTtenForcing([cp.asarray(slot)], [60.0],
                                    active_minutes=0.0)
