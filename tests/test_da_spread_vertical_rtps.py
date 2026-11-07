"""Physical-column covariance and RTPS over-relaxation checks for Lane 8."""

from dataclasses import replace
import hashlib
import math

import numpy as np
import pytest

from woof.da import perturb
from woof.da.letkf import (
    GridGeometry, GriddedObs, LetkfConfig, LetkfDiagnostics, LetkfError,
    Localization, analyze,
)


@pytest.mark.parametrize("nz,scale", [(24, 6.0), (12, 8.0), (8, 2.0)])
def test_physical_endpoints_have_the_prescribed_separation(nz, scale):
    """A top-bottom pair is nz-1 levels apart, rather than one FFT interval."""
    geometry = perturb.vertical_wrap_correlations(nz, scale)
    halo = math.ceil(4.0 * scale)
    assert geometry["fft_levels"] == nz + 2 * halo
    assert geometry["physical_crop_levels"] == [halo, halo + nz]
    assert geometry["periodic_seam_in_physical_column"] is False
    assert geometry["nearest_periodic_image_lag_levels"] > 8.0 * scale
    for key, lag in (("adjacent_interior", 1),
                     ("half_column", nz // 2),
                     ("top_to_bottom_seam", nz - 1)):
        expected = math.exp(-(lag / scale) ** 2 / 2.0)
        # At the smallest admitted scale, truncating the continuum Gaussian
        # spectrum at Nyquist leaves a sub-1e-8 discrete-spectrum difference.
        assert geometry[key] == pytest.approx(expected, abs=1e-8)


def test_real_draw_has_no_artificial_top_bottom_lock():
    """Thousands of independent horizontal columns expose the old seam."""
    field, info = perturb.gaussian_random_field(
        (24, 64, 64), seed=43, name="vertical_probe", dx_km=1.0, dy_km=1.0,
        length_scale_km=0.0, vertical_scale_levels=6.0, xp=np)
    measured = np.corrcoef(field[0].ravel(), field[-1].ravel())[0, 1]
    assert measured == pytest.approx(
        info["vertical_wrap"]["top_to_bottom_seam"], abs=0.06)
    assert float(np.sqrt(np.mean(field ** 2))) == pytest.approx(1.0, abs=0.05)
    assert field.flags.owndata


def test_padded_white_noise_and_analytic_variance_are_reproducible():
    """The stamp identifies the actual independent-noise domain and crop."""
    shape = (12, 8, 10)
    scale = 8.0
    field, info = perturb.gaussian_random_field(
        shape, seed=7, name="column_probe", dx_km=1.0, dy_km=1.0,
        length_scale_km=0.0, vertical_scale_levels=scale, xp=np)
    noise_shape = tuple(info["noise_shape"])
    generator = np.random.Generator(np.random.Philox(
        key=int(info["stream_key_hex"], 16)))
    noise = generator.standard_normal(noise_shape, dtype=np.float64)
    assert hashlib.sha256(noise.tobytes()).hexdigest() == info["noise_sha256"]
    frequency = np.fft.fftfreq(noise_shape[0])
    amplitude = np.exp(-0.25 * (2 * np.pi * frequency * scale) ** 2)
    variance = float(np.mean(amplitude ** 2))
    filtered = np.fft.ifft(
        np.fft.fft(noise, axis=0) * amplitude[:, None, None], axis=0).real
    start, stop = info["vertical_crop_levels"]
    assert stop - start == shape[0]
    assert info["pre_normalization_analytic_rms"] == pytest.approx(math.sqrt(variance))
    np.testing.assert_allclose(field, filtered[start:stop] / math.sqrt(variance),
                               rtol=1e-12, atol=1e-12)
    second, stamp = perturb.gaussian_random_field(
        shape, seed=7, name="column_probe", dx_km=1.0, dy_km=1.0,
        length_scale_km=0.0, vertical_scale_levels=scale, xp=np)
    np.testing.assert_array_equal(field, second)
    assert stamp == info


def test_zero_vertical_scale_still_uses_the_original_noise_shape():
    shape = (6, 8, 10)
    _, info = perturb.gaussian_random_field(
        shape, seed=5, name="uncorrelated_probe", dx_km=1.0, dy_km=1.0,
        length_scale_km=0.0, xp=np)
    assert tuple(info["noise_shape"]) == shape
    assert info["vertical_crop_levels"] == [0, shape[0]]
    assert info["vertical_wrap"]["top_to_bottom_seam"] == pytest.approx(0.0, abs=1e-15)


def test_fft_admission_prices_padding_and_its_own_plan_shapes():
    # mass_balance "none": since d51f0cf32 the default hydrostatic balance
    # holds float64 column captures that dominate this small census, so
    # the draw's padded FFT, which this cell prices, would not set the
    # peak.  The balance's own price is pinned in tests/test_da_cycle_memory.py
    # (test_the_census_prices_the_streamfunction_draw_and_the_mass_balance).
    cfg = perturb.PerturbationConfig(
        dx_km=1.0, dy_km=1.0, mass_balance="none",
        fields=(perturb.FieldPerturbation(
            "theta", 0.5, 4.0, vertical_scale_levels=6.0),))
    # The default still prices at least everything the draw needs
    # (reproduces a847c4701 on development branch).
    assert perturb.device_working_bytes(
        replace(cfg, mass_balance="hydrostatic"), (24, 32, 32)) >= \
        perturb.device_working_bytes(cfg, (24, 32, 32))
    mass_shape = (24, 32, 32)
    padded_shape = (72, 32, 32)
    assert perturb._draw_shapes(cfg, mass_shape) == (padded_shape,)
    without = perturb.device_working_bytes(cfg, mass_shape)
    with_plan = perturb.device_working_bytes(
        cfg, mass_shape, {padded_shape: (1 << 30, 2 << 30)})
    assert with_plan >= 2 << 30
    assert with_plan > without
    no_vertical = replace(cfg, fields=(replace(cfg.fields[0], vertical_scale_levels=0.0),))
    assert without > perturb.device_working_bytes(no_vertical, mass_shape)
    # Host FFTs return only the physical crop to the device.
    host = replace(cfg, fft_host=True)
    assert perturb.device_working_bytes(host, mass_shape) == perturb.device_working_bytes(
        replace(no_vertical, fft_host=True), mass_shape)


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_vertical_taper_ends_and_default_identity(kind):
    np.testing.assert_array_equal(perturb.vertical_taper(12, xp=np), np.ones(12))
    top = perturb.vertical_taper(12, top_width_levels=3, kind=kind, xp=np)
    assert top[-1] == 0.0
    assert top[0] == 1.0
    np.testing.assert_array_equal(top[:-3], np.ones(9))
    assert np.all(np.diff(top) <= 0.0)
    both = perturb.vertical_taper(12, 3, 3, kind=kind, xp=np)
    assert both[0] == both[-1] == 0.0
    np.testing.assert_array_equal(both[3:9], np.ones(6))
    np.testing.assert_array_equal(both, both[::-1])
    overlap = perturb.vertical_taper(3, 4, 4, kind=kind, xp=np)
    assert np.all((overlap >= 0.0) & (overlap <= 1.0))


@pytest.mark.parametrize("width", [-1, 1.5])
def test_vertical_taper_refuses_an_invalid_ramp(width):
    with pytest.raises(ValueError, match="non-negative integer"):
        perturb.vertical_taper(12, top_width_levels=width, xp=np)


def _filter_case(error=0.3):
    values = np.array([-1.5, -0.5, 0.5, 1.5])
    prior = {"theta": np.broadcast_to(values[:, None, None, None], (4, 1, 1, 5)).copy()}
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0, heights_m=np.array([500.0]))
    mask = np.array([[[False, False, True, False, False]]])
    obs = GriddedObs(name="spread_probe", values=np.zeros((1, 1, 5)),
                     errors=error, simulated=prior["theta"].copy(), mask=mask)
    # The cutoff reaches neighboring columns, as required by the filter.
    # Endpoints are exactly at the cutoff, so their weights remain zero.
    cfg = LetkfConfig(localization=Localization(horizontal_m=2000.0, vertical_m=500.0),
                      analysis_fields=("theta",), rtps_alpha=0.0)
    return grid, prior, obs, cfg


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.08, 1.20, 2.0])
def test_rtps_accepts_finite_non_negative_over_relaxation(alpha):
    _, _, _, cfg = _filter_case()
    assert replace(cfg, rtps_alpha=alpha).rtps_alpha == alpha


@pytest.mark.parametrize("alpha", [-0.1, float("nan"), float("inf")])
def test_rtps_refuses_invalid_spread_relaxation(alpha):
    _, _, _, cfg = _filter_case()
    with pytest.raises(LetkfError, match="finite and non-negative"):
        replace(cfg, rtps_alpha=alpha)


def test_rtpp_keeps_convex_mixture_bounds():
    _, _, _, cfg = _filter_case()
    with pytest.raises(LetkfError, match="RTPP must lie"):
        replace(cfg, relaxation="rtpp", rtps_alpha=1.08)


@pytest.mark.parametrize("alpha", [1.08, 1.20])
def test_rtps_over_relaxes_contracted_spread_without_moving_the_mean(alpha):
    grid, prior, obs, cfg = _filter_case()
    base = analyze(prior, [obs], grid, cfg)["theta"]
    relaxed = analyze(prior, [obs], grid, replace(cfg, rtps_alpha=alpha))["theta"]
    sb = prior["theta"].std(axis=0, ddof=1)
    sa = (prior["theta"] + base).std(axis=0, ddof=1)
    result = prior["theta"] + relaxed
    target = (1 - alpha) * sa + alpha * sb
    np.testing.assert_allclose(result.std(axis=0, ddof=1), target, atol=1e-12)
    np.testing.assert_allclose(relaxed.mean(axis=0), base.mean(axis=0), atol=1e-12)
    assert result.std(axis=0, ddof=1)[0, 0, 2] > sb[0, 0, 2]
    np.testing.assert_array_equal(relaxed[:, :, :, (0, 4)], np.zeros((4, 1, 1, 2)))


@pytest.mark.parametrize("with_observation", [False, True])
def test_negative_rtps_target_collapses_spread_without_sign_reversal(with_observation):
    grid, prior, obs, cfg = _filter_case(error=1000.0)
    diagnostics = LetkfDiagnostics()
    # sqrt(rho)=10 and alpha=2 give the inactive target factor -8.
    # The very weak active observation also leaves sa > 2 sb.
    inc = analyze(prior, [obs] if with_observation else [], grid,
                  replace(cfg, prior_inflation=100.0, rtps_alpha=2.0),
                  diagnostics=diagnostics)["theta"]
    np.testing.assert_allclose(prior["theta"] + inc, 0.0, atol=1e-12)
    assert diagnostics.active_points == 3 * int(with_observation)
