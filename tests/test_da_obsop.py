"""EXPERIMENTAL (WOOF v1.2): radar observation operators.

Deliberately CPU-only: this module imports no CuPy, so it stays outside
the ``gpu`` marker and the whole file runs on a machine with no device.

The reflectivity assertions are the point of the Z tests.  The operator
is an adapter over ``woof.core.refl`` / ``woof.verify.npref``, so
comparing it against those mirrors alone would be circular; the crafted
single-category cases below are therefore checked against closed forms
evaluated in-test from the WRF constants, and the multi-column case is
checked against the mirrors to catch index/assembly bugs the closed
forms cannot see.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core import constants as c
from woof.da import obsop


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _one(value):
    """The sole element of a one-element array, as a Python float."""
    array = np.asarray(value)
    assert array.size == 1, f"expected one element, got {array.shape}"
    return float(array.reshape(-1)[0])


def _state(nz=4, ny=3, nx=5, *, pressure=8.0e4, temperature=290.0):
    """A minimal duck-typed state carrying only what the operators read."""
    s = SimpleNamespace()
    s.u = np.zeros((nz, ny, nx + 1), np.float32)
    s.v = np.zeros((nz, ny + 1, nx), np.float32)
    s.w = np.zeros((nz + 1, ny, nx), np.float32)
    s.php = np.zeros((nz + 1, ny, nx), np.float32)
    s.phb = np.asarray(
        np.float32(c.G) * np.linspace(0.0, 12000.0, nz + 1), np.float32)
    s.sina = np.zeros((ny, nx), np.float32)
    s.cosa = np.ones((ny, nx), np.float32)
    s.p = np.full((nz, ny, nx), pressure, np.float32)
    s.thp = np.zeros((nz, ny, nx), np.float32)
    theta = temperature * (np.float32(c.P0) / pressure) ** np.float32(c.RCP)
    s.thb = np.full((nz,), theta, np.float32)
    for name in ("qv", "qr", "nr", "qs", "ns", "qg", "ng"):
        setattr(s, name, np.zeros((nz, ny, nx), np.float32))
    s.qv[...] = 1.0e-2
    return s


def _flat_geometry(state, site_lat, site_lon, *, height_m=0.0):
    """Every column at the same lat/lon offset from a single site."""
    nz, ny, nx = state.p.shape
    lat = np.full((ny, nx), site_lat, np.float64)
    lon = np.full((ny, nx), site_lon, np.float64)
    hgt = np.full((nz, ny, nx), height_m, np.float64)
    return obsop.GridGeometry(latitude_deg=lat, longitude_deg=lon,
                              height_m=hgt)


def _uniform_wind(state, u=0.0, v=0.0, w=0.0):
    state.u[...] = np.float32(u)
    state.v[...] = np.float32(v)
    state.w[...] = np.float32(w)
    return state


# --------------------------------------------------------------------------
# destaggering
# --------------------------------------------------------------------------

def test_destaggering_recovers_a_linear_field_at_mass_points():
    """A field linear in the staggered index averages to the midpoint."""
    nz, ny, nx = 3, 4, 5
    u = np.arange(nz * ny * (nx + 1), dtype=np.float64).reshape(
        nz, ny, nx + 1)
    v = np.arange(nz * (ny + 1) * nx, dtype=np.float64).reshape(
        nz, ny + 1, nx)
    w = np.arange((nz + 1) * ny * nx, dtype=np.float64).reshape(
        nz + 1, ny, nx)

    um = obsop.destagger_u(u)
    vm = obsop.destagger_v(v)
    wm = obsop.destagger_w(w)

    assert um.shape == (nz, ny, nx)
    assert vm.shape == (nz, ny, nx)
    assert wm.shape == (nz, ny, nx)
    # Each mass value is the mean of the two faces that bracket it.
    np.testing.assert_allclose(um, 0.5 * (u[:, :, :-1] + u[:, :, 1:]))
    np.testing.assert_allclose(vm, 0.5 * (v[:, :-1, :] + v[:, 1:, :]))
    np.testing.assert_allclose(wm, 0.5 * (w[:-1] + w[1:]))


def test_destaggering_picks_the_right_axis():
    """A ramp in x must survive destaggering in x and stay flat in y."""
    nz, ny, nx = 2, 3, 4
    # u = face index, so mass point i sits at i + 0.5.
    u = np.tile(np.arange(nx + 1, dtype=np.float64), (nz, ny, 1))
    um = obsop.destagger_u(u)
    expected = np.arange(nx, dtype=np.float64) + 0.5
    for k in range(nz):
        for j in range(ny):
            np.testing.assert_allclose(um[k, j], expected)


def test_earth_relative_rotation_is_the_identity_on_an_idealized_grid():
    """sina = 0, cosa = 1 is the DomainState default; it must not rotate."""
    u = np.array([[[3.0, -4.0]]])
    v = np.array([[[5.0, 12.0]]])
    sina = np.zeros((1, 2))
    cosa = np.ones((1, 2))
    ue, vn = obsop.earth_relative_winds(u, v, sina, cosa)
    np.testing.assert_array_equal(ue, u)
    np.testing.assert_array_equal(vn, v)


def test_earth_relative_rotation_matches_the_geo_em_convention():
    """alpha = 90 deg sends grid-x onto -north and grid-y onto +east.

    With ``u_earth = u*cosa - v*sina`` and ``v_earth = u*sina + v*cosa``
    (woof/static/projection.py:21-30), alpha = +90 deg takes a pure
    grid-relative westerly (u = 1, v = 0) to a pure southerly-pointing
    earth vector (east 0, north +1).
    """
    u = np.array([[[1.0]]])
    v = np.array([[[0.0]]])
    sina = np.ones((1, 1))
    cosa = np.zeros((1, 1))
    ue, vn = obsop.earth_relative_winds(u, v, sina, cosa)
    np.testing.assert_allclose(ue, [[[0.0]]], atol=0.0)
    np.testing.assert_allclose(vn, [[[1.0]]], atol=0.0)
    # The rotation preserves speed, whatever the angle.
    for angle in np.linspace(0.0, 2.0 * np.pi, 17):
        ue, vn = obsop.earth_relative_winds(
            np.array([[[3.0]]]), np.array([[[4.0]]]),
            np.full((1, 1), np.sin(angle)), np.full((1, 1), np.cos(angle)))
        np.testing.assert_allclose(np.hypot(ue, vn), 5.0)


# --------------------------------------------------------------------------
# beam geometry
# --------------------------------------------------------------------------

def test_beam_unit_vector_is_a_unit_vector():
    state = _state()
    geometry = obsop.GridGeometry(
        latitude_deg=np.array([[35.0, 35.2], [35.4, 35.6]]),
        longitude_deg=np.array([[-97.0, -96.8], [-96.6, -96.4]]),
        height_m=np.broadcast_to(
            np.array([500.0, 3000.0, 9000.0])[:, None, None], (3, 2, 2)
        ).copy())
    site = obsop.RadarSite(35.3, -97.3, 380.0, name="analytic")
    beam = obsop.beam_geometry(geometry, site)
    east, north, up = beam.unit_vector_enu()
    magnitude = np.sqrt(east ** 2 + north ** 2 + up ** 2)
    # The chord length carries a cancellation of order (1 - cos theta),
    # so a few ulp of float64 slack is the accurate tolerance here.
    np.testing.assert_allclose(magnitude, 1.0, rtol=1e-10)


def test_equal_height_beam_elevation_is_half_the_effective_central_angle():
    """A chord between two points at equal radius makes angle theta/2.

    That is exact spherical geometry, and it pins the 4/3-earth
    substitution: the effective central angle must be the real-earth one
    divided by 4/3, so el = (3/8) * gamma.
    """
    separation_deg = 1.0
    state = _state(nz=1, ny=1, nx=1)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=0.0)
    site = obsop.RadarSite(0.0, -separation_deg, 0.0)

    beam = obsop.beam_geometry(geometry, site)
    gamma = math.radians(separation_deg)
    theta = gamma / obsop.EFFECTIVE_EARTH_RATIO

    np.testing.assert_allclose(beam.sin_elevation, math.sin(0.5 * theta),
                               rtol=1e-9)
    np.testing.assert_allclose(beam.cos_elevation, math.cos(0.5 * theta),
                               rtol=1e-9)
    np.testing.assert_allclose(beam.ground_range_m,
                               obsop.EARTH_RADIUS_M * gamma, rtol=1e-12)


def test_a_spherical_earth_would_give_a_steeper_beam_than_four_thirds():
    """The refraction ratio is essential, not decoration."""
    state = _state(nz=1, ny=1, nx=1)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=5000.0)
    site = obsop.RadarSite(0.0, -1.0, 0.0)

    four_thirds = obsop.beam_geometry(geometry, site)
    spherical = obsop.beam_geometry(geometry, site,
                                    effective_earth_ratio=1.0)
    # Inflating the earth flattens the apparent climb to a fixed height.
    assert _one(four_thirds.sin_elevation) < _one(spherical.sin_elevation)


def test_azimuth_is_exact_on_the_cardinal_great_circles():
    """Due-east and due-north targets get exactly (1, 0) and (0, 1)."""
    state = _state(nz=1, ny=1, nx=1)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=0.0)

    east_of_radar = obsop.beam_geometry(
        geometry, obsop.RadarSite(0.0, -1.0, 0.0))
    assert _one(east_of_radar.sin_azimuth) == 1.0
    assert _one(east_of_radar.cos_azimuth) == 0.0

    west_of_radar = obsop.beam_geometry(
        geometry, obsop.RadarSite(0.0, 1.0, 0.0))
    assert _one(west_of_radar.sin_azimuth) == -1.0
    assert _one(west_of_radar.cos_azimuth) == 0.0

    north_of_radar = obsop.beam_geometry(
        geometry, obsop.RadarSite(-1.0, 0.0, 0.0))
    assert _one(north_of_radar.sin_azimuth) == 0.0
    assert _one(north_of_radar.cos_azimuth) == 1.0


def test_a_column_over_the_radar_gets_a_vertical_beam():
    state = _state(nz=1, ny=1, nx=1)
    geometry = _flat_geometry(state, 40.0, -100.0, height_m=4000.0)
    beam = obsop.beam_geometry(geometry,
                               obsop.RadarSite(40.0, -100.0, 0.0))
    assert _one(beam.sin_elevation) == pytest.approx(1.0, abs=1e-15)
    assert _one(beam.cos_elevation) == pytest.approx(0.0, abs=1e-15)


def test_a_target_coincident_with_the_radar_has_no_beam_direction():
    state = _state(nz=1, ny=1, nx=1)
    geometry = _flat_geometry(state, 40.0, -100.0, height_m=0.0)
    beam = obsop.beam_geometry(geometry,
                               obsop.RadarSite(40.0, -100.0, 0.0))
    assert np.isnan(beam.sin_elevation).all()
    assert np.isnan(beam.cos_elevation).all()


# --------------------------------------------------------------------------
# radial velocity: analytic projections
# --------------------------------------------------------------------------

def _analytic_case(u=0.0, v=0.0, w=0.0, *, site, separation_deg=None,
                   fall_speed="none", height_m=0.0):
    state = _uniform_wind(_state(nz=1, ny=1, nx=1), u=u, v=v, w=w)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=height_m)
    vr = obsop.radial_velocity(state, site, geometry, fall_speed=fall_speed,
                               dtype=np.float64)
    return float(vr[0, 0, 0])


def test_pure_westerly_with_the_radar_due_west_is_the_exact_projection():
    """u = +10 m/s, radar 1 deg west on the equator, target at sea level.

    Both points sit at the effective-earth radius, so the beam elevation
    is exactly half the effective central angle and the azimuth is
    exactly due east.  Vr is therefore 10*cos(theta/2) with no free
    parameters at all.
    """
    separation = 1.0
    theta = math.radians(separation) / obsop.EFFECTIVE_EARTH_RATIO
    got = _analytic_case(u=10.0,
                         site=obsop.RadarSite(0.0, -separation, 0.0))
    assert got == pytest.approx(10.0 * math.cos(0.5 * theta), rel=1e-9)
    # Outbound: a westerly recedes from a radar sitting to its west.
    assert got > 0.0


def test_pure_westerly_with_the_radar_due_east_is_inbound():
    """Same wind, radar on the other side: same magnitude, opposite sign."""
    separation = 1.0
    theta = math.radians(separation) / obsop.EFFECTIVE_EARTH_RATIO
    got = _analytic_case(u=10.0, site=obsop.RadarSite(0.0, separation, 0.0))
    assert got == pytest.approx(-10.0 * math.cos(0.5 * theta), rel=1e-9)


def test_pure_westerly_across_a_due_north_beam_projects_to_exactly_zero():
    """A cross-beam wind contributes nothing, to the last bit."""
    got = _analytic_case(u=10.0, site=obsop.RadarSite(-1.0, 0.0, 0.0))
    assert got == 0.0


def test_pure_southerly_along_a_due_north_beam_is_the_exact_projection():
    separation = 1.0
    theta = math.radians(separation) / obsop.EFFECTIVE_EARTH_RATIO
    got = _analytic_case(v=10.0, site=obsop.RadarSite(-separation, 0.0, 0.0))
    assert got == pytest.approx(10.0 * math.cos(0.5 * theta), rel=1e-9)


def test_a_column_over_the_radar_sees_only_vertical_motion():
    """Directly overhead the beam is vertical: horizontal wind drops out."""
    got = _analytic_case(u=25.0, v=-13.0, w=3.0,
                         site=obsop.RadarSite(0.0, 0.0, 0.0),
                         height_m=4000.0)
    assert got == pytest.approx(3.0, rel=1e-12)


def test_a_45_degree_azimuth_splits_the_wind_by_the_expected_cosine():
    """Equatorial radar to the south-west; check against a hand projection."""
    state = _uniform_wind(_state(nz=1, ny=1, nx=1), u=10.0, v=10.0)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=0.0)
    site = obsop.RadarSite(-1.0, -1.0, 0.0)
    beam = obsop.beam_geometry(geometry, site)
    east, north, up = beam.unit_vector_enu()
    expected = 10.0 * _one(east) + 10.0 * _one(north)
    got = _one(obsop.radial_velocity(state, site, geometry,
                                      fall_speed="none", dtype=np.float64))
    assert got == pytest.approx(expected, rel=1e-12)
    # A south-west radar and a south-westerly wind: strongly outbound.
    assert got > 13.0


def test_grid_relative_winds_are_rotated_before_projection():
    """A non-identity sina/cosa must change the answer, and by the
    same amount as rotating the wind by hand."""
    separation = 1.0
    site = obsop.RadarSite(0.0, -separation, 0.0)
    rotated = _state(nz=1, ny=1, nx=1)
    _uniform_wind(rotated, u=10.0)
    # alpha = 90 deg: the grid-relative westerly becomes a pure northward
    # earth vector, which is perpendicular to a due-east beam.
    rotated.sina[...] = 1.0
    rotated.cosa[...] = 0.0
    geometry = _flat_geometry(rotated, 0.0, 0.0, height_m=0.0)
    got = _one(obsop.radial_velocity(rotated, site, geometry,
                                      fall_speed="none", dtype=np.float64))
    assert got == pytest.approx(0.0, abs=1e-12)

    unrotated = _one(obsop.radial_velocity(
        rotated, site, geometry, fall_speed="none", rotate_to_earth=False,
        dtype=np.float64))
    theta = math.radians(separation) / obsop.EFFECTIVE_EARTH_RATIO
    assert unrotated == pytest.approx(10.0 * math.cos(0.5 * theta), rel=1e-9)


def test_the_fall_speed_term_enters_with_the_vertical_component():
    """Vr = ... + (w - vt)*sin(el); overhead that is exactly w - vt."""
    state = _uniform_wind(_state(nz=1, ny=1, nx=1), w=5.0)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=4000.0)
    site = obsop.RadarSite(0.0, 0.0, 0.0)
    vt = np.full((1, 1, 1), 2.0)
    got = _one(obsop.radial_velocity(state, site, geometry, fall_speed=vt,
                                      dtype=np.float64))
    assert got == pytest.approx(3.0, rel=1e-12)


# --------------------------------------------------------------------------
# radial velocity: fail-closed contract
# --------------------------------------------------------------------------

def test_default_fall_speed_refuses_to_guess():
    """Dropping the fall-speed term silently would change the physics."""
    state = _state(nz=1, ny=1, nx=1)
    geometry = _flat_geometry(state, 0.0, 0.0)
    with pytest.raises(ValueError, match="reflectivity_dbz"):
        obsop.radial_velocity(state, obsop.RadarSite(0.0, -1.0, 0.0),
                              geometry)


def test_an_unknown_fall_speed_mode_is_rejected():
    state = _state(nz=1, ny=1, nx=1)
    geometry = _flat_geometry(state, 0.0, 0.0)
    with pytest.raises(ValueError, match="must be 'reflectivity'"):
        obsop.radial_velocity(state, obsop.RadarSite(0.0, -1.0, 0.0),
                              geometry, fall_speed="zero")


def test_a_geometry_on_the_wrong_grid_is_rejected():
    state = _state(nz=2, ny=3, nx=4)
    wrong = obsop.GridGeometry(
        latitude_deg=np.zeros((3, 5)), longitude_deg=np.zeros((3, 5)),
        height_m=np.zeros((2, 3, 5)))
    with pytest.raises(ValueError, match="does not match the state"):
        obsop.radial_velocity(state, obsop.RadarSite(0.0, -1.0, 0.0), wrong,
                              fall_speed="none")


def test_geometry_from_a_state_without_geolocation_fails_closed():
    state = _state()
    with pytest.raises(ValueError, match="latitude and longitude"):
        obsop.GridGeometry.from_state(state)


def test_geometry_from_a_state_uses_the_states_own_heights():
    state = _state(nz=4, ny=2, nx=2)
    ny, nx = 2, 2
    geometry = obsop.GridGeometry.from_state(
        state, latitude_deg=np.zeros((ny, nx)),
        longitude_deg=np.zeros((ny, nx)))
    z_full = np.asarray(state.phb)[:, None, None] / np.float32(c.G)
    expected = np.broadcast_to(0.5 * (z_full[:-1] + z_full[1:]),
                               (4, ny, nx))
    np.testing.assert_allclose(geometry.height_m, expected, rtol=1e-6)


def test_a_swapped_latitude_longitude_pair_is_caught():
    """The commonest geolocation bug there is; it must not survive to
    produce a plausible-looking wrong beam."""
    lon2d, lat2d = np.meshgrid(np.linspace(-97.2, -97.0, 2),
                               np.linspace(35.0, 35.2, 2))
    height = np.zeros((1, 2, 2))
    # Correct order builds fine.
    obsop.GridGeometry(latitude_deg=lat2d, longitude_deg=lon2d,
                       height_m=height)
    with pytest.raises(ValueError, match="swapped"):
        obsop.GridGeometry(latitude_deg=lon2d, longitude_deg=lat2d,
                           height_m=height)


def test_nonfinite_geometry_is_rejected():
    good = np.zeros((2, 2))
    with pytest.raises(ValueError, match="latitude_deg must be finite"):
        obsop.GridGeometry(latitude_deg=np.full((2, 2), np.nan),
                           longitude_deg=good, height_m=np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="height_m must be finite"):
        obsop.GridGeometry(latitude_deg=good, longitude_deg=good,
                           height_m=np.full((1, 2, 2), np.inf))


def test_a_malformed_radar_site_is_rejected():
    with pytest.raises(ValueError, match="latitude"):
        obsop.RadarSite(91.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="altitude"):
        obsop.RadarSite(0.0, 0.0, float("nan"))


# --------------------------------------------------------------------------
# fall speed
# --------------------------------------------------------------------------

def test_reflectivity_fall_speed_matches_sun_and_crook_by_hand():
    """The density correction is (ps/p)**0.4, and ps is SURFACE pressure.

    Sun and Crook write ``vt = 5.40 * (ps/p)**0.4 * (rho*qr)**0.125`` with
    ``ps`` the pressure at the ground, and WRFDA's da_radial_velocity.inc
    makes it unambiguous by taking ``ps`` as a separate argument and
    forming ``alpha = (ps/p)**0.4``.  Substituting the thermodynamic
    reference constant P0 = 1000 hPa is not the same closure: over
    elevated terrain it is a systematic high bias in vt, constant through
    the column, and it pushes positive-away radial velocity inbound by
    ``-dvt*sin(elevation)``.
    """
    dbz = np.array([[[40.0]]])
    pressure = np.array([[[7.0e4]]])
    surface = np.array([[8.5e4]])
    active = np.ones((1, 1, 1), bool)
    got = _one(obsop.reflectivity_fall_speed(
        dbz, pressure, active, surface_pressure=surface)[0, 0, 0])
    expected = 5.40 * (8.5e4 / 7.0e4) ** 0.4 * 10.0 ** (
        0.125 * (40.0 - 43.1) / 17.5)
    assert got == pytest.approx(expected, rel=1e-12)
    # Sanity: a 40 dBZ rain shaft falls at a few m/s, not tens.
    assert 4.0 < got < 9.0


def test_the_fixed_reference_pressure_is_not_the_surface_pressure():
    """The size of the substitution the old code made, at three elevations.

    (100000/ps)**0.4 is +4.3% at 900 hPa, +9.3% at 800 hPa and +15.3% at
    700 hPa.  A 15% error in the hydrometeor fall speed is a large fraction
    of the vertical-motion signal a Doppler operator exists to extract.
    """
    dbz = np.full((1, 1, 1), 40.0)
    pressure = np.full((1, 1, 1), 5.0e4)
    active = np.ones((1, 1, 1), bool)
    for ps_hpa, bias in ((900.0, 0.043), (800.0, 0.093), (700.0, 0.153)):
        accurate = _one(obsop.reflectivity_fall_speed(
            dbz, pressure, active,
            surface_pressure=np.full((1, 1), ps_hpa * 100.0))[0, 0, 0])
        wrong = _one(obsop.reflectivity_fall_speed(
            dbz, pressure, active,
            surface_pressure=np.full((1, 1), c.P0))[0, 0, 0])
        assert wrong / accurate == pytest.approx(1.0 + bias, abs=5e-4)


def test_the_fall_speed_refuses_to_guess_a_surface_pressure():
    """Fail closed rather than substitute a constant for a field.

    The module already refuses to drop the fall-speed term silently; the
    same argument applies to quietly replacing its surface pressure with
    1000 hPa, which changes the physics without saying so.
    """
    dbz = np.full((1, 1, 1), 40.0)
    pressure = np.full((1, 1, 1), 7.0e4)
    active = np.ones((1, 1, 1), bool)
    with pytest.raises(ValueError, match="surface_pressure"):
        obsop.reflectivity_fall_speed(dbz, pressure, active)


def test_the_fall_speed_z_leg_is_the_bound_kessler_psd():
    """Sun and Crook's Z = 43.1 + 17.5*log10(rho*qr) is not an outside
    assumption: at rho*qr = 1 g/m3 the Smith-1975 PSD the Kessler
    fallback implements gives exactly 43.1 dBZ, which is what makes the
    fall speed self-consistent with the bound Z operator on mp=1.
    """
    rho_qr = 1.0e-3                      # kg m-3, i.e. 1 g m-3
    lam = (math.pi * 1000.0 * 8.0e6 / rho_qr) ** 0.25
    ze = math.gamma(7.0) * 8.0e6 * lam ** -7.0 * 1.0e18
    assert 10.0 * math.log10(ze) == pytest.approx(
        obsop.SUN_CROOK_Z_INTERCEPT_DBZ, abs=0.05)

    # And at that reflectivity the fall speed collapses to the bare
    # coefficient times the density correction.
    dbz = np.full((1, 1, 1), obsop.SUN_CROOK_Z_INTERCEPT_DBZ)
    pressure = np.full((1, 1, 1), c.P0)
    active = np.ones((1, 1, 1), bool)
    # ps == p, so the density correction is exactly 1 and only the
    # coefficient is left -- which is the point being checked.
    got = _one(obsop.reflectivity_fall_speed(
        dbz, pressure, active,
        surface_pressure=np.full((1, 1), c.P0))[0, 0, 0])
    assert got == pytest.approx(obsop.SUN_CROOK_VT_COEFF, rel=1e-12)


def test_fall_speed_rises_with_reflectivity_and_with_altitude():
    active = np.ones((1, 1, 3), bool)
    ps = np.full((1, 3), 9.5e4)
    rising_z = obsop.reflectivity_fall_speed(
        np.array([[[10.0, 35.0, 60.0]]]), np.full((1, 1, 3), 8.0e4), active,
        surface_pressure=ps)
    assert np.all(np.diff(rising_z[0, 0]) > 0.0)
    # Thinner air -> faster fall, via the (ps/p)**0.4 correction.
    thinning = obsop.reflectivity_fall_speed(
        np.full((1, 1, 3), 40.0),
        np.array([[[9.0e4, 7.0e4, 4.0e4]]]), active, surface_pressure=ps)
    assert np.all(np.diff(thinning[0, 0]) > 0.0)


def test_fall_speed_is_gated_off_where_no_precipitation_exists():
    """The power law does not vanish at the -35 dBZ floor, so the activity
    gate is what keeps clear air from acquiring a 2 m/s downdraught."""
    dbz = np.full((1, 1, 2), -35.0)
    pressure = np.full((1, 1, 2), 8.0e4)
    active = np.array([[[False, True]]])
    got = obsop.reflectivity_fall_speed(dbz, pressure, active,
                                        surface_pressure=np.full((1, 2), 9.5e4))
    assert float(got[0, 0, 0]) == 0.0
    assert float(got[0, 0, 1]) > 1.0


def test_fall_speed_honours_its_clamp():
    dbz = np.array([[[200.0]]])
    pressure = np.array([[[1.0e4]]])
    active = np.ones((1, 1, 1), bool)
    got = _one(obsop.reflectivity_fall_speed(
        dbz, pressure, active, surface_pressure=np.full((1, 1), 1.0e5),
        max_ms=30.0))
    assert got == 30.0


def test_precipitating_mixing_ratio_excludes_cloud_species():
    state = _state(nz=1, ny=1, nx=1)
    state.qr[...] = 1.0e-3
    state.qs[...] = 2.0e-3
    state.qg[...] = 3.0e-4
    state.qc = np.full((1, 1, 1), 9.0e-3, np.float32)
    state.qi = np.full((1, 1, 1), 9.0e-3, np.float32)
    got = _one(obsop.precipitating_mixing_ratio(state))
    assert got == pytest.approx(3.3e-3, rel=1e-6)


# ---------------------------------------------- F6 per-species activity gate


def _activity_state(values):
    """A one-cell state carrying the named species at the given values."""
    state = _state(nz=1, ny=1, nx=1)
    state.qh = np.zeros((1, 1, 1), np.float32)
    for name in obsop.PRECIPITATING_SPECIES:
        getattr(state, name)[...] = np.float32(values.get(name, 0.0))
    return state


def test_four_individually_inactive_species_do_not_activate_the_fall_speed():
    """The probe that reopened F6: 4 x 3e-10 kg/kg summed past the gate.

    Every species is below the reflectivity authority's own 1e-9 kg/kg
    activity threshold, so the bound Z operator adds nothing for any of
    them -- the cell is empty as far as H_Z is concerned.  The old gate
    summed the four to 1.2e-9, cleared 1e-9, and switched on a 5.244 m/s
    fall speed in that cell: a Doppler operator sedimenting hydrometeors
    its own reflectivity operator says are not there.
    """

    state = _activity_state({name: 3.0e-10
                             for name in obsop.PRECIPITATING_SPECIES})
    summed = _one(obsop.precipitating_mixing_ratio(state))
    assert summed > obsop.Q_ACTIVE_THRESHOLD, (
        "the probe is only a probe if the SUM does clear the threshold")
    assert not bool(np.asarray(obsop.precipitating_activity_mask(state))[
        0, 0, 0])

    dbz = np.full((1, 1, 1), 40.0)
    pressure = np.full((1, 1, 1), 8.0e4)
    got = _one(obsop.reflectivity_fall_speed(
        dbz, pressure, obsop.precipitating_activity_mask(state),
        surface_pressure=np.full((1, 1), 9.5e4)))
    assert got == 0.0
    # And the speed the summed gate switched on, from the closed form
    # rather than from "not zero": a regression that only asserted
    # inequality would pass on a gate that had merely become noisy.
    ungated = _one(obsop.reflectivity_fall_speed(
        dbz, pressure, np.ones((1, 1, 1), bool),
        surface_pressure=np.full((1, 1), 9.5e4)))
    hand = 5.40 * (9.5e4 / 8.0e4) ** 0.4 * 10.0 ** (
        0.125 * (40.0 - 43.1) / 17.5)
    assert ungated == pytest.approx(hand, rel=1e-9)
    assert ungated > 5.0, "several m/s of invented sedimentation"


@pytest.mark.parametrize("species", obsop.PRECIPITATING_SPECIES)
@pytest.mark.parametrize("value, active", [
    (0.0, False),
    (1.0e-10, False),
    (1.0e-9, False),           # strictly greater, as the authority is
    (1.0000001e-9, True),
    (1.0e-6, True),
])
def test_the_activity_gate_is_per_species_at_its_own_boundary(species, value,
                                                              active):
    """One species at a time, straddling 1e-9 from both sides.

    Keyed on the species as well as the value: a gate that read only
    ``qr`` would pass a single-value test and fail this one, and a gate
    that summed would pass this one and fail the four-species probe.
    """

    state = _activity_state({species: value})
    got = bool(np.asarray(obsop.precipitating_activity_mask(state))[0, 0, 0])
    assert got is active


def test_the_activity_gate_activates_on_any_one_active_species():
    state = _activity_state({"qs": 5.0e-9, "qr": 0.0, "qg": 0.0, "qh": 0.0})
    assert bool(np.asarray(
        obsop.precipitating_activity_mask(state))[0, 0, 0]) is True


def test_a_state_with_no_precipitating_species_refuses_rather_than_gating():
    state = _state(nz=1, ny=1, nx=1)
    for name in ("qr", "qs", "qg"):
        setattr(state, name, None)
    with pytest.raises(ValueError, match="no precipitating hydrometeor"):
        obsop.precipitating_activity_mask(state)


def test_the_summed_mixing_ratio_is_refused_as_an_activity_gate():
    """The old argument must not still work; it would still be wrong.

    ``reflectivity_fall_speed`` took the summed mixing ratio and
    thresholded it internally.  Accepting a float array now -- however it
    were interpreted -- would leave every existing caller on the summed
    gate, so the boolean mask is required by dtype and the refusal names
    the function that builds it.
    """

    state = _activity_state({name: 3.0e-10
                             for name in obsop.PRECIPITATING_SPECIES})
    with pytest.raises(ValueError, match="BOOLEAN mask"):
        obsop.reflectivity_fall_speed(
            np.full((1, 1, 1), 40.0), np.full((1, 1, 1), 8.0e4),
            obsop.precipitating_mixing_ratio(state),
            surface_pressure=np.full((1, 1), 9.5e4))


@pytest.mark.parametrize("value, sediments", [
    (3.0e-10, False),          # four of these summed past the old gate
    (3.0e-9, True),            # each one active on its own
])
def test_the_radial_velocity_route_uses_the_per_species_gate(value,
                                                             sediments):
    """The end-to-end path, not only the helper.

    Overhead beam, so ``Vr = w - vt`` exactly and the fall speed is
    readable straight off the answer.  With every species at 3e-10 the
    reflectivity operator sees an empty cell, so a filter must get pure
    air motion; with every species active the same call must still
    sediment, or the fix would be a gate that never fires.
    """

    state = _uniform_wind(_state(nz=1, ny=1, nx=1), w=5.0)
    state.qh = np.zeros((1, 1, 1), np.float32)
    for name in obsop.PRECIPITATING_SPECIES:
        getattr(state, name)[...] = np.float32(value)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=4000.0)
    site = obsop.RadarSite(0.0, 0.0, 0.0)
    got = _one(obsop.radial_velocity(
        state, site, geometry, reflectivity_dbz=np.full((1, 1, 1), 40.0),
        surface_pressure=np.full((1, 1), 9.5e4), dtype=np.float64))
    if sediments:
        assert got < 5.0 - 1.0
    else:
        assert got == pytest.approx(5.0, rel=1e-12), (
            "no species is active, so there is no fall speed: the radial "
            "velocity is the vertical air motion alone")


# --------------------------------------------------------------------------
# reflectivity operator
# --------------------------------------------------------------------------

def _hand_rho(qv, t, p):
    """refl10cm_hm's own density (module_mp_morr_two_moment.F:4540-4542)."""
    return 0.622 * p / (287.0 * t * (max(qv, 1.0e-10) + 0.622))


def test_kessler_reflectivity_matches_the_smith_1975_closed_form():
    """Single-category (rain-only) state against the derived fallback.

    woof/core/refl.py:30-43 derives Ze = Gamma(7)*N0r*lambda**-7 with
    lambda = (pi*rho_w*N0r/(rho*qr))**0.25, N0r = 8e6 m-4, rho_w = 1000.
    Evaluated here from those constants, not from the mirror.
    """
    state = _state(nz=1, ny=1, nx=1, pressure=9.0e4, temperature=295.0)
    state.qr[...] = 1.0e-3
    cfg = SimpleNamespace(mp_physics=1)

    got = float(obsop.simulated_reflectivity(state, cfg)[0, 0, 0])

    rho = _hand_rho(1.0e-2, 295.0, 9.0e4)
    lam = (math.pi * 1000.0 * 8.0e6 / (rho * 1.0e-3)) ** 0.25
    ze = math.gamma(7.0) * 8.0e6 * lam ** -7.0
    expected = max(-35.0, 10.0 * math.log10(ze * 1.0e18))
    assert got == pytest.approx(expected, rel=1e-5)
    # A 1 g/kg rain shaft is a solidly detectable echo.
    assert 35.0 < got < 50.0


def test_morrison_rain_only_matches_the_two_moment_closed_form():
    """Crafted single-category Morrison state against the exponential PSD.

    xam_r = pi*997/6 and xmu_r = 0 (module_mp_morr_two_moment.F:528-542)
    give lamr**3 = pi*997*nr/qr and ze_rain = 720*nr*rho/lamr**6
    (F:4547-4551, :4609).  With no snow or graupel there is no melting
    integral, so this is the whole answer.
    """
    qr, nr = 1.0e-3, 1.0e4
    qv, t, p = 1.0e-2, 295.0, 9.0e4
    state = _state(nz=1, ny=1, nx=1, pressure=p, temperature=t)
    state.qr[...] = qr
    state.nr[...] = nr
    cfg = SimpleNamespace(mp_physics=10, morr_rimed_ice=1)

    got = float(obsop.simulated_reflectivity(state, cfg)[0, 0, 0])

    rho = _hand_rho(qv, t, p)
    lamr = (math.pi * 997.0 * nr / qr) ** (1.0 / 3.0)
    ze = 720.0 * nr * rho / lamr ** 6
    expected = max(-35.0, 10.0 * math.log10(ze * 1.0e18))
    assert got == pytest.approx(expected, rel=1e-5)


def test_clear_air_returns_the_wrapper_floor_everywhere():
    """No hydrometeors at all is -35 dBZ, the WRF wrapper floor."""
    state = _state(nz=3, ny=2, nx=2)
    for mp in (1, 10):
        cfg = SimpleNamespace(mp_physics=mp, morr_rimed_ice=1)
        got = obsop.simulated_reflectivity(state, cfg)
        np.testing.assert_array_equal(got, np.full(got.shape, -35.0,
                                                   np.float32))


def test_the_three_dimensional_driver_assembles_columns_in_the_right_order():
    """Every column distinct, each checked against the certified mirror.

    The closed forms above cannot catch a transposed index; this can.
    """
    from woof.verify import npref

    nz, ny, nx = 4, 3, 5
    state = _state(nz=nz, ny=ny, nx=nx, pressure=8.5e4, temperature=288.0)
    rng_free = np.arange(nz * ny * nx, dtype=np.float64).reshape(nz, ny, nx)
    state.qr[...] = (1.0e-4 * (1.0 + rng_free)).astype(np.float32)
    cfg = SimpleNamespace(mp_physics=1)

    got = obsop.simulated_reflectivity(state, cfg)

    qv = np.asarray(state.qv, np.float64)
    qr = np.asarray(state.qr, np.float64)
    p = np.asarray(state.p, np.float64)
    theta = np.asarray(state.thb, np.float64)[:, None, None]
    t = theta * (p / c.P0) ** c.RCP
    for j in range(ny):
        for i in range(nx):
            expected = npref.np_refl10cm_kessler_column(
                qv[:, j, i], qr[:, j, i], t[:, j, i], p[:, j, i])
            np.testing.assert_allclose(got[:, j, i], expected, rtol=1e-5)
    # The distinct-column premise actually holds.
    assert len(np.unique(np.round(got, 4))) == nz * ny * nx


def test_reflectivity_increases_monotonically_with_rain_mass():
    state = _state(nz=1, ny=1, nx=4)
    state.qr[0, 0, :] = np.float32([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
    cfg = SimpleNamespace(mp_physics=1)
    got = obsop.simulated_reflectivity(state, cfg)[0, 0]
    assert np.all(np.diff(got) > 0.0)


def test_thompson_has_no_host_mirror_and_says_so():
    """mp_physics=8 fails closed on the CPU path rather than substituting
    another scheme's Z formulation."""
    state = _state(nz=1, ny=1, nx=1)
    cfg = SimpleNamespace(mp_physics=8)
    with pytest.raises(NotImplementedError, match="Thompson"):
        obsop.simulated_reflectivity(state, cfg)


def test_nssl_has_no_host_mirror_and_names_itself():
    """mp_physics=18 on the CPU path is a NAMED refusal, not a silent gap.

    NSSL is a flagship campaign scheme, so "not on the host" has to say so
    by name and say why -- five ice categories with their own moments make
    it a scheme port rather than an adapter over one of the mirrored three,
    which is exactly why there is no float64 column mirror to fall back on.
    The device path does support it (see the GPU dispatch gate); the host
    refusal must never read as absence.
    """
    state = _state(nz=1, ny=1, nx=1)
    cfg = SimpleNamespace(mp_physics=18)
    with pytest.raises(NotImplementedError, match="NSSL"):
        obsop.simulated_reflectivity(state, cfg)


def test_an_unsupported_scheme_is_refused_by_number():
    """A scheme with no formulation at all is named, not mapped to a default."""
    state = _state(nz=1, ny=1, nx=1)
    cfg = SimpleNamespace(mp_physics=99)
    with pytest.raises(NotImplementedError, match="99"):
        obsop.simulated_reflectivity(state, cfg)


def _p3_state(nz=3, ny=2, nx=2):
    """A P3-shaped state: one ice category, no qs and no qg."""
    state = _state(nz=nz, ny=ny, nx=nx, pressure=7.0e4, temperature=268.0)
    for absent in ("qs", "ns", "qg", "ng"):
        setattr(state, absent, None)
    for present in ("qi", "ni", "qir", "qib"):
        setattr(state, present, np.zeros((nz, ny, nx), np.float32))
    return state


def test_p3_has_its_own_operator_and_does_not_read_as_an_unknown_scheme():
    """mp_physics=50 must not read the way mp_physics=99 reads, and now it
    reads as what it is: a scheme with an H_Z(x).

    P3's Z used to exist only inside ``p3_main``'s final diagnostics loop,
    which also updates the state (module_mp_p3.F:4722-4895), so it was a
    named refusal.  The Z half of that loop now runs on local copies
    (woof/core/kernels/p3_zdiag.cu, host replay
    ``woof.core.p3_device.p3_reflectivity_host``), and the operator
    returns it.  A selector that does not exist is still refused by number.
    """
    state = _p3_state()
    dbz = obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=50))
    assert dbz.shape == state.p.shape and dbz.dtype == np.float32
    # A state with no rain and no ice reads the ONE clear-air value the
    # floor table records for P3, everywhere, exactly.
    assert np.all(dbz == np.float32(obsop.CLEAR_AIR_FLOOR_DBZ[50]))
    assert obsop.clear_air_floor_dbz(50) == obsop.CLEAR_AIR_FLOOR_DBZ[50]
    with pytest.raises(NotImplementedError, match="99"):
        obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=99))


def test_a_glaciated_p3_column_is_echo_and_the_state_is_not_moved():
    """The concrete breakage the old refusal prevented, now answered.

    P3 has ONE ice category and allocates no qs and no qg at all, so the
    only ``compute_refl_10cm`` branch a P3 state satisfies is the rain-only
    Kessler fallback, which would read a glaciated updraft as clear air.
    The P3 operator reads the ice through the scheme's own lookup table,
    and -- the property that makes it an observation operator -- leaves
    every field of the state it observes byte for byte as it was, even
    where the forecast loop would have clamped ni or dumped sub-QSMALL
    condensate into vapour and theta.
    """
    state = _p3_state()
    state.qi[0] = 1.0e-3
    state.ni[0] = 1.0e4
    state.qir[0] = 2.0e-4
    state.qib[0] = 4.0e-7
    state.qr[1] = 5.0e-4
    state.nr[1] = 1.0e3
    # sub-QSMALL condensate the forecast loop would evaporate
    state.qr[2] = 1.0e-16
    state.qi[2] = 1.0e-16
    # an ice number far past the scheme's total-Ni cap
    state.ni[0, 0, 0] = 1.0e9
    before = {name: np.array(getattr(state, name), copy=True)
              for name in ("qv", "thp", "qr", "nr", "qi", "ni", "qir", "qib",
                           "p")}
    dbz = obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=50))
    floor = np.float32(obsop.CLEAR_AIR_FLOOR_DBZ[50])
    assert np.all(dbz[0] > floor + 20.0), dbz[0]
    assert np.all(dbz[1] > floor + 20.0), dbz[1]
    assert np.all(dbz[2] == floor)
    assert np.all(np.isfinite(dbz))
    for name, value in before.items():
        assert np.array_equal(getattr(state, name), value), name


def test_every_native_z_refusal_carries_a_reason_and_its_authority():
    """The table is a record of decisions, in the DELIBERATE_STALE_SITES
    pattern: a selector may only be excluded WITH its reason.

    It is empty today, and that is a decision too: mp=9 left it when its Z
    block was lifted into milbrandt2_zet.cu, and mp=50 when the Z half of
    P3's final diagnostics was lifted into p3_zdiag.cu.  ``mp_physics=18``
    never belonged: NSSL's ``radardd02`` is a separate, pure Fortran
    diagnostic, and the criterion for this table is separability from the
    scheme's own state update, not membership of the 1/6/8/10/16/28 family.
    """
    table = obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP
    for mp, reason in table.items():
        assert f"mp_physics={mp}" in reason, (
            f"the reason for mp_physics={mp} does not name it")
        # A named scheme's authority is its own Fortran 90 source.
        assert ".F:" in reason or ".F90:" in reason, (
            f"the reason for mp_physics={mp} cites no WRF authority line")
    assert 18 not in table
    assert 50 not in table
    # The P3 family shares p3_zdiag.cu's H(x); NTU (mp=56) landed with its
    # own state-reading operator, woof.core.ntu.reflectivity.
    assert set(obsop.STATE_REFLECTIVITY_OPERATORS) == {50, 51, 52, 53, 56}


def test_reflectivity_requires_a_moist_state():
    state = _state(nz=1, ny=1, nx=1)
    state.qv = None
    with pytest.raises(ValueError, match="moist state"):
        obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=1))


def test_temperature_and_pressure_must_arrive_together():
    state = _state(nz=1, ny=1, nx=1)
    with pytest.raises(ValueError, match="supplied together"):
        obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=1),
                                     temperature=state.p)


def test_missing_morrison_moments_are_named():
    state = _state(nz=1, ny=1, nx=1)
    state.nr = None
    state.qr[...] = 1.0e-3
    cfg = SimpleNamespace(mp_physics=10, morr_rimed_ice=1)
    with pytest.raises(ValueError, match="nr"):
        obsop.simulated_reflectivity(state, cfg)


# --------------------------------------------------------------------------
# end-to-end: the operator pair on one state
# --------------------------------------------------------------------------

def test_the_operator_pair_runs_end_to_end_on_one_state():
    """H_Z feeds the fall speed that H_Vr consumes -- one Z, two uses."""
    state = _uniform_wind(_state(nz=4, ny=3, nx=3), u=12.0, v=-4.0, w=2.0)
    state.qr[...] = 1.0e-3
    cfg = SimpleNamespace(mp_physics=1)
    ny, nx = 3, 3
    lon2d, lat2d = np.meshgrid(np.linspace(-97.2, -97.0, nx),
                               np.linspace(35.0, 35.2, ny))
    geometry = obsop.GridGeometry(
        latitude_deg=lat2d, longitude_deg=lon2d,
        height_m=np.broadcast_to(
            np.linspace(500.0, 9000.0, 4)[:, None, None], (4, ny, nx)).copy())
    site = obsop.RadarSite(35.33, -97.28, 370.0, name="analytic-site")

    # A real surface pressure field, varying across the domain, because
    # the density factor is (ps/p)**0.4 and a constant would hide a
    # broadcasting error.
    psfc = np.linspace(9.0e4, 9.6e4, ny * nx).reshape(ny, nx)

    dbz = obsop.simulated_reflectivity(state, cfg)
    vr = obsop.radial_velocity(state, site, geometry, reflectivity_dbz=dbz,
                               surface_pressure=psfc)

    assert dbz.shape == (4, ny, nx)
    assert vr.shape == (4, ny, nx)
    assert vr.dtype == np.float32
    assert np.all(np.isfinite(vr))
    # A 12 m/s wind cannot produce a 40 m/s radial component.
    assert np.max(np.abs(vr)) < 20.0
    # Passing the Z explicitly and letting the operator find it agree.
    vr_self = obsop.radial_velocity(state, site, geometry, cfg=cfg,
                                    surface_pressure=psfc)
    np.testing.assert_allclose(vr, vr_self, rtol=0.0, atol=0.0)


def test_the_operators_are_pure_and_leave_the_state_untouched():
    state = _uniform_wind(_state(nz=2, ny=2, nx=2), u=7.0, v=3.0, w=1.0)
    state.qr[...] = 5.0e-4
    cfg = SimpleNamespace(mp_physics=1)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=2000.0)
    before = {name: np.array(getattr(state, name), copy=True)
              for name in ("u", "v", "w", "p", "thp", "qv", "qr")}

    obsop.simulated_reflectivity(state, cfg)
    obsop.radial_velocity(state, obsop.RadarSite(0.0, -0.5, 0.0), geometry,
                          cfg=cfg, surface_pressure=np.full((2, 2), 9.6e4))

    for name, original in before.items():
        np.testing.assert_array_equal(getattr(state, name), original)


def test_repeated_calls_are_bitwise_identical():
    state = _uniform_wind(_state(nz=3, ny=2, nx=2), u=9.0, v=2.0, w=-1.0)
    state.qr[...] = 8.0e-4
    cfg = SimpleNamespace(mp_physics=1)
    geometry = _flat_geometry(state, 0.0, 0.0, height_m=3000.0)
    site = obsop.RadarSite(0.0, -0.7, 100.0)
    psfc = np.full((2, 2), 9.7e4)
    first = obsop.radial_velocity(state, site, geometry, cfg=cfg,
                                  surface_pressure=psfc)
    second = obsop.radial_velocity(state, site, geometry, cfg=cfg,
                                   surface_pressure=psfc)
    np.testing.assert_array_equal(first, second)
