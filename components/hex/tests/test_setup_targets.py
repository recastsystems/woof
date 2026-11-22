"""Setup-following targets for the moving fine region (synthetic states, CPU only)."""
import json
import math

import numpy as np
import pytest

from woof.hex import storm_tracker as st

CENTRE = (36.0, -98.0)


def cloud(n=40000, radius_km=1200.0, seed=3):
    rng = np.random.default_rng(seed)
    r = radius_km * np.sqrt(rng.random(n))
    a = 2 * np.pi * rng.random(n)
    return st.unproject(CENTRE[0], CENTRE[1], r * np.sin(a), r * np.cos(a))


def columns(lat, lon, moist_centre, *, moist_radius_km=250.0, nlev=40):
    """A conditionally unstable sounding everywhere; low-level moisture only in a disc; 30 m/s of 0-6 km shear."""
    n = lat.size
    x, y = st.project(moist_centre[0], moist_centre[1], lat, lon)
    moist = np.exp(-(np.hypot(x, y) / moist_radius_km) ** 2)
    psfc = np.full(n, 97000.0)
    p = psfc[:, None] * np.linspace(1.0, 0.12, nlev)[None, :]
    z_approx = 7500.0 * np.log(psfc[:, None] / p)
    t = np.maximum(303.0 - 0.0075 * z_approx, 205.0)                # 7.5 K/km to the tropopause
    theta = t * (1.0e5 / p) ** (st.RD / st.CP)
    qv = (0.003 + 0.013 * moist[:, None]) * np.exp(-z_approx / 2500.0)
    u = np.broadcast_to(5.0 * z_approx / 1000.0, p.shape).copy()     # 30 m/s at 6 km
    v = np.zeros_like(u)
    return dict(pressure=p, theta=theta, qv=qv, u=u, v=v, terrain=np.full(n, 400.0), psfc=psfc)


def test_mlcape_and_shear_are_plausible():
    lat, lon = np.array([CENTRE[0], CENTRE[0] + 8]), np.array([CENTRE[1], CENTRE[1]])
    cape, shear = st.mlcape_and_shear(columns(lat, lon, CENTRE))
    assert 1000.0 < cape[0] < 6000.0          # the moist column
    assert cape[1] < 0.2 * cape[0]            # the dry one
    assert shear == pytest.approx([30.0, 30.0], abs=4.0)       # 5 m/s per km on the approximate heights


def test_cape_shear_target_finds_the_moist_warm_sector():
    lat, lon = cloud()
    target = st.destination(*CENTRE, 60.0, 120.0)
    target = (float(target[0]), float(target[1]))
    fields = st.StateFields(lat=lat, lon=lon, columns=columns(lat, lon, target))
    config = st.EnvConfig(max_step_km=None)     # one frame's localisation; the per-check step cap has its own test
    fix = st.EnvFixer(config).fix(0.0, fields, CENTRE, 300.0)
    assert fix.ok, fix.reason
    raw = fix.components['cape_shear']['raw_centre']
    assert st.distance_km(raw[0], raw[1], *target) < 25.0
    # the fix moves toward the area by the blend, so one frame's centroid jump is damped
    assert st.distance_km(fix.lat, fix.lon, *target) == pytest.approx((1 - config.fix_blend) * 120.0, abs=25.0)


def test_cape_shear_target_says_why_without_columns():
    lat, lon = cloud(2000)
    fix = st.EnvFixer().fix(0.0, st.StateFields(lat=lat, lon=lon), CENTRE, 300.0)
    assert not fix.ok and 'columns' in fix.reason


def surface(lat, lon, line_centre, bearing_deg):
    """A dryline through line_centre along bearing_deg: 20 C dewpoint east of it, -5 C west."""
    x, y = st.project(line_centre[0], line_centre[1], lat, lon)
    b = math.radians(bearing_deg)
    across = x * math.cos(b) - y * math.sin(b)        # positive to the east/right of the line
    td = 273.15 + 7.5 + 12.5 * np.tanh(across / 20.0)
    p = np.full(lat.size, 90000.0)
    es = 611.2 * np.exp(17.67 * (td - 273.15) / (td - 29.65))
    r = 0.622 * es / (p - es)
    return dict(t2=np.full(lat.size, 302.0), q2=r / (1 + r), psfc=p)


def test_boundary_target_follows_the_dryline_aligned_with_the_drawn_segment():
    lat, lon = cloud()
    line = st.destination(*CENTRE, 270.0, 80.0)
    line = (float(line[0]), float(line[1]))
    fields = st.StateFields(lat=lat, lon=lon, surface=surface(lat, lon, line, 10.0))
    a = st.destination(*line, 10.0, 150.0)
    b = st.destination(*line, 190.0, 150.0)
    config = st.BoundaryConfig(segment=[[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]], fix_blend=1.0)
    fix = st.BoundaryFixer(config).fix(0.0, fields, CENTRE, 200.0)
    assert fix.ok, fix.reason
    x, y = st.project(*line, fix.lat, fix.lon)
    across = x * math.cos(math.radians(10.0)) - y * math.sin(math.radians(10.0))
    assert abs(across) < 15.0                                   # on the line
    axis = fix.components['boundary']['axis_deg']
    assert min(abs(axis - 10.0), 180 - abs(axis - 10.0)) < 15.0  # oriented like it


def test_low_target_finds_a_synoptic_low():
    lat, lon = cloud()
    low = st.destination(*CENTRE, 45.0, 150.0)
    low = (float(low[0]), float(low[1]))
    x, y = st.project(low[0], low[1], lat, lon)
    r = np.hypot(x, y)
    mslp = (1012.0 - 14.0 * np.exp(-(r / 600.0) ** 2)) * 100.0
    fields = st.StateFields(lat=lat, lon=lon, mslp_pa=mslp)
    fix = st.TCFixer(st.low_config(fix_blend=1.0)).fix(0.0, fields, CENTRE, 300.0)
    assert fix.ok, fix.reason
    assert st.distance_km(fix.lat, fix.lon, *low) < 40.0
    # the low target's own blend moves half way: a broad low's minimum jumps between frames (IFS f360 lee low)
    damped = st.TCFixer(st.low_config()).fix(0.0, fields, CENTRE, 300.0)
    assert st.distance_km(damped.lat, damped.lon, *low) == pytest.approx(75.0, abs=25.0)


def test_cape_shear_target_follows_the_core_not_the_sprawl():
    """GFS 2026-10-23 18Z f354: one 570,000 km2 area whose centroid drifted 700 km away from the strongest part."""
    lat, lon = cloud(radius_km=1400.0)
    core = st.destination(*CENTRE, 90.0, 300.0)
    core = (float(core[0]), float(core[1]))
    wide = columns(lat, lon, st.destination(*CENTRE, 270.0, 300.0), moist_radius_km=900.0)
    strong = columns(lat, lon, core, moist_radius_km=150.0)
    wide['qv'] = 0.7 * wide['qv'] + 0.6 * strong['qv']
    fields = st.StateFields(lat=lat, lon=lon, columns=wide)
    fix = st.EnvFixer(st.EnvConfig(fix_blend=1.0, search_km=700.0, core_reach_km=None, max_step_km=None)).fix(
        0.0, fields, CENTRE, 700.0)
    assert fix.ok, fix.reason
    assert fix.components['cape_shear']['core_km2'] < fix.components['cape_shear']['area_km2']
    assert st.distance_km(fix.lat, fix.lon, *core) < 120.0


def test_cape_shear_target_stays_with_its_storms():
    """GFS 2026-10-23 18Z f354: storms used up the CAPE they grew from, the index core slid away from them
    and the region left the storms. Storms inside the area now hold the target; with no storms it is the core."""
    lat, lon = cloud(radius_km=1400.0)
    core = st.destination(*CENTRE, 270.0, 250.0)
    core = (float(core[0]), float(core[1]))
    cols = columns(lat, lon, core, moist_radius_km=450.0)
    storms_at = st.destination(*CENTRE, 90.0, 150.0)
    x, y = st.project(float(storms_at[0]), float(storms_at[1]), lat, lon)
    refl = np.where(np.hypot(x, y) < 120.0, 50.0, 0.0)
    config = st.EnvConfig(fix_blend=1.0, search_km=700.0, core_reach_km=None, max_step_km=None)
    quiet = st.EnvFixer(config).fix(0.0, st.StateFields(lat=lat, lon=lon, columns=cols), CENTRE, 700.0)
    stormy = st.EnvFixer(config).fix(0.0, st.StateFields(lat=lat, lon=lon, columns=cols, object_field=refl),
                                     CENTRE, 700.0)
    assert quiet.ok and stormy.ok
    assert quiet.components['cape_shear']['storm_share'] == 0.0
    assert stormy.components['cape_shear']['storm_km2'] > 30000.0 and 'storms' in stormy.reason
    to_storms = lambda f: st.distance_km(f.lat, f.lon, float(storms_at[0]), float(storms_at[1]))
    assert to_storms(stormy) < to_storms(quiet) - 100.0
    far = refl * 0.0
    x2, y2 = st.project(*st.destination(*CENTRE, 90.0, 1300.0), lat, lon)
    far[np.hypot(x2, y2) < 100.0] = 50.0                        # storms far outside the area do not count
    other = st.EnvFixer(config).fix(0.0, st.StateFields(lat=lat, lon=lon, columns=cols, object_field=far),
                                    CENTRE, 700.0)
    assert other.components['cape_shear']['storm_share'] == 0.0


def test_cape_shear_target_moves_like_weather_not_across_the_country():
    """Same run, +16 h: the storms died after dark, the one connected area's peak was a storm-free ocean
    plume 900 km away and the fixes ran 100-150 km per check. The core is now taken near the prediction and one
    fix moves at most max_step_km."""
    lat, lon = cloud(radius_km=1400.0)
    far = st.destination(*CENTRE, 250.0, 550.0)
    far = (float(far[0]), float(far[1]))
    wide = columns(lat, lon, st.destination(*CENTRE, 250.0, 250.0), moist_radius_km=700.0)
    strong = columns(lat, lon, far, moist_radius_km=150.0)
    wide['qv'] = 0.7 * wide['qv'] + 0.6 * strong['qv']
    fields = st.StateFields(lat=lat, lon=lon, columns=wide)
    free = st.EnvFixer(st.EnvConfig(search_km=700.0, core_reach_km=None, max_step_km=None)).fix(0.0, fields, CENTRE, 700.0)
    held = st.EnvFixer(st.EnvConfig(search_km=700.0)).fix(0.0, fields, CENTRE, 700.0)
    assert free.ok and held.ok
    assert st.distance_km(free.lat, free.lon, *CENTRE) > 150.0                  # the old target leapt toward the plume
    assert st.distance_km(held.lat, held.lon, *CENTRE) <= 45.0 + 1.0
    assert held.components['cape_shear']['step_capped'] or held.components['cape_shear']['step_km'] <= 45.0


def test_path_target_goes_where_the_user_said():
    rows = [[0.0, 35.0, -100.0], [36000.0, 36.0, -95.0]]
    session = st.TrackSession(st.TrackerConfig.from_dict(dict(
        mode='path', initial_center_deg=[35.0, -100.0], path=dict(rows=rows),
        placement=dict(interior_radius_km=146.0, core_margin_km=48.0, check_minutes=30.0))))
    fix = session.observe(18000.0, None)
    assert fix.ok and 'not a tracker' in fix.reason
    assert fix.lat == pytest.approx(35.5) and fix.lon == pytest.approx(-97.5, abs=0.01)
    assert fix.steering_ms[0] > 0                               # heading east, as drawn


@pytest.mark.parametrize('mode', st.TRACKER_MODES)
def test_every_mode_builds_a_session_and_names_what_it_reads(mode):
    config = st.TrackerConfig.from_dict(dict(mode=mode, initial_center_deg=list(CENTRE),
                                             path=dict(rows=[[0, *CENTRE]])))
    st.TrackSession(config)
    needs = st.history_needs(config)
    assert needs['columns'] == (mode == 'cape_shear') and needs['surface'] == (mode == 'boundary')
    assert needs['state'] == (mode != 'path')
    assert needs['storm_field'] == ('refl10cm' if mode == 'cape_shear' else None)


def test_author_setup_case_from_a_forecast_lead(tmp_path):
    from woof.hex.moving_author import author_case
    case = author_case(tmp_path / 'case.json', cycle='2026-10-23T18', point=[36.0, -98.0], target='cape-shear',
                       hours=24, source='gfs', fine_km=3.0, background_km=96.0, fine_radius_km=300.0,
                       lead_hours=354, check_minutes=30)
    assert case['start_time'] == '2026-11-07_12:00:00'   # 18Z + 14 d 18 h
    assert case['init_cycle'] == '2026-10-23_18:00:00' and case['init_lead_hours'] == 354
    assert case['tracker']['mode'] == 'cape_shear'
    assert case['storm']['initial_center_deg'] == [36.0, -98.0]
    # GFS runs to 384 h: a 24 h run from f354 ends at f378, inside the source
    assert 'beyond_source' not in case
    written = json.loads((tmp_path / 'case.json').read_text())
    assert written['authoring']['init_lead_hours'] == 354


def test_author_from_the_last_hour_names_the_free_run(tmp_path):
    from woof.hex.moving_author import author_case
    case = author_case(tmp_path / 'case.json', cycle='2026-10-23T18', point=[36.0, -98.0], target='low',
                       hours=24, source='gfs', fine_km=3.0, background_km=96.0, from_last_hour=True)
    assert case['init_lead_hours'] == 384
    assert case['beyond_source']['free_running_hours'] == 24
    assert 'free-running beyond the source' in case['beyond_source']['note']


def test_stage_plans_the_source_cycle_at_the_lead(tmp_path):
    from woof.hex import init_sources
    from woof.hex.moving_author import author_case
    author_case(tmp_path / 'case.json', cycle='2026-10-23T18', point=[36.0, -98.0], target='cape-shear',
                hours=24, source='gfs', fine_km=3.0, background_km=96.0, lead_hours=354)
    row = init_sources.resolve_source(init_sources.load_table(), 'gfs')
    plans = init_sources.plan_products(row, init_sources.Cycle.parse('2026-10-23T18'), None, 354)
    names = [str(getattr(plan, 'filename', plan)) for plan in plans]
    assert any('f354' in name for name in names), names


def test_history_frames_feed_the_setup_targets(tmp_path):
    """The controller reads the model columns and 2 m fields a setup target needs from a closed history frame."""
    netCDF4 = pytest.importorskip('netCDF4')
    from woof.hex.moving_run import history_fields
    lat, lon = cloud(3000)
    cols = columns(lat, lon, CENTRE)
    n, nlev = lat.size, cols['pressure'].shape[1]
    path = tmp_path / 'history.nc'
    with netCDF4.Dataset(path, 'w') as d:
        d.createDimension('Time', 1)
        d.createDimension('nCells', n)
        d.createDimension('nVertLevels', nlev)
        put = lambda name, dims, value: d.createVariable(name, 'f8', dims).__setitem__(Ellipsis, value)
        put('latCell', ('nCells',), np.radians(lat))
        put('lonCell', ('nCells',), np.radians(lon))
        put('ter', ('nCells',), cols['terrain'])
        for name, key in (('pressure', 'pressure'), ('theta', 'theta'), ('qv', 'qv'), ('u_zonal', 'u'),
                          ('v_meridional', 'v')):
            put(name, ('Time', 'nCells', 'nVertLevels'), cols[key][None])
        for name, value in (('surface_pressure', cols['psfc']), ('u10', cols['u'][:, 0]), ('v10', cols['v'][:, 0]),
                            ('t2m', np.full(n, 300.0)), ('q2', np.full(n, 0.01))):
            put(name, ('Time', 'nCells'), value[None])
    fields = history_fields(st, path, CENTRE, 300.0, columns=True, surface=True)
    assert fields.columns is not None and fields.columns['qv'].shape[1] == nlev
    assert fields.surface is not None and np.allclose(fields.surface['t2'], 300.0)
    fix = st.EnvFixer().fix(0.0, fields, CENTRE, 300.0)
    assert fix.ok, fix.reason
