from datetime import datetime
from dataclasses import asdict
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from woof.config import RunConfig
from woof.tc.config import DEFAULTS, drop_inert_tc, validate_tc
from woof.tc.curves import coefficients, CURVES, make_table
from woof.tc.cli import configuration
from woof.tc.diagnostics import inland_decay, vortex_candidates


@pytest.mark.parametrize('name', CURVES)
def test_tables_are_physical(name):
    table = make_table(name, xp=np)
    assert table.shape == (1201, 3)
    assert table.dtype == np.float32
    assert table.flags.c_contiguous
    assert np.all(np.isfinite(table))
    assert np.all(table > 0)


def test_published_high_wind_limits():
    cd, ch, cq = coefficients([52., 60., 72.], 'bell2012', xp=np)
    np.testing.assert_allclose(cd, .0024)
    np.testing.assert_allclose(ch, .001)
    np.testing.assert_allclose(cq, .001)
    np.testing.assert_allclose(coefficients([33., 60., 100.], 'donelan2004', xp=np)[0], .0025)
    np.testing.assert_allclose(coefficients([30., 60.], 'emanuel2003-adapted', xp=np), .002)


def test_capped_roughness_has_log_law_drag():
    # Eq. 8 cap z0=0.00285 m, not a separately fitted drag number.
    cd, _, _ = coefficients([60., 90.], 'gz-option2', xp=np)
    np.testing.assert_allclose(cd, (.4 / np.log(10 / .00285))**2, rtol=1e-10)


def test_inert_serialization_and_no_device_allocation():
    from woof.tc.surface import create_surface
    cfg = RunConfig(nx=12, ny=12, nz=8, dx=1000., dy=1000., ztop=10000., dt=3., run_seconds=6.)
    values = asdict(cfg)
    drop_inert_tc(values)
    assert not any(k.startswith('tc_') for k in values)
    assert create_surface(cfg, {}) is None
    changed = dict(DEFAULTS, tc_spray='bao2000')
    drop_inert_tc(changed)
    assert changed['tc_spray'] == 'bao2000'


@pytest.mark.parametrize('changes,match', [
    ({'tc_fetch_m': float('nan')}, 'finite'),
    ({'tc_horizontal_length_m': -1}, 'nonnegative'),
    ({'tc_wave_roughness': True}, 'explicit'),
    ({'tc_spray': 'bao2000', 'sf_sfclay_physics': 2}, 'MM5'),
    ({'tc_spray_partition': 1.01}, 'between'),
])
def test_invalid_combinations(changes, match):
    cfg = SimpleNamespace(**(dict(DEFAULTS, sf_sfclay_physics=91,
        sf_surface_physics=2, isfflx=1, ocean_config='', diff_opt=2, km_opt=4) | changes))
    with pytest.raises(ValueError, match=match):
        validate_tc(cfg)


@pytest.mark.parametrize('les', [False, True])
def test_generated_hierarchy_validates(les, tmp_path):
    from woof.experiment import build_experiment_from_config_tables
    tables = tomllib.loads(configuration(start=datetime(2022,9,26), latitude=25.,
        longitude=-83., les=les, arm='barr'))
    exp = build_experiment_from_config_tables(tables, source='test', base_dir=tmp_path)
    assert [d.run.dx for d in exp.domains] == ([9000.,3000.,1000.,200.,40.] if les else [9000.,3000.,1000.])


def test_landfall_brackets_and_return_to_sea():
    rows = [dict(time_hours=t, max_wind_ms=u, centre_land_fraction=l)
            for t,u,l in [(0,60,0),(1,54,1),(2,48,1),(3,50,0),(4,40,1)]]
    out = inland_decay(rows)
    assert len(out) == 3
    assert out[0]['landfall_time_bracket_hours'] == [0,1]
    assert out[1]['wind_ratio'] == .8
    assert out[2]['landfall_time_bracket_hours'] == [3,4]
    with pytest.raises(ValueError, match='increasing'):
        inland_decay(rows[::-1])


def test_vortex_hemisphere_and_resolution():
    w = np.zeros((25,25)); zeta = np.zeros_like(w)
    w[12,12] = 21.; zeta[12,15] = -.21
    kwargs = dict(height_m=500., dx_m=100., dy_m=100., xp=np)
    assert vortex_candidates(w,zeta,latitude=-20.,**kwargs)[12,15]
    assert not vortex_candidates(w,zeta,latitude=20.,**kwargs).any()
    with pytest.raises(ValueError, match='unresolved'):
        vortex_candidates(w,zeta,latitude=20.,**(kwargs | dict(dx_m=1000.)))


def test_prepared_state_reusable_across_tc_arms_but_restart_identity_changes():
    from woof.ingest.prepared_cache import effective_prepared_domain_config
    from woof.tc.config import DEFAULTS
    off = {'run': {'dx': 3000., **DEFAULTS}}
    on = {'run': {**off['run'], 'tc_flux_curve': 'gz-option2', 'tc_spray': 'barr2023'}}
    assert effective_prepared_domain_config(off) == effective_prepared_domain_config(on)
    assert off != on
    assert effective_prepared_domain_config(on) != effective_prepared_domain_config(
        {'run': {**on['run'], 'dx': 1000.}})


def test_relocation_preserves_pending_timing_without_reusing_spatial_buffers():
    from woof.tc.receipt import restore_cost
    pending = object()
    meter = SimpleNamespace(calls=17, pending=[pending])
    for move in range(2):
        fresh = SimpleNamespace(cost=object(), spray_diag=object())
        diagnostic = fresh.spray_diag
        restore_cost(SimpleNamespace(tc_surface=fresh), meter)
        assert fresh.cost is meter
        assert fresh.cost.pending == [pending]
        assert fresh.cost.calls == 17 + move
        assert fresh.spray_diag is diagnostic
        fresh.cost.calls += 1
    restore_cost(None, meter)
    restore_cost(SimpleNamespace(tc_surface=None), meter)
    untouched = object()
    driver = SimpleNamespace(tc_surface=SimpleNamespace(cost=untouched))
    restore_cost(driver, None)
    assert driver.tc_surface.cost is untouched


@pytest.mark.parametrize('sfclay', [1, 5, 91])
@pytest.mark.parametrize('lsm', [0, 1, 2, 3, 4])
def test_shared_surface_pairings(sfclay, lsm):
    cfg = SimpleNamespace(**(dict(DEFAULTS, sf_sfclay_physics=sfclay,
        sf_surface_physics=lsm, isfflx=1, ocean_config='', diff_opt=2, km_opt=4)
        | dict(tc_flux_curve='bell2012')))
    validate_tc(cfg)


def test_mpas_shared_exchange_selectors():
    from woof.core.mpas_column_suite import resolve_generation, FROZEN_SELECTION
    for surface in ('revised_mm5', 'mynn'):
        chosen = dict(tc_flux_curve='bell2012', tc_spray='off')
        assert resolve_generation(chosen, microphysics_scheme='wsm6',
            selection=dict(FROZEN_SELECTION, surface_layer_scheme=surface)) == chosen


def test_twelve_three_one_hierarchy(tmp_path):
    from woof.experiment import build_experiment_from_config_tables
    tables = tomllib.loads(configuration(start=datetime(2024, 10, 7),
        latitude=23., longitude=-94., parent_dx_m=12000., arm='cblast'))
    eta = tables['shared']['eta_levels']
    assert len(eta) == tables['shared']['nz'] + 1
    assert eta[0] == 1. and eta[-1] == 0.
    assert all(a > b for a, b in zip(eta, eta[1:]))
    exp = build_experiment_from_config_tables(tables, source='test', base_dir=tmp_path)
    assert [d.run.dx for d in exp.domains] == [12000., 3000., 1000.]
    assert all(d.run.tc_flux_curve == 'bell2012' for d in exp.domains)
