"""Native NWP planes survive the common checkpoint and tiled carriers."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core import nwp_diagnostics as nwp
from woof.core.state import refresh_model_time
from woof.io import restart
from test_restart import _cfg, _fill_setup, _shim_state


def test_gsd_owns_interval_windows_when_both_packages_are_requested(monkeypatch):
    # The shipped HRRR recipe selects both. Its fork windows also allow
    # sub-minute output, unlike the stock NWP integer-minute divisor.
    cfg = _cfg(mp_physics=28, nwp_diagnostics=1, gsd_diagnostics=1,
               output_interval_s=20)
    assert not nwp.active(cfg)
    assert nwp.scratch_shapes(cfg) == {}
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    before = {}
    from woof.core.gsd_diagnostics import HISTORY
    for index, slot in enumerate(HISTORY):
        value = state.existing_scratch(slot)
        value[...] = np.float32(index + .25)
        before[slot] = value.tobytes()
    # The stock writer must leave every GSD-owned word untouched.
    nwp.update(state, cfg)
    assert {slot: state.existing_scratch(slot).tobytes() for slot in HISTORY} == before


def test_all_nwp_windows_roundtrip_with_nonzero_words(monkeypatch, tmp_path):
    cfg = _cfg(nwp_diagnostics=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    patterns = {}
    for index, name in enumerate((*nwp.HISTORY, 'up_heli_max')):
        pattern = np.arange(24, dtype=np.float32).reshape(4, 6) + np.float32(index + .25)
        source.existing_scratch(name)[...] = pattern
        patterns[name] = pattern
    # Work buffers and the cached step mirror are rebuilt from DomainClock.
    source._native_nwp_prepared = True
    source._nwp_itimestep = 301
    path = restart.write_restart(tmp_path / 'nwp.npz', source, cfg)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for name, pattern in patterns.items():
        assert fresh.existing_scratch(name).tobytes() == pattern.tobytes(), name
    with np.load(path, allow_pickle=False) as archive:
        assert all('scratch/'+name in archive.files for name in patterns)
        assert 'scratch/nwp_rho' not in archive.files


@pytest.mark.parametrize('step,epoch', [(0, 0), (300, 0), (0, 1800), (17, 1800)])
def test_native_step_comes_from_domain_clock_for_restart_and_delayed_nest(step, epoch):
    clock = SimpleNamespace(ticks=epoch+step*12, step_ticks=12, tick_den=1,
                            step_count=step, spec=SimpleNamespace(start_ticks=epoch))
    state = SimpleNamespace(_scratch={'nwp_rho': object()})
    refresh_model_time(state, clock)
    assert state._nwp_itimestep == step+1
    assert state.elapsed_seconds == epoch+step*12
    assert state.domain_start_offset == epoch


def test_native_window_is_not_reset_by_a_history_or_tracker_consumer(monkeypatch):
    from woof.core.uh_diag import reset_up_heli_max
    from woof.core.hail import reset_window
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    state._native_nwp_prepared = True
    for name in ('up_heli_max', 'hail_maxk1', 'hail_max2d'):
        state.existing_scratch(name)[...] = np.float32(17.25)
    reset_up_heli_max(state)
    reset_window(state)
    for name in ('up_heli_max', 'hail_maxk1', 'hail_max2d'):
        assert (state.existing_scratch(name) == np.float32(17.25)).all()


def test_native_wrf_restart_continues_every_nwp_history_window():
    from woof.ingest import wrfrst
    for name in (*nwp.HISTORY, 'up_heli_max'):
        assert wrfrst.WRFRST_DISPOSITIONS[name.upper()].kind == 'read'
        assert wrfrst.scratch_restore_plan(name, 0, nwp_diagnostics=1) == ('fill', name.upper())
    # Retain the historical tracking-only warm-start rule when NWP is off.
    assert wrfrst.scratch_restore_plan('up_heli_max', 0)[0] == 'cold'


@pytest.mark.parametrize('shortwave,swint,calls,fields,expected', [
    (4, 0, 1, ('swddir', 'swddif', 'swddni'), True),
    (5, 0, 2, ('swddir', 'swddif', 'swddni'), True),
    (1, 1, 1, ('swddir', 'swddif', 'swddni'), True),
    (4, 0, 0, ('swddir', 'swddif', 'swddni'), False),
    (4, 0, 1, ('swddir', 'swddif'), False),
    (1, 0, 1, ('swddir', 'swddif', 'swddni'), False),
])
def test_resume_preserves_solar_history_between_radiation_calls(
        shortwave, swint, calls, fields, expected):
    # A nighttime producer can carry all-zero fluxes. Provenance depends
    # on the scheme and producer clock, never on the numerical flux values.
    driver = SimpleNamespace(
        refl_config=_cfg(ra_lw_physics=shortwave, ra_sw_physics=shortwave,
                         swint_opt=swint),
        call_counts={'radiation': calls}, _swdd_from_scheme=False)
    restart._restore_solar_split_provenance(
        driver, {name: np.zeros((4, 6), np.float32) for name in fields})
    assert driver._swdd_from_scheme is expected
