"""Active groundwater state must survive the complete driver, not just a leaf."""
import numpy as np
import pytest
from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize('opt_run', [1, 2])
def test_groundwater_cold_start_and_live_writeback(opt_run, monkeypatch):
    import cupy as cp
    from test_noahmp_runtime import _build
    from woof.core.dycore import step
    from woof.core import noahmp_runtime as runtime
    from woof.io.wrfout import state_frame

    monkeypatch.delenv(runtime.HOST_COLD_START_ENV, raising=False)
    _, _, device = _build(nx=6, ny=4, opt_run=opt_run)
    expected = {name: cp.asnumpy(device.fields[name]).tobytes()
                for name in ('zwtxy', 'waxy', 'wtxy')}
    for name, value in (('zwtxy', 2.5), ('waxy', 4900.), ('wtxy', 4900.)):
        np.testing.assert_array_equal(cp.asnumpy(device.fields[name]), np.float32(value))
    monkeypatch.setenv(runtime.HOST_COLD_START_ENV, '1')
    _, _, host = _build(nx=6, ny=4, opt_run=opt_run)
    assert expected == {name: cp.asnumpy(host.fields[name]).tobytes() for name in expected}
    monkeypatch.delenv(runtime.HOST_COLD_START_ENV)
    state, cfg, live = _build(nx=6, ny=4, opt_run=opt_run)
    before = cp.asnumpy(live.fields['zwtxy']).copy()
    step(state, cfg)
    assert np.any(cp.asnumpy(live.fields['zwtxy']) != before)
    frame = state_frame(state)
    for native, carrier in (('ZWT', 'zwtxy'), ('WA', 'waxy'), ('WT', 'wtxy')):
        np.testing.assert_array_equal(frame[native], cp.asnumpy(live.fields[carrier]))


def test_groundwater_native_feedback_flags_and_allocation():
    from woof.core.noahmp_nest import feedback_fields
    from woof.core.noahmp_runtime import state_fields_2d, NOAHMP_STATE_2D
    assert state_fields_2d() == NOAHMP_STATE_2D
    for option in (1, 2):
        assert {'zwtxy', 'wtxy'} <= set(state_fields_2d(option))
        assert {'zwtxy', 'wtxy', 'waxy'} <= set(feedback_fields(option))


def test_miguez_macho_fan_feedback_matches_native_registry_flags():
    from woof.core.noahmp_nest import feedback_fields
    from woof.core.noahmp_groundwater_slab import CARRIERS
    selected=set(feedback_fields(5)) & set(CARRIERS)
    assert selected == {"zwtxy","ht","fdepthxy","eqzwt","rechclim","riverbedxy"}

@pytest.mark.parametrize("minutes,dt,expected",[(30.,12.,150),(0.,12.,1),(.2,12.,1),(.5,12.,3)])
def test_wtddt_native_positive_nint_cadence(minutes,dt,expected):
    from woof.core.noahmp_groundwater_slab import steps
    assert steps(minutes,dt)==expected

@pytest.mark.parametrize("minutes",[-1.,float("nan"),float("inf"),True])
def test_invalid_wtddt_is_named(minutes):
    from woof.core.noahmp_groundwater_slab import validate
    with pytest.raises(ValueError,match="wtddt"):
        validate(minutes)


def test_stock_run5_and_physics_wtddt_survive_import(tmp_path):
    from test_namelist_import import _import_with
    text,_=_import_with(tmp_path,extra_physics=" wtddt=15., num_soil_layers=4,",extra_input="&noah_mp opt_run=5 /\n",inp_filter=lambda x:x.replace("sf_surface_physics = 2, 2","sf_surface_physics = 4, 4").replace("num_soil_layers = 5","num_soil_layers = 4"))
    assert "opt_run = 5" in text
    assert "wtddt = 15.0" in text


def test_groundwater_cadence_identity_is_absent_off_and_binding_on():
    from dataclasses import replace
    from test_noahmp_no_consumer_knobs import _noahmp_cfg
    from woof.io.restart import configuration_echo,_require_config_match,RestartMismatchError
    off=_noahmp_cfg()
    echo=configuration_echo(off)
    assert "wtddt" not in echo
    _require_config_match(echo,replace(off,wtddt=90.),"pre-groundwater.npz")
    on=replace(off,opt_run=5)
    active=configuration_echo(on)
    assert active["wtddt"]==30.
    with pytest.raises(RestartMismatchError,match="wtddt"):
        _require_config_match(active,replace(on,wtddt=15.),"active-groundwater.npz")
