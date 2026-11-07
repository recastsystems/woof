"""Radiation phase must survive import, scheduling and continuation together."""
from dataclasses import replace
import pytest
from woof.config import RunConfig,validate_radiation_driver_options
from woof.io.restart import configuration_echo,_require_config_match
from woof.ingest.wrfrst import _last_radiation_step
from woof.core.physics import _radiation_step_due

@pytest.mark.parametrize('offset,calls',[(0,[1,6,11]),(-1,[1,5,10])])
def test_continuation_holds_the_actual_previous_radiation_call(offset,calls):
    actual=[s for s in range(1,13) if _radiation_step_due(s,5,1.,offset)]
    assert actual==calls
    assert [_last_radiation_step(s,5,offset) for s in range(1,13)]==[max(c for c in calls if c<=s) for s in range(1,13)]

def test_offset_zero_keeps_old_checkpoint_configuration():
    cfg=RunConfig(nx=8,ny=8,nz=8,dx=3000.,dy=3000.,dt=12.,ztop=20000.,run_seconds=120.)
    echo=configuration_echo(cfg)
    assert 'ra_call_offset' not in echo
    assert configuration_echo(replace(cfg,ra_call_offset=-1))['ra_call_offset']==-1
    _require_config_match(echo,cfg,'older-checkpoint')

@pytest.mark.parametrize('bad',[True,1,-2,0.0])
def test_offset_is_a_native_integer_choice(bad):
    cfg=RunConfig(nx=8,ny=8,nz=8,dx=3000.,dy=3000.,dt=12.,ztop=20000.,run_seconds=120.,ra_call_offset=bad)
    with pytest.raises(ValueError,match='ra_call_offset'):
        validate_radiation_driver_options(cfg)

@pytest.mark.parametrize('offset,calls',[(0,[1,11,21]),(-1,[1,10,20])])
def test_cam_absorption_uses_the_radiation_phase(offset,calls):
    from woof.core.cam3 import absorption_due
    assert [s for s in range(1,22) if absorption_due((s-1)*12.,12.,60.,120.,first=s==1,ra_call_offset=offset)]==calls


def test_old_default_checkpoint_keeps_scalar_transport_on_resume():
    from woof.io.restart import RestartMismatchError
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=3000., dy=3000., dt=12.,
                    ztop=20000., run_seconds=120.)
    echo = configuration_echo(cfg)
    _require_config_match(echo, cfg, "old-default")
    with pytest.raises(RestartMismatchError, match="_legacy_h_sca"):
        _require_config_match(echo, replace(cfg, _legacy_h_sca=False), "old-default")
    explicit = replace(cfg, h_sca_adv_order=5, _legacy_h_sca=False)
    _require_config_match(configuration_echo(explicit), explicit, "explicit-order")
