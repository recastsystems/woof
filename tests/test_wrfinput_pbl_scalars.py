from types import SimpleNamespace
import numpy as np
import pytest
from woof.ingest.wrfinput_pbl import ALL_PBL_WRFINPUT,pbl_input_map,restore_pbl_scalars
from woof.ingest.wrfinput import MAPPED_WRFINPUT,WRFINPUT_DIMENSIONS,missing_required_wrfinput

CFG=SimpleNamespace(bl_pbl_physics=16,moist=True,mp_physics=1)

def test_registered_pbl_input_is_mapped_and_partial_pairs_are_refused():
    assert pbl_input_map(CFG)=={'pek_adv':'pek_adv','pep_adv':'pep_adv'}
    assert ALL_PBL_WRFINPUT <= MAPPED_WRFINPUT
    for name in ALL_PBL_WRFINPUT:
        assert WRFINPUT_DIMENSIONS[name]==('bottom_top','south_north','west_east')
        assert name not in missing_required_wrfinput({},CFG)
    assert 'pep_adv' in missing_required_wrfinput({'pek_adv':np.zeros((3,2,4),np.float32)},CFG)


def test_pbl_restore_preserves_every_word_and_rk_copy():
    raw={name:np.linspace(-0.0,0.01,24,dtype=np.float32).reshape(3,2,4) for name in ALL_PBL_WRFINPUT}
    state=SimpleNamespace(**{name:np.zeros((3,2,4),np.float32) for name in (*ALL_PBL_WRFINPUT,*(x+'0' for x in ALL_PBL_WRFINPUT))})
    restore_pbl_scalars(state,raw,CFG,np)
    for name in ALL_PBL_WRFINPUT:
        assert getattr(state,name).tobytes()==raw[name].tobytes()
        assert getattr(state,name+'0').tobytes()==raw[name].tobytes()

@pytest.mark.parametrize('damage',['missing','shape','state'])
def test_pbl_restore_refuses_lost_carrier(damage):
    raw={name:np.zeros((3,2,4),np.float32) for name in ALL_PBL_WRFINPUT}
    state=SimpleNamespace(**{name:np.zeros((3,2,4),np.float32) for name in ALL_PBL_WRFINPUT})
    if damage=='missing':raw.pop('pek_adv')
    if damage=='shape':raw['pek_adv']=raw['pek_adv'][:1]
    if damage=='state':del state.pek_adv
    with pytest.raises(ValueError):restore_pbl_scalars(state,raw,CFG,np)

@pytest.mark.parametrize('selector',[0,1,2,5,7,10,11,12])
def test_other_pbl_schemes_keep_their_input_contract(selector):
    cfg=SimpleNamespace(bl_pbl_physics=selector,moist=True,mp_physics=1)
    assert not pbl_input_map(cfg)
    assert not ALL_PBL_WRFINPUT.intersection(missing_required_wrfinput({},cfg))
    restore_pbl_scalars(SimpleNamespace(),{},cfg,np)


def test_cold_start_with_another_pbl_initializes_registered_i0_carriers():
    state=SimpleNamespace(**{name:np.ones((3,2,4),np.float32) for name in (*ALL_PBL_WRFINPUT,*(x+'0' for x in ALL_PBL_WRFINPUT))})
    restore_pbl_scalars(state,{},CFG,np)
    for name in ALL_PBL_WRFINPUT:
        assert not getattr(state,name).any()
        assert not getattr(state,name+'0').any()
