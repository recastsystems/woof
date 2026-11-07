"""Native QNSE can publish +inf detrainment above a stopped plume.

The step-112 native witness is docs/public/receipts/qnse-native-detr-witness.json.
This tests the health contract, not a repair to the diagnostic's words.
"""
import numpy as np
import pytest
from conftest import requires_gpu
from woof.core.health import field_from_array,validate_fields_cpu

@pytest.mark.parametrize('value,flux,healthy',[
    (np.inf,0.,True),(np.inf,1e-20,False),
    (-np.inf,0.,False),(np.nan,0.,False),(1.,0.,True)])
def test_native_inactive_detr_contract(value,flux,healthy):
    values=np.array([value],np.float32)
    before=values.view(np.uint32).copy()
    field=field_from_array('surface.detr_edkf',values,
        auxiliary=np.array([flux],np.float32),aux_mode='inactive-positive-infinity')
    assert validate_fields_cpu([field]).ok == healthy
    np.testing.assert_array_equal(values.view(np.uint32),before)


@requires_gpu
@pytest.mark.gpu
def test_gpu_native_detr_contract_and_first_failure():
    cp=pytest.importorskip('cupy')
    from woof.core.health import StateHealthValidator
    values=cp.array([np.inf,np.inf,-np.inf,np.nan],np.float32)
    flux=cp.array([0.,1e-20,0.,0.],np.float32)
    field=field_from_array('surface.detr_edkf',values,
        auxiliary=flux,aux_mode='inactive-positive-infinity')
    from types import SimpleNamespace
    scratch={}
    def allocate(shape,slot):
        if slot not in scratch:scratch[slot]=cp.zeros(shape,cp.float32)
        return scratch[slot]
    validator=StateHealthValidator(SimpleNamespace(scratch=allocate),
                                  field_provider=lambda:[field])
    result=validator.validate(phase='native-detr-test')
    assert not result.ok and result.first_bad_flat_index==1
    assert bool(cp.isposinf(values[0]))


@pytest.mark.parametrize('device',[False,pytest.param(True, marks=[requires_gpu, pytest.mark.gpu])])
def test_terminal_level_uses_outgoing_flux_and_preserves_words(device):
    xp=pytest.importorskip('cupy') if device else np
    # Both terminal levels have current flux, but only column 0 has
    # exactly zero outgoing flux. The top plane has no outgoing level.
    values=xp.array([[[np.inf,np.inf]],[[np.inf,1.]]],np.float32)
    flux=xp.array([[[1.,1.]],[[0.,1.]]],np.float32)
    before=values.view(xp.uint32).copy()
    field=field_from_array('surface.detr_edkf',values,auxiliary=flux,
        aux_mode='inactive-positive-infinity',plane_size=2)
    if device:
        from woof.core.health import StateHealthValidator
        from types import SimpleNamespace
        scratch={}
        def allocate(shape,slot):
            if slot not in scratch:scratch[slot]=xp.zeros(shape,xp.float32)
            return scratch[slot]
        result=StateHealthValidator(SimpleNamespace(scratch=allocate),
            field_provider=lambda:[field]).validate(phase='terminal-level')
    else:
        result=validate_fields_cpu([field])
    assert not result.ok and result.first_bad_flat_index==1
    assert bool(xp.array_equal(values.view(xp.uint32),before))


def test_streamed_domain_binds_full_outgoing_activity_instead_of_template():
    from test_store_health_gate import _state,_store,_auxiliaries,NZ,NY,NX,ROWS
    from woof.core.health import validate_store_fields
    state,carriers=_state(ROWS)
    state.physics.scheme_dispatch={'bl_pbl_physics':'_run_qnse_pbl'}
    for name in ('massflux_edkf','detr_edkf'):
        array=np.zeros((NZ,ROWS,NX),np.float32)
        state.physics.fields[name]=array
        carriers['fields/'+name]=array
    store=_store()
    store['fields/massflux_edkf']=np.zeros((NZ,NY,NX),np.float32)
    store['fields/detr_edkf']=np.zeros((NZ,NY,NX),np.float32)
    store['fields/detr_edkf'][0,-1,-1]=np.inf
    store['fields/massflux_edkf'][0,-1,-1]=1.
    def scan():
        return validate_store_fields(state,carriers,store,
            domain_shape=(NY,NX),auxiliaries=_auxiliaries())[0]
    assert scan().ok
    store['fields/massflux_edkf'][1,-1,-1]=1.
    result=scan()
    assert not result.ok and result.first_bad_field=='surface.detr_edkf'
