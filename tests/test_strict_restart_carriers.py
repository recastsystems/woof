"""Every diagnostic read before the next strict diagnosis is checkpointed."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.io import restart

@pytest.mark.parametrize('slot', ['wrf_exact_muts', 'wrf_exact_mu_halo'])
def test_strict_cross_step_plane_is_serialized(slot):
    words = np.array([0x3f800001, 0x3f800002], dtype=np.uint32)
    a = words.view(np.float32).reshape(1, 2)
    state = SimpleNamespace(_scratch={slot: a})
    assert restart.classify_scratch_slot(slot) == 'carry'
    assert slot in restart.CHECKPOINTED_CARRIED_SCRATCH_SLOTS
    assert restart._scratch_manifest(state)['scratch/'+slot] is a

def test_strict_pressure_uses_the_canonical_state_namespace():
    a = np.array([0x3f800001, 0xbf800002], dtype=np.uint32).view(np.float32)
    state = SimpleNamespace(p_perturbation=a)
    assert restart.classify_state_attr('p_perturbation') == 'serialize'
    assert restart.state_manifest(state)['state/p_perturbation'] is a
    assert 'acoustic/p_perturbation' not in restart.state_manifest(state)

def test_default_without_strict_planes_keeps_empty_manifest():
    assert restart._scratch_manifest(SimpleNamespace(_scratch={})) == {}
    assert restart.state_manifest(SimpleNamespace()) == {}

def test_lazy_strict_pressure_shape_is_validated_before_restore():
    state = SimpleNamespace(p=np.zeros((2,3,4),np.float32))
    with pytest.raises(restart.RestartMismatchError):
        restart._validate_member_namespaces(
            {'acoustic/p_perturbation':np.zeros((1,3,4),np.float32)},
            state,None,'test.npz',5)
