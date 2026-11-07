"""A moved precipitation archive remains audited state at the final digest."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof import state_digest
from woof.core.nest_accumulation import META,PREFIX
from woof.core.physics_continuation import STRIP_ACCUMULATORS


def test_all_carried_nest_archives_pass_the_digest_census():
    scratch={PREFIX+slot:np.ones((4,5),dtype=np.float32) for slot in STRIP_ACCUMULATORS}
    scratch[META]=np.array([1,0,0,0],dtype=np.int64)
    manifest=state_digest._canonical_extra_manifest(SimpleNamespace(_scratch=scratch))
    assert set(manifest)=={'scratch/'+slot for slot in scratch}
    for slot,value in scratch.items():
        assert manifest['scratch/'+slot] is value


def test_unknown_archive_fields_still_fail_the_census():
    with pytest.raises(RuntimeError,match='not a concrete registered member'):
        state_digest._canonical_extra_manifest(SimpleNamespace(_scratch={PREFIX+'unknown':np.zeros((4,5))}))


def test_restart_archive_uses_saved_extents_and_rejects_bad_origin():
    from woof.io.restart import _validated_nest_archive, RestartMismatchError
    state=SimpleNamespace(u=np.zeros((2,4),dtype=np.float32),p=np.zeros((3,2,3),dtype=np.float32),
                          _scratch={PREFIX+'mp_graupelnc':np.zeros((2,5),dtype=np.float32)})
    plane=np.arange(20,dtype=np.float32).reshape(4,5)
    source={'scratch/'+PREFIX+'mp_graupelnc':plane,'scratch/'+META:np.array([1,1,0,0],dtype=np.int64)}
    archive=_validated_nest_archive(source,state)
    assert archive[PREFIX+'mp_graupelnc'] is plane
    assert state._scratch[PREFIX+'mp_graupelnc'].shape==(2,5)
    source['scratch/'+META]=np.array([4,1,0,0],dtype=np.int64)
    with pytest.raises(RestartMismatchError,match='outside its saved lattice'):
        _validated_nest_archive(source,state)
