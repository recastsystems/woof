"""Recorded scheme versions survive strict allocation-authority replay."""
import copy

import pytest

from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist, specification
from woof.hex.mpasphys.state_roles import resolve_receipt


@pytest.mark.parametrize('mode,version', [('exact', 'pinned'), ('correct', 'pinned'),
                                         ('correct', 'mynn-v6.1')])
def test_recorded_mynn_selection_is_replayed_and_tampering_refused(mode, version):
    spec = specification(table_key='mpasphys_rrfs_44c0849')
    text = ("&physics config_physics_suite='hrrrv5' /\n&"
            + spec['backend_namelist_group'] + '\n'
            + spec['mynn_versions']['selector'] + '=' + repr(version) + '\n/')
    backends = {slot: 'hex-cuda' for slot in spec['slots']}
    original = resolve_namelist(text, backends=backends, physics_mode=mode, table_key='mpasphys_rrfs_44c0849').receipt()
    replay, _, _ = resolve_receipt(original)
    assert replay.receipt() == original
    assert replay.mynn_version == version
    changed = copy.deepcopy(original)
    changed['submodule_pins']['TEMPO'] = '0' * 40
    with pytest.raises(ValueError, match='disagree'):
        resolve_receipt(changed)
