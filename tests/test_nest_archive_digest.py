"""Moving-nest precipitation archives must survive final state hashing."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.state_digest import _canonical_extra_manifest, _canonical_digest_document
from woof.core.nest_accumulation import PREFIX, META
from woof.core.physics_continuation import STRIP_ACCUMULATORS


def test_archive_payload_and_origin_are_part_of_manifest():
    scratch = {PREFIX + name: np.ones((3, 4), dtype='f4') for name in STRIP_ACCUMULATORS}
    scratch[META] = np.array([2, 3, 0, 0], dtype='i8')
    manifest = _canonical_extra_manifest(SimpleNamespace(_scratch=scratch))
    assert set(manifest) == {'scratch/' + name for name in scratch}
    assert all(manifest['scratch/' + name] is value for name, value in scratch.items())


def test_unregistered_archive_field_is_not_silently_ignored():
    with pytest.raises(RuntimeError, match='not a concrete registered member'):
        _canonical_extra_manifest(SimpleNamespace(_scratch={PREFIX + 'unknown': np.zeros((2, 2))}))
