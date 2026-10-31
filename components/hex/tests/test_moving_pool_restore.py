from types import SimpleNamespace
import numpy as np
import pytest

from woof.hex.mpasphys.moving_mpi_v842 import PoolWords


@pytest.mark.parametrize('padding', [0, 1, 3])
@pytest.mark.parametrize('kind', ['real', 'integer'])
def test_bulk_restore_matches_cold_copy_cast_put_and_owns_words(padding, kind):
    dtype = np.float32 if kind == 'real' else np.int32
    row = dict(pool='p', pool_name='field', name='field', shape=[4, 9+padding],
               kind=kind, nlevels=2)
    cold = np.arange(4*(9+padding), dtype=dtype).reshape(4, 9+padding)
    class Hydration:
        def field_buffer(self, *args):
            return cold
    actual = PoolWords(SimpleNamespace(fields=[row]), Hydration())
    expected = PoolWords(SimpleNamespace(fields=[row]), Hydration())
    source = np.linspace(-2., 2., 36).reshape(4, 9)
    if kind == 'real':
        source[0, 0] = -0.
    jobs = []
    for level in (1, 2):
        words = source+level
        old = expected.get_field('p', 'field', level=level)
        old[..., :9] = words.astype(dtype)
        expected.put_field('p', 'field', old, level=level, nlevels=2)
        jobs.append((row, level, True, words))
    actual.restore_mapped_fields(jobs, 9)
    for key, words in expected.written_words().items():
        assert actual.data[key].tobytes() == words.tobytes()
        assert actual.data[key].flags.f_contiguous
    for job in jobs:
        job[-1][:] = 77
    assert actual.data['p', 'field', 1][0, 0] != 77
    assert np.array_equal(cold, np.arange(cold.size, dtype=dtype).reshape(cold.shape))


def test_bulk_restore_keeps_unset_nan_payload_and_rejects_new_nonfinite():
    row = dict(pool='p', pool_name='f', name='f', shape=[1, 3], kind='real', nlevels=1)
    words = np.array([[0x7fa12345, 0xffc07654, 0x80000000]], dtype=np.uint32).view(np.float32)
    pool = PoolWords(SimpleNamespace(fields=[row]))
    pool.restore_mapped_fields([(row, 1, False, words)], 3)
    assert pool.data['p', 'f', 1].tobytes() == words.tobytes()
    with pytest.raises(ValueError, match='non-finite'):
        pool.restore_mapped_fields([(row, 1, True, words)], 3)
