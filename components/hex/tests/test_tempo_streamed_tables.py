"""Bounded TEMPO packing preserves the fast launch's original table words."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.mpasphys.rrfs import tempo_v303 as tempo

@pytest.mark.parametrize('kind', [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.gpu
def test_streamed_tables_match_full_binary64_pack(monkeypatch, kind):
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA table conversion comparison')
    monkeypatch.setenv('WOOF_HEX_NUMERICS', 'fast')
    monkeypatch.setattr(tempo._hex_numerics, '_mode', 'fast')
    rng = np.random.default_rng(303)
    count = (1 << 20) + 17
    value = rng.uniform(-5, 5, count).astype(kind)
    if kind in (np.float32, np.float64):
        value[:7] = np.array([0., -0., np.finfo(np.float32).tiny / 2,
            -np.finfo(np.float32).tiny / 2, np.finfo(np.float32).tiny,
            1.0000000596046448, -1.0000000596046448], dtype=kind)
    monkeypatch.setattr(tempo, 'TABLE_LAYOUT', [('tiny_table', (count, 1, 1, 1, 1))])
    native = SimpleNamespace(build={'source_pin': next(iter(tempo.TEMPO_FORK_PINS))},
        get_table=lambda key: (value, (1,)))
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'single')
    expected = tempo.fast_tables(tempo._session_tables(native, (False,) * 4, cp))
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'compact')
    actual = tempo._session_tables(native, (False,) * 4, cp)
    assert actual.dtype == cp.float32
    np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),
                                  cp.asnumpy(expected).view(np.uint32))
