"""Native stream orientation is part of the bitwise replay contract."""
from pathlib import Path
import struct

import numpy as np
import pytest

from tools.cam5_wrf471_oracle.capture import read


def test_native_stream_preserves_fortran_column_order_and_integer_words(tmp_path):
    p = tmp_path / 'cu7-00001-00001-00001-in.bin'
    # Native write A(i,k,j): i changes fastest, then k, then j.
    # Two horizontal axes are deliberately unequal to catch a transpose.
    words = np.arange(12, dtype='<f4')
    surface = np.array([0, -1, 2, -3, 4, -5], dtype='<i4')
    p.write_bytes(struct.pack('<8i3f2i', 20261101, 3, 2, 2, 1, 1, 1, 1,
                             60., 12000., 0., 9, 1)
                  + words.tobytes() + surface.tobytes())
    meta, got = read(p, [dict(name='a', shape='mass', dtype='<f4'),
                        dict(name='kpbl', shape='surface', dtype='<i4')])
    assert meta['nz'] == 2 and meta['ny'] == 2 and meta['nx'] == 3
    np.testing.assert_array_equal(got['a'], [[[0, 1, 2], [6, 7, 8]],
                                           [[3, 4, 5], [9, 10, 11]]])
    np.testing.assert_array_equal(got['kpbl'], [[0, -1, 2], [-3, 4, -5]])
    assert got['a'].flags.c_contiguous and got['kpbl'].flags.c_contiguous


@pytest.mark.parametrize('payload, match', [(b'', 'truncated capture header'),
    (struct.pack('<8i', 0, 1, 1, 1, 1, 1, 1, 1), 'capture magic'),
    (struct.pack('<8i', 20261101, -1, 1, 1, 1, 1, 1, 1), 'capture bounds')])
def test_rejects_unusable_native_header(tmp_path, payload, match):
    p = tmp_path / 'cu7-00001-00001-00001-in.bin'; p.write_bytes(payload)
    with pytest.raises(ValueError, match=match):
        read(p, [])
