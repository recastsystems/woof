"""The production Rust startup must reproduce WRF's pinned table words."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from woof.core import p3_rust, p3_tables


@pytest.fixture(autouse=True)
def require_built_p3_bridge():
    try:
        p3_rust.resolve()
    except FileNotFoundError as error:
        pytest.skip(str(error))


def test_rust_startup_matches_all_native_table_words():
    ice, collection, *rain = p3_rust.init_two_moment(p3_tables.p3_table_root())
    for array, digest in ((ice, p3_tables.FORTRAN_ITAB_SHA256),
                          (collection, p3_tables.FORTRAN_ITABCOLL_SHA256)):
        assert hashlib.sha256(array.astype('<f4').tobytes()).hexdigest() == digest
    raw = np.fromfile(Path(__file__).parent / 'data/oracles/p3/wrf461_rain_tables.bin', '<f4')
    for i, actual in enumerate(rain):
        expected = raw[i*3000:(i+1)*3000].reshape(300, 10, order='F')
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('damage', ['version', 'size', 'digest'])
def test_rust_startup_refuses_changed_authority(tmp_path, damage):
    asset = p3_tables.TABLE_1_2MOM_ASSET
    raw = (Path(p3_tables.p3_table_root()) / asset.filename).read_bytes()
    if damage == 'version':
        raw = raw.replace(b'5.4_2momI', b'5.5_2momI', 1)
        expected = 'names version'
    elif damage == 'size':
        raw = raw[:-1]
        expected = 'has size'
    else:
        raw = raw[:100] + bytes([raw[100] ^ 1]) + raw[101:]
        expected = 'SHA-256'
    (tmp_path / asset.filename).write_bytes(raw)
    with pytest.raises(ValueError, match=expected):
        p3_rust.init_two_moment(str(tmp_path))
