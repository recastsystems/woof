"""CPU checks for table word transport and the production compile recipe."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from woof.core.wrf_table_rust import read_f32_records

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("strict", [False, True])
def test_compile_recipe_selects_fast_default_and_native_replay(strict):
    env = dict(os.environ, GPUWM_NUMERICS="strict" if strict else "fast",
               GPUWM_WRF_EXACT="1" if strict else "0", CUDA_VISIBLE_DEVICES="-1")
    code = """
from woof.core.kernels import module_source, module_options
source=module_source('ishmael')
options=module_options('ishmael')
assert '--ftz=false' in options
assert ('--fmad=false' in options) == STRICT
assert ('#define ISHMAEL_FAST_NUMERICS 1' in source) != STRICT
""".replace("STRICT", repr(strict))
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)


@pytest.fixture
def table(tmp_path):
    # Includes signed zeros, subnormal moments, infinities and a NaN payload.
    words = np.array([0, 0x80000000, 1, 0x3f800000, 0x7f800000, 0x7fc01234], ">u4")
    raw = b"".join((8).to_bytes(4, "big") + words[i:i+2].tobytes()
                   + (8).to_bytes(4, "big") for i in range(0, 6, 2))
    path = tmp_path / "records.bin"
    path.write_bytes(raw)
    return path, raw, words.astype(np.uint32)


@pytest.fixture
def rust_bridge():
    from woof.core.wrf_table_rust import _reader
    try:
        _reader()
    except FileNotFoundError:
        pytest.skip("rw-tempo bridge must be built for Rust table tests")


def test_table_preserves_all_value_words(table, rust_bridge):
    path, raw, words = table
    got = read_f32_records(path, sha256=hashlib.sha256(raw).hexdigest(), count=3, values=2)
    np.testing.assert_array_equal(got.view(np.uint32).ravel(), words)


def test_corrupt_record_marker_is_rejected_even_with_matching_hash(table, rust_bridge):
    path, raw, _ = table
    bad = bytearray(raw)
    bad[15] ^= 1
    path.write_bytes(bad)
    with pytest.raises(ValueError, match="markers"):
        read_f32_records(path, sha256=hashlib.sha256(bad).hexdigest(), count=3, values=2)


def test_table_wrong_digest_is_rejected(table, rust_bridge):
    path, _, _ = table
    with pytest.raises(ValueError, match="SHA-256"):
        read_f32_records(path, sha256="0" * 64, count=3, values=2)


def test_truncated_table_is_rejected(table, rust_bridge):
    path, raw, _ = table
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="bytes"):
        read_f32_records(path, sha256=hashlib.sha256(raw[:-1]).hexdigest(), count=3, values=2)
