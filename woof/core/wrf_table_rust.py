"""Setup-only ctypes seam for pinned WRF float32 sequential table records."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os

import numpy as np


@lru_cache(maxsize=1)
def _reader():
    from woof.core.tempo_rust import load

    library = load()
    try:
        marker = library.gpuwm_wrf_table_f32_v1
        marker.restype = ctypes.c_uint32
        if marker() != 1:
            raise AttributeError("table ABI mismatch")
        reader = library.gpuwm_wrf_table_read_f32
    except AttributeError as error:
        raise RuntimeError(
            "Rust WRF table reader is missing; rebuild the bridge with "
            "`cd tools/rustwx; cargo build --release -p rw-tempo --offline`"
        ) from error
    reader.argtypes = (
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    )
    reader.restype = ctypes.c_int32
    return library, reader


def read_f32_records(path, *, sha256: str, count: int, values: int):
    """Rust verifies a single immutable read and fills native float32 storage.

    Host allocation and the buffer view are orchestration. File reads,
    hashing, marker validation and endian decoding all execute in Rust.
    """
    if count <= 0 or values <= 0:
        raise ValueError("WRF table record counts must be positive")
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError("WRF table SHA-256 must be 64 lowercase hex digits")
    library, reader = _reader()
    data = np.empty((count, values), dtype=np.float32)
    encoded_path = os.fsencode(path)
    status = reader(encoded_path, len(encoded_path), sha256.encode("ascii"),
                    count, values, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    data.size)
    if status:
        from woof.core.tempo_rust import _last_error
        raise ValueError(f"WRF table {path}: {_last_error(library)}")
    return data
