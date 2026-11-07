"""CUDA P3 two- and three-moment startup through the Rust rw-p3 bridge.

Python owns allocation and the C ABI only. Rust reads and validates WRF's
table, parses independent table blocks in parallel, and integrates independent
rain-size rows in parallel. There is no Python table processing in mp50/51/52/53.
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

P3_BRIDGE_ENV = "WOOF_P3_BRIDGE"
ABI_VERSION = 1
ITAB_WORDS = 14000
COLL_WORDS = 60000
RAIN_WORDS = 3000
_LIBRARY = None


def library_name():
    if os.name == "nt":
        return "rw_p3.dll"
    return "librw_p3.dylib" if os.uname().sysname == "Darwin" else "librw_p3.so"


def resolve():
    from woof.bridges import default_bridge_dir, packaged_bridge_dir
    from woof.rustwx import crate_dir

    override = os.environ.get(P3_BRIDGE_ENV)
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"{P3_BRIDGE_ENV} names a missing file: {path}")
        return path
    root = Path(__file__).resolve().parents[2]
    candidates = (
        crate_dir() / "target" / "release" / library_name(),
        crate_dir() / "target" / "debug" / library_name(),
        root / "libexec" / "bridges" / library_name(),
        packaged_bridge_dir() / library_name(),
        default_bridge_dir() / library_name(),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "P3 requires the Rust rw-p3 startup bridge; build it with "
        "cd tools/rustwx; cargo build --release -p rw-p3 --offline. Searched: "
        + ", ".join(map(str, candidates)))


def _load():
    global _LIBRARY
    if _LIBRARY is None:
        lib = ctypes.CDLL(str(resolve()))
        lib.gpuwm_p3_abi_version.restype = ctypes.c_uint32
        if lib.gpuwm_p3_abi_version() != ABI_VERSION:
            raise RuntimeError("rw-p3 ABI mismatch; rebuild the P3 bridge")
        for name in ("gpuwm_p3_init_2mom", "gpuwm_p3_init_3mom", "gpuwm_p3_init_table2"):
            call = getattr(lib, name)
            call.argtypes = (ctypes.c_char_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
            call.restype = ctypes.c_int32
        lib.gpuwm_p3_last_error.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        lib.gpuwm_p3_last_error.restype = ctypes.c_size_t
        _LIBRARY = lib
    return _LIBRARY


@lru_cache(maxsize=1)
def pinned_native_rain():
    """WRF 4.7.1 REAL32 startup constants for the pinned GNU13/glibc2.39 oracle."""
    lib = _load()
    try:
        call = lib.gpuwm_p3_native_gnu239_rain
    except AttributeError as exc:
        raise RuntimeError("strict MP52/53 needs the rebuilt rw-p3 bridge") from exc
    call.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    call.restype = ctypes.c_int32
    blob = np.empty(3 * RAIN_WORDS, dtype=np.float32)
    if call(blob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), blob.size):
        raise RuntimeError("rw-p3 rejected its native rain table allocation")
    return tuple(blob[i * RAIN_WORDS:(i + 1) * RAIN_WORDS].reshape(300, 10)
                 for i in range(3))


@lru_cache(maxsize=4)
def init_two_moment(root: str):
    """Return caller-owned float32 arrays in the CUDA table layout."""
    lib = _load()
    words = ITAB_WORDS + COLL_WORDS + 3 * RAIN_WORDS
    blob = np.empty(words, dtype=np.float32)
    path = os.fsencode(root)
    status = lib.gpuwm_p3_init_2mom(
        path, len(path), blob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), words)
    if status:
        length = lib.gpuwm_p3_last_error(None, 0)
        error = ctypes.create_string_buffer(length)
        lib.gpuwm_p3_last_error(error, length)
        message = error.raw.decode("utf-8", "replace")
        if "missing or unreadable" in message:
            raise FileNotFoundError(message)
        if "names version" in message:
            from woof.core.p3_tables import P3TableVersionError
            raise P3TableVersionError(message)
        raise ValueError(message)
    ice = blob[:ITAB_WORDS].reshape(5, 4, 50, 14)
    end = ITAB_WORDS + COLL_WORDS
    coll = blob[ITAB_WORDS:end].reshape(5, 4, 50, 30, 2)
    rain = tuple(blob[end + i * RAIN_WORDS:end + (i + 1) * RAIN_WORDS].reshape(300, 10)
                 for i in range(3))
    return ice, coll, *rain

ITAB_WORDS_3 = 165000
COLL_WORDS_3 = 660000

@lru_cache(maxsize=4)
def init_three_moment(root: str):
    """Return caller-owned float32 arrays in the CUDA table layout."""
    lib = _load()
    words = ITAB_WORDS_3 + COLL_WORDS_3 + 3 * RAIN_WORDS
    blob = np.empty(words, dtype=np.float32)
    path = os.fsencode(root)
    status = lib.gpuwm_p3_init_3mom(
        path, len(path), blob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), words)
    if status:
        length = lib.gpuwm_p3_last_error(None, 0)
        error = ctypes.create_string_buffer(length)
        lib.gpuwm_p3_last_error(error, length)
        message = error.raw.decode("utf-8", "replace")
        if "missing or unreadable" in message:
            raise FileNotFoundError(message)
        if "names version" in message:
            from woof.core.p3_tables import P3TableVersionError
            raise P3TableVersionError(message)
        raise ValueError(message)
    ice = blob[:ITAB_WORDS_3].reshape(11, 5, 4, 50, 15)
    end = ITAB_WORDS_3 + COLL_WORDS_3
    coll = blob[ITAB_WORDS_3:end].reshape(11, 5, 4, 50, 30, 2)
    rain = tuple(blob[end + i * RAIN_WORDS:end + (i + 1) * RAIN_WORDS].reshape(300, 10)
                 for i in range(3))
    return ice, coll, *rain

@lru_cache(maxsize=4)
def init_table2(root: str):
    """Allocate the two collection arrays; Rust parses their pinned bytes."""
    lib = _load()
    words = 250000
    blob = np.empty(2 * words, dtype=np.float32)
    path = os.fsencode(root)
    status = lib.gpuwm_p3_init_table2(
        path, len(path), blob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), blob.size)
    if status:
        length = lib.gpuwm_p3_last_error(None, 0)
        error = ctypes.create_string_buffer(length)
        lib.gpuwm_p3_last_error(error, length)
        message = error.raw.decode("utf-8", "replace")
        if "missing or unreadable" in message:
            raise FileNotFoundError(message)
        if "names version" in message:
            from woof.core.p3_tables import P3TableVersionError
            raise P3TableVersionError(message)
        raise ValueError(message)
    shape = (5, 4, 25, 5, 4, 25)
    return blob[:words].reshape(shape), blob[words:].reshape(shape)
