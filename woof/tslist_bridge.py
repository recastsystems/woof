"""Batch-only bindings to the Rust station reader and writer.

The owner supplies a Rust-validated placement plan. This module only marshals
metadata, pinned-array addresses and opaque checkpoint receipts across the ABI.
No weather arithmetic, formatting or file processing happens in Python.
"""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import sys


BRIDGE_ENV = "WOOF_TSLIST_LIBRARY"
ABI_VERSION = 1


def library_name():
    return ("gpuwm_tslist.dll" if sys.platform == "win32" else
            "libgpuwm_tslist.dylib" if sys.platform == "darwin" else
            "libgpuwm_tslist.so")


def resolve():
    from woof.bridges import find_artifact
    selected = find_artifact(BRIDGE_ENV, library_name(),
                             crate_relative="tools/tslist_bridge", executable=False)
    if selected is None:
        raise RuntimeError(
            "Rust tslist writer is not built. Build tools/tslist_bridge with "
            "cargo build --release --offline and set WOOF_TSLIST_LIBRARY to its library.")
    return selected


def load(path=None):
    selected = Path(path) if path is not None else resolve()
    library = ct.CDLL(str(selected.resolve()))
    library.gpuwm_tslist_abi_version.restype = ct.c_uint32
    if library.gpuwm_tslist_abi_version() != ABI_VERSION:
        raise RuntimeError("unsupported Rust tslist ABI")
    library.gpuwm_tslist_open.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_uint32,
        ct.c_uint8, ct.c_size_t, ct.c_char_p, ct.c_char_p]
    library.gpuwm_tslist_open.restype = ct.c_void_p
    library.gpuwm_tslist_rewind_initial.argtypes = library.gpuwm_tslist_open.argtypes[:-1]
    library.gpuwm_tslist_rewind_initial.restype = ct.c_void_p
    library.gpuwm_tslist_validate.argtypes = library.gpuwm_tslist_open.argtypes
    library.gpuwm_tslist_validate.restype = ct.c_int32
    library.gpuwm_tslist_write.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_size_t]
    library.gpuwm_tslist_write.restype = ct.c_int32
    library.gpuwm_tslist_checkpoint.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
    library.gpuwm_tslist_checkpoint.restype = ct.c_ssize_t
    library.gpuwm_tslist_close.argtypes = [ct.c_void_p]
    library.gpuwm_tslist_close.restype = None
    library.gpuwm_tslist_last_error.argtypes = [ct.c_void_p, ct.c_size_t]
    library.gpuwm_tslist_last_error.restype = ct.c_size_t
    library.gpuwm_tslist_read.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_uint32,
                                         ct.c_void_p, ct.c_size_t]
    library.gpuwm_tslist_read.restype = ct.c_ssize_t
    library.gpuwm_tslist_place.argtypes = [ct.c_char_p, ct.c_void_p, ct.c_void_p,
        ct.c_size_t, ct.c_int32, ct.c_int32, ct.c_void_p, ct.c_size_t]
    library.gpuwm_tslist_place.restype = ct.c_ssize_t
    return library


def _error(library):
    size = library.gpuwm_tslist_last_error(None, 0)
    buffer = ct.create_string_buffer(size)
    library.gpuwm_tslist_last_error(buffer, size)
    return RuntimeError(buffer.value.decode("utf-8", errors="replace"))


def _text_result(library, function, *args):
    size = function(*args, None, 0)
    if size < 0:
        raise _error(library)
    buffer = ct.create_string_buffer(size)
    if function(*args, buffer, size) != size:
        raise _error(library)
    return buffer.value.decode("ascii")


def read_stations(path, maximum, *, ideal=False, library=None):
    """Rust performs file I/O and the native fixed-column READ."""
    lib = load(library)
    mode, plan = _text_result(lib, lib.gpuwm_tslist_read,
        os.fsencode(path), maximum, bool(ideal)).split("\n", 1)
    return mode == "1", plan


def place_stations(plan, x, y, nx, ny, *, library=None):
    """Rust rounds native REAL coordinates and selects each containing grid."""
    lib = load(library)
    if x.dtype.name != "float64" or y.dtype.name != "float64" or x.shape != y.shape:
        raise ValueError("tslist placement requires paired float64 coordinates")
    return _text_result(lib, lib.gpuwm_tslist_place, plan.encode("ascii"),
        x.ctypes.data, y.ctypes.data, x.size, nx, ny)


def validate_files(saved):
    library = load()
    if library.gpuwm_tslist_validate(os.fsencode(saved["directory"]),
            saved["plan"].encode("ascii"), bool(saved["geographic"]),
            saved["domain"], saved["levels"], saved["start"].encode("ascii"),
            saved["files"].encode("ascii")):
        raise _error(library)


class RustTslistWriter:
    """One serialized worker owns write_batch; drain before receipt or close."""
    def __init__(self, directory, placements, *, geographic, domain, levels,
                 start, saved=None, library=None, rewind_initial=False):
        if type(domain) is not int or not 1 <= domain <= 99:
            raise ValueError("tslist domain must be 1-99")
        if type(levels) is not int or not 1 <= levels <= 999:
            raise ValueError("tslist levels must be 1-999")
        self.library = load(library)
        if rewind_initial and saved is not None:
            raise ValueError("initial tslist rollback cannot replace a checkpoint receipt")
        args = (
            os.fsencode(directory), placements.encode("ascii"), bool(geographic),
            domain, levels, start.encode("ascii"))
        if rewind_initial:
            self.handle = self.library.gpuwm_tslist_rewind_initial(*args)
        else:
            self.handle = self.library.gpuwm_tslist_open(*args,
                None if saved is None else saved.encode("ascii"))
        if not self.handle:
            raise _error(self.library)

    def write_batch(self, values, samples):
        if not self.handle:
            raise RuntimeError("Rust tslist writer is closed")
        if (str(values.dtype) != "float32" or not values.flags.c_contiguous
                or type(samples) is not int or samples < 0):
            raise ValueError("tslist writer requires a contiguous float32 batch")
        if self.library.gpuwm_tslist_write(self.handle, values.ctypes.data, values.size, samples):
            raise _error(self.library)

    def checkpoint(self):
        if not self.handle:
            raise RuntimeError("Rust tslist writer is closed")
        needed = self.library.gpuwm_tslist_checkpoint(self.handle, None, 0)
        if needed < 0:
            raise _error(self.library)
        buffer = ct.create_string_buffer(needed)
        if self.library.gpuwm_tslist_checkpoint(self.handle, buffer, needed) != needed:
            raise _error(self.library)
        return buffer.value.decode("ascii")

    def close(self):
        if self.handle:
            self.library.gpuwm_tslist_close(self.handle)
            self.handle = None
