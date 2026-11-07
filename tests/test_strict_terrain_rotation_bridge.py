"""Real Rust boundary checks for strict static rotation initialization."""
import ctypes
import os

import numpy as np
import pytest

from woof.static import rust_bridge


def test_scalar_rotation_keeps_signed_zero_and_endpoint_words():
    values = np.array([[0.0, -0.0], [1.0, -1.0]], dtype=np.float32)
    try:
        result = rust_bridge.scalar_rotation_asin_f32(values)
    except (FileNotFoundError, rust_bridge.StaticBridgeError) as error:
        if os.environ.get("WOOF_REQUIRE_TERRAIN_SCALAR") == "1":
            raise
        pytest.skip(str(error))
    assert result.shape == values.shape
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.view(np.uint32),
                                 [[0, 0x80000000], [0x3fc90fdb, 0xbfc90fdb]])


def test_prior_real_bridge_names_the_terrain_breakage(monkeypatch):
    previous = os.environ.get("WOOF_TERRAIN_PRIOR_BRIDGE")
    if not previous:
        pytest.skip("the previous real bridge is required")
    library = ctypes.CDLL(previous)
    assert not hasattr(library, "gpuwm_static_scalar_asin_f32")
    monkeypatch.setattr(rust_bridge, "load", lambda: library)
    with pytest.raises(rust_bridge.StaticBridgeError, match="SLP_AZI and SWNORM.*Noah"):
        rust_bridge.scalar_rotation_asin_f32(np.zeros((2, 2), np.float32))


def test_unknown_native_asin_profile_still_refuses(monkeypatch):
    """Mutating the real bridge's probe must not admit an unqualified libm."""
    try:
        library = rust_bridge.load()
        original = library.gpuwm_static_scalar_asin_f32
    except (FileNotFoundError, AttributeError, rust_bridge.StaticBridgeError) as error:
        if os.environ.get("WOOF_REQUIRE_TERRAIN_SCALAR") == "1":
            raise
        pytest.skip(str(error))
    pointer = ctypes.POINTER(ctypes.c_float)
    original.argtypes = [pointer, pointer, ctypes.c_size_t]
    original.restype = ctypes.c_int32

    class ChangedIntrinsic:
        def __call__(self, source, destination, count):
            status = original(source, destination, count)
            if status == 0 and count.value == 3:
                words = np.ctypeslib.as_array(destination, shape=(3,)).view(np.uint32)
                words[0] ^= np.uint32(0x10)
            return status

    class ChangedLibrary:
        gpuwm_static_scalar_asin_f32 = ChangedIntrinsic()

        def __getattr__(self, name):
            return getattr(library, name)

    monkeypatch.setattr(rust_bridge, "load", lambda: ChangedLibrary())
    with pytest.raises(rust_bridge.StaticBridgeError, match="cannot match the scalar reference"):
        rust_bridge.scalar_reference_asin_f32(np.zeros((2, 2), np.float32))
def test_scalar_asin_matches_qualified_native_fortran_probe_words():
    values = np.array([0xbd137442, 0xbd222103, 0xbd289002], np.uint32)
    try:
        result = rust_bridge.scalar_reference_asin_f32(values.view(np.float32))
    except (FileNotFoundError, rust_bridge.StaticBridgeError) as error:
        if os.environ.get("WOOF_REQUIRE_TERRAIN_SCALAR") == "1":
            raise
        pytest.skip(str(error))
    actual = tuple(int(v) for v in result.view(np.uint32))
    assert actual in ((0xbd137c6b, 0xbd222bdc, 0xbd289c33),
                      (0xbd137c6a, 0xbd222bdb, 0xbd289c32))
