"""The MPAS-suite wind reconstruction names the check that raised its one invalid word.

Brought from fork jacobbieker/woof PR 495 (Apache-2.0).
"""

import types

import numpy as np
import pytest

from woof.hex.mpasphys.runtime_v842 import wind_reconstruction_refusal

NAMES = ("edges_on_cell", "n_edges_on_cell", "coeffs_reconstruct", "lat_cell", "lon_cell")

# The refusal takes the run's array module.  A host adapter covers it on a CPU-only runner; CuPy runs it too.
HOST = types.SimpleNamespace(array_equal=np.array_equal, isfinite=np.isfinite, uint8=np.uint8,
                             asarray=np.asarray, asnumpy=np.asarray)


def _device():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.asarray(np.zeros(1, np.float32))
    except Exception as error:  # no card on this runner
        pytest.skip(f"no CUDA device: {error}")
    return cupy


@pytest.fixture(params=["host", "device"])
def cp(request):
    return HOST if request.param == "host" else _device()


def _geometry(cp):
    geometry = types.SimpleNamespace()
    for name in NAMES:
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        setattr(geometry, f"_{name}", cp.asarray(values))
        setattr(geometry, f"_sealed_{name}", cp.asarray(values.copy()))
    return geometry


def test_a_non_finite_normal_velocity_is_not_called_a_geometry_fault(cp):
    velocity = np.zeros((80, 10), np.float32)
    velocity[78, 3] = np.nan
    velocity[79, 4] = np.inf
    message = wind_reconstruction_refusal(cp, _geometry(cp), cp.asarray(velocity), 1)
    assert "non-finite normal velocity" in message
    assert "2 edge values on 2 levels (1-based: 79, 80)" in message
    assert "geometry equals its sealed copy" in message
    assert "differs" not in message


def test_a_changed_geometry_names_the_array(cp):
    geometry = _geometry(cp)
    geometry._sealed_coeffs_reconstruct = cp.asarray(np.ones((3, 4), np.float32))
    message = wind_reconstruction_refusal(cp, geometry, cp.asarray(np.zeros((80, 10), np.float32)), 1)
    assert "differs from its sealed copy in coeffs_reconstruct" in message


def test_neither_is_reported_as_an_invalid_coefficient_or_index(cp):
    message = wind_reconstruction_refusal(cp, _geometry(cp), cp.asarray(np.zeros((80, 10), np.float32)), 1)
    assert "normal velocity is finite now" in message
