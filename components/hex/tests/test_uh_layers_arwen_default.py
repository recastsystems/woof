"""The ArWen-physics route writes the 2-5 km updraft-helicity maximum by default, as every MPAS-physics run does.

THE BREAKAGE THIS GUARDS (both-routes matrix, woof hex 0.4.14 candidate, 2026-11-06): the MPAS route rendered
composite_reflectivity_uh, uh_2to5km and 1km_reflectivity_uh2to5km and the WOOF route rendered none of them, because
its history carried no updraft_helicity_max unless the run passed --uh-layers on (global leg: 225 vs 207 maps).
CPU tier: a NumPy stand-in for CuPy; the kernel itself is not launched.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import uh_layers


class _Cp:
    float32 = np.float32
    asarray = staticmethod(np.asarray)
    zeros = staticmethod(np.zeros)
    asnumpy = staticmethod(np.array)  # a device-to-host copy
    ascontiguousarray = staticmethod(np.ascontiguousarray)

    class RawKernel:
        def __init__(self, source, name):
            self.name = name


def _mesh():
    # Two triangles' vertices over three cells (one-based, as the grid file stores them).
    return SimpleNamespace(cellsOnVertex=np.array([[1, 2, 3], [3, 2, 1]]),
                           kiteAreasOnVertex=np.full((2, 3), 0.5, dtype=np.float32),
                           areaCell=np.ones(3, dtype=np.float32))


@pytest.mark.parametrize("value,names", [
    (None, ("updraft_helicity_max",)),
    ("", ("updraft_helicity_max",)),
    ("1", uh_layers.NAMES),
    ("on", uh_layers.NAMES),
    ("0", None),
    ("off", None),
])
def test_the_arwen_route_writes_the_2_to_5_km_maximum_unless_turned_off(monkeypatch, value, names):
    if value is None:
        monkeypatch.delenv(uh_layers.ENV, raising=False)
    else:
        monkeypatch.setenv(uh_layers.ENV, value)
    layers = uh_layers.for_arwen(_Cp(), _mesh(), n_cells=3)
    if names is None:
        assert layers is None
        return
    assert layers.names == names
    layers.acc[:] = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    taken = uh_layers.arwen_history_arrays(layers)
    assert tuple(taken) == names
    np.testing.assert_array_equal(taken["updraft_helicity_max"], [0.0, 1.0, 2.0])   # band 0 is 2-5 km
    assert not layers.acc.any()                                                      # reset at the frame


def test_the_forecast_driver_arms_the_layers_without_the_variable():
    from pathlib import Path
    source = Path(uh_layers.__file__).with_name("drivers").joinpath("run_cuda_v841_forecast.py").read_text()
    arm = source[source.index("uh_frames = {}"):source.index("capture_finish_seconds = 0.0")]
    assert "WOOF_HEX_UH_LAYERS" not in arm and "for_arwen(" in arm
