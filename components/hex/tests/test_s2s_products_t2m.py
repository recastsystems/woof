"""S2S products read 2 m temperature from either physics route's history (hex 0.4.15 fit16-rows).

THE BREAKAGE THIS PINS (verification host, 2026-11-07): a 7-day 60 km S2S run on the MPAS-physics route (the hex default)
finished its forecast and export, then stopped in products with KeyError: 't2'; that route writes MPAS-A's t2m.
"""
import numpy as np
import pytest
from netCDF4 import Dataset

from woof.hex.s2s_products import read_native


def _history(path, temperature_name):
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", 3)
        ds.createDimension("nVertLevels", 2)
        for name, dims in ((temperature_name, ("Time", "nCells")), ("rainc", ("Time", "nCells")),
                           ("rainnc", ("Time", "nCells")), ("pressure", ("Time", "nCells", "nVertLevels")),
                           ("theta", ("Time", "nCells", "nVertLevels")), ("qv", ("Time", "nCells", "nVertLevels")),
                           ("uReconstructZonal", ("Time", "nCells", "nVertLevels")),
                           ("uReconstructMeridional", ("Time", "nCells", "nVertLevels"))):
            variable = ds.createVariable(name, "f4", dims)
            variable[0] = np.full(variable.shape[1:], 290.0 if name == temperature_name else 1.0, np.float32)


@pytest.mark.parametrize("name", ["t2", "t2m"])
def test_two_metre_temperature_from_either_route(tmp_path, name):
    path = tmp_path / f"history-{name}.nc"
    _history(path, name)
    data = read_native(path)
    assert np.allclose(np.asarray(data["t2"]), 290.0)
    assert set(data) == {"t2", "rainc", "rainnc", "pressure", "theta", "qv", "u", "v"}
