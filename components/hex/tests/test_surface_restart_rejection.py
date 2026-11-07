"""Rejected ocean restarts must not overwrite the live surface pool.

These exercise host-side validation only, without a CUDA allocation or launch.
The copy API raises if a rejected checkpoint reaches it.
"""
from datetime import datetime
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from woof.hex.arwen_surface_update import ArwenSurfaceUpdate


@pytest.mark.parametrize("mismatch", ["mode", "parameters", "shape", "dtype"])
def test_incompatible_restart_rejected_before_first_copy(monkeypatch, mismatch):
    def unexpected_copy(value):
        pytest.fail("rejected checkpoint reached the device copy API")

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(asarray=unexpected_copy))
    service = ArwenSurfaceUpdate.__new__(ArwenSurfaceUpdate)
    service.start = datetime(2022, 9, 26)
    service.n = 2
    service.waves = None
    service.reader = SimpleNamespace(times=[service.start])
    service.owned = {"sst": np.array([300, 301], dtype="f4"),
                     "ocean_temperature": np.array([299, 300], dtype="f8")}
    service.ocean = SimpleNamespace(options={"depth_m": 50}, steps=9)
    saved = dict(identity=dict(albedo_initialization="fractional-ice-v1",
        start=service.start.isoformat(), cells=service.n,
        times=[service.start.isoformat()]),
        arrays={key: value + 1 for key, value in service.owned.items()},
        ocean=dict(options={"depth_m": 50}, steps=12))
    if mismatch == "mode":
        saved["ocean"] = None
    elif mismatch == "parameters":
        saved["ocean"]["options"]["depth_m"] = 25
    elif mismatch == "shape":
        saved["arrays"]["ocean_temperature"] = np.zeros(3, dtype="f8")
    else:
        saved["arrays"]["ocean_temperature"] = np.zeros(2, dtype="f4")
    original = {key: value.tobytes() for key, value in service.owned.items()}
    with pytest.raises(ValueError, match="ocean mode|ocean parameters|layout"):
        service.restore(saved)
    assert {key: value.tobytes() for key, value in service.owned.items()} == original
    assert service.ocean.steps == 9
