"""Public domain surface and the pinned earlier downstream import contract."""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from woof.domain import API_VERSION, Domain, WarmBubble, _sounding, _placement_grid


def test_import_without_cupy_or_device():
    code = """
import sys
class NoCupy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cupy' or fullname.startswith('cupy.'):
            raise AssertionError('domain import touched CuPy')
sys.meta_path.insert(0, NoCupy())
from woof.domain import API_VERSION, Domain, DomainSnapshot, WarmBubble
assert API_VERSION == '1.0'
assert 'cupy' not in sys.modules
print('import_without_cupy=True')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "import_without_cupy=True" in result.stdout


INVENTORY = json.loads(Path(__file__).with_name("domain_api_compat.json").read_text())


@pytest.mark.parametrize("entry", INVENTORY,
                         ids=[r["module"] + "." + r["name"] for r in INVENTORY])
def test_earlier_downstream_import_and_signature(entry):
    value = getattr(importlib.import_module(entry["module"]), entry["name"])
    if "fields" in entry:
        signature = inspect.signature(value)
        assert set(entry["fields"]) <= set(signature.parameters)
    if "signature" not in entry:
        return
    old, current = entry["signature"], inspect.signature(value)
    marker = object()
    # Every formerly accepted named parameter and every old positional form.
    posnames = old["positional"]
    positional = [marker] * len(posnames)
    keywords = dict.fromkeys(old["kwonly"], marker)
    current.bind(*positional, **keywords)
    named = dict.fromkeys(posnames[old["posonly"]:] + old["kwonly"], marker)
    current.bind(*([marker] * old["posonly"]), **named)
    # A newly required argument would break the old minimal call.
    current.bind(*([marker] * old["required"]),
                 **dict.fromkeys(old["required_kwonly"], marker))
    if old["kwargs"]:
        assert any(p.kind == p.VAR_KEYWORD for p in current.parameters.values())
    if old["varargs"]:
        assert any(p.kind == p.VAR_POSITIONAL for p in current.parameters.values())


def test_height_pressure_and_file_soundings(tmp_path):
    z = np.array([0, 1000, 3000], dtype=float)
    theta = np.array([300, 305, 320], dtype=float)
    result = _sounding(None, z, None, theta, None, None, None, None, 3000)
    np.testing.assert_array_equal(result[0], z)
    assert result[-1] == 100000
    result = _sounding({"p": [100000, 80000, 50000], "theta": theta},
                       None, None, None, None, None, None, None, 3000)
    assert result[0][0] == 0 and np.all(np.diff(result[0]) > 0)
    path = tmp_path / "input_sounding"
    path.write_text("1000 300 14\n10 301 13 4 -2\n4000 320 1 10 5\n")
    result = _sounding(path, None, None, None, None, None, None, None, 3000)
    assert result[0][0] == 0 and result[1][0] == 300
    assert result[2][0] == 0.014 and result[3][0] == 4


@pytest.mark.parametrize("updates", [
    {"z": [100, 1000]}, {"z": [0, 500]}, {"z": [0, 0]},
    {"theta": [300, float('nan')]}, {"qv": [-1, 0]},
    {"p": [100000, 110000], "z": None}, {"u": [1, 2, 3]},
])
def test_invalid_sounding_refuses_lost_or_invented_atmosphere(updates):
    args = dict(sounding=None, z=[0, 1000], p=None, theta=[300, 310],
                qv=None, u=None, v=None, p_surf=None, ztop=1000)
    args.update(updates)
    with pytest.raises(ValueError):
        _sounding(**args)


def test_public_version_and_bubble_validation():
    assert API_VERSION == "1.0"
    assert callable(Domain.from_state)
    with pytest.raises(ValueError):
        WarmBubble(radii=(0, 1, 1))


@pytest.mark.parametrize("lat,lon", [(35.5, -97.0), (0.0, 0.0), (-35.0, 179.9)])
def test_placement_uses_engine_grid_and_preserves_registration(lat, lon):
    from woof.static.projection import MercatorGrid
    grid = _placement_grid(16, 12, 1000, 1000, lat, lon)
    assert isinstance(grid, MercatorGrid)
    center = grid.ij_to_latlon(8.5, 6.5)
    np.testing.assert_allclose(center, (lat, lon), atol=1.e-12)
    latitude, longitude = grid.latlon_mass()
    assert latitude.shape == longitude.shape == (12, 16)
    sina, cosa = grid.rotation_m()
    np.testing.assert_array_equal(sina, np.zeros((12, 16)))
    np.testing.assert_array_equal(cosa, np.ones((12, 16)))
    assert _placement_grid(16, 12, 1000, 1000, None, None) is None


@pytest.mark.parametrize("lat,lon", [(35, None), (None, -97), (90, 0),
                                      (-90, 0), (float("nan"), 0), (0, 181)])
def test_invalid_placement_cannot_silently_mislocate_radar(lat, lon):
    with pytest.raises(ValueError, match="placement"):
        _placement_grid(16, 12, 1000, 1000, lat, lon)


def test_valid_time_is_utc_epoch_plus_live_model_clock():
    state = SimpleNamespace(elapsed_seconds=2.5)
    cfg = SimpleNamespace(nx=16, ny=12, dx=1000, dy=1000)
    assert Domain(state, cfg).grid is None
    assert Domain(state, cfg).valid_time is None
    epoch = datetime(2020, 12, 31, 23, 59, 59)
    d = Domain(state, cfg, start_time=epoch)
    assert d.valid_time == datetime(2021, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)
    state.elapsed_seconds = 10
    assert d.valid_time == epoch.replace(tzinfo=timezone.utc) + timedelta(seconds=10)
    with pytest.raises(TypeError, match="datetime"):
        Domain(state, cfg, start_time="2020-01-01")
    with pytest.raises(ValueError, match="dimensions/spacing"):
        Domain(state, cfg, grid=_placement_grid(17, 12, 1000, 1000, 35, -97))
