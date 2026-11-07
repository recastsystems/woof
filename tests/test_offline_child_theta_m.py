"""Selected thermal interpolation through the real archived-child reader."""
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from conftest import requires_netcdf_bridge
from woof.core import constants as c
from woof.offline_child import (
    OfflineChildContractError, OfflineChildPlacement,
    build_offline_child_domain_state, interpolate_parent_initial_state,
    interpolate_parent_boundary_snapshot,
)
from test_offline_child import _history, _physics_binding, _mp28_child_config

pytestmark = requires_netcdf_bridge


def _moist_archive(path):
    _history(path, datetime(2021, 6, 1), ny=38, nx=40)
    with netCDF4.Dataset(path, "a") as ds:
        ds.USE_THETA_M = 1
        dry = ds.variables["T"][:].filled()
        k, j, i = np.indices(dry.shape[1:])
        dry[0] = 2 + 0.12 * i + 0.04 * j + 3 * k
        qv = np.asarray(0.003 + 0.0001 * i + 0.0002 * k, dtype=np.float32)[None]
        ds.variables["T"][:] = dry
        ds.variables["QVAPOR"][:] = qv
        thm = (dry + np.float32(300)) * (np.float32(1) + np.float32(c.RVOVRD) * qv) - np.float32(300)
        ds.createVariable("THM", "f4", ds.variables["T"].dimensions)[:] = thm


@pytest.mark.parametrize("route", ["inherited", "vertical", "terrain"])
def test_archive_selected_thermal_initial_state_and_boundary(route, tmp_path):
    path = tmp_path / "parent.nc"
    _moist_archive(path)
    placement = OfflineChildPlacement(parent_nx=40, parent_ny=38,
        child_nx=30, child_ny=28, parent_grid_ratio=1,
        i_parent_start=4, j_parent_start=4)
    cfg = replace(_mp28_child_config(mp=8), nx=30, ny=28, use_theta_m=1,
                  nz=3 if route == "vertical" else 2,
                  eta_levels=(1., .7, .3, 0.) if route == "vertical" else None)
    args = dict(physics_binding=_physics_binding(tmp_path), child_cfg=cfg,
                backend="cpu", child_eta_levels=cfg.eta_levels)
    if route == "terrain":
        args["geography"] = SimpleNamespace(
            terrain=np.full((28, 30), 420., dtype=np.float64), receipt={})
    initial = interpolate_parent_initial_state(path, placement, **args)
    boundary = interpolate_parent_boundary_snapshot(path, placement, **args)
    state = build_offline_child_domain_state(initial, cfg, array_module=np)
    assert initial.receipt["use_theta_m"] == boundary.receipt["use_theta_m"] == 1
    assert np.max(np.abs(initial.fields["THM"] - initial.fields["T"])) > 1
    dry = (initial.fields["THM"] + np.float32(300)) / (
        np.float32(1) + np.float32(c.RVOVRD) * initial.microphysics["qv"])
    np.testing.assert_allclose(state.total_theta(), dry, rtol=0, atol=4e-5)
    np.testing.assert_allclose(initial.fields["T"] + np.float32(300), dry, rtol=0, atol=4e-5)
    # The constant parent mass makes coupling and horizontal interpolation
    # commute here; the boundary therefore carries the selected THM column.
    weight = state.c1h[:, None, None] * state.total_mu()[None] + state.c2h[:, None, None]
    np.testing.assert_allclose(boundary.fields["theta"] / weight,
                               initial.fields["THM"], rtol=3e-6, atol=3e-5)
    if route == "vertical":
        assert initial.receipt["vertical_remap"]["max_relative_drift"] < 1e-12


def test_selected_archive_requires_native_thermal_field(tmp_path):
    path = tmp_path / "dry-parent.nc"
    _history(path, datetime(2021, 6, 1), ny=18, nx=20)
    from test_offline_child import _mp28_placement
    cfg = replace(_mp28_child_config(mp=8), use_theta_m=1)
    with pytest.raises(OfflineChildContractError, match="USE_THETA_M=1"):
        interpolate_parent_initial_state(path, _mp28_placement(), child_cfg=cfg,
                                         physics_binding=_physics_binding(tmp_path))
