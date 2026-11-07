"""Terrain-drag inputs must follow domain geography into every rank halo."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from woof.static.orographic import GWD_FIELDS, required_static_fields
from tilestream import driver, multigpu


@pytest.fixture
def host_arrays(monkeypatch):
    # These checks inspect the CPU geography contract, without a CUDA context.  The geography check imports
    # woof.io.restart (-> model -> state) lazily; import it before the stub, as a full suite run has, because
    # state binds cupy at import and the stub has no float32 (red whenever this file ran alone).
    import woof.io.restart  # noqa: F401
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(ndarray=()))


def _host_state(topo, gwd, ny=29, nx=37):
    plane = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
    drag = SimpleNamespace(
        topo_wind=topo, gwd_opt=gwd,
        ctopo=plane + 0.25 if topo else None,
        ctopo2=plane + 0.5 if topo else None,
        gwd={name: plane + index + 1 for index, name in
             enumerate(GWD_FIELDS.get(gwd, ()))},
        kpblmax=12 if gwd == 3 else None)
    return SimpleNamespace(p=np.empty((4, ny, nx), np.float32),
                           physics=SimpleNamespace(terrain_drag=drag))


@pytest.mark.parametrize("topo", [0, 1, 2])
@pytest.mark.parametrize("gwd", [0, 1, 3])
def test_geography_carries_every_drag_array_without_recomputing(host_arrays, topo, gwd):
    state = _host_state(topo, gwd)
    actual = driver.geography_inventory(state)
    drag = state.physics.terrain_drag
    expected = {f"terrain_drag/gwd/{name}": value
                for name, value in drag.gwd.items()}
    if topo:
        expected.update({"terrain_drag/ctopo": drag.ctopo,
                         "terrain_drag/ctopo2": drag.ctopo2})
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] is value, name
    # kpblmax is vertical setup, not a horizontal plane to window.
    assert not any("kpblmax" in name for name in actual)


@pytest.mark.parametrize("grid", [(1, 2), (2, 1), (2, 2)])
def test_gsl_geography_windows_include_every_ragged_halo_cell(host_arrays, grid):
    state = _host_state(1, 3)
    geography = driver.geography_inventory(state)
    assert len(geography) == 22
    ny, nx = state.p.shape[-2:]
    specs = multigpu.plan_split(nx, ny, 5, gy=grid[0], gx=grid[1], periodic=False)
    for window in specs:
        for name, source in geography.items():
            tile = np.full((window.cny, window.cnx), np.nan, source.dtype)
            for transfer in window.gather("mass"):
                tile[transfer.tile_key] = source[transfer.full_key]
            expected = source[window.cj0:window.cj0 + window.cny,
                              window.ci0:window.ci0 + window.cnx]
            assert tile.tobytes() == expected.tobytes(), (grid, window, name)
            assert np.isfinite(tile).all(), (grid, window, name)


@pytest.mark.parametrize("missing", ["terrain_drag/ctopo", "terrain_drag/ctopo2",
                                    "terrain_drag/gwd/VARLS", "terrain_drag/gwd/CONSS",
                                    "terrain_drag/gwd/OA4SS", "terrain_drag/gwd/OL1LS"])
def test_missing_drag_geography_is_a_named_failure(host_arrays, missing):
    state = _host_state(1, 3)
    keys = set(driver.geography_inventory(state)) - {missing}
    with pytest.raises(driver.TiledRunError, match="terrain_drag"):
        driver.assert_geography_gathered(state, keys=keys)


@pytest.mark.parametrize("topo,gwd", [(0, 1), (0, 3), (1, 0), (2, 0), (1, 3), (2, 3)])
def test_tile_constructor_allocates_only_requested_static_planes(topo, gwd):
    from woof.core.streaming import _tile_terrain_drag_static

    cfg = replace(multigpu.forced_config(37, 29, 8), topo_wind=topo, gwd_opt=gwd)
    terrain = np.zeros((cfg.ny, cfg.nx), np.float64)
    static = _tile_terrain_drag_static(cfg, terrain)
    assert set(static) == {"HGT_M", *required_static_fields(topo, gwd)}
    assert static["HGT_M"] is terrain
    for name in required_static_fields(topo, gwd):
        assert static[name].shape == terrain.shape
        assert static[name].dtype == np.float32
        assert np.count_nonzero(static[name]) == 0


def test_disabled_tile_drag_allocates_no_static_planes():
    from woof.core.streaming import _tile_terrain_drag_static

    cfg = multigpu.forced_config(37, 29, 8)
    terrain = np.zeros((29, 37))
    assert _tile_terrain_drag_static(cfg, terrain) is None


def test_static_drag_is_priced_once_on_host_and_never_as_seam_traffic():
    from woof.core.devices import DeviceOptions
    from woof.core.devices_memory import estimate_devices, inventory_shapes
    from woof.experiment import experiment_from_run_config

    cfg = replace(multigpu.forced_config(120, 90, 12),
                  bl_pbl_physics=1, sf_sfclay_physics=1)
    active = replace(cfg, gwd_opt=3)
    options = DeviceOptions(count=2, ids=(0, 1))

    def price(run):
        exp = experiment_from_run_config(run, datetime(2026, 1, 1, tzinfo=timezone.utc))
        return estimate_devices(replace(exp, devices=options))

    baseline, dragged = price(cfg), price(active)
    assert dragged["host_store_bytes"] - baseline["host_store_bytes"] == 20 * cfg.nx * cfg.ny * 4
    assert not any("terrain_drag/" in name for name in inventory_shapes(active))
    assert dragged["host_staging_bytes"] == baseline["host_staging_bytes"]


def test_topo_slab_context_has_a_transient_memory_price():
    from woof.core.physics_inventory import terrain_drag_transient_shapes

    cfg = replace(multigpu.forced_config(37, 3, 8), topo_wind=1, bl_pbl_physics=1)
    assert terrain_drag_transient_shapes(cfg)["terrain_drag/topo_work"] == (7, 5, 37)
    assert terrain_drag_transient_shapes(replace(cfg, topo_wind=2))[
        "terrain_drag/topo_work"] == (3, 3, 37)
