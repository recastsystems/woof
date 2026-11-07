"""Terrain drag statics reach prepared tile buffers before any step."""
from types import SimpleNamespace
from collections import defaultdict
import sys

import numpy as np
import pytest

from woof.core.terrain_drag import TerrainDrag
from woof.static.orographic import GWD_FIELDS, required_static_fields


def _drag(topo, gwd, shape=(3, 4)):
    return TerrainDrag(
        topo_wind=topo, gwd_opt=gwd,
        ctopo=np.zeros(shape, np.float32) if topo else None,
        ctopo2=np.zeros(shape, np.float32) if topo else None,
        gwd={name: np.zeros(shape, np.float32)
             for name in GWD_FIELDS[gwd]} if gwd else None,
        kpblmax=7 if gwd == 3 else None)


@pytest.mark.parametrize("topo,gwd", [(0, 0), (1, 0), (2, 1), (0, 3), (1, 3)])
def test_static_geography_contains_every_drag_array(monkeypatch, topo, gwd):
    from woof.core.physics_inventory import terrain_drag_array_shapes
    from tilestream import physics_inventory  # Import before the array-type shim.
    from tilestream.driver import geography_inventory, assert_geography_gathered

    # Only the array type check needs this module on the CPU path.
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(ndarray=np.ndarray))
    drag = _drag(topo, gwd)
    state = SimpleNamespace(p=np.zeros((7, 3, 4), np.float32),
                            physics=SimpleNamespace(terrain_drag=drag))
    got = geography_inventory(state)
    cfg = SimpleNamespace(topo_wind=topo, gwd_opt=gwd, ny=3, nx=4)
    assert {key: value.shape for key, value in got.items()} == (
        terrain_drag_array_shapes(cfg))
    assert all(got[key] is value for key, value in drag.geography().items())
    assert_geography_gathered(state, keys=got)
    if got:
        # Uniform neutral buffers still need the gather. Otherwise their
        # zero statistics would silently disable drag on every rank.
        with pytest.raises(RuntimeError, match="terrain-drag-static"):
            assert_geography_gathered(state, keys=())


@pytest.mark.parametrize("topo,gwd", [(1, 0), (2, 1), (0, 3), (1, 3)])
def test_prepared_factory_builds_matching_neutral_statics(monkeypatch, topo, gwd):
    from woof.core import streaming
    from woof.io import restart
    from tilestream import harness

    cfg = SimpleNamespace(topo_wind=topo, gwd_opt=gwd, ny=3, nx=4)
    driver = SimpleNamespace(terrain_drag=_drag(topo, gwd))
    observed = {}

    def build(tile_cfg, *_args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(existing_scratch=lambda *_: None), None

    monkeypatch.setattr(streaming, "_domain_start_time", lambda *_: None)
    monkeypatch.setattr(streaming, "domain_vertical_coord", lambda *_: object())
    monkeypatch.setattr(streaming, "_impose_domain_setup", lambda *_: 0)
    monkeypatch.setattr(streaming, "prime_lazy_carriers", lambda *_: ())
    monkeypatch.setattr(restart, "lifecycle_window_slots", lambda *_: ())
    monkeypatch.setattr(harness, "neutral_geography", lambda cfg: SimpleNamespace(
        lat=np.zeros((cfg.ny, cfg.nx)), lon=np.zeros((cfg.ny, cfg.nx))))
    monkeypatch.setattr(harness, "make_physics_state", build)
    factory = streaming.prepared_tile_state_factory(
        SimpleNamespace(physics=driver, existing_scratch=lambda *_: None), cfg)
    tile = factory(cfg)
    assert tile._tile_buffer
    statics = observed["terrain_drag_static"]
    assert set(statics) == {"HGT_M", *required_static_fields(topo, gwd)}
    for values in statics.values():
        assert values.shape == (cfg.ny, cfg.nx)
        assert values.dtype == np.float32
        assert not np.any(values)
    driver.terrain_drag = None
    with pytest.raises(streaming.StreamingRefused, match="no terrain-drag geography"):
        factory(cfg)


@pytest.mark.parametrize("start,rows", [(0, 2), (2, 3), (5, 2)])
def test_slab_topo_wind_uses_true_neighbour_rows(monkeypatch, start, rows):
    from woof.core import terrain_drag
    from woof.ingest.prepared_store import _complete_slab_topo_wind

    monkeypatch.setitem(sys.modules, "cupy", np)
    ny, nx = 7, 4
    height = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
    variance = height + 400.0
    mask = np.ones((rows, nx), np.float32)
    mask[:, ::2] = 2.0
    drag = _drag(1, 0, (rows, nx))
    original_ctopo, original_ctopo2 = drag.ctopo, drag.ctopo2
    state = SimpleNamespace(physics=SimpleNamespace(
        terrain_drag=drag, fields={"xland": mask}))
    first, last = max(0, start - 1), min(ny, start + rows + 1)
    before = start - first

    def coefficients(ht, xland, *, topo_wind, var_sso):
        assert topo_wind == 1
        np.testing.assert_array_equal(ht, height[first:last].astype(np.float32))
        np.testing.assert_array_equal(var_sso, variance[first:last])
        np.testing.assert_array_equal(xland[before:before + rows], mask)
        # Distinct values expose selecting the wrong coefficient rows.
        return ht.copy(), ht + np.float32(0.25), np.zeros_like(ht)

    monkeypatch.setattr(terrain_drag, "topo_wind_coefficients", coefficients)
    _complete_slab_topo_wind(
        state, SimpleNamespace(topo_wind=1, nx=nx), {"VAR_SSO": variance},
        SimpleNamespace(terrain_z=height), start, rows, ny)
    assert drag.ctopo is original_ctopo and drag.ctopo2 is original_ctopo2
    np.testing.assert_array_equal(drag.ctopo, height[start:start + rows])
    np.testing.assert_array_equal(drag.ctopo2, height[start:start + rows] + 0.25)


@pytest.mark.parametrize("option", [0, 2])
def test_column_local_topo_options_need_no_neighbour_rebuild(option):
    from woof.ingest.prepared_store import _complete_slab_topo_wind

    _complete_slab_topo_wind(None, SimpleNamespace(topo_wind=option),
                             None, None, 2, 3, 7)


@pytest.mark.parametrize("option", [1, 3])
def test_gwd_runs_on_a_tile_with_the_gathered_columns(monkeypatch, option):
    from woof.core import physics

    monkeypatch.setattr(physics, "cp", np)
    shape = (4, 3, 5)
    plane = np.ones(shape[1:], np.float32)
    driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(_tile_buffer=True, sina=plane, cosa=plane)
    driver.fields = {name: plane for name in ("pblh", "xland", "br")}
    driver.fields["kpbl"] = np.ones(shape[1:], np.int32)
    driver.sase_active = False
    driver.bldt_seconds = 24.0
    rates = {name: np.zeros(shape, np.float32) for name in ("du", "dv")}
    atmosphere = {"u": np.ones(shape, np.float32)}
    called = []

    def apply(atm, du, dv, **kwargs):
        assert atm is atmosphere
        assert kwargs["xland"] is plane and kwargs["kpbl"] is driver.fields["kpbl"]
        assert kwargs["dx"] == 3000.0 and kwargs["dt"] == 24.0
        du[...] = 2.0
        dv[...] = -3.0
        called.append(True)

    drag = SimpleNamespace(gwd_opt=option, apply_gwd=apply)
    driver._apply_gwd(drag, SimpleNamespace(dx=3000.0, spp_pbl=0), rates,
                      atmosphere)
    assert called == [True]
    assert np.all(rates["du"] == 2.0) and np.all(rates["dv"] == -3.0)


def test_topo_wind_reaches_ysu_on_a_tile(monkeypatch):
    from woof.core import physics

    monkeypatch.setattr(physics, "cp", np)
    plane = np.ones((3, 5), np.float32)
    driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(_tile_buffer=True)
    driver.fields = defaultdict(lambda: plane)
    driver.terrain_drag = _drag(1, 0, plane.shape)
    driver.bldt_seconds = 24.0
    driver._wrf_rthraten = None
    driver.rthratenlw = driver.rthratensw = np.zeros((4, 3, 5), np.float32)
    driver.urban = None
    atmosphere = defaultdict(lambda: np.ones((4, 3, 5), np.float32))

    class ReachedKernel(Exception):
        pass

    def launch(*args, **kwargs):
        assert kwargs["topo"][0] is driver.terrain_drag.ctopo
        assert kwargs["topo"][1] is driver.terrain_drag.ctopo2
        assert kwargs["u10_out"].shape == plane.shape
        raise ReachedKernel

    monkeypatch.setattr(physics, "launch_ysu", launch)
    with pytest.raises(ReachedKernel):
        driver._run_ysu(atmosphere, SimpleNamespace(
            ysu_topdown_pblmix=0, ysu_version="wrf_480"))
