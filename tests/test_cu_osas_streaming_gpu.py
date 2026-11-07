"""OSAS column identity and persistent diagnostics through reused tile buffers."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.gpu
def test_osas_draw_uses_gathered_domain_column_indices():
    import cupy as cp
    from woof.core import column_cumulus_r8 as cc
    from tilestream import driver, gather, spec

    domain = SimpleNamespace(p=cp.empty((3, 12, 16), cp.float32))
    owner = cc.ColumnCumulus(95)
    owner.ensure_carriers(domain)
    domain.physics = SimpleNamespace(cumulus_callable=owner)
    geo = driver.geography_inventory(domain)
    assert set(geo) == {"cumulus/column_indices"}
    want = cc.random_input_host(12 * 16, step=17).reshape(12, 16)
    # Ragged periodic windows exercise both wrap and reusing a buffer.
    for tspec in spec.plan_tiles(16, 12, 7, 5, 2, periodic=True):
        ny, nx = tspec.cny, tspec.cnx
        tile = SimpleNamespace(p=cp.empty((3, ny, nx), cp.float32))
        twin = cc.ColumnCumulus(95)
        twin.ensure_carriers(tile)
        tile.physics = SimpleNamespace(cumulus_callable=twin)
        gather.gather_tile(geo, tile, tspec, cp.cuda.get_current_stream(),
                           inventory_fn=driver.geography_inventory, nz=3)
        got = cp.asnumpy(cc.random_input(
            ny * nx, step=17, module_name="cu_osas",
            column_indices=twin.column_indices)).reshape(ny, nx)
        expected = np.empty((ny, nx), np.float64)
        tspec.apply_gather(want, expected, "mass")
        np.testing.assert_array_equal(got.view(np.uint64), expected.view(np.uint64))


@pytest.mark.gpu
@pytest.mark.parametrize("injected", (False, True))
@pytest.mark.parametrize("warm_steps", (0, 1))
def test_osas_resident_and_reused_host_tiles_have_identical_carriers(injected, warm_steps, tmp_path):
    import cupy as cp
    from woof.core import column_cumulus_r8 as cc, streaming
    from woof import wrf_exact
    from tilestream import physics_inventory
    from tilestream import test_join as join

    cfg = join.join_cfg(nx=80, ny=64, nz=29, rung="mp10", dt=3.,
                        cu_physics=95, cudt_minutes=.1, run_seconds=9.)
    first, _ = join.build_domain(cfg, warmup=0)
    last, _ = join.build_domain(cfg, seed=join.SEED + 1, warmup=0)
    boundaries = join.domain_boundaries(cfg, first, last, seconds=3600.)

    def prepared():
        state, _ = join.build_domain(cfg, boundaries=boundaries, warmup=0)
        if injected:
            state.physics.cumulus_callable.inject_random(
                cp.asarray(cc.random_input_host(cfg.nx * cfg.ny, step=19)))
        if warm_steps:
            # Both cold streaming and resident-to-streamed continuation.
            from woof.core.dycore import step
            step(state, cfg, refl_10cm_due=False)
        return state

    resident, streamed = prepared(), prepared()
    if wrf_exact.ENABLED and not warm_steps:
        from woof.io import restart
        # Capacity exists before a producer has run. A cold checkpoint must
        # retain that distinction, including an injected replay input.
        streaming.prime_lazy_carriers(streamed, cfg)
        path = restart.write_restart(tmp_path / "cold.npz", streamed, cfg)
        assert restart.read_restart_header(path)["wrf_mass_carriers_cold"] is True
        streamed = prepared()
        restart.restore_restart(path, streamed, cfg)
        assert streamed._wrf_mass_carriers_cold is True
    resident_step = streaming.make_stepper(resident, cfg, streaming.OFF)
    options = streaming.StreamingOptions(mode="on", tile_nx=16, tile_ny=16,
                                          nbuffers=1, halo=16, store="host",
                                          max_redundancy=False)
    node = SimpleNamespace(parent=None, cfg=SimpleNamespace(grid_id=1))
    tile_step = streaming.make_stepper(
        streamed, cfg, options, build=streaming.prepared_domain_builder(node))
    for _ in range(3):
        resident_step(resident, cfg, refl_10cm_due=False)
        tile_step(streamed, cfg, refl_10cm_due=False)
    cp.cuda.get_current_stream().synchronize()
    expected = physics_inventory.carrier_inventory(resident)
    actual = tile_step.store
    for name, value in expected.items():
        want = cp.asnumpy(value)
        got = np.asarray(actual[name])
        assert np.isfinite(want).all(), name
        assert want.dtype == got.dtype and want.shape == got.shape, name
        assert want.tobytes() == got.tobytes(), name
    assert streamed.physics.cumulus_callable.scheme.cu_physics == 95
