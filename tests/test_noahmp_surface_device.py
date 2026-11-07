"""Current surface identities stay on the device during selected stepping."""
import numpy as np
import pytest

from conftest import requires_gpu


@requires_gpu
def test_selected_surface_step_is_cuda_graph_capturable(monkeypatch):
    import cupy as cp
    from test_noahmp_runtime import _build
    from woof.core.physics import _prepare_atmosphere
    from woof.core.health_ledger import HealthLedger, deferring

    state, cfg, driver = _build(nx=6, ny=4, opt_btr=2, opt_rsf=2)
    atmosphere = _prepare_atmosphere(state)
    ledger = HealthLedger()
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream, deferring(ledger):
        for _ in range(2):
            driver._run_noahmp(atmosphere, cfg, 2)
        stream.synchronize()
        def refuse(*args, **kwargs):
            raise AssertionError("selected Noah-MP performed a host readback")
        monkeypatch.setattr(cp, "asnumpy", refuse)
        stream.begin_capture()
        driver._run_noahmp(atmosphere, cfg, 2)
        graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    monkeypatch.undo()
    ledger.drain()


@requires_gpu
@pytest.mark.parametrize("field,value", [("xland", 2), ("xice", 1),
                                          ("ivgtyp", 16), ("isltyp", 7)])
@pytest.mark.parametrize("integer_width", [32, 64])
def test_device_layout_reads_current_identity_bytes(field, value, integer_width):
    import cupy as cp
    from woof.core.noahmp_runtime import NoahmpRuntimeParameters
    from woof.core.noahmp_surface_device import layout, parameter_pool
    from woof.core.health_ledger import HealthLedger, deferring

    params = NoahmpRuntimeParameters(opt_btr=2, opt_rsf=2)
    fields = dict(xland=cp.ones((3, 4), dtype=cp.float32),
                  xice=cp.zeros((3, 4), dtype=cp.float32),
                  ivgtyp=cp.full((3, 4), 10, dtype=cp.int32),
                  isltyp=cp.full((3, 4), 6, dtype=f"int{integer_width}"))
    parameter_pool(params, 4)
    ledger = HealthLedger()
    with deferring(ledger):
        original = layout(params, fields, nsoil=4)
        fields[field][1, 2] = value
        changed = layout(params, fields, nsoil=4)
    ledger.drain()
    assert int(original["census"]["land"]) == 12
    legacy = params.surface_layout(fields, nsoil=4, chunk_width=3)
    for name in ("sea_ice", "glacier", "vegtyp", "soiltyp"):
        np.testing.assert_array_equal(changed[name].get(), legacy[name].get())
    assert {k:int(v) for k,v in changed["census"].items()} == legacy["census"]
    active = changed["active"].get()
    np.testing.assert_array_equal(changed["land_j"].get()[active], legacy["land_j"].get())
    np.testing.assert_array_equal(changed["land_i"].get()[active], legacy["land_i"].get())
    pool = changed["pool"]
    rows = changed["row"][changed["active"]]
    expected = params.slab_parameter_chunks(legacy["keys"], legacy["inverse"],
                                          nsoil=4, chunk_width=65536)
    for name, values in pool.items():
        actual = cp.ascontiguousarray(values[rows]).get().view(np.uint32)
        wanted = expected[0][name].get().view(np.uint32)
        np.testing.assert_array_equal(actual, wanted, err_msg=name)


@requires_gpu
@pytest.mark.parametrize("water_columns", [0, 2, 6])
def test_device_surface_matches_compact_written_words(monkeypatch, water_columns):
    """Changing launch capacity cannot change a single model carrier word."""
    import cupy as cp
    from test_noahmp_runtime import _build
    from woof.core.physics import _prepare_atmosphere
    from woof.core import noahmp_surface_device as device
    from woof.core.health_ledger import HealthLedger, deferring

    dense = _build(nx=6, ny=4, water_columns=water_columns, opt_btr=2, opt_rsf=2)
    compact = _build(nx=6, ny=4, water_columns=water_columns, opt_btr=2, opt_rsf=2)
    original = device.layout

    def compact_layout(params, fields, **kwargs):
        result = original(params, fields, **kwargs)
        active = result["active"]
        for key in ("land_j", "land_i", "gather_j", "gather_i", "row"):
            result[key] = result[key][active]
        result["active"] = cp.ones(result["land_j"].size, dtype=cp.bool_)
        return result

    for step in range(1, 4):
        ledger = HealthLedger()
        with deferring(ledger):
            state, cfg, driver = dense
            driver._run_noahmp(_prepare_atmosphere(state), cfg, step)
            with monkeypatch.context() as patch:
                patch.setattr(device, "layout", compact_layout)
                state, cfg, driver = compact
                driver._run_noahmp(_prepare_atmosphere(state), cfg, step)
        ledger.drain()
        for name, value in dense[2].fields.items():
            if isinstance(value, cp.ndarray):
                assert value.get().tobytes() == compact[2].fields[name].get().tobytes(), (step, name)


@requires_gpu
@pytest.mark.parametrize("nx,ny", [(6, 4), (384, 192)])
def test_selected_surface_resident_and_transient_allocations_are_priced(nx, ny):
    import cupy as cp
    from test_noahmp_runtime import _build
    from test_noahmp_column_slab import _peak_demand_hook
    from woof.core.physics import _prepare_atmosphere
    from woof.core.preflight import noahmp_lsm_cache_shapes, noahmp_lsm_transient_shapes
    from woof.core.health_ledger import HealthLedger, deferring

    state, cfg, driver = _build(nx=nx, ny=ny, nz=10, opt_btr=2, opt_rsf=2)
    atmosphere = _prepare_atmosphere(state)
    ledger = HealthLedger()
    with deferring(ledger):
        driver._run_noahmp(atmosphere, cfg, 2)
    ledger.drain()
    pool = next(iter(driver.noahmp_params._device_parameter_pools.values()))
    held = sum(v.nbytes for v in pool[0].values()) + pool[1].nbytes + 4 * 8
    price = sum(int(np.prod(shape)) for shape in noahmp_lsm_cache_shapes(cfg).values())
    assert held <= price
    demand = _peak_demand_hook()
    ledger = HealthLedger()
    with demand, deferring(ledger):
        driver._run_noahmp(atmosphere, cfg, 2)
    ledger.drain()
    transient = sum(int(np.prod(shape)) for shape in noahmp_lsm_transient_shapes(cfg).values())
    assert demand.peak <= transient, (demand.peak, transient)
    print(dict(grid=[nx, ny], held_bytes=held, resident_price=price,
               peak_transient_bytes=demand.peak, transient_price=transient))
