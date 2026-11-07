"""Cold and warm CUDA graph captures reproduce the eager production path.

The eager reference used to be a frozen copy of the pre-speedup kernels and
gather; it was retired when the sedimentation became a transliteration of
WRF v4.6.1 (bitwise at stage 1 of tools/nssl2_wrf461_column_oracle), since
it pinned bits WRF does not produce.  A captured graph must still replay the
eager production gather and sedimentation bit for bit, cold or warm, without
touching the gamma-table cache under capture.
"""
import numpy as np
import pytest


@pytest.mark.parametrize("nz", [49, 65])
@pytest.mark.parametrize("warm", [False, True])
def test_graph_capture_matches_eager_base(nz, warm, monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    from woof.core import nssl2_driver_support as driver
    from woof.core.kernels import load_module

    # Preflight may compile modules before capture. No eager driver call or
    # gamma table initialization is needed to capture fresh workspace buffers.
    load_module("nssl2_driver_support").compile()

    rng = np.random.default_rng(1810 + nz)
    shape = (nz, 1, 7)
    rho_host = rng.uniform(0.3, 1.2, shape).astype(np.float32)
    rho = cp.asarray(rho_host)
    dz = cp.asarray(rng.uniform(60, 400, shape).astype(np.float32))
    temperature = cp.asarray(rng.uniform(235, 285, shape).astype(np.float32))
    fields_host = []
    for index in range(16):
        if index < 7:
            values = rng.uniform(0, 0.003, shape).astype(np.float32)
            values[:, :, 0] = 0
        elif index < 14:
            values = rng.uniform(1, 2e4, shape).astype(np.float32)
        else:
            values = fields_host[5 if index == 14 else 6] / np.float32(500)
        fields_host.append(values)
    fields = [cp.asarray(value) for value in fields_host]
    rates = [cp.asarray(rng.uniform(0, 1e-6, shape).astype(np.float32)) for _ in range(4)]
    kwargs = dict(temperature_k=temperature, first_step=True, cu_used=True,
                  qrcuten=rates[0], qscuten=rates[1], qicuten=rates[2], qccuten=rates[3])
    def workspace():
        return driver.NSSL2DriverWorkspace(
            cp.empty((16, *shape), dtype=cp.float32),
            cp.empty((5, 1, 7), dtype=cp.float32), shape,
            cp.empty((1, 7), dtype=cp.float32))
    expected = workspace()
    actual = workspace()
    inputs = [rho, dz, temperature, *fields, *rates]
    inputs_before = [value.get().view(np.uint32).copy() for value in inputs]
    # The eager reference takes the route a capture takes: no gamma table.
    with monkeypatch.context() as patch:
        # The table is keyed by the shape coefficients too (032b43f5a0).
        patch.setattr(driver, "_velocity_gamma_table",
                      lambda device_id, shapes=driver.DEFAULT_SHAPES: None)
        driver.gather_initialize_and_sediment(
            rho, dz, *fields, 15.0, workspace=expected, **kwargs)
    driver._cached_velocity_gamma_table.cache_clear()
    if warm:
        with cp.cuda.Stream(non_blocking=True):
            driver._velocity_gamma_table(cp.cuda.runtime.getDevice())
    cp.cuda.runtime.deviceSynchronize()
    cached_before = driver._cached_velocity_gamma_table.cache_info()
    captured_names = []
    original_get_kernel = driver.get_kernel
    def get_kernel(module, name):
        if cp.cuda.get_current_stream().is_capturing():
            captured_names.append(name)
        return original_get_kernel(module, name)
    monkeypatch.setattr(driver, "get_kernel", get_kernel)
    # The sediment launches resolve through nssl2_shape.shape_kernel since
    # the shape-coefficient port (032b43f5a0); record those too.
    original_shape_kernel = driver.shape_kernel
    def shape_kernel(module, name, *args, **kwargs):
        if cp.cuda.get_current_stream().is_capturing():
            captured_names.append(name)
        return original_shape_kernel(module, name, *args, **kwargs)
    monkeypatch.setattr(driver, "shape_kernel", shape_kernel)
    cached_function = driver._cached_velocity_gamma_table
    def guarded_cache(device_id, shapes=driver.DEFAULT_SHAPES):
        assert not cp.cuda.get_current_stream().is_capturing()
        return cached_function(device_id, shapes)
    monkeypatch.setattr(driver, "_cached_velocity_gamma_table", guarded_cache)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        try:
            driver.gather_initialize_and_sediment(
                rho, dz, *fields, 15.0, workspace=actual, **kwargs)
        finally:
            graph = stream.end_capture()
    assert not any("cached" in name or "parallel" in name or "gamma" in name
                   for name in captured_names)
    assert len(captured_names) == 7
    assert cached_function.cache_info() == cached_before
    for _ in range(2):
        with stream:
            for buffer in (actual.state, actual.category_surface_export, actual.ignored_accumulator):
                buffer.fill(np.float32(np.nan))
            graph.launch(stream=stream)
        stream.synchronize()
        for reference, observed in zip(
                (expected.state, expected.category_surface_export, expected.ignored_accumulator),
                (actual.state, actual.category_surface_export, actual.ignored_accumulator)):
            np.testing.assert_array_equal(reference.get().view(np.uint32), observed.get().view(np.uint32))
    eager = workspace()
    driver.gather_initialize_and_sediment(rho, dz, *fields, 15.0, workspace=eager, **kwargs)
    for reference, observed in zip(
            (expected.state, expected.category_surface_export, expected.ignored_accumulator),
            (eager.state, eager.category_surface_export, eager.ignored_accumulator)):
        np.testing.assert_array_equal(reference.get().view(np.uint32), observed.get().view(np.uint32))
    for before, after in zip(inputs_before, inputs):
        np.testing.assert_array_equal(before, after.get().view(np.uint32))
    cached_function.cache_clear()
