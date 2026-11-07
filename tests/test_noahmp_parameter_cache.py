"""Parameter reuse must follow the complete current column identity mapping."""

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.noahmp_runtime import NoahmpRuntimeParameters


@requires_gpu
@pytest.mark.parametrize("chunk_width", [1, 3, 65536])
def test_parameter_cache_reuse_and_mapping_invalidation(chunk_width):
    import cupy as cp
    from woof.core.noahmp_column_slab import parameter_fields

    params = NoahmpRuntimeParameters()
    keys = np.array([10006, 16007], dtype=np.int64)
    inverse = np.array([0, 1, 0, 1], dtype=np.int64)

    def get():
        return params.slab_parameter_chunks(
            keys, inverse, nsoil=4, chunk_width=chunk_width)

    def check(chunks):
        rows = [params.slab_row(vegtyp=int(key // 1000),
                                soiltyp=(int(key % 1000),) * 4,
                                slopetype=1, soilcolor=4, nsoil=4)
                for key in keys]
        for number, start in enumerate(range(0, len(inverse), chunk_width)):
            expected = parameter_fields(rows, inverse[start:start + chunk_width])
            for name, array in expected.items():
                actual = cp.asnumpy(chunks[number][name])
                wanted = cp.asnumpy(array)
                np.testing.assert_array_equal(actual.view(np.uint32),
                                              wanted.view(np.uint32))

    original = get()
    assert get() is original
    check(original)
    # An in-place identity edit must invalidate even with unchanged dimensions.
    inverse[0] = 1
    remapped = get()
    assert remapped is not original
    check(remapped)
    keys[1] = 16006
    replaced = get()
    assert replaced is not remapped
    check(replaced)
    inverse = inverse[:2]
    shortened = get()
    assert shortened is not replaced
    check(shortened)
    different_width = params.slab_parameter_chunks(
        keys, inverse, nsoil=4, chunk_width=2)
    if chunk_width != 2:
        assert different_width is not shortened


@requires_gpu
def test_slab_consumers_leave_cached_parameters_unchanged():
    import cupy as cp
    from woof.core.dycore import step
    from test_noahmp_runtime import _build

    state, cfg, driver = _build(nx=6, ny=4)
    step(state, cfg)
    cache = driver.noahmp_params._slab_parameter_cache
    before = [{name: cp.asnumpy(array).view(np.uint32).copy()
               for name, array in chunk.items()} for chunk in cache[1]]
    for _ in range(3):
        step(state, cfg)
    assert driver.noahmp_params._slab_parameter_cache is cache
    for chunk, expected in zip(cache[1], before):
        for name, words in expected.items():
            np.testing.assert_array_equal(cp.asnumpy(chunk[name]).view(np.uint32),
                                          words)


@requires_gpu
@pytest.mark.parametrize("field,value", [("xland", 2), ("xice", 1),
                                         ("ivgtyp", 16), ("isltyp", 7)])
@pytest.mark.parametrize("integer_width", [32, 64])
def test_surface_layout_detects_each_in_place_identity_edit(field, value,
                                                           integer_width):
    import cupy as cp

    params = NoahmpRuntimeParameters()
    fields = {"xland": cp.ones((3, 4), dtype=cp.float32),
              "xice": cp.zeros((3, 4), dtype=cp.float32),
              "ivgtyp": cp.full((3, 4), 10, dtype=cp.int32),
              "isltyp": cp.full((3, 4), 6,
                                dtype=cp.int32 if integer_width == 32 else cp.int64)}
    original = params.surface_layout(fields, nsoil=4, chunk_width=3)
    assert params.surface_layout(fields, nsoil=4, chunk_width=3) is original
    fields[field][1, 2] = value
    changed = params.surface_layout(fields, nsoil=4, chunk_width=3)
    assert changed is not original
    assert params.surface_layout(fields, nsoil=4, chunk_width=3) is changed
    # An uncached classification on the exact same inputs must agree.
    fresh = NoahmpRuntimeParameters().surface_layout(fields, nsoil=4, chunk_width=3)
    assert changed["census"] == fresh["census"]
    for name in ("sea_ice", "glacier", "vegtyp", "soiltyp", "land_j", "land_i"):
        np.testing.assert_array_equal(cp.asnumpy(changed[name]), cp.asnumpy(fresh[name]))
    np.testing.assert_array_equal(changed["keys"], fresh["keys"])
    np.testing.assert_array_equal(changed["inverse"], fresh["inverse"])


@requires_gpu
def test_surface_layout_tracks_threshold_and_chunk_width():
    import cupy as cp

    params = NoahmpRuntimeParameters()
    fields = {"xland": cp.ones((2, 4), dtype=cp.float32),
              "xice": cp.full((2, 4), 0.1, dtype=cp.float32),
              "ivgtyp": cp.full((2, 4), 10, dtype=cp.int32),
              "isltyp": cp.full((2, 4), 6, dtype=cp.int32)}
    land = params.surface_layout(fields, nsoil=4, chunk_width=3)
    assert land["census"]["land"] == 8
    params.xice_threshold = 0.02
    ice = params.surface_layout(fields, nsoil=4, chunk_width=3)
    assert ice["census"]["sea_ice"] == 8
    assert ice["census"]["land"] == 0
    assert params.surface_layout(fields, nsoil=4, chunk_width=1) is not ice


@requires_gpu
def test_the_held_caches_fit_the_preflight_price():
    """What the two caches keep on the card between calls, measured on an
    all-land grid with int64 identity grids, is at most the per-column bytes
    preflight prices (woof.core.preflight.noahmp_lsm_cache_shapes)."""
    import cupy as cp
    from woof.core.noahmp_runtime import slab_cache_bytes_per_column

    ny, nx = 5, 7
    params = NoahmpRuntimeParameters()
    fields = {"xland": cp.ones((ny, nx), dtype=cp.float32),
              "xice": cp.zeros((ny, nx), dtype=cp.float32),
              "ivgtyp": cp.full((ny, nx), 10, dtype=cp.int64),
              "isltyp": cp.full((ny, nx), 6, dtype=cp.int64)}
    fields["ivgtyp"][0, :3] = 16
    layout = params.surface_layout(fields, nsoil=4, chunk_width=4)
    assert layout["census"]["land"] == ny * nx
    params.slab_parameter_chunks(layout["keys"], layout["inverse"], nsoil=4,
                                 chunk_width=4)

    def device_bytes(value):
        if isinstance(value, cp.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(device_bytes(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(device_bytes(v) for v in value)
        return 0

    held = (device_bytes(params._surface_layout_cache)
            + device_bytes(params._slab_parameter_cache))
    assert held <= ny * nx * slab_cache_bytes_per_column()


@requires_gpu
def test_prepared_footprint_switch_has_no_host_classification(monkeypatch):
    import cupy as cp
    from woof.core.noahmp_runtime import SLAB_COLUMN_CHUNK
    from woof.core.noahmp_slab_libm import take_slab_rows

    params = NoahmpRuntimeParameters()
    params.static_surface_identity = True
    fields = {"xland": cp.ones((3, 4), dtype=cp.float32),
              "xice": cp.zeros((3, 4), dtype=cp.float32),
              "ivgtyp": cp.full((3, 4), 10, dtype=cp.int32),
              "isltyp": cp.full((3, 4), 6, dtype=cp.int32)}
    expected = {}
    snapshots = {}
    for token in (0, 1):
        fields["xland"][0, 0] = 1 + token
        fields["ivgtyp"][1, 1] = 10 + 6 * token
        params.prepare_surface_footprint(token, fields, nsoil=4)
        saved = params._surface_footprints[token][2]
        expected[token] = params.slab_parameter_chunks(
            saved["keys"], saved["inverse"], nsoil=4,
            chunk_width=SLAB_COLUMN_CHUNK)[0]
        snapshots[token] = {k: v.copy() for k, v in fields.items()}

    original = cp.asnumpy
    def refuse(*args, **kwargs):
        raise AssertionError("prepared footprint attempted D2H classification")
    for token in (0, 1, 0, 1):
        for k, v in snapshots[token].items():
            fields[k][...] = v
        with monkeypatch.context() as m:
            m.setattr(cp, "asnumpy", refuse)
            params.bind_surface_footprint(token)
            layout = params.surface_layout(fields, nsoil=4,
                                            chunk_width=SLAB_COLUMN_CHUNK)
            actual = take_slab_rows(layout["device_inverse"],
                                     layout["parameter_table"],
                                     len(layout["keys"]))
        assert layout["census"]["land"] == 12 - token
        for name, wanted in expected[token].items():
            np.testing.assert_array_equal(original(actual[name]).view(np.uint32),
                                          original(wanted).view(np.uint32))

    fields["isltyp"][2, 2] = 7
    with pytest.raises(RuntimeError, match="static surface identity changed"):
        params.surface_layout(fields, nsoil=4, chunk_width=SLAB_COLUMN_CHUNK)
