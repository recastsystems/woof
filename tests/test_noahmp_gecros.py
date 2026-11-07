"""Native active-crop replay, thermal emergence, and carried-state recovery."""
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

DATA = Path(__file__).resolve().parents[1] / "woof/data/gecros/oracle"


@requires_gpu
def test_active_native_471_replay_has_zero_differing_words():
    import cupy as cp
    from woof.core.noahmp_gecros_slab import module

    records = np.fromfile(DATA / "inputs.bin", np.float32).reshape(-1, 78)
    expected = np.fromfile(DATA / "native.bin", np.float32).reshape(-1, 66)
    n = len(records)
    x = cp.asarray(np.ascontiguousarray(records[:, :18]))
    state = cp.asarray(np.ascontiguousarray(records[:, 18:]))
    flux = cp.empty((n, 6), cp.float32)
    module().get_function("k_gecros_replay")(
        ((n + 63) // 64,), (64,), (state, x, flux, np.int32(n)))
    actual = np.concatenate((cp.asnumpy(state), cp.asnumpy(flux)), axis=1)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@requires_gpu
def test_cold_initializer_matches_native_and_preserves_reserved_words():
    import cupy as cp
    from woof.core.noahmp_gecros_slab import module

    state = cp.zeros((1, 60), cp.float32)
    state[0, 58:] = cp.asarray([123., -456.], cp.float32)
    module().get_function("k_gecros_init")(
        (1,), (64,), (cp.asarray([40.], cp.float32), state, np.int32(1)))
    expected = np.fromfile(DATA / "initial.bin", np.float32)
    expected[58:] = [123., -456.]
    np.testing.assert_array_equal(cp.asnumpy(state).ravel().view("u4"), expected.view("u4"))


@requires_gpu
def test_thermal_emergence_starts_both_crops_and_leaves_warm_states_alone():
    import cupy as cp
    from woof.core.noahmp_gecros_slab import module

    m = module()
    initial = np.fromfile(DATA / "initial.bin", np.float32)
    state = cp.asarray(np.tile(initial, (5, 1)))
    # Wheat, maize, frozen wheat, non-crop, and an already emerged crop.
    crop = cp.asarray([1, 2, 1, 0, 1], cp.int32)
    soil = cp.full((5, 7), 303.15, cp.float32)
    soil[2, 3] = 273.15
    state[4, 40] = 1
    before = cp.asnumpy(state[4]).copy()
    dt = cp.full(5, 3600., cp.float32)
    for _ in range(120):
        m.get_function("k_gecros_emerge")((1,), (64,), (state, crop, soil, dt, np.int32(5)))
    actual = cp.asnumpy(state)
    assert np.all(actual[:2, 40] == 1)
    assert np.all(actual[2:4, 40] == -1)
    assert actual[2, 42] == 0
    np.testing.assert_array_equal(actual[3], initial)
    np.testing.assert_array_equal(actual[4], before)


@requires_gpu
def test_full_column_emerges_and_restart_keeps_all_sixty_words(tmp_path):
    import cupy as cp
    from woof.core.dycore import step
    from woof.core.noahmp_gecros_slab import STATE
    from woof.io.restart import restore_restart, write_restart
    from test_noahmp_runtime import _build

    state, cfg, driver = _build(nx=6, ny=4, opt_crop=2)
    mask = driver.fields["cropcat"] > 0
    assert bool(cp.any(mask))
    # A carried crop just short of emergence, under real surface forcing.
    driver.fields["gecros43xy"][mask] = cp.float32(80.999)
    for _ in range(3):
        step(state, cfg)
    assert bool(cp.all(driver.fields["gecros41xy"][mask] > 0))
    assert bool(cp.all(cp.isfinite(driver.fields["hfx"])))
    path = tmp_path / "crop.npz"
    write_restart(path, state, cfg)
    restored, restored_cfg, restored_driver = _build(nx=6, ny=4, opt_crop=2)
    restore_restart(path, restored, restored_cfg)
    for name in STATE:
        np.testing.assert_array_equal(cp.asnumpy(restored_driver.fields[name]), cp.asnumpy(driver.fields[name]))
    step(state, cfg)
    step(restored, restored_cfg)
    for name in (*STATE, "hfx", "lh", "lai", "smois", "tslb"):
        np.testing.assert_array_equal(cp.asnumpy(restored_driver.fields[name]), cp.asnumpy(driver.fields[name]))


@requires_gpu
def test_native_harvest_gate_and_reset_have_zero_differing_words():
    import cupy as cp
    from woof.core.noahmp_gecros_slab import module
    records = np.fromfile(DATA / "harvest-inputs.bin", np.float32).reshape(-1, 62)
    expected = np.fromfile(DATA / "harvest-native.bin", np.float32).reshape(-1, 60)
    n = len(records)
    state = cp.asarray(np.ascontiguousarray(records[:, 2:]))
    crop = cp.asarray(records[:, 1].astype(np.int32))
    dt = cp.asarray(np.ascontiguousarray(records[:, 0]))
    module().get_function("k_gecros_harvest")(
        ((n + 63) // 64,), (64,), (state, crop, dt, np.int32(n)))
    np.testing.assert_array_equal(cp.asnumpy(state).view("u4"), expected.view("u4"))
