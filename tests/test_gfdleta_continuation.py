"""GFDL memory follows ground through capture, relocation and restore."""
from types import SimpleNamespace
import numpy as np
from woof.core.physics_continuation import capture_continuation, shift_continuation, restore_continuation
from woof.core.nest_relocation import Placement, plan_relocation
from woof.gfdleta_contract import GFDLETA_FIELDS, GFDLETA_INTEGER_FIELDS


def test_cloud_memory_moves_bitwise_and_new_ground_has_no_convection():
    shape = (12, 12)
    old = {name: np.arange(144, dtype=np.int32 if name in GFDLETA_INTEGER_FIELDS else np.float32).reshape(shape).copy()
           for name in GFDLETA_FIELDS}
    state = SimpleNamespace(existing_scratch=lambda name: None)
    driver = SimpleNamespace(fields=old, rthratenlw=np.zeros((4, *shape), np.float32))
    captured = capture_continuation(state, driver)
    plan = plan_relocation(placement_from=Placement(grid_id=2, i_parent_start=10, j_parent_start=10),
                           placement_to=Placement(grid_id=2, i_parent_start=11, j_parent_start=10, generation=1),
                           parent_grid_ratio=3, child_nx=12, child_ny=12)
    shifted = shift_continuation(captured, plan)
    new = {name: np.full_like(value, -99) for name, value in old.items()}
    target = SimpleNamespace(fields=new, rthratenlw=np.zeros_like(driver.rthratenlw))
    receipt = restore_continuation(state, target, shifted)
    (dj, sj), (di, si) = plan.window(shape)
    for name, value in old.items():
        np.testing.assert_array_equal(new[name][dj, di].view(np.uint32), value[sj, si].view(np.uint32))
        assert not np.array_equal(new[name], value)
        assert np.all(new[name][:, -3:] == (5 if name == 'hbot' else 0))
        assert f'fields/{name}' in receipt['slots_moved']
        assert new[name].dtype == value.dtype


def test_native_counter_history_schema_retains_integer_type():
    from woof.io.wrf_output_schema import GFDLETA_OUTPUT_FIELDS, HISTORY_FIELDS_BY_NETCDF_NAME
    assert set(GFDLETA_OUTPUT_FIELDS) == set(GFDLETA_FIELDS)
    for name in GFDLETA_INTEGER_FIELDS:
        row = HISTORY_FIELDS_BY_NETCDF_NAME[name.upper()]
        assert row.dtype == 'i4' and row.field_type == 106
        assert row.wrf_history is False


def test_streamed_twin_keeps_policy_and_owns_its_tables(monkeypatch):
    import sys
    from woof.core.gfdleta import GFDLEtaRadiation
    from woof.core.streaming import _tile_scheme
    # Construction only needs these NumPy-compatible CuPy array operations.
    monkeypatch.setitem(sys.modules, 'cupy', np)
    domain = GFDLEtaRadiation('native-start', np.ones((9, 9)), np.zeros((9, 9)),
                             p_top=5000., co2tf=0, longwave=False,
                             shortwave=True, column_chunk=17)
    domain.tables = object()
    domain.update_count = 42
    tile = _tile_scheme(domain, np.full((4, 5), 32.), np.full((4, 5), -80.))
    assert tile is not domain and tile.tables is None
    assert tile.update_count == 0
    assert (tile.start_time, tile.p_top, tile.co2tf, tile.longwave,
            tile.shortwave, tile.column_chunk) == ('native-start', 5000., 0, False, True, 17)
    np.testing.assert_array_equal(tile.latitude_deg, np.full((4, 5), 32., np.float32))
    np.testing.assert_array_equal(tile.longitude_deg, np.full((4, 5), -80., np.float32))
