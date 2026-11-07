"""WRF interval fluxes: user doors, device recurrence, output and restart."""
from dataclasses import replace

import numpy as np
import pytest

from woof.config import validate_run_config
from woof.core import avgflx
from woof.io import restart
from test_restart import _cfg, _fill_setup, _shim_state


@pytest.mark.parametrize('value', [-1, 2, True, 1.0, '1'])
def test_invalid_switch(value):
    with pytest.raises(ValueError, match='do_avgflx_em'):
        validate_run_config(_cfg(do_avgflx_em=value))


def test_convective_refusal_names_actual_missing_producer():
    with pytest.raises(ValueError, match='cu_physics=93'):
        validate_run_config(_cfg(do_avgflx_em=1, do_avgflx_cugd=1))


def test_legacy_toml_and_registry_door(tmp_path):
    from woof.config import load_config
    from woof.physics_registry import physics_registry
    path = tmp_path / 'flux.toml'
    path.write_text('''[grid]
nx=12
ny=12
nz=10
dx=2000.0
dy=2000.0
ztop=8000.0
[dynamics]
dt=3.0
do_avgflx_em=1
[run]
run_seconds=60.0
''')
    assert load_config(path).do_avgflx_em == 1
    row = physics_registry()['parameters']['do_avgflx_em']
    assert row['enum'] == [0, 1] and row['default'] == 0
    assert row['per_domain'] is True


@pytest.mark.parametrize('column,expected', [('1', [1, 0]), ('0, 1', [0, 1]), ('1, 1', [1, 1])])
def test_namelist_per_domain_roundtrip_and_support_report(tmp_path, column, expected):
    from test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    from woof.namelist_compat import analyze_namelists
    from woof.experiment import load_experiment
    pair = _pair(tmp_path, inp=INPUT_TEXT.replace('&dynamics', '&dynamics\n do_avgflx_em = ' + column + ','))
    text, _ = import_namelists(*pair)
    path = tmp_path / 'avg.toml'
    path.write_text(text)
    exp = load_experiment(path)
    assert [dc.run.do_avgflx_em for dc in exp.domains] == expected
    report = analyze_namelists(*pair)
    assert 'UNCLASSIFIED_NAMELIST_SETTING' not in str(report)


def test_memory_and_first_frame(monkeypatch):
    from woof.core.preflight import scratch_slot_registry
    from woof.io.history_layout import live_state_history_fields
    on_cfg, off_cfg = _cfg(do_avgflx_em=1), _cfg()
    on, off = _shim_state(on_cfg, monkeypatch), _shim_state(off_cfg, monkeypatch)
    on_slots, off_slots = scratch_slot_registry(on_cfg), scratch_slot_registry(off_cfg)
    for slot, shape in avgflx.scratch_shapes(on_cfg).items():
        assert slot not in off_slots
        assert on_slots[slot] == shape
        value = on.existing_scratch(slot)
        assert value.shape == shape and not value.any()
        assert off.existing_scratch(slot) is None
    fields = live_state_history_fields(on)
    assert set(fields) - set(live_state_history_fields(off)) == {v[0] for v in avgflx.HISTORY.values()}
    assert on.existing_scratch('avgflx_clock').dtype == np.int32
    assert {'rk_ru_m', 'rk_rv_m', 'rk_ww_m'} <= set(on_slots)


@pytest.mark.parametrize('pending', [0, 1])
def test_checkpoint_carries_mean_count_and_pending_reset(monkeypatch, tmp_path, pending):
    cfg = _cfg(do_avgflx_em=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    for slot in avgflx.HISTORY:
        source.existing_scratch(slot)[...] = np.float32(3.125)
    source.existing_scratch('avgflx_clock')[0] = 37
    if pending:
        avgflx.history_written(source)
    path = restart.write_restart(tmp_path / 'flux.npz', source, cfg)
    target = _shim_state(cfg, monkeypatch)
    _fill_setup(target)
    restart.restore_restart(path, target, cfg)
    for slot in avgflx.PERSISTENT:
        assert target.existing_scratch(slot).tobytes() == source.existing_scratch(slot).tobytes()


def test_move_clears_whole_interval(monkeypatch):
    from types import SimpleNamespace
    from woof.core.physics_continuation import shift_continuation, restore_continuation
    state = _shim_state(_cfg(do_avgflx_em=1), monkeypatch)
    captured = {slot: state.existing_scratch(slot) for slot in avgflx.PERSISTENT}
    for array in captured.values():
        array[...] = 7
    # avgflx is reset before any overlap plan is consulted.
    moved = shift_continuation(captured, None)
    assert all(not array.any() for array in moved.values())
    assert all((array == 7).all() for array in captured.values())
    # Relocation rebuilds an already allocated child before restoring these
    # values. Its integer clock must not be requested as floating scratch.
    rebuilt = _shim_state(_cfg(do_avgflx_em=1), monkeypatch)
    restore_continuation(rebuilt, SimpleNamespace(), moved)
    for slot in avgflx.PERSISTENT:
        restored = rebuilt.existing_scratch(slot)
        assert restored.dtype == captured[slot].dtype
        assert not restored.any()


@pytest.mark.gpu
def test_device_recurrence_and_independent_history_windows():
    import cupy as cp
    from woof.core.state import DomainState
    configs = [_cfg(do_avgflx_em=1), _cfg(nx=9, ny=7, do_avgflx_em=1)]
    states = [DomainState(cfg) for cfg in configs]
    means = [{k: np.zeros(shape, np.float32) for k, shape in avgflx.scratch_shapes(c).items() if k in avgflx.HISTORY} for c in configs]
    counts = [0, 0]
    rng = np.random.default_rng(414)
    for step in range(23):
        for domain, (state, cfg) in enumerate(zip(states, configs)):
            if step % (4 + domain) == 0:
                previous = [cp.asnumpy(state.existing_scratch(k)).tobytes() for k in avgflx.HISTORY]
                avgflx.history_written(state)
                assert previous == [cp.asnumpy(state.existing_scratch(k)).tobytes() for k in avgflx.HISTORY]
                counts[domain] = 0
                for value in means[domain].values():
                    value.fill(0)
            fluxes = [rng.normal(0, 1e6, v.shape).astype(np.float32) for v in means[domain].values()]
            avgflx.update(state, cfg, tuple(cp.asarray(v) for v in fluxes))
            count = np.float32(counts[domain])
            for (slot, old), flux in zip(means[domain].items(), fluxes):
                old[...] = (count * old + flux) / (count + np.float32(1))
                assert old.tobytes() == cp.asnumpy(state.existing_scratch(slot)).tobytes()
            counts[domain] += 1
            clock = cp.asnumpy(state.existing_scratch('avgflx_clock'))
            assert (clock[0] == counts[domain]).all() and not clock[1].any()


@pytest.mark.gpu
def test_whole_dycore_restart_and_inertness(tmp_path):
    import cupy as cp
    from woof.core.dycore import step
    from woof.io.wrfout import state_frame, WrfoutWriter
    from test_mp_spec_zone_ring import _square_moist_state
    from netCDF4 import Dataset
    off, off_cfg = _square_moist_state(dt=3, open_x=True, open_y=True)
    on, cfg = _square_moist_state(dt=3, open_x=True, open_y=True, do_avgflx_em=1)
    split, _ = _square_moist_state(dt=3, open_x=True, open_y=True, do_avgflx_em=1)
    for state in (off, on, split):
        state.u[...] = 8.0
        state.v[...] = -3.0
    for i in range(20):
        for state, config in ((off, off_cfg), (on, cfg), (split, cfg)):
            step(state, config)
        if i in (4, 9, 14):
            avgflx.history_written(on)
            avgflx.history_written(split)
        if i in (7, 9):  # Mid-window and immediately after a history reset.
            path = restart.write_restart(tmp_path / f'split-{i}.npz', split, cfg)
            split, _ = _square_moist_state(dt=3, open_x=True, open_y=True, do_avgflx_em=1)
            restart.restore_restart(path, split, cfg)
    frames = {name: state_frame(state) for name, state in [('off', off), ('on', on), ('split', split)]}
    assert any(np.any(frames['on'][v[0]]) for v in avgflx.HISTORY.values())
    for key, value in frames['on'].items():
        assert value.tobytes() == frames['split'][key].tobytes(), key
        if key in frames['off']:
            assert value.tobytes() == frames['off'][key].tobytes(), key
    path = tmp_path / 'wrfout.nc'
    with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy, field_schema=frames['on'], engine='python') as writer:
        writer.write_frame('1999-05-03_12:01:00', frames['on'])
    with Dataset(path) as ds:
        for (slot, (symbol, units, description)), stagger in zip(avgflx.HISTORY.items(), 'XYZ'):
            var = ds[symbol]
            assert var.stagger == stagger and var.units == units
            assert var.description == description and var.FieldType == 104
            assert var.MemoryOrder == 'XYZ'
            assert var[0].tobytes() == frames['on'][symbol].tobytes()


@pytest.mark.gpu
def test_ranked_history_reset_preserves_count_and_mean():
    import cupy as cp
    from woof.core import dycore, streaming
    from woof.core.devices import DeviceOptions
    from tilestream.ranks_gate import fixture, make_ranked
    from tilestream.multigpu import forced_config
    from types import SimpleNamespace
    cfg = replace(forced_config(96, 80, 12), do_avgflx_em=1)
    state, bundle = fixture(cfg)
    options = DeviceOptions(count=2, ids=(0, 0))
    streamed = make_ranked(bundle, cfg, streaming.ranked_decision(cfg, options), options, None, 'threads')
    proxy = SimpleNamespace(_streamed_domain=streamed, existing_scratch=lambda _: None)
    run = streamed.tiled_run
    take = streaming.streamed_store_inventory()
    try:
        for i in range(6):
            dycore.step(state, cfg)
            run.sweep(1)
            if i in (1, 3):
                avgflx.history_written(state)
                avgflx.history_written(proxy)
        actual = run.store
        for slot in avgflx.PERSISTENT:
            name = 'scratch/' + slot
            assert cp.asnumpy(take(state)[name]).tobytes() == actual[name].tobytes(), name
    finally:
        run.close()


@pytest.mark.gpu
def test_native_strict_real_flux_capture():
    from pathlib import Path
    import cupy as cp
    from woof.core.state import DomainState
    cfg = _cfg(nx=7, ny=7, nz=7, do_avgflx_em=1)
    state = DomainState(cfg)
    files = sorted((Path(__file__).parent / 'data/oracles/avgflx_wrf461_strict239').glob('*.bin'))
    assert len(files) == 9
    slices = [(slice(7), slice(7), slice(8)),
              (slice(7), slice(8), slice(7)),
              (slice(8), slice(7), slice(7))]
    for path in files:
        raw = path.read_bytes()
        count = int(np.frombuffer(raw[:4], dtype='>i4')[0])
        blocks = np.frombuffer(raw[4:], dtype='>f4').astype(np.float32)
        blocks = blocks.reshape(9, 8, 8, 8).transpose(0, 2, 1, 3)
        fluxes = []
        expected = []
        for i, (slot, select) in enumerate(zip(avgflx.HISTORY, slices)):
            state.existing_scratch(slot)[...] = cp.asarray(blocks[i][select])
            fluxes.append(cp.asarray(np.ascontiguousarray(blocks[3+i][select])))
            expected.append(np.ascontiguousarray(blocks[6+i][select]))
        state.existing_scratch('avgflx_clock')[0] = count
        state.existing_scratch('avgflx_clock')[1] = 0
        avgflx.update(state, cfg, fluxes)
        for slot, want in zip(avgflx.HISTORY, expected):
            assert cp.asnumpy(state.existing_scratch(slot)).tobytes() == want.tobytes(), (path.name, slot)


@pytest.mark.gpu
def test_dry_acoustic_fluxes_do_not_change_the_forecast():
    import cupy as cp
    from woof.core.state import init_at_rest
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.dycore import step
    from woof.io.wrfout import state_frame
    cfg = _cfg(nx=12, ny=12, nz=10, dt=3.0)
    on_cfg = replace(cfg, do_avgflx_em=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 300.0 + 0.003 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    off, on = [init_at_rest(c, coord, base) for c in (cfg, on_cfg)]
    for state in (off, on):
        state.u[...] = 8.0
        state.v[...] = -3.0
    for i in range(20):
        step(off, cfg)
        step(on, on_cfg)
        if i in (4, 9, 14):
            avgflx.history_written(on)
    before, after = state_frame(off), state_frame(on)
    for key, value in before.items():
        assert value.tobytes() == after[key].tobytes(), key
    assert np.any(after['AVGFLX_RUM']) and np.any(after['AVGFLX_RVM'])
    assert (cp.asnumpy(on.existing_scratch('avgflx_clock'))[0] == 5).all()
