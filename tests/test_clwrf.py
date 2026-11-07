"""Stock WRF climate counters, window rules and lifecycle."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
from woof.core.clwrf import HISTORY, INPUTS, PERSISTENT

ROOT = Path(__file__).resolve().parents[1]

def test_stock_namelist_interval_and_native_missing_interval_error(tmp_path):
    from test_namelist_import import _generic_hierarchy_pair
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    text = inp.read_text().replace("&time_control", "&time_control\n output_diagnostics=1, auxhist3_interval=60,")
    inp.write_text(text)
    toml, _ = import_namelists(wps, inp, name="stock-clwrf")
    target = tmp_path / "stock.toml"; target.write_text(toml)
    cfg = load_experiment(target).root.run
    assert (cfg.output_diagnostics, cfg.auxhist3_interval) == (1, 60)
    inp.write_text(text.replace("auxhist3_interval=60,", ""))
    with pytest.raises(ValueError, match="NO 'auxhist3_interval'"):
        import_namelists(wps, inp, name="missing-clwrf-interval")

def test_stock_route_export_and_reimport(tmp_path):
    from test_route_round_trip_fields import _emit, _replay
    _, paths = _emit(tmp_path, "stock-clwrf", output_diagnostics=1, auxhist3_interval=60)
    cfg = _replay(paths).root.run
    assert cfg.output_diagnostics == 1 and cfg.auxhist3_interval == 60
    text = paths['namelist_input'].read_text()
    assert 'diag_int ' not in text and 'wind_int ' not in text

def test_native_interval_columns_keep_omitted_nest_defaults(tmp_path):
    from test_namelist_import import _generic_hierarchy_pair
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    wps, inp = _generic_hierarchy_pair(tmp_path, 2)
    original = inp.read_text()
    inp.write_text(original.replace('&time_control',
        '&time_control\n output_diagnostics=1, auxhist3_interval=60,'))
    with pytest.raises(ValueError, match='every active domain'):
        import_namelists(wps, inp, name='missing-nest-interval')
    inp.write_text(original.replace('&time_control',
        '&time_control\n output_diagnostics=1, auxhist3_interval=60,30,'))
    text, _ = import_namelists(wps, inp, name='native-interval-columns')
    path = tmp_path / 'columns.toml'; path.write_text(text)
    runs = load_experiment(path).domains
    assert [domain.run.auxhist3_interval for domain in runs] == [60, 30]

def test_stock_carriers_history_and_integer_restart(monkeypatch, tmp_path):
    from test_restart import _cfg, _shim_state, _fill_setup
    from woof.io import restart
    from woof.io.history_layout import produced_history_shapes, live_state_history_fields
    from woof.core import preflight
    cfg = _cfg(output_diagnostics=1, auxhist3_interval=60)
    state = _shim_state(cfg, monkeypatch); _fill_setup(state)
    assert len(HISTORY) == 36
    assert {symbol for symbol, _, _ in HISTORY.values()} == {
        "T2MIN", "T2MAX", "TT2MIN", "TT2MAX", "T2MEAN", "T2STD",
        "Q2MIN", "Q2MAX", "TQ2MIN", "TQ2MAX", "Q2MEAN", "Q2STD",
        "SKINTEMPMIN", "SKINTEMPMAX", "TSKINTEMPMIN", "TSKINTEMPMAX", "SKINTEMPMEAN", "SKINTEMPSTD",
        "U10MAX", "V10MAX", "SPDUV10MAX", "TSPDUV10MAX", "U10MEAN", "V10MEAN", "SPDUV10MEAN",
        "U10STD", "V10STD", "SPDUV10STD", "RAINCVMAX", "RAINCVMEAN", "RAINCVSTD", "TRAINCVMAX",
        "RAINNCVMAX", "RAINNCVMEAN", "RAINNCVSTD", "TRAINNCVMAX"}
    assert state.existing_scratch("clwrf_nsteps").dtype == np.int32
    assert state.existing_scratch("t2min") is None
    for index, name in enumerate(PERSISTENT):
        state.existing_scratch(name)[...] = index + 1
    fields = live_state_history_fields(state)
    shapes = produced_history_shapes(cfg)
    for name, (symbol, _, _) in HISTORY.items():
        assert fields[symbol] is state.existing_scratch(name)
        assert shapes[symbol] == (cfg.ny, cfg.nx)
        assert preflight.scratch_slot_lifetime(name).kind == "carrying"
    path = restart.write_restart(tmp_path / "stock.npz", state, cfg)
    target = _shim_state(cfg, monkeypatch); _fill_setup(target)
    restart.restore_restart(path, target, cfg)
    for name in PERSISTENT:
        np.testing.assert_array_equal(target.existing_scratch(name), state.existing_scratch(name))
    assert target.existing_scratch("clwrf_nsteps").dtype == np.int32

def test_native_warm_start_maps_registry_fields_and_broadcasts_scalar_counter():
    from woof.ingest import wrfrst
    for slot, (symbol, _, _) in HISTORY.items():
        assert wrfrst.WRFRST_DISPOSITIONS[symbol].kind == "read"
        assert wrfrst.scratch_restore_plan(slot, 0) == ("fill", symbol)
    assert wrfrst.WRFRST_DISPOSITIONS["NSTEPS"].kind == "read"
    assert wrfrst.scratch_restore_plan("clwrf_nsteps", 0) == ("fill", "NSTEPS")
    value = wrfrst._host_field({"NSTEPS": np.array(11, dtype=np.int32)},
        "NSTEPS", (2, 3), label="scratch/clwrf_nsteps")
    np.testing.assert_array_equal(value, np.full((2, 3), 11, dtype=np.int32))

@requires_gpu
def test_native_stock_words_at_every_call():
    import cupy as cp
    from woof.core.clwrf import launch
    path = ROOT / "tests/data/oracles/clwrf/vectors.npz"
    with np.load(path) as vectors:
        output = {name: cp.full((2, 3), -123.25, dtype=cp.float32) for name in HISTORY}
        counter = cp.zeros((2, 3), dtype=cp.int32)
        for index, (values, clock) in enumerate(zip(vectors["inputs"], vectors["clock"])):
            dt, xtime, seconds, restart, interval = clock
            launch({name: cp.asarray(value) for name, value in zip(INPUTS, values)}, output, counter,
                   dt=dt, xtime=xtime, seconds=seconds, interval=interval)
            actual = np.stack([cp.asnumpy(output[name]).view(np.uint32) for name in HISTORY])
            np.testing.assert_array_equal(actual, vectors["expected"][index], err_msg=f"call {index}")
            np.testing.assert_array_equal(cp.asnumpy(counter), vectors["counters"][index])

@requires_gpu
def test_stock_climate_is_trajectory_inert():
    import cupy as cp
    from woof.core.dycore import step
    from woof.core.physics import initialize_physics
    from test_mp_spec_zone_ring import _square_moist_state
    off, cfg0 = _square_moist_state(mp=6, open_x=True, open_y=True, dt=2., hailcast_opt=0)
    on, cfg1 = _square_moist_state(mp=6, open_x=True, open_y=True, dt=2., hailcast_opt=0,
                                 output_diagnostics=1, auxhist3_interval=1)
    initialize_physics(off, cfg0); initialize_physics(on, cfg1)
    for _ in range(3):
        step(off, cfg0); step(on, cfg1)
    for name in ('u', 'v', 'w', 'thp', 'php', 'mup', 'p', 'al', 'alt', 'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'h_diabatic'):
        a, b = getattr(off, name, None), getattr(on, name, None)
        if a is not None:
            np.testing.assert_array_equal(cp.asnumpy(a).view(np.uint32), cp.asnumpy(b).view(np.uint32), err_msg=name)

@requires_gpu
def test_stock_climate_survives_pinned_tile_reuse():
    import cupy as cp
    from tilestream import gather, physics_inventory, spec
    from woof.config import RunConfig
    from woof.core.clwrf import launch
    cfg = RunConfig(nx=8, ny=6, nz=3, dx=3000., dy=3000., ztop=10000.,
                    dt=20., run_seconds=240., output_diagnostics=1, auxhist3_interval=1)
    def allocate(c):
        return {**{name: cp.zeros((c.ny, c.nx), cp.float32) for name in INPUTS},
                **{"scratch/" + name: cp.zeros((c.ny, c.nx), cp.int32 if name == "clwrf_nsteps" else cp.float32)
                   for name in PERSISTENT}}
    resident = allocate(cfg)
    store = {name: gather.pinned_copy(value) for name, value in resident.items()}
    tile_cfg = replace(cfg, nx=6, ny=5)
    tile = allocate(tile_cfg)
    tiles = spec.plan_tiles(cfg.nx, cfg.ny, 4, 3, halo=1, periodic=True)
    def sample(arrays, seconds):
        launch(arrays, {name: arrays['scratch/' + name] for name in HISTORY}, arrays['scratch/clwrf_nsteps'],
               dt=cfg.dt, xtime=seconds / 60., seconds=seconds, interval=cfg.auxhist3_interval)
    for call in range(12):
        for index, name in enumerate(INPUTS):
            values = (np.arange(48, dtype=np.float32).reshape(6, 8) + index + call % 5) / 16
            resident[name].set(values); store[name][...] = values
        sample(resident, call * cfg.dt)
        for geometry in tiles:
            gather.gather_tile(store, tile, geometry, inventory_fn=physics_inventory.carrier_inventory, nz=cfg.nz)
            sample(tile, call * cfg.dt)
            gather.scatter_tile(tile, store, geometry, inventory_fn=physics_inventory.carrier_inventory, nz=cfg.nz)
        cp.cuda.get_current_stream().synchronize()
        for name in PERSISTENT:
            np.testing.assert_array_equal(store['scratch/' + name].view(np.uint32),
                cp.asnumpy(resident['scratch/' + name]).view(np.uint32), err_msg=f'{call}: {name}')
