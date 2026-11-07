"""Measured policy rows and the previously missing MPAS launch gate."""
from argparse import Namespace
import json
import pytest
from woof.hex import device_admission as memory, forecast_door as door
from woof.hex.compact_memory import CompactFootprintModel


@pytest.fixture
def card(monkeypatch):
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'compact')
    from woof.hex import numerics
    monkeypatch.setattr(numerics, '_mode', 'fast')
    return memory.KNOWN_CARDS['16gib-70sm']


@pytest.mark.parametrize('route,suite', [('mpas', 'convection_permitting'), ('arwen', 'arwen_mpas_column')])
def test_route_rows_and_all_admission_surfaces_agree(card, route, suite):
    model = memory.model_for_run(card, route=route, suite=suite, cells=40962)
    assert isinstance(model, CompactFootprintModel)
    for cells, peak in model.peak_knots:
        assert model.predict_bytes(cells) == peak
        required = model.required_bytes(cells)
        assert required == memory.route_required_free_bytes(cells, model, route=route)
        for free, admitted in [(required, True), (required-1, False)]:
            verdict = door.admission_verdict(mesh='test', cells=cells,
                free_bytes=free, total_bytes=16<<30, headroom_bytes=None, registry={}, model=model)
            assert verdict.admitted is admitted
            assert verdict.required_bytes == required
        assert model.max_cells(required) >= cells
    assert model.required_bytes(163842) > 12<<30
    assert model.max_cells(1<<40) == model.peak_knots[-1][0]
    assert not isinstance(memory.model_for_run(card, route=route, suite=suite,
        cells=model.peak_knots[-1][0]+1), CompactFootprintModel)


def test_rows_do_not_leak_to_other_cards_extents_suites_or_policies(card, monkeypatch):
    cases = [dict(card=memory.KNOWN_CARDS['24gib-128sm']), dict(levels=59),
        dict(configuration='limited-area'), dict(suite='hrrrv5')]
    for change in cases:
        kwargs = dict(card=card, route='mpas', suite='convection_permitting', cells=40962)
        kwargs.update(change)
        assert not isinstance(memory.model_for_run(**kwargs), CompactFootprintModel)
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'single')
    assert not isinstance(memory.model_for_run(card, route='mpas'), CompactFootprintModel)


def test_mpas_gate_reads_input_extent_and_records_refusal_before_prewarm(tmp_path, card, monkeypatch):
    from netCDF4 import Dataset
    init = tmp_path/'init.nc'
    with Dataset(init, 'w') as dataset:
        dataset.createDimension('nCells', 163842)
        dataset.createDimension('nVertLevels', 55)
    arguments = Namespace(init=init, out=tmp_path/'forecast', mesh='x1.163842')
    monkeypatch.setattr(door, 'read_card_profile', lambda: card)
    monkeypatch.setattr(door, 'measure_device_memory', lambda: (11<<30, 16<<30))
    with pytest.raises(Exception, match='Not enough GPU memory.*Forecast has not started'):
        door.admit_native_device(arguments, Namespace(suite='convection_permitting'))
    record = json.loads((arguments.out/'device-memory-admission.json').read_text())
    assert record['admitted'] is False
    assert record['before_kernel_prewarm'] is True
    assert record['cells'] == 163842
    assert record['measured_free_bytes'] == 11<<30
    monkeypatch.setattr(door, 'measure_device_memory', lambda: (16<<30, 16<<30))
    assert door.admit_native_device(arguments, Namespace(suite='convection_permitting')).admitted


def test_qualification_does_not_hide_workspace_numerics_or_physics_overrides(card, monkeypatch):
    args = dict(card=card, route='mpas', suite='convection_permitting', cells=40962)
    assert isinstance(memory.model_for_run(**args), CompactFootprintModel)
    assert not isinstance(memory.model_for_run(**args, qualified_defaults=False), CompactFootprintModel)
    monkeypatch.setenv('WOOF_HEX_THOMPSON_WORKSPACE_MIB', '1536')
    assert not isinstance(memory.model_for_run(**args), CompactFootprintModel)
    monkeypatch.delenv('WOOF_HEX_THOMPSON_WORKSPACE_MIB')
    from woof.hex import numerics
    monkeypatch.setattr(numerics, '_mode', 'strict')
    assert isinstance(memory.model_for_run(**args), CompactFootprintModel)
    assert not isinstance(memory.model_for_run(**{**args, 'cells': 163842}), CompactFootprintModel)


def test_installed_regional_peak_is_refused_before_prewarm(tmp_path, card, monkeypatch):
    from netCDF4 import Dataset
    init = tmp_path / 'init.nc'
    with Dataset(init, 'w') as dataset:
        dataset.createDimension('nCells', 4499)
        dataset.createDimension('nVertLevels', 59)
    args = Namespace(init=init, out=tmp_path / 'forecast', mesh='regional', lbc_dir=tmp_path)
    model = memory.model_for_run(card, configuration='limited-area', levels=59,
                                route='mpas', suite='hrrrv5', cells=4499)
    assert model.predict_bytes(4499) == 3914 * memory.MIB
    required = model.required_bytes(4499)
    assert required == 4170 * memory.MIB
    monkeypatch.setattr(door, 'read_card_profile', lambda: card)
    monkeypatch.setattr(door, 'measure_device_memory', lambda: (required - 1, 16 << 30))
    with pytest.raises(Exception, match='Not enough GPU memory.*Forecast has not started'):
        door.admit_native_device(args, Namespace(suite='hrrrv5'))
    record = json.loads((args.out / 'device-memory-admission.json').read_text())
    assert record['before_kernel_prewarm'] is True
    assert record['admitted'] is False
    monkeypatch.setattr(door, 'measure_device_memory', lambda: (required, 16 << 30))
    assert door.admit_native_device(args, Namespace(suite='hrrrv5')).admitted


# 149,322: the 96 -> 3 km zoom globe, whose restarts the uniform-globe curve priced at 26,037 MiB against a
# measured 17,632 MiB peak and refused (IFS-9KM-DEFAULT-2026-11-04).
@pytest.mark.parametrize('cells,peak_mib', [(40962, 8602), (149322, 17632), (163842, 18566)])
def test_measured_5090_peak_admits_the_freed_capacity_before_prewarm(tmp_path, monkeypatch, cells, peak_mib):
    from netCDF4 import Dataset
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'compact')
    from woof.hex import numerics
    monkeypatch.setattr(numerics, '_mode', 'fast')
    card = memory.KNOWN_CARDS['32gib-170sm']
    model = memory.model_for_run(card, route='mpas', suite='convection_permitting', cells=cells)
    assert isinstance(model, CompactFootprintModel)
    assert model.predict_bytes(cells) == peak_mib * memory.MIB
    init = tmp_path/'init.nc'
    with Dataset(init, 'w') as dataset:
        dataset.createDimension('nCells', cells)
        dataset.createDimension('nVertLevels', 55)
    args = Namespace(init=init, out=tmp_path/'forecast', mesh=f'x1.{cells}')
    monkeypatch.setattr(door, 'read_card_profile', lambda: card)
    required = model.required_bytes(cells)
    assert required < 19 << 30
    monkeypatch.setattr(door, 'measure_device_memory', lambda: (required-1, 32<<30))
    with pytest.raises(Exception, match='Not enough GPU memory.*Forecast has not started'):
        door.admit_native_device(args, Namespace(suite='convection_permitting'))
    record = json.loads((args.out/'device-memory-admission.json').read_text())
    assert record['before_kernel_prewarm'] is True
    assert record['admitted'] is False
    monkeypatch.setattr(door, 'measure_device_memory', lambda: (required, 32<<30))
    assert door.admit_native_device(args, Namespace(suite='convection_permitting')).admitted
    # The measured extents do not qualify a mesh outside them, another card,
    # vertical grid, arithmetic mode or allocation policy.
    for change in (dict(cells=40961), dict(cells=163843), dict(card=memory.KNOWN_CARDS['24gib-128sm']),
                   dict(levels=59), dict(suite='hrrrv5')):
        kwargs = dict(card=card, route='mpas', suite='convection_permitting', cells=cells)
        kwargs.update(change)
        assert not isinstance(memory.model_for_run(**kwargs), CompactFootprintModel)
    monkeypatch.setattr(numerics, '_mode', 'strict')
    assert not isinstance(memory.model_for_run(card, route='mpas', cells=cells), CompactFootprintModel)
