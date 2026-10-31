"""Runtime class admission requires intact, independent, finite forecast evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from woof.hex import regional_mint as mint
from woof.hex.cuda_backend.regional_admission import (
    RegionalClassKey, admitted_class, admitted_class_for_key,
)
from woof.hex.drivers import run_cuda_regional_mint as driver


def evidence(tmp_path):
    key = RegionalClassKey.build(boundary_zone_width=7, n_vert_levels=55,
                                finest_edge_m=1234.567, dt_seconds=5, kernel_set='a' * 64)
    run = dict(schema='gpuwm-hex.regional-mint-run/v1', instrument='run_cuda_regional_mint',
               pid=1, minted_without_anchor_gate=True, numerics='strict', key=key.as_dict(),
               start_time='2026-10-27_18:00:00', stop_time='2026-10-27_18:01:15',
               steps=driver.STEPS, config_sha256='b' * 64, n_cells=100,
               bdy_mask_sha256='c' * 64, card='test-card',
               driver_sha256=hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest(),
               frames=[dict(step=step, finite={name: True for name in driver.FIELDS},
                            field_sha256={name: 'd' * 64 for name in driver.FIELDS})
                       for step in range(driver.STEPS + 1)])
    pair = [run, copy.deepcopy(run)]
    pair[1]['pid'] = 2
    document = dict(schema=mint.SCHEMA, key=key.as_dict(), class_id=mint.class_id(key),
                    parent='parent', mesh_row='cull', generated_utc='2026-10-20T00:00:00Z', runs=[])
    for index, item in enumerate(pair):
        path = tmp_path / f'run{index}.json'
        path.write_text(json.dumps(item))
        document['runs'].append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    path = tmp_path / 'class.mint.json'
    path.write_text(json.dumps(document))
    return key, pair, path


def test_intact_runtime_class_is_presented_to_forecast_admission(tmp_path, monkeypatch):
    key, pair, path = evidence(tmp_path)
    monkeypatch.setenv('WOOF_HEX_REGIONAL_CONTRACT_DIR', str(tmp_path))
    assert admitted_class_for_key(key).class_id == mint.class_id(key)
    assert admitted_class(mint.class_id(key)).key == key
    # Editing a measured run lapses the class in both lookup paths.
    (tmp_path / 'run1.json').write_text('{}')
    assert admitted_class_for_key(key) is None
    assert admitted_class(mint.class_id(key)) is None


@pytest.mark.parametrize('defect', [
    'same_process', 'nonfinite', 'different_field', 'different_key', 'missing_step',
    'different_window', 'missing_window', 'shortened_sample',
])
def test_pair_rejects_unmeasured_or_different_runs(tmp_path, defect):
    key, pair, path = evidence(tmp_path)
    if defect == 'same_process':
        pair[1]['pid'] = pair[0]['pid']
    elif defect == 'nonfinite':
        pair[1]['frames'][3]['finite']['theta'] = False
    elif defect == 'different_field':
        pair[1]['frames'][3]['field_sha256']['theta'] = 'e' * 64
    elif defect == 'different_key':
        pair[1]['key']['kernel_set_sha256'] = 'e' * 64
    elif defect == 'different_window':
        pair[1].update(start_time='2026-10-27_19:00:00', stop_time='2026-10-27_19:01:15')
    elif defect == 'missing_window':
        pair[1].pop('stop_time')
    elif defect == 'shortened_sample':
        for run in pair:
            run['steps'] = 6
            run['frames'] = run['frames'][:7]
    else:
        pair[1]['frames'].pop()
    with pytest.raises(ValueError):
        mint.checked_pair(pair, key)


def test_door_mints_fresh_key_and_passes_class_to_geometry_deck(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof.hex import regional_door as door
    from woof.hex.cuda_backend import regional_admission as admission

    key, pair, path = evidence(tmp_path)
    generated = dict(cull=dict(grid='grid.nc', static='static.nc', lbc_dir='lbc', row='new-cull'),
                     vertical=dict(n_vert_levels=55), parent=dict(dt_seconds=5))
    value = dict(paths=dict(contract=str(tmp_path), init='init.nc'), start_time='start')
    monkeypatch.setattr(door, '_generated', lambda value: generated)
    monkeypatch.setattr(door, 'regional_class_key', lambda *a, **kw: key)
    monkeypatch.setattr(admission, 'admitted_class_for_key', lambda key: None)
    calls = []

    def earn(measured, mesh, plan, status):
        calls.append(measured)
        return mint.read_class(path)

    def deck(command, **kwargs):
        assert command[command.index('--class-id') + 1] == mint.class_id(key)
        (tmp_path / 'contract.json').write_text('{}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mint, 'mint', earn)
    monkeypatch.setattr(door.subprocess, 'run', deck)
    result = door._stage_contract(None, value, SimpleNamespace(note=lambda line: None))
    assert calls == [key]
    assert len(result['mint_receipts']) == 2
    assert result['class_id'] == mint.class_id(key)


@pytest.mark.parametrize('dt, expected_steps', [(600, 6), (240, 15), (120, 15)])
def test_mint_stays_inside_parent_window(tmp_path, monkeypatch, dt, expected_steps):
    """One hour of boundaries must not die at mint step 7 on a 600 s clock.

    Exercise both real mint entry points and LbcInventory on CPU. Only the
    mesh/physics/device implementation and process launch are replaced.
    """
    from datetime import datetime, timedelta
    import sys
    from types import SimpleNamespace as NS
    import numpy as np
    from woof.hex import lbc

    start = datetime(2026, 10, 27, 18)
    stop = start + timedelta(hours=1)
    stamp = lambda time: time.strftime('%Y-%m-%d_%H:%M:%S')
    boundary_paths = []
    for minutes in (0, 20, 40, 60):
        path = tmp_path / f'lbc.{stamp(start + timedelta(minutes=minutes))}.nc'
        path.touch()
        boundary_paths.append(path)
    monkeypatch.setattr(lbc, 'read_lbc_valid_time',
                        lambda path: datetime.strptime(path.name[4:-3], '%Y-%m-%d_%H:%M:%S'))
    inventory = lbc.LbcInventory(boundary_paths)
    array = np.ones((1, 1), dtype=np.float32)
    mesh = NS(bdyMaskCell=np.array([7]), bdyMaskEdge=array, bdyMaskVertex=array,
              dcEdge=np.array([108581.3515625]))
    state = NS(rho=array, scalars=[array])
    saved = NS(normal_velocity=array, vertical_velocity=array, theta_m=array,
               pressure_perturbation=array)
    reference = NS(pressure_base=array)
    clocks = []

    def assemble(*args, **kwargs):
        clock = [kwargs['start_time']]
        clocks.append(clock)
        atmosphere = NS(state=state, saved=saved)

        def step():
            inventory.earliest_strictly_after(clock[-1])
            clock.append(clock[-1] + timedelta(seconds=dt))
            return NS(atmosphere=atmosphere)

        return NS(atmosphere=atmosphere, step_device=step, configuration_sha256='b' * 64,
                  regional_v841=NS(n_cells_solve=1, history_slice=lambda value: value),
                  cp=NS(asnumpy=lambda value: value,
                        cuda=NS(runtime=NS(getDeviceProperties=lambda index: {'name': 'CPU stub'}))))

    replacements = {
        'woof.hex.config_v841': NS(V841DryDycoreConfig=lambda **kw: NS(**kw)),
        'woof.hex.regional_v841': NS(RVORD_F32=np.float32(0.608)),
        'woof.hex.cuda_regional_forecast_v841': NS(assemble_regional_driver_v841=assemble),
        'woof.hex.driver': NS(load_mpas_initial_state=lambda *a, **kw: (state, reference, saved),
                             load_mpas_vertical_grid=lambda *a, **kw: NS(
                                 vertical_grid=NS(zz=array), terrain_metrics=None)),
        'woof.hex.dynamics_v841': NS(load_v841_reference_wind_profiles=lambda *a: None),
        'woof.hex.mesh': NS(Mesh=NS(from_netcdf=lambda *a, **kw: mesh),
                           regional_boundary_mask_digest=lambda masks: 'c' * 64),
    }
    for name, module in replacements.items():
        monkeypatch.setitem(sys.modules, name, module)
    from woof.hex.cuda_backend.regional_admission import kernel_set_sha256
    key = RegionalClassKey.build(boundary_zone_width=7, n_vert_levels=1,
                                finest_edge_m=float(mesh.dcEdge.min()), dt_seconds=dt,
                                kernel_set=kernel_set_sha256())
    generated = dict(cull=dict(grid='grid', static='static', lbc_dir=str(tmp_path), row='cull'),
                     parent=dict(row='parent'))
    value = dict(paths=dict(contract=str(tmp_path / 'contract'), init='init'),
                 start_time=stamp(start), stop_time=stamp(stop), fine_dx_m=120000)

    def launch(command, **kwargs):
        monkeypatch.setattr(sys, 'argv', command[2:])
        assert driver.main() == 0
        path = Path(command[command.index('--out') + 1])
        receipt = json.loads(path.read_text())
        receipt['pid'] = len(clocks)  # Simulate the two independent subprocess IDs.
        path.write_text(json.dumps(receipt))
        return NS(returncode=0)

    monkeypatch.setattr(mint.subprocess, 'run', launch)
    result = mint.mint(key, generated, value, NS(note=lambda line: None))
    assert len(clocks) == 2
    assert all(len(clock) == expected_steps + 1 and clock[-1] <= stop for clock in clocks)
    assert f'{expected_steps} steps each' in result.basis
    for path in result.mint_receipts:
        receipt = json.loads(Path(path).read_text())
        assert receipt['steps'] == expected_steps
        assert len(receipt['frames']) == expected_steps + 1


@pytest.mark.parametrize('stop', ['18:00:00', '18:05:00', '19:05:00'])
def test_mint_refuses_an_empty_or_partial_step_window_before_launch(tmp_path, monkeypatch, stop):
    from types import SimpleNamespace
    from woof.hex.errors import MpasPortError

    key = RegionalClassKey.build(boundary_zone_width=7, n_vert_levels=55,
                                finest_edge_m=108581.3515625, dt_seconds=600, kernel_set='a' * 64)
    value = dict(start_time='2026-10-27_18:00:00', stop_time='2026-10-27_' + stop)
    monkeypatch.setattr(mint.subprocess, 'run', lambda *a, **kw: pytest.fail('launched a mint'))
    with pytest.raises(MpasPortError, match='partial step would integrate past the admitted boundary window'):
        mint.mint(key, {}, value, SimpleNamespace(note=lambda line: None))
