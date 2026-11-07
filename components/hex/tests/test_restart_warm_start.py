"""A native restart is a real restart on every MPAS-suite row (restart-check, x1.163842 hrrrv5, 2026-11-06).

Breakages pinned here:
* the hrrrv5 row (mpasphys_rrfs_44c0849) and the MPAS-A row had no initial-state selector, so
  config_do_restart = true cold-started the dynamics from init.nc beside restart-read physics;
* with the restart selected, the dycore loaders refused "zgrid=None" because an RRFS restart holds no
  zgrid (it is in the invariant stream) and they read only --init with cold_start=True;
* eight same-host rank threads assembling the RRFS source cache raced in _link_relative;
* a missing prebuilt kernel pack logged a routine note while every module compiled on one thread.
"""
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from woof.hex.physics_suites import load_table
from woof.hex.mpasphys.forecast_v842 import initial_state_control, initial_state_input


def _suite_rows():
    return [key for key, row in load_table().items()
            if key.startswith('mpasphys_') and isinstance(row, dict)]


@pytest.mark.parametrize('key', _suite_rows())
def test_every_suite_row_selects_the_restart_stream(key):
    config = SimpleNamespace(table_key=key, execution_options={})
    control, _ = initial_state_control(config, '&restart\n config_do_restart = .true.\n/\n')
    assert control.initial_stream == 'restart'
    assert control.initialize_coupled_diagnostics is False
    cold, _ = initial_state_control(config, '&restart\n config_do_restart = .false.\n/\n')
    assert cold.initial_stream == 'input'
    assert cold.initialize_coupled_diagnostics is True


def test_no_row_carries_its_own_selector_copy():
    assert not [key for key in _suite_rows() if 'initial_state_selector' in load_table()[key]]


def test_da_cycling_stream_follows_the_row_core():
    text = '&restart\n config_do_DAcycling = .true.\n/\n'
    rrfs, _ = initial_state_control(SimpleNamespace(table_key='mpasphys_rrfs_44c0849', execution_options={}), text)
    mpas, _ = initial_state_control(SimpleNamespace(table_key='mpasphys_v842', execution_options={}), text)
    assert (rrfs.initial_stream, mpas.initial_stream) == ('da_state', 'input')
    both, _ = initial_state_control(SimpleNamespace(table_key='mpasphys_rrfs_44c0849', execution_options={}),
                                    '&restart\n config_do_restart = .true.\n config_do_DAcycling = .true.\n/\n')
    assert both.initial_stream == 'restart' and both.initialize_coupled_diagnostics is True


def test_hrrrv5_restart_binding_is_the_initial_state(tmp_path):
    cold, restart = tmp_path / 'init.nc', tmp_path / 'restart.2023-04-01_12.00.00.nc'
    config = SimpleNamespace(table_key='mpasphys_rrfs_44c0849', execution_options={})
    args = SimpleNamespace(init=cold, start_time=None)
    assert initial_state_input(args, config, '&restart config_do_restart = .true. /', None,
                               {'restart': restart}) == ('restart', restart)


def _nc(path, names):
    from netCDF4 import Dataset
    with Dataset(path, 'w') as dataset:
        dataset.createDimension('nCells', 2)
        for name in names:
            dataset.createVariable(name, 'f4', ('nCells',))[:] = np.zeros(2, np.float32)
    return path


def test_warm_start_reads_the_vertical_grid_from_the_invariant_stream(tmp_path):
    from woof.hex.mpasphys.runtime_v842 import native_state_sources
    restart = _nc(tmp_path / 'restart.nc', ('rho_zz', 'theta_m'))
    invariant = _nc(tmp_path / 'invariant.nc', ('zgrid', 'zz'))
    layout = SimpleNamespace(configs={'config_do_restart': True})
    args = SimpleNamespace(init=restart, _initial_state_cold_start=False, _initial_state_invariant=invariant)
    assert native_state_sources(args, layout) == (restart, invariant, invariant, False)


def test_warm_start_keeps_a_restart_that_carries_its_grid(tmp_path):
    from woof.hex.mpasphys.runtime_v842 import native_state_sources
    restart = _nc(tmp_path / 'restart.nc', ('rho_zz', 'zgrid'))
    invariant = _nc(tmp_path / 'invariant.nc', ('zgrid',))
    layout = SimpleNamespace(configs={'config_do_restart': True})
    args = SimpleNamespace(init=restart, _initial_state_cold_start=False, _initial_state_invariant=invariant)
    assert native_state_sources(args, layout) == (restart, restart, None, False)


def test_cold_start_sources_are_unchanged(tmp_path):
    from woof.hex.mpasphys.runtime_v842 import native_state_sources
    init = tmp_path / 'init.nc'
    layout = SimpleNamespace(configs={'config_do_restart': False})
    args = SimpleNamespace(init=init, _initial_state_cold_start=True)
    assert native_state_sources(args, layout) == (init, init, None, True)


def test_source_assembly_has_one_assembler(tmp_path, monkeypatch):
    from woof.hex.mpasphys import resources_build as rb
    inside, peak, lock = [0], [0], threading.Lock()

    def slow(cache, **options):
        with lock:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        time.sleep(0.05)
        with lock:
            inside[0] -= 1
        return {'path': str(cache), 'sha256': '0' * 64}

    monkeypatch.setattr(rb, '_assemble_rrfs_source_unlocked', slow)
    threads = [threading.Thread(target=rb.assemble_rrfs_source, args=(tmp_path,), kwargs={'pin_set': 'rrfs'})
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 1


def test_installed_record_keeps_every_concurrent_row(tmp_path):
    from woof.hex.mpasphys import resources_build as rb
    rows = [{f'key-{index}': {'path': str(tmp_path / str(index)), 'sha256': f'{index:064d}'}} for index in range(8)]
    threads = [threading.Thread(target=rb.record_installed, args=(tmp_path, row)) for row in rows]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert set(rb.read_installed(tmp_path)['resources']) == {f'key-{index}' for index in range(8)}


def test_missing_kernel_pack_is_a_defect_naming_the_nvrtc():
    from woof.hex.cuda_backend.aot import pack_defect_line
    line = pack_defect_line({'nvrtc': [13, 4], 'nvrtc_package': 'nvidia-cuda-nvrtc==13.4.59',
                             'nvrtc_library_sha256': '5fc18c75a772c5fe'}, 'no compatible pack is downloaded')
    assert line.startswith('DEFECT: ') and '\n' not in line
    assert 'NVRTC 13.4 (nvidia-cuda-nvrtc==13.4.59, library sha256 5fc18c75a772)' in line
