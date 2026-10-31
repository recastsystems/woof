"""RRFS core.F:139-150: one initial stream shared by physics and dynamics."""
from pathlib import Path
from types import SimpleNamespace

from woof.hex.mpasphys.forecast_v842 import initial_state_input
from woof.hex.mpasphys.rrfs.cold_start_rrfs import select_initial_state_stream


def test_restart_precedes_da_with_fortran_case_insensitive_options():
    assert select_initial_state_stream({'config_do_dacycling': True}) == 'da_state'
    assert select_initial_state_stream({'CONFIG_DO_RESTART': True, 'config_do_dacycling': True}) == 'restart'


def test_da_binding_overrides_cold_input(tmp_path):
    cold, da = tmp_path / 'cold.nc', tmp_path / 'analysis.nc'
    config = SimpleNamespace(table_key='mpasphys_rrfs', execution_options={})
    args = SimpleNamespace(init=cold, start_time=None)
    name, path = initial_state_input(args, config,
        '&restart config_do_DAcycling = .true. /', None, {'da_state': da})
    assert (name, path) == ('da_state', da)
    assert args.init == cold


def test_da_time_template_uses_deck_directory(tmp_path):
    cold, da = tmp_path / 'cold.nc', tmp_path / 'analysis.2023-03-31_13.00.00.nc'
    da.touch()
    streams = tmp_path / 'streams.atmosphere'
    streams.write_text('<streams><stream name="da_state" type="input;output" '
        'filename_template="analysis.$Y-$M-$D_$h.$m.$s.nc" input_interval="initial_only" '
        'output_interval="none" /></streams>')
    config = SimpleNamespace(table_key='mpasphys_rrfs', execution_options={})
    args = SimpleNamespace(init=cold, start_time='2023-03-31_13:00:00')
    assert initial_state_input(args, config, '&restart config_do_DAcycling=.true. /',
                               streams, {}) == ('da_state', da)


def test_missing_da_file_does_not_fall_back_to_cold_input(tmp_path):
    import pytest
    cold = tmp_path / 'cold.nc'
    cold.touch()
    streams = tmp_path / 'streams.atmosphere'
    streams.write_text('<streams><stream name="da_state" type="input;output" '
        'filename_template="missing.nc" input_interval="initial_only" /></streams>')
    config = SimpleNamespace(table_key='mpasphys_rrfs', execution_options={})
    args = SimpleNamespace(init=cold, start_time='2023-03-31_13:00:00')
    with pytest.raises(FileNotFoundError, match='da_state'):
        initial_state_input(args, config, '&restart config_do_DAcycling=.true. /', streams, {})


def test_mutable_da_members_override_registry_default_package():
    from xml.etree import ElementTree as ET
    from woof.hex.mpasphys.pool_v842 import RegistryLayout

    layout = SimpleNamespace(
        root=ET.fromstring('<registry><streams><stream name="input" type="input">'
            '<var name="cold_only" /></stream><stream name="da_state" type="input;output" '
            'packages="jedi_da"><var name="u" /></stream></streams></registry>'),
        enabled=lambda element: element.get('packages') != 'jedi_da',
        configs={'config_do_restart': False},
        fields=[dict(pool='state', name=name, kind='real', dimensions=['nCells'],
                     shape=[2], components=[], initial='0', required=False)
                for name in ('u', 'cold_only')])
    request = RegistryLayout.request(layout, [], initial_state=('da_state', ('u',)))
    assert {f['name']: f['read_input'] for f in request['fields']} == {'u': True, 'cold_only': False}


def test_initial_only_output_uses_cycle_time():
    from datetime import datetime
    from woof.hex.streams_rrfs import Stream
    stream = Stream('probe', dict(type='output', filename_template='probe.$h.nc',
                                  output_interval='initial_only', filename_interval='none'), ())
    assert stream.filename(datetime(2023, 3, 31, 13), datetime(2023, 3, 31, 12)) == 'probe.13.nc'


def test_mutable_include_copies_fields_without_old_package_gates(tmp_path):
    from xml.etree import ElementTree as ET
    from woof.hex.mpasphys.forecast_v842 import read_streams
    layout = SimpleNamespace(packages={'no_invariant_stream'}, fields=[
        dict(name='mesh_field', pool='mesh', components=[]),
        dict(name='state_field', pool='state', components=[])],
        root=ET.fromstring('<registry><streams>'
            '<stream name="invariant"><var name="mesh_field"/></stream>'
            '<stream name="restart"><stream name="invariant" packages="no_invariant_stream"/>'
            '<var name="state_field"/></stream></streams></registry>'))
    layout.enabled = lambda e: not e.get('packages') or bool(set(e.get('packages').split(';')) & layout.packages)
    path = tmp_path / 'streams'
    path.write_text('<streams><immutable_stream name="invariant" type="input" '
        'input_interval="initial_only" filename_template="invariant.nc"/>'
        '<immutable_stream name="restart" type="input;output" filename_template="restart.nc"/>'
        '<stream name="probe" type="output" filename_template="probe.nc"><stream name="restart"/></stream></streams>')
    streams = {s.name: s for s in read_streams(path, layout)}
    assert streams['restart'].fields == ('state_field',)
    assert streams['probe'].fields == ('mesh_field', 'state_field')


def test_real_da_coupled_state_matches_native_initial():
    import os
    import numpy as np
    import pytest
    from netCDF4 import Dataset
    from woof.hex.mesh import Mesh
    from woof.hex.driver import load_mpas_vertical_grid, load_mpas_initial_state
    folder = os.environ.get('WOOF_HEX_DA_REFERENCE_CASE')
    if not folder:
        pytest.skip('set WOOF_HEX_DA_REFERENCE_CASE to grid.nc, da_state.nc, native_initial.nc')
    folder = Path(folder)
    mesh = Mesh.from_netcdf(folder/'grid.nc', folder/'da_state.nc', native_parent_chain=True)
    grid = load_mpas_vertical_grid(folder/'da_state.nc', mesh, config_coef_3rd_order=.25)
    state, _, saved = load_mpas_initial_state(folder/'da_state.nc', mesh, grid.vertical_grid,
        terrain_metrics=grid.terrain_metrics, cold_start=True, return_saved_diagnostics=True)
    fields = {'rho_zz': state.rho, 'theta_m': saved.theta_m, 'ru': state.rho_u,
              'rw': state.rho_w, 'rho_p': saved.density_perturbation,
              'rtheta_p': saved.rho_theta_perturbation}
    with Dataset(folder/'native_initial.nc') as native:
        for name, value in fields.items():
            expected = np.asarray(native[name][0]).T.copy()
            assert np.array_equal(value.view(np.uint32), expected.view(np.uint32)), name
