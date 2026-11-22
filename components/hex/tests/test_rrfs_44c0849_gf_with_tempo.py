"""hrrrv5's physics with MPAS-A Grell-Freitas (cu_grell_freitas) on the GPU, on the rrfs-44c0849 route.

THE BREAKAGE THIS PREVENTS: a 10 km regional control asked for hrrrv5 with ``config_convection_scheme =
'cu_grell_freitas'`` (20 Nov 2026). The 44c0849 table had no CUDA entry for it, so convection resolved to native-cpu
and the run needed a native RRFS library; with the entry added by hand, the resident glue refused the selection
("convection=cu_grell_freitas (needs off)") before step 1. The run went to a rebuilt native library with convection on
the CPU: 35 to 40 minutes for 18 h against 2 to 3 minutes for the plain GPU run.

Nothing in the fork stood in the way. Native 44c0849 runs the same scheme file as MPAS-A v8.4.2
(physics_wrf/module_cu_gf.mpas.F, sha256 b63086af, the file gf_mpas_v842 transcribes), with the same call;
its convection glue for the scheme differs only by reading and writing rvcuten(k,k), which leaves every word as it
was (driver_rrfs.convection_from_mpas/convection_to_mpas reproduce it). The 1d6e370e row already routes the scheme
to gf_mpas_v842, and the resident glue carries GF's tendencies for GF-Li and New Tiedtke.
"""
import pytest

from woof.hex.physics_suites import load_table
from woof.hex.mpasphys import gf_mpas_v842, route_table
from woof.hex.mpasphys.resident_v842 import ported_route_refusal, resident_requested
from woof.hex.mpasphys.rrfs.driver_rrfs import Backend, resolve_namelist
from woof.hex.mpasphys.runtime_v842 import framework_only

ROW = 'mpasphys_rrfs_44c0849'
DIMS = dict(nCells=3, nCellsSolve=3, nEdges=6, nEdgesSolve=6)
HRRRV5 = dict(microphysics='mp_tempo', pbl='bl_mynnedmf', surface_layer='sf_mynnsfclay', land_surface='sf_ruc',
              gwdo='bl_ugwp_gwdo', radiation_lw='rrtmg_lw', radiation_sw='rrtmg_sw')


def _plain(x):
    if hasattr(x, 'items'):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _deck(convection, suite='hrrrv5'):
    return f"&physics\nconfig_physics_suite='{suite}'\nconfig_convection_scheme='{convection}'\n/\n"


def test_44c0849_routes_mpas_grell_freitas_to_the_cuda_port():
    table = load_table()
    entry = _plain(table[ROW]['slots']['convection']['cuda_entries']['cu_grell_freitas'])
    assert entry == _plain(table['mpasphys_rrfs']['slots']['convection']['cuda_entries']['cu_grell_freitas'])
    assert entry == {'module': 'gf_mpas_v842', 'call': 'run_slot', 'bind': 'bind_native_pools'}
    # The port transcribes the file native 44c0849 compiles (its physics_wrf/module_cu_gf.mpas.F).
    assert gf_mpas_v842.SOURCE_SHA256 == 'b63086afc7f7213d1ec04de13a2b4a77dde718167c22304a030741da05fb5baf'


def test_44c0849_resident_targets_carry_hrrrv5_with_grell_freitas():
    targets = [_plain(t) for t in load_table()[ROW]['resident']['targets']]
    assert dict(HRRRV5, convection='cu_grell_freitas') in targets


def test_hrrrv5_with_grell_freitas_runs_on_the_resident_gpu_glue(monkeypatch):
    config = resolve_namelist(_deck('cu_grell_freitas'), default_backend='hex-cuda', table_key=ROW)
    assert config.selectors['convection'] == 'cu_grell_freitas'
    assert config.selectors['microphysics'] == 'mp_tempo'
    assert config.backends['convection'] == Backend.HEX_CUDA
    if not framework_only(config):
        pytest.skip('the ported RRFS framework is not in this distribution')
    assert ported_route_refusal(config) is None
    monkeypatch.setenv('WOOF_HEX_MPAS_RESIDENT', 'auto')
    assert resident_requested(config, DIMS)


def test_the_pairing_is_labelled_as_a_variant_never_as_hrrrv5_alone():
    name, row = route_table.named_variant('hrrrv5', {'config_convection_scheme': 'cu_grell_freitas'})
    assert name == 'hrrrv5 + Grell-Freitas'
    assert row['differences_from_suite']
    # GF-Li keeps its own name.
    assert route_table.named_variant('hrrrv5', {'config_convection_scheme': 'cu_grell_freitas_li'})[0] == \
        'hrrrv5 + storm scheme'


def test_a_refused_choice_names_the_schemes_that_run_on_the_gpu_with_the_rest():
    """The 1d6e370e row (--physics-pin-set rrfs) still admits no cumulus scheme with TEMPO but New Tiedtke: the
    refusal says which choices do run with the run's other physics, as a flag a user can pass."""
    config = resolve_namelist(_deck('cu_grell_freitas'), default_backend='hex-cuda', table_key='mpasphys_rrfs')
    if not framework_only(config):
        pytest.skip('the ported RRFS framework is not in this distribution')
    refusal = ported_route_refusal(config)
    assert refusal is not None
    assert 'config_convection_scheme=cu_ntiedtke' in refusal
    assert 'config_convection_scheme=off' in refusal
    assert 'cu_grell_freitas (' not in refusal.split('runs on the GPU')[-1]


def test_a_cpu_slot_without_a_library_stops_at_the_door_with_the_gpu_choices(monkeypatch):
    """THE BREAKAGE: hrrrv5 with a cumulus scheme the row has no GPU slot for printed "PHYSICS ON CPU", fetched 1 GB of
    RRFS tables (115 s) and stopped on "the packaged physics manifest has no assets for these pin sets"."""
    from argparse import Namespace
    from woof.hex.mpasphys import forecast_v842, resources_build
    config = resolve_namelist(_deck('cu_tiedtke'), default_backend='hex-cuda', table_key=ROW)
    assert str(config.backends['convection']) == 'native-cpu'
    monkeypatch.delenv('MPAS_NATIVE_LIBRARY', raising=False)

    def not_installed(*_args, **_kwargs):
        raise resources_build.PhysicsResourceError('not installed')

    monkeypatch.setattr(resources_build, 'resolve_resource', not_installed)
    refusal = forecast_v842.cpu_slot_refusal(Namespace(native_physics_library=None), config)
    assert 'convection=cu_tiedtke' in refusal
    for scheme in ('off', 'cu_ntiedtke', 'cu_grell_freitas_li', 'cu_grell_freitas'):
        assert f'--physics config_convection_scheme={scheme}' in refusal
    # A named library, or a selection with every slot on the GPU, goes on.
    assert forecast_v842.cpu_slot_refusal(Namespace(native_physics_library='/site/libmpas.so'), config) is None
    gpu = resolve_namelist(_deck('cu_grell_freitas'), default_backend='hex-cuda', table_key=ROW)
    assert forecast_v842.cpu_slot_refusal(Namespace(native_physics_library=None), gpu) is None
