"""RRFS route rows must reach the resident driver without a library fallback."""
import pytest

from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist
from woof.hex.mpasphys.resident_v842 import ported_route_refusal, resident_requested
from woof.hex.mpasphys.runtime_v842 import framework_only


@pytest.mark.parametrize('sub3d', [0, 1, 2])
def test_ntiedtke_uses_gpu_with_inert_gfl_setting(sub3d):
    config = resolve_namelist(
        "&physics\nconfig_physics_suite='hrrrv5'\n"
        "config_convection_scheme='cu_ntiedtke'\n"
        f"config_gfl_sub3d={sub3d}\n/\n", default_backend='hex-cuda')
    assert config.options['config_gfl_sub3d'] == sub3d
    assert str(config.backends['convection']) == 'hex-cuda'
    assert framework_only(config)
    assert ported_route_refusal(config) is None
    assert resident_requested(config, {'nCells': 2562, 'nCellsSolve': 2562,
                                       'nEdges': 7680, 'nEdgesSolve': 7680})


@pytest.mark.parametrize('suite', ['rrfs_convection_permitting', 'rrfs_mesoscale_reference'])
def test_alias_slots_use_ported_resident_route(suite):
    config = resolve_namelist(f"&physics\nconfig_physics_suite='{suite}'\n/\n",
                              default_backend='hex-cuda')
    assert all(str(config.backends[slot]) == 'hex-cuda'
               for slot in config.selectors if config.active(slot))
    assert framework_only(config)
    assert ported_route_refusal(config) is None
    assert resident_requested(config, {'nCells': 2562, 'nCellsSolve': 2562,
                                       'nEdges': 7680, 'nEdgesSolve': 7680})


@pytest.mark.parametrize('suite', ['rrfs_convection_permitting', 'rrfs_mesoscale_reference'])
def test_alias_descriptors_register_with_fork_rules(suite):
    from woof.hex.mpasphys.rrfs.driver_rrfs import SlotRegistry, specification
    config = resolve_namelist(f"&physics\nconfig_physics_suite='{suite}'\n/\n",
                              default_backend='hex-cuda')
    registry = SlotRegistry()
    for slot, scheme in config.selectors.items():
        if not config.active(slot):
            continue
        row = specification()['slots'][slot]['cuda_entries'][scheme]
        registry.register(slot, scheme, config.backends[slot], lambda call: None,
                          corrections=row.get('corrections', ()))
    registry.require(config)


def test_noah_land_row_owns_post_lsm_diagnostics():
    from importlib import import_module
    from woof.hex.mpasphys.rrfs.driver_rrfs import specification
    from woof.hex.mpasphys.sfclay_mm5_v842 import run_native_diagnostics
    row = specification()['slots']['land_surface']['cuda_entries']['sf_noah']
    module = import_module('woof.hex.mpasphys.' + row['module'])
    assert getattr(module, row['post_lsm_diagnostics']) is run_native_diagnostics


def test_fork_adapters_do_not_replace_v842_rows():
    from woof.hex.physics_suites import load_table
    slots = load_table()['mpasphys_v842']['slots']
    assert slots['pbl']['cuda_entries']['bl_mynn']['module'] == 'bl_mynn_v842'
    row = slots['surface_layer']['cuda_entries']['sf_monin_obukhov_rev']
    assert row['module'] == 'sfclay_mm5_v842'
    assert tuple(row['corrections']) == ('RP3',)
