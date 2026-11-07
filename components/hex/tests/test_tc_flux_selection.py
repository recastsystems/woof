"""Both horizontal engines select the same shared surface formulation."""
import pytest
from woof.hex import physics_suites as ps


@pytest.mark.parametrize('suite', ['arwen_mpas_column', 'rrfs_v2_dev'])
def test_shared_tc_selection(suite):
    if suite == 'arwen_mpas_column':
        pytest.importorskip('cupy')  # its selection is planned by the engine's CuPy column seam
    chosen = ps.resolve_physics(suite=suite, overrides={
        'tc_flux_curve': 'bell2012', 'tc_spray': 'off'}, nominal_dx_m=3000.)
    assert chosen.generation['tc_flux_curve'] == 'bell2012'
    assert chosen.generation['tc_spray'] == 'off'
    assert chosen.seam in ('engine', 'column_suite')


def test_invalid_tc_formulation_is_refused():
    with pytest.raises(ps.PhysicsSelectionRefusal, match='tc_flux_curve'):
        ps.resolve_physics(suite='arwen_mpas_column', overrides={'tc_flux_curve': 'invented'})



def test_inactive_final_receipt_is_inert():
    from types import SimpleNamespace
    from woof.hex.tc_flux import final_receipt
    assert final_receipt(object(), steps=100) is None
    backend = SimpleNamespace(_seam=SimpleNamespace(
        _driver=SimpleNamespace(tc_surface=None)))
    assert final_receipt(backend, steps=100) is None


def test_native_mynn_uses_extension_without_polluting_registry():
    from types import SimpleNamespace
    from woof.hex.mpasphys.forecast_v842 import native_route
    arguments = SimpleNamespace(physics_suite='convection_permitting',
        physics=["config_pbl_scheme='bl_mynn'", 'tc_flux_curve=bell2012', 'tc_spray=off'])
    config, text = native_route(arguments)
    assert config.selectors['surface_layer'] == 'sf_mynn'
    assert config.tc_surface == {'tc_flux_curve': 'bell2012', 'tc_spray': 'off'}
    assert config.receipt()['tc_surface'] == config.tc_surface
    assert 'tc_flux_curve' not in text and 'tc_flux_curve' not in config.options
    arguments.physics = ["config_pbl_scheme='bl_mynn'", 'tc_flux_curve=off']
    off, _ = native_route(arguments)
    assert not off.tc_surface and 'tc_surface' not in off.receipt()


def test_native_cpu_referee_cannot_silently_accept_tc_exchange():
    from types import SimpleNamespace
    from woof.hex.mpasphys.forecast_v842 import native_route
    with pytest.raises(ValueError, match='CUDA MYNN or MM5'):
        native_route(SimpleNamespace(physics_suite='convection_permitting',
            physics_default_backend='native-cpu', physics=['tc_flux_curve=bell2012']))
