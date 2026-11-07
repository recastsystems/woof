"""Admission and inventory tests for the shared registry-bound column path."""
import numpy as np
from woof.config import RunConfig, validate_run_config
from woof.physics_registry import column_microphysics_binding, consumer_row_gaps
from woof.core.state import DomainState
from woof.core.device_inventory import state_array_shapes
from woof.ingest.wrfinput import active_moisture_inventory
from woof.io.history_layout import live_state_history_fields
from woof.io.wrf_output_schema import GSFCGCE4ICE_OUTPUT_FIELDS
from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS


def config():
    return RunConfig(nx=16, ny=16, nz=40, dx=3000.0, dy=3000.0, ztop=20000.0, run_seconds=18.0, dt=6.0, moist=True, mp_physics=7)


def test_bare_scheme_selector_validates_and_has_every_consumer_row():
    from woof.config import _MP_UNPORTED_SCHEMES
    cfg = config()
    validate_run_config(cfg)
    assert 7 not in _MP_UNPORTED_SCHEMES
    assert consumer_row_gaps(cfg) == []
    row = column_microphysics_binding(7)
    assert row['launcher'] == 'launch_gsfcgce4ice'
    assert row['transported_extras'] == ['qi', 'qs', 'qg', 'qh']
    assert 'nh' not in row['volume_fields']


def test_real_host_setup_allocation_matches_price_and_preserves_carried_fields():
    cfg = config()
    state = DomainState(cfg, array_module=np)
    row = column_microphysics_binding(7)
    prices = state_array_shapes(cfg)
    for name in row['volume_fields']:
        value = getattr(state, name)
        assert value.dtype == np.float32
        assert value.shape == prices[name] == (40, 16, 16)
        assert name in STATE_SERIALIZED_ATTRS
    for name in row['transported_extras']:
        assert getattr(state, name + '0').shape == prices[name + '0']
    assert getattr(state, 'qndrop', None) is None
    assert getattr(state, 'qnh', None) is None


def test_wrfinput_contract_requires_hail_but_no_number_moments():
    required, allowed = active_moisture_inventory(config())
    assert 'QHAIL' in required
    assert not {'QNHAIL', 'QNDROP', 'QNRAIN'} & allowed


def test_history_carries_every_process_accumulator_flux_and_radius_with_units():
    cfg = config()
    state = DomainState(cfg, array_module=np)
    output = live_state_history_fields(state)
    row = column_microphysics_binding(7)
    for name, attr in row['history_fields'].items():
        assert output[name] is getattr(state, attr)
        assert GSFCGCE4ICE_OUTPUT_FIELDS[attr].units
    assert output['QHAIL'] is state.qh


def test_same_scheme_nested_and_offline_inventories_carry_hail_once():
    from woof.core.nest_fields import nest_field_kinds
    from woof.offline_child import _transported_source_fields
    cfg = config()
    fields = nest_field_kinds(cfg)
    assert fields.count('qh') == 1
    assert _transported_source_fields(7) == ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh')


def test_mixed_edges_refuse_the_unvalidated_hail_water_conversion():
    import pytest
    from dataclasses import replace
    from woof.core.microphysics_transition import resolve_microphysics_transition
    cfg = config()
    with pytest.raises(ValueError, match='hail mass folding'):
        resolve_microphysics_transition(replace(cfg, mp_physics=6), cfg)


def test_standalone_helper_inventory_names_real_launchers_and_their_scope():
    import woof.core.gsfcgce4ice as implementation
    from woof.physics_registry import physics_registry
    option = physics_registry()['components']['microphysics']['options']['goddard4ice-mp7']
    helpers = option['extensions']['standalone_helpers']
    assert set(helpers) == {'negcor', 'auto_conversion', 'refl10cm_gsfc'}
    for entry in helpers.values():
        assert callable(getattr(implementation, entry['launcher']))
        assert entry['default_active'] is False
        assert entry['reason']
