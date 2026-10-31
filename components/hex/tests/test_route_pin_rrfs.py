"""Pin selection and installed declarations; these are not numerical proofs."""
from types import SimpleNamespace

import pytest

from woof.hex.mpasphys.forecast_v842 import native_route
from woof.hex.mpasphys.route_table import framework_entry, stream_contract
from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist
from woof.hex.physics_suites import load_table

KEY='mpasphys_rrfs_44c0849'


def test_explicit_pin_and_default_can_coexist():
    new,_=native_route(SimpleNamespace(physics_suite='hrrrv5',physics_pin_set='rrfs-44c0849'))
    old,_=native_route(SimpleNamespace(physics_suite='hrrrv5',physics_pin_set='rrfs'))
    assert old.pin_set=='rrfs'
    assert new.table_key==KEY and new.pin_set=='rrfs-44c0849'
    assert new.receipt()['model_commit']==load_table()[KEY]['model_commit']
    assert new.options['num_soil_layers']==9
    assert new.options['config_mp_aero_emission'] is False
    assert new.options['calc_prate_max'] is False
    assert new.options['config_microphysics_top']==45000.0
    assert 'calc_prate_max' not in old.options
    assert 'prate_max' in stream_contract(new).registry.data['fields']
    with pytest.raises(ValueError,match='unknown fork Registry field'):
        stream_contract(old).registry.field_definition('prate_max')


def test_named_pin_without_suite_match_is_an_error():
    with pytest.raises(ValueError,match='has no route'):
        native_route(SimpleNamespace(physics_suite='hrrrv5',physics_pin_set='typo'))


def test_new_options_reach_only_the_selected_config(tmp_path):
    path=tmp_path/'namelist.atmosphere'
    path.write_text('&physics config_physics_suite="hrrrv5", calc_prate_max=.true. /')
    new,text=native_route(SimpleNamespace(physics_namelist=path,physics_pin_set='rrfs-44c0849'))
    assert new.options['calc_prate_max'] is True
    assert new.options['num_soil_layers']==4  # Native Registry default with a supplied namelist.
    with pytest.raises(ValueError,match='no physics option'):
        native_route(SimpleNamespace(physics_namelist=path,physics_pin_set='rrfs'))


def test_framework_storage_authenticates_each_pin_without_global_switch():
    old=resolve_namelist('&physics config_physics_suite="hrrrv5" /')
    new=resolve_namelist('&physics config_physics_suite="hrrrv5" /',table_key=KEY)
    a=framework_entry(old,'backend')()
    b=framework_entry(new,'backend')()
    assert a.source_pin==old.receipt()['model_commit']
    assert b.source_pin==new.receipt()['model_commit']
    assert a.data!=b.data
    assert 'cldtop_cooling_p' not in a.abi['variables']
    assert 'cldtop_cooling_p' in b.abi['variables']
    assert framework_entry(new,'registry_source')()==b.data
    assert b.metadata_sha256()!=a.metadata_sha256()


def test_default_route_is_per_suite():
    """44c0849 is the default for hrrrv5, the suite proven on it (whole 6 h strict forecast, lam7k hour); the
    other RRFS aliases keep the 1d6e370e row until they are proven on the new pin."""
    default,_=native_route(SimpleNamespace(physics_suite='hrrrv5'))
    assert default.table_key==KEY and default.pin_set=='rrfs-44c0849'
    for suite in ('rrfs_mesoscale_reference','rrfs_convection_permitting'):
        other,_=native_route(SimpleNamespace(physics_suite=suite))
        assert other.pin_set=='rrfs', suite
