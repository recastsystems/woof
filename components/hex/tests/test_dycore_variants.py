"""Variant routing depends on table metadata and preserves default identity."""
from dataclasses import asdict
import pytest
from woof.hex.config_v841 import V841DryDycoreConfig
from woof.hex.config_rrfs import RRFSDryDycoreConfig
from woof.hex.dycore_variants import resolve_dycore_variant,dycore_driver_type

def test_default_is_same_object():
    config=V841DryDycoreConfig()
    before=asdict(config)
    for row in (None,{}, {'name':'arbitrary','source_pin':'arbitrary'}):
        assert resolve_dycore_variant(config,row) is config
    assert asdict(config)==before
    assert dycore_driver_type(config).__name__=='CudaDryDycoreDriver'

def test_arbitrary_names_same_variant():
    config=V841DryDycoreConfig()
    rows=[dict(name=name,source_pin=pin,dycore=dict(variant='mpas-v8.3.1-noaa',defaults={'config_epssm':.17,'config_lbc_w':'nearest'})) for name,pin in [('first','abc'),('renamed','def')]]
    selected=[resolve_dycore_variant(config,row) for row in rows]
    assert all(isinstance(c,RRFSDryDycoreConfig) for c in selected)
    assert selected[0]==selected[1]
    assert selected[0].config_epssm==.17
    assert dycore_driver_type(selected[0]).__name__=='CudaRRFSDycoreDriver'

def test_explicit_values_win_over_table_defaults():
    row={'dycore':{'variant':'mpas-v8.3.1-noaa','defaults':{'config_epssm':.1}}}
    result=resolve_dycore_variant(V841DryDycoreConfig(),row,overrides={'config_epssm':.2})
    assert result.config_epssm==.2
    assert result.config_scalar_eddy_mix is False

def test_native_provider_metadata_and_factories():
    from woof.hex.mpasphys.config_v842 import NativeMpasDycoreConfigV842
    from woof.hex.dycore_variants import dycore_kernel_cache_type,dycore_regional_constructor
    original=NativeMpasDycoreConfigV842(config_physics_suite='arbitrary-native-provider')
    selected=resolve_dycore_variant(original,{'dycore':{'variant':'mpas-v8.3.1-noaa'}})
    assert selected.config_physics_suite==original.config_physics_suite
    assert dycore_kernel_cache_type(original).__name__=='KernelCache'
    assert dycore_kernel_cache_type(selected).__name__=='RRFSKernelCache'
    assert dycore_regional_constructor(selected).__name__=='assemble_regional_driver_rrfs'


def test_new_noaa_pin_uses_upstream_dynamics_without_relabeling_old_fork():
    from woof.hex.config_rrfs import NoaaV841DycoreConfig
    from woof.hex.cuda_driver import CudaDryDycoreDriver
    from woof.hex.cuda_driver_rrfs import CudaRRFSDycoreDriver
    from woof.hex.dycore_variants import dycore_kernel_cache_type, dycore_regional_constructor
    original = V841DryDycoreConfig(config_dynamics_split_steps=3, config_split_dynamics_transport=True)
    new = resolve_dycore_variant(original, {'name': 'arbitrary-new-pin', 'model_commit': 'arbitrary-source',
        'pin_set': 'arbitrary-pin', 'dycore': {
        'variant': 'mpas-v8.4.1-noaa', 'defaults': {
            'prognostic_les_viscosity_floor': 0.0,
            'config_lbc_w': 'nearest', 'config_scalar_eddy_mix': True}}})
    old = resolve_dycore_variant(original, {'dycore': {'variant': 'mpas-v8.3.1-noaa'}})
    assert isinstance(new, NoaaV841DycoreConfig)
    assert dycore_driver_type(new) is CudaDryDycoreDriver
    assert dycore_driver_type(old) is CudaRRFSDycoreDriver
    assert dycore_regional_constructor(new).__name__ == 'open_regional_forecast_v841'
    assert dycore_kernel_cache_type(new).__name__ == 'RRFSKernelCache'
    assert new.config_epssm == 0.0 and old.config_epssm == 0.1
    assert new.config_lbc_w == 'nearest' and new.config_scalar_eddy_mix
    assert new.source_commit == 'arbitrary-source' and new.physics_pin_set == 'arbitrary-pin'
    CudaDryDycoreDriver._validate_config(new)

@pytest.mark.parametrize('entry',[{'variant':'absent'}, {'variant':'mpas-v8.3.1-noaa','defaults':{'misspelled':1}}])
def test_invalid_metadata_reports_unimplemented_selection(entry):
    with pytest.raises(ValueError):resolve_dycore_variant(V841DryDycoreConfig(),{'dycore':entry})

@pytest.mark.parametrize('splits,split_transport',[(1,False),(1,True),(2,True),(3,True),(4,True)])
def test_fork_schedule_validation(splits,split_transport):
    from woof.hex.cuda_driver_rrfs import CudaRRFSDycoreDriver
    config=RRFSDryDycoreConfig(config_dynamics_split_steps=splits,
                             config_split_dynamics_transport=split_transport)
    config.validate()
    CudaRRFSDycoreDriver._validate_config(config)


