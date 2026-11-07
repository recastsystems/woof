"""WRF's one-moment radiation flags precede all radius carrier reads."""
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('use_mp_re',[0,1])
def test_one_moment_radiation_never_reads_unwritten_radii(use_mp_re):
    from woof.core.rrtmg_legacy import legacy_scheme_has_req
    from woof.core.radiation_composition import radiation_radius_provider_fields
    class Unwritten:
        def __getattr__(self,name):
            raise AssertionError(f'read unwritten one-moment carrier {name}')
    assert legacy_scheme_has_req(18,use_mp_re,nssl_2moment_on=0)==(0,0,0)
    assert radiation_radius_provider_fields(Unwritten(),18,use_mp_re,
               nssl_2moment_on=0)==((None,)*6,False)


def test_two_moment_radiation_still_consumes_native_carriers():
    from woof.core.rrtmg_legacy import legacy_scheme_has_req
    from woof.core.radiation_composition import radiation_radius_provider_fields
    state=SimpleNamespace(effc=10,effi=30,effs=80)
    assert legacy_scheme_has_req(18,1)==(1,1,1)
    assert radiation_radius_provider_fields(state,18)==((30,10,None,80,None,None),False)


def test_modern_adapter_retains_ice_and_snow_without_scheme_radii():
    from woof.core.rrtmgp import cloud_optics_scheme,scheme_is_ice_active,scheme_has_snow_species
    assert cloud_optics_scheme(18)=="nssl"
    scheme=cloud_optics_scheme(18,nssl_2moment_on=0)
    assert scheme=="nssl1" and scheme_is_ice_active(scheme) and scheme_has_snow_species(scheme)


@pytest.mark.gpu
def test_modern_one_moment_merged_snow_path():
    from conftest import requires_gpu
    if requires_gpu.args[0]:pytest.skip('GPU unavailable')
    import cupy as cp
    import numpy as np
    from woof.core.rrtmgp import hydrometeor_paths
    p=cp.asarray([[90000,80000,70000]],dtype=cp.float32)
    qc=cp.full((1,2),1e-4,dtype=cp.float32)
    qi=qc*2;qs=qc*3
    paths=hydrometeor_paths(p,qc,qi=qi,qs=qs,microphysics='nssl1',
             effc=cp.full_like(qc,8),effi=cp.full_like(qc,30),
             effs=cp.full_like(qc,cp.nan),validate=False,
             snow_treatment='wrf-rrtmg-130um-snow-discount')
    # has_reqs=0 merges all snow mass; the unwritten snow radius is ignored.
    np.testing.assert_allclose(cp.asnumpy(paths.ciwp),cp.asnumpy((qi+qs)*cp.float32(10000*1000/9.80665)),rtol=2e-7)
