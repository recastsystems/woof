"""Real-card CCN coefficient treatment and an in-branch one-bit control.

These are production-seam treatment tests. They do not claim WRF parity.
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from conftest import requires_gpu

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/nssl2_wrf461_variant_oracle'))
import cases
import compare
import run_gpu


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('label',['cccn-low','cccn-high'])
def test_coefficient_is_treated_in_the_real_production_seam(label):
    mode=cases.MODE_BY_LABEL[label]
    resolved=run_gpu.admit(mode)
    actual,inputs=run_gpu.run_mode(mode,resolved,3,6.0,2)
    default,control_inputs=run_gpu.run_mode(replace(mode,cccn=0.5e9),resolved,3,6.0,2)
    assert inputs==control_inputs
    assert all(np.isfinite(a).all() for a in actual.values())
    assert not np.array_equal(actual['qnn'].view(np.uint32),default['qnn'].view(np.uint32))


@pytest.mark.gpu
@requires_gpu
def test_one_bit_mutation_in_the_coefficient_branch_is_caught(monkeypatch):
    import cupy as cp
    from woof.core import kernels,nssl2_nucond
    from woof.certify.kernel_manifest import record_module
    mode=cases.MODE_BY_LABEL['cccn-low']
    resolved=run_gpu.admit(mode)
    reference,_=run_gpu.run_mode(mode,resolved,3,6.0,2)
    source=kernels.module_source_int_defines(
        'nssl2_nucond', (('NSSL2_VARIABLE_CCCN', 1),))
    site='    int predicted_ccn)\n#endif\n{'
    assert source.count(site)==1
    source=source.replace(site,site+"""
    if (nssl_cccn != 0.5e9f) {
        nssl_cccn = __uint_as_float(__float_as_uint(nssl_cccn) ^ 1u);
        background_ccn_per_kg = __fdiv_rn(nssl_cccn, 1.225f);
    }
""")
    options=kernels.module_options('nssl2_nucond')
    module=cp.RawModule(code=source,options=options,name_expressions=None)
    module.compile()
    record_module('test.nssl2_cccn.one_bit',source=source,options=options,module=module)
    normal=nssl2_nucond.get_kernel_int_defines
    monkeypatch.setattr(nssl2_nucond,'get_kernel_int_defines',lambda name,function,defines:
                        module.get_function(function) if name=='nssl2_nucond'
                        else normal(name,function,defines))
    mutated,_=run_gpu.run_mode(mode,resolved,3,6.0,2)
    result=compare.compare_mode(reference,mutated,'strict')
    assert not result['passed']
    assert result['fields']['qnn']['cells_differing'] > 0
