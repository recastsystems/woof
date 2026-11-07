"""Production-only preludes and NVCC-only flags must not break AMD builds."""
from pathlib import Path
from woof.hip_source import options_for_amd
from tools.check_hip_kernels import production_loader_unit


def test_amd_accepts_relaxed_division_and_register_hints():
    for dash in ('-', '--'):
        result=options_for_amd((dash+'prec-div=false',dash+'prec-sqrt=false',
                                dash+'maxrregcount=112','--ftz=false'))
        assert result==('-fhip-fp32-correctly-rounded-divide-sqrt',
                        '-fno-cuda-flush-denormals-to-zero')


def test_snow_inventory_uses_standalone_production_source():
    from woof.core import snow_papers
    source,options=production_loader_unit('snow_papers')
    assert source==(Path(snow_papers.__file__).with_name('kernels')/'snow_papers.cu').read_text()
    assert '#define G ' not in source
    assert options==('-std=c++17','--ftz=false')


def test_ocean_inventory_includes_production_pwp_header():
    import pytest
    pytest.importorskip('hexcore.mpasphys.ocean_pwp_v842')
    from importlib.resources import files
    from woof.hex.mpasphys.ocean_pwp_v842 import CUDA_SOURCE
    from woof.hex import numerics
    source,options=production_loader_unit('ocean_pwp')
    assert source==files('hexcore').joinpath('kernels/ocean_pwp.cuh').read_text()+CUDA_SOURCE
    assert options==tuple(v for v in numerics.options(('--std=c++17',))
                          if v.lstrip('-').split('=',1)[0]!='ftz')
