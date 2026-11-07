"""NVIDIA keeps canonical PTX barriers; AMD receives portable operations."""
import pytest
from woof.hip_source import _translate_assembly, translate


def test_nvidia_retains_validated_canonical_assembly():
    source='__device__ float f(float a,float b) {float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;}'
    assert _translate_assembly(source,nvidia=True)==source
    assert 'r = __fadd_rn(a, b);' in _translate_assembly(source)
    assert 'asm("add.rn.f32' in translate(source,nvidia=True)
    assert 'r = gpuwm_fadd_rn(a, b);' in translate(source)


def test_nvidia_still_refuses_unknown_translation_unit_assembly():
    with pytest.raises(ValueError,match='cannot translate'):
        _translate_assembly('asm("unreviewed.ptx %0;" : "=f"(r));',nvidia=True)


def test_amd_keeps_nvrtc_local_index_alias_without_stdint_collision():
    source='typedef long long int64_t; __device__ int64_t index(int64_t x) {return x+1;}'
    assert 'typedef long long gpuwm_cuda_int64_t;' in translate(source)
    assert 'gpuwm_cuda_int64_t index(gpuwm_cuda_int64_t x)' in translate(source)
    assert source in translate(source,nvidia=True)


def test_amd_keeps_stdint_selection_for_rtc_guarded_alias():
    source='''#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#else
typedef long long int64_t;
#endif
__device__ int64_t index(int64_t x) {return x+1;}
'''
    assert 'gpuwm_cuda_int64_t' not in translate(source)


def test_local_alias_does_not_rename_earlier_standard_type():
    source='''#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#endif
__device__ int64_t first(int64_t x) {return x;}
typedef long long int64_t;
__device__ int64_t second(int64_t x) {return x;}
'''
    translated=translate(source)
    assert 'int64_t first(int64_t x)' in translated
    assert 'gpuwm_cuda_int64_t second(gpuwm_cuda_int64_t x)' in translated
