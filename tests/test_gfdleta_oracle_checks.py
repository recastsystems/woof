"""Completion comparison must reject errors without masking a field."""
import importlib.util
from pathlib import Path
import pytest

P=Path(__file__).parents[1]/'tools/gfdleta_wrf461_oracle/validate_gfdleta.py'
spec=importlib.util.spec_from_file_location('gfdl_validate',P)
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)

def test_one_bit_co2_miss_is_not_tolerated():
    ref={('CO251','13'):('CO251',0x3f000000)}
    got={('CO251','13'):('CO251',0x3f000001)}
    result=check.compare_words(ref,got)
    assert not result['pass']
    assert result['fields']['CO251']['max_ulp']==1
    assert result['fields']['CO251']['word_misses']==1

def test_missing_and_extra_keys_fail():
    ref={('CO251','1'):('CO251',0x3f800000)}
    assert not check.compare_words(ref,{})['pass']
    assert not check.compare_words({},ref)['pass']

def test_signed_zero_requires_word_identity():
    ref={('GLW','1'):('GLW',0)};got={('GLW','1'):('GLW',0x80000000)}
    result=check.compare_words(ref,got)
    assert not result['pass'] and result['fields']['GLW']['word_misses']==1

@pytest.mark.parametrize('body',['field,index,bits\nCO251,1,7F800000\n','field,index,bits\nCO251,1,3F800000\nCO251,1,3F800000\n','field,index,bits\n','field,index,bits\nCO251,1,XYZ\n','field,index,bits\nCO251,1,3F800000,extra\n'])
def test_invalid_fixture_is_rejected(tmp_path,body):
    path=tmp_path/'fixture.csv';path.write_text(body)
    with pytest.raises(ValueError):check.read_words(path)

def test_new_module_has_separate_no_ftz_options():
    from woof.core.kernels import module_options
    from woof.wrf_exact import ENABLED
    expected = ('-std=c++17','--ftz=false','--fmad=false','-DGPUWM_GFDLETA_STRICT=1') if ENABLED else ('-std=c++17','--ftz=false')
    assert module_options('gfdleta_init')==expected
    assert module_options('thompson')==('-std=c++17',)

@pytest.mark.parametrize('version,selected', [('2.39', 0), ('2.43', 1)])
def test_log10_binding_tracks_the_named_referee(monkeypatch, version, selected):
    from woof.core.kernels import _extra_header_text
    monkeypatch.setenv('GPUWM_WRF_GLIBC', version)
    for name in ('gfdleta_math_probe', 'gfdleta_co2_init', 'gfdleta_column'):
        assert _extra_header_text(name).startswith(
            f'#define GPUWM_GFDLETA_GLIBC243 {selected}\n')
