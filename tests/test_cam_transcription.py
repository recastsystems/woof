"""Regression checks for source conversion mistakes that alter WRF radiation."""
from pathlib import Path
import importlib.util,sys
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools/cam_wrf461_oracle'))
spec=importlib.util.spec_from_file_location('cam_transcribe',root/'tools/cam_wrf461_oracle/transcribe.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_realnu_assignment_is_not_a_declaration():
    assert module.declarations('realnu = (1./beta)*te')==[]
    assert module.declarations('real(r8) realnu')[0][0]=='realnu'


def test_default_real_promotion_keeps_its_original_word():
    converted=module.numeric_literals('0.7')
    assert float(converted)==float(np.float32(0.7))
    assert float(converted)!=0.7


def test_explicit_r8_literals_keep_binary64_kind():
    assert module.numeric_literals('0.7_r8')=='0.7d0'


def test_spaced_array_declaration():
    assert module.declarations('real(r8) abo3 (nspint)')[0][:3] == ('abo3','double precision',['nspint'])


def test_inline_operator_literal():
    assert module.numeric_literals('if (g.gt.1._r8) call report') == 'if (g.gt.1.d0) call report'


def test_loop_exit_and_cycle_labels():
    from normalize_arrays import explicit
    code,_=explicit(['do','if(i==2) cycle','if(i==3) exit','end do'],{})
    assert code == ['do while(.true.)','if(i==2) goto 9002','if(i==3) goto 9003','9002 continue','end do','9003 continue']


def test_whole_array_copy_is_elementwise():
    from normalize_arrays import explicit
    code,_=explicit(['left=right'],{'left':['2','3'],'right':['2','3']})
    assert any('right(1+cam_s1_0,1+cam_s1_1)' in s for s in code)


def test_select_case_preserves_default_branch():
    from normalize_arrays import explicit
    code,_=explicit(['select case(choice)','case(8)','x=1','case(5)','x=2','case default','x=3','end select'],{})
    assert code==['if(choice==8) then','x=1','else if(choice==5) then','x=2','else','x=3','endif']


def test_driver_storage_covers_interface_and_retained_absorption_fields():
    import json
    meta=json.loads((root/'woof/core/kernels/cam3_emissivity.json').read_text())['camrad']
    fields={x['name']:x for x in meta['parameters']}
    assert fields['rthratenlw']['dtype']=='float'
    assert fields['rthratenlw']['count']==33
    assert fields['abstot_3d']['count']==33*33
    assert fields['absnxt_3d']['count']==33*4


def test_driver_gas_values_do_not_replace_saved_co2_path_constant():
    source=(root/'woof/core/kernels/cam3_emissivity.cu').read_text()
    wrapper=source[source.index('void cam_driver('):]
    assert 'state->driver_gases[0]=gases[column*5+0]' in wrapper
    assert 'state->cg_co2vmr__=gases' not in wrapper


def test_vertical_specialization_keeps_nested_indices_separate():
    from specialize_vertical import local_arrays
    text='doublereal a[1089]; double x=a[i + k*33 - 34]; double y=a[b[35] + k*33 -34];'
    result=local_arrays(text,{'a':{'dtype':'double precision','shape':['pverp','pverp']}})
    assert 'CAM_LEVELS' in result
    assert 'b[35]' in result
    assert 'a[1089]' not in result


def test_workspace_stride_aligns_each_column():
    import json
    from pathlib import Path
    metadata=json.loads(Path('woof/core/kernels/cam3_emissivity.json').read_text())
    for levels in (1,31,32,48,64,96):
        stride=eval(metadata['camrad']['workspace_expression'],{'CAM_LEVELS':levels})
        assert stride > 0 and stride % 8 == 0
        assert all((column*stride)%8 == 0 for column in range(64))


def test_vertical_specialization_restores_strength_reduced_stride():
    from specialize_vertical import local_arrays
    source='doublereal q[96]; q[i + (k + (species << 5)) - 34] = value;'
    result=local_arrays(source,{'q':{'dtype':'double precision','shape':['pcols','pver','3']}})
    assert '<< 5' not in result
    assert 'CAM_LEVELS' in result


def test_vertical_specialization_restores_folded_stride_multiple():
    from specialize_vertical import local_arrays
    source='integer istrtd[578]; istrtd[68]=nconfig+1; istrtd[k+68]=2;'
    result=local_arrays(source,{'istrtd':{'dtype':'integer','shape':['0:pverp','0:16']}})
    assert '[68]' not in result and 'k+68' not in result
    assert 'CAM_LEVELS' in result
