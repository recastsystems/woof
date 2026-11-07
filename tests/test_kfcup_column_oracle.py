"""Grade the resumed KF-CuP source against WRF 4.6.1 captured output words."""
import numpy as np
import pytest
from woof.verify.kfcup_oracle import (
    configs, load_capture, pack_column_inputs, expected_outputs,
    unpack_outputs, differing_words,
)
from woof.verify.kfcup_host import run_columns

@pytest.mark.parametrize('cfg', configs(), ids=lambda c: c.name)
def test_kfcup_complete_column_words(cfg):
    cap = load_capture(cfg.index)
    assert cap.ncol >= 64, 'acceptance capture needs at least 64 columns'
    buffers = pack_column_inputs(cap)
    status = run_columns(buffers, dt=cfg.dt, dx=cfg.dx, ntst=cfg.stepcu,
                         numbins=cfg.numbins, thbinsize=cfg.thbinsize,
                         rbinsize=cfg.rbinsize, mindeepfreq=cfg.mindeepfreq,
                         minshallowfreq=cfg.minshallowfreq, flags=cfg.flags)
    assert status == 0, f'KF-CuP kernel status {status}'
    differences = differing_words(unpack_outputs(buffers), expected_outputs(cap))
    assert not any(differences.values()), {k:v for k,v in differences.items() if v}

@pytest.mark.parametrize('cfg', configs(), ids=lambda c: c.name)
def test_kfcup_trigger_history_words(cfg):
    from woof.verify.kfcup_host import run_history
    cap = load_capture(cfg.index)
    got = cap.level_in['w0avg'][:cap.nz].copy()
    run_history(got, cap.level_in['w'].copy(), stepcu=cfg.stepcu)
    np.testing.assert_array_equal(got.view(np.uint32),
                                  cap.level_out['w0avg'].view(np.uint32))

@pytest.mark.parametrize('icloud,forced', [(1,False),(1,True),(3,False),(3,True)])
def test_kfcup_radiation_merge_words(icloud, forced):
    from woof.verify.kfcup_host import run_radiation
    from woof.verify.kfcup_oracle import _rows, word
    rows = [r for r in _rows('kfcup-rad-levels.csv')
            if int(r['icloud']) == icloud and (r['shallowcu_forced_ra']=='T') == forced]
    assert len({r['case'] for r in rows}) >= 64
    values = {key: np.array([word(r[key]) for r in rows], np.float32)
              for key in ('shall','cubot','cutop','qc_in','cldfra_cup_in','cldfra1')}
    values['flag'] = np.array([int(r['cldfra1_flag']) for r in rows],np.int32)
    values['k'] = np.array([int(r['k']) for r in rows],np.int32)
    got = run_radiation(values, icloud=icloud, forced=forced)
    for key in ('cldfra','qc','cldfra_cup'):
        expected = np.array([word(r[key]) for r in rows], np.float32)
        np.testing.assert_array_equal(got[key].view(np.uint32), expected.view(np.uint32))


def test_kfcup_trigger_bit_mutation_is_caught():
    from woof.core.kernels import module_source
    source = module_source('kfcup')
    needle = '    if (sl.cupflag) {'
    assert source.count(needle) == 1
    changed = source.replace(needle, '    if (col == 0) sl.cupflag = !sl.cupflag;\n'+needle)
    cfg = configs()[0]
    cap = load_capture(cfg.index)
    b = pack_column_inputs(cap)
    run_columns(b,dt=cfg.dt,dx=cfg.dx,ntst=cfg.stepcu,numbins=cfg.numbins,
                thbinsize=cfg.thbinsize,rbinsize=cfg.rbinsize,mindeepfreq=cfg.mindeepfreq,
                minshallowfreq=cfg.minshallowfreq,flags=cfg.flags,source=changed)
    differences = differing_words(unpack_outputs(b), expected_outputs(cap))
    assert differences['cupflag'] == 1
    assert sum(differences.values()) > 1, 'trigger mutation must alter physical outputs'
    print('TRIGGER_MUTATION', {k:v for k,v in differences.items() if v})


def test_kfcup_radiation_one_bit_mutation_is_caught():
    from woof.core.kernels import module_source
    from woof.verify.kfcup_host import run_radiation
    from woof.verify.kfcup_oracle import _rows, word
    source = module_source('kfcup')
    needle = 'k <= (int)cutop) cf = cup;'
    assert source.count(needle) == 1
    changed = source.replace(needle, 'k <= (int)cutop) cf = __uint_as_float(__float_as_uint(cup)^1u);')
    rows = [r for r in _rows('kfcup-rad-levels.csv')
            if r['icloud']=='3' and r['shallowcu_forced_ra']=='F']
    v={key:np.array([word(r[key]) for r in rows],np.float32)
       for key in ('shall','cubot','cutop','qc_in','cldfra_cup_in','cldfra1')}
    v['flag']=np.array([int(r['cldfra1_flag']) for r in rows],np.int32)
    v['k']=np.array([int(r['k']) for r in rows],np.int32)
    got=run_radiation(v,icloud=3,forced=False,source=changed)['cldfra'].view(np.uint32)
    exp=np.array([word(r['cldfra']) for r in rows],np.float32).view(np.uint32)
    xor=got^exp
    assert np.count_nonzero(xor)>0
    assert np.all(xor[xor!=0]==1), 'every planted change must be exactly the low bit'
    print('RADIATION_ONE_BIT_MUTATION',np.count_nonzero(xor))


def test_kfcup_default_compile_options_preserve_wrf_operations():
    from woof.core.kernels import module_options
    assert module_options('kfcup') == ('-std=c++17','--fmad=false','--ftz=false')
    assert module_options('kf') == ('-std=c++17',)


def test_kfcup_history_preserves_subnormal_rounding():
    from woof.verify.kfcup_host import run_history
    avg=np.array([[3]],np.uint32).view(np.float32)
    w=np.zeros((2,1),np.float32)
    run_history(avg,w,stepcu=1)
    # 3 units of the least positive subnormal, divided by two, ties to 2.
    np.testing.assert_array_equal(avg.view(np.uint32),np.array([[2]],np.uint32))
