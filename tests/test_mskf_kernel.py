"""MSKF fixture and parity checks. CUDA checks run only in the card queue."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
from woof.verify.mskf_oracle import load_fixture, grade, run_port


def test_fixture_integrity_and_coverage():
    root = Path('tests/data/oracles/mskf')
    for line in (root / 'oracle-sha256sums.txt').read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    fx = load_fixture(root)
    assert np.isfinite(fx.out).all()
    assert fx.ncase >= 64
    assert {int(c['dx']) for c in fx.meta['cases']} >= {1000, 3000, 9000, 27000}
    assert {c['aercu_opt'] for c in fx.meta['cases']} == {0, 1, 2}
    assert grade(fx, fx.out.copy()).differing == 0


def test_planted_one_bit_is_caught():
    fx = load_fixture()
    changed = fx.out.copy()
    changed.view(np.uint32)[0, 0] ^= np.uint32(1)
    g = grade(fx, changed)
    assert (g.differing, g.max_ulp) == (1, 1)
    assert g.per_field['rthcuten'] == (1, 1)


@requires_gpu
def test_column_oracle_no_aerosol():
    fx = load_fixture()
    cases = [i for i, c in enumerate(fx.meta['cases']) if c['aercu_opt'] == 0]
    got = run_port(fx, cases)
    g = grade(fx, got, cases)
    assert g.differing == 0, g


@requires_gpu
def test_column_oracle_with_aerosol():
    fx = load_fixture()
    cases = [i for i, c in enumerate(fx.meta['cases']) if c['aercu_opt'] > 0]
    got = run_port(fx, cases)
    g = grade(fx, got, cases)
    assert g.differing == 0, g


@requires_gpu
@pytest.mark.parametrize('ncol', [65, 129])
def test_tile_tail_matches_column_oracle(monkeypatch, ncol):
    import cupy as cp
    from woof.core import mskf
    from woof.verify.mskf_oracle import case_columns, SENTINEL, FIELDS_3D, FIELDS_2D
    fx = load_fixture()
    case = case_columns(fx, 0)
    nz = fx.nz
    sc, ci = case['scalars'], case['ints']
    inputs = {n: cp.broadcast_to(cp.asarray(case['cols'][n])[:, None, None],
                               (nz, 1, ncol)).copy() for n in mskf.INPUT_3D}
    inputs.update({n: cp.full((1, ncol), sc[n], dtype=cp.float32)
                   for n in mskf.INPUT_2D})
    w = cp.broadcast_to(cp.asarray(case['cols']['w'])[:, None, None],
                        (nz + 1, 1, ncol)).copy()
    mskf.update_w0avg(inputs['w0avg'], w, dt=sc['dt'], cudt_minutes=sc['cudt'],
                      stepcu=ci['stepcu'], adaptive=bool(ci['adapt_step']))
    outputs = {n: cp.full((nz, 1, ncol), SENTINEL, dtype=cp.float32)
               for n in mskf.OUTPUT_3D}
    outputs.update({n: cp.full((1, ncol), SENTINEL, dtype=cp.float32)
                    for n in mskf.OUTPUT_2D})
    outputs['nca'][...] = np.float32(sc['nca'])
    outputs['cu_act'] = cp.zeros((1, ncol), dtype=cp.int32)
    # Limit workspace to 64 columns. The arithmetic still runs on the GPU.
    monkeypatch.setattr(mskf, '_multiprocessors', lambda: 1)
    phase = mskf.MskfPhase(bool(ci['warm_rain']), bool(ci['f_qr']),
                         bool(ci['f_qi']), bool(ci['f_qs']))
    assert mskf.launch_mskf(inputs, outputs, dx=sc['dx'], dt=sc['dt'],
                            cudt_minutes=sc['cudt'], phase=phase) == 0
    words = [cp.asnumpy(inputs[n] if n == 'w0avg' else outputs[n]).reshape(nz, ncol)
             for n in FIELDS_3D]
    words.extend(cp.asnumpy(outputs[n]).astype(np.float32).reshape(1, ncol)
                 for n in FIELDS_2D)
    actual = np.concatenate(words).T.copy()
    expected = np.repeat(fx.out[[0]], ncol, axis=0)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))



def test_layout_refuses_wrong_horizontal_shape():
    from woof.core import mskf
    inputs = {n: np.zeros((40, 1, 2), np.float32) for n in mskf.INPUT_3D}
    inputs.update({n: np.zeros((1, 2), np.float32) for n in mskf.INPUT_2D})
    outputs = {n: np.zeros((40, 1, 2), np.float32) for n in mskf.OUTPUT_3D}
    outputs.update({n: np.zeros((1, 2), np.float32) for n in mskf.OUTPUT_2D})
    outputs['cu_act'] = np.zeros((1, 2), np.int32)
    assert mskf.validate_layout(inputs, outputs) == (40, 2)
    inputs['u'] = np.zeros((40, 1, 1), np.float32)
    with pytest.raises(ValueError, match='outside the field'):
        mskf.validate_layout(inputs, outputs)
    inputs['u'] = np.zeros((40, 1, 2), np.float32)
    outputs['cu_act'] = np.zeros((1, 2), np.float32)
    with pytest.raises(ValueError, match='cu_act'):
        mskf.validate_layout(inputs, outputs)



def test_signed_zero_bit_difference_is_not_accepted():
    fx = load_fixture()
    changed = fx.out.copy()
    c, i = (int(x) for x in np.argwhere(changed == 0)[0])
    changed.view(np.uint32)[c, i] ^= np.uint32(0x80000000)
    assert grade(fx, changed).differing == 1


@requires_gpu
def test_device_one_bit_change_is_caught():
    import cupy as cp
    from woof.core import mskf
    from woof.verify.mskf_oracle import run_port_case
    fx = load_fixture()
    actual = run_port_case(fx, 0)
    assert grade(fx, actual[None], [0]).differing == 0
    def planted(inputs, outputs, **kw):
        status = mskf.launch_mskf(inputs, outputs, **kw)
        words = outputs['rthcuten'].view(cp.uint32).ravel()
        words[0] ^= cp.uint32(1)
        return status
    corrupted = run_port_case(fx, 0, kernel_override=planted)
    result = grade(fx, corrupted[None], [0])
    assert (result.differing, result.max_ulp) == (1, 1)
    assert result.per_field_error['rthcuten']['reference_sha256'] != result.per_field_error['rthcuten']['port_sha256']



def test_aerosol_integer_and_real_powers_have_distinct_lowering():
    from tools.mskf_wrf461_oracle.transcribe_aerosol import expr, Variable
    symbols={'x': Variable('x','double',[])}
    assert expr('x**4',symbols) == 'ma_powi<4>(x)'
    assert expr('x**4._r8',symbols) == 'ma_pow(x,4.)'
    assert expr('x**2.5_r8',symbols) == 'ma_pow(x,2.5)'



def test_aerosol_supersaturation_count_is_a_constant():
    from pathlib import Path
    p=Path('woof/core/kernels/mskf_aerosol.cuh')
    assert 'static constexpr int psat=6;' in p.read_text()



def test_mskf_registry_has_the_full_column_contract():
    from woof.physics_registry import physics_registry
    row=physics_registry()['components']['cumulus']['options']['multiscale-kf']
    assert row['implemented'] and row['selectors']=={'cu_physics':11}
    contract=row['extensions']['column_contract']
    assert {'rucuten','rvcuten'} <= set(contract['result_mapping'])
    assert {'cldfra_dp','cldfra_sh','qc_cu','qi_cu','efcs','efis','efss'} <= set(contract['radiation_feedback'])


def test_aercu_pair_refusal_names_missing_handoff():
    from woof.core.morrison_aero import refuse_aercu_opt
    with pytest.raises(ValueError,match='effective-radius handoff'):
        refuse_aercu_opt(1,cu_physics=11,mp_physics=10)
    with pytest.raises(ValueError,match='effective-radius handoff'):
        refuse_aercu_opt(2,cu_physics=1,mp_physics=40)
    refuse_aercu_opt(0,cu_physics=11,mp_physics=40)
    for arm in (1,2):
        refuse_aercu_opt(arm,cu_physics=11,mp_physics=40)


@requires_gpu
def test_aerosol_initialization_matches_recorded_wrf_words():
    import cupy as cp
    from woof.core.kernels import get_kernel
    ref=np.fromfile('tests/data/oracles/mskf/mphy-constants.bin','<f8')
    got=cp.empty(ref.size,cp.float64)
    ws=cp.empty(512*42*32,cp.float64)
    get_kernel('mskf_aerosol_replay','mskf_aero_constants')((1,),(32,),(got,ws))
    np.testing.assert_array_equal(got.get().view('u8'),ref.view('u8'))
