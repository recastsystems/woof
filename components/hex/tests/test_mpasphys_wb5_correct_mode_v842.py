"""Correct mode WB5 (cluster mpas-water-budget): WSM6's below-ground rain-out with the remap's profile.

Native v8.4.2 mp_wsm6.F90:1984 and :2254 book the part of the straddling sedimentation cell that falls below
the ground at the flat cell mean, while the remap moved the rest with a piecewise-linear profile, so column
plus surface water is not conserved. The native correct-mode build books it with the same profile. The
hex-cuda WSM6 slot compiles its scheme from text with those two statements corrected; exact mode compiles
the original text. CPU checks here; the whole-column words against the native build are in
tests/mpas_wb5_v842/test_mpasphys_wb5_columns_gpu_v842.py.
"""
from woof.hex.mpasphys import wsm6_v842
from woof.hex.mpasphys.driver_v842 import CORRECTIONS_V842, resolve_namelist

from test_mpasphys_correct_mode_v842 import deck


def test_WB5_is_requested_for_WSM6_only_and_settles_per_implementation():
    native = resolve_namelist(deck(mode='correct', skip='RP3,WB3'))                    # WSM6, native-cpu
    assert [(g.finding, g.slot, g.scheme, g.owner) for g in native.gaps] == [
        ('WB5', 'microphysics', 'mp_wsm6', CORRECTIONS_V842['WB5']['owners']['microphysics'])]
    cuda = resolve_namelist(deck(mode='correct', skip='RP3'), backends={'microphysics': 'hex-cuda'})
    assert {'WB3', 'WB5'} <= cuda.corrections and not cuda.gaps
    thompson = resolve_namelist(deck('convection_permitting', mode='correct'),
                                backends={'microphysics': 'hex-cuda'})
    assert 'WB5' not in thompson.corrections and not any(g.finding == 'WB5' for g in thompson.gaps)
    skipped = resolve_namelist(deck(mode='correct', skip='RP3,WB5'), backends={'microphysics': 'hex-cuda'})
    assert skipped.corrections == {'WB3'} and skipped.correct_skip == {'RP3', 'WB5'}


def test_WB5_correct_text_changes_exactly_the_two_rain_out_statements():
    exact = wsm6_v842.cuda_source('/* maths */')
    assert exact == wsm6_v842.cuda_source('/* maths */', corrections=())
    corrected = wsm6_v842.cuda_source('/* maths */', corrections=('WB5',))
    changed = [(a, b) for a, b in zip(exact.splitlines(), corrected.splitlines()) if a != b]
    assert len(exact.splitlines()) == len(corrected.splitlines()) and len(changed) == 2
    for (before, after), line in zip(changed, (1984, 2254)):
        assert before.endswith(f'// mp_wsm6.F90:{line}') and 'mul(qa[' in before
        # th = (0.0-za(k))/dza(k); precip += dza(k)*(qmi(k)*th + 0.5*(qpi(k)-qmi(k))*th*th), Fortran order.
        assert after.lstrip().startswith('th = div(sub(0.0f, za[') and 'mul(qa[' not in after
        assert 'mul(dza[((k)-(1))*(1)], add(mul(qmi[((k)-(1))*(1)], th), mul(mul(mul(0.5f, sub(qpi[' in after
    both = wsm6_v842.cuda_source('/* maths */', corrections=('WB3', 'WB5'))
    assert sum(a != b for a, b in zip(exact.splitlines(), both.splitlines())) == 3
