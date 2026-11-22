"""Correct mode WB3 (cluster mpas-water-budget): precipitable water from the dry density.

Native v8.4.2 precip_to_MPAS divides rho_p by (1+qv) although microphysics_from_MPAS already made rho_p the
dry density (mpas_atmphys_driver_microphysics.F:614-617), so precipw is 0.6 % low on average. The hex-cuda
WSM6, Thompson and Kessler slots carry the same diagnostic; in correct mode each compiles its CUDA text with
that one statement corrected, and exact mode compiles the original text byte for byte.

CPU checks here: settlement per backend, the text substitution, and that each slot forwards the run's
corrections. The words themselves, against the native correct-mode build, are held by the GPU test
tests/mpas_wb3_v842/test_mpasphys_wb3_kernels_gpu_v842.py.
"""
from types import SimpleNamespace

import pytest

from woof.hex.mpasphys import kessler_v842, thompson_v842, wsm6_v842
from woof.hex.mpasphys.driver_v842 import CORRECTIONS_V842, correct_source, resolve_namelist, slot_corrections

from test_mpasphys_correct_mode_v842 import deck

OWNERS = CORRECTIONS_V842['WB3']['owners']['microphysics']


def gap_keys(config):
    return [(g.finding, g.slot, g.scheme, g.backend, g.owner) for g in config.gaps]


def test_WB3_is_requested_for_active_microphysics_and_settles_per_implementation():
    native = resolve_namelist(deck(mode='correct', skip='RP3,WB5'))                   # WSM6, native-cpu
    assert 'WB3' not in native.corrections
    assert gap_keys(native) == [('WB3', 'microphysics', 'mp_wsm6', 'native-cpu', OWNERS['mp_wsm6'])]
    for text, scheme in ((deck(mode='correct', skip='RP3'), 'mp_wsm6'),
                         (deck('convection_permitting', mode='correct', skip='RP1,RP3'), 'mp_thompson'),
                         (deck('none', mode='correct', config_microp_scheme='mp_kessler'), 'mp_kessler')):
        cuda = resolve_namelist(text, backends={'microphysics': 'hex-cuda'})
        assert cuda.selectors['microphysics'] == scheme
        assert 'WB3' in cuda.corrections and not any(g.finding == 'WB3' for g in cuda.gaps), scheme
    aerosols = resolve_namelist(deck('none', mode='correct', config_microp_scheme='mp_thompson_aerosols'),
                                backends={'microphysics': 'hex-cuda'})
    assert ('WB3', 'microphysics', 'mp_thompson_aerosols', 'hex-cuda', OWNERS['mp_thompson_aerosols']) in gap_keys(aerosols)
    dry = resolve_namelist(deck('none', mode='correct'))
    assert dry.selectors['microphysics'] == 'off' and 'WB3' not in dry.corrections and not dry.gaps
    assert resolve_namelist(deck()).corrections == frozenset()


def one_line_change(exact: str, corrected: str):
    before, after = exact.splitlines(), corrected.splitlines()
    assert len(before) == len(after)
    return [(a, b) for a, b in zip(before, after) if a != b]


@pytest.mark.parametrize('module,exact_text,correct_text', [
    (wsm6_v842, lambda: wsm6_v842.cuda_source('/* maths */'),
     lambda: wsm6_v842.cuda_source('/* maths */', corrections=('WB3',))),
    (thompson_v842, lambda: thompson_v842.cuda_source(device_math_source='/* maths */'),
     lambda: thompson_v842.cuda_source(device_math_source='/* maths */', corrections=('WB3',))),
    (kessler_v842, lambda: kessler_v842.CUDA_SOURCE + kessler_v842.CUDA_RELHUM_SOURCE,
     lambda: correct_source(kessler_v842.CUDA_SOURCE + kessler_v842.CUDA_RELHUM_SOURCE,
                            kessler_v842.CORRECT_MODE_SOURCE, ('WB3',), 'kessler_v842')),
])
def test_WB3_correct_text_changes_only_the_density_statement(module, exact_text, correct_text):
    (old, new, count), = module.CORRECT_MODE_SOURCE['WB3']
    exact, corrected = exact_text(), correct_text()
    assert exact.count(old) == count == 1 and new not in exact
    changed = one_line_change(exact, corrected)
    assert len(changed) == 1
    before, after = changed[0]
    assert old in before and new in after and before.replace(old, new) == after
    # The exact statement divides by (1+qv); the corrected one is the dry density itself.
    assert '1.' in old and '1.' not in new and 'rho' in new


def test_correct_source_refuses_text_that_moved():
    table = {'WB3': (('float rho_a=old;', 'float rho_a=new;', 1),)}
    assert correct_source('x', table, (), 'here') == 'x'
    with pytest.raises(RuntimeError, match='occurs 0 times'):
        correct_source('float rho_a=other;', table, ('WB3',), 'here')


def call_with(corrections, **fields):
    return SimpleNamespace(slot='microphysics', corrections=frozenset(corrections), **fields)


def test_slot_corrections_keep_only_the_implementations_findings():
    assert slot_corrections(call_with({'WB3', 'RP6'}), {'WB3': ()}) == ('WB3',)
    assert slot_corrections(call_with(()), {'WB3': ()}) == ()
    assert slot_corrections(SimpleNamespace(), {'WB3': ()}) == ()


@pytest.mark.parametrize('corrections,expected', [((), None), (('WB3', 'RP6'), ('WB3',))])
def test_kessler_and_wsm6_slots_forward_the_runs_corrections(monkeypatch, corrections, expected):
    seen = {}
    monkeypatch.setattr(kessler_v842, 'run_mpas', lambda **kw: seen.setdefault('kessler', kw))
    call = call_with(corrections, pools={}, dt=10.0, options={'config_bucket_rainnc': 0.0},
                     flags=SimpleNamespace(rain_bucket=False, diagnostics=False))
    kessler_v842.run_slot(call)
    assert seen['kessler'].get('corrections') == expected
    monkeypatch.setattr(wsm6_v842, 'step', lambda *a, **kw: seen.setdefault('wsm6', kw))
    call = call_with(corrections, pools={'fields': {}, 'constants': None, 'options': {}},
                     options={'wsm6_device_math_header': '/* maths */'})
    wsm6_v842.cuda_entry(call)
    assert seen['wsm6'].get('corrections') == expected


@pytest.mark.parametrize('corrections,expected', [((), None), (('WB3',), ('WB3',))])
def test_thompson_slot_forwards_the_runs_corrections_to_the_precipitation_writeback(monkeypatch, corrections,
                                                                                     expected):
    pytest.importorskip('cupy')
    seen = []
    monkeypatch.setattr(thompson_v842, 'launch_precip', lambda phase, *a, **kw: seen.append((phase, kw)))
    for name in ('launch_conversion', 'launch_driver', 'launch_radar'):
        monkeypatch.setattr(thompson_v842, name, lambda *a, **kw: None)
    pools = {'thompson_v842': {'n_microp': 1, 'device_math_source': '/* maths */', 'fields': {},
                               'model': None, 'mesh': None, 'diagnostics': {}, 'tendencies': None,
                               'constants': None, 'surface': None, 'precip': None, 'tables': None,
                               'l_acrain': False, 'dt_microp': 10.0, 'mp_top_level': 1,
                               'has_reqc': 1, 'has_reqi': 1, 'has_reqs': 1}}
    call = call_with(corrections, scheme=thompson_v842.SCHEME, backend=thompson_v842.BACKEND, pools=pools,
                     dt=10.0, options={'config_bucket_rainnc': 0.0}, flags=SimpleNamespace(diagnostics=False))
    thompson_v842._run_bound_slot(call)
    assert [phase for phase, _ in seen] == ['from', 'to']
    assert seen[0][1].get('corrections') is None                 # the reset phase computes no precipw
    assert seen[1][1].get('corrections') == expected
