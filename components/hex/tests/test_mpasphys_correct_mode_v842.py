"""Correct-mode switch of the MPAS-A v8.4.2 physics driver (TRIAGE.md cluster 4: RP1, RP3, RP4, RP6).

Each test that names a finding fails on the exact (matched MPAS) behaviour and passes in correct mode.
These are control tests of the driver seam; the forecast receipts live with the native runs.
"""
from fractions import Fraction

import pytest

from woof.hex.mpasphys.driver_v842 import (
    CORRECTIONS_V842, RP6_RESTART_GAP, SURFACE_PRECIPITATION_WINDOW, Backend, NativeDriverHooks, PhysicsDriver, PhysicsManager,
    SlotRegistry, resolve_namelist,
)
from woof.hex.mpasphys.forecast_v842 import split_physics_mode


def deck(suite='mesoscale_reference', mode=None, skip=None, **options):
    values = {'config_physics_suite': suite, **options}
    text = '&physics\n' + '\n'.join(f'{k} = {v!r}' for k, v in values.items()) + '\n/'
    nhyd = {}
    if mode is not None:
        nhyd['config_physics_mode'] = mode
    if skip is not None:
        nhyd['config_physics_correct_skip'] = skip
    if nhyd:
        text = '&nhyd_model\n' + '\n'.join(f'{k} = {v!r}' for k, v in nhyd.items()) + '\n/\n' + text
    return text


class RecordingHooks:
    def __init__(self):
        self.operations = []

    def invoke(self, operation, call):
        self.operations.append(operation)

    def restart_state(self):
        return {}

    def restore_restart_state(self, payload):
        assert payload == {}


def driver_for(config, dt, *, declare=None):
    """Fake slots: native-cpu ones declare nothing; ``declare`` maps slot -> declared finding ids."""
    hooks, registry, seen = RecordingHooks(), SlotRegistry(), []
    for slot in config.selectors:
        if not config.active(slot) or config.native_embedded(slot):
            continue
        def callback(call):
            hooks.operations.append(call.slot)
            seen.append(call)
        registry.register(slot, config.selectors[slot], config.backends[slot], callback,
                          corrections=(declare or {}).get(slot, ()))
    driver = PhysicsDriver(config, PhysicsManager(config, dt), registry, hooks)
    driver.initialize(None)
    hooks.operations.clear()
    return driver, hooks, seen


def run_steps(driver, dt, steps):
    for n in range(steps):
        driver.begin_step(itimestep=n + 1, time_seconds=n * dt, pools=None)
        driver.physics_get_tend(rk_step=1, dynamics_substep=1, pools=None)
        driver.finish_step(pools=None)


def test_exact_is_the_default_and_its_receipt_is_unchanged():
    plain = resolve_namelist(deck())
    assert plain.mode == 'exact' and plain.corrections == frozenset()
    assert set(plain.receipt()) == {'schema', 'suite', 'selectors', 'options', 'backends'}
    assert resolve_namelist(deck(mode='exact')) == plain


def test_a_misspelt_mode_is_refused_by_name():
    with pytest.raises(ValueError, match='config_physics_mode'):
        resolve_namelist(deck(mode='corect'))
    with pytest.raises(ValueError, match='unknown findings'):
        resolve_namelist(deck(mode='correct', skip='RP9'))
    with pytest.raises(ValueError, match="only with config_physics_mode='correct'"):
        resolve_namelist(deck(skip='RP3'))


def test_correct_mode_receipt_names_its_applied_findings_and_named_gaps():
    config = resolve_namelist(deck(mode='correct', config_pbl_interval='00:12:00'))
    assert config.mode == 'correct'
    # YSU suite: no MYNN finding is active.  RP3 is active (config_frac_seaice defaults true) but its slots
    # run native-cpu, which can never implement it: a named gap, not a refusal.
    assert config.corrections == {'RP6'}
    owners = {finding: row['owners'] for finding, row in CORRECTIONS_V842.items()}
    seaice = owners['RP3']['seaice']
    sfclay = owners['RP3']['surface_layer']['sf_monin_obukhov_rev']
    wsm6 = owners['WB5']['microphysics']
    assert wsm6 == owners['WB3']['microphysics']['mp_wsm6']
    assert [(g.finding, g.slot, g.scheme, g.backend, g.owner) for g in config.gaps] == [
        ('RP3', 'seaice', 'sf_noah', 'native-cpu', seaice),
        ('RP3', 'surface_layer', 'sf_monin_obukhov_rev', 'native-cpu', sfclay),
        ('WB3', 'microphysics', 'mp_wsm6', 'native-cpu', wsm6),
        ('WB5', 'microphysics', 'mp_wsm6', 'native-cpu', wsm6)]
    receipt = config.receipt()
    assert receipt['physics_mode'] == 'correct' and receipt['corrections'] == ['RP6']
    block = receipt['correct_mode']
    assert block['mode'] == 'correct' and block['applied'] == ['RP6'] and block['skipped'] == []
    assert [(g['finding'], g['slot'], g['reason'], g['owner']) for g in block['gaps']] == [
        ('RP3', 'seaice', 'slot does not implement it', seaice),
        ('RP3', 'surface_layer', 'slot does not implement it', sfclay),
        ('WB3', 'microphysics', 'slot does not implement it', wsm6),
        ('WB5', 'microphysics', 'slot does not implement it', wsm6)]
    assert all('never implement' in g['note'] and g['breakage'] for g in block['gaps'])
    skipped = resolve_namelist(deck(mode='correct', skip='RP3,WB3,WB5', config_pbl_interval='00:12:00'))
    assert skipped.corrections == {'RP6'} and skipped.receipt()['correct_skip'] == ['RP3', 'WB3', 'WB5']
    assert skipped.gaps == () and skipped.gap_warning() is None


def test_gap_warning_is_loud_multiline_and_names_every_gap():
    config = resolve_namelist(deck('convection_permitting', mode='correct'))
    warning = config.gap_warning()
    lines = warning.splitlines()
    assert lines[0].startswith('CORRECT MODE: named gaps') and 'WARNING' in lines[0]
    assert len(lines) == 2 + len(config.gaps) and all(line.startswith('CORRECT MODE:') for line in lines)
    for gap in config.gaps:
        assert any(gap.finding in line and gap.slot in line for line in lines)
        # The printed lines name the gap and its reason; the run record's row says which part of hex closes it.
        assert gap.owner and not any(gap.owner in line or "owner" in line for line in lines)
    assert 'exact' in lines[-1]
    assert resolve_namelist(deck('convection_permitting')).gap_warning() is None


def test_RP4_tke_advection_is_refused_in_correct_mode_only():
    exact = resolve_namelist(deck('convection_permitting', config_mynn_tkeadvect=True))
    assert exact.options['config_mynn_tkeadvect'] is True       # matched MPAS accepts and resets TKE
    with pytest.raises(ValueError, match='RP4'):
        resolve_namelist(deck('convection_permitting', mode='correct', config_mynn_tkeadvect=True))
    assert resolve_namelist(deck('convection_permitting', mode='correct', skip='RP4',
                                 config_mynn_tkeadvect=True)).mode == 'correct'


def declaring(**slots):
    """A Declarations callable: slot -> finding ids its implementation applies, whatever the scheme."""
    return lambda slot, scheme, backend: slots.get(slot, ())


def test_RP1_RP3_pristine_native_slots_are_named_gaps_not_refusals():
    config = resolve_namelist(deck('convection_permitting', mode='correct', skip='WB3'))
    # Native-cpu MYNN, sea ice and surface layer are pristine v8.4.2: RP1 and RP3 cannot be applied, so
    # they are left out of the run whole and listed (correct is the default, and a gap is never a refusal).
    assert config.corrections == frozenset()
    owners = {finding: row['owners'] for finding, row in CORRECTIONS_V842.items()}
    assert {(g.finding, g.slot): g.owner for g in config.gaps} == {
        ('RP1', 'pbl'): owners['RP1']['pbl'], ('RP7', 'pbl'): owners['RP7']['pbl'],
        ('RP3', 'seaice'): owners['RP3']['seaice'], ('RP3', 'surface_layer'): owners['RP3']['surface_layer']['sf_mynn']}
    assert all(g.reason == 'slot does not implement it' and g.backend == 'native-cpu' for g in config.gaps)
    driver, _, seen = driver_for(config, 360)           # runs: no refusal
    run_steps(driver, 360, 1)
    assert seen and all(call.corrections == frozenset() for call in seen)
    # The published hex-cuda MYNN declares RP1 in its table descriptor: the table
    # settlement applies it, so selecting the hex-cuda slot closes the gap.
    cuda_mynn = resolve_namelist(deck('convection_permitting', mode='correct', skip='RP3,WB3,RP7'),
                                 backends={'pbl': 'hex-cuda'})
    assert cuda_mynn.corrections == {'RP1'} and cuda_mynn.gaps == ()
    # Implementations that declare the corrections apply them.
    backends = {'pbl': 'hex-cuda', 'seaice': 'hex-cuda', 'surface_layer': 'hex-cuda'}
    cuda = resolve_namelist(deck('convection_permitting', mode='correct', skip='WB3,RP7'), backends=backends,
                            declarations=declaring(pbl=('RP1',), seaice=('RP3',), surface_layer=('RP3',)))
    assert cuda.corrections == {'RP1', 'RP3'} and cuda.gaps == ()
    driver_for(cuda, 360, declare={'pbl': ('RP1',), 'seaice': ('RP3',), 'surface_layer': ('RP3',)})
    # A finding is applied whole or not at all: one slot without RP3 makes all of RP3 a gap.
    partial = resolve_namelist(deck('convection_permitting', mode='correct', skip='WB3,RP7'), backends=backends,
                               declarations=declaring(pbl=('RP1',), seaice=('RP3',)))
    assert partial.corrections == {'RP1'}
    assert [(g.finding, g.slot, g.backend) for g in partial.gaps] == [('RP3', 'surface_layer', 'hex-cuda')]
    driver, _, seen = driver_for(partial, 360, declare={'pbl': ('RP1',), 'seaice': ('RP3',)})
    run_steps(driver, 360, 1)
    assert all(call.corrections == {'RP1'} for call in seen)
    # Registered implementations that contradict the declarations the run was settled with are refused:
    # the receipt would name RP3 applied while the surface layer runs it uncorrected.
    with pytest.raises(RuntimeError, match='do not declare'):
        driver_for(cuda, 360, declare={'pbl': ('RP1',), 'seaice': ('RP3',)})
    # Skipping the findings is an explicit, receipted choice: no correction and no gap.
    skipped = resolve_namelist(deck('convection_permitting', mode='correct', skip='RP1,RP3,RP7,WB3'))
    assert skipped.corrections == frozenset() and skipped.gaps == ()
    driver_for(skipped, 360)


def test_registry_declarations_settle_the_same_way_as_the_table():
    config = resolve_namelist(deck('convection_permitting', mode='correct'))
    driver, _, _ = driver_for(config, 360)
    again = resolve_namelist(deck('convection_permitting', mode='correct'),
                             declarations=driver.registry.declared)
    assert again == config


def test_slot_calls_carry_the_active_corrections():
    config = resolve_namelist(deck(mode='correct'), backends={'seaice': 'hex-cuda', 'surface_layer': 'hex-cuda'},
                              declarations=declaring(seaice=('RP3',), surface_layer=('RP3',)))
    driver, _, seen = driver_for(config, 360, declare={'seaice': ('RP3',), 'surface_layer': ('RP3',)})
    run_steps(driver, 360, 1)
    assert all(call.corrections == {'RP3'} for call in seen)
    exact = resolve_namelist(deck())
    driver, _, seen = driver_for(exact, 360)
    run_steps(driver, 360, 1)
    assert all(call.corrections == frozenset() for call in seen)


#: The dt_pbl schemes (the surface layer has no timestep; it is counted separately).
SURFACE_GROUP = ('land_surface', 'seaice', 'pbl', 'gwdo')


def integrated_seconds(mode, interval, dt, steps):
    """Time each surface/PBL scheme integrates over the run: calls x the step each call is given."""
    config = resolve_namelist(deck(mode=mode, skip='RP3' if mode == 'correct' else None,
                                   config_pbl_interval=interval))
    driver, _, seen = driver_for(config, dt)
    run_steps(driver, dt, steps)
    totals = {'surface_layer_calls': 0}
    for call in seen:
        if call.slot in SURFACE_GROUP:
            totals[call.slot] = totals.get(call.slot, Fraction(0)) + Fraction(call.dt)
        elif call.slot == 'surface_layer':
            totals['surface_layer_calls'] += 1
    return totals


def test_RP6_pbl_interval_integrates_elapsed_time_only_in_correct_mode():
    dt, steps = 360, 12                       # 72 min of model time, pbl interval twice the step
    elapsed = dt * steps
    exact = integrated_seconds('exact', '00:12:00', dt, steps)
    correct = integrated_seconds('correct', '00:12:00', dt, steps)
    # Matched MPAS: dt_pbl = 720 s on every 360 s step, so the dt_pbl schemes integrate twice the forecast.
    assert exact['pbl'] == exact['seaice'] == exact['gwdo'] == 2 * elapsed
    assert exact['surface_layer_calls'] == steps
    # Correct mode: called every second step with dt_pbl, so every scheme integrates the elapsed time.
    calls = correct.pop('surface_layer_calls')
    assert set(correct) == set(SURFACE_GROUP)
    assert all(total == elapsed for total in correct.values()), correct
    assert calls == steps // 2


def test_RP6_correct_mode_with_interval_equal_to_dt_calls_exactly_as_before():
    dt = 360
    sequences = []
    for mode, interval in (('exact', 'none'), ('correct', '00:06:00')):
        config = resolve_namelist(deck(mode=mode, skip='RP3' if mode == 'correct' else None,
                                       config_pbl_interval=interval))
        driver, hooks, _ = driver_for(config, dt)
        run_steps(driver, dt, 5)
        sequences.append(hooks.operations)
    # The precipitation window brackets each call; a one-step window hands over the step's own words
    # (test_RP6_precipitation_window_words and the native harness), so only its glue is added.
    assert sequences[1].count('accumulate_surface_precipitation') == 5
    assert [op for op in sequences[1] if op not in SURFACE_PRECIPITATION_WINDOW] == sequences[0]


def window_steps(config, dt, steps):
    driver, hooks, _ = driver_for(config, dt)
    per_step = []
    for n in range(steps):
        hooks.operations.clear()
        driver.begin_step(itimestep=n + 1, time_seconds=n * dt, pools=None)
        per_step.append(list(hooks.operations))
        driver.physics_get_tend(rk_step=1, dynamics_substep=1, pools=None)
        driver.finish_step(pools=None)
    return per_step


def test_RP6_every_step_precipitation_reaches_the_land_surface_and_sea_ice_calls():
    """Fails on the first RP6 fix (the cadence without the window): the land-surface and sea-ice schemes read only
    the call step's rainncv/raincv, so the precipitation of every step between calls never reached them (native
    x1.2562 6 h, 40 min interval: 48 % of the snow arrived)."""
    dt, steps = 360, 12
    config = resolve_namelist(deck(mode='correct', skip='RP3', config_pbl_interval='00:12:00'))
    per_step = window_steps(config, dt, steps)
    for n, ops in enumerate(per_step):
        # every step adds its precipitation, before any surface scheme runs
        assert ops.count('accumulate_surface_precipitation') == 1
        assert all(ops.index('accumulate_surface_precipitation') < ops.index(s)
                   for s in ('surface_layer', 'land_surface') if s in ops)
        if n % 2 == 0:                                   # call steps: the window brackets land surface and sea ice
            order = [ops.index(s) for s in ('open_surface_precipitation_window', 'land_surface', 'seaice',
                                            'close_surface_precipitation_window', 'atmphys_sfc_diagnostics',
                                            'pbl')]
            assert order == sorted(order), ops
        else:                                            # between calls nothing reads or empties the window
            assert 'land_surface' not in ops and 'open_surface_precipitation_window' not in ops \
                and 'close_surface_precipitation_window' not in ops


def test_RP6_window_is_absent_in_exact_mode_and_without_an_interval():
    for text in (deck(config_pbl_interval='00:12:00'), deck(mode='correct', skip='RP3'),
                 deck(mode='correct', skip='RP3,RP6', config_pbl_interval='00:12:00')):
        config = resolve_namelist(text)
        ops = [op for step in window_steps(config, 360, 4) for op in step]
        assert not set(ops) & set(SURFACE_PRECIPITATION_WINDOW), text


def test_RP6_window_glue_is_required_before_any_native_mutation():
    class Native:
        abi = {'entries': {}}
    hooks = NativeDriverHooks(Native(), None, after_call=None, external={}, save_fields=None, restore_fields=None)
    correct = resolve_namelist(deck(mode='correct', skip='RP3', config_pbl_interval='00:12:00'))
    assert set(SURFACE_PRECIPITATION_WINDOW) <= set(hooks.missing_operations(correct))
    exact = resolve_namelist(deck(config_pbl_interval='00:12:00'))
    assert not set(SURFACE_PRECIPITATION_WINDOW) & set(hooks.missing_operations(exact))


def test_RP6_noahmp_is_a_named_gap_until_it_steps_by_dt_pbl():
    noahmp = deck('none', mode='correct', config_lsm_scheme='sf_noahmp',
                  config_sfclayer_scheme='sf_monin_obukhov_rev', config_pbl_scheme='bl_ysu',
                  config_pbl_interval='00:12:00', config_frac_seaice=False)
    config = resolve_namelist(noahmp)
    # Native Noah-MP steps by the model dt (noahmpinit.F:289), so the cadence cannot be applied to it: RP6 is
    # left out whole (the run keeps native's every-step calls) and named, closed by the Noah-MP port.
    assert config.corrections == frozenset()
    assert [(g.finding, g.slot, g.scheme, g.owner) for g in config.gaps] == [
        ('RP6', 'land_surface', 'sf_noahmp', CORRECTIONS_V842['RP6']['owners']['land_surface'])]
    exact = resolve_namelist(noahmp.replace("config_physics_mode = 'correct'", "config_physics_mode = 'exact'"))
    sequences = []
    for each in (config, exact):
        driver, hooks, _ = driver_for(each, 360)
        run_steps(driver, 360, 4)
        sequences.append(hooks.operations)
    assert sequences[0] == sequences[1] and sequences[0].count('land_surface') == 4
    noah = resolve_namelist(noahmp.replace('sf_noahmp', 'sf_noah'))
    assert noah.corrections == {'RP6'} and noah.gaps == ()
    driver_for(noah, 360)                     # native Noah already steps by dt_pbl (driver_lsm.F:726)
    declared = resolve_namelist(noahmp, declarations=declaring(land_surface=('RP6',)))
    assert declared.corrections == {'RP6'} and declared.gaps == ()
    driver_for(declared, 360, declare={'land_surface': ('RP6',)})


def test_RP6_native_file_restart_is_a_named_gap_not_a_refusal():
    restart = deck(mode='correct', skip='RP3,WB3,WB5', config_pbl_interval='00:12:00') \
        + '\n&restart\nconfig_do_restart = .true.\n/\n'
    config = resolve_namelist(restart)
    assert config.corrections == frozenset()
    assert [(g.finding, g.slot) for g in config.gaps] == [('RP6', 'restart')]
    assert 'precipitation window' in config.gaps[0].reason
    assert config.gaps[0].owner == RP6_RESTART_GAP['owner']
    cold = resolve_namelist(restart.replace('.true.', '.false.'))
    assert cold.corrections == {'RP6'} and cold.gaps == ()


def test_restart_keeps_the_pbl_cadence_phase():
    config = resolve_namelist(deck(mode='correct', skip='RP3', config_pbl_interval='00:12:00'))
    a = PhysicsManager(config, 360)
    due = [(a.advance(t), a.pbl_due)[1] for t in (0, 360, 720)]
    assert due == [True, False, True]
    b = PhysicsManager(config, 360)
    b.restore_restart_state(a.restart_state())
    a.advance(1080); b.advance(1080)
    assert a.pbl_due is b.pbl_due is False


def test_door_takes_the_mode_keys_out_before_pristine_registry_readers():
    plain = deck()
    assert split_physics_mode(plain) == (plain, None, ())
    text, mode, skip = split_physics_mode(deck(mode='correct', skip='RP3, RP6'))
    assert mode == 'correct' and skip == ('RP3', 'RP6')
    assert 'config_physics_mode' not in text and 'config_physics_correct_skip' not in text
    assert resolve_namelist(text, physics_mode=mode, correct_skip=skip).correct_skip == {'RP3', 'RP6'}


def test_every_correction_names_its_breakage():
    # The MPAS suite route's findings (an MPAS slot implements each, or the resolver refuses it: RP4), and
    # the findings that act on the ArWen route and the IAU and boundary doors ("hex_route", read by
    # woof.hex.correct_mode).  WB5 is on both: the hex-cuda WSM6 slot and the engine's WSM6.
    assert {k for k, row in CORRECTIONS_V842.items() if row['slots'] or k == 'RP4'} == {
        'RP1', 'RP3', 'RP4', 'RP6', 'RP7', 'RP8', 'WB3', 'WB5'}
    assert {k for k, row in CORRECTIONS_V842.items() if 'hex_route' in row} == {
        'WB4', 'WB5', 'IN1', 'IN2', 'IN3', 'IN4', 'IN5'}
    assert set(CORRECTIONS_V842) == {'RP1', 'RP3', 'RP4', 'RP6', 'RP7', 'RP8', 'WB3', 'WB4', 'WB5',
                                     'IN1', 'IN2', 'IN3', 'IN4', 'IN5'}
    assert all(row['breakage'] for row in CORRECTIONS_V842.values())
    assert Backend.NATIVE_CPU.value == 'native-cpu'


def test_every_slot_a_finding_lives_in_names_a_follow_up_owner():
    from woof.hex.physics_suites import load_table
    slots = load_table()['mpasphys_v842']['slots']
    for finding, row in CORRECTIONS_V842.items():
        assert set(row['owners']) == set(row['slots']), finding
        for slot, schemes in row['slots'].items():
            owner = row['owners'][slot]
            for scheme in (schemes or [s for s in slots[slot]['schemes'] if s != 'off']):
                assert isinstance(owner, str) or owner.get(scheme), (finding, slot, scheme)
