"""Routing and non-growing recorded inventory for regenerated headline decks."""
import argparse
import copy
import json
from pathlib import Path

import pytest
from woof.hex.verify_mpas import VerificationRefusal, build_headline, digest


NAMELIST = "&physics\n    config_physics_suite = 'convection_permitting'\n/\n"
#: The ten slots convection_permitting runs (hl-3's landed list); ngw and oml are off in it.
ACTIVE = ('config_sfclayer_backend', 'config_lsm_backend', 'config_convection_backend', 'config_radt_cld_backend',
          'config_microp_backend', 'config_seaice_backend', 'config_pbl_backend', 'config_gwdo_backend',
          'config_radt_lw_backend', 'config_radt_sw_backend')


@pytest.fixture
def case(tmp_path):
    def ref(name):
        path = tmp_path / name
        path.write_text(NAMELIST if name == 'namelist' else name)
        return {'path': name, 'sha256': digest(path)}
    parent = dict(schema='mpas-verification-deck-v1', id='parent', pin_set='pin',
                  mesh='mesh', date='date', physics='on', suite='suite',
                  process_count=1, ranks=1, duration_steps=3,
                  inputs={k: ref(k) for k in ('namelist', 'streams', 'init')},
                  native=ref('publication'),
                  hex_command=['woof hex', 'forecast', '--native-physics-library', '{resource:native-library}'])
    path = tmp_path / 'parent.json'
    path.write_text(json.dumps(parent))
    index = dict(schema='mpas-gate-index-v1', gates={'G1': [{'path': str(path), 'sha256': digest(path)}],
                 'G2': [{'path': 'parent.json', 'sha256': digest(path)}]}, landed_decks=['parent'])
    entry = dict(pin_set='pin', mesh='mesh', date='date', deck='parent', stream='s',
                 field='deferred', process_count=1, differing_words=7, max_ulp=0,
                 missing_samples=1, first_differing_step=0,
                 compared_words=7, statuses=['field missing in hex'])
    baseline = dict(schema='mpas-ratchet-v1', entries=[entry])
    for name, value in [('index.json', index), ('baseline.json', baseline)]:
        (tmp_path / name).write_text(json.dumps(value))
    slots = tmp_path / 'slots.txt'
    slots.write_text("# integrator list\nconfig_sfclayer_backend=hex-cuda # landed\n")
    args = argparse.Namespace(index=tmp_path/'index.json', ratchet=tmp_path/'baseline.json',
        slots=slots, parent_deck='parent', headline_id='headline', out=tmp_path/'generated')
    return args, parent, index, baseline


def read(path):
    return json.loads(path.read_text())


def test_land_then_regenerate_preserves_retired_allowance(case):
    args, parent, index, baseline = case
    assert build_headline(args) == 0
    generated = read(args.out / 'headline-deck.json')
    assert generated['headline_backends']['surface_layer'] == 'hex-cuda'
    assert set(generated['headline_backends'].values()) == {'native-cpu', 'hex-cuda'}
    # The landed slot plus the two slots convection_permitting leaves off (they run nothing).
    assert {k for k, v in generated['headline_backends'].items() if v == 'hex-cuda'} == {
        'surface_layer', 'ngw', 'oml'}
    # Active slots still on the CPU: the native library stays.
    assert generated['hex_command'][2:4] == ['--native-physics-library', '{resource:native-library}']
    assert generated['native']['sha256'] == parent['native']['sha256']
    assert Path(generated['native']['path']).is_absolute()
    first_index = read(args.out / 'mpas-gate-index.json')
    assert first_index['gates']['G1'] == index['gates']['G1']
    assert first_index['gates']['G2'][0]['sha256'] == index['gates']['G2'][0]['sha256']
    assert Path(first_index['gates']['G2'][0]['path']) == args.index.parent/'parent.json'
    assert first_index['landed_decks'] == ['parent', 'headline']
    first = read(args.out / 'ratchet.json')
    assert first['entries'][:1] == baseline['entries']
    # An adopted written field retires its missing allowance. A later landing
    # must preserve zero, rather than copying the old parent's missing words.
    first['entries'][1].update(differing_words=0, missing_samples=0, statuses=[], first_differing_step=None)
    (args.out / 'ratchet.json').write_text(json.dumps(first))
    args.slots.write_text('config_sfclayer_backend=hex-cuda\nconfig_lsm_backend=hex-cuda\n')
    args.index, args.ratchet = args.out/'mpas-gate-index.json', args.out/'ratchet.json'
    assert build_headline(args) == 0
    second = read(args.out / 'ratchet.json')
    assert second['entries'] == first['entries']
    assert read(args.out/'headline-deck.json')['headline_backends']['land_surface'] == 'hex-cuda'
    assert len(read(args.out/'mpas-gate-index.json')['gates']['G2']) == 2


def test_an_all_gpu_headline_names_no_native_library(case):
    args, *_ = case
    args.slots.write_text(''.join(f'{name}=hex-cuda\n' for name in ACTIVE))
    assert build_headline(args) == 0
    generated = read(args.out / 'headline-deck.json')
    assert set(generated['headline_backends'].values()) == {'hex-cuda'}
    assert '--native-physics-library' not in generated['hex_command']
    overlay = generated['hex_command'][generated['hex_command'].index('--physics') + 1]
    assert "config_ngw_backend='hex-cuda'" in overlay and "config_oml_backend='hex-cuda'" in overlay


@pytest.mark.parametrize('text', ['typo=hex-cuda', 'config_lsm_backend=off',
    'config_lsm_backend=native-cpu', 'config_lsm_backend',
    'config_lsm_backend=hex-cuda\nconfig_lsm_backend=hex-cuda'])
def test_invalid_slot_cannot_silently_leave_cpu(case, text):
    args, *_ = case
    args.slots.write_text(text)
    with pytest.raises(VerificationRefusal, match='landed-slot'):
        build_headline(args)
    assert not args.out.exists()


def test_changed_parent_input_is_not_relabelled(case):
    args, *_ = case
    (args.index.parent/'init').write_text('changed')
    with pytest.raises(VerificationRefusal, match='changed pinned input'):
        build_headline(args)


def test_same_id_foreign_context_rejected(case):
    args, _, _, baseline = case
    foreign = copy.deepcopy(baseline['entries'][0]);foreign.update(deck='headline', process_count=4)
    baseline['entries'].append(foreign)
    args.ratchet.write_text(json.dumps(baseline))
    with pytest.raises(VerificationRefusal, match='different native context'):
        build_headline(args)


def test_resource_packet_and_dependencies_survive_slot_regeneration(case):
    args, *_ = case
    decoder = args.index.parent/'decoder'
    decoder.write_bytes(b'certified decoder')
    directory = args.index.parent/'tables';directory.mkdir()
    table = directory/'table';table.write_bytes(b'native table')
    packet = args.index.parent/'resources.json'
    packet.write_text(json.dumps({'scheme': {'table_directory': str(directory),
        'decoder': {'path': str(decoder), 'sha256': digest(decoder)}}}))
    args.physics_resources = packet
    build_headline(args)
    first = read(args.out/'headline-deck.json')
    assert '--physics-resources' in first['hex_command']
    refs = first['hex_support_inputs'].values()
    assert {str(decoder), str(table)} <= {ref['path'] for ref in refs}
    args.index, args.ratchet = args.out/'mpas-gate-index.json', args.out/'ratchet.json'
    args.physics_resources = None
    build_headline(args)
    assert read(args.out/'headline-deck.json')['hex_support_inputs'] == first['hex_support_inputs']
    decoder.write_bytes(b'changed executable')
    with pytest.raises(VerificationRefusal, match='changed pinned input'):
        build_headline(args)


def _repin_parent(args, parent, hex_command):
    """Rewrite the indexed parent deck with another hex_command and re-pin it in the index."""
    parent = dict(parent, hex_command=hex_command)
    path = args.index.parent/'parent.json'
    path.write_text(json.dumps(parent))
    index = read(args.index)
    for refs in index['gates'].values():
        for ref in refs:
            ref['sha256'] = digest(path)
    args.index.write_text(json.dumps(index))


def test_a_parent_that_names_physics_mode_exact_is_accepted_and_carried(case):
    """Correct mode (2026-10-14): every proof names exact; the headline keeps it, once."""
    args, parent, _, _ = case
    _repin_parent(args, parent, ['woof hex', 'forecast', '--physics-mode', 'exact'])
    assert build_headline(args) == 0
    command = read(args.out/'headline-deck.json')['hex_command']
    assert command.count('--physics-mode') == 1 and command[command.index('--physics-mode') + 1] == 'exact'
    assert '--physics' in command


@pytest.mark.parametrize('bad', [['--physics-mode', 'exact', '--physics-mode', 'correct'],
                                 ['--physics-mode', 'precise']])
def test_a_parent_with_a_malformed_physics_mode_is_refused(case, bad):
    args, parent, _, _ = case
    _repin_parent(args, parent, ['woof hex', 'forecast', *bad])
    with pytest.raises(VerificationRefusal, match='physics-mode'):
        build_headline(args)
