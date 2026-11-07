"""A start and the first boundary share a valid time, not a source or level count."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from woof.hex import regional_door as door
from woof.hex.cli import build_parser


def request(tmp_path, *flags):
    args = build_parser().parse_args([
        'regional', '--point=37,-96.5', '--radius-km', '300', '--fine-dx-m', '3000',
        '--background-km', '96', '--cycle', '2026-10-14T03', '--hours', '2',
        '--work-dir', str(tmp_path), *flags])
    return args, door.plan(args)


def test_start_horizon_does_not_limit_gfs_boundaries(tmp_path):
    _, v = request(tmp_path, '--source', 'rrfs', '--boundary-cycle', '2026-10-14T00', '--hours', '24')
    assert v['boundary_source']['id'] == 'gfs'
    assert v['boundary_lead_hours'] == 3
    assert v['start_time'] == '2026-10-14_03:00:00'
    start, boundary = door._met_series(v)
    assert start['hours'] == 0
    assert boundary['cycle'] == '2026-10-14T00'
    assert boundary['lead_hours'] == 3 and boundary['hours'] == 24
    assert start['paths']['met'] != boundary['paths']['met']
    with pytest.raises(door.RegionalRefusal, match='rrfs boundary cycle.*run.s end'):
        request(tmp_path, '--source', 'rrfs', '--boundary-source', 'rrfs', '--hours', '24')


def test_old_boundary_cycle_is_refused_by_source_and_cycle(tmp_path):
    with pytest.raises(door.RegionalRefusal, match='gfs boundary cycle 2026-09-27T00.*run.s end'):
        request(tmp_path, '--source', 'rrfs', '--boundary-cycle', '2026-09-27T00')


def test_coverage_refuses_in_plan_before_any_work(tmp_path):
    with pytest.raises(door.RegionalRefusal, match='reaches past RRFS.*map'):
        request(tmp_path, '--source', 'rrfs', '--point=0,20', '--boundary-cycle', '2026-10-14T00')
    assert not list(tmp_path.iterdir())


def test_independent_receipt_levels_reach_separate_producer_calls(tmp_path, monkeypatch):
    args, v = request(tmp_path, '--source', 'rrfs', '--cycle', '2026-10-14T00', '--forecast-arg=--physics-suite=hrrrv5')
    for directory, levels, soil in [('met', 46, 9), ('lbc-met', 58, 4)]:
        root = tmp_path / directory
        root.mkdir()
        (root / 'intermediate-receipt.json').write_text(json.dumps({'levels': {
            'count': levels, 'init_switches': {'--nfgsoillevels': soil}}}))
    assert door._first_guess(v)['nfglevels'] == 46
    assert door._first_guess(v)['nfgsoillevels'] == 9
    assert door._first_guess(v, boundary=True)['nfglevels'] == 58
    assert door._first_guess(v, boundary=True)['nfgsoillevels'] == 4
    # The producer is deliberately invoked twice, with a per-call level count.
    from woof.hex.mpasphys.rrfs.cold_start_rrfs import case7_jobs, case9_jobs, producer_for_suite
    _, row = producer_for_suite('hrrrv5')
    init = case7_jobs(row, static=Path('s'), met=Path('m'), start_time=v['start_time'],
                     nfglevels=46, nfgsoillevels=9, aerosols=Path('a'), artifact=Path('c'),
                     out=Path('i'), levels_file=Path('l'), regional=True)
    lbc = case9_jobs(row, init=Path('i'), intervals=[(v['start_time'], Path('b'))], start_time=v['start_time'],
                    nfglevels=58, aerosols=Path('a'), out_dir=Path('o'), carrier=Path('c'))
    assert init[-1]['config']['n_fg_levels'] == 46
    assert lbc[-1]['config']['n_fg_levels'] == 58


@pytest.mark.parametrize('source', ['hrrr', 'gem-hrdps', 'gem-rdps'])
def test_same_source_uses_the_original_single_series_argv_and_bytes(tmp_path, monkeypatch, source):
    args, value = request(tmp_path, '--source', source, '--cycle', '2026-10-14T00')
    assert door._met_series(value) == [value]
    assert door._met_series(value)[0] is value
    assert value['paths']['lbc_met'] == value['paths']['met']
    monkeypatch.setattr(door, '_generated', lambda v: {'_path': 'original-plan.json'})
    calls = []
    payload = b'WPS bytes from the unchanged intermediate worker\x00\xff'
    def cli(argv, status):
        calls.append(argv)
        out = Path(argv[argv.index('--out-dir')+1]); out.mkdir(parents=True)
        (out / 'MET:2026-10-14_00').write_bytes(payload)
        return 0
    monkeypatch.setattr(door, '_cli', cli)
    result = door._stage_intermediate(args, value, SimpleNamespace(note=lambda s: None))
    assert result['returncode'] == 0
    # Literal pre-split command shape, including ordering. No source-specific rewrite is applied.
    assert calls == [['intermediate', '--source', source, '--from-plan', 'original-plan.json',
                      '--grib-dir', value['paths']['grib'], '--cycle', '2026-10-14T00', '--hours', '0-2',
                      '--out-dir', value['paths']['met'], '--workers', str(args.workers)]]
    assert (Path(value['paths']['met']) / 'MET:2026-10-14_00').read_bytes() == payload


def test_named_blend_is_row_driven_and_does_not_replace_defaults():
    from woof.hex.init_sources import load_table, resolve_source
    from woof.hex.blend_start import blend_rows, inscribed_box
    t = load_table(); row = resolve_source(t, 'rrfs-gfs')
    outside, inside = blend_rows(t, row)
    assert (inside['id'], outside['id']) == ('rrfs', 'gfs')
    box = inscribed_box(inside['intermediate']['row'], spacing_deg=.2)
    assert box['south'] < 37 < box['north'] and box['west'] < -96.5 < box['east']
    assert build_parser().parse_args(['mesh-plan', '--point=37,-96.5']).source is None


def test_unpublished_gfs_boundary_cycle_is_refused_before_work(tmp_path):
    with pytest.raises(door.RegionalRefusal, match='gfs boundary cycle 2026-10-14T03 is not published'):
        request(tmp_path, '--source', 'rrfs')


def test_country_producer_splits_start_and_older_boundary_cycle(tmp_path):
    from woof.hex.country_door import plan
    args = build_parser().parse_args(['country', '--point=37,-96.5', '--radius-km', '300',
        '--fine-dx-m', '3000', '--background-km', '96', '--source', 'rrfs', '--cycle', '2026-10-14T03',
        '--boundary-source', 'gfs', '--boundary-cycle', '2026-10-14T00', '--hours', '24',
        '--forecast-arg=--physics-suite=hrrrv5', '--work-dir', str(tmp_path)])
    v = plan(args)
    assert v['boundary_cycle'] == '2026-10-14T00' and v['boundary_lead_hours'] == 3
    assert v['commands']['start_intermediate'][v['commands']['start_intermediate'].index('--hours')+1] == '0'
    assert '--physics-suite' in v['commands']['start_intermediate']
    assert 'start-met' in v['commands']['producer_init'][2]
    assert v['boundary_endpoints'][0]['valid_time'] == '2026-10-14_03:00:00'
    assert v['boundary_endpoints'][-1]['valid_time'] == '2026-10-15_03:00:00'


def test_producer_cut_reuses_template_without_prepare(tmp_path, monkeypatch):
    from woof.hex.mpasphys.rrfs import cold_start_rrfs as producer
    monkeypatch.setattr(producer, 'static_missing_fields', lambda *a: [])
    monkeypatch.setattr(producer, 'resolve_worker', lambda *a: Path('worker'))
    monkeypatch.setattr(producer, 'resolve_aerosol_table', lambda *a: Path('aerosol'))
    monkeypatch.setattr(producer, '_refuse_cold_land_soil', lambda *a: None)
    captured = []
    monkeypatch.setattr(producer, 'run_jobs', lambda worker, jobs, logs: captured.extend(jobs) or [])
    template = tmp_path / 'whole-cut.nc'
    producer.init_on_template(suite='hrrrv5', template=template, met=Path('met'), out=tmp_path / 'init.nc',
        start_time='2026-10-14_00:00:00', nfglevels=46, nfgsoillevels=9)
    assert [job['stage'] for job in captured] == ['case7']
    config = captured[0]['config']
    assert {config[key] for key in ('static_path', 'capsule_path', 'reference_path')} == {str(template)}
    assert config['n_fg_levels'] == 46 and config['n_fg_soil_levels'] == 9
