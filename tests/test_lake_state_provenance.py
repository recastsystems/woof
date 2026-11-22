"""Defect 5 (2026-11-12 runner list): an auto lake request must not fall back quietly.

A daily lake analysis is posted the day after it is valid, so the newest
day answers HTTP 404 for any start near the wall clock (measured on GLSEA:
2026-11-11T12Z gave 404 "greater than the axis maximum" while 2026-11-10T12Z
was served).  The run asked one day, fell back to the forcing, and said so
in one stderr line that no plan, dry run or run record carried.  These run
on the CPU: the network and the Rust sampler are stood in for.
"""
from datetime import datetime, timezone
import io
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

import woof.ingest.lake_state as obs


START = datetime(2026, 11, 12, 0, tzinfo=timezone.utc)
BOX = dict(south=42., north=43., west=-81., east=-80.)


class _Served:
    def __init__(self, body):
        self._body = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self._body.read(n)


def _server(posted, asked, *, down=False):
    """A stand-in ERDDAP that serves only the analysis days in ``posted``."""
    def urlopen(url, timeout=None):
        asked.append(url)
        if down:
            raise URLError('network is unreachable')
        if any(day in url for day in posted):
            return _Served(b'CDF\x01' + b'\x00' * 64)
        raise HTTPError(url, 404, 'Not Found', {}, None)
    return urlopen


def _water(row):
    return next(f for f in row['field'] if f.get('dataset') == 'GLSEA_ACSPO_GCS')


def test_the_newest_day_missing_takes_the_day_before_inside_the_row_age(monkeypatch, tmp_path):
    row = obs.source_rows()['glsea']
    asked = []
    monkeypatch.setattr(obs, 'urlopen', _server(('2026-11-10T12',), asked))
    path, url = obs.acquire_field(row, _water(row), START, BOX,
                                  tmp_path)
    assert path.is_file() and '2026-11-10T12:00:00Z' in url
    assert ['2026-11-11T12:00:00Z' in asked[0], '2026-11-10T12:00:00Z' in asked[1]] == [True, True]


def test_days_asked_stop_at_the_row_age_limit():
    row = obs.source_rows()['glsea']   # analysis_hour 12, max_age_hours 48
    assert [s.isoformat()[:13] for s in obs.analysis_stamps(row, START)] == ['2026-11-11T12', '2026-11-10T12']
    noon = datetime(2026, 11, 12, 12, tzinfo=timezone.utc)
    assert [s.isoformat()[:13] for s in obs.analysis_stamps(row, noon)] == [
        '2026-11-12T12', '2026-11-11T12', '2026-11-10T12']


def test_no_served_day_names_every_day_and_its_answer(monkeypatch, tmp_path):
    row = obs.source_rows()['glsea']
    asked, attempts = [], []
    monkeypatch.setattr(obs, 'urlopen', _server((), asked))
    with pytest.raises(obs.LakeAnalysisUnavailable,
                       match=r'GLSEA_ACSPO_GCS served no analysis within 48 h before 2026-11-12T00:00Z '
                             r'\(2026-11-11T12:00:00Z HTTP 404, 2026-11-10T12:00:00Z HTTP 404\)'):
        obs.acquire_field(row, _water(row), START, BOX, tmp_path, attempts)
    assert len(asked) == 2


def test_an_unreachable_server_is_not_asked_for_earlier_days(monkeypatch, tmp_path):
    row = obs.source_rows()['glsea']
    asked = []
    monkeypatch.setattr(obs, 'urlopen', _server((), asked, down=True))
    with pytest.raises(obs.LakeAnalysisUnavailable, match='URLError'):
        obs.acquire_field(row, _water(row), START, BOX, tmp_path)
    assert len(asked) == 1


def _stub_rust(monkeypatch):
    monkeypatch.setattr(obs, 'coordinate_coverage', lambda lat, lon, row: dict(covered=float(np.size(lat))))
    monkeypatch.setattr(obs, 'decode_field', lambda path, row, field, valid_time, lat, lon: (
        np.full(np.shape(lat), 280.), dict(field=field['name'], valid_time='2026-11-10T12:00:00+00:00',
                                           age_hours=36., covered_targets=int(np.size(lat)))))


def test_auto_binds_the_day_before_and_its_line_says_which_day_and_why(monkeypatch, tmp_path):
    _stub_rust(monkeypatch)
    monkeypatch.setattr(obs, 'urlopen', _server(('2026-11-10T12',), []))
    values, receipt = obs.load_state('auto', START, np.array([42.2]), np.array([-80.6]), cache=tmp_path)
    assert receipt['status'] == 'bound' and receipt['requested'] == 'auto' and receipt['source'] == 'glsea'
    assert set(values) == {'water_temperature', 'ice_fraction'}
    attempts = receipt['fetch_attempts']['water_temperature']
    assert [(a['time'], a['outcome']) for a in attempts] == [
        ('2026-11-11T12:00:00Z', 'HTTP 404'), ('2026-11-10T12:00:00Z', 'fetched')]
    line = obs.describe(receipt)
    assert line.startswith('lake state: glsea water_temperature, ice_fraction from the analysis of '
                           '2026-11-10T12:00Z (36 h before the start)'), line
    assert '(the newer 2026-11-11T12:00Z: HTTP 404)' in line


def test_auto_with_nothing_served_says_forcing_and_why_in_one_line(monkeypatch, tmp_path, capsys):
    _stub_rust(monkeypatch)
    monkeypatch.setattr(obs, 'urlopen', _server((), []))
    monkeypatch.setenv('WOOF_LAKE_RECEIPT_DIR', str(tmp_path / 'receipts'))
    values, receipt = obs.load_state('auto', START, np.array([42.2]), np.array([-80.6]), cache=tmp_path)
    assert values == {} and receipt['status'] == 'forcing_fallback'
    receipt = obs.publish_receipt(receipt)
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith('WARNING lake state: forcing, because glsea gave nothing usable: '
                               'water_temperature: GLSEA_ACSPO_GCS served no analysis within 48 h'), lines[0]
    assert '2026-11-10T12:00:00Z HTTP 404' in lines[0] and receipt['receipt_path'] in lines[0]
    assert receipt['fetch_attempts']['ice_fraction'][0]['outcome'] == 'HTTP 404'


def test_a_forcing_run_says_so_without_touching_the_card(capsys):
    _, receipt = obs.initialize_surface({}, SimpleNamespace(lake_state_source='forcing'),
                                        valid_time=START, latitude=None, longitude=None)
    assert receipt['status'] == 'retained_forcing' and receipt['source'] == 'forcing'
    assert capsys.readouterr().err.strip() == 'lake state: forcing, because lake_state_source = "forcing"'


def test_the_plan_names_the_lake_source_before_anything_is_spent():
    auto = SimpleNamespace(lake_state_source='auto')
    great_lakes = obs.plan_sentence(auto, start_time=START, area='35.29,-98.96,44.63,-86.04')
    assert great_lakes.startswith('lake state: auto -- at the start the run asks glsea (NOAA GLERL'), great_lakes
    assert "if any of the domain's lake cells is inside 38.9N to 50.6N, 92.4W to 75.9W; otherwise" in great_lakes
    assert 'up to 48 h before 2026-11-12T00:00Z' in great_lakes and 'lake_state_source = "forcing"' in great_lakes
    elsewhere = obs.plan_sentence(auto, start_time=START, area='8.0,-76.0,36.0,-44.0')
    assert elsewhere.startswith('lake state: forcing (lake_state_source = "auto", and no automatic')
    forcing = obs.plan_sentence(SimpleNamespace(lake_state_source='forcing'), start_time=START)
    assert forcing == 'lake state: forcing (lake_state_source = "forcing"); no lake analysis is requested'
    licensed = obs.plan_sentence(SimpleNamespace(lake_state_source='era5-land', lake_state_path='/d/lakes.nc'))
    assert licensed.startswith('lake state: era5-land (Copernicus') and '/d/lakes.nc' in licensed


def _box_count(lat, lon, row):
    """The Rust coverage entry's count, for a host whose bridge predates it."""
    lat = np.asarray(lat, float)
    lon = (np.asarray(lon, float) + 180.) % 360. - 180.
    inside = ((lat >= row.get('south', -90.)) & (lat <= row.get('north', 90.))
              & (lon >= row.get('west', -180.)) & (lon <= row.get('east', 180.)))
    return dict(covered=float(inside.sum()))


def _experiment(map_proj, ref_lat, ref_lon, truelat1, truelat2, nx, ny, dx):
    from woof.experiment import ProjectionConfig
    projection = ProjectionConfig(map_proj=map_proj, ref_lat=ref_lat, ref_lon=ref_lon,
                                  truelat1=truelat1, truelat2=truelat2, stand_lon=ref_lon)
    run = SimpleNamespace(lake_state_source='auto', nx=nx, ny=ny, dx=dx)
    return SimpleNamespace(projection=projection, domains=[SimpleNamespace(run=run)], start_time=START)


# The 2026-11-12 card runs' domains (sea6, ne48 from `woof domain`; OKC from the
# met_em door), and a polar 12 km domain over the Beaufort Sea in place of that set.
SEA6 = ('lambert', 31.0, -62.0, 21.0, 41.0, 60, 48, 12000.)      # open Atlantic, GFS
SEA6_FETCH_AREA = '13.32,-80.94,48.63,-43.06'                     # its padded fetch box
OKC = ('lambert', 36.00000381, -97.5, 30.0, 60.0, 150, 150, 3000.)  # met_em, no area
BEAUFORT = ('polar', 71.0, -150.0, 71.0, 71.0, 120, 100, 12000.)   # met_em, no area
NE48 = ('lambert', 41.5, -75.5, 31.5, 51.5, 90, 72, 12000.)       # Erie and Ontario


@pytest.mark.parametrize('domain, area', [(SEA6, SEA6_FETCH_AREA), (OKC, None), (BEAUFORT, None)],
                         ids=['sea6-gfs', 'okc-met_em', 'beaufort-met_em'])
def test_a_domain_no_automatic_analysis_reaches_is_planned_with_no_request(monkeypatch, domain, area):
    """Defect 3 of the 2026-11-12 card runs: the plan promised a GLSEA request for
    each of these domains, deciding from the padded fetch area (the 31N box
    reaches 48.6N and 80.9W) or from no area at all on the met_em door.
    The run made no request.  The plan now asks the run's own question of
    the domain's own cells."""
    monkeypatch.setattr(obs, 'coordinate_coverage', _box_count)
    line = obs.experiment_plan_sentence(_experiment(*domain), area=area)
    assert line.startswith('lake state: forcing (lake_state_source = "auto", and no automatic lake '
                           'analysis reaches this domain: glsea covers only 38.9N to 50.6N, 92.4W to '
                           '75.9W)'), line
    assert line.endswith('no lake analysis is requested') and 'asks' not in line


def test_a_domain_over_the_great_lakes_is_planned_with_the_request_and_its_cell_count(monkeypatch):
    monkeypatch.setattr(obs, 'coordinate_coverage', _box_count)
    experiment = _experiment(*NE48)
    lat, lon = obs.root_cells(experiment)
    inside = int(_box_count(lat, lon, obs.source_rows()['glsea'])['covered'])
    assert lat.shape == (72, 90) and 0 < inside < 72 * 90
    line = obs.experiment_plan_sentence(experiment, area='22.38,-97.50,60.44,-53.50')
    assert line.startswith('lake state: auto -- at the start the run asks glsea (NOAA GLERL'), line
    assert (f"if any of the domain's lake cells is inside 38.9N to 50.6N, 92.4W to 75.9W, where {inside} "
            "of the domain's 6480 cells are; otherwise no lake analysis is requested") in line


# A 12 km domain on the Ohio valley (37N 85W, 720 km): its northern rows are
# inside GLSEA's box, and its lakes (Kentucky Lake, Lake Barkley, Lake
# Cumberland) are south of it.
OHIO_VALLEY = ('lambert', 37.0, -85.0, 27.0, 47.0, 60, 60, 12000.)
OHIO_VALLEY_LAKES = (np.array([36.75, 36.90, 36.95]), np.array([-88.10, -88.00, -85.00]))


def test_a_domain_that_enters_the_box_with_its_lakes_outside_it_is_not_promised_a_request(monkeypatch, tmp_path):
    """The plan is written before the lake mask exists, so for a domain an
    automatic row reaches it states the run's condition instead of a
    request.  The plan said "the run asks glsea" for this domain; the run,
    counting its lake cells, asks nothing."""
    monkeypatch.setattr(obs, 'coordinate_coverage', _box_count)
    experiment = _experiment(*OHIO_VALLEY)
    lat, lon = obs.root_cells(experiment)
    inside = int(_box_count(lat, lon, obs.source_rows()['glsea'])['covered'])
    assert 0 < inside < lat.size
    line = obs.experiment_plan_sentence(experiment)
    asks, condition = line.split(') if ', 1)
    assert asks.startswith('lake state: auto -- at the start the run asks glsea (NOAA GLERL'), line
    assert condition.startswith("any of the domain's lake cells is inside 38.9N to 50.6N, 92.4W to 75.9W, "
                                f"where {inside} of the domain's 3600 cells are; otherwise no lake analysis "
                                "is requested; lake cells the analysis does not cover keep the forcing's "
                                'lake state'), line
    # The run's half, on this domain's lake cells: nothing is fetched.
    monkeypatch.setattr(obs, 'urlopen', lambda *a, **k: pytest.fail('nothing may be fetched'))
    values, receipt = obs.load_state('auto', START, *OHIO_VALLEY_LAKES, cache=tmp_path)
    assert values == {} and receipt['status'] == 'outside_source_coverage'
    assert receipt['analysis_fetch'] == 'none' and receipt['checked'][0]['lake_cells_inside'] == 0
    assert obs.describe(receipt).endswith('holds 0 of the 3); no lake analysis requested')


def test_the_run_record_names_the_sources_it_checked_when_none_covers_the_lakes(monkeypatch, capsys, tmp_path):
    """The run's half of the same decision: lakes outside every automatic
    row fetch nothing, and report.json / run-receipt.json say which rows
    were checked and how many lake cells each holds."""
    monkeypatch.setattr(obs, 'coordinate_coverage', _box_count)
    monkeypatch.setattr(obs, 'urlopen', lambda *a, **k: pytest.fail('nothing may be fetched'))
    monkeypatch.setenv('WOOF_LAKE_RECEIPT_DIR', str(tmp_path))
    values, receipt = obs.load_state('auto', START, np.array([35.6, 36.2, 34.9]),
                                     np.array([-97.1, -95.6, -98.3]), cache=tmp_path)
    assert values == {} and receipt['status'] == 'outside_source_coverage'
    assert receipt['target_cells'] == 3 and receipt['analysis_fetch'] == 'none'
    assert receipt['checked'] == [dict(source='glsea', extent='38.9N to 50.6N, 92.4W to 75.9W',
                                       lake_cells_inside=0)]
    obs.publish_receipt(receipt)
    line = capsys.readouterr().err.strip()
    assert line.startswith("lake state: forcing, because no automatic lake analysis covers this domain's "
                           "lake cells (lake_state_source = 'auto'; glsea covers 38.9N to 50.6N, 92.4W to "
                           "75.9W and holds 0 of the 3); no lake analysis requested; receipt "), line


def test_lake_cells_in_hand_decide_the_plan_outright(monkeypatch, tmp_path):
    """A door whose input files hold the lake mask hands the plan the lake
    cells, and the plan then makes the run's decision instead of stating
    its condition.  The Ohio valley domain above was told "the run asks
    glsea if any of the domain's lake cells is inside ..."; with its three
    lake cells in hand the plan says none is, and nothing is requested."""
    monkeypatch.setattr(obs, 'coordinate_coverage', _box_count)
    auto = SimpleNamespace(lake_state_source='auto')
    outside = obs.plan_sentence(auto, start_time=START, lakes=OHIO_VALLEY_LAKES)
    assert outside == ('lake state: forcing (lake_state_source = "auto", and no automatic lake analysis '
                       "covers this domain's 3 lake cells: glsea covers only 38.9N to 50.6N, 92.4W to 75.9W "
                       "and holds none of them); the lakes keep the forcing's own state and no lake "
                       'analysis is requested'), outside
    # The run's half, on the same cells: nothing is fetched.
    monkeypatch.setattr(obs, 'urlopen', lambda *a, **k: pytest.fail('nothing may be fetched'))
    _, receipt = obs.load_state('auto', START, *OHIO_VALLEY_LAKES, cache=tmp_path)
    assert receipt['status'] == 'outside_source_coverage' and receipt['analysis_fetch'] == 'none'
    # Two cells on Lake Erie and one on Kentucky Lake: the row is asked, and the count is the run's.
    mixed = (np.array([42.2, 41.9, 36.75]), np.array([-81.2, -82.5, -88.10]))
    inside = obs.plan_sentence(auto, start_time=START, lakes=mixed)
    assert inside.startswith('lake state: auto -- at the start the run asks glsea (NOAA GLERL'), inside
    assert ("up to 48 h before 2026-11-12T00:00Z), whose extent 38.9N to 50.6N, 92.4W to 75.9W holds 2 of "
            "the domain's 3 lake cells; lake cells the analysis does not cover keep the forcing's lake "
            'state') in inside, inside
    assert ' if ' not in inside and 'otherwise' not in inside
    assert obs.automatic_coverage(*mixed)[0][1] == 2
    # No lake cell at all: the run's own no_lake_cells answer, said at plan time.
    none = obs.plan_sentence(auto, start_time=START, lakes=(np.empty(0), np.empty(0)))
    assert none == 'lake state: no lake cells in this domain; no lake analysis is requested'
    # A choice other than auto does not read the cells.
    forcing = obs.plan_sentence(SimpleNamespace(lake_state_source='forcing'), lakes=mixed)
    assert forcing == 'lake state: forcing (lake_state_source = "forcing"); no lake analysis is requested'


def test_the_run_says_why_it_binds_none_of_the_lake_cells_its_input_file_holds(monkeypatch, capsys):
    """The 2026-11-12 OKC wrfinput card run: the run line and report.json said
    "no lake cells in this domain" of a domain whose frames carry 197 lake
    cells.  The wrfinput door builds its driver from the land use, where
    real.exe has folded the lakes into water, and restores LAKEMASK
    afterwards.  The driver build is told the file's count, and the bind
    says why it binds nothing.  numpy stands in for cupy on the two calls
    this branch makes."""
    import sys
    monkeypatch.setitem(sys.modules, 'cupy', np)
    monkeypatch.setattr(obs, 'load_state', lambda *a, **k: pytest.fail('nothing may be fetched'))
    cfg = SimpleNamespace(lake_state_source='auto')
    grid = dict(valid_time=START, latitude=np.zeros((2, 3)), longitude=np.zeros((2, 3)))

    def build(lakemask):
        return obs.initialize_surface({'tsk': np.zeros((2, 3)), 'lakemask': lakemask}, cfg, **grid)

    file_mask = np.array([[0., 1., 1.], [1., 0., 0.]])
    values, receipt = obs.knowing_input_lake_cells(file_mask, build)(np.zeros((2, 3)))
    assert values is None and receipt == dict(
        schema='gpuwm.lake-init.v1', status='retained_forcing', source='forcing', requested='auto',
        analysis_fetch='none', input_lake_cells=3,
        reason="the lake binding reads a domain's lake cells from its land use (LU_INDEX) while the lake "
               "model is off, and LU_INDEX names none of the 3 lake cells the input files' LAKEMASK holds, "
               'so no lake analysis is requested')
    assert capsys.readouterr().err.strip() == 'lake state: forcing, because ' + receipt['reason']
    assert obs._INPUT_LAKE_CELLS.get() is None
    # A file with no lake cells, a file with no LAKEMASK, and a build no door wrapped: the plain answer.
    for wrapped in (obs.knowing_input_lake_cells(np.zeros((2, 3)), build),
                    obs.knowing_input_lake_cells(None, build), build):
        _, receipt = wrapped(np.zeros((2, 3)))
        assert receipt == dict(schema='gpuwm.lake-init.v1', status='no_lake_cells', requested='auto')
        assert capsys.readouterr().err.strip() == (
            'lake state: no lake cells in this domain; no lake analysis requested')


def test_a_store_tile_buffer_binds_nothing_and_prints_nothing(capsys):
    _, receipt = obs.initialize_surface({}, SimpleNamespace(lake_state_source='auto'), valid_time=START,
                                        latitude=None, longitude=None, cold_start=False)
    assert receipt['status'] == 'store_buffer' and capsys.readouterr().err == ''
    assert obs.describe(receipt).startswith('lake state: gathered from the prepared store')


def test_a_resumed_run_records_that_its_lakes_came_from_the_checkpoint(capsys, tmp_path):
    """Both runners build the driver, so initialize_surface binds afresh, and
    then restore the checkpoint over it, lake fields included.  The run record
    named that overwritten bind as what the lakes started from (item 5 review)."""
    _, fresh = obs.initialize_surface({}, SimpleNamespace(lake_state_source='forcing'),
                                      valid_time=START, latitude=None, longitude=None)
    bound = dict(schema='gpuwm.lake-init.v1', status='bound', source='glsea', requested='auto',
                 receipt_path=str(tmp_path / 'lake-init-x.json'))
    drivers = [SimpleNamespace(lake_initialization_receipt=bound),
               SimpleNamespace(lake_initialization_receipt=fresh),
               SimpleNamespace(lake_initialization_receipt=None)]
    capsys.readouterr()
    checkpoint = tmp_path / 'checkpoint-000012'
    marked = obs.mark_restored(drivers, checkpoint=checkpoint)
    assert len(marked) == 2 and drivers[2].lake_initialization_receipt is None
    first = drivers[0].lake_initialization_receipt
    assert first['status'] == 'restored_from_checkpoint' and first['checkpoint'] == str(checkpoint)
    assert first['requested'] == 'auto'
    assert first['start_up_bind'] == dict(status='bound', source='glsea', receipt_path=bound['receipt_path'])
    line = obs.describe(first)
    assert line == (f'lake state: restored from the checkpoint {checkpoint}; the start-up bind '
                    '(bound, glsea) was overwritten by it')
    assert capsys.readouterr().err.strip() == line
    warm = obs.mark_restored([SimpleNamespace(lake_initialization_receipt=fresh)],
                             source='the WRF restart files (wrfrst)', checkpoint=None)
    assert obs.describe(warm[0]).startswith('lake state: restored from the WRF restart files (wrfrst); ')


def test_a_resume_fetches_no_lake_analysis_and_says_so(capsys, monkeypatch, tmp_path):
    """A resume restores the lake fields over whatever the start-up bind put
    there, so the bind was a download nobody used (item 5 follow-up).  Under
    the runner's resume wrapper the drivers built before the restore fetch
    nothing; mark_restored records it and ends the skip, so a nest spawned
    later binds as an unbroken run's would."""
    def no_fetch(*args, **kwargs):
        raise AssertionError('a resume fetched the lake analysis')

    monkeypatch.setattr(obs, 'load_state', no_fetch)
    checkpoint = tmp_path / 'checkpoint-000012'
    seen = {}

    @obs.skips_analysis_on_resume
    def runner(inputs, *, restart=None, initialization=None):
        _, receipt = obs.initialize_surface(
            {}, SimpleNamespace(lake_state_source='auto'), valid_time=START,
            latitude=np.zeros((2, 2)), longitude=np.zeros((2, 2)))
        seen['built'] = receipt
        driver = SimpleNamespace(lake_initialization_receipt=receipt)
        seen['marked'] = obs.mark_restored([driver], checkpoint=restart)
        seen['after'] = obs._RESUME_SOURCE.get()
        return driver

    capsys.readouterr()
    driver = runner(None, restart=checkpoint)
    assert seen['built']['status'] == 'restore_pending'
    assert seen['after'] is None and obs._RESUME_SOURCE.get() is None
    receipt = driver.lake_initialization_receipt
    assert receipt['status'] == 'restored_from_checkpoint'
    assert receipt['analysis_fetch'] == 'skipped' and 'start_up_bind' not in receipt
    assert receipt['requested'] == 'auto'
    line = f'lake state: restored from the checkpoint {checkpoint}; no lake analysis was fetched'
    assert obs.describe(receipt) == line
    # One line: the skipped bind printed nothing of its own.
    assert capsys.readouterr().err.strip() == line

    warm = runner(None, initialization=SimpleNamespace(warm_start_clocks={1: 0}))
    assert seen['built']['source'] == 'the WRF restart files (wrfrst)'
    assert warm.lake_initialization_receipt['analysis_fetch'] == 'skipped'


def test_a_fresh_run_is_not_touched_by_the_resume_wrapper(monkeypatch):
    @obs.skips_analysis_on_resume
    def runner(inputs, *, restart=None, initialization=None):
        return obs._RESUME_SOURCE.get()

    assert runner(None) is None
    assert runner(None, initialization=SimpleNamespace(warm_start_clocks=None)) is None


def test_both_prepared_runners_skip_the_fetch_on_a_resume():
    from woof import prepared_domain_tree_forecast as tree
    from woof import prepared_single_domain_forecast as single

    wrapper = obs.skips_analysis_on_resume(lambda inputs: inputs).__code__
    assert tree.run_prepared_tree.__wrapped__.__code__ is wrapper
    assert single.run_prepared_forecast.__wrapped__.__code__ is wrapper


def test_a_resume_plan_names_the_checkpoint_not_a_request():
    experiment = SimpleNamespace(domains=[SimpleNamespace(run=SimpleNamespace(lake_state_source='auto'))],
                                 start_time=START)
    line = obs.experiment_plan_sentence(experiment, restart='ckpt-6')
    assert line == ('lake state: a resume takes its lakes from the checkpoint ckpt-6; '
                    'no lake analysis is fetched')
    assert obs.experiment_plan_sentence(experiment).startswith('lake state: auto')
