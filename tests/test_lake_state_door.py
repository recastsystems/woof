"""``--lake-state`` on the met_em and wrfinput doors (2026-11-12 runner defect 4).

Both doors build their configuration from the producing namelist, which has
no lake-state key, so every such run took ``lake_state_source = "auto"`` and
asked for the Great Lakes surface analysis whenever the domain held lake
cells, with no switch to keep the input files' own lakes.  These tests hold
the switch from the command line, through the worker argv, into every
domain's RunConfig, on both doors, and hold a plain run byte-identical.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import requires_netcdf_bridge

TOML = '[shared]\nnz = 2\n\n[[domain]]\ngrid_id = 1\n'


def test_no_choice_leaves_the_resolved_toml_byte_identical():
    from woof.ingest.lake_state import select_lake_state
    assert select_lake_state(TOML) == (TOML, None)
    assert select_lake_state(TOML, None, None) == (TOML, None)


@pytest.mark.parametrize('source', ['forcing', 'auto', 'glsea'])
def test_a_choice_is_written_once_into_the_shared_table(source):
    import tomllib
    from woof.ingest.lake_state import select_lake_state
    text, note = select_lake_state(TOML, source)
    assert tomllib.loads(text)['shared']['lake_state_source'] == source
    assert text.count('lake_state_source') == 1
    assert note.startswith(f'lake state: {source} (--lake-state)')
    # A second application replaces, it does not stack.
    again, _ = select_lake_state(text, 'forcing')
    assert again.count('lake_state_source') == 1
    assert tomllib.loads(again)['shared']['lake_state_source'] == 'forcing'


def test_the_choices_are_the_source_table_rows():
    from woof.ingest.lake_state import lake_state_choices, source_rows
    assert lake_state_choices() == ('auto', 'forcing', *source_rows())


def test_a_row_without_a_download_needs_its_file_and_records_it(tmp_path):
    import tomllib
    from woof.ingest.lake_state import select_lake_state, source_rows
    offline = [rid for rid, row in source_rows().items()
               if any('url' not in field for field in row['field'])]
    assert offline, 'the table carries a row read from a local file'
    row = offline[0]
    with pytest.raises(ValueError, match='--lake-state-path FILE'):
        select_lake_state(TOML, row)
    with pytest.raises(ValueError, match='is not a file'):
        select_lake_state(TOML, row, tmp_path / 'absent.nc')
    data = tmp_path / 'lake.nc'
    data.write_bytes(b'CDF\x01')
    text, note = select_lake_state(TOML, row, data)
    shared = tomllib.loads(text)['shared']
    assert shared['lake_state_source'] == row
    assert Path(shared['lake_state_path']) == data.resolve()
    assert str(data.resolve()) in note


@pytest.mark.parametrize('source, path, message', [
    ('nowhere', None, 'not a lake-state source'),
    ('forcing', 'lake.nc', 'reads no lake file'),
    ('auto', 'lake.nc', 'chooses its row by coverage'),
    (None, 'lake.nc', 'give --lake-state ROW'),
])
def test_contradictory_requests_are_refused_by_name(source, path, message):
    from woof.ingest.lake_state import select_lake_state
    with pytest.raises(ValueError, match=message):
        select_lake_state(TOML, source, path)


_NAMELIST = """\
&time_control
 run_hours = 3,
 start_year = 2021, start_month = 06, start_day = 01, start_hour = 00,
 end_year = 2021, end_month = 06, end_day = 01, end_hour = 03,
 interval_seconds = 10800,
 input_from_file = .true.,
 history_interval = 60,
 restart = .false.,
 restart_interval = 1440,
/
&domains
 time_step = 60,
 max_dom = 1,
 e_we = 6, e_sn = 6, e_vert = 9,
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0,
 p_top_requested = 20000,
 num_metgrid_levels = 4, sfcp_to_sfcp = .true.,
 dx = 12000.0, dy = 12000.0,
 grid_id = 1, parent_id = 0, i_parent_start = 1, j_parent_start = 1,
 parent_grid_ratio = 1, parent_time_step_ratio = 1,
/
&physics
 mp_physics = 10, ra_lw_physics = 4, ra_sw_physics = 4, radt = 12,
 sf_sfclay_physics = 91, sf_surface_physics = 2, bl_pbl_physics = 11,
 cu_physics = 1,
/
&dynamics
 hybrid_opt = 2, etac = 0.2, w_damping = 1, epssm = 0.5,
 diff_opt = 2, km_opt = 4, mix_full_fields = .true.,
 diff_6th_opt = 2, diff_6th_factor = 0.12, diff_6th_slopeopt = 1,
 base_temp = 2.90D2, damp_opt = 3, zdamp = 5000., dampcoef = 0.2,
 khdif = 0, kvdif = 0, non_hydrostatic = .true., use_theta_m = 0,
 moist_adv_opt = 1,
/
&bdy_control
 spec_bdy_width = 5, spec_zone = 1, relax_zone = 4,
 specified = .true., nested = .false.,
/
"""


@pytest.fixture
def metgrid_directory(tmp_path):
    """Two real met_em files (00 and 03 UTC) and the namelist that made them."""
    from test_metem_ingest import write_case
    directory = tmp_path / 'met'
    directory.mkdir()
    for stamp in ('2021-06-01_00:00:00', '2021-06-01_03:00:00'):
        write_case(directory / f"met_em.d01.{stamp.replace(':', '_')}.nc", time=stamp,
                   global_override={'FLAG_PSFC': 1, 'FLAG_SOILHGT': 1, 'GRID_ID': 1,
                                    'PARENT_ID': 1, 'I_PARENT_START': 1,
                                    'J_PARENT_START': 1, 'PARENT_GRID_RATIO': 1})
    (directory / 'namelist.input').write_text(_NAMELIST)
    return directory


@requires_netcdf_bridge
def test_the_metem_door_writes_the_choice_into_every_domain(metgrid_directory):
    from woof.metem_door import resolve_metem_run
    plain = resolve_metem_run(metgrid_directory)
    assert 'lake_state' not in plain.toml_text
    assert plain.lake_state_note is None
    assert {d.run.lake_state_source for d in plain.experiment.domains} == {'auto'}
    chosen = resolve_metem_run(metgrid_directory, lake_state='forcing')
    assert {d.run.lake_state_source for d in chosen.experiment.domains} == {'forcing'}
    assert 'lake_state_source = "forcing"' in chosen.toml_text
    assert 'no lake analysis is requested' in chosen.lake_state_note
    # The flag changes only that key: the rest of the resolved TOML is the plain run's.
    assert chosen.toml_text.replace('lake_state_source = "forcing"\n', '') == plain.toml_text


@pytest.mark.parametrize('door, flag', [('metem_forecast', '--met-em'),
                                        ('wrfinput_forecast', '--wrfinput')])
def test_gpuwm_run_hands_the_choice_to_both_doors(monkeypatch, door, flag):
    import importlib
    from woof import cli, capabilities, provenance_gate
    module = importlib.import_module(f'woof.{door}')
    launcher = 'run_metem_forecast' if door == 'metem_forecast' else 'run_wrf_forecast'
    calls = []
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    monkeypatch.setattr(module, launcher, lambda *a, **kw: calls.append(kw) or 0)
    assert cli.main(['run', flag, 'inputs', '--outdir', 'forecast',
                     '--lake-state', 'forcing']) == 0
    assert calls[0]['lake_state'] == 'forcing'
    assert 'lake_state_path' not in calls[0]
    # Absent, nothing is passed: the door keeps the configuration's own default.
    assert cli.main(['run', flag, 'inputs', '--outdir', 'forecast']) == 0
    assert 'lake_state' not in calls[1]


def test_the_config_route_refuses_the_door_flag():
    from woof import cli
    with pytest.raises(SystemExit) as stopped:
        cli.main(['run', 'case.toml', '--lake-state', 'forcing'])
    assert stopped.value.code == 2


@pytest.mark.parametrize('module', ['metem_forecast', 'wrfinput_forecast'])
def test_both_worker_parsers_read_the_flags(module):
    import importlib
    parser = importlib.import_module(f'woof.{module}').build_parser()
    door = '--met-em' if module == 'metem_forecast' else '--wrfinput'
    args = parser.parse_args([door, 'in', '--outdir', 'out', '--lake-state', 'era5-land',
                              '--lake-state-path', 'lake.nc'])
    assert (args.lake_state, args.lake_state_path) == ('era5-land', 'lake.nc')
    plain = parser.parse_args([door, 'in', '--outdir', 'out'])
    assert (plain.lake_state, plain.lake_state_path) == (None, None)


def test_the_supervised_workers_carry_the_choice(tmp_path, monkeypatch, capsys):
    from test_wrfinput_metem_products import _pair, _supervised
    from test_metem_forecast import _launcher_stub_run
    from woof import metem_door, metem_forecast, wrfinput_door
    from woof.wrfinput_forecast import run_wrf_forecast
    calls = {}
    _supervised(monkeypatch, calls)
    seen = {}
    note = 'lake state: forcing (--lake-state); stub'
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run',
        lambda *a, **kw: seen.setdefault('wrfinput', kw)
        and SimpleNamespace(experiment=None, lake_state_note=note))
    assert run_wrf_forecast(tmp_path / 'wrf', tmp_path / 'out', render_products='none',
                            lake_state='forcing') == 0
    assert seen['wrfinput']['lake_state'] == 'forcing'
    assert _pair(calls['argv'], '--lake-state') == ['--lake-state', 'forcing']
    assert 'woof run --wrfinput: ' + note in capsys.readouterr().out

    def metem(directory, **kw):
        seen['metem'] = kw
        stub = _launcher_stub_run()
        stub.lake_state_note = note
        return stub
    monkeypatch.setattr(metem_door, 'resolve_metem_run', metem)
    data = tmp_path / 'lake.nc'
    data.write_bytes(b'CDF\x01')
    assert metem_forecast.run_metem_forecast(tmp_path / 'met', tmp_path / 'out2',
        render_products='none', lake_state='era5-land', lake_state_path=data) == 0
    assert seen['metem']['lake_state'] == 'era5-land'
    assert _pair(calls['argv'], '--lake-state') == ['--lake-state', 'era5-land']
    assert _pair(calls['argv'], '--lake-state-path') == ['--lake-state-path', str(data.resolve())]
    assert 'woof run --met-em: ' + note in capsys.readouterr().out


def test_an_explicit_auto_at_the_met_em_door_still_gets_the_plan_line(tmp_path, monkeypatch, capsys):
    """``--lake-state auto`` printed only the flag's own line, which names no
    source and says nothing of this domain.  The door now prints the
    plan's lake line after it, as an unflagged run's plan review does."""
    from test_wrfinput_metem_products import _supervised
    from test_metem_forecast import _launcher_stub_run
    from woof import metem_door, metem_forecast
    _supervised(monkeypatch, {})
    note = 'lake state: auto (--lake-state); stub'

    def metem(directory, **kw):
        stub = _launcher_stub_run()
        stub.lake_state_note = note
        return stub
    monkeypatch.setattr(metem_door, 'resolve_metem_run', metem)
    assert metem_forecast.run_metem_forecast(tmp_path / 'met', tmp_path / 'out',
                                             render_products='none', lake_state='auto') == 0
    out = capsys.readouterr().out
    assert 'woof run --met-em: ' + note in out
    assert 'woof run --met-em: lake state: auto -- at the start the run asks glsea (NOAA GLERL' in out
    assert "if any of the domain's lake cells is inside 38.9N to 50.6N, 92.4W to 75.9W; otherwise" in out


def _input_file(path, lake_rows, *, latitude, longitude, islake=21, lakemask_rows=None):
    """A 4 x 5 input file across GLSEA's southern edge (38.9N), with lake cells on ``lake_rows``.

    Rows 0 and 1 (38.0N, 38.5N) are south of GLSEA's box and rows 2 and 3
    (39.0N, 39.5N) are inside it; every column (86W to 84W) is inside its
    longitudes.  LU_INDEX carries the lake category on the named rows,
    columns 1 to 3, and cropland elsewhere.  ``lakemask_rows`` adds a
    LAKEMASK record that is 1 on those rows' same columns, as real.exe
    writes one into wrfinput.
    """
    import netCDF4 as nc
    import numpy as np
    lat, lon = np.meshgrid(np.array([38.0, 38.5, 39.0, 39.5]), np.linspace(-86.0, -84.0, 5), indexing='ij')
    land_use = np.full((4, 5), 12.0)
    for row in lake_rows:
        land_use[row, 1:4] = 21.0
    with nc.Dataset(path, 'w', format='NETCDF3_CLASSIC') as ds:
        for name, size in (('Time', 1), ('south_north', 4), ('west_east', 5)):
            ds.createDimension(name, size)
        ds.setncatts(dict(ISLAKE=islake, ISWATER=17, MMINLU='MODIFIED_IGBP_MODIS_NOAH'))
        records = [('LU_INDEX', land_use), (latitude, lat), (longitude, lon)]
        if lakemask_rows is not None:
            mask = np.zeros((4, 5))
            for row in lakemask_rows:
                mask[row, 1:4] = 1.0
            records.append(('LAKEMASK', mask))
        for name, values in records:
            ds.createVariable(name, 'f4', ('Time', 'south_north', 'west_east'))[0] = values
    return path


def _auto_experiment(*grid_ids, start=None, sf_lake_physics=0):
    from datetime import datetime
    run = SimpleNamespace(lake_state_source='auto', sf_lake_physics=sf_lake_physics)
    return SimpleNamespace(start_time=start or datetime(2026, 11, 12), run_seconds=3600.0, domains=tuple(
        SimpleNamespace(grid_id=gid, parent_id=gid - 1, run=run) for gid in grid_ids))


def _supervised_reading_files(monkeypatch):
    """``_supervised``, with the NetCDF reader's own subprocess left real.

    ``_supervised`` replaces ``subprocess.run`` to record the worker argv;
    the plan line reads the input file through the Rust reader, which is a
    subprocess too, so only the worker launch is recorded here.
    """
    import subprocess
    from test_wrfinput_metem_products import _supervised
    real = subprocess.run
    _supervised(monkeypatch, {})
    recorder = subprocess.run
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: (
        recorder if '--_worker' in command else real)(command, **kwargs))


UNREAD = ('lake state: forcing (lake_state_source = "auto"; the lake binding reads a domain' "'s lake cells from "
          'its land use (LU_INDEX) while the lake model is off, and LU_INDEX names none of the 6 lake cells the '
          "input files' LAKEMASK holds); the lakes keep the input files' own state and no lake analysis is "
          'requested')
OUTSIDE = ('lake state: forcing (lake_state_source = "auto", and no automatic lake analysis covers this '
           "domain's 6 lake cells: glsea covers only 38.9N to 50.6N, 92.4W to 75.9W and holds none of them); "
           "the lakes keep the forcing's own state and no lake analysis is requested")
INSIDE = ("), whose extent 38.9N to 50.6N, 92.4W to 75.9W holds 3 of the domain's 6 lake cells; lake cells "
          "the analysis does not cover keep the forcing's lake state")


@requires_netcdf_bridge
@pytest.mark.parametrize('latitude, longitude', [('XLAT_M', 'XLONG_M'), ('XLAT', 'XLONG')],
                         ids=['met_em', 'wrfinput'])
def test_the_input_files_own_lake_cells_decide_the_plan(tmp_path, monkeypatch, latitude, longitude):
    """Both input doors hold the lake mask at plan review (LU_INDEX == ISLAKE,
    the run's own lakemask), so the plan is the run's decision.  This
    domain's northern rows are inside GLSEA's box: with its lakes south of
    the box nothing is asked, and the run asks nothing of the same cells."""
    import numpy as np
    import woof.ingest.lake_state as lakes
    from test_lake_state_provenance import _box_count
    monkeypatch.setattr(lakes, 'coordinate_coverage', _box_count)
    names = dict(latitude=latitude, longitude=longitude)
    south = _input_file(tmp_path / 'south.nc', (0, 1), **names)
    lat, lon = lakes.input_lake_cells(south, **names)
    np.testing.assert_allclose(sorted(lat), [38.0] * 3 + [38.5] * 3)
    np.testing.assert_allclose(sorted(set(lon)), [-85.5, -85.0, -84.5])
    assert lakes.input_plan_sentences(_auto_experiment(1), {1: south}, **names) == [OUTSIDE]
    monkeypatch.setattr(lakes, 'urlopen', lambda *a, **k: pytest.fail('nothing may be fetched'))
    _, receipt = lakes.load_state('auto', _auto_experiment(1).start_time, lat, lon, cache=tmp_path)
    assert receipt['status'] == 'outside_source_coverage'
    assert receipt['checked'][0]['lake_cells_inside'] == 0 and receipt['target_cells'] == 6
    # Lakes on a row inside the box and a row outside it: GLSEA is asked, for the run's own count.
    across = _input_file(tmp_path / 'across.nc', (1, 2), **names)
    line, = lakes.input_plan_sentences(_auto_experiment(1), {1: across}, **names)
    assert line.startswith('lake state: auto -- at the start the run asks glsea (NOAA GLERL') and INSIDE in line
    assert lakes.automatic_coverage(*lakes.input_lake_cells(across, **names))[0][1] == 3
    # A land-use table with no lake category (ISLAKE = -1) names no lake cell, as in the run.
    usgs = _input_file(tmp_path / 'usgs.nc', (2, 3), islake=-1, **names)
    assert lakes.input_plan_sentences(_auto_experiment(1), {1: usgs}, **names) == [
        'lake state: no lake cells in this domain; no lake analysis is requested']
    # Each domain binds its own lakes, so a nested run gets a line per domain.
    nested = lakes.input_plan_sentences(_auto_experiment(1, 2), {1: south, 2: across}, **names)
    assert nested[0] == OUTSIDE.replace('lake state: ', 'lake state: d01: ')
    assert nested[1].startswith('lake state: d02: auto -- at the start the run asks glsea') and INSIDE in nested[1]
    # A resume and a choice other than auto read no file and are one line.
    assert lakes.input_plan_sentences(_auto_experiment(1, 2), {}, restart='ckpt.nc', **names) == [
        'lake state: a resume takes its lakes from the checkpoint ckpt.nc; no lake analysis is fetched']
    forcing = _auto_experiment(1, 2)
    forcing.domains[0].run.lake_state_source = 'forcing'
    assert lakes.input_plan_sentences(forcing, {}, **names) == [
        'lake state: forcing (lake_state_source = "forcing"); no lake analysis is requested']


@requires_netcdf_bridge
def test_a_wrfinput_lake_model_run_is_planned_on_the_files_own_lakemask(tmp_path, monkeypatch):
    """real.exe folds the lake category into water in wrfinput's LU_INDEX and
    keeps the lakes in LAKEMASK.  The wrfinput run starts its lake mask
    from LAKEMASK when the lake model is on and from LU_INDEX == ISLAKE
    when it is off, so the plan reads the same record for the same run."""
    import woof.ingest.lake_state as lakes
    from woof.wrfinput_forecast import lake_plan_lines
    from test_lake_state_provenance import _box_count
    monkeypatch.setattr(lakes, 'coordinate_coverage', _box_count)
    folded = _input_file(tmp_path / 'wrfinput_d01', (), latitude='XLAT', longitude='XLONG',
                         lakemask_rows=(1, 2))

    def resolved(sf_lake_physics):
        return SimpleNamespace(experiment=_auto_experiment(1, sf_lake_physics=sf_lake_physics),
                               wrfinput_paths={1: folded}, restart_paths=None)
    # Lake model off: the run binds nothing, and not because the domain has no lakes.
    assert lake_plan_lines(resolved(0)) == [UNREAD]
    line, = lake_plan_lines(resolved(1))
    assert line.startswith('lake state: auto -- at the start the run asks glsea (NOAA GLERL') and INSIDE in line
    # The met_em door never reads that record: its run starts the mask from LU_INDEX alone.
    assert lakes.input_plan_sentences(_auto_experiment(1, sf_lake_physics=1), {1: folded},
                                      latitude='XLAT', longitude='XLONG') == [
        'lake state: no lake cells in this domain; no lake analysis is requested']


@requires_netcdf_bridge
def test_the_met_em_door_prints_the_decision_not_the_condition(tmp_path, monkeypatch, capsys):
    """The met_em door printed "the run asks glsea if any of the domain's
    lake cells is inside ..." for a domain GLSEA's box reaches, although it
    holds the files that answer it."""
    import woof.ingest.lake_state as lakes
    from test_lake_state_provenance import _box_count
    from test_metem_forecast import _launcher_stub_run
    from woof import metem_door, metem_forecast
    monkeypatch.setattr(lakes, 'coordinate_coverage', _box_count)
    _supervised_reading_files(monkeypatch)
    south = _input_file(tmp_path / 'met_em.d01.2021-06-01_00_00_00.nc', (0, 1),
                        latitude='XLAT_M', longitude='XLONG_M')

    def metem(directory, **kw):
        stub = _launcher_stub_run()
        stub.paths = {1: (south, *stub.paths[1][1:])}
        return stub
    monkeypatch.setattr(metem_door, 'resolve_metem_run', metem)
    assert metem_forecast.run_metem_forecast(tmp_path / 'met', tmp_path / 'out', render_products='none') == 0
    out = capsys.readouterr().out
    assert 'woof run --met-em: ' + OUTSIDE + '\n' in out
    assert 'asks glsea' not in out


@requires_netcdf_bridge
def test_the_wrfinput_door_prints_a_lake_plan_line_and_its_dry_run_carries_it(tmp_path, monkeypatch, capsys):
    """A plain ``woof run --wrfinput`` printed no lake line at plan review
    and its dry run named no lake source, although the default is ``auto``
    and the wrfinput files hold the lake mask."""
    import json
    import woof.ingest.lake_state as lakes
    from test_lake_state_provenance import _box_count
    from woof import wrfinput_door, wrfinput_go
    from woof.wrfinput_forecast import run_wrf_forecast
    monkeypatch.setattr(lakes, 'coordinate_coverage', _box_count)
    _supervised_reading_files(monkeypatch)
    names = dict(latitude='XLAT', longitude='XLONG')
    south = _input_file(tmp_path / 'wrfinput_d01', (0, 1), **names)
    across = _input_file(tmp_path / 'wrfinput_d02', (1, 2), **names)

    def resolved(paths, **more):
        return SimpleNamespace(experiment=_auto_experiment(*paths), wrfinput_paths=paths, grid_ids=tuple(paths),
                               namelist_input=tmp_path / 'namelist.input', lake_state_note=None, **more)
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda *a, **kw: resolved({1: south}))
    assert run_wrf_forecast(tmp_path / 'wrf', tmp_path / 'out', render_products='none') == 0
    out = capsys.readouterr().out
    assert 'woof run --wrfinput: ' + OUTSIDE + '\n' in out
    assert out.count('lake state:') == 1
    # A WRF warm start restores its lakes from the wrfrst files: no request is named.
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run',
                        lambda *a, **kw: resolved({1: south}, restart_paths={1: tmp_path / 'wrfrst_d01'}))
    assert run_wrf_forecast(tmp_path / 'wrf', tmp_path / 'out2', render_products='none') == 0
    assert ('woof run --wrfinput: lake state: a resume takes its lakes from the WRF restart files (wrfrst); '
            'no lake analysis is fetched') in capsys.readouterr().out
    # `woof go --wrfinput DIR --dry-run`: the same lines, one per domain.
    assert wrfinput_go._dry_run(resolved({1: south, 2: across}), tmp_path, tmp_path / 'out3') == 0
    plan = json.loads(capsys.readouterr().out)['lake_state_plan']
    assert plan[0] == OUTSIDE.replace('lake state: ', 'lake state: d01: ')
    assert plan[1].startswith('lake state: d02: auto -- at the start the run asks glsea') and INSIDE in plan[1]
    flagged = resolved({1: south})
    flagged.lake_state_note = 'lake state: forcing (--lake-state); stub'
    assert wrfinput_go._dry_run(flagged, tmp_path, tmp_path / 'out4', lake_state='forcing') == 0
    record = json.loads(capsys.readouterr().out)
    assert record['lake_state'] == flagged.lake_state_note and 'lake_state_plan' not in record


def _go_args(*tokens):
    import argparse
    from woof.go_cli import _add_go_arguments
    parser = argparse.ArgumentParser()
    _add_go_arguments(parser)
    return parser.parse_args(tokens)


def test_gpuwm_go_carries_the_choice_on_its_wrf_input_routes(tmp_path, monkeypatch):
    from datetime import datetime
    import woof.cli
    import woof.wrfinput_door as door
    import woof.wrfinput_forecast as runner
    from woof.go_cli import GoRefusal, go_main
    resolved = []
    monkeypatch.setattr(door, 'resolve_wrfinput_run', lambda p, **kw: resolved.append(kw) or SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime(2024, 5, 25, 18), run_seconds=21600),
        grid_ids=(1,), namelist_input=Path('case/namelist.input'),
        lake_state_note='lake state: forcing (--lake-state); stub'))
    calls = []
    monkeypatch.setattr(runner, 'run_wrf_forecast', lambda *a, **kw: calls.append(kw) or 0)
    assert go_main(_go_args('--wrfinput', 'case', '--outdir', str(tmp_path / 'o'),
                            '--lake-state', 'forcing')) == 0
    assert resolved[-1]['lake_state'] == 'forcing' and calls[-1]['lake_state'] == 'forcing'
    # The met_em directory route hands the flag to `woof run --met-em`.
    met = tmp_path / 'met'
    met.mkdir()
    (met / 'namelist.input').write_text(_NAMELIST)
    commands = []
    monkeypatch.setattr(woof.cli, 'main', lambda command: commands.append(command) or 0)
    assert go_main(_go_args(str(met), '--lake-state', 'forcing')) == 0
    assert commands[-1][:2] == ['run', '--met-em']
    assert commands[-1][commands[-1].index('--lake-state') + 1] == 'forcing'
    # The CONFIG route reads its own lake_state_source key.
    with pytest.raises(GoRefusal, match='WRF-input execution'):
        go_main(_go_args('forecast.toml', '--lake-state', 'forcing'))
