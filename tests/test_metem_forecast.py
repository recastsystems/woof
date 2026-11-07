"""Public metgrid launch and its shared initialization contracts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_netcdf_bridge


@pytest.mark.parametrize('arguments', [[], ['case.toml','--met-em','met'],
    ['--wrfinput','wrf','--met-em','met']])
def test_exclusive_input_choice_precedes_readiness(monkeypatch, arguments):
    from woof import cli, capabilities
    monkeypatch.setattr(capabilities,'require_for_command',lambda *a:pytest.fail('invalid choice reached readiness'))
    with pytest.raises(SystemExit) as stopped:
        cli.main(['run',*arguments])
    assert stopped.value.code == 2


def test_public_metgrid_dispatch_and_existing_config(monkeypatch):
    from woof import cli, capabilities, provenance_gate, metem_forecast
    calls=[]
    monkeypatch.setattr(capabilities,'require_for_command',lambda *a:None)
    monkeypatch.setattr(provenance_gate,'announce',lambda *a:None)
    monkeypatch.setattr(metem_forecast,'run_metem_forecast',lambda *a,**kw:calls.append((a,kw)) or 0)
    assert cli.main(['run','--met-em','met','--outdir','forecast','--run-seconds','120']) == 0
    assert calls[0][0] == (Path('met'),Path('forecast'))
    assert calls[0][1]['run_seconds'] == 120
    parsed=cli.build_parser().parse_args(['run','case.toml'])
    assert parsed.config == Path('case.toml') and parsed.met_em is None


def test_metgrid_context_does_not_spoof_wrf_boundary_representation(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    text=INPUT_TEXT.replace(' use_theta_m = 0,',' use_theta_m = 1,')
    paths=_pair(tmp_path,inp=text)
    _, report=import_namelists(*paths,metgrid_initialization=True)
    assert not any(item.key == 'use_theta_m' for item in report.substitutions)
    with pytest.raises(ValueError,match='distinct input contracts'):
        import_namelists(*paths,metgrid_initialization=True,wrf_boundary_use_theta_m=1)


@pytest.mark.parametrize('spelling', [' use_theta_m = 1,', ''])
def test_moist_theta_on_the_metem_door_is_selected_and_announced(tmp_path, spelling, capsys):
    from types import SimpleNamespace
    from test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    from woof.wrfinput_forecast import announce_wrf_substitutions
    text = INPUT_TEXT.replace(' use_theta_m = 0,', spelling)
    toml, report = import_namelists(*_pair(tmp_path, inp=text), metgrid_initialization=True)
    assert 'use_theta_m = 1' in toml
    assert not any(item.key == 'use_theta_m' for item in report.fixed)
    assert not any(item.key == 'use_theta_m' for item in report.substitutions)
    announce_wrf_substitutions(SimpleNamespace(substitution_report=report), tmp_path/'r.json')
    out = capsys.readouterr().out
    assert 'use_theta_m=1' in out and 'moist potential temperature' in out


def test_shared_deep_soil_formula_preserves_native_bytes():
    from woof.static.build import deep_soil_temperature_at_terrain
    rng=np.random.default_rng(217)
    for dtype in (np.float32,np.float64):
        soil=(280+rng.random((5,7))*15).astype(dtype)
        terrain=(rng.random((5,7))*3000).astype(dtype)
        mask=(rng.random((5,7))>.3).astype(dtype)
        old=np.where(mask>.5,soil-.0065*terrain,soil)
        new=deep_soil_temperature_at_terrain(soil,terrain,mask)
        assert new.dtype == old.dtype
        assert new.tobytes() == old.tobytes()


@pytest.mark.parametrize('value', [1, 'false', None])
def test_soil_public_routers_do_not_coerce_truthy_fractional_flag(value):
    from woof.ingest.ruc_soil import preprocess_land_surface_soil, preprocess_ruc_soil
    for call,kw in ((preprocess_land_surface_soil,{'sf_surface_physics':2}),
                    (preprocess_ruc_soil,{})):
        with pytest.raises(TypeError,match='fractional_seaice must be boolean'):
            call({},soil_type=None,fractional_seaice=value,**kw)


def pressure_case(levels):
    return SimpleNamespace(path=Path('met_em.d01.test'),snapshot=SimpleNamespace(levels_hpa=np.array(levels),fields={}),
        attributes={'FLAG_PSFC':1,'FLAG_SOILHGT':1},geometry={'num_metgrid_levels':len(levels)+1})


def test_hybrid_pressure_representation_resolves_wrf_sfcp_override():
    from woof.metem_door import metgrid_initialization_controls
    run=SimpleNamespace(controls={})
    assert metgrid_initialization_controls(pressure_case([100,300,700,950]),run)['sfcp_to_sfcp'] is True
    descending = pressure_case([950,700,300,100])
    with pytest.raises(ValueError,match='sfcprs3 requires FLAG_SLP'):
        metgrid_initialization_controls(descending,run)
    descending.attributes['FLAG_SLP'] = 1
    with pytest.raises(ValueError,match='declared PMSL field'):
        metgrid_initialization_controls(descending,run)
    descending.snapshot.fields['PMSL'] = np.full((1,1),101100.)
    assert metgrid_initialization_controls(descending,run)['sfcp_to_sfcp'] is False
    run.controls={'domains':{'sfcp_to_sfcp':[True]}}
    assert metgrid_initialization_controls(pressure_case([950,700,300,100]),run)['sfcp_to_sfcp'] is True
    missing=pressure_case([950,700,300,100]);missing.attributes['FLAG_PSFC']=0
    with pytest.raises(ValueError,match='FLAG_PSFC'):
        metgrid_initialization_controls(missing,run)


def test_metgrid_metadata_missing_directory_is_actionable(tmp_path):
    from woof.metem_door import resolve_metem_run
    with pytest.raises(ValueError,match='producing namelist.input'):
        resolve_metem_run(tmp_path)


@requires_netcdf_bridge
def test_scalar_soil_coordinate_is_named_refusal(tmp_path):
    from test_metem_ingest import case
    from woof.ingest.metem import read_met_em, MetgridRefusal
    def scalar(ds): ds.createVariable('SOIL_LEVELS','f4',())[...]=1
    with pytest.raises(MetgridRefusal,match='soil depth dimension, not a scalar'):
        read_met_em(case(tmp_path,mutate=scalar))


def _launcher_stub_run(*, flags=(), mp_physics=28):
    """A resolved-run shape the launcher can plan-review with no files.

    ``parse_met_em_name`` reads names only and the door already computes
    coverage from them, so nothing here needs a met_em on disk.  It carries
    ``metadata`` and a one-domain ``experiment.domains`` because plan
    review now asks the analyzed-field capability question too, which is
    the same inventory ``prepare_metem_run`` asks it of.
    """
    from datetime import datetime
    root = SimpleNamespace(nz=2, base_temp=290., mp_physics=mp_physics)
    return SimpleNamespace(
        toml_text='[shared]\nnz = 2\neta_levels = [1.0, 0.5, 0.0]\n',
        coverage_seconds=21600.0,
        source_top_pressure_pa=None,
        interval_seconds=10800,
        metadata={1: SimpleNamespace(
            global_attributes=dict(flags),
            path=Path('met_em.d01.2021-06-01_00_00_00.nc'))},
        paths={1: (Path('met_em.d01.2021-06-01_00_00_00.nc'),
                   Path('met_em.d01.2021-06-01_03_00_00.nc'),
                   Path('met_em.d01.2021-06-01_06_00_00.nc'))},
        experiment=SimpleNamespace(
            start_time=datetime(2021, 6, 1, 0, 0), run_seconds=3600.0,
            vertical=SimpleNamespace(eta_levels=(1.0, 0.5, 0.0), p_top=5000.),
            domains=(SimpleNamespace(grid_id=1, parent_id=0, run=root),),
            root=SimpleNamespace(run=root)),
        controls={})


def test_met_em_window_is_bounded_by_the_forcing_not_by_the_producing_namelist():
    """A 6 h met_em directory is a 6 h directory whatever its namelist says.

    ``prepare_metem_run`` used to bound ``--run-seconds`` by
    ``run.experiment.run_seconds``, the duration of the namelist that
    happened to produce the files, so forcing already on disk could not be
    used.  What limits the window is the series, and the door now carries
    it as ``coverage_seconds``, measured from the file NAMES alone.
    """
    from dataclasses import fields
    from datetime import datetime
    from woof.metem_door import MetemRun, metem_coverage_seconds
    from woof.metem_forecast import metem_window_seconds
    run = _launcher_stub_run()
    assert 'coverage_seconds' in {item.name for item in fields(MetemRun)}
    assert metem_coverage_seconds(run.paths, run.experiment.start_time) == 21600.0
    assert run.experiment.run_seconds == 3600.0
    assert metem_window_seconds(run, 21600) == 21600.0
    assert metem_window_seconds(run, 7200) == 7200.0
    with pytest.raises(ValueError, match='2021-06-01_06:00:00') as refused:
        metem_window_seconds(run, 25200)
    assert '21600 s' in str(refused.value)
    # A FRACTIONAL DURATION INSIDE COVERAGE RUNS.  It is what the base
    # door accepted (any finite 0 < run_seconds <= coverage), and nothing
    # in this check needs a whole second: run_seconds is resolved on a
    # microsecond lattice (woof/experiment.py:2987), the step-grid
    # question is a rational division by dt (woof/experiment.py:2347),
    # and the bound is an exact float comparison.
    assert metem_window_seconds(run, 3600.5) == 3600.5
    assert metem_window_seconds(run, 21599.999) == 21599.999
    assert metem_window_seconds(run, 21600.0) == 21600.0
    with pytest.raises(ValueError, match='2021-06-01_06:00:00') as beyond:
        metem_window_seconds(run, 21600.5)
    said = str(beyond.value)
    # And a fractional request that misses is quoted as itself, not
    # rounded into a number the caller never asked for.
    assert 'run of 21600.5 s' in said
    assert 'only 21600 s' in said
    assert 'whole number of seconds' not in said
    assert 'output step grid' not in said
    with pytest.raises(ValueError, match='finite, positive'):
        metem_window_seconds(run, 0)
    with pytest.raises(ValueError, match='finite, positive'):
        metem_window_seconds(run, float('inf'))
    # Every refusal this function raises names a way out, including the
    # one for an argument that is not a number at all.
    for bad in (0, float('inf'), float('nan'), 'six hours'):
        with pytest.raises(ValueError) as named:
            metem_window_seconds(run, bad)
        assert '21600 s this forcing covers' in str(named.value)


def test_both_source_doors_bound_the_window_through_one_function(monkeypatch):
    """Two doors never disagree about one configuration."""
    from datetime import datetime
    from woof import metem_forecast, wrfinput_forecast
    from woof.ingest import preflight
    seen = []

    def record(run_seconds, *, coverage_seconds, source, last_valid_time):
        seen.append((source, float(coverage_seconds), last_valid_time))
        return float(run_seconds)

    monkeypatch.setattr(preflight, 'check_forcing_window', record)
    metem_forecast.metem_window_seconds(_launcher_stub_run(), 7200)
    wrfinput_forecast.wrfinput_window_seconds(SimpleNamespace(
        coverage=SimpleNamespace(coverage_seconds=21600.0,
                                 end=datetime(2021, 6, 1, 6, 0))), 7200)
    assert [item[0] for item in seen] == ['woof run --met-em', 'woof run --wrfinput']
    assert {item[1] for item in seen} == {21600.0}
    assert {item[2] for item in seen} == {datetime(2021, 6, 1, 6, 0)}
    # And the real function, reached from the wrfinput door, says the same
    # sentence the met_em door gets.
    monkeypatch.undo()
    with pytest.raises(ValueError, match='2021-06-01_06:00:00'):
        wrfinput_forecast.wrfinput_window_seconds(SimpleNamespace(
            coverage=SimpleNamespace(coverage_seconds=21600.0,
                                     end=datetime(2021, 6, 1, 6, 0))), 25200)


def _fail_if_reached(name):
    def reached(*arguments, **keywords):
        pytest.fail(f'{name} was reached for a configuration refused at plan review')
    return reached


def test_a_refused_window_or_ladder_never_reaches_the_gpu_or_makes_an_outdir(
        monkeypatch, tmp_path):
    """Refusals fire at plan review, not after the run has started.

    Both of these used to be raised inside the spawned worker, after
    ``select_gpu``, after the GPU file lock, and after ``outdir.mkdir``.
    """
    import subprocess
    from woof import metem_door, metem_forecast, supervisor
    monkeypatch.setattr(metem_door, 'resolve_metem_run',
                        lambda *a, **kw: _launcher_stub_run())
    for name in ('select_gpu', 'GPUFileLock', 'preflight_exclusive_gpu'):
        monkeypatch.setattr(supervisor, name, _fail_if_reached(name))
    monkeypatch.setattr(subprocess, 'run', _fail_if_reached('the forecast worker'))
    outdir = tmp_path/'forecast'
    with pytest.raises(ValueError, match='names no ladder'):
        metem_forecast.run_metem_forecast(tmp_path, outdir, vertical_grid='stretched')
    assert not outdir.exists()
    with pytest.raises(ValueError, match='2021-06-01_06:00:00'):
        metem_forecast.run_metem_forecast(tmp_path, outdir, run_seconds=25200)
    assert not outdir.exists()


def test_the_wrfinput_door_refuses_its_window_before_the_gpu_or_the_outdir(
        monkeypatch, tmp_path):
    """One function, both doors, and now ONE TIME: plan review.

    ``check_forcing_window`` was reachable from this door only through
    ``prepare_wrf_run``, which the supervised parent reaches inside the
    spawned worker, past ``select_gpu``, the GPU file lock and
    ``outdir.mkdir``.  The shared check therefore fired after the run had
    started on one of the two doors it claims to serve.
    """
    import subprocess
    from datetime import datetime
    from woof import supervisor, wrfinput_door, wrfinput_forecast
    resolved = []

    def stub(directory, **keywords):
        resolved.append(directory)
        return SimpleNamespace(coverage=SimpleNamespace(
            coverage_seconds=21600.0, end=datetime(2021, 6, 1, 6, 0)))

    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', stub)
    for name in ('select_gpu', 'GPUFileLock', 'preflight_exclusive_gpu'):
        monkeypatch.setattr(supervisor, name, _fail_if_reached(name))
    monkeypatch.setattr(subprocess, 'run', _fail_if_reached('the forecast worker'))
    outdir = tmp_path/'forecast'
    with pytest.raises(ValueError, match='2021-06-01_06:00:00') as refused:
        wrfinput_forecast.run_wrf_forecast(tmp_path, outdir, run_seconds=25200)
    assert 'woof run --wrfinput' in str(refused.value)
    assert not outdir.exists()
    assert resolved == [tmp_path]


def test_a_fractional_window_inside_the_forcing_runs_on_both_doors(monkeypatch):
    """Never refuse solely because a number is not round.

    The base wrfinput door accepted any finite ``0 < run_seconds <=
    coverage_seconds`` (woof/wrfinput_forecast.py:143 at f08085092), and
    a fractional duration lands on the microsecond lattice
    (woof/experiment.py:2987) and divides by dt like any other
    (woof/experiment.py:2347).
    """
    from datetime import datetime
    from woof import metem_forecast, wrfinput_forecast
    run = SimpleNamespace(coverage=SimpleNamespace(
        coverage_seconds=21600.0, end=datetime(2021, 6, 1, 6, 0)))
    assert wrfinput_forecast.wrfinput_window_seconds(run, 7200.5) == 7200.5
    assert wrfinput_forecast.wrfinput_window_seconds(run, 0.25) == 0.25
    assert metem_forecast.metem_window_seconds(_launcher_stub_run(), 7200.5) == 7200.5


def test_both_doors_carry_the_shared_gpu_remedy_their_own_preflight_prints(
        monkeypatch, tmp_path):
    """``GPUPreflightError`` names ``--allow-shared-gpu``; the doors have it.

    ``preflight_exclusive_gpu`` tells the operator to pass
    ``--allow-shared-gpu``, and these are the two routes that call it, so
    the remedy has to exist here or the message is an instruction the
    caller cannot follow.
    """
    import subprocess
    from woof import metem_door, metem_forecast, supervisor, wrfinput_forecast, wrfinput_door
    calls = []

    class _Lock:
        def __init__(self, *arguments, **keywords):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            return False

    monkeypatch.setattr(metem_door, 'resolve_metem_run',
                        lambda *a, **kw: _launcher_stub_run())
    monkeypatch.setattr(supervisor, 'select_gpu',
                        lambda uuid=None: SimpleNamespace(uuid='GPU-stub'))
    monkeypatch.setattr(supervisor, 'GPUFileLock', _Lock)
    monkeypatch.setattr(supervisor, 'preflight_exclusive_gpu',
                        lambda uuid, **keywords: calls.append(keywords))
    monkeypatch.setattr(subprocess, 'run',
                        lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run',
                        lambda *a, **kw: _launcher_stub_run())
    source = tmp_path/'source'
    source.mkdir()
    outdir = tmp_path/'forecast'
    # This test exercises physical-card supervision, so remove the
    # host-only gate mask that deliberately bypasses that owner.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert metem_forecast.run_metem_forecast(source, outdir) == 0
    assert calls[-1]['allow_shared_gpu'] is False
    assert metem_forecast.run_metem_forecast(source, outdir, allow_shared_gpu=True) == 0
    assert calls[-1]['allow_shared_gpu'] is True
    assert wrfinput_forecast.run_wrf_forecast(source, outdir) == 0
    assert calls[-1]['allow_shared_gpu'] is False
    assert wrfinput_forecast.run_wrf_forecast(source, outdir, allow_shared_gpu=True) == 0
    assert calls[-1]['allow_shared_gpu'] is True


def test_both_module_parsers_accept_the_flag_and_main_forwards_it(monkeypatch):
    from woof import metem_forecast, provenance_gate, wrfinput_forecast
    forwarded = []
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    monkeypatch.setattr(metem_forecast, 'run_metem_forecast',
                        lambda *a, **kw: forwarded.append(('met-em', kw)) or 0)
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *a, **kw: forwarded.append(('wrfinput', kw)) or 0)
    assert wrfinput_forecast.build_parser().parse_args(
        ['--wrfinput', 'w', '--outdir', 'o']).allow_shared_gpu is False
    assert wrfinput_forecast.build_parser().parse_args(
        ['--wrfinput', 'w', '--outdir', 'o', '--allow-shared-gpu']).allow_shared_gpu is True
    assert metem_forecast.main(['--met-em', 'm', '--outdir', 'o', '--allow-shared-gpu']) == 0
    assert forwarded[-1] == ('met-em', forwarded[-1][1])
    assert forwarded[-1][1]['allow_shared_gpu'] is True
    assert wrfinput_forecast.main(['--wrfinput', 'w', '--outdir', 'o', '--allow-shared-gpu']) == 0
    assert forwarded[-1][1]['allow_shared_gpu'] is True
    assert metem_forecast.main(['--met-em', 'm', '--outdir', 'o']) == 0
    assert forwarded[-1][1]['allow_shared_gpu'] is False


def test_an_unsupported_analyzed_field_is_refused_before_the_gpu_or_the_outdir(
        monkeypatch, tmp_path):
    """The one refusal C-040 keeps fires at plan review, like the other two.

    ``check_analyzed_scalar_capability`` was reachable only from
    ``metgrid_initialization_controls`` and ``metgrid_memory_admission``,
    both of which run inside ``prepare_metem_run`` -- which the supervised
    launcher reaches after ``select_gpu``, after the GPU file lock and
    after ``outdir.mkdir``.  Everything the question needs is in the
    resolved run, so the launcher asks it before line one of the launch.
    """
    import subprocess
    from woof import metem_door, metem_forecast, supervisor
    monkeypatch.setattr(metem_door, 'resolve_metem_run',
                        lambda *a, **kw: _launcher_stub_run(
                            flags={'FLAG_QNBCA': 1}))
    for name in ('select_gpu', 'GPUFileLock', 'preflight_exclusive_gpu'):
        monkeypatch.setattr(supervisor, name, _fail_if_reached(name))
    monkeypatch.setattr(subprocess, 'run', _fail_if_reached('the forecast worker'))
    outdir = tmp_path/'forecast'
    with pytest.raises(ValueError, match='FLAG_QNBCA') as refused:
        metem_forecast.run_metem_forecast(tmp_path, outdir)
    assert 'Regenerate met_em without QNBCA' in str(refused.value)
    assert not outdir.exists()


def test_both_capability_doors_ask_one_function_of_one_inventory(monkeypatch):
    """Two doors never disagree about one configuration."""
    from woof import metem_door
    seen = []
    monkeypatch.setattr(metem_door, 'check_analyzed_scalar_capability',
                        lambda attributes, cfg: seen.append((attributes, cfg)))
    run = _launcher_stub_run(flags={'FLAG_QNBCA': 1})
    metem_door.check_analyzed_scalar_capabilities(run)
    metem_door.check_analyzed_scalar_capabilities(run, run.experiment)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0][0] == {'FLAG_QNBCA': 1}


def test_a_ladder_is_not_pinned_to_the_producing_namelist_level_count(tmp_path):
    """C-124's other half: the level count is the ladder's, not the namelist's.

    ``resolve_metem_vertical`` validated every ladder against
    ``run.experiment.root.run.nz``, so an authored ``explicit:PATH`` ladder
    had to carry exactly the producing namelist's ``e_vert`` and no count
    other than it could ever be asked for.  The resolved ladder owns the
    count now, and the emitted ``[shared] nz`` is rewritten to match.
    """
    from woof.metem_forecast import METEM_MINIMUM_E_VERT, resolve_metem_vertical
    run = _launcher_stub_run()
    assert run.experiment.root.run.nz == 2
    ladder = tmp_path/'ladder.toml'
    eta = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    ladder.write_text('eta_levels = [' + ', '.join(repr(v) for v in eta) + ']\n',
                      encoding='utf-8')
    text, policy, receipt = resolve_metem_vertical(
        run, run.toml_text, vertical_grid=f'explicit:{ladder}')
    assert receipt['e_vert'] == 6
    assert '\nnz = 5\n' in text and '\nnz = 2\n' not in text
    assert repr(eta) in text
    # And the count can be asked for by name, with no ladder file at all.
    text, policy, receipt = resolve_metem_vertical(
        run, run.toml_text, vertical_grid='native', vertical_levels=6)
    assert receipt['e_vert'] == 6 and '\nnz = 5\n' in text
    # A ladder file and an explicit count that disagree are two answers to
    # one question, so neither silently wins.
    with pytest.raises(ValueError, match='--vertical-levels 7'):
        resolve_metem_vertical(run, run.toml_text,
                               vertical_grid=f'explicit:{ladder}',
                               vertical_levels=7)
    with pytest.raises(ValueError, match='--vertical-levels'):
        resolve_metem_vertical(run, run.toml_text,
                               vertical_levels=METEM_MINIMUM_E_VERT - 1)


def test_a_refused_level_count_never_reaches_the_gpu_or_makes_an_outdir(
        monkeypatch, tmp_path):
    """``--vertical-levels`` is resolved at plan review with the rest."""
    import subprocess
    from woof import metem_door, metem_forecast, supervisor
    monkeypatch.setattr(metem_door, 'resolve_metem_run',
                        lambda *a, **kw: _launcher_stub_run())
    for name in ('select_gpu', 'GPUFileLock', 'preflight_exclusive_gpu'):
        monkeypatch.setattr(supervisor, name, _fail_if_reached(name))
    monkeypatch.setattr(subprocess, 'run', _fail_if_reached('the forecast worker'))
    outdir = tmp_path/'forecast'
    with pytest.raises(ValueError, match='--vertical-levels'):
        metem_forecast.run_metem_forecast(tmp_path, outdir, vertical_levels=3)
    assert not outdir.exists()


def test_the_module_door_takes_a_level_count_and_hands_it_on(monkeypatch):
    """The parser and the launcher agree about the new selector."""
    from woof import metem_forecast, provenance_gate
    seen = {}
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    monkeypatch.setattr(metem_forecast, 'run_metem_forecast',
                        lambda *a, **kw: seen.update(kw) or 0)
    assert metem_forecast.main([
        '--met-em', 'met', '--outdir', 'out', '--vertical-levels', '60']) == 0
    assert seen['vertical_levels'] == 60


@pytest.mark.parametrize('selector', ['native', 'wrf-auto', 'explicit:eta.txt'])
def test_common_cli_forwards_metem_ladder_levels_and_shared_gpu(monkeypatch, selector):
    from woof import cli, capabilities, provenance_gate, metem_forecast
    calls = []
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    monkeypatch.setattr(metem_forecast, 'run_metem_forecast',
                        lambda *a, **kw: calls.append(kw) or 0)
    assert cli.main(['run', '--met-em', 'met', '--vertical-grid', selector,
                     '--vertical-levels', '32', '--allow-shared-gpu']) == 0
    assert calls[-1]['vertical_grid'] == selector
    assert calls[-1]['vertical_levels'] == 32
    assert calls[-1]['allow_shared_gpu'] is True


def test_common_cli_forwards_shared_gpu_to_wrfinput(monkeypatch):
    from woof import cli, capabilities, provenance_gate, wrfinput_forecast
    calls = []
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *a, **kw: calls.append(kw) or 0)
    assert cli.main(['run', '--wrfinput', 'wrf', '--allow-shared-gpu']) == 0
    assert calls[-1]['allow_shared_gpu'] is True
    assert 'vertical_levels' not in calls[-1]


def test_common_cli_rejects_metem_levels_on_wrfinput_before_readiness(monkeypatch):
    from woof import cli, capabilities
    monkeypatch.setattr(capabilities, 'require_for_command',
                        lambda *a: pytest.fail('bad option placement reached readiness'))
    with pytest.raises(SystemExit) as stopped:
        cli.main(['run', '--wrfinput', 'wrf', '--vertical-levels', '32'])
    assert stopped.value.code == 2


def test_metem_worker_defines_terrain_helper_before_entrypoint():
    """Execute the worker module up to main, where its helpers must exist."""
    import runpy,sys
    from types import SimpleNamespace
    previous=sys.gettrace()
    def entry(frame,event,arg):
        if event=='call' and frame.f_code.co_name=='main' and frame.f_globals.get('__name__')=='__main__' and frame.f_globals.get('__package__')=='woof':
            helper=frame.f_globals.get('_smooth_cg_case')
            assert callable(helper), 'met_em worker starts before its terrain helper is defined'
            case=object()
            assert helper(SimpleNamespace(smooth_cg_topo=False),case,{}) is case
            raise SystemExit(0)
        return entry
    try:
        sys.settrace(entry)
        with pytest.raises(SystemExit) as stopped:
            runpy.run_module('woof.metem_forecast',run_name='__main__')
        assert stopped.value.code==0
    finally:sys.settrace(previous)
