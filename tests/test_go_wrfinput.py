"""The stock-namelist go door delegates to the production input runtime."""
import argparse
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import pytest
from woof.go_cli import _add_go_arguments, GoRefusal
from woof.go_wrfinput import launch

@pytest.fixture(autouse=True)
def resolved_stock_bundle(monkeypatch):
    from types import SimpleNamespace
    from datetime import datetime, timezone
    import woof.wrfinput_door as door
    monkeypatch.setattr(door, 'resolve_wrfinput_run', lambda *a, **k:
        SimpleNamespace(experiment=SimpleNamespace(start_time=datetime(2000,1,24,12,tzinfo=timezone.utc))))


def args(*flags):
    parser=argparse.ArgumentParser()
    _add_go_arguments(parser)
    return parser.parse_args(['--wrfinput','inputs','--outdir','output','--run-stamp','off',*flags])

def test_go_delegates_native_inputs_without_changing_namelist_physics(monkeypatch):
    import woof.wrfinput_forecast as runtime
    calls=[]
    monkeypatch.setattr(runtime,'run_wrf_forecast',lambda *a,**k:calls.append((a,k)) or 0)
    assert launch(args('--run-seconds','10800','--allow-shared-gpu','--no-verify-visuals','--products','none'))==0
    positional,controls=calls.pop()
    assert positional==(Path('inputs'),Path('output'))
    assert controls['run_seconds']==10800 and controls['allow_shared_gpu']
    assert 'rrtmg_variant' not in controls
    assert controls['render_products']=='none'

@pytest.mark.parametrize('flags',[['config.toml'],['--source','gfs'],['--hours','2'],['--history-interval','60']])
def test_native_inputs_refuse_controls_the_runtime_would_ignore(flags):
    with pytest.raises(GoRefusal,match='remove CONFIG|do not consume'):
        launch(args(*flags))
"""A real.exe bundle reaches the same validated forecast owner through go."""
from pathlib import Path
import pytest
from woof.cli import build_parser
from woof.go_cli import _go_wrfinput, GoRefusal


def test_native_bundle_launch_preserves_run_controls(tmp_path, monkeypatch):
    import woof.wrfinput_door as door
    import woof.wrfinput_forecast as forecast
    calls = []
    monkeypatch.setattr(door, "resolve_wrfinput_run", lambda *a, **k: calls.append((a, k)) or SimpleNamespace(experiment=SimpleNamespace(start_time=datetime(2000,1,24,12))))
    monkeypatch.setattr(forecast, "run_wrf_forecast", lambda *a, **k: calls.append((a, k)) or 0)
    args = build_parser().parse_args(["go", "--wrfinput", str(tmp_path), "--outdir", str(tmp_path / "out"),
                                      "--no-supervise", "--products", "none", "--run-stamp", "off"])
    assert _go_wrfinput(args) == 0
    assert len(calls) == 2
    assert calls[1][0] == (tmp_path, tmp_path / "out")
    assert calls[1][1]["exclusive_gpu"] is False
    assert "rrtmg_variant" not in calls[1][1]
    assert calls[1][1]["render_products"] == "none"


@pytest.mark.parametrize("extra", [["--prepare-only"], ["other.toml"], ["--tiles", "on"],
                                    ["--history-interval", "100"], ["--devices", "2"]])
def test_native_bundle_rejects_controls_it_cannot_apply(tmp_path, extra):
    args = build_parser().parse_args(["go", "--wrfinput", str(tmp_path), *extra])
    with pytest.raises(GoRefusal, match="remove CONFIG|do not consume"):
        _go_wrfinput(args)
"""Stock WRF go launches keep the existing native-input execution contract."""
from pathlib import Path
import pytest
from woof import cli


@pytest.mark.parametrize('explicit',[False,True])
def test_go_stock_directory_preserves_restart_checkpoints_and_products(tmp_path,monkeypatch,explicit):
    (tmp_path/'namelist.input').write_text('&physics mp_physics=52 /')
    (tmp_path/'wrfinput_d01').touch()
    calls=[]
    monkeypatch.setattr(cli.capabilities,'require_for_command',lambda *a:None)
    monkeypatch.setattr(cli,'_dispatch',lambda args:calls.append(args) or 0)
    prefix=['go','--wrfinput',str(tmp_path)] if explicit else ['go',str(tmp_path)]
    assert cli.main(prefix+['--outdir',str(tmp_path/'out'),'--restart','checkpoint.nc',
        '--checkpoint-at','end','--products','refc,t2m','--allow-shared-gpu','--no-supervise'])==0
    args=calls[0]
    assert args.command == 'go'
    assert (args.wrfinput == tmp_path and args.config is None) if explicit else (
        args.config == tmp_path and args.wrfinput is None)
    assert args.restart==Path('checkpoint.nc') and args.checkpoint_at=='end'
    assert args.render_products=='refc,t2m' and args.allow_shared_gpu and args.no_supervise


def test_go_toml_keeps_preparation_route(tmp_path,monkeypatch):
    config=tmp_path/'run.toml';config.write_text('[experiment]\n')
    calls=[]
    monkeypatch.setattr(cli.capabilities,'require_for_command',lambda *a:None)
    monkeypatch.setattr(cli,'_dispatch',lambda args:calls.append(args) or 0)
    assert cli.main(['go',str(config),'--products','refc'])==0
    assert calls[0].command=='go' and calls[0].config==config


def test_stock_go_refuses_unused_acquisition_flags(tmp_path,monkeypatch):
    (tmp_path/'namelist.input').write_text('&physics mp_physics=52 /')
    (tmp_path/'wrfinput_d01').touch()
    monkeypatch.setattr(cli,'_dispatch',lambda args:pytest.fail('must refuse before launch'))
    with pytest.raises(SystemExit) as error:
        cli.main(['go',str(tmp_path),'--cycle','2024-05-25T18'])
    assert error.value.code==2


from pathlib import Path
import argparse
import pytest


def arguments(*flags):
    from woof.go_cli import _add_go_arguments
    parser=argparse.ArgumentParser()
    _add_go_arguments(parser)
    args=parser.parse_args(["--run-stamp","off",*map(str, flags)]);args.command="go"
    return args


def test_go_wrfinput_uses_the_existing_stock_namelist_runner(monkeypatch,tmp_path):
    import woof.wrfinput_forecast as runner
    from woof.wrfinput_go import launch
    seen={}
    def run(directory,outdir,**kwargs):
        seen.update(directory=directory,outdir=outdir,**kwargs);return 23
    monkeypatch.setattr(runner,"run_wrf_forecast",run)
    args=arguments("--wrfinput",str(tmp_path/"real"),"--outdir",str(tmp_path/"out"),
                   "--allow-shared-gpu","--no-supervise","--products","2m_temperature")
    assert launch(args)==23
    assert seen["directory"]==tmp_path/"real" and seen["outdir"]==tmp_path/"out"
    assert seen["render_products"]=="2m_temperature"
    assert seen["allow_shared_gpu"] and not seen["exclusive_gpu"]


@pytest.mark.parametrize("flags,match",[
    (("other.toml",),"CONFIG"),
    (("--prepare-only",),"--prepare-only"),
    (("--prepared-root","other"),"--prepared-root"),
    (("--cycle","latest"),"--cycle"),
])
def test_go_wrfinput_refuses_controls_without_consumers(flags,match,tmp_path):
    from woof.go_cli import GoRefusal
    from woof.wrfinput_go import launch
    args=arguments("--wrfinput",str(tmp_path),*flags)
    with pytest.raises(GoRefusal,match=match):launch(args)


from pathlib import Path
import argparse
import pytest

from woof.go_cli import _add_go_arguments, GoRefusal
from woof.go_wrfinput import launch_if_selected


def slab_args(*tokens):
    parser = argparse.ArgumentParser()
    _add_go_arguments(parser)
    return parser.parse_args(["--run-stamp", "off", *tokens])


def test_native_go_uses_production_driver_and_preserves_controls(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("woof.wrfinput_forecast.run_wrf_forecast",
                        lambda *a, **k: calls.append((a, k)) or 0)
    parsed = slab_args("--wrfinput", str(tmp_path), "--outdir", str(tmp_path / "out"),
                  "--checkpoint-at", "end", "--restart", "saved", "--products", "none",
                  "--no-supervise", "--allow-shared-gpu")
    assert launch_if_selected(parsed) == 0
    positional, options = calls[0]
    assert positional == (tmp_path, tmp_path / "out")
    assert options["checkpoint_at"] == "end"
    assert options["restart"] == Path("saved")
    assert options["render_products"] == "none"
    assert options["exclusive_gpu"] is False
    assert options["allow_shared_gpu"] is True


@pytest.mark.parametrize("tokens", [("--cycle", "latest"), ("--prepare-only",),
                                    ("--tiles", "on"), ("--recipe", "surface-state")])
def test_native_go_refuses_unconsumed_controls(tokens):
    with pytest.raises(GoRefusal, match="do not consume"):
        launch_if_selected(slab_args("--wrfinput", "native", *tokens))


@pytest.mark.parametrize("option", [("--checkpoint-at", "end"), ("--run-seconds", "0")])
def test_config_route_unchanged_and_native_options_named(option):
    assert launch_if_selected(slab_args("experiment.toml")) is None
    with pytest.raises(GoRefusal, match="requires --wrfinput"):
        launch_if_selected(slab_args("experiment.toml", *option))


def test_directory_cannot_run_identical_copies_as_an_ensemble():
    with pytest.raises(ValueError, match="one trajectory|copies|members"):
        launch_if_selected(slab_args("--wrfinput", "native", "--members", "2"))


"""Stock WRF go uses the existing input executor and refuses unused flags."""
import os
from pathlib import Path
import pytest

from woof.cli import build_parser
from woof.go_cli import GoRefusal
from woof.wrf_go import run, validate


def gfdl_arguments(*extra):
    return build_parser(go_only=True).parse_args(['go', '--wrfinput', 'native', '--run-stamp', 'off', *extra])


@pytest.mark.parametrize('extra,flag', [
    (['config.toml'], 'CONFIG'),
    (['--prepared-root', 'prepared'], '--prepared-root'),
    (['--cycle', '2026-10-16T09'], '--cycle'),
    (['--tiles', 'off'], '--tiles'),
])
def test_unused_configuration_control_is_refused_before_launch(extra, flag):
    with pytest.raises(GoRefusal, match=flag):
        validate(gfdl_arguments(*extra))


def test_go_passes_stock_input_and_execution_controls_to_its_owner(monkeypatch, tmp_path):
    from woof import wrfinput_forecast
    calls = []
    monkeypatch.setenv('WOOF_VERIFY_VISUALS', '1')
    def launch(directory, outdir, **kwargs):
        calls.append((directory, outdir, kwargs, os.environ['WOOF_VERIFY_VISUALS']))
        return 0
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast', launch)
    args = gfdl_arguments('--outdir', str(tmp_path/'result'), '--run-seconds', '10800',
                     '--no-supervise', '--allow-shared-gpu', '--products', '2m_temperature',
                     '--no-verify-visuals', '--restart', 'checkpoint.npz')
    # Exercise the real decorated public go entry, including its scopes.
    assert args.func(args) == 0
    directory, outdir, kwargs, verify = calls.pop()
    assert directory == Path('native') and outdir == tmp_path/'result'
    assert kwargs == dict(run_seconds=10800., restart=Path('checkpoint.npz'),
                          exclusive_gpu=False, allow_shared_gpu=True,
                          render_products='2m_temperature', benchmark=False)
    assert verify == '0' and os.environ['WOOF_VERIFY_VISUALS'] == '1'


def test_go_inherits_verification_preference(monkeypatch):
    from woof import wrfinput_forecast
    monkeypatch.setenv('WOOF_VERIFY_VISUALS', '0')
    def launch(*args, **kwargs):
        assert os.environ['WOOF_VERIFY_VISUALS'] == '0'
        return 0
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast', launch)
    assert run(gfdl_arguments()) == 0


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_nonpositive_duration_is_refused(value):
    with pytest.raises(GoRefusal, match='positive'):
        validate(gfdl_arguments('--run-seconds', value))


"""A stock namelist directory reaches the same input and render implementation."""
import argparse
import pytest

from woof import go_cli


def _args(directory, *extra):
    parser = argparse.ArgumentParser()
    go_cli._add_go_arguments(parser)
    return parser.parse_args([str(directory), '--run-stamp', 'off', '--no-supervise', *extra])


def test_go_wrf_directory_preserves_render_and_restart_options(tmp_path, monkeypatch):
    from woof import wrfinput_forecast
    (tmp_path / 'namelist.input').write_text('&physics cu_physics=2 /\n')
    calls = []
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *a, **kw: calls.append((a, kw)) or 7)
    args = _args(tmp_path, '--outdir', str(tmp_path / 'out'),
                 '--products', 't2', '--restart', str(tmp_path / 'restart'))
    assert go_cli._go_wrfinput(args) == 7
    assert len(calls) == 1
    positional, controls = calls[0]
    assert positional == (tmp_path, tmp_path / 'out')
    assert controls['restart'] == tmp_path / 'restart'
    assert controls['render_products'] == 't2'
    assert controls['exclusive_gpu'] is False


def test_go_wrf_directory_refuses_unconsumed_acquisition_flag(tmp_path):
    (tmp_path / 'namelist.input').touch()
    with pytest.raises(go_cli.GoRefusal, match='--hours'):
        go_cli._go_wrfinput(_args(tmp_path, '--hours', '7'))


def test_go_toml_keeps_its_existing_route(tmp_path, monkeypatch):
    from woof import cli
    config = tmp_path / 'experiment.toml'
    config.write_text('[experiment]\nname = "route-test"\n')
    calls = []
    monkeypatch.setattr(cli.capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(cli, '_dispatch', lambda args: calls.append(args) or 0)
    assert cli.main(['go', str(config)]) == 0
    assert calls[0].config == config and calls[0].wrfinput is None


def test_go_wrf_dry_run_launches_nothing(tmp_path, capsys, monkeypatch):
    import json
    from types import SimpleNamespace
    from datetime import datetime
    from woof import wrfinput_door, wrfinput_forecast
    (tmp_path / 'namelist.input').touch()
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda *a, **k:
        SimpleNamespace(experiment=SimpleNamespace(start_time=datetime(2000, 1, 24, 12),
                                                  run_seconds=120),
                        grid_ids=(1,), namelist_input=tmp_path / 'namelist.input'))
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *a, **k: pytest.fail('dry run launched forecast'))
    assert go_cli._go_wrfinput(_args(tmp_path, '--dry-run', '--products', 't2')) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['route'] == 'native-wrf' and report['run_seconds'] == 120
    assert report['input'] == str(tmp_path)
