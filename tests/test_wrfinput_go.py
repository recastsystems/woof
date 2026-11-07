"""The go directory route retains native ownership and never drops flags."""
from pathlib import Path
from types import SimpleNamespace
import datetime
import pytest
from woof.cli import build_parser
from woof.go_cli import GoRefusal, go_main


def test_go_wrfinput_delegates_the_real_runner(monkeypatch):
    from woof import wrfinput_forecast as runner, wrfinput_door
    calls = []
    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run", lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime.datetime(2023, 3, 31))))
    monkeypatch.setattr(runner, "run_wrf_forecast",
                        lambda *a, **kw: calls.append((a, kw)) or 0)
    args = build_parser(go_only=True).parse_args([
        "go", "--wrfinput", "native", "--outdir", "forecast",
        "--allow-shared-gpu", "--products", "none", "--run-stamp", "off"])
    args.interactive = True  # CLI dispatch annotates interactive terminals.
    assert go_main(args) == 0
    assert calls == [((Path("native"), Path("forecast")),
                     dict(restart=None, allow_shared_gpu=True, render_products="none",
                          exclusive_gpu=True, benchmark=False))]


def test_public_go_wrfinput_preserves_product_controls(monkeypatch, tmp_path):
    from woof import cli, capabilities, provenance_gate
    seen = []
    monkeypatch.setattr(provenance_gate, "announce", lambda *args: None)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *args: None)
    monkeypatch.setattr(cli, "_dispatch", lambda args: seen.append(args) or 0)
    assert cli.main(["go", "--wrfinput", str(tmp_path), "--products", "none",
                     "--no-verify-visuals", "--allow-shared-gpu"]) == 0
    args, = seen
    assert args.command == "go" and args.wrfinput == tmp_path
    assert args.render_products == "none" and args.verify_visuals is False
    assert args.allow_shared_gpu


@pytest.mark.parametrize('mode', ['flag', 'directory'])
def test_public_native_go_applies_explain_and_history_controls(mode, monkeypatch, tmp_path, capsys):
    import os
    from woof import cli, capabilities, provenance_gate, wrfinput_door, wrfinput_forecast
    from woof.explain import warn
    (tmp_path / 'namelist.input').write_text('&time_control /\n')
    (tmp_path / 'wrfinput_d01').write_bytes(b'')
    monkeypatch.setattr(provenance_gate, 'announce', lambda *args: None)
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *args: None)
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime.datetime(2023, 3, 31))))
    monkeypatch.setenv('GPUWM_HISTORY_COMPRESSION', 'deflate')
    seen = []
    def run(*args, **kwargs):
        warn('control scope probe', 'explain scope reached native runner')
        seen.append(('explain scope reached native runner' in capsys.readouterr().err,
                     os.environ['GPUWM_HISTORY_COMPRESSION']))
        return 0
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast', run)
    source = ['--wrfinput', str(tmp_path)] if mode == 'flag' else [str(tmp_path)]
    assert cli.main(['go', *source, '--run-stamp', 'off', '--no-verify-visuals',
                     '--products', 'none', '--explain', '--history-compression', 'none']) == 0
    assert seen == [(True, 'none')]


@pytest.mark.parametrize("extra", [["--prepared-root", "wrong"],
                                  ["--cycle", "2000-01-24T12"],
                                  ["--prepare-only"], ["another.toml"]])
def test_go_wrfinput_never_silently_drops_an_input_control(extra):
    args = build_parser(go_only=True).parse_args(["go", "--wrfinput", "native", *extra])
    with pytest.raises(GoRefusal):
        go_main(args)
def arguments(directory, *flags):
    return build_parser(go_only=True).parse_args(["go", str(directory), *flags])


def test_native_route_preserves_restart_products_and_output_owner(tmp_path, monkeypatch):
    from woof import wrfinput_forecast,wrfinput_door
    seen=[]
    monkeypatch.setattr(wrfinput_door,"resolve_wrfinput_run",lambda _:SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime.datetime(2023,3,31))))
    monkeypatch.setattr(wrfinput_forecast,"run_wrf_forecast",
                        lambda source,target,**kw: seen.append((source,target,kw)) or 7)
    args=arguments(tmp_path,"--outdir",str(tmp_path/"out"),"--no-supervise",
                   "--products","none","--restart","checkpoint","--run-stamp","off")
    # Dispatch adds this context field after argparse. It is not a CLI setting.
    args.interactive=False
    assert go_main(args)==7
    assert seen==[(tmp_path,tmp_path/"out",dict(restart=Path("checkpoint"),
                  render_products="none",exclusive_gpu=False,allow_shared_gpu=False,
                  benchmark=False))]


@pytest.mark.parametrize("flags",[("--prepare-only",),("--cycle","latest"),
                                 ("--tiles","on"),])
def test_native_route_refuses_settings_it_cannot_consume(tmp_path, monkeypatch, flags):
    from woof import wrfinput_forecast
    monkeypatch.setattr(wrfinput_forecast,"run_wrf_forecast",lambda *a,**kw:pytest.fail("must refuse before launch"))
    with pytest.raises(GoRefusal,match="would ignore the requested setting"):
        go_main(arguments(tmp_path,*flags))


def test_config_route_does_not_accept_native_only_worker_control(tmp_path):
    config=tmp_path/"run.toml"
    config.write_text("")
    with pytest.raises(GoRefusal,match="configuration launches retain"):
        go_main(arguments(config,"--no-supervise"))


def test_metem_go_keeps_duration_members_and_shared_supervision(tmp_path, monkeypatch):
    from woof import cli
    (tmp_path / "namelist.input").write_text("&time_control /\n")
    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 0)
    assert go_main(arguments(tmp_path, "--run-seconds", "120", "--members", "2",
                             "--keep-member-files", "--allow-shared-gpu")) == 0
    forwarded = cli.build_parser().parse_args(calls[0])
    assert forwarded.met_em == tmp_path
    assert forwarded.run_seconds == 120
    assert forwarded.members == 2 and forwarded.keep_member_files
    assert forwarded.allow_shared_gpu and not forwarded.no_supervise


@pytest.mark.parametrize("flags", [("--benchmark",), ("--dry-run",),
                                 ("--run-stamp", "off")])
def test_metem_go_refuses_controls_absent_from_its_runner(tmp_path, flags):
    (tmp_path / "namelist.input").write_text("&time_control /\n")
    with pytest.raises(GoRefusal, match="would ignore the requested setting"):
        go_main(arguments(tmp_path, *flags))


"""Native WRF go selection reaches the real runner and consumes its controls."""
from pathlib import Path
from unittest.mock import patch
import pytest
from woof import cli
from woof.go_cli import GoRefusal, go_main
from woof.wrfinput_go import run_wrf_go, refuse_unrouted_go_controls


def args(*tail):
    return cli.build_parser(go_only=True).parse_args(['go', *tail])


def test_public_go_dispatches_native_inputs_and_checkpoint_controls(monkeypatch, tmp_path):
    monkeypatch.setattr('woof.capabilities.require_for_command', lambda command: None)
    inputs = tmp_path / 'inputs'
    output = tmp_path / 'output'
    monkeypatch.setattr('woof.wrfinput_door.resolve_wrfinput_run', lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime.datetime(2023, 3, 31))))
    with patch('woof.wrfinput_forecast.run_wrf_forecast', return_value=0) as run:
        result = cli.main(['go', '--wrfinput', str(inputs), '--outdir', str(output),
                           '--run-stamp', 'off', '--allow-shared-gpu', '--benchmark',
                           '--run-seconds', '10800', '--checkpoint-at', 'end',
                           '--restart', str(tmp_path / 'gpuwmrst')])
    assert result == 0
    positional, controls = run.call_args
    assert positional == (inputs, output)
    assert controls['run_seconds'] == 10800
    assert controls['checkpoint_at'] == 'end'
    assert controls['restart'] == tmp_path / 'gpuwmrst'
    assert controls['benchmark'] is True and controls['allow_shared_gpu'] is True
    assert not output.exists()  # the real runner owns output admission


@pytest.mark.parametrize('tail, name', [
    (['forecast.toml'], 'exactly one'),
    (['--prepared-root', '/bundle'], '--prepared-root'),
    (['--source', 'gfs'], '--source'),
    (['--devices', '2'], '--devices'),
    (['--prepare-only'], '--prepare-only'),
    (['--cyclone-products', '/request.json'], '--cyclone-products'),
])
def test_native_input_route_refuses_unused_controls_before_loading_inputs(tail, name):
    with pytest.raises(GoRefusal, match=name):
        go_main(args('--wrfinput', '/absent-inputs', *tail))


def test_config_route_refuses_a_native_only_control():
    with pytest.raises(ValueError, match='--benchmark.*would not consume'):
        refuse_unrouted_go_controls(args('forecast.toml', '--benchmark'))


def test_native_dry_run_checks_window_without_claiming_output(tmp_path):
    namespace = args('--wrfinput', '/inputs', '--outdir', str(tmp_path/'out'), '--dry-run', '--run-seconds', '10800')
    with patch('woof.wrfinput_door.resolve_wrfinput_run', return_value=SimpleNamespace(
            experiment=SimpleNamespace(start_time=datetime.datetime(2023, 3, 31)),
            grid_ids=[1], namelist_input=Path('/inputs/namelist.input'))) as resolve:
        with patch('woof.wrfinput_forecast.wrfinput_window_seconds', return_value=10800):
            with patch('woof.wrfinput_forecast.run_wrf_forecast') as run:
                assert run_wrf_go(namespace) == 0
    resolve.assert_called_once()
    run.assert_not_called()
    assert not (tmp_path/'out').exists()


"""The stock namelist go route uses the ordinary runtime and flag contracts."""
from types import SimpleNamespace
from pathlib import Path

import pytest
from woof import wrfinput_go
from woof.go_cli import GoRefusal


@pytest.mark.parametrize('key,flag', [('source', '--source'), ('config', 'CONFIG'),
                                    ('devices', '--devices'), ('render_section', '--section')])
def test_stock_route_names_unconsumed_flags(key, flag):
    args = build_parser(go_only=True).parse_args(['go', '--wrfinput', '.', '--run-stamp', 'off'])
    setattr(args, key, 'requested')
    with pytest.raises(GoRefusal, match=flag):
        wrfinput_go.launch(args)


def test_stock_route_keeps_one_input_ensemble_admission():
    args = build_parser(go_only=True).parse_args(['go', '--wrfinput', '.', '--members', '3'])
    with pytest.raises(ValueError, match='one trajectory'):
        wrfinput_go.launch(args)


def test_stock_route_forwards_checkpoint_restart_and_shared_card(monkeypatch, tmp_path):
    import woof.wrfinput_door as door
    import woof.wrfinput_forecast as runtime
    monkeypatch.setattr(door, 'resolve_wrfinput_run', lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime.datetime(2023, 3, 31))))
    seen = {}
    def run(directory, outdir, **kwargs):
        seen.update(directory=directory, outdir=outdir, **kwargs)
        return 0
    monkeypatch.setattr(runtime, 'run_wrf_forecast', run)
    args = build_parser(go_only=True).parse_args([
        'go', '--wrfinput', str(tmp_path), '--outdir', str(tmp_path/'out'),
        '--products', 'none', '--restart', str(tmp_path/'checkpoint'),
        '--run-seconds', '120', '--checkpoint-at', '2024-05-25_20:00:00',
        '--allow-shared-gpu', '--no-verify-visuals', '--no-supervise', '--run-stamp', 'off'])
    assert wrfinput_go.launch(args) == 0
    assert seen['directory'] == tmp_path
    for name in ('restart', 'run_seconds', 'checkpoint_at', 'allow_shared_gpu'):
        assert seen[name] == getattr(args, name)
    assert seen['exclusive_gpu'] is False
