"""The ordinary go door preserves its public parser without unrelated imports."""
import argparse
import pytest
from woof import cli


def command(parser):
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices['go']


def test_go_help_defaults_and_options_match_full_surface():
    full, focused = cli.build_parser(), cli.build_parser(go_only=True)
    assert command(full).format_help() == command(focused).format_help()
    for argv in [
        ['go', 'forecast.toml'],
        ['go', 'forecast.toml', '--prepared-root', '/prepared', '--restart', '/restart', '--wps-namelist', '/wps', '--devices', '1', '--products', 'none', '--run-stamp', 'off'],
        ['go', 'forecast.toml', '--cycle', '2026-02-28T00', '--whole-cycle', '--prepare-only', '--explain'],
        ['go', 'forecast.toml', '--dry-run', '--data-dir', '/forcing', '--geog-root', '/geog'],
        ['go', 'forecast.toml', '--readiness'],
    ]:
        expected = vars(full.parse_args(argv))
        expected.pop('ingest_preflight_handler', None)  # Check-only default leaks from its registrar.
        assert expected == vars(focused.parse_args(argv))


@pytest.mark.parametrize('tail', [['forecast.toml', '--devices', 'bad'], ['forecast.toml', '--unknown-option']])
def test_go_refusal_matches_full_surface(tail, capsys):
    messages=[]
    for parser in [cli.build_parser(), cli.build_parser(go_only=True)]:
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(['go', *tail])
        assert stopped.value.code == 2
        messages.append(capsys.readouterr().err)
    assert messages[0] == messages[1]


def test_go_dispatch_uses_focused_parser(monkeypatch, capsys):
    original=cli.build_parser
    calls=[]
    def build(**kwargs):
        calls.append(kwargs)
        assert kwargs == {'go_only': True}
        return original(**kwargs)
    monkeypatch.setattr(cli, 'build_parser', build)
    with pytest.raises(SystemExit) as stopped:
        cli.main(['go', '--help'])
    assert stopped.value.code == 0
    assert calls == [{'go_only': True}]
    assert '--prepared-root' in capsys.readouterr().out


@pytest.mark.parametrize('selector', [['--wrfinput', '/case'], ['--wrfinput=/case']])
def test_go_stock_wrf_inputs_reach_the_existing_run_parser(selector, monkeypatch, capsys):
    original = cli.build_parser
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(cli, 'build_parser', build)
    with pytest.raises(SystemExit) as stopped:
        cli.main(['go', *selector, '--help'])
    assert stopped.value.code == 0
    assert calls == [{'go_only': True}]
    assert '--restart-from' in capsys.readouterr().out
    with pytest.raises(SystemExit) as stopped:
        cli.main(['go', *selector, '--unknown-option'])
    assert stopped.value.code == 2
    assert '--unknown-option' in capsys.readouterr().err
