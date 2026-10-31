"""``woof hex --help`` lists every door in an install without the engine, and an engine door refuses by name.

THE BREAKAGE THIS HOLDS SHUT (hex 0.4.10 gate, HXCI packaging job, 2026-10-29): the doors that take
``--output-format`` import ``woof.output_formats`` while their parsers are built, so the public CI packaging
job (the built wheel installed with ``--no-deps``) stopped at ``woof hex --help`` with ModuleNotFoundError.
"""

from __future__ import annotations

import sys

import pytest

from woof.hex import cli


@pytest.fixture
def no_engine(monkeypatch):
    """Imports of the engine fail as they do where it is not installed."""
    for name in [n for n in sys.modules if n == "woof" or n.startswith("woof.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "woof", None)
    monkeypatch.setitem(sys.modules, "woof.output_formats", None)


def _choices(parser):
    return dict(parser._subparsers._group_actions[0].choices)


def test_help_lists_every_door_without_the_engine(no_engine, capsys):
    names = {name for row in cli._DOORS for name in row[0]}
    assert names <= set(_choices(cli.build_parser()))
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    assert "country" in capsys.readouterr().out


def test_an_engine_door_refuses_by_name_without_the_engine(no_engine, capsys):
    assert cli.main(["country", "--row", "x", "--cycle", "2026-10-29T12"]) == 2
    error = capsys.readouterr().err
    assert "country door needs the woof engine" in error and "woof>=2.9.10" in error


def test_an_engine_free_door_still_builds_alone_without_the_engine(no_engine):
    assert "version" in _choices(cli.build_parser("version"))


def test_a_missing_module_that_is_not_the_engine_still_raises(monkeypatch):
    def broken(commands):
        raise ModuleNotFoundError("No module named 'somethingelse'", name="somethingelse")
    monkeypatch.setattr(cli, "_DOORS", ((("broken",), broken),) + cli._DOORS)
    with pytest.raises(ModuleNotFoundError):
        cli.build_parser()
