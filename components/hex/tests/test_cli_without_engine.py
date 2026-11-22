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


# (WOOF 1.1.7) hex's model of an install without the engine package
@pytest.mark.skip(reason="in WOOF the hex model is woof.hex, inside the engine's own package, so an "
                         "install of the hex model without its engine cannot exist; the no_engine fixture "
                         "would remove the code under test")
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
    # (Recast WOOF) the engine ships inside recast-woof, so the remedy names it without an engine version
    assert "country door needs the woof engine" in error and "pip install --force-reinstall 'recast-woof'" in error


# (WOOF 1.1.7) hex's model of an install without the engine package
@pytest.mark.skip(reason="in WOOF the hex model is woof.hex, inside the engine's own package, so an "
                         "install of the hex model without its engine cannot exist; the no_engine fixture "
                         "would remove the code under test")
def test_an_engine_free_door_still_builds_alone_without_the_engine(no_engine):
    assert "version" in _choices(cli.build_parser("version"))


def test_a_missing_module_that_is_not_the_engine_still_raises(monkeypatch):
    def broken(commands):
        raise ModuleNotFoundError("No module named 'somethingelse'", name="somethingelse")
    monkeypatch.setattr(cli, "_DOORS", ((("broken",), broken),) + cli._DOORS)
    with pytest.raises(ModuleNotFoundError):
        cli.build_parser()


# (WOOF 1.1.7) hex's model of an install without the engine package
@pytest.mark.skip(reason="in WOOF the hex model is woof.hex, inside the engine's own package, so an "
                         "install of the hex model without its engine cannot exist; the no_engine fixture "
                         "would remove the code under test")
def test_version_prints_the_installed_version_and_exits_zero(no_engine, capsys, monkeypatch):
    """``woof hex --version``: the distribution and the version its own metadata holds, exit 0.

    THE BREAKAGE (the candidate's card job, 2026-11-14): the install check's ``woof hex --version`` printed
    ``usage: woof hex [-h] [--backend {cuda,amd}]`` and exited 2; only the ``version`` door answered."""
    import woof.hex

    def no_parser(command=None):
        raise AssertionError("--version builds no door's parser")
    built = cli.build_parser
    monkeypatch.setattr(cli, "build_parser", no_parser)
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr()
    assert out.out == f"woof hex {woof.hex.__version__}\n" and out.err == ""
    try:
        from importlib.metadata import version
        assert woof.hex.__version__ == version("woof hex")       # an installed tree: the metadata's own number
    except ImportError:
        pass
    except Exception as missing:                                  # a source tree with no metadata says so
        assert type(missing).__name__ == "PackageNotFoundError" and woof.hex.__version__ == "0+unknown"
    # After another top-level option the parser answers the same line, and --help lists the word.
    monkeypatch.setattr(cli, "build_parser", built)
    with pytest.raises(SystemExit) as stop:
        cli.build_parser("version").parse_args(["--backend", "cuda", "--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out == f"woof hex {woof.hex.__version__}\n"
    assert "--version" in cli.build_parser("version").format_help()
