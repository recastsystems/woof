"""The door table behind ``build_parser(command)`` and the once-per-file NVRTC digest.

THE BREAKAGE THESE HOLD SHUT (member-leg start-up measure, RTX 5090,
2026-10-28): every ``woof hex forecast`` imported all 36 doors before its own
opened, every spawn compile worker re-imported the forecast, init and cull
doors through ``__mp_main__``, and one forecast start hashed the 115 MB NVRTC
library 8-12 times on its main thread.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from woof.hex import cli


def _commands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    return dict(parser._subparsers._group_actions[0].choices)


def _word(value) -> str:
    """A value as parse_args sees it; a function (a handler, a ``type=`` converter) by its qualified name,
    since an adder that defines one inside itself makes a new object, with a new address, per build."""

    if callable(value) and hasattr(value, "__qualname__"):
        return f"{getattr(value, '__module__', '')}.{value.__qualname__}"
    return repr(value)


def _shape(parser: argparse.ArgumentParser):
    """Everything parse_args reads from a door's parser, recursively through its own sub-commands."""

    rows = []
    for action in parser._actions:
        row = (type(action).__name__, tuple(action.option_strings), action.dest, _word(action.default),
               _word(action.type), repr(action.choices) if not isinstance(action, argparse._SubParsersAction) else None,
               repr(action.nargs), action.required, _word(action.const), action.help)
        rows.append(row)
        if isinstance(action, argparse._SubParsersAction):
            rows.append(tuple((name, _shape(child)) for name, child in action.choices.items()))
    return rows, sorted((key, _word(value)) for key, value in parser._defaults.items())


def test_every_row_registers_exactly_its_commands_in_the_full_order():
    full = list(_commands(cli.build_parser()))
    assert full == [name for names, _add in cli._DOORS for name in names]
    for names, add in cli._DOORS:
        alone = argparse.ArgumentParser(prog="woof hex").add_subparsers(dest="command", required=True)
        add(alone)
        assert tuple(alone.choices) == names


@pytest.mark.parametrize("command", [name for names, _add in cli._DOORS for name in names])
def test_a_command_built_alone_is_the_same_parser(command):
    alone = _commands(cli.build_parser(command))
    assert command in alone
    assert _shape(alone[command]) == _shape(_commands(cli.build_parser())[command])


def test_unknown_or_option_first_words_build_every_door():
    every = list(_commands(cli.build_parser()))
    assert list(_commands(cli.build_parser("no-such-door"))) == every
    assert cli._command_word(["--help"]) is None and cli._command_word([]) is None
    assert cli._command_word(["forecast", "--hours", "1"]) == "forecast"


def test_forecast_arguments_parse_identically():
    argv = ["forecast", "--hours", "1", "--dt", "18.0", "--out", "fc", "--physics-mode", "correct",
            "--no-products", "--iau-window-seconds", "600.0", "--balance-series"]
    alone = vars(cli.build_parser("forecast").parse_args(argv))
    full = vars(cli.build_parser().parse_args(argv))
    assert {key: repr(value) for key, value in alone.items()} == {key: repr(value) for key, value in full.items()}


def test_importing_the_cli_imports_no_door():
    code = ("import sys, json, woof.hex.cli; "
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('hexcore.'))))")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(Path(cli.__file__).parents[1]),
                                                                       os.environ.get("PYTHONPATH")])))
    loaded = json.loads(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                       check=True, timeout=120).stdout.strip().splitlines()[-1])
    doors = {"woof.hex.forecast_door", "woof.hex.init_door", "woof.hex.cull_door", "woof.hex.country_door",
             "woof.hex.ensemble", "woof.hex.global_door", "woof.hex.swath.door", "woof.hex.cycle.door"}
    assert not doors & set(loaded), loaded


def test_checkout_defaults_are_read_on_first_use():
    from woof.hex.forecast_door import PROJECT_ROOT

    assert cli.PROJECT_ROOT == PROJECT_ROOT
    expected = None if PROJECT_ROOT is None else PROJECT_ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.grid.nc"
    assert cli.DEFAULT_GRID == expected
    mesh_check = _commands(cli.build_parser("mesh-check"))["mesh-check"]
    assert mesh_check.get_default("grid") == expected


def test_nvrtc_library_is_hashed_once_per_file(tmp_path, monkeypatch):
    compile_platform = pytest.importorskip("woof.certify.compile_platform")
    from woof.hex.cuda_backend import compile_contract

    library = tmp_path / "libnvrtc.so.13"
    library.write_bytes(b"first")
    reads = []

    def nvrtc_library():
        reads.append(1)
        data = library.read_bytes()
        import hashlib

        return {"status": "resolved", "path": str(library), "sha256": hashlib.sha256(data).hexdigest()}

    nvrtc_library.__module__ = compile_platform.__name__
    monkeypatch.setattr(compile_platform, "_loaded_library_path", lambda: str(library))
    monkeypatch.setattr(compile_platform, "nvrtc_library", nvrtc_library)
    compile_contract.hash_nvrtc_library_once()
    first = compile_platform.nvrtc_library()
    assert compile_platform.nvrtc_library() == first and len(reads) == 1
    compile_contract.hash_nvrtc_library_once()  # idempotent: never wraps twice
    assert compile_platform.nvrtc_library.__wrapped__ is nvrtc_library
    library.write_bytes(b"replaced bytes")  # a rewritten file: new size and ctime, hashed again
    second = compile_platform.nvrtc_library()
    assert len(reads) == 2 and second["sha256"] != first["sha256"]


def test_a_stand_in_nvrtc_library_is_left_alone(monkeypatch):
    compile_platform = pytest.importorskip("woof.certify.compile_platform")
    from woof.hex.cuda_backend import compile_contract

    stand_in = lambda: {"status": "resolved", "sha256": "a" * 64}  # noqa: E731
    monkeypatch.setattr(compile_platform, "nvrtc_library", stand_in)
    compile_contract.hash_nvrtc_library_once()
    assert compile_platform.nvrtc_library is stand_in
