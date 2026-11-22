"""Correct mode is the preset rows' default; exact stays one flag away.

Correct mode is the standard. Every shipped forecast preset row
(woof.hex.forecast_preset) declares physics.mode "correct", and the forecast
door takes the row's mode when the command line names none.

THE BREAKAGES THESE TESTS HOLD SHUT:
* a preset run silently getting the matched (uncorrected) arithmetic because
  the door's own default stayed exact while the ruling says correct;
* an explicit ``--physics-mode exact`` being overridden by the preset row
  (the matched configuration must stay one flag away, byte for byte);
* the preset receipt claiming a mode the run did not use (the run's own mode
  is recorded by the door, and only in correct mode).
"""
from __future__ import annotations

import argparse

from woof.hex import correct_mode, forecast_door, forecast_preset


def test_every_shipped_preset_row_is_correct_mode():
    for name in forecast_preset.preset_names():
        row = forecast_preset.resolve_preset(name)
        assert row.physics_mode == correct_mode.PHYSICS_MODE_CORRECT, name
        assert "physics_mode" not in row.receipt(), name


def test_the_door_takes_the_preset_mode_when_none_is_named():
    row = forecast_preset.resolve_preset(None)
    named_none = argparse.Namespace(physics_mode=None, physics_correct_skip=None)
    assert forecast_door._physics_mode_fields(named_none, row) == {
        "physics_mode": "correct", "physics_correct_skip": ()}


def test_an_explicit_exact_wins_over_the_preset():
    row = forecast_preset.resolve_preset(None)
    exact = argparse.Namespace(physics_mode="exact", physics_correct_skip=None)
    assert forecast_door._physics_mode_fields(exact, row) == {
        "physics_mode": "exact", "physics_correct_skip": ()}
    # exact hands the driver no flag at all: the argument vector of before
    assert correct_mode.door_argv("exact") == []


def test_the_command_line_default_is_the_preset_row():
    parser = forecast_door.build_parser() if hasattr(forecast_door, "build_parser") else None
    if parser is None:
        import inspect
        source = inspect.getsource(forecast_door)
        assert ('"--physics-mode", choices=("exact", "correct"), default=None' in source
                or '"--physics-mode", choices=("correct", "exact"), default=None' in source)
        assert source.count('"--physics-mode", choices=') == 1
        return
    actions = {a.dest: a for a in parser._actions}
    assert actions["physics_mode"].default is None


def test_a_row_with_an_unknown_mode_is_refused():
    try:
        forecast_preset.ForecastPreset(
            name="typo", summary="x", suite=forecast_preset.ENGINE_COLUMN_SUITE,
            surface_pbl_hold_seconds=None, evidence="x", physics_mode="Correct")
    except ValueError as error:
        assert "physics_mode" in str(error)
    else:
        raise AssertionError("a misspelled physics_mode must be refused")
