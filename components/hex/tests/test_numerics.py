"""The numerics knob (owner ruling 2026-10-16: strict numerics is opt-in, never the default).

CPU only.  The battery runs strict (tests/conftest.py); these tests clear that and check the door default,
the option mapping, the one-mode-per-process refusal and the proof harness's appended flag.
"""
from __future__ import annotations

import pytest

from woof.hex import numerics


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.delenv(numerics.ENV, raising=False)
    numerics._reset_for_tests()
    yield numerics
    numerics._reset_for_tests()


def test_default_is_fast(fresh):
    assert fresh.mode() == "fast"
    assert fresh.resolve(None) == ("fast", "default")
    assert fresh.kernel_base() == ("--std=c++17", "--fmad=true", "-DMPAS_FTZ_FALLBACK_ENABLED=0",
                                   "-DHEX_NUMERICS_FAST=1")


def test_strict_is_byte_identical_to_the_proved_tuples(fresh):
    strict = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
              "--ptxas-options=-O0")
    assert fresh.options(strict, numerics="strict") == strict
    assert fresh.kernel_base("strict") == ("--std=c++17", "--fmad=false")


def test_fast_options(fresh):
    strict = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
              "--ptxas-options=-O0", "-DRUC_ICE_ENTRY_CONTEXT")
    fast = fresh.options(strict)
    assert fast == ("--std=c++17", "-DRUC_ICE_ENTRY_CONTEXT", "--fmad=true", "--ftz=true", "--prec-div=true",
                    "--prec-sqrt=true", "-DMPAS_FTZ_FALLBACK_ENABLED=0", "-DHEX_NUMERICS_FAST=1")
    land = fresh.options(strict, keep_ftz=True)
    assert "--ftz=false" in land and "--ftz=true" not in land and "--fmad=true" in land
    # the central base plus strict extras (the carrier caches) does not duplicate flags or defines
    combined = fresh.options((*fresh.kernel_base(), "--fmad=false", "--ftz=false"))
    assert len(combined) == len(set(combined))
    assert not any("fast_math" in option for option in combined)


def test_one_mode_per_process(fresh):
    fresh.options(("--fmad=false",))  # a compile under fast
    with pytest.raises(fresh.NumericsError):
        fresh.set_mode("strict")
    assert fresh.set_mode("fast") == "fast"


def test_fast_preparation_identity_retains_unfused_arithmetic(fresh):
    flags = fresh.options((*fresh.kernel_base(), "-DGPUWM_PREPARATION_IDENTITY=1", "--fmad=false"))
    assert "--fmad=false" in flags and "--fmad=true" not in flags
    assert "-DHEX_NUMERICS_FAST=1" in flags
    assert "--fmad=true" in fresh.options(fresh.kernel_base())


def test_resolve_cli_namelist_env(fresh, monkeypatch):
    namelist = "&nhyd_model\n    config_dt = 600.0\n    config_numerics = 'strict'\n/\n"
    assert fresh.resolve(None, namelist) == ("strict", "namelist config_numerics")
    assert fresh.resolve("strict", namelist) == ("strict", "--numerics")
    with pytest.raises(fresh.NumericsError):
        fresh.resolve("fast", namelist)
    monkeypatch.setenv(fresh.ENV, "strict")
    assert fresh.resolve(None) == ("strict", fresh.ENV)
    with pytest.raises(fresh.NumericsError):
        fresh.normalise("fastest")


def test_cli_flag_on_the_forecast_door(fresh):
    import argparse
    from woof.hex.cli import build_parser, _select_numerics
    commands = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    for door in ("forecast", "init"):
        if door in commands.choices:
            assert any("--numerics" in a.option_strings for a in commands.choices[door]._actions), door
    arguments = argparse.Namespace(numerics="strict")
    _select_numerics(arguments)
    assert fresh.mode() == "strict" and arguments.numerics_source == "--numerics"


def test_proof_harness_appends_strict():
    from woof.hex.verify_mpas import proof_numerics
    command = ["woof hex", "forecast", "--out", "{out}"]
    run, record = proof_numerics(command)
    assert run[-2:] == ["--numerics", "strict"] and record["appended"]
    run, record = proof_numerics([*command, "--numerics", "fast"])
    assert run == [*command, "--numerics", "fast"] and record["mode"] == "fast" and not record["appended"]


def test_namelist_key_is_taken_out_before_native(fresh):
    from woof.hex.mpasphys.forecast_v842 import split_physics_mode
    text, mode, _ = split_physics_mode("&nhyd_model\nconfig_dt=600.0\nconfig_numerics='strict'\n/\n")
    assert "config_numerics" not in text and mode is None


def test_strict_receipt_requests_but_does_not_claim_proof(fresh):
    fresh.set_mode("strict")
    receipt = fresh.receipt_block()
    assert receipt["bitwise_proof_requested"]
    assert receipt["bitwise_to_native"] is False


def test_native_parser_never_receives_numerics_in_other_group(fresh):
    from woof.hex.mpasphys.forecast_v842 import split_physics_mode
    text, mode, _ = split_physics_mode("&physics\nconfig_numerics='fast'\n/\n")
    assert 'config_numerics' not in text and mode is None
