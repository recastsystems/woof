"""Bundle transport and real forecast configuration parsing, without model execution."""
from __future__ import annotations

import argparse
import json

import pytest

from woof.hex import analysis_bundle as ab
from woof.hex.analysis_door import _apply_forecast_defaults
from woof.hex.forecast_door import add_forecast_arguments
from woof.hex.mpasphys.forecast_v842 import _native_route
from woof.hex.stochastic import StochasticConfig

from test_analysis_bundle import files, make  # noqa: F401


STOCHASTIC = '{"sppt":true, "skeb":true, "seed":53, "member":3}'
PIN = ["--physics-suite", "hrrrv5", "--physics-pin-set", "rrfs-44c0849",
       "--physics-default-backend", "hex-cuda", "--stochastic", STOCHASTIC]
PROFILE = ["--physics-suite", "convection_permitting", "--physics-profile", "hrrr-like",
           "--physics-default-backend", "hex-cuda", "--stochastic", STOCHASTIC]
ALL_CONTROLS = PROFILE + ["--physics-pin-set", "rrfs-44c0849"]


def arguments(flags=()):
    parser = argparse.ArgumentParser()
    add_forecast_arguments(parser)
    return parser.parse_args(flags)


@pytest.mark.parametrize("controls", [PIN, PROFILE, ALL_CONTROLS])
def test_controls_roundtrip_through_manifest_and_forecast_parser(tmp_path, files, controls):
    bundle = make(tmp_path, files, forecast_args=controls)
    loaded = ab.load_bundle(bundle.root)
    assert loaded.manifest["forecast"]["args"] == controls
    assert [item for pair in ab.parse_forecast_args(controls) for item in pair if item is not None] == controls
    run = arguments()
    receipt = _apply_forecast_defaults(run, loaded)
    assert receipt["run_named"] == []
    for flag, value in ab.parse_forecast_args(controls):
        assert getattr(run, flag[2:].replace("-", "_")) == value
    assert run.stochastic == STOCHASTIC  # preserve JSON bytes, not only decoded values
    config = StochasticConfig.parse(run.stochastic)
    assert config.sppt and config.skeb and config.member == 3 and config.seed == 53
    assert config.member_seed == StochasticConfig.parse(STOCHASTIC).member_seed


@pytest.mark.parametrize("controls", [PIN, PROFILE])
def test_carried_controls_reach_the_real_physics_route(tmp_path, files, controls):
    run = arguments()
    _apply_forecast_defaults(run, make(tmp_path, files, forecast_args=controls))
    config, _ = _native_route(run, default_mode="correct")
    assert all(str(config.backends[slot]) == "hex-cuda"
               for slot in config.selectors if config.active(slot))
    if controls == PIN:
        assert config.pin_set == "rrfs-44c0849" and config.table_key == "mpasphys_rrfs_44c0849"
    else:
        assert config.options["hex_physics_profile"] == "hrrr-like"
        assert config.selectors["microphysics"] == "mp_thompson_aerosols"


def test_explicit_run_controls_win_without_mutating_the_bundle(tmp_path, files):
    bundle = make(tmp_path, files, forecast_args=ALL_CONTROLS)
    before = ab.manifest_bytes(bundle.manifest)
    explicit = ["--physics-pin-set", "rrfs", "--physics-profile", "hrrr-like-aerosol",
                "--physics-default-backend", "native-cpu", "--stochastic",
                '{"sppt":true,"skeb":true,"seed":71,"member":9}']
    run = arguments(explicit)
    receipt = _apply_forecast_defaults(run, bundle)
    for flag, value in ab.parse_forecast_args(explicit):
        assert getattr(run, flag[2:].replace("-", "_")) == value
        assert flag[2:].replace("-", "_") in receipt["run_named"]
    assert ab.manifest_bytes(bundle.manifest) == before


def test_absent_controls_do_not_change_defaults_or_manifest(tmp_path, files):
    bundle = make(tmp_path, files)
    run = arguments()
    before_args = vars(run).copy()
    before_manifest = ab.manifest_bytes(bundle.manifest)
    assert _apply_forecast_defaults(run, bundle) is None
    assert vars(run) == before_args
    assert ab.manifest_bytes(bundle.manifest) == before_manifest
    assert "forecast" not in bundle.manifest


@pytest.mark.parametrize("flag", ["--physics-pin-set", "--physics-profile",
                                  "--physics-default-backend", "--stochastic"])
@pytest.mark.parametrize("following", [[], ["--dt", "18.75"]])
def test_new_controls_require_a_value(flag, following):
    with pytest.raises(ab.BundleRefusal, match=f"{flag} takes a value"):
        ab.parse_forecast_args([flag, *following])


@pytest.mark.parametrize("flag,value", [
    ("--physics-pin-set", "unknown-pin"), ("--physics-pin-set", "relative/path"),
    ("--physics-profile", "unknown-profile"), ("--physics-profile", "https://example.invalid/profile"),
    ("--physics-default-backend", "auto"), ("--physics-default-backend", ""),
    ("--stochastic", "{broken"), ("--stochastic", "[]"), ("--stochastic", "null"),
    ("--stochastic", '{"sppt":1}'), ("--stochastic", '{"skeb":"true"}'),
    ("--stochastic", '{"member":-1}'), ("--stochastic", '{"seed":1.5}'),
    ("--stochastic", '{"member":true}'), ("--stochastic", '{"seed":18446744073709551616}'),
    ("--stochastic", '{"amplitude":NaN}'), ("--stochastic", '{"length_m":"bad"}'),
    ("--stochastic", '{"extra":"relative/path"}'),
])
def test_invalid_controls_are_refused_at_the_bundle_boundary(tmp_path, files, flag, value):
    with pytest.raises(ab.BundleRefusal, match=flag):
        ab.parse_forecast_args([flag, value])
    bundle = make(tmp_path, files)
    bundle.manifest["forecast"] = {"args": [flag, value]}
    with pytest.raises(ab.BundleRefusal):
        ab.validate_manifest(bundle.manifest)


@pytest.mark.parametrize("controls,match", [
    (ALL_CONTROLS, "has no route"),
    (PROFILE + ["--physics-mode", "exact"], "requires mode correct"),
    (PROFILE + ["--physics-default-backend", "native-cpu"], "require the CUDA resident route"),
])
def test_transport_does_not_bypass_runtime_route_refusals(tmp_path, files, controls, match):
    run = arguments()
    _apply_forecast_defaults(run, make(tmp_path, files, forecast_args=controls))
    with pytest.raises(ValueError, match=match):
        _native_route(run, default_mode="correct")


def test_existing_repeatable_controls_and_values_keep_their_order():
    controls = ["--physics", "config_microp_scheme=mp_thompson_aerosols",
                "--physics", "config_pbl_scheme=bl_mynn", "--dt", "18.75",
                "--fullstep-local-timestep", "--stochastic", STOCHASTIC]
    assert ab.parse_forecast_args(controls) == [
        ("--physics", "config_microp_scheme=mp_thompson_aerosols"),
        ("--physics", "config_pbl_scheme=bl_mynn"), ("--dt", "18.75"),
        ("--fullstep-local-timestep", None), ("--stochastic", STOCHASTIC)]
    assert json.loads(STOCHASTIC)["sppt"] is True
