"""Correct mode at the forecast door and in the proof harness (coordinator ruling 2026-10-14).

- A bare MPAS-suite door run resolves correct, lists its named gaps in the run record and prints them loudly.
- ``--physics-mode exact`` gives the route, namelist text and receipt of the code before the switch.
- A dry run with no suite needs no mode and is never refused for one.
- ``verify_mpas`` runs ``--physics-mode exact`` when a deck names no mode, and records the mode it ran.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from woof.hex import forecast_door as door
from woof.hex.mpasphys import forecast_v842
from woof.hex.mpasphys.forecast_v842 import native_route

from test_forecast_door import (  # noqa: F401  (autouse fixtures)
    _namespace,
    _proven_architecture,
    _reference_card,
    _registry,
    _satisfied_seam_pin,
)

#: The receipt keys of an exact-mode physics record, the same as before the switch existed.
EXACT_RECEIPT_KEYS = {"schema", "suite", "selectors", "options", "backends"}


def door_arguments(tmp_path: Path, *extra: str) -> argparse.Namespace:
    """A door argument vector for the explicit MPAS route with dummy native providers (preflight only)."""
    # The streams deck is parsed as XML before the run (rrfs/match deck_stream_inputs), so its placeholder is an empty deck.
    for name in ("library.so", "reader", "source", "init.nc", "streams.atmosphere"):
        (tmp_path / name).write_text("<streams/>" if name == "streams.atmosphere" else "placeholder")
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    return parser.parse_args([
        "--mesh", "x1.2562", "--init", str(tmp_path / "init.nc"), "--hours", "1",
        "--streams", str(tmp_path / "streams.atmosphere"), "--out", str(tmp_path / "out"),
        "--native-physics-library", str(tmp_path / "library.so"),
        "--native-pool-reader", str(tmp_path / "reader"), "--mpas-source", str(tmp_path / "source"),
        "--preflight", *extra])


@pytest.fixture
def native_preflight(monkeypatch):
    """Stand in for the Rust pool reader and the stream reader; everything else is the real route."""
    # These door tests fake the forecast itself; the ported-route suite refusal
    # (resident_v842.ported_route_refusal, tested in test_mpasphys_resident_v842) is not their subject.
    monkeypatch.setattr("woof.hex.mpasphys.resident_v842.ported_route_refusal", lambda config: None)
    layout = SimpleNamespace(configs={"config_start_time": "2023-03-31_12:00:00", "config_dt": 360.0,
                                      "config_run_duration": "0_01:00:00", "config_do_restart": False})
    monkeypatch.setattr(forecast_v842, "decode_registry_pools", lambda *a, **k: layout)
    monkeypatch.setattr(forecast_v842, "read_streams", lambda *a, **k: ())
    # The placeholder init.nc is not NetCDF: stand in for the spacing the suite-route step reads from the
    # mesh files (forecast_door._mesh_file_spacing_m, routeb-door b4477dcd) with x1.2562's 480 km.
    monkeypatch.setattr(door, "_mesh_file_spacing_m", lambda arguments: 480_000.0)
    # ... and for the init xtime the suite route takes its start from (forecast_v842.bind_init_start,
    # routeb-door 1146680c), the same 2023-03-31 12Z the stand-in layout names.
    from datetime import datetime
    from woof.hex import lbc
    monkeypatch.setattr(lbc, "_open_lbc", lambda path: None)
    monkeypatch.setattr(lbc, "_read_xtime", lambda dataset, path: datetime(2023, 3, 31, 12))


def mode_lines(err: str) -> list[str]:
    """stderr without the suite route's own step and start lines (routeb-door b4477dcd, 1146680c)."""
    return [line for line in err.splitlines()
            if not line.startswith(("woof hex: MPAS suite timestep ", "woof hex: MPAS suite start "))]


def run_door(arguments) -> dict:
    assert door._run_forecast(arguments) == 0
    return json.loads((Path(arguments.out) / "native-physics-request.json").read_text())["physics"]


def test_a_bare_mpas_suite_door_run_resolves_correct_and_lists_the_gaps(tmp_path, native_preflight, capsys):
    physics = run_door(door_arguments(tmp_path, "--physics-suite", "convection_permitting"))
    assert physics["physics_mode"] == "correct"
    block = physics["correct_mode"]
    # Every slot runs its GPU port by default, so the GPU MYNN (RP1, RP7) and Thompson (WB3) apply their fixes.
    assert block["mode"] == "correct" and block["applied"] == ["RP1", "RP7", "WB3"]
    gaps = {(g["finding"], g["slot"]): g for g in block["gaps"]}
    # RP3 lives in the surface layer and sea ice; the GPU sea ice does not carry it, so RP3 is not applied.
    assert set(gaps) == {("RP3", "seaice")}
    assert all({"finding", "slot", "reason", "owner"} <= set(g) for g in gaps.values())
    assert gaps["RP3", "seaice"]["owner"] == "p18-seaice"
    assert gaps["RP3", "seaice"]["backend"] == "hex-cuda"
    assert all(g["reason"] == "slot does not implement it" for g in gaps.values())
    err = capsys.readouterr().err
    assert "PHYSICS MODE correct" in err
    assert any(line.startswith("CORRECT MODE: named gaps (1)") for line in err.splitlines())
    warnings = [line for line in err.splitlines() if line.startswith("CORRECT MODE:")]
    assert len(warnings) >= 1 + len(gaps)
    for finding, slot in gaps:
        assert any(finding in line and slot in line for line in warnings)
    assert "PHYSICS ON CPU" not in err


#: The ten slots the convection_permitting suite turns on (ngw and oml are off in it).
CONVECTION_PERMITTING_ACTIVE = ("microphysics", "convection", "pbl", "surface_layer", "land_surface", "seaice",
                                "gwdo", "radiation_lw", "radiation_sw", "cloudiness")


def test_a_bare_mpas_suite_runs_every_slot_on_the_gpu(tmp_path, native_preflight, capsys):
    """The defect: a run that selected an MPAS suite ran MPAS's Fortran on the CPU in every slot."""
    physics = run_door(door_arguments(tmp_path, "--physics-suite", "convection_permitting",
                                      "--physics-mode", "exact"))
    assert {slot: physics["backends"][slot] for slot in CONVECTION_PERMITTING_ACTIVE} == dict.fromkeys(
        CONVECTION_PERMITTING_ACTIVE, "hex-cuda")
    assert "cpu_slots" not in physics
    assert mode_lines(capsys.readouterr().err) == []


def test_a_slot_without_a_gpu_port_runs_native_and_says_so(tmp_path, native_preflight, capsys):
    physics = run_door(door_arguments(tmp_path, "--physics-suite", "mesoscale_reference", "--physics-mode",
                                      "exact", "--physics", "config_radt_lw_scheme='cam_lw'",
                                      "--physics", "config_radt_sw_scheme='cam_sw'"))
    assert physics["backends"]["radiation_lw"] == physics["backends"]["radiation_sw"] == "native-cpu"
    assert physics["backends"]["microphysics"] == "hex-cuda"
    cpu = {row["slot"]: row for row in physics["cpu_slots"]}
    assert set(cpu) == {"radiation_lw", "radiation_sw"}
    assert cpu["radiation_lw"]["scheme"] == "cam_lw" and "no GPU slot for cam_lw" in cpu["radiation_lw"]["reason"]
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("PHYSICS ON CPU:")]
    assert lines[0].startswith("PHYSICS ON CPU: 2 active MPAS physics slot(s)")
    assert any("radiation_lw=cam_lw" in line for line in lines)
    assert any("radiation_sw=cam_sw" in line for line in lines)


def test_native_cpu_stays_selectable_as_the_referee(tmp_path, native_preflight, capsys):
    physics = run_door(door_arguments(tmp_path, "--physics-suite", "convection_permitting", "--physics-mode",
                                      "exact", "--physics-default-backend", "native-cpu",
                                      "--physics", "config_microp_backend='hex-cuda'"))
    assert physics["backends"]["microphysics"] == "hex-cuda"
    assert all(physics["backends"][slot] == "native-cpu" for slot in CONVECTION_PERMITTING_ACTIVE
               if slot != "microphysics")
    assert [row["slot"] for row in physics["cpu_slots"]] == [s for s in CONVECTION_PERMITTING_ACTIVE
                                                             if s != "microphysics"]
    assert "PHYSICS ON CPU: 9 active MPAS physics slot(s)" in capsys.readouterr().err
    # A slot named native-cpu by its own selector says so.
    physics = run_door(door_arguments(tmp_path, "--physics-suite", "convection_permitting", "--physics-mode",
                                      "exact", "--physics", "config_pbl_backend='native-cpu'"))
    assert [row["reason"] for row in physics["cpu_slots"]] == ["selected with config_pbl_backend='native-cpu'"]


def test_the_default_backend_flag_needs_an_mpas_suite_route(tmp_path):
    arguments = SimpleNamespace(physics_namelist=None, physics_suite=None, physics=None, physics_mode=None,
                                physics_correct_skip=None, physics_default_backend="native-cpu")
    with pytest.raises(ValueError, match="--physics-default-backend applies to an explicit MPAS suite route"):
        native_route(arguments)


def test_a_named_correct_mode_and_skip_reach_the_route(tmp_path, native_preflight, capsys):
    physics = run_door(door_arguments(tmp_path, "--physics-suite", "mesoscale_reference",
                                      "--physics-mode", "correct", "--physics-correct-skip", "RP3",
                                      "--physics-correct-skip", "WB3,WB5"))
    assert physics["correct_mode"] == {"mode": "correct", "applied": [], "skipped": ["RP3", "WB3", "WB5"],
                                       "gaps": []}
    assert "CORRECT MODE: named gaps" not in capsys.readouterr().err


def test_physics_mode_exact_gives_the_route_and_receipt_of_before(tmp_path, native_preflight, capsys):
    arguments = door_arguments(tmp_path, "--physics-suite", "convection_permitting", "--physics-mode", "exact")
    physics = run_door(arguments)
    assert set(physics) == EXACT_RECEIPT_KEYS
    assert mode_lines(capsys.readouterr().err) == []
    # The library route with no mode is the route of the code before the switch: same config, same text.
    explicit = native_route(arguments, default_mode="correct")
    arguments.physics_mode = None
    assert explicit == native_route(arguments)
    assert explicit[0].mode == "exact" and explicit[0].corrections == frozenset() and explicit[0].gaps == ()


def test_namelist_and_command_line_modes_must_agree(tmp_path):
    namelist = tmp_path / "namelist.atmosphere"
    namelist.write_text("&nhyd_model\nconfig_physics_mode = 'exact'\n/\n"
                        "&physics\nconfig_physics_suite = 'mesoscale_reference'\n/\n")
    arguments = SimpleNamespace(physics_namelist=namelist, physics_suite=None, physics=None,
                                physics_mode="correct", physics_correct_skip=None)
    with pytest.raises(ValueError, match="two modes"):
        native_route(arguments, default_mode="correct")
    arguments.physics_mode = None          # the namelist's exact wins over the door default
    assert native_route(arguments, default_mode="correct")[0].mode == "exact"


def fresh(tmp_path: Path, name: str) -> Path:
    """_namespace creates its checkout directory, so each namespace gets a directory of its own."""
    path = tmp_path / name
    path.mkdir()
    return path


def test_a_dry_run_with_no_suite_needs_no_mode(tmp_path, monkeypatch):
    arguments = _namespace(fresh(tmp_path, "bare"))
    assert native_route(arguments, default_mode="correct") == (None, "")
    assert native_route(_namespace(fresh(tmp_path, "exact"), physics_mode="exact"),
                        default_mode="correct") == (None, "")
    # Past the MPAS route the door resolves the request as before; stop it there.
    class Reached(Exception):
        pass

    def reached(arguments, *a, **k):
        raise Reached
    resolve_request = door.resolve_request
    monkeypatch.setattr(door, "resolve_request", reached)
    with pytest.raises(Reached):
        door._run_forecast(arguments)
    # On that (ArWen) route the bare run takes its preset row's mode, correct (lane H1), and
    # --physics-mode exact hands the driver the argument vector of before: no mode flag at all.
    same = fresh(tmp_path, "argv")
    plain = door.build_driver_argv(resolve_request(_namespace(same), registry=_registry()))
    (same / "woof").rmdir()
    exact = door.build_driver_argv(resolve_request(_namespace(same, physics_mode="exact"),
                                                   registry=_registry()))
    assert "--physics-mode" not in exact
    at = plain.index("--physics-mode")
    assert plain[at:at + 2] == ["--physics-mode", "correct"]
    assert plain[:at] + plain[at + 2:] == exact


def test_a_named_mode_without_an_mpas_suite_goes_to_the_arwen_route(tmp_path):
    # --physics-mode and --physics-correct-skip on a run with no MPAS suite route reach the ArWen door,
    # which reads the same option (lane H1); native_route does not take the run.
    assert native_route(_namespace(fresh(tmp_path, "mode"), physics_mode="correct"),
                        default_mode="correct") == (None, "")
    assert native_route(_namespace(fresh(tmp_path, "skip"), physics_correct_skip=["WB5"]),
                        default_mode="correct") == (None, "")
    # The ArWen door refuses a skip of a finding its route never applies (an A/B arm that equals its
    # control), by name.
    with pytest.raises(Exception, match="WOOF route"):
        door._physics_mode_fields(argparse.Namespace(physics_mode="correct", physics_correct_skip=["RP3"]))


def test_a_namelist_mode_without_an_mpas_suite_is_refused_by_name(tmp_path):
    namelist = tmp_path / "namelist.atmosphere"
    namelist.write_text("&nhyd_model\nconfig_physics_mode = 'correct'\n/\n")
    with pytest.raises(ValueError, match="explicit MPAS suite route"):
        native_route(_namespace(fresh(tmp_path, "nl"), physics_namelist=namelist), default_mode="correct")


# ---------------------------------------------------------------------------
# the proof harness names exact
# ---------------------------------------------------------------------------
def _proof_environment():
    from woof.hex import verification_environment as env
    return dict(schema=env.SCHEMA, errors=[], pins=dict(
        engine_version="2.9.2", engine_git_sha="a" * 40, engine_record_sha256="b" * 64,
        math_headers={"glibc_flt32.cuh": "c" * 64}, cuda_toolkit="nvcc release 13.0",
        cuda_runtime=13000, nvrtc=[13, 0], cupy_version="13.0",
        driver="610.57", gpu_uuid="GPU-example", numerics="strict"))


def _deck(tmp_path: Path, hex_command: list[str], namelist: str = "&physics\n/\n") -> Path:
    from woof.hex.verify_mpas import digest
    files = {"namelist": namelist, "streams": "<streams/>", "init": "init"}
    inputs = {}
    for name, text in files.items():
        path = tmp_path / f"{name}.input"
        path.write_text(text)
        inputs[name] = {"path": path.name, "sha256": digest(path)}
    environment = tmp_path / "expected-environment.json"
    environment.write_text(json.dumps(_proof_environment()))
    deck = dict(schema="mpas-verification-deck-v1", id="deck", pin_set="pins", mesh="x1.2562", date="d",
                process_count=1, inputs=inputs, native={"path": "native.json", "sha256": "0"},
                hex_command=hex_command,
                strict_environment={"path": environment.name, "sha256": digest(environment)})
    path = tmp_path / "deck.json"
    path.write_text(json.dumps(deck))
    return path


COMMAND = ["woof hex", "forecast", "--physics-namelist", "{namelist}", "--streams", "{streams}",
           "--init", "{init}", "--out", "{out}"]


@pytest.mark.parametrize("hex_command,namelist,expected_tail,source", [
    (COMMAND, "&physics\n/\n", ["--physics-mode", "exact"], "verify_mpas default (the deck names no mode)"),
    (COMMAND + ["--physics-mode", "exact"], "&physics\n/\n", ["--physics-mode", "exact"], "hex_command"),
    (COMMAND, "&nhyd_model\nconfig_physics_mode = 'correct'\n/\n", ["--out", "{out}"],
     "namelist config_physics_mode"),
])
def test_verify_mpas_runs_exact_when_the_deck_names_no_mode(tmp_path, monkeypatch, hex_command, namelist,
                                                            expected_tail, source):
    from woof.hex import verify_mpas as verifier
    from woof.hex import verification_environment as environment
    deck = _deck(tmp_path, hex_command, namelist)
    # Keep the production pin comparison; this test double replaces only device discovery.
    monkeypatch.setattr(environment, "capture", lambda *a, **k: _proof_environment())
    monkeypatch.setattr(verifier, "comparator_path", lambda given: Path("/bin/true"))
    monkeypatch.setattr(verifier, "native_manifest",
                        lambda *a: (tmp_path / "native.json", {"outputs": [], "resolved_output_root": tmp_path}))
    seen = []

    class ModelLaunched(Exception):
        pass

    def fake_run(argv, *a, **k):
        seen.append(list(argv))
        raise ModelLaunched
    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    args = SimpleNamespace(deck=deck, work_root=tmp_path / "work", comparator=None, hex_output=None,
                           failed_model_exit=None)
    with pytest.raises(ModelLaunched):
        verifier.verify(args)
    argv = seen[0]
    assert argv[:3] == [verifier.sys.executable, "-m", "woof.hex.cli"]
    out = str((tmp_path / "work").resolve() / "hex-output")
    # Every proof names strict kernel numerics (owner ruling 2026-10-16: fast is the door default).
    assert argv[-2:] == ["--numerics", "strict"]
    argv = argv[:-2]
    # A deck naming no default backend keeps running the native referee it was written for.
    assert argv[-2:] == ["--physics-default-backend", "native-cpu"]
    argv = argv[:-2]
    assert argv[-len(expected_tail):] == [part.format_map({"out": out}) for part in expected_tail]
    assert argv.count("--physics-mode") == (0 if "correct" in namelist else 1)
    # The deck file itself is never edited.
    assert json.loads(deck.read_text())["hex_command"] == hex_command
    command, record = verifier.proof_physics_mode(hex_command, tmp_path / "namelist.input")
    assert record["source"] == source and record["mode"] == ("correct" if "correct" in namelist else "exact")


def test_verify_mpas_keeps_a_named_default_backend():
    from woof.hex import verify_mpas as verifier
    named = COMMAND + ["--physics-default-backend", "hex-cuda"]
    assert verifier.proof_default_backend(named) == named
    assert verifier.proof_default_backend(COMMAND) == COMMAND + ["--physics-default-backend", "native-cpu"]


def test_model_command_accepts_one_physics_mode_and_refuses_a_malformed_one():
    from woof.hex.verify_mpas import VerificationRefusal, validate_model_command
    validate_model_command(COMMAND)
    validate_model_command(COMMAND + ["--physics-mode", "exact"])
    validate_model_command(COMMAND + ["--physics-mode=correct"])
    for bad in (["--physics-mode", "exact", "--physics-mode", "exact"], ["--physics-mode", "corect"],
                ["--physics-mode"]):
        with pytest.raises(VerificationRefusal, match="physics-mode"):
            validate_model_command(COMMAND + bad)


@pytest.mark.parametrize("analysis", [False, True])
def test_input_validation_keeps_suite_cycling_preflight(tmp_path, native_preflight, analysis):
    from test_mpasphys_checkpoint_v842 import write_checkpoint

    carrier, *_ = write_checkpoint(tmp_path)
    flags = ["--physics-suite", "convection_permitting", "--restart-from", str(carrier),
             "--history-every-minutes", "6", "--dt", "60"]
    if analysis:
        flags += ["--da-analysis", str(tmp_path / "analysis.nc")]
    args = door_arguments(tmp_path, *flags)
    assert door._run_forecast(args) == 0
    receipt = json.loads((tmp_path / "out/native-physics-request.json").read_text())
    assert receipt["steps"] == 54
