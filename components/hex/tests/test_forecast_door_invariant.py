"""The forecast door hands the native invariant stream to the driver by default.

THE BREAKAGE THIS PREVENTS (measured 2026-10-05 on
RRFS v2's conus12km, physics off): RRFS v2 configures MPAS-A's invariant
stream, so the native model reads its vertical grid from
conus12km.invariant.nc_L60_GFS.  The driver learned --invariant, but the door
never passed it, so a bare ``woof hex forecast`` still read the grid from
init.nc (2.7 mm off in zgrid, 8e-5 relative in zz), which moved hex from
3.6e-5 to 2.0e-4 K RMS theta against NOAA's MPAS at hour 1.  Fixed means
default-on: the row names the file and the door selects it when it is there.

CPU only.  The device seams are pinned by test_forecast_door's autouse
fixtures, imported below.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tomllib

import pytest

from woof.hex import forecast_door as door

from test_forecast_door import (  # noqa: F401  (autouse fixtures)
    X1_CELLS,
    _namespace,
    _proven_architecture,
    _reference_card,
    _registry,
    _satisfied_seam_pin,
)

ROOT = Path(__file__).resolve().parents[1]
INVARIANT = "x1.40962.invariant.nc_L60_GFS"


def _levels_file(path: Path, layers: int) -> Path:
    from netCDF4 import Dataset

    with Dataset(path, mode="w") as dataset:
        dataset.createDimension("nVertLevels", layers)
        dataset.createDimension("nVertLevelsP1", layers + 1)
    return path


def _declaring_registry() -> dict[str, door.MeshRow]:
    registry = _registry()
    registry["x1.40962"] = replace(registry["x1.40962"], invariant_file=INVARIANT)
    return registry


def _case(tmp_path: Path, *, invariant_layers: int | None = 59, **overrides):
    arguments = _namespace(tmp_path, **overrides)
    _levels_file(Path(arguments.init), 59)
    if invariant_layers is not None:
        _levels_file(Path(arguments.grid).parent / INVARIANT, invariant_layers)
    return arguments


def test_a_bare_run_on_a_declaring_row_reads_the_invariant(tmp_path: Path) -> None:
    arguments = _case(tmp_path)
    request = door.resolve_request(arguments, registry=_declaring_registry())
    expected = (Path(arguments.grid).parent / INVARIANT).absolute()
    assert request.invariant == expected
    assert "found beside the grid" in request.invariant_selection
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--invariant") + 1] == str(expected)


def test_the_driver_parser_receives_the_file(tmp_path: Path) -> None:
    from woof.hex.drivers.run_cuda_v841_forecast import parse_args

    request = door.resolve_request(_case(tmp_path), registry=_declaring_registry())
    assert parse_args(door.build_driver_argv(request)).invariant == request.invariant


def test_the_receipt_names_the_vertical_grid_and_why(tmp_path: Path) -> None:
    request = door.resolve_request(_case(tmp_path), registry=_declaring_registry())
    receipt = door.build_receipt(
        request=request, admission=None, bind_receipt=None, driver_receipt=None,
        history=[], driver_argv=door.build_driver_argv(request), seconds=0.1,
        status="refused_by_admission",
    )
    assert receipt["vertical_grid"]["invariant"] == str(request.invariant)
    assert receipt["vertical_grid"]["selection"] == request.invariant_selection


def test_an_absent_declared_file_runs_on_the_init_grid_and_says_so(tmp_path: Path) -> None:
    request = door.resolve_request(
        _case(tmp_path, invariant_layers=None), registry=_declaring_registry())
    assert request.invariant is None
    assert INVARIANT in request.invariant_selection
    assert "--invariant" in request.invariant_selection
    assert "--invariant" not in door.build_driver_argv(request)


def test_a_row_with_no_stream_passes_nothing(tmp_path: Path) -> None:
    request = door.resolve_request(_case(tmp_path), registry=_registry())
    assert request.invariant is None
    assert "declares no invariant stream" in request.invariant_selection
    assert "--invariant" not in door.build_driver_argv(request)


def test_no_invariant_declines_the_present_file(tmp_path: Path) -> None:
    request = door.resolve_request(
        _case(tmp_path, no_invariant=True), registry=_declaring_registry())
    assert request.invariant is None
    assert "declined by --no-invariant" in request.invariant_selection
    assert "--invariant" not in door.build_driver_argv(request)


def test_an_explicit_file_wins_over_the_row(tmp_path: Path) -> None:
    other = _levels_file(tmp_path / "elsewhere.invariant.nc", 59)
    request = door.resolve_request(
        _case(tmp_path, invariant=other), registry=_declaring_registry())
    assert request.invariant == other.absolute()
    assert request.invariant_selection == "named by --invariant"


def test_both_flags_are_refused(tmp_path: Path) -> None:
    arguments = _case(tmp_path, invariant=tmp_path / "x.nc", no_invariant=True)
    with pytest.raises(door.ForecastDoorRefusal, match="--no-invariant"):
        door.resolve_request(arguments, registry=_declaring_registry())


def test_a_missing_explicit_file_is_refused_by_name(tmp_path: Path) -> None:
    arguments = _case(tmp_path, invariant=tmp_path / "absent.invariant.nc")
    with pytest.raises(door.ForecastDoorRefusal, match="--invariant"):
        door.resolve_request(arguments, registry=_declaring_registry())


def test_a_run_directory_of_links_reads_the_targets_and_finds_the_invariant_beside_the_links(
        tmp_path: Path) -> None:
    """THE BREAKAGE: the door refused a linked --init, so users passed the resolved file and lost what they staged
    beside the link (the RRFS invariant: 12,800,256 of 29,644,818 words different by f006 on x1.2562 hrrrv5)."""
    arguments = _case(tmp_path, invariant_layers=None)
    store, run = tmp_path / "store", tmp_path / "run"
    store.mkdir()
    run.mkdir()
    targets = {}
    for key in ("grid", "static", "init"):
        named = Path(getattr(arguments, key))
        targets[key] = named.replace(store / named.name).resolve()
        (run / named.name).symlink_to(targets[key])
        setattr(arguments, key, run / named.name)
    _levels_file(store / INVARIANT, 59)
    (run / INVARIANT).symlink_to(store / INVARIANT)
    request = door.resolve_request(arguments, registry=_declaring_registry())
    # The drivers get the files the links reach (they refuse a link), and the invariant beside the named grid.
    assert (request.grid, request.static, request.init) == (targets["grid"], targets["static"], targets["init"])
    assert request.invariant == (store / INVARIANT).resolve()
    assert "found beside the grid" in request.invariant_selection


def test_a_link_to_a_missing_file_is_refused_naming_its_target(tmp_path: Path) -> None:
    arguments = _case(tmp_path)
    (tmp_path / "linked.init.nc").symlink_to(tmp_path / "gone.init.nc")
    arguments.init = tmp_path / "linked.init.nc"
    with pytest.raises(door.ForecastDoorRefusal, match="gone.init.nc, which does not exist"):
        door.resolve_request(arguments, registry=_declaring_registry())


def test_a_different_level_set_is_refused_before_the_card(tmp_path: Path) -> None:
    arguments = _case(tmp_path, invariant_layers=64)
    with pytest.raises(door.ForecastDoorRefusal, match="65 vertical interfaces"):
        door.resolve_request(arguments, registry=_declaring_registry())


def test_preflight_collects_a_level_mismatch(tmp_path: Path) -> None:
    arguments = _case(tmp_path, invariant_layers=64, preflight=True)
    request = door.resolve_request(arguments, registry=_declaring_registry())
    assert any("vertical interfaces" in problem for problem in request.input_problems)


def test_the_cli_defaults_select_the_row() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    arguments = parser.parse_args([])
    assert arguments.invariant is None
    assert arguments.no_invariant is False


def test_the_rrfs_rows_declare_their_invariant_stream() -> None:
    registry = door.load_registry()
    assert registry["conus12km"].invariant_file == "conus12km.invariant.nc_L60_GFS"
    assert registry["conus3km"].invariant_file == "conus3km.invariant.nc_L60_GFS"
    assert registry["x1.40962"].invariant_file is None


def test_the_rrfs_settings_file_names_the_rows_file() -> None:
    settings = tomllib.loads((ROOT / "tools/rrfs_h6/RRFS-V2.toml").read_text())
    registry = door.load_registry()
    assert settings["vertical"]["invariant"] == registry[settings["mesh"]].invariant_file


# Whole-model verification policy shares this registered CPU door tier.
"""Whole-model gate policy; stored-word and real-door tests live in rw-mpas."""
import importlib.util
import sys


from woof.hex.verify_mpas import aggregate, gate_receipt_failures, ratchet_failures, tier, VerificationRefusal


CONTEXT = {"pin_set": "pins", "mesh": "mesh", "date": "date", "deck": "deck", "process_count": 1}


def field(step=0, words=0, status="compared", ulp=0):
    return {"stream": "history", "field": "rho", "step": step, "time": str(step),
            "status": status, "differing_words": words, "compared_words": 3, "max_ulp": ulp}


def test_ratchet_uses_pin_set_and_counts_missing_fields():
    baseline = aggregate([field(0), field(1)], CONTEXT)
    candidate = aggregate([field(0), field(1, 1, ulp=1)], CONTEXT)
    assert any("differing_words 0 -> 1" in reason for reason in ratchet_failures(candidate, baseline))
    missing = aggregate([field(0), field(1, 3, "field missing in hex")], CONTEXT)
    assert missing["entries"][0]["missing_samples"] == 1
    assert any("missing_samples" in reason for reason in ratchet_failures(missing, baseline))
    other_pins = aggregate([field(0), field(1, 1, ulp=1)], {**CONTEXT, "pin_set": "new-pins"})
    assert ratchet_failures(other_pins, candidate)  # unknown pin set starts at zero


def test_ratchet_cannot_drop_fields_shorten_forecast_or_move_first_difference():
    baseline = aggregate([field(0), field(2, 1, ulp=1)], CONTEXT)
    earlier = aggregate([field(0), field(1, 1, ulp=1)], CONTEXT)
    assert any("moved earlier" in reason for reason in ratchet_failures(earlier, baseline))
    fewer = aggregate([field(0)], CONTEXT)
    assert any("fewer native words" in reason for reason in ratchet_failures(fewer, baseline))
    missing = aggregate([{**field(), "field": "new"}], CONTEXT)
    assert any("inventoried field absent" in reason for reason in ratchet_failures(missing, baseline))
    baseline["entries"].append(baseline["entries"][0])
    with pytest.raises(VerificationRefusal, match="duplicate"):
        ratchet_failures(earlier, baseline)


def test_duplicate_comparison_rows_and_negative_counters_refuse():
    with pytest.raises(VerificationRefusal, match="repeated"):
        aggregate([field(), field()], CONTEXT)
    with pytest.raises(VerificationRefusal, match="counter"):
        aggregate([field(words=-1)], CONTEXT)


@pytest.mark.parametrize("missing_status", ["field missing in hex", "output time missing in hex", "type or dimensions differ in hex"])
def test_missing_schema_never_qualifies_for_spread(missing_status):
    result = {"fields": [field(status=missing_status)], "spread_samples": [{"passed": True}]}
    assert tier(result, ["routine residue"])[0] is None


def test_tiers_do_not_allow_one_step_to_excuse_later_spread_failure():
    assert tier({"fields": [field(), field(1)]}, [])[0] == "BITWISE"
    result = {"fields": [field(), field(1), field(2, 1)], "spread_samples": [{"passed": True}]}
    assert tier(result, ["expf source order"])[0] == "ONE-STEP-IDENTICAL"
    assert tier(result, [])[0] is None
    result["spread_samples"] = [{"passed": False, "stream": "history", "step": 2}]
    assert tier(result, ["expf source order"])[0] is None
    result["fields"] = [field(), field(3, 1)]
    result["spread_samples"] = [{"passed": True}]
    assert tier(result, ["expf source order"])[0] == "INSIDE-NATIVE-SPREAD"


def test_gate_refuses_comparison_only_and_truncated_forecasts():
    deck = {"process_count": 1, "expected_samples": {"history": [0, 1, 2]}, "duration_steps": 2}
    receipt = {"process_count": 1, "passed": True, "tier": "BITWISE", "acceptance_tier": "BITWISE", "evidence_kind": "COMPARISON-ONLY",
               "comparison": {"fields": [field(0), field(1), field(2)]}}
    assert gate_receipt_failures(receipt, deck, "G2")
    receipt["evidence_kind"] = "FORECAST"
    assert not gate_receipt_failures(receipt, deck, "G1")
    receipt["comparison"]["fields"].pop()
    assert len(gate_receipt_failures(receipt, deck, "G1")) == 2
    receipt["tier"] = "INSIDE-NATIVE-SPREAD"
    assert any("zero differing words" in reason for reason in gate_receipt_failures(receipt, deck, "G1"))


def citation_checker():
    path = Path(__file__).resolve().parents[1] / "tools/check_mpas_citations.py"
    spec = importlib.util.spec_from_file_location("f2_citations", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_routine_citation_cannot_inherit_another_routines_source(tmp_path):
    source = tmp_path / "woof/hex/mpasphys"
    native = tmp_path / "mpas/src"
    source.mkdir(parents=True)
    native.mkdir(parents=True)
    (native / "scheme.F").write_text("module scheme\nend module scheme\n")
    (source / "scheme.py").write_text('''"""Module source scheme.F:1-2"""
def cited():
    """scheme.F:1-2"""
    return 1
def uncited():
    return 2
def nested():
    def child():
        """scheme.F:1-2"""
        return 3
    return child()
''')
    failures = citation_checker().check_routine_coverage(source.parent, native.parent)
    assert len(failures) == 2
    assert any("uncited" in f for f in failures)
    assert any("nested" in f for f in failures)


def test_cuda_citation_checks_adjacent_routine_comment(tmp_path):
    source = tmp_path / "hexcore"
    native = tmp_path / "mpas/src"
    source.mkdir()
    native.mkdir(parents=True)
    (native / "scheme.F").write_text("module scheme\nend module scheme\n")
    (source / "cuda_diagnostics.py").write_text('''CODE = r"""
// scheme.F:1-2
__global__ void cited() { }
__device__ float uncited(float a) { return a; }
"""
''')
    failures = citation_checker().check_routine_coverage(source, native.parent)
    assert len(failures) == 1 and "uncited" in failures[0]


def test_campaign_source_symlink_uses_pinned_canonical_file(tmp_path, monkeypatch):
    import hashlib
    checker = citation_checker()
    source = tmp_path / "hexcore"
    native = tmp_path / "MPAS-Model-v8.4.2"
    source.mkdir()
    (native / "src/core_atmosphere").mkdir(parents=True)
    (native / "src/core_init_atmosphere").mkdir()
    data = native / "src/core_atmosphere/mpas_shared.F"
    data.write_text("module shared\nend module shared\n")
    (native / "src/core_init_atmosphere/mpas_shared.F").symlink_to(data)
    (source / "init.py").write_text('"""src/core_init_atmosphere/mpas_shared.F:1-2"""\n')
    manifest = tmp_path / "MPAS-Model-v8.4.2.sha256"
    manifest.write_text(hashlib.sha256(data.read_bytes()).hexdigest() +
                        "  ./src/core_atmosphere/mpas_shared.F\n")
    monkeypatch.setattr(checker, "SOURCE_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert checker.check_campaign_citations(source, native) == []
    data.write_text("module altered\nend module altered\n")
    assert "SHA-256 differs" in checker.check_campaign_citations(source, native)[0]


def test_model_command_binds_inputs_to_model_flags_once():
    from woof.hex.verify_mpas import validate_model_command
    command = ["woof hex", "forecast", "--physics-namelist", "{namelist}",
               "--streams={streams}", "--init", "{init}", "--out", "{out}"]
    validate_model_command(command)
    # Mentioning a pinned init as descriptive provenance cannot conceal the
    # different init actually supplied to the forecast. Duplicate switches
    # cannot silently replace the verifier's binding in argparse.
    for changed in (command + ["--init", "other.nc"],
                    command[:5] + ["--init", "other.nc", "--init-source", "{init}", "--out", "{out}"],
                    ["woof hex", "doctor"] + command[2:], ["woof hex"]):
        with pytest.raises(VerificationRefusal):
            validate_model_command(changed)


def test_campaign_retains_committed_legacy_windows(tmp_path, monkeypatch):
    import hashlib
    checker = citation_checker()
    source = tmp_path / "hexcore"
    source.mkdir()
    native = tmp_path / "MPAS-Model-v8.4.2"
    native.mkdir()
    manifest = tmp_path / "MPAS-Model-v8.4.2.sha256"
    manifest.write_text("")
    monkeypatch.setattr(checker, "SOURCE_MANIFEST_SHA256", hashlib.sha256(b"").hexdigest())
    monkeypatch.setattr(checker, "VENDOR_ROOT", tmp_path / "legacy")
    citation = "MPAS-Model-v8.2.3/src/core_atmosphere/Registry.xml:1520-1532"
    (source / "legacy.py").write_text(f'"""{citation}"""\n')
    assert checker.check_campaign_citations(source, native) == []
    (source / "legacy.py").write_text(f'"""{citation.replace("1532", "1533")}"""\n')
    assert "older MPAS release" in checker.check_campaign_citations(source, native)[0]
    (source / "legacy.py").write_text(f'"""{citation}"""\n')
    archived = checker.VENDOR_ROOT / "src/core_atmosphere"
    archived.mkdir(parents=True)
    (archived / "Registry.xml").write_text("changed\n" * 1532)
    assert "source-window pin mismatch" in checker.check_campaign_citations(source, native)[0]


def test_lander_gate_interface_refuses_missing_index_with_receipt(tmp_path):
    import subprocess
    import json
    tree = Path(__file__).resolve().parents[1]
    out = tmp_path / "gate-out"
    result = subprocess.run(["bash", str(tree / "tools/mpas_gate.sh"), "--gates", "G1,G2",
                             "--index", str(tmp_path / "absent-index.json"), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    receipt = json.loads((out / "gate.json").read_text())
    assert receipt["passed"] is False and receipt["stages"] == {}
    assert "absent-index.json" in receipt["failures"][0]
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema": "mpas-gate-index-v1", "gates": {"G1": [], "G2": []}, "landed_decks": []}))
    missing_baseline = tmp_path / "absent-ratchet.json"
    second_out = tmp_path / "missing-baseline"
    result = subprocess.run(["bash", str(tree / "tools/mpas_gate.sh"), "--gates", "G1,G2",
                             "--index", str(index), "--ratchet", str(missing_baseline), "--out", str(second_out)],
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    receipt = json.loads((second_out / "gate.json").read_text())
    assert receipt["stages"] == {} and "absent-ratchet.json" in receipt["failures"][0]
    from woof.hex.verify_mpas import gate_parser
    assert gate_parser().parse_args(["--gates", "G1,G2", "--out", str(out)]).gates == ["G1", "G2"]
    for gates in ("", "G1,G1", "G1,", "G1,G9"):
        with pytest.raises(SystemExit):
            gate_parser().parse_args(["--gates", gates, "--out", str(out)])


def test_stream_filename_clock_reaches_all_records_and_refuses_ambiguous_clock():
    import xml.etree.ElementTree as ET
    from woof.hex.verify_mpas import stream_filename_times
    stream = ET.fromstring('<stream output_interval="0_01:00:00" filename_interval="none"/>')
    tokens = dict(Y="2000", M="01", D="01", h="23", m="00", s="00")
    assert stream_filename_times(stream, tokens, 3) == ["2000-01-01_23:00:00",
        "2000-01-02_00:00:00", "2000-01-02_01:00:00"]
    stream.set("filename_interval", "output_interval")
    with pytest.raises(VerificationRefusal, match="one record per file"):
        stream_filename_times(stream, tokens, 3)
    with pytest.raises(VerificationRefusal, match="complete filename timestamp"):
        stream_filename_times(stream, {"Y": "2000"}, 1)


@pytest.mark.parametrize("value", [None, True, False, 0, -1, "1", 1.0])
def test_proof_requires_a_positive_integer_process_count(value):
    from woof.hex.verify_mpas import process_count
    with pytest.raises(VerificationRefusal, match="process_count"):
        process_count(value)


def test_ratchet_refuses_different_or_unwitnessed_process_counts():
    baseline = aggregate([field()], CONTEXT)
    candidate = aggregate([field()], {**CONTEXT, "process_count": 2})
    assert any("process count" in f for f in ratchet_failures(candidate, baseline))
    baseline["entries"][0].pop("process_count")
    assert any("process count" in f for f in ratchet_failures(candidate, baseline))


@pytest.mark.parametrize("grade", ["BITWISE", "ONE-STEP-IDENTICAL", "INSIDE-NATIVE-SPREAD"])
def test_gate_checks_actual_words_and_count_independently_of_claimed_grade(grade):
    deck = {"process_count": 1, "expected_samples": {"history": [0, 1]}, "duration_steps": 1}
    receipt = {"process_count": 1, "passed": True, "tier": grade,
               "evidence_kind": "FORECAST", "comparison": {"fields": [field(), field(1, 1)]}}
    assert any("diagnostic grades" in f for f in gate_receipt_failures(receipt, deck, "G2"))
    receipt["comparison"]["fields"][1] = field(1)
    receipt["process_count"] = 2
    assert any("process count" in f for f in gate_receipt_failures(receipt, deck, "G2"))


def test_spread_acceptance_authenticates_math_artifacts_and_replay(tmp_path):
    import hashlib
    import json
    from woof.hex.verify_mpas import acceptance_tier, maths_library_witness
    def pin(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    strict, optimized = tmp_path / "strict.json", tmp_path / "optimized.json"
    strict.write_text("strict reference bytes")
    optimized.write_text("optimized reference bytes")
    deck = {"pin_set": "pins", "process_count": 1, "declared_libm_residue": ["expf"]}
    libraries = {}
    for name in ("native_strict", "native_optimized", "hex"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        libraries[name] = pin(path)
    replay = {"schema": "mpas-maths-replay-v1", "process_count": 1,
              "libraries": {k: v["sha256"] for k, v in libraries.items()},
              "routines": ["expf"], "tested_words": 100, "differing_words": 0}
    replay_path = tmp_path / "replay.json"
    witness = {"schema": "mpas-maths-library-witness-v1", "pin_set": "pins", "process_count": 1,
               "strict_native_manifest_sha256": pin(strict)["sha256"],
               "optimized_native_manifest_sha256": pin(optimized)["sha256"],
               "declared_libm_residue": ["expf"], "libraries": libraries}
    path = tmp_path / "witness.json"
    def publish():
        replay_path.write_text(json.dumps(replay))
        witness["replay"] = pin(replay_path)
        path.write_text(json.dumps(witness))
        deck["maths_library_witness"] = pin(path)
    publish()
    validated = maths_library_witness(deck, tmp_path, strict, optimized)
    result = {"fields": [field(), field(1, 1)]}
    assert acceptance_tier(result, "ONE-STEP-IDENTICAL", None)[0] is None
    assert acceptance_tier(result, "INSIDE-NATIVE-SPREAD", None)[0] is None
    assert acceptance_tier(result, "ONE-STEP-IDENTICAL", validated)[0] == "INSIDE-NATIVE-SPREAD"
    assert acceptance_tier({"fields": [field()]}, "BITWISE", None) == ("BITWISE", [])
    replay["differing_words"] = 1
    publish()
    with pytest.raises(VerificationRefusal, match="zero-word witness"):
        maths_library_witness(deck, tmp_path, strict, optimized)
    replay["differing_words"] = 0
    witness["process_count"] = 2
    publish()
    with pytest.raises(VerificationRefusal, match="process count"):
        maths_library_witness(deck, tmp_path, strict, optimized)
    witness["process_count"] = 1
    publish()
    (tmp_path / "hex").write_bytes(b"changed math artifact")
    with pytest.raises(VerificationRefusal, match="changed pinned input"):
        maths_library_witness(deck, tmp_path, strict, optimized)


def test_default_citation_mode_authenticates_file_without_requiring_routine_coverage(tmp_path, monkeypatch):
    import hashlib
    checker = citation_checker()
    source, native = tmp_path / "hexcore", tmp_path / "MPAS-Model-v8.4.2"
    source.mkdir()
    (native / "src").mkdir(parents=True)
    data = native / "src/module_scheme.F"
    data.write_text("module scheme\nend module scheme\n")
    physics = source / "mpasphys"
    physics.mkdir()
    (physics / "scheme.py").write_text('"""module_scheme.F:1-2"""\ndef uncited():\n    return 1\n')
    manifest = tmp_path / "MPAS-Model-v8.4.2.sha256"
    manifest.write_text(hashlib.sha256(data.read_bytes()).hexdigest() + "  ./src/module_scheme.F\n")
    monkeypatch.setattr(checker, "SOURCE_ROOT", source)
    monkeypatch.setattr(checker, "SOURCE_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert checker.main(["--mpas-source", str(native)]) == 0
    assert checker.main(["--mpas-source", str(native), "--routine-coverage"]) == 1


def test_native_incomplete_water_inventory_cannot_pass_nonzero_spread():
    from woof.hex.verify_mpas import acceptance_tier
    inventory = {"water_integral_complete": False,
                 "water_integral_limitation": "native streams omit cumulative evaporation"}
    result = {"fields": [field(0), field(2, 1)]}
    accepted, failures = acceptance_tier(result, "INSIDE-NATIVE-SPREAD", {"process_count": 1}, inventory)
    assert accepted is None and "cumulative evaporation" in failures[0]
    result["fields"] = [field(0), field(2)]
    assert acceptance_tier(result, "BITWISE", None, inventory) == ("BITWISE", [])


def test_gate_rejects_recovered_outer_failure_even_with_zero_words():
    deck = {"process_count": 1, "expected_samples": {"history": [0, 1]}, "duration_steps": 1}
    receipt = {"process_count": 1, "passed": True, "tier": "BITWISE", "acceptance_tier": "BITWISE",
               "evidence_kind": "FORECAST", "failed_model_exit": 124,
               "comparison": {"fields": [field(0), field(1)]}}
    assert any("interrupted outer model job" in failure
               for failure in gate_receipt_failures(receipt, deck, "G2"))


def test_wrf_driver_is_not_a_bare_mpas_citation():
    checker = citation_checker()
    assert not checker.is_mpas_citation("module_microphysics_driver.F")
    assert not checker.is_mpas_citation("module_sf_noahmplsm.F")
    assert checker.is_mpas_citation("MPAS-Model-v8.4.2/module_microphysics_driver.F")
    assert checker.is_mpas_citation("module_unknown_physics.F")
    assert checker.is_mpas_citation("mpas_atmphys_driver_microphysics.F")


def test_historical_rrfs_window_keeps_its_source_authority(tmp_path, monkeypatch):
    import hashlib
    checker = citation_checker()
    root = tmp_path / "rrfs"
    source = root / checker.RRFS_SOURCE_FILE
    source.parent.mkdir(parents=True)
    source.write_text("native\n" * 1154)
    monkeypatch.setattr(checker, "RRFS_SOURCE_ROOT", root)
    monkeypatch.setattr(checker, "RRFS_SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    occurrence = checker.CitationOccurrence(checker.RRFS_CITATION, "src/hexcore/cuda_ntiedtke_column_v841.py", 41)
    assert checker.check_rrfs_window(checker.RRFS_CITATION, [occurrence]) == []
    # Neither an adjacent window nor a different port inherits this pin.
    assert checker.check_rrfs_window(checker.RRFS_CITATION.replace("1154", "1155"), [occurrence]) is None
    other = checker.CitationOccurrence(checker.RRFS_CITATION, "src/hexcore/new_port.py", 1)
    assert checker.check_rrfs_window(checker.RRFS_CITATION, [occurrence, other]) is None
    source.write_text("changed\n" * 1154)
    assert "source SHA-256 mismatch" in checker.check_rrfs_window(checker.RRFS_CITATION, [occurrence])[0]
    source.unlink()
    assert "source unavailable" in checker.check_rrfs_window(checker.RRFS_CITATION, [occurrence])[0]


@pytest.mark.parametrize("interrupted", [False, True])
def test_gate_owns_tmpdir_scratch_and_cleans_it_on_failure(tmp_path, monkeypatch, interrupted):
    from woof.hex import verify_mpas as verifier
    import os
    from types import SimpleNamespace
    temporary_root = tmp_path / "data-tmp"
    monkeypatch.setenv("TMPDIR", str(temporary_root))
    monkeypatch.setenv("PYTEST_ADDOPTS", "--basetemp=/old-root/dumps -ra --basetemp '/another root/dumps'")
    out = tmp_path / "receipts"
    args = verifier.gate_parser().parse_args(["--stage", "G0", "--out", str(out)])
    seen = []
    def execute(command, **kwargs):
        if "--basetemp" in command:
            scratch = Path(command[command.index("--basetemp") + 1])
            assert scratch.is_relative_to(temporary_root)
            assert not scratch.is_relative_to(out)
            assert kwargs["env"]["TMPDIR"] == str(scratch.parent)
            assert kwargs["env"]["PYTEST_ADDOPTS"] == "-ra"
            (Path(kwargs["env"]["TMPDIR"]) / "subprocess-temporary-dump").write_bytes(b"also disposable")
            scratch.mkdir()
            (scratch / "native-dump").write_bytes(b"disposable")
            seen.append(scratch.parent)
            if interrupted:
                raise KeyboardInterrupt()
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(verifier.subprocess, "run", execute)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            verifier.gate(args)
    else:
        assert verifier.gate(args) == 1
        assert (out / "gate.json").is_file()
        assert (out / "G0-0.log").is_file()
    assert seen and all(not path.exists() for path in seen)
    assert os.environ["TMPDIR"] == str(temporary_root)


def test_gate_without_tmpdir_keeps_plain_fixture_outside_git(tmp_path, monkeypatch):
    from woof.hex import verify_mpas as verifier
    import os
    import subprocess
    work = tmp_path / "lane"
    checkout = work / "hex"
    module = checkout / "tree/src/hexcore/verify_mpas.py"
    module.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    monkeypatch.setattr(verifier, "__file__", str(module))
    monkeypatch.delenv("TMPDIR", raising=False)
    seen = []
    def check_plain_fixture(args, scratch):
        plain = scratch / "not-a-repository"
        plain.mkdir()
        result = subprocess.run(["git", "-C", str(plain), "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True)
        assert result.returncode != 0
        assert scratch.is_relative_to(work) and not scratch.is_relative_to(checkout)
        seen.append(scratch)
        return 1
    monkeypatch.setattr(verifier, "_gate", check_plain_fixture)
    assert verifier.gate(object()) == 1
    assert seen and not seen[0].exists()
    assert "TMPDIR" not in os.environ


def test_gate_forecast_proof_cap_and_timeout_recovery_use_tmpdir(tmp_path, monkeypatch):
    from woof.hex import verify_mpas as verifier
    from types import SimpleNamespace
    import json
    temporary_root = tmp_path / "data-tmp"
    monkeypatch.setenv("TMPDIR", str(temporary_root))
    # a shared host's queues, named the way woof.hex.site_runner reads them
    monkeypatch.setenv("HEX_GPU_QUEUE", "gpu-queue")
    monkeypatch.setenv("HEX_BUILD_QUEUE", "build-queue")
    deck = {"id": "test-deck", "process_count": 1, "duration_steps": 2,
            "expected_samples": {"history": [0, 2]}}
    path = tmp_path / "deck.json"
    verifier.write_json(path, deck)
    index = tmp_path / "index.json"
    verifier.write_json(index, {"schema": "mpas-gate-index-v1", "gates": {"G2": [
        {"path": str(path), "sha256": verifier.digest(path)}]}, "landed_decks": [deck["id"]]})
    ratchet = tmp_path / "ratchet.json"
    verifier.write_json(ratchet, {"schema": "mpas-ratchet-v1", "entries": []})
    out = tmp_path / "receipts"
    args = verifier.gate_parser().parse_args(["--gates", "G2", "--index", str(index),
                                             "--ratchet", str(ratchet), "--out", str(out)])
    assert args.minutes == 120
    for minutes in (0, 121):
        with pytest.raises(SystemExit):
            verifier.gate_parser().parse_args(["--stage", "G1", "--out", str(out), "--minutes", str(minutes)])
    seen = []
    def execute(command, **kwargs):
        work = Path(command[command.index("--work-root") + 1])
        assert work.is_relative_to(temporary_root) and not work.is_relative_to(out)
        work.mkdir()
        seen.append(work)
        if command[0] == "gpu-queue":
            assert command[:7] == ["gpu-queue", "run", "--proof", "--minutes", "120", "--", "build-queue"]
            (work / "hex.log").write_text("actual model trace retained\n")
            (work / "hex-output").mkdir()
            return SimpleNamespace(returncode=124)
        assert command[2:4] == ["woof.hex.cli", "verify-mpas"]
        assert command[command.index("--failed-model-exit") + 1] == "124"
        partial = Path(command[command.index("--hex-output") + 1])
        assert partial.is_relative_to(temporary_root)
        verifier.write_json(Path(command[command.index("--receipt") + 1]), {
            "passed": False, "evidence_kind": "FORECAST-FAILED", "process_count": 1,
            "failed_model_exit": 124, "comparison": {"fields": []}})
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(verifier.subprocess, "run", execute)
    assert verifier.gate(args) == 1
    result = json.loads((out / "gate.json").read_text())
    assert not result["passed"] and result["stages"]["G2"]["decks"][0]["exit_code"] == 124
    assert (out / "G2-0-model.log").read_text() == "actual model trace retained\n"
    assert len(seen) == 2 and all(not path.exists() for path in seen)


@pytest.mark.parametrize("complete_comparison,existing_output", [(False, False), (True, False), (True, True)])
def test_forecast_output_is_retired_only_after_complete_word_receipt(tmp_path, monkeypatch, complete_comparison, existing_output):
    from woof.hex import verify_mpas as verifier
    from types import SimpleNamespace
    import json
    output = tmp_path / "hex-output"
    output.mkdir()
    evidence = output / "closed-output.nc"
    evidence.write_bytes(b"closed forecast evidence")
    args = SimpleNamespace(receipt=tmp_path / "receipt.json", work_root=tmp_path,
                           hex_output=output if existing_output else None)
    def attempt(arguments):
        arguments._owns_work = True
        arguments._context = {"process_count": 1}
        if not complete_comparison:
            raise verifier.VerificationRefusal("native output disappeared after forecast completion")
        return {"passed": False, "process_count": 1, "comparison": {"fields": [
            {"field": "theta", "differing_words": 1}]}}
    monkeypatch.setattr(verifier, "verify", attempt)
    assert verifier.run(args) == 1
    receipt = json.loads(args.receipt.read_text())
    assert receipt["process_count"] == 1
    assert output.exists() == (existing_output or not complete_comparison)
    if output.exists():
        assert evidence.read_bytes() == b"closed forecast evidence"
    if not complete_comparison:
        assert receipt["evidence_kind"] == "REFUSAL" and "comparison" not in receipt


@pytest.mark.parametrize("stage", ["G1", "G2"])
def test_recorded_missing_gate_retains_inventory_and_retires_allowance(stage):
    from copy import deepcopy
    from woof.hex.verify_mpas import recorded_missing_failures
    deck = {**{k: CONTEXT[k] for k in ("pin_set", "mesh", "date")}, "id": CONTEXT["deck"],
            "process_count": 1, "expected_samples": {"history": [0, 1]}, "duration_steps": 1}
    rows = [field(), field(1), {**field(words=3, status="field missing in hex"), "field": "pv"},
            {**field(1, 3, "field missing in hex"), "field": "pv"}]
    baseline = aggregate(rows, CONTEXT)
    receipt = {**CONTEXT, "passed": True, "evidence_kind": "FORECAST", "tier": None,
               "acceptance_tier": "BITWISE-WRITTEN-FIELDS", "comparison": {"fields": rows}}
    assert not recorded_missing_failures(receipt["comparison"], CONTEXT, baseline)
    assert not gate_receipt_failures(receipt, deck, stage, baseline)
    assert any("diagnostic grades" in f or "missing" in f
               for f in gate_receipt_failures(receipt, deck, stage, {"schema": "mpas-ratchet-v1", "entries": []}))
    fixed = deepcopy(rows)
    for row in fixed:
        row.update(status="compared", differing_words=0)
    fixed_receipt = {**receipt, "tier": "BITWISE", "acceptance_tier": "BITWISE", "comparison": {"fields": fixed}}
    assert not gate_receipt_failures(fixed_receipt, deck, stage, baseline)
    adopted = aggregate(fixed, CONTEXT)
    assert gate_receipt_failures(receipt, deck, stage, adopted)  # cannot disappear after owner's landing
    assert gate_receipt_failures(receipt, deck, "G3", baseline)  # no foundation waiver in the nightly gate


@pytest.mark.parametrize("defect", ["numeric", "new-field", "corrupt", "shape", "time", "rise", "rank", "failed-model", "context", "all-missing"])
def test_recorded_missing_gate_rejects_every_unrecorded_or_computed_defect(defect):
    from copy import deepcopy
    deck = {**{k: CONTEXT[k] for k in ("pin_set", "mesh", "date")}, "id": CONTEXT["deck"],
            "process_count": 1, "expected_samples": {"history": [0, 1]}, "duration_steps": 1}
    original = [field(), field(1), {**field(words=3, status="field missing in hex"), "field": "pv"}]
    baseline = aggregate(original, CONTEXT)
    rows = deepcopy(original)
    receipt = {**CONTEXT, "passed": True, "evidence_kind": "FORECAST", "tier": None,
               "acceptance_tier": "BITWISE-WRITTEN-FIELDS", "comparison": {"fields": rows}}
    if defect == "numeric":
        # Even an old nonzero numerical baseline cannot authorize a computed difference.
        rows[0]["differing_words"] = 1
        baseline = aggregate(rows, CONTEXT)
    elif defect == "new-field": rows[-1]["field"] = "new-pv"
    elif defect == "corrupt": rows[-1]["status"] = "unreadable hex output"
    elif defect == "shape": rows[-1]["status"] = "type or dimensions differ in hex"
    elif defect == "time": rows[-1]["status"] = "output time missing in hex"
    elif defect == "rise": rows.append({**rows[-1], "step": 1})
    elif defect == "rank": baseline["entries"][-1]["process_count"] = 2
    elif defect == "failed-model": receipt["failed_model_exit"] = 143
    elif defect == "context": receipt["pin_set"] = "other-pins"
    elif defect == "all-missing":
        for row in rows: row.update(status="field missing in hex", differing_words=3)
        baseline = aggregate(rows, CONTEXT)
    assert gate_receipt_failures(receipt, deck, "G1", baseline)
