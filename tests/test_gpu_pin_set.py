"""The declared GPU pin set is real, and the card stage reads it.

tests/gpu_pin_set.txt is the ONE place the set lives: tools/release/
precut_gpu_gate.py runs it on the release node's card before a cut, and this
file holds it to the tree, the way tools/release/precut_gate.contract_files
reads the publication workflow.  The controls below drive the gate's main()
with every remote and archive operation replaced by a recorded response; no
connection, device or project suite is started.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import pytest
from _dycore_oracle_routes import DYCORE_PARITY_FILES

ROOT = Path(__file__).resolve().parents[1]
HEAD = "0123456789abcdef0123456789abcdef01234567"
REMOTE = "/tmp/precut-gpu-control"
#: The pins that shipped red on the 2.7.4 and 2.7.5 tips with nothing running
#: them, and the Omega column kernel's bit-identity file, whose readings sit
#: under tests/data/receipts/omega-column-scan/.
REQUIRED = {"tests/test_coriolis_map.py", "tests/test_mp8_frozen.py",
            "tests/test_omega_column_scan.py"}
#: Compiled WRF diffusion fixtures and every full-output device word pin
#: must stay in the release card stage. Dropping one retires a routine's
#: independent Fortran comparison while leaving its source tests green.
WRF_DIFFUSION_REQUIRED = {
    "tests/test_diff6_wrf471_parity.py",
    "tests/test_deformation_wrf471_parity.py",
    "tests/test_horizontal_diffusion_wrf471_parity.py",
    "tests/test_vertical_diffusion_wrf471_parity.py",
    "tests/test_constant_diffusion_wrf471_parity.py",
    "tests/test_diffusion_drivers_wrf471_parity.py",
}
#: The pins the enlarged set found on the release node after the stage's
#: first run: three of them red there (the Shin-Hong ULP row and the bl=1
#: run-state hash keyed by an NVRTC build nobody had recorded, and the SASE
#: device golden pinned on one card), four green.
KEYED_BY_COMPILER_OR_CARD = {
    "tests/test_shinhong_runtime.py::test_off_path_bl1_micro_run_state_hash_is_pinned",
    "tests/test_shinhong_wrf461_parity.py::test_shinhong_cuda_column_holds_its_measured_distance_from_wrf",
    "tests/test_sase_gpu.py::test_dynamic_solve_device_real_lift_golden",
    "tests/test_feedback.py::test_feedback_zero_output_is_pinned_and_costs_nothing",
    "tests/test_ysu_wrf461_parity.py::test_ysu_cuda_column_holds_its_measured_distance_from_wrf",
    "tests/test_ysu_wrf461_parity.py::test_ysu_is_bitwise_wrf_outside_the_flushed_subnormal_lanes",
    "tests/test_noah_wrf461_parity.py::test_noah_cuda_column_holds_its_measured_distance_from_wrf",
}
#: Two more max-ULP tables asserted for equality that the set had missed
#: while drawing other rows from their files: the Shin-Hong partition curves
#: (one flat table, keyed by neither compiler nor card) and the UH kernel's
#: table pinned at zero.  Both green on the release node's card.
EQUALITY_ULP_TABLES = {
    "tests/test_shinhong_wrf461_parity.py::test_shinhong_cuda_partition_curves_hold_their_distance_from_glibc",
    "tests/test_uh_wrf461_parity.py::test_cuda_kernel_matches_the_oracle_at_the_pinned_ulp",
}
#: Every Tiedtke kernel row that asserts a DEVICE result bitwise equal to
#: the Fortran oracle's recorded columns.  They are pins for the card
#: stage's purpose by the same reasoning the stage exists for: a device
#: result asserted for equality that no gate runs is what shipped red
#: twice.  Declared per row rather than per file because the same files
#: carry CPU rows comparing the numpy mirror with the oracle, and those
#: open no device.
NTIEDTKE_DEVICE_BITWISE = {
    "tests/test_ntiedtke_adjust_parity.py::test_kernel_level_outputs_are_bitwise",
    "tests/test_ntiedtke_adjust_parity.py::test_kernel_surface_fluxes_are_bitwise",
    "tests/test_ntiedtke_cloud_depth_parity.py::test_kernel_flip_and_sum_are_exact",
    "tests/test_ntiedtke_cuascn_parity.py::test_kernel_level_outputs_are_bitwise",
    "tests/test_ntiedtke_cuascn_parity.py::test_kernel_integer_outputs_and_wup_are_exact",
    "tests/test_ntiedtke_cudlfsn_parity.py::test_kernel_level_outputs_are_bitwise",
    "tests/test_ntiedtke_cudlfsn_parity.py::test_kernel_scalars_are_exact",
    "tests/test_ntiedtke_cudlfsn_parity.py::test_the_kernel_LEAVES_untouched_class2_levels_alone",
    "tests/test_ntiedtke_cuflxn_parity.py::test_kernel_level_outputs_are_bitwise",
    "tests/test_ntiedtke_cuflxn_parity.py::test_kernel_precipitation_fluxes_are_bitwise",
    "tests/test_ntiedtke_cuflxn_parity.py::test_kernel_prain_is_bitwise",
    "tests/test_ntiedtke_kedis_parity.py::test_kernel_ptte_is_bitwise",
    "tests/test_ntiedtke_mprofile_parity.py::test_kernel_outputs_are_bitwise",
    "tests/test_ntiedtke_mrescale_parity.py::test_kernel_outputs_are_bitwise",
    "tests/test_ntiedtke_prep_parity.py::test_cuda_cuinin_is_bitwise",
    "tests/test_ntiedtke_prep_parity.py::test_cuda_cutypen_is_bitwise_across_all_ktype_arms",
    "tests/test_ntiedtke_prep_parity.py::test_cuda_cubasmcn_is_bitwise_on_both_sides_of_the_guard",
    "tests/test_ntiedtke_prep_parity.py::test_cuda_mfub_is_bitwise",
    "tests/test_ntiedtke_prep_parity.py::test_cuda_closure_is_bitwise_across_every_arm",
    "tests/test_ntiedtke_prep_parity.py::test_cuda_reproduces_every_prep_word",
    "tests/test_ntiedtke_prep_parity.py::test_cuda_matches_the_mirror_exactly",
    "tests/test_ntiedtke_prep_parity.py::test_cuda_scale_factors_survive_both_branches_in_one_launch",
    "tests/test_ntiedtke_uscale_parity.py::test_kernel_level_outputs_are_bitwise",
    "tests/test_ntiedtke_uscale_parity.py::test_kernel_scalar_outputs_are_exact",
}
#: The rows in the same files that reach a device and are NOT results, so
#: that the classification below is exhaustive and a new row cannot join a
#: file and stay unclassified.  Each would be worth running; none is a pin.
#:
#: The 20 local-frame rows assert a compiled kernel's local_size_bytes is
#: 0, which is a launch-cost property of the artifact rather than a value
#: the oracle has an opinion about, and they are recognised by that
#: attribute rather than by name.  The four named below assert a
#: behaviour with no oracle number behind it: a seeded array comes back
#: zeroed, a sentinel is gone everywhere, deep-only slots are left alone
#: on columns that did not take the deep branch, and cutypen's parcel
#: outputs and final scratch are one bit pattern at 32, 64 and 128 threads
#: per block.  That last row (5310e64b2, the default-pieces speed lane's
#: trial pruning) compares the kernel with itself at three launch widths;
#: the oracle's numbers are held by the cutypen rows above.
NTIEDTKE_DEVICE_NOT_A_RESULT = {
    "tests/test_ntiedtke_cloud_depth_parity.py::test_kernel_zeroes_the_downdraft_arrays",
    "tests/test_ntiedtke_mrescale_parity.py::test_the_kernel_writes_every_level",
    "tests/test_ntiedtke_prep_parity.py::test_the_kernel_leaves_deep_only_slots_alone_on_other_columns",
    "tests/test_ntiedtke_prep_parity.py::test_cutypen_launch_widths_preserve_outputs_and_final_scratch",
}
#: What makes a device row a launch-cost row rather than a result row.
LAUNCH_COST_MARK = "local_size_bytes"
#: The test whose exclusion is an argument rather than a shrug, so the
#: argument is checked: it imports these four names from the frozen
#: module and re-hashes the same files, which is why running it on the
#: card would learn nothing that tests/test_mp8_frozen.py does not.
MP8_REASSERTION = (
    "tests/test_thompson_aerosol_gpu.py",
    "test_mp8_freeze_receipt_still_holds_at_the_wp12b_tip",
    ("THOMPSON_CU_SHA256", "THOMPSON_COMPILED_SOURCE_SHA256",
     "THOMPSON_PY_SHA256", "TABLE_SET_ID_PIN"))
CARD = {"name": "Control Card", "compute_capability": "8.9", "cupy": "14.2.0",
        "cuda_runtime": 13020, "cuda_driver": 13020, "device_count": 1,
        "nvrtc": "13.3.33", "nvrtc_build_id": "CL-0", "nvrtc_library_sha256": "0" * 64,
        "compiles": True}
OTHER = "4242, /opt/other-venv/bin/python, 3176 MiB"
#: Spelled once so the scans below can be written without an escape
#: inside the source they are reading.
NEWLINE = chr(10)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("precut_gpu_gate_control", ROOT / "tools/release/precut_gpu_gate.py")


def test_the_declared_set_names_the_pins_that_shipped_red():
    files = set(gate.pin_files(gate.pin_set(ROOT)))
    assert REQUIRED <= files, sorted(REQUIRED - files)
    entries = set(gate.pin_set(ROOT))
    assert KEYED_BY_COMPILER_OR_CARD <= entries, sorted(KEYED_BY_COMPILER_OR_CARD - entries)
    assert EQUALITY_ULP_TABLES <= entries, sorted(EQUALITY_ULP_TABLES - entries)
    assert NTIEDTKE_DEVICE_BITWISE <= entries, sorted(
        NTIEDTKE_DEVICE_BITWISE - entries)


def test_the_declared_set_keeps_every_compiled_wrf_diffusion_oracle():
    files = set(gate.pin_files(gate.pin_set(ROOT)))
    assert WRF_DIFFUSION_REQUIRED <= files, sorted(WRF_DIFFUSION_REQUIRED - files)


def test_the_declared_set_keeps_every_compiled_wrf_dycore_oracle():
    files = set(gate.pin_files(gate.pin_set(ROOT)))
    assert DYCORE_PARITY_FILES <= files, sorted(DYCORE_PARITY_FILES - files)


def test_compiled_wrf_dycore_cpu_fixture_rows_remain_selectable():
    """A CuPy import in a helper must not retire the CPU fixture checks."""
    import ast
    from conftest import _cupy_scope

    for file in sorted(DYCORE_PARITY_FILES):
        path = ROOT / file
        whole, _functions = _cupy_scope(str(path))
        assert not whole, f"{file}: a device import marks every CPU fixture row GPU"
        source = path.read_text(encoding="utf-8")
        fixture_rows = []
        for node in ast.parse(source).body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            decorators = [ast.get_source_segment(source, value) or ""
                          for value in node.decorator_list]
            is_device = "requires_gpu" in decorators
            is_marked = "pytest.mark.gpu" in decorators
            if is_device:
                assert is_marked, f"{file}::{node.name}: device row survives the CPU marker selection"
            elif not is_marked:
                fixture_rows.append(node.name)
        assert fixture_rows, f"{file}: no CPU fixture checks remain"


def test_every_ntiedtke_device_row_in_the_tree_is_classified():
    """Every row that reaches a card is a declared pin or a stated reason.

    A row of this shape that is in neither table is a device result
    asserted for equality that no gate runs, which is the defect the card
    stage exists for.  So the question is not "are the declared ones
    there" but "is anything unaccounted for": the scan finds every test
    in these files that reaches a device, and the two tables have to
    cover it exactly.
    """
    import ast

    reaching = set()
    launch_cost = set()
    for path in sorted((ROOT / "tests").glob("test_ntiedtke_*parity*.py")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tree = ast.parse(text)
        device_fixtures = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = NEWLINE.join(lines[node.lineno - 1:node.end_lineno])
            decorated = NEWLINE.join(lines[d.lineno - 1]
                                     for d in node.decorator_list)
            if "fixture" in decorated and "importorskip" in body:
                device_fixtures.add(node.name)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test_")):
                continue
            body = NEWLINE.join(lines[node.lineno - 1:node.end_lineno])
            arguments = {argument.arg for argument in node.args.args}
            if not (arguments & device_fixtures or "importorskip" in body):
                continue
            entry = f"tests/{path.name}::{node.name}"
            reaching.add(entry)
            if LAUNCH_COST_MARK in body:
                launch_cost.add(entry)

    assert len(reaching) > 40, (
        f"the scan found only {len(reaching)} device rows; it is broken and "
        "every count below is vacuous")
    assert launch_cost, "no launch-cost rows found; the mark has moved"
    results = reaching - launch_cost
    assert results == NTIEDTKE_DEVICE_BITWISE | NTIEDTKE_DEVICE_NOT_A_RESULT, {
        "unclassified": sorted(
            results - NTIEDTKE_DEVICE_BITWISE - NTIEDTKE_DEVICE_NOT_A_RESULT),
        "declared but gone": sorted(
            (NTIEDTKE_DEVICE_BITWISE | NTIEDTKE_DEVICE_NOT_A_RESULT)
            - results)}
    assert not (NTIEDTKE_DEVICE_BITWISE & NTIEDTKE_DEVICE_NOT_A_RESULT)
    assert not (NTIEDTKE_DEVICE_NOT_A_RESULT & set(gate.pin_set(ROOT))), (
        "a row excluded with a reason is also declared as a pin")


def test_the_mp8_re_assertion_is_excluded_on_an_argument_that_holds():
    """It cannot be red where tests/test_mp8_frozen.py is green."""
    file, name, imported = MP8_REASSERTION
    entries = set(gate.pin_set(ROOT))
    assert not any(entry.startswith(f"{file}::{name}") or entry == file
                   for entry in entries)
    source = (ROOT / file).read_text(encoding="utf-8")
    assert f"def {name}(" in source, f"{name} is gone; retire the exclusion"
    body = source.split(f"def {name}(", 1)[1].split(NEWLINE + "def ", 1)[0]
    assert "import test_mp8_frozen as frozen" in body
    for value in imported:
        assert f"frozen.{value}" in body, (
            f"{name} no longer takes {value} from the frozen module, so the "
            "reason it is excluded has stopped being true")
    frozen_source = (ROOT / "tests" / "test_mp8_frozen.py").read_text(
        encoding="utf-8")
    for value in imported:
        assert f"{value}" in frozen_source


def test_every_entry_resolves_to_a_test_in_the_tree():
    entries = gate.pin_set(ROOT)
    assert len(entries) == len(set(entries)), "duplicate entries"
    for entry in entries:
        file, _, name = entry.partition("::")
        source = (ROOT / file).read_text(encoding="utf-8")
        if name:
            assert re.search(rf"^def {re.escape(name)}\(", source, re.M), (
                f"{entry} names a test that is not defined in {file}")


def test_every_entry_is_a_gpu_pin_or_the_mp8_freeze():
    for file in gate.pin_files(gate.pin_set(ROOT)):
        source = (ROOT / file).read_text(encoding="utf-8")
        assert (file == "tests/test_mp8_frozen.py" or "cupy" in source
                or "requires_gpu" in source), (
            f"{file} opens no device; it does not belong in the card stage")


def test_the_card_stage_carries_no_second_copy_of_the_machinery():
    source = (ROOT / "tools/release/precut_gpu_gate.py").read_text(encoding="utf-8")
    for name in ("candidate_commit", "ship_candidate", "identity_probe",
                 "remote_prefix", "remote_test_command", "read_report",
                 "completed_result", "remote_directory", "save_evidence",
                 "evidence_path", "home_directory_redaction",
                 "without_machine_paths", "record_name"):
        assert callable(getattr(gate.base, name)), name
        assert not re.search(rf"^def {name}\(", source, re.M), (
            f"precut_gpu_gate.py defines its own {name}")
    assert gate.completed_result is gate.base.completed_result


def _report(rows):
    """A completed JUnit report; rows are (file, name, outcome, message)."""
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", name="pytest", tests=str(len(rows)),
                          errors="0",
                          failures=str(sum(r[2] == "failure" for r in rows)),
                          skipped=str(sum(r[2] == "skipped" for r in rows)))
    for file, name, outcome, message in rows:
        row = ET.SubElement(suite, "testcase", file=file, name=name,
                            classname=file.removesuffix(".py").replace("/", "."))
        if outcome != "passed":
            ET.SubElement(row, outcome, message=message).text = message
    return ET.tostring(root, encoding="unicode")


ENTRIES = ["tests/test_a.py::test_pin", "tests/test_b.py"]
GREEN = _report([("tests/test_a.py", "test_pin", "passed", ""),
                 ("tests/test_b.py", "test_other", "passed", "")])


def _probe(card=CARD, apps="", extra=""):
    return (extra + "WOOF-CARD " + json.dumps(card) + "\nARWEN-SMI\nControl Card, 610.57.04\n"
            "ARWEN-APPS\n" + apps)


def _sampler(during="", after=""):
    return f"ARWEN-APPS-DURING\n{during}ARWEN-APPS-AFTER\n{after}ARWEN-APPS-END\n"


def drive(tmp_path, monkeypatch, *, xml=GREEN, child_exit=0, card_output=None,
          entries=ENTRIES, recheck=(), sampler=None, remote=REMOTE):
    """recheck: the idle re-check's answers in order (the last one repeats);
    sampler: the DURING/AFTER blocks the pin run prints, None for the
    complete idle record and "" for a run that printed none; remote: the
    scratch root, which a caller sets to see what reaches the receipt."""
    tree = tmp_path / "tree"
    (tree / ".git").mkdir(parents=True, exist_ok=True)
    calls = []
    probe = _probe() if card_output is None else card_output
    sampler = _sampler() if sampler is None else sampler
    rechecks = list(recheck)
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append(command)
        rc, stdout = 0, ""
        if command[0] == "git":
            if "rev-parse" in command:
                stdout = HEAD + "\n"
            elif "status" in command:
                stdout = ""
            elif "archive" in command:
                target = next(v.split("=", 1)[1] for v in command if v.startswith("--output="))
                Path(target).write_bytes(b"archive stand-in")
            else:
                raise AssertionError(command)
        elif command[0] == "ssh":
            last = command[-1]
            if last.startswith("cat -- "):
                stdout = xml
            elif "-m pytest" in last:
                rc, stdout = child_exit, remote + "/woof/__init__.py\npins ran\n" + sampler
            elif "ARWEN-CARD" in last:
                stdout = probe
            elif last.startswith("echo WOOF-APPS;"):
                answer = rechecks.pop(0) if len(rechecks) > 1 else (rechecks[0] if rechecks else "")
                stdout = "ARWEN-APPS\n" + answer
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, rc, stdout, "")

    monkeypatch.setattr(gate.base.subprocess, "run", fake_run)
    monkeypatch.setattr(gate.time, "sleep", sleeps.append)
    monkeypatch.setattr(gate, "IDLE_WAIT_SECONDS", 90)
    if entries is not None:
        monkeypatch.setattr(gate, "pin_set_with_source",
                            lambda _: (list(entries), "candidate"))
    rc = gate.main(["--tree", str(tree), "--node", "build-account",
                    "--python", "/never/run/python", "--remote-dir", remote,
                    "--evidence-dir", str(tmp_path / "evidence")])
    receipts = sorted((tmp_path / "evidence").glob(f"precut-gpu-gate-{HEAD[:9]}*.json"),
                      key=lambda p: p.stat().st_mtime_ns)
    receipt = json.loads(receipts[-1].read_text()) if receipts else {}
    return rc, receipt, calls


def test_a_completed_run_passes_on_the_card_serially_and_records_it(tmp_path, monkeypatch):
    rc, receipt, calls = drive(tmp_path, monkeypatch)
    assert rc == 0
    assert receipt["schema"] == "arwen.precut-gpu-gate.v1"
    assert receipt["status"] == "PASS"
    assert receipt["candidate"] == HEAD
    assert receipt["tests"] == ENTRIES
    card = receipt["card"]
    assert card["name"] == "Control Card"
    assert card["driver"] == "610.57.04"
    assert card["compute_capability"] == "8.9"
    assert card["cuda_runtime"] == 13020
    assert card["nvrtc"] == "13.3.33"
    assert card["nvrtc_library_sha256"] == "0" * 64
    assert card["other_compute_processes_at_probe"] == []
    assert card["other_compute_processes"] == []
    assert card["other_compute_processes_during"] == []
    assert card["other_compute_processes_after"] == []
    assert card["idle_wait_seconds"] == 0
    assert receipt["skipped"] == []
    assert "previous_receipt" not in receipt
    remote_test_command = next(c[-1] for c in calls if c[0] == "ssh" and "-m pytest" in c[-1])
    assert "-n 0" in remote_test_command
    assert "GPUWM_NO_LOCAL_GPU" not in remote_test_command
    assert " -m " not in remote_test_command.split("-m pytest", 1)[1].split(" &", 1)[0]
    assert "-p tools.release.precut_gate" in remote_test_command
    for entry in ENTRIES:
        assert entry in remote_test_command
    # The card is sampled while the pins run, the pytest process and its
    # children excepted, and the command exits with pytest's own status.
    assert "pgrep -P" in remote_test_command
    assert "ARWEN-APPS-DURING" in remote_test_command
    assert remote_test_command.endswith('exit "$ARWEN_RC"')
    # The card is read on the shipped tree, through the same identity probe,
    # and the probe compiles a kernel and reads the tree's own fingerprint.
    probe_command = next(c[-1] for c in calls if c[0] == "ssh" and "ARWEN-CARD" in c[-1])
    assert "woof.__file__" in probe_command
    assert "GPUWM_NO_LOCAL_GPU" not in probe_command
    assert "compile_platform_fingerprint" in probe_command
    assert "ElementwiseKernel" in probe_command


def test_a_skipped_pin_is_named_in_the_receipt_with_its_reason(tmp_path, monkeypatch):
    reason = ("the phase-2 step capture is per card and none is committed for "
              "'Control Card'")
    xml = _report([("tests/test_a.py", "test_pin", "skipped", reason),
                   ("tests/test_b.py", "test_other", "passed", "")])
    rc, receipt, _ = drive(tmp_path, monkeypatch, xml=xml)
    assert rc == 0
    assert receipt["skipped"] == [{"test": "tests/test_a.py::test_pin", "reason": reason}]


@pytest.mark.parametrize("reason", ["no CUDA GPU / cupy",
                                    "GPUWM_NO_LOCAL_GPU=1: GPU work belongs on the rented device",
                                    "device verification needs CUDA",
                                    "no CUDA device"])
def test_a_skip_that_saw_no_card_refuses_the_stage(tmp_path, monkeypatch, reason):
    xml = _report([("tests/test_a.py", "test_pin", "skipped", reason),
                   ("tests/test_b.py", "test_other", "passed", "")])
    rc, receipt, _ = drive(tmp_path, monkeypatch, xml=xml)
    assert rc != 0
    assert receipt["status"] == "ERROR"
    assert "no card" in receipt["error"]


def test_a_red_pin_fails_the_candidate(tmp_path, monkeypatch):
    xml = _report([("tests/test_a.py", "test_pin", "failure", "assert False"),
                   ("tests/test_b.py", "test_other", "passed", "")])
    rc, receipt, _ = drive(tmp_path, monkeypatch, xml=xml, child_exit=1)
    assert rc == 1
    assert receipt["status"] == "FAIL"
    assert receipt["failures"] == ["tests/test_a.py::test_pin"]


def test_a_node_without_a_card_cannot_run_the_stage(tmp_path, monkeypatch):
    rc, receipt, calls = drive(tmp_path, monkeypatch,
                               card_output="ModuleNotFoundError: No module named 'cupy'\n")
    assert rc != 0
    assert "no usable CUDA device" in receipt["error"]
    assert not any(c[0] == "ssh" and "-m pytest" in c[-1] for c in calls)


def test_an_interpreter_without_a_toolkit_refuses_the_stage_naming_it(tmp_path, monkeypatch):
    """getDeviceProperties needs no headers and no NVRTC, so a cupy with no
    toolkit answered the probe and the stage's first real run recorded every
    pin red with the same RuntimeError instead of refusing."""
    card = dict(CARD, compiles=False, nvrtc="unresolved")
    output = _probe(card, extra="WOOF-CARD-NO-COMPILE RuntimeError: Failed to find CUDA headers. "
                                "Please install the CUDA Toolkit or set CUDA_PATH\n")
    rc, receipt, calls = drive(tmp_path, monkeypatch, card_output=output)
    assert rc != 0
    assert receipt["status"] == "ERROR"
    assert "cannot compile a CUDA kernel" in receipt["error"]
    assert "Failed to find CUDA headers" in receipt["error"]
    assert "--cuda-path" in receipt["error"]
    assert not any(c[0] == "ssh" and "-m pytest" in c[-1] for c in calls)


@pytest.mark.parametrize("child_exit", [2, 137, 255])
def test_an_interrupted_pin_process_never_passes(tmp_path, monkeypatch, child_exit):
    rc, receipt, _ = drive(tmp_path, monkeypatch, child_exit=child_exit)
    assert rc != 0
    assert receipt["status"] == "ERROR"


def test_a_card_shared_at_the_probe_is_waited_for_and_then_run(tmp_path, monkeypatch):
    output = _probe(apps=OTHER + "\n")
    rc, receipt, calls = drive(tmp_path, monkeypatch, card_output=output,
                               recheck=[OTHER + "\n", OTHER + "\n", ""])
    assert rc == 0
    assert receipt["status"] == "PASS"
    assert receipt["card"]["other_compute_processes_at_probe"] == [OTHER]
    assert receipt["card"]["other_compute_processes"] == []
    assert receipt["card"]["idle_wait_seconds"] == 3 * gate.IDLE_POLL_SECONDS
    rechecks = [c for c in calls if c[0] == "ssh" and c[-1].startswith("echo WOOF-APPS;")]
    assert len(rechecks) == 3
    # The pins ran only after the card was free.
    pytest_index = next(i for i, c in enumerate(calls) if c[0] == "ssh" and "-m pytest" in c[-1])
    assert all(i < pytest_index for i, c in enumerate(calls)
               if c[0] == "ssh" and c[-1].startswith("echo WOOF-APPS;"))


def test_a_card_still_shared_at_the_bound_refuses_the_stage(tmp_path, monkeypatch):
    output = _probe(apps=OTHER + "\n")
    rc, receipt, calls = drive(tmp_path, monkeypatch, card_output=output,
                               recheck=[OTHER + "\n"])
    assert rc != 0
    assert receipt["status"] == "ERROR"
    assert "discarded reading" in receipt["error"]
    assert "other-venv/bin/python" in receipt["error"]
    assert receipt["card"]["name"] == "Control Card"
    assert not any(c[0] == "ssh" and "-m pytest" in c[-1] for c in calls)


@pytest.mark.parametrize("during,after", [(OTHER + "\n", ""), ("", OTHER + "\n")])
def test_a_process_that_shared_the_card_while_the_pins_ran_discards_the_reading(tmp_path, monkeypatch, during, after):
    """Four of the stage's first five real runs recorded another process
    holding 2.5 to 3.3 GB of the card and were still PASS receipts; a
    capture taken while another process shares the card is a discarded
    reading, whatever the pins said."""
    rc, receipt, calls = drive(tmp_path, monkeypatch, sampler=_sampler(during, after))
    assert rc != 0
    assert receipt["status"] == "ERROR"
    assert "discarded reading" in receipt["error"]
    assert "other-venv/bin/python" in receipt["error"]
    assert receipt["card"]["other_compute_processes_during"] == ([OTHER] if during else [])
    assert receipt["card"]["other_compute_processes_after"] == ([OTHER] if after else [])
    assert "counts" not in receipt
    assert not any(c[0] == "ssh" and c[-1].startswith("cat -- ") for c in calls)


def test_a_run_without_the_sampler_record_is_refused(tmp_path, monkeypatch):
    rc, receipt, _ = drive(tmp_path, monkeypatch, sampler="")
    assert rc != 0
    assert receipt["status"] == "ERROR"
    assert "sampler" in receipt["error"]


def test_a_sampled_process_is_one_row_however_often_it_was_seen(tmp_path, monkeypatch):
    during = "4242, /opt/other-venv/bin/python, 2500 MiB\n" + OTHER + "\n"
    rc, receipt, _ = drive(tmp_path, monkeypatch, sampler=_sampler(during, ""))
    assert rc != 0
    assert receipt["card"]["other_compute_processes_during"] == [OTHER]


def test_a_repeated_run_for_one_commit_keeps_the_earlier_receipt(tmp_path, monkeypatch):
    """A discarded card-stage reading must stay on record beside the run
    that replaced it, so a second run for the same commit writes a numbered
    receipt naming the one it follows."""
    rc, first, _ = drive(tmp_path, monkeypatch, sampler=_sampler(OTHER + "\n", ""))
    assert rc != 0 and first["status"] == "ERROR"
    first_path = tmp_path / "evidence" / f"precut-gpu-gate-{HEAD[:9]}.json"
    first_sha = hashlib.sha256(first_path.read_bytes()).hexdigest()
    rc, second, _ = drive(tmp_path, monkeypatch)
    assert rc == 0 and second["status"] == "PASS"
    assert (tmp_path / "evidence" / f"precut-gpu-gate-{HEAD[:9]}.2.json").exists()
    assert json.loads(first_path.read_text())["status"] == "ERROR", "the earlier receipt was replaced"
    assert second["previous_receipt"] == {"path": first_path.name, "sha256": first_sha}
    rc, third, _ = drive(tmp_path, monkeypatch)
    assert (tmp_path / "evidence" / f"precut-gpu-gate-{HEAD[:9]}.3.json").exists()
    assert third["previous_receipt"]["path"] == f"precut-gpu-gate-{HEAD[:9]}.2.json"


def test_a_candidate_that_predates_the_set_runs_the_gate_tree_copy_and_says_so(tmp_path, monkeypatch):
    """integrate/2.7.6 at fc639c51f has no tests/gpu_pin_set.txt; the stage
    still runs there, with the gate's own copy, and the receipt names it."""
    real = gate.pin_set(ROOT)
    files = gate.pin_files(real)
    rows = [(file, "test_control", "passed", "") for file in files]
    tree = tmp_path / "tree"
    for file in files:
        (tree / file).parent.mkdir(parents=True, exist_ok=True)
        (tree / file).write_text("def test_control():\n    assert True\n", encoding="utf-8")
    assert not (tree / gate.PIN_SET).exists()
    rc, receipt, calls = drive(tmp_path, monkeypatch, xml=_report(rows), entries=None)
    assert rc == 0
    assert receipt["tests"] == real
    # Named by the COMMIT the copy sits at, not by the path it sits in: a
    # path said which worktree on which machine held it, which is a
    # machine path in a committed receipt and the less useful of the two
    # answers, since the commit is the same wherever it is checked out.
    assert receipt["pin_set_source"] == "gate tree " + HEAD
    remote_test_command = next(c[-1] for c in calls if c[0] == "ssh" and "-m pytest" in c[-1])
    for entry in real:
        assert entry in remote_test_command


def test_the_node_interpreter_survives_git_bash_path_conversion(tmp_path, monkeypatch):
    """Git Bash rewrites a leading-slash argument into its own prefix; the
    stage's first real run sent C:/Program Files/Git/opt/.../python to the
    node.  The remote command must carry the POSIX path."""
    tree = tmp_path / "tree"
    (tree / ".git").mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        stdout = ""
        if command[0] == "git" and "rev-parse" in command:
            stdout = HEAD + "\n"
        elif command[0] == "git" and "archive" in command:
            target = next(v.split("=", 1)[1] for v in command if v.startswith("--output="))
            Path(target).write_bytes(b"archive stand-in")
        elif command[0] == "ssh" and command[-1].startswith("cat -- "):
            stdout = GREEN
        elif command[0] == "ssh" and "-m pytest" in command[-1]:
            stdout = "pins ran\n" + _sampler()
        elif command[0] == "ssh" and "ARWEN-CARD" in command[-1]:
            stdout = _probe()
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(gate.base.subprocess, "run", fake_run)
    monkeypatch.setattr(gate, "pin_set_with_source", lambda _: (list(ENTRIES), "candidate"))
    rc = gate.main(["--tree", str(tree), "--node", "build-account",
                    "--python", "C:/Program Files/Git/opt/venv/bin/python",
                    "--remote-dir", "C:/Program Files/Git" + REMOTE,
                    "--cuda-path", "C:/Program Files/Git/opt/cuda13",
                    "--evidence-dir", str(tmp_path / "evidence")])
    assert rc == 0
    remote_commands = [c[-1] for c in calls if c[0] == "ssh"]
    assert all("Program Files" not in command for command in remote_commands), remote_commands
    assert any("/opt/venv/bin/python -m pytest" in command for command in remote_commands)
    assert any("CUDA_PATH=/opt/cuda13" in command for command in remote_commands)
    assert gate.base.remote_path("/opt/venv/python") == "/opt/venv/python"
    assert gate.base.remote_path(None) is None


def test_a_candidate_with_its_own_set_runs_that_set(tmp_path):
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests/test_own.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    (tree / gate.PIN_SET).write_text("# own\ntests/test_own.py::test_x\n", encoding="utf-8")
    assert gate.pin_set_with_source(tree) == (["tests/test_own.py::test_x"], "candidate")


def test_an_entry_whose_file_is_missing_is_refused(tmp_path):
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / gate.PIN_SET).write_text("tests/test_gone.py::test_x\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="not a test file in the candidate"):
        gate.pin_set(tree)
    (tree / gate.PIN_SET).write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="declares no pin tests"):
        gate.pin_set(tree)


#: The shipped-tree scan lives with the snapshot builder, which
#: RELEASE-EXCLUDE keeps out of a published tree; there is nothing to
#: check against in one.
BUILDER = ROOT / "work" / "build_release_snapshot.py"
requires_builder = pytest.mark.skipif(
    not BUILDER.is_file(),
    reason="work/build_release_snapshot.py is not in this tree "
           "(published snapshot: the builder is publisher scaffolding)")


def _shapes():
    """A scratch root and a tree, in the two shapes that would ship.

    Assembled rather than written out, for the reason
    tests/test_release_snapshot_machine_paths.py gives about its own
    fixtures: the scan reads this file too and cannot tell a control's
    synthetic home from a real one, which is correct, since it is
    looking for the shape.
    """
    back = chr(92)
    return ("/ho" + "me/account/work/card-stage",
            "C:" + back + "Users" + back + "account" + back + "work" + back + "tree")


@requires_builder
def test_the_receipt_it_writes_carries_no_machine_path(tmp_path, monkeypatch):
    """THE BREAKAGE THIS PREVENTS: these receipts are committed beside the
    tests they back, and the release snapshot refuses to build over a
    developer-absolute path in a shipped file. The stage wrote the scratch
    root three times per receipt and the node's tracebacks under it, and
    that is what stopped the 2.7.6 snapshot.

    Driven end to end with a home-shaped scratch root, then the writer
    driven directly with a Windows-profile-shaped tree, and the verdict
    is the scan's own, not a second opinion about what a machine path is.
    """

    scan = _load("snapshot_scan", BUILDER)
    scratch, tree_shape = _shapes()

    rc, receipt, _ = drive(tmp_path, monkeypatch, remote=scratch)
    assert rc == 0 and receipt["status"] == "PASS"
    written = tmp_path / "evidence" / f"precut-gpu-gate-{HEAD[:9]}.json"
    assert scan.machine_path_violations(
        written.read_text(encoding="utf-8")) == []
    # What is left is the run's identity, which is what a reader wants:
    # the run directory's own name below the root the operator chose, and
    # the report's own name inside it.
    assert receipt["remote_directory"].startswith("run-")
    assert receipt["structured_report"].startswith("gpu-pins-")
    assert "/" not in receipt["remote_directory"]
    assert "/" not in receipt["structured_report"]
    assert scratch not in json.dumps(receipt)

    target = tmp_path / "evidence" / "written-directly.json"
    gate.base.save_evidence(target, {
        "pin_set_source": "gate tree " + tree_shape,
        "output_tail": "Traceback\n  File " + scratch + "/repo/tests/t.py\n",
        "card": {"other_compute_processes": [
            "4242, " + scratch + "/venv/bin/python, 390 MiB"]},
        "error": "held the card: " + scratch + "/venv/bin/python",
    })
    assert scan.machine_path_violations(
        target.read_text(encoding="utf-8")) == []
