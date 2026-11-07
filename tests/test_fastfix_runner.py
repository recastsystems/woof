"""A focused run must retain failures, empty collection, and unrun broad gates."""
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "tools/battery/run_fastfix.py"
    spec = importlib.util.spec_from_file_location("fastfix_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_empty_junit_cannot_be_a_pass(tmp_path):
    path = tmp_path / "empty.xml"
    path.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    with pytest.raises(RuntimeError, match="no test cases"):
        runner().junit_summary(path)


def test_failed_and_skipped_tests_are_separate_from_passes(tmp_path):
    path = tmp_path / "counts.xml"
    path.write_text('<testsuite><testcase/><testcase><failure/></testcase>'
                    '<testcase><error/></testcase><testcase><skipped/></testcase></testsuite>')
    assert runner().junit_summary(path) == {"collected": 4, "passed": 1, "failed": 2, "skipped": 1}


@pytest.mark.parametrize("path, expected", [
    ("woof/core/moist.py", "numerical"),
    ("woof/core/kernels/a.cu", "native/kernel"),
    ("tools/arwen-tui/src/main.rs", "native/kernel"),
    ("pyproject.toml", "dependency"),
    ("configs/starter.toml", "numerical"),
    ("woof/cli.py", "indirect consumers"),
])
def test_import_selection_does_not_claim_broader_qualification(path, expected):
    assert any(expected in reason for reason in runner().broader_checks([path]))


def test_failed_process_retains_actual_exit_and_log(tmp_path):
    import os
    result = runner().run_command([sys.executable, "-c", "print('named failure'); raise SystemExit(7)"],
                                  tmp_path, dict(os.environ), tmp_path / "failure.log")
    assert result["returncode"] == 7
    assert "named failure" in (tmp_path / "failure.log").read_text()
    assert result["log_sha256"] == runner().digest(tmp_path / "failure.log")


@pytest.fixture
def source_checkout(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "tracked.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
    subprocess.run([
        "git", "-C", str(root), "-c", "user.name=Runner Test",
        "-c", "user.email=runner-test@example.invalid",
        "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={root / 'no-hooks'}",
        "commit", "-qm", "Seed the isolated runner fixture",
    ], check=True)
    module = runner()
    monkeypatch.setattr(module, "ROOT", root)
    return module, root


def test_untracked_source_addition_and_edit_change_the_recorded_identity(source_checkout):
    module, root = source_checkout
    before = module.source_state()
    (root / "woof").mkdir()
    added = root / "woof/new_source.py"
    added.write_text("VALUE = 2\n")
    first = module.source_state()
    added.write_text("VALUE = 3\n")
    second = module.source_state()

    assert first["untracked_inputs"] == second["untracked_inputs"] == ["woof/new_source.py"]
    assert len({state["tracked_content_sha256"] for state in [before, first, second]}) == 3
    assert before["revision"] == first["revision"] == second["revision"]
    assert not any(state["tracked_dirty"] for state in [before, first, second])


def test_untracked_suite_requires_development_flag_and_reaches_selection(
        source_checkout, monkeypatch, tmp_path):
    module, root = source_checkout
    (root / "tests").mkdir()
    (root / "tests/test_added.py").write_text("def test_added(): pass\n")
    output = tmp_path / "evidence"
    arguments = ["--source", str(root), "--base", "HEAD", "--output", str(output)]
    with pytest.raises(SystemExit) as refusal:
        module.main(arguments)
    assert refusal.value.code == 2
    assert not output.exists()

    selected_inputs = []
    def select(paths):
        selected_inputs.extend(paths)
        return ["tests/test_added.py"]
    monkeypatch.setattr(module, "load_selector", lambda: SimpleNamespace(
        changed_files=lambda *_: [], select=select, load_durations=lambda: {},
        _ordered=lambda selected, _: [(path, 0, "fixture") for path in selected]))
    assert module.main([*arguments, "--include-working-tree"]) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert selected_inputs == receipt["touched"] == ["tests/test_added.py"]
    assert receipt["status"] == "PLAN_ONLY"
    assert receipt["release_authorized"] is False


def test_entirely_skipped_and_uncollected_suites_are_named(tmp_path):
    report = tmp_path / "partial.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.test_passed.TestClass"/>'
        '<testcase classname="tests.test_skipped"><skipped/></testcase>'
        '<testcase classname="tests.test_missing_neighbor"/></testsuite>')
    module = runner()
    assert module.unexecuted_suites(report, [
        "tests/test_passed.py", "tests/test_skipped.py", "tests/test_missing.py",
    ]) == ["tests/test_skipped.py", "tests/test_missing.py"]
    assert module.junit_summary(report) == {
        "collected": 3, "passed": 2, "failed": 0, "skipped": 1}


def test_installed_checks_refuse_tracked_or_untracked_development_source(
        source_checkout, tmp_path, capsys):
    module, root = source_checkout
    for change in ("tracked", "untracked"):
        (root / "tracked.py").write_text("VALUE = 2\n" if change == "tracked" else "VALUE = 1\n")
        if change == "untracked":
            (root / "new_source.py").write_text("VALUE = 3\n")
        output = tmp_path / ("installed-" + change)
        with pytest.raises(SystemExit) as refusal:
            module.main([
                "--source", str(root), "--base", "HEAD", "--output", str(output),
                "--include-working-tree", "--installed-python", sys.executable,
                "--engine-wheel", str(tmp_path / "engine.whl"),
                "--companion-wheel", str(tmp_path / "companion.whl"),
                "--artifact-proof", str(tmp_path / "proof.json"),
            ])
        assert refusal.value.code == 2
        assert "installed checks require a clean committed source tree" in capsys.readouterr().err
        assert not output.exists()


def _source_probe_environment(tmp_path, monkeypatch):
    for name, directory in [("woof", "woof"), ("woof_data", "recast-woof-data/woof_data"),
                            ("gpuwm_kernels", "gpuwm-kernels/gpuwm_kernels"),
                            ("gpuwm_kernels_startup", "gpuwm-kernels-startup/gpuwm_kernels_startup")]:
        module = ModuleType(name)
        module.__file__ = str(tmp_path / directory / "__init__.py")
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["source-probe", str(tmp_path), "2.7.0"])
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "2.7.0")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [])


@pytest.mark.parametrize("name", ["woof", "woof_data", "gpuwm_kernels", "gpuwm_kernels_startup"])
def test_source_probe_refuses_either_package_from_another_checkout(
        tmp_path, monkeypatch, name):
    _source_probe_environment(tmp_path, monkeypatch)
    sys.modules[name].__file__ = str(tmp_path / "other" / name / "__init__.py")
    with pytest.raises(RuntimeError, match="imports another source tree"):
        exec(runner().SOURCE_PROBE, {})


@pytest.mark.parametrize("name", ["woof", "recast-woof-data", "gpuwm-kernels", "gpuwm-kernels-startup"])
def test_source_probe_refuses_either_packages_wrong_metadata(tmp_path, monkeypatch, name):
    _source_probe_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(importlib.metadata, "version", lambda key: "0.0.1" if key == name else "2.7.0")
    with pytest.raises(RuntimeError, match="mismatched metadata: " + name):
        exec(runner().SOURCE_PROBE, {})


def test_source_probe_accepts_matching_origins_and_metadata(tmp_path, monkeypatch, capsys):
    _source_probe_environment(tmp_path, monkeypatch)
    exec(runner().SOURCE_PROBE, {})
    result = json.loads(capsys.readouterr().out)
    assert result["version"] == "2.7.0"
    assert Path(result["engine"]) == tmp_path / "woof/__init__.py"
    assert Path(result["companion"]) == tmp_path / "recast-woof-data/woof_data/__init__.py"
    assert Path(result["kernels"]) == tmp_path / "gpuwm-kernels/gpuwm_kernels/__init__.py"
