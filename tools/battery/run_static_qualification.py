#!/usr/bin/env python3
"""Qualify native static arithmetic against independently run Python references."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "tools/battery/static_qualification.json"


def _verdict(passed, failed, skipped, operational_errors, **details):
    return {"status": ("operational_failure" if operational_errors else
                       "assertion_failure" if failed else "passed"),
            "passed": passed, "failed": failed, "skipped": skipped,
            "errors": 0, "operational_errors": operational_errors, **details}


def _comparison_lines(source: str, policy: dict) -> set[int]:
    allowed = set(policy["rust_qualification_tests"] + policy["rust_assertion_helpers"])
    result, function, body = set(), "", []
    lines = source.splitlines()
    for index, line in enumerate(lines, 1):
        definition = re.match(r"\s*fn\s+(\w+)\s*[<(]", line)
        if definition:
            function, body = definition.group(1), []
        body.append(line)
        if function in allowed and re.search(r"\bassert(?:_eq|_ne)?!\s*\(", line):
            result.add(index)
        literal = policy.get("rust_comparison_panics", {}).get(function)
        if (literal and "panic!(" in line and
                '"' + literal + '"' in "\n".join(lines[index:index + 2]) and
                "if observed != expected {" in "\n".join(body)):
            result.add(index)
    return result


def classify_native(output: str, returncode: int, policy: dict, source: str) -> dict:
    expected = set(policy["rust_functional_tests"] + policy["rust_qualification_tests"])
    cases = re.findall(r"^test (\S+) \.\.\. (ok|FAILED|ignored)\s*$", output, re.M)
    sums = re.findall(r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored", output, re.M)
    passed, failed, skipped = map(int, sums[0]) if len(sums) == 1 else (0, 0, 0)
    errors = []
    if len(sums) != 1 or len(cases) != len(expected) or {name for name, _ in cases} != expected:
        errors.append("missing, duplicate, or unexpected native test results")
    if (passed, failed, skipped) != tuple(sum(status == want for _, status in cases)
                                        for want in ("ok", "FAILED", "ignored")):
        errors.append("native summary disagrees with individual results")
    if skipped:
        errors.append("native qualification tests must actually execute")
    failed_names = {name for name, status in cases if status == "FAILED"}
    if failed_names & set(policy["rust_functional_tests"]):
        errors.append("a required functional native control failed")
    if returncode not in (0, 101) or (returncode == 0) != (failed == 0):
        errors.append(f"unexpected native process status {returncode}")
    panic_matches = list(re.finditer(r"thread '([^']+)'(?:\s+\(\d+\))? panicked at ([^\n]+?):(\d+):(\d+):", output))
    panics = [match.groups() for match in panic_matches]
    allowed_lines = _comparison_lines(source, policy)
    allowed_file = Path(policy["rust_source"]).name
    if failed and (len(panics) != failed or {name for name, *_ in panics} != failed_names or
                   any(Path(path.replace("\\", "/")).name != allowed_file or int(line) not in allowed_lines
                       for _name, path, line, _column in panics)):
        errors.append("a native failure was not an identified comparison assertion")
    source_lines = source.splitlines()
    for match in panic_matches:
        name, _path, line, _column = match.groups()
        index = int(line) - 1
        if 0 <= index < len(source_lines) and "panic!(" in source_lines[index]:
            message = next((line.strip() for line in output[match.end():].splitlines() if line.strip()), "")
            if (name not in policy.get("rust_comparison_panics", {}) or not re.fullmatch(
                    r"NPZ bytes differ first at offset \d+: [0-9a-fA-F]{2} != [0-9a-fA-F]{2}", message)):
                errors.append("native comparison panic did not report the declared byte mismatch")
    return _verdict(passed, failed, skipped, errors, failed_tests=sorted(failed_names),
                    observed_tests=[name for name, _ in cases], tests=len(cases))


def classify_python(xml: Path, returncode: int, policy: dict) -> dict:
    expected = {"::".join(name.split("::")[1:]): name for name in policy["python_tests"]}
    errors, seen, failed_names = [], [], []
    passed = failed = skipped = errored = 0
    try:
        cases = ET.parse(xml).getroot().findall(".//testcase")
    except (OSError, ET.ParseError) as error:
        return _verdict(0, 0, 0, [f"no usable Python result report: {error}"])
    for case in cases:
        key = case.get("classname", "").rsplit(".", 1)[-1] + "::" + case.get("name", "")
        seen.append(key)
        if case.find("error") is not None:
            errored += 1
            errors.append(f"Python setup/collection error: {key}")
        elif (failure := case.find("failure")) is not None:
            failed += 1
            failed_names.append(key)
            terminal = (failure.text or "").strip().splitlines()
            terminal = terminal[-1] if terminal else ""
            if (failure.get("type") not in (None, "AssertionError") or
                    not re.search(r":\d+: AssertionError(?:$|:)", terminal)):
                errors.append(f"Python failure is not an assertion: {key}")
        elif (skip := case.find("skipped")) is not None:
            skipped += 1
            reason = policy.get("python_allowed_skips", {}).get(expected.get(key, ""))
            if not reason or reason not in ((skip.get("message") or "") + (skip.text or "")):
                errors.append(f"undeclared Python skip: {key}")
        else:
            passed += 1
    if len(seen) != len(expected) or set(seen) != set(expected):
        errors.append("missing, duplicate, or unexpected Python test results")
    if returncode not in (0, 1) or (returncode == 0) != (failed == 0 and not errors):
        errors.append(f"unexpected Python process status {returncode}")
    return _verdict(passed, failed, skipped, errors, failed_tests=failed_names,
                    observed_tests=seen, tests=len(cases), errors=errored)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    target, output = args.target_dir.resolve(), args.output.resolve()
    if target == ROOT or ROOT in target.parents or output == ROOT or ROOT in output.parents:
        parser.error("target and evidence directories must be outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    report = {"schema": "arwen.static-qualification-result.v1", "system": platform.system(),
              "machine": platform.machine(), "python": platform.python_version(),
              "scope": "exact native-platform Python comparisons and pinned portable sampling bounds",
              "status": "operational_failure", "commands": []}
    environment = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_TERM_COLOR="never",
                       GPUWM_NO_LOCAL_GPU="1", GPUWM_NPMATH_SWEEP=str(output / "sweep"))
    environment["GPUWM_STATIC_LANE1_GOLDENS"] = str(output / "native-goldens")
    environment["WOOF_STATIC_BRIDGE"] = str(target / "debug" /
        ("static_fields.dll" if sys.platform == "win32" else "libstatic_fields.so"))

    def run(label, command, cwd=ROOT):
        result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                                encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        path = output / f"{label}.log"
        path.write_text(result.stdout, encoding="utf-8")
        report["commands"].append({"label": label, "command": command,
                                   "returncode": result.returncode, "log": path.name})
        print(f"{label}: exit {result.returncode}; full output {path}", flush=True)
        return result

    exitcode = 2
    try:
        import numpy
        report["numpy"] = numpy.__version__
        if numpy.__version__ != "2.2.6":
            raise RuntimeError("the declared comparison profile requires NumPy 2.2.6")
        workspace = ROOT / "tools/rustwx"
        prepare = [
            ("native-reference", [sys.executable, str(ROOT / "tools/static_rust_port/extract_lane1_goldens.py"),
                                  "--output", str(output / "native-goldens"), "--require-numpy", "2.2.6"], ROOT),
            ("build", ["cargo", "build", "--locked", "--offline", "-p", "static-fields", "--lib"], workspace),
            ("test-build", ["cargo", "test", "--locked", "--offline", "-p", "static-fields", "--test", "lane1_goldens", "--no-run"], workspace),
            ("bridge-load", [sys.executable, "-c", "from woof.static import rust_bridge; reason=rust_bridge.unavailable_reason(); print(reason or 'PASS: static bridge loads'); assert reason is None, reason"], ROOT),
            ("sweep", [sys.executable, str(ROOT / "tools/static_rust_port/gen_npmath_sweep.py"), str(output / "sweep")], ROOT),
        ]
        for label, command, cwd in prepare:
            if run(label, command, cwd).returncode:
                raise RuntimeError(f"required {label} prerequisite failed")
        # Use the declared qualification row, not a second list of Cargo args.
        sys.path.insert(0, str(ROOT))
        from tools.battery.run_cargo_gates import read_manifest
        entries = [entry for entry in read_manifest() if entry.shard == "qualification"]
        if len(entries) != 1 or entries[0].package != "static-fields":
            raise RuntimeError("expected exactly the declared static qualification target")
        native = run("rust-comparisons", entries[0].invocation(), workspace)
        source = (ROOT / policy["rust_source"]).read_text(encoding="utf-8")
        report["rust"] = classify_native(native.stdout, native.returncode, policy, source)
        junit = output / "python-comparisons.xml"
        python = run("python-comparisons", [sys.executable, "-m", "pytest", "-q", "-ra",
                     "-p", "no:cacheprovider", "--junitxml", str(junit), *policy["python_tests"]])
        report["python_comparisons"] = classify_python(junit, python.returncode, policy)
        results = [report["rust"], report["python_comparisons"]]
        if any(result["status"] == "operational_failure" for result in results):
            raise RuntimeError("a prerequisite, functional control, or test execution failed; see raw reports")
        comparisons_failed = any(result["failed"] for result in results)
        report["qualification_complete"] = not any(result["failed"] or result["skipped"] for result in results)
        report["status"] = ("COMPARISON_FAILURE" if comparisons_failed else
                            "INCOMPLETE" if not report["qualification_complete"] else "COMPARISONS_PASSED")
        # A skip reaching here is one the policy declares (python_allowed_skips: the full build without a
        # WPS_GEOG tree, which no hosted runner carries); any other skip is an operational failure above.
        # It leaves the record INCOMPLETE but is not a failed comparison. Breakage (2.9.12 R7a route Q):
        # exiting 1 on it turned both "Static qualification" jobs, which gate publication, red on every
        # hosted run.
        exitcode = 1 if comparisons_failed else 0
    except (OSError, ValueError, RuntimeError) as error:
        report["operational_error"] = str(error)
        report["status"] = "operational_failure"
    finally:
        report["evidence"] = [{"path": path.relative_to(output).as_posix(),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "bytes": path.stat().st_size}
                              for path in sorted(output.rglob("*")) if path.is_file()]
        (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [f"Static platform qualification: **{report['status']}** ({report['system']})"]
        for key in ("rust", "python_comparisons"):
            if key in report:
                result = report[key]
                lines.append(f"{key}: {result['passed']} passed, {result['failed']} failed, {result['skipped']} skipped, {result['errors']} errors; {result['status']}")
        lines.append("Full raw logs, JUnit and qualification.json are retained as the qualification artifact. Linux installed preparation, forecast and restart acceptance remain separate release requirements.")
        summary = "\n\n".join(lines) + "\n"
        print(summary, flush=True)
        (output / "summary.md").write_text(summary, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(summary)
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
