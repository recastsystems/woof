"""Run a changed-area CPU batch and optional installed command sweep, with evidence.

This is a development/patch qualification leg, not permission to publish. A
release also needs the artifact verifier and the applicable numerical baseline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
MARKS = "not gpu and not slow and not network"


def load_selector():
    path = ROOT / "tools/battery/fastfix.py"
    spec = importlib.util.spec_from_file_location("arwen_patch_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_state():
    """Hash current tracked bytes, including deletions; never label a dirty tree HEAD."""
    rows = []
    untracked = [p.decode("utf-8") for p in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if p]
    # Private work/ is a scratch/evidence tree, not shipped runtime source. A
    # public cut has no such inputs. Relevant untracked product/test files must
    # never silently borrow HEAD's identity.
    untracked = sorted(p for p in untracked if not p.startswith("work/"))
    tracked = [p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p]
    for relative in sorted(set(tracked) | set(untracked)):
        path = ROOT / relative
        rows.append([relative, digest(path) if path.is_file() else None])
    return {
        "revision": git("rev-parse", "HEAD").decode().strip(),
        "tracked_content_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "tracked_dirty": bool(git("diff", "HEAD", "--name-only")),
        "untracked_inputs": untracked,
        "runner_sha256": digest(__file__),
    }


def broader_checks(touched):
    """Name the work an import selector cannot qualify; never imply it ran."""
    reasons = set()
    for path in touched:
        if path.startswith(("woof/core/", "tilestream/", "configs/", "woof/data/", "recast-woof-data/")):
            reasons.add("forecast/numerical inputs changed: run the affected numerical, GPU and forecast gates")
        if path.endswith((".rs", ".cu", "Cargo.toml", "Cargo.lock")) or "/vendor/" in path:
            reasons.add("native/kernel inputs changed: rebuild and run the affected native/ABI/numerical gates")
        if path in {"pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in"} or path.startswith((".github/", ".cargo/")):
            reasons.add("build/dependency inputs changed: repeat clean installs and the affected platform checks")
        if not (path.startswith(("docs/", "tests/")) or path in {"README.md", "CHANGELOG.md"}):
            reasons.add("runtime or tooling changed: review indirect consumers and run the touched user workflow")
    return sorted(reasons)


def junit_summary(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
    skipped = sum(c.find("skipped") is not None for c in cases)
    if not cases:
        raise RuntimeError("pytest reported no test cases; a green empty batch proves nothing")
    return {"collected": len(cases), "passed": len(cases) - failed - skipped,
            "failed": failed, "skipped": skipped}


def unexecuted_suites(report, selected):
    cases = list(ET.parse(report).getroot().iter("testcase"))
    missing = []
    for path in selected:
        module = path.removesuffix(".py").replace("/", ".")
        executed = [case for case in cases if
                    (case.get("classname", "") == module or case.get("classname", "").startswith(module + "."))
                    and case.find("skipped") is None]
        if not executed:
            missing.append(path)
    return missing


SOURCE_PROBE = r'''
import importlib.metadata as md, json, pathlib, sys
import woof, woof_data, gpuwm_kernels, gpuwm_kernels_startup
root = pathlib.Path(sys.argv[1]).resolve()
for module, directory in ((woof, root / "woof"), (woof_data, root / "recast-woof-data/woof_data"),
                          (gpuwm_kernels, root / "gpuwm-kernels/gpuwm_kernels"), (gpuwm_kernels_startup, root / "gpuwm-kernels-startup/gpuwm_kernels_startup")):
    if pathlib.Path(module.__file__).resolve().parent != directory:
        raise RuntimeError("test environment imports another source tree: " + str(module.__file__))
for name in ("woof", "recast-woof-data", "gpuwm-kernels", "gpuwm-kernels-startup"):
    if md.version(name) != sys.argv[2]:
        raise RuntimeError("test environment has mismatched metadata: " + name + " " + md.version(name))
print(json.dumps({"python": sys.executable, "engine": woof.__file__, "companion": woof_data.__file__, "kernels": gpuwm_kernels.__file__, "version": sys.argv[2], "packages": sorted((d.metadata.get("Name", ""), d.version) for d in md.distributions())}))
'''


# Runs with -I from an external directory, under the explicitly supplied wheel
# environment. Enumerate the installed parser so newly added commands cannot be
# missed by a stale handwritten list. Parsing --help never runs a handler.
INSTALLED_SWEEP = r'''
import argparse, contextlib, hashlib, importlib.metadata as md, io, json, pathlib, subprocess, sys, zipfile
import woof, woof_data, gpuwm_kernels, gpuwm_kernels_startup
from woof.cli import build_parser
from woof import bridge_assets, tui_cli
prefix = pathlib.Path(sys.prefix).resolve()
for module in (woof, woof_data, gpuwm_kernels, gpuwm_kernels_startup):
    if not pathlib.Path(module.__file__).resolve().is_relative_to(prefix):
        raise RuntimeError("expected a wheel installed in this environment: " + str(module.__file__))
for name in ("woof", "recast-woof-data", "gpuwm-kernels", "gpuwm-kernels-startup"):
    direct = md.distribution(name).read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise RuntimeError("editable installation is not an installed artifact check: " + name)
if any(md.version(name) != sys.argv[2] for name in ("woof", "recast-woof-data", "gpuwm-kernels", "gpuwm-kernels-startup")):
    raise RuntimeError("engine and companion versions disagree")
proof = json.loads(pathlib.Path(sys.argv[5]).read_text())
if proof.get("schema") != "gpuwm-release-artifact-proof-v2" or proof.get("status") != "PASS" or proof.get("mode") != "cut" or proof.get("source_rev") != sys.argv[1]:
    raise RuntimeError("artifact proof does not qualify this revision")
wheel_records = [proof["wheel"], *proof.get("distributions", [])]
engine_wheel = pathlib.Path(sys.argv[3])
kernels_wheel = engine_wheel.parent / ("gpuwm_kernels-" + sys.argv[2] + "-py3-none-any.whl")
startup_wheel = engine_wheel.parent / ("gpuwm_kernels_startup-" + sys.argv[2] + "-py3-none-any.whl")
for wheel in (engine_wheel, pathlib.Path(sys.argv[4]), kernels_wheel, startup_wheel):
    wheel_sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if not any(r.get("filename") == wheel.name and r.get("sha256") == wheel_sha for r in wheel_records):
        raise RuntimeError("wheel does not match artifact proof: " + str(wheel))
wheel_checks = []
for name, package, wheel in (("woof", "woof", engine_wheel), ("recast-woof-data", "woof_data", pathlib.Path(sys.argv[4])),
                             ("gpuwm-kernels", "gpuwm_kernels", kernels_wheel), ("gpuwm-kernels-startup", "gpuwm_kernels_startup", startup_wheel)):
    distribution = md.distribution(name)
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename.endswith("/RECORD"): continue
            parts = pathlib.PurePosixPath(member.filename).parts
            if not parts or ".." in parts or pathlib.PurePosixPath(member.filename).is_absolute(): raise RuntimeError("unsafe wheel member")
            if not (parts[0] == package or parts[0].endswith(".dist-info")):
                raise RuntimeError("wheel installation layout needs explicit verification: " + member.filename)
            installed = pathlib.Path(distribution.locate_file(member.filename)).resolve()
            if not installed.is_relative_to(prefix) or installed.read_bytes() != archive.read(member):
                raise RuntimeError("installed file differs from expected wheel: " + str(installed))
            checked += 1
    if not checked: raise RuntimeError("wheel has no verified members")
    wheel_checks.append({"path": str(wheel), "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "verified_members": checked})
binary = tui_cli.require_tui()
if not binary.resolve().is_relative_to(prefix):
    raise RuntimeError("installed TUI came from outside the wheel environment: " + str(binary))
bridge_assets.verify_source_revision(binary.read_bytes(), expected=sys.argv[1], label="installed TUI")
parser = build_parser()
commands = []
def walk(node, words):
    if words:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            try:
                parser.parse_args(words + ["--help"])
            except SystemExit as error:
                if error.code != 0: raise
            else:
                raise RuntimeError("help did not exit for " + repr(words))
        if not output.getvalue().strip(): raise RuntimeError("empty help: " + repr(words))
        commands.append({"argv": words + ["--help"], "help_sha256": hashlib.sha256(output.getvalue().encode()).hexdigest()})
    for action in node._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in sorted(action.choices.items()): walk(child, words + [name])
walk(parser, [])
if not commands: raise RuntimeError("installed parser has no commands")
for argv in (["--help"], ["version"], ["tui", "--snapshot", "terminal.html"]):
    subprocess.run([sys.executable, "-I", "-m", "woof.cli", *argv], check=True, stdout=sys.stderr)
for name in ("terminal.html", "terminal.cells.json"):
    path = pathlib.Path(name)
    if not path.is_file() or not path.stat().st_size: raise RuntimeError("missing native TUI output: " + name)
print(json.dumps({"python": sys.executable, "engine": md.version("woof"), "companion": md.version("recast-woof-data"), "wheels": wheel_checks, "tui_path": str(binary), "tui_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "commands": commands, "snapshot_sha256": hashlib.sha256(pathlib.Path("terminal.html").read_bytes()).hexdigest()}))
'''


def run_command(argv, cwd, env, log):
    started = time.monotonic()
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        result = subprocess.run([str(x) for x in argv], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    return {"argv": [str(x) for x in argv], "cwd": str(cwd), "returncode": result.returncode,
            "seconds": round(time.monotonic() - started, 3), "log": str(log), "log_sha256": digest(log)}


def main(argv=None):
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT, help="checkout to test; permits an external runner during qualification")
    parser.add_argument("--base", required=True, help="the exact accepted base revision or tag")
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory outside the checkout")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="provisioned CPU test interpreter")
    parser.add_argument("--installed-python", type=Path, help="optional isolated wheel environment; adds real command/TUI checks")
    parser.add_argument("--engine-wheel", type=Path, help="exact platform wheel installed in the checked environment")
    parser.add_argument("--companion-wheel", type=Path, help="exact matching companion wheel")
    parser.add_argument("--artifact-proof", type=Path, help="PASS from verify_release_artifacts.py on the same source revision")
    parser.add_argument("--execute", action="store_true", help="run the selected batch; default only writes the reviewable plan")
    parser.add_argument("--include-working-tree", action="store_true", help="include uncommitted tracked changes; receipt is development evidence")
    args = parser.parse_args(argv)
    ROOT = args.source.resolve(strict=True)
    installed_inputs = (args.installed_python, args.engine_wheel, args.companion_wheel, args.artifact_proof)
    if any(installed_inputs) and not all(installed_inputs):
        parser.error("installed checks require --installed-python, --engine-wheel, --companion-wheel and --artifact-proof together")
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(ROOT): parser.error("output must be outside the source checkout")
    if output.exists(): parser.error("output must be a new directory")
    before = source_state()
    if (before["tracked_dirty"] or before["untracked_inputs"]) and not args.include_working_tree:
        parser.error("source changes are present; commit first or explicitly request development evidence")
    if args.installed_python and (before["tracked_dirty"] or before["untracked_inputs"]):
        parser.error("installed checks require a clean committed source tree; commit source changes before checking installed wheels")
    selector = load_selector()
    base = git("rev-parse", "--verify", args.base + "^{commit}").decode().strip()
    touched = selector.changed_files(base, "HEAD")
    if args.include_working_tree:
        touched += [p for p in git("diff", "HEAD", "--name-only").decode().splitlines() if p]
        touched += before["untracked_inputs"]
    touched = sorted(set(touched))
    selected = selector.select(touched)
    rows = selector._ordered(selected, selector.load_durations())
    if not rows: parser.error("selection is empty")
    missing = [name for name in selected if not (ROOT / name).is_file()]
    if missing: parser.error("selected suites are missing: " + repr(missing))
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "arwen.fastfix-run.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_revision": base, "source": before, "touched": touched,
            "selected": selected, "markexpr": MARKS, "broader_checks_required": broader_checks(touched),
            "release_authorized": False, "reused_test_results": [], "steps": []}
    def save(status):
        plan["status"] = status
        (output / "receipt.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    save("PLAN_ONLY")
    print(f"{len(rows)} suites selected; plan: {output / 'receipt.json'}", flush=True)
    if not args.execute: return 0
    env = dict(os.environ)
    env.update(GPUWM_NO_LOCAL_GPU="1", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    for key in ("PYTHONSAFEPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    report = output / "pytest.xml"
    pytest = [args.python.resolve(strict=True), "-m", "pytest", "-q", "-p", "no:cacheprovider",
              "-m", MARKS, "--durations=0", "--junitxml=" + str(report), *[p for p, _, _ in rows]]
    started = time.monotonic()
    save("RUNNING")
    try:
        runtime = run_command([args.python.resolve(strict=True), "-c", SOURCE_PROBE, ROOT, version], ROOT, env, output / "source-environment.log")
        plan["steps"].append(runtime)
        if runtime["returncode"]:
            raise RuntimeError("source test environment is bound to the wrong tree or version; see source-environment.log")
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(run_command, pytest, ROOT, env, output / "pytest.log")]
            if args.installed_python:
                installed_env = {k: v for k, v in env.items() if not k.upper().startswith(("GPUWM_", "WOOF_", "CUPY_"))}
                installed_env["GPUWM_NO_LOCAL_GPU"] = "1"
                jobs.append(pool.submit(run_command, [args.installed_python.resolve(strict=True), "-I", "-c", INSTALLED_SWEEP,
                                        before["revision"], version, args.engine_wheel.resolve(strict=True),
                                        args.companion_wheel.resolve(strict=True), args.artifact_proof.resolve(strict=True)],
                                        output, installed_env, output / "installed-commands.log"))
            plan["steps"].extend(job.result() for job in jobs)
        plan["tests"] = junit_summary(report)
        plan["junit_sha256"] = digest(report)
        plan["unexecuted_suites"] = unexecuted_suites(report, selected)
        plan["durations"] = selector.durations_from_batched_report((output / "pytest.log").read_text(encoding="utf-8"))
        after = source_state()
        if after != before: raise RuntimeError("source changed during the run; results cannot be bound to one source state")
        passed = all(step["returncode"] == 0 for step in plan["steps"]) and plan["tests"]["failed"] == 0
        complete = plan["tests"]["skipped"] == 0 and not plan["unexecuted_suites"]
        plan["installed_commands"] = "FRESH" if args.installed_python else "NOT_RUN"
        plan["seconds"] = round(time.monotonic() - started, 3)
        save(("FOCUSED_CHECKS_PASS" if complete else "INCOMPLETE") if passed else "FAIL")
        print(json.dumps({"status": plan["status"], "seconds": plan["seconds"], "tests": plan["tests"],
                          "installed_commands": plan["installed_commands"], "receipt": str(output / "receipt.json")}))
        return 0 if passed and complete else 1
    except Exception as error:
        plan["error"] = f"{type(error).__name__}: {error}"
        plan["seconds"] = round(time.monotonic() - started, 3)
        save("FAIL")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
