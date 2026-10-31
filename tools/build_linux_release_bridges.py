"""Build and qualify the Linux release payload inside the pinned manylinux image.

This script also runs inside the same image under a daemonless container runner.
The source checkout, compiler copy, Cargo target and qualified output are explicit;
the script never obtains native artifacts from a checkout's target directories.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

IMAGE = ("quay.io/pypa/manylinux_2_28_x86_64@sha256:"
         "53390351aeb4688114b02c36a23b3e6ce1166ee9b7afc5df1a4f776354fc764c")
RUST_VERSION = "1.94.0"
# region_global_dealias builds before rustwx: both reach the dealias crate,
# and dealias's panic = "abort" profile left in a shared target broke a later
# rustwx build at a new revision into the same target (reproduced by 2.9.0
# lane A7: rustwx, then dealias, then rustwx again).  The cut's own sorted
# order already builds dealias first.
WORKSPACES = ("tools/grib1_bridge", "tools/region_global_dealias",
              "tools/rustwx", "tools/rw_icechunk", "tools/rw_wps",
              "tools/zarr_bridge", "tools/nextgen_hydrofabric",
              "tools/nextgen_forcing", "tools/nextgen_verify", "tools/nextgen_output",
              "tools/hydro_world", "tools/hydro_world_params",
              "tools/nextgen_reservoir_prepare", "tools/hydro_world_init",
              "tools/ocean_prepare", "tools/rw-bathy", "tools/wind-table")
CONTEXT_NAME = ".arwen-manylinux-build-context.json"
CONTEXT_SCHEMA = "arwen.manylinux.build-context.v1"


class QualificationError(RuntimeError):
    """The native bytes have not met the release compatibility contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Commands:
    def __init__(self, logs: Path, environment: dict[str, str]):
        logs.mkdir(parents=True, exist_ok=False)
        self.logs, self.environment, self.records = logs, environment, []

    def run(self, label: str, command: list[str], *, cwd: Path) -> str:
        print(f"{label}: starting", flush=True)
        started = now()
        result = subprocess.run(command, cwd=cwd, env=self.environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = self.logs / f"{len(self.records):03d}-{label}.stdout"
        stderr = stdout.with_suffix(".stderr")
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        self.records.append({"label": label, "command": command, "cwd": str(cwd),
                             "started_utc": started, "finished_utc": now(),
                             "returncode": result.returncode,
                             "stdout": file_identity(stdout), "stderr": file_identity(stderr)})
        require(result.returncode == 0,
                f"{label} exited {result.returncode}; see {stderr}")
        print(f"{label}: passed", flush=True)
        return result.stdout.decode("utf-8", errors="replace")


def source_identity(source: Path, revision: str, commands: Commands) -> dict:
    actual = commands.run("source-revision", ["git", "-C", str(source),
                          "rev-parse", "HEAD"], cwd=source).strip()
    status = commands.run("source-status", ["git", "-C", str(source), "status",
                          "--porcelain=v1", "--untracked-files=all"], cwd=source)
    require(actual == revision, f"source HEAD {actual} does not equal {revision}")
    require(not status.strip(), f"source checkout is not clean: {status}")
    return {"path": str(source), "revision": actual, "clean": True}


def toolchain_identity(root: Path, commands: Commands, source: Path) -> dict:
    rustc, cargo = root / "bin/rustc", root / "bin/cargo"
    rustc_vv = commands.run("rustc-version", [str(rustc), "-Vv"], cwd=source).strip()
    cargo_version = commands.run("cargo-version", [str(cargo), "-V"], cwd=source).strip()
    require(f"release: {RUST_VERSION}" in rustc_vv.splitlines(), rustc_vv)
    require(cargo_version.startswith(f"cargo {RUST_VERSION} "), cargo_version)
    require("host: x86_64-unknown-linux-gnu" in rustc_vv.splitlines(), rustc_vv)
    files = []
    for path in sorted({rustc, cargo, *[p for p in (root / "lib").rglob("*") if p.is_file()]}):
        require(path.resolve().is_relative_to(root.resolve()),
                f"toolchain file resolves outside its copied tree: {path}")
        identity = file_identity(path)
        identity["path"] = path.relative_to(root).as_posix()
        files.append(identity)
    require(any(p["path"].endswith(".rlib") for p in files), "copied Rust standard library is missing")
    return {"rustc_vv": rustc_vv, "cargo_version": cargo_version, "files": files}


def target_context(target: Path, *, image: str, toolchain: dict, revision: str,
                   reuse: bool) -> tuple[Path, dict]:
    target.mkdir(parents=True, exist_ok=True)
    marker = target / CONTEXT_NAME
    entries = list(target.iterdir())
    if marker.is_file():
        require(reuse or entries == [marker],
                "populated Cargo target requires explicit --reuse-target")
        context = json.loads(marker.read_text(encoding="utf-8"))
        for key, expected in (("schema", CONTEXT_SCHEMA), ("image", image),
                              ("glibc", "2.28"), ("toolchain", toolchain),
                              ("target_was_empty", True)):
            require(context.get(key) == expected, f"Cargo target provenance mismatch: {key}")
        require(re.fullmatch(r"[0-9a-f]{40}", context.get("initial_source_rev", "")) is not None,
                "Cargo target marker has no initial source revision")
    else:
        require(not entries, "refusing an unmarked, nonempty Cargo target")
        context = {"schema": CONTEXT_SCHEMA, "image": image, "glibc": "2.28",
                   "toolchain": toolchain, "target_was_empty": True,
                   "initial_source_rev": revision, "reuse_source_revisions": [],
                   "created_utc": now()}
        write_json(marker, context, exclusive=True)
    revisions = context.setdefault("reuse_source_revisions", [])
    require(isinstance(revisions, list), "invalid Cargo source revision history")
    if revision not in revisions:
        revisions.append(revision)
    context["current_source_rev"] = revision
    context.setdefault("build_history", []).append({"source_rev": revision,
        "started_utc": now(), "status": "BUILDING", "workspaces": list(WORKSPACES)})
    write_json(marker, context)
    return marker, context


def load_policy(commands: Commands, source: Path) -> tuple[dict, dict]:
    auditwheel = shutil.which("auditwheel", path=commands.environment.get("PATH"))
    require(auditwheel is not None, "auditwheel is required inside the pinned manylinux image")
    executable = Path(auditwheel).resolve()
    with executable.open(encoding="utf-8") as stream:
        first_line = stream.readline().strip()
    require(first_line.startswith("#!/"), "auditwheel must have an absolute Python shebang")
    launcher = shlex.split(first_line[2:])
    interpreter = Path(launcher[0])
    require(interpreter.is_file(), f"auditwheel Python is missing: {interpreter}")
    version = commands.run("auditwheel-version", [str(executable), "--version"], cwd=source).strip()
    query = ("import importlib.metadata as m,json; d=m.distribution('auditwheel'); "
             "p=[str(d.locate_file(f)) for f in d.files "
             "if str(f).endswith('auditwheel/policy/manylinux-policy.json')]; "
             "assert len(p)==1,p; print(json.dumps({'path':p[0],'version':d.version}))")
    located = json.loads(commands.run("auditwheel-policy", [*launcher, "-I", "-c", query], cwd=source))
    path = Path(located["path"])
    policies = json.loads(path.read_text(encoding="utf-8"))
    selected = [p for p in policies if p.get("name") == "manylinux_2_28"]
    require(len(selected) == 1, "auditwheel has no unique manylinux_2_28 policy")
    policy = selected[0]
    versions = policy.get("symbol_versions", {}).get("x86_64", {})
    require(all(versions.get(name) for name in ("GLIBC", "GLIBCXX", "GCC", "CXXABI")),
            "manylinux policy omits a required symbol-version family")
    require("2.28" in versions["GLIBC"] and "2.29" not in versions["GLIBC"],
            "manylinux policy does not establish the glibc 2.28 boundary")
    require(bool(policy.get("lib_whitelist")), "manylinux dependency policy is missing")
    require(shutil.which("readelf", path=commands.environment.get("PATH")) is not None,
            "readelf is required inside the pinned manylinux image")
    retained = commands.logs / "manylinux-policy.json"
    retained.write_bytes(path.read_bytes())
    return policy, {"file": file_identity(path), "auditwheel": file_identity(executable),
                    "version": version, "distribution_version": located["version"],
                    "retained_file": file_identity(retained), "policy": policy}


def inspect_elf(header: str, dynamic: str, versions: str, symbols: str,
                policy: dict) -> dict:
    require("ELF64" in header and "little endian" in header
            and "Advanced Micro Devices X86-64" in header, "artifact is not a little-endian x86-64 ELF")
    require(re.search(r"Type:\s+(?:DYN|EXEC)\b", header) is not None,
            "artifact is not an ELF executable or shared library")
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", dynamic)
    require(not re.search(r"\((?:RPATH|RUNPATH)\).*?\[[^\]]+\]", dynamic),
            "artifact retains RPATH/RUNPATH instead of baseline system dependencies")
    whitelist = set(policy["lib_whitelist"])
    # The ELF loader is an intrinsic part of this glibc target, not an
    # optional external library; auditwheel likewise treats it separately.
    whitelist.add("ld-linux-x86-64.so.2")
    require(set(needed) <= whitelist,
            f"DT_NEEDED exceeds manylinux policy: {sorted(set(needed) - whitelist)}")
    required: dict[str, list[str]] = {}
    active = False
    library = None
    for line in versions.splitlines():
        if line.startswith("Version ") and " section " in line:
            active = line.startswith("Version needs section ")
            library = None
        if not active:
            continue
        file_match = re.search(r"\bFile:\s+(\S+)", line)
        if file_match:
            library = file_match.group(1)
            required.setdefault(library, [])
        name_match = re.search(r"\bName:\s+(\S+)", line)
        if name_match:
            require(library is not None, "ELF version requirement has no library")
            required[library].append(name_match.group(1))
    require(set(required) <= set(needed), "ELF version requirements have no matching DT_NEEDED")
    allowed = policy["symbol_versions"]["x86_64"]
    for library, names in required.items():
        for name in names:
            family, separator, version = name.partition("_")
            require(bool(separator) and version in allowed.get(family, []),
                    f"{library} requires unsupported symbol version {name}")
    undefined = {line.split()[7].split("@", 1)[0] for line in symbols.splitlines()
                 if len(line.split()) >= 8 and line.split()[6] == "UND"}
    for library in needed:
        forbidden = undefined & set(policy.get("blacklist", {}).get(library, []))
        require(not forbidden, f"{library} uses policy-blacklisted symbols: {sorted(forbidden)}")
    return {"needed": sorted(needed), "required_symbol_versions": required}


def loader_closure(text: str, policy: dict) -> list[dict]:
    """Validate every resolved dependency, including transitive libraries."""
    whitelist = set(policy["lib_whitelist"]) | {"ld-linux-x86-64.so.2"}
    libraries = []
    for line in text.splitlines():
        if not line.strip() or re.fullmatch(r"\s*linux-vdso\.so\.1 \(0x[0-9a-f]+\)\s*", line):
            continue
        match = re.fullmatch(r"\s*(\S+) => (/\S+) \(0x[0-9a-f]+\)\s*", line)
        if match:
            name, path = match.groups()
        else:
            intrinsic = re.fullmatch(r"\s*(/\S*/ld-linux-x86-64\.so\.2) \(0x[0-9a-f]+\)\s*", line)
            require(intrinsic is not None, f"unresolved or unrecognized loader dependency: {line}")
            name, path = "ld-linux-x86-64.so.2", intrinsic.group(1)
        require(name in whitelist, f"transitive dependency exceeds manylinux policy: {name}")
        parts = PurePosixPath(path).parts
        require(".." not in parts and (path.startswith(("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/"))),
                f"dependency resolves outside baseline system libraries: {path}")
        libraries.append({"soname": name, "path": path})
    require(bool(libraries), "loader produced no resolved dependency evidence")
    return libraries


def qualify_artifacts(release: Path, source: Path, revision: str,
                      policy: dict, commands: Commands,
                      workspaces: tuple[str, ...] = WORKSPACES) -> list[dict]:
    sys.path.insert(0, str(source))
    bridge_assets = importlib.import_module("woof.bridge_assets")
    bridges = importlib.import_module("woof.bridges")
    doctor = importlib.import_module("woof.doctor")
    for module in (bridge_assets, bridges, doctor):
        require(Path(module.__file__).resolve().is_relative_to(source),
                f"native declaration imported outside the payload source: {module.__file__}")
    readelf = shutil.which("readelf", path=commands.environment.get("PATH"))
    require(readelf is not None, "readelf is required to verify native compatibility")
    declarations = bridge_assets.BUNDLED_ARTIFACTS
    require(bool(declarations), "release declaration contains no native artifacts")
    require(len({a.name for a in declarations}) == len(declarations),
            "release declaration repeats a native artifact name")
    results = []
    for artifact in declarations:
        if artifact.crate not in workspaces:
            continue
        path = release / bridge_assets.artifact_filename(artifact, "linux-x86_64")
        require(path.is_file(), f"missing declared native artifact: {artifact.name}")
        before = file_identity(path)
        payload = path.read_bytes()
        require(payload.startswith(b"\x7fELF"), f"{artifact.name} is not an ELF artifact")
        if artifact.vendored:
            require(bridges.BRIDGE_ABI_MARKERS[artifact.name] in payload,
                    f"vendored ABI marker missing: {artifact.name}")
        else:
            bridge_assets.verify_source_revision(payload, expected=revision, label=str(path))
        outputs = {label: commands.run(f"{artifact.name}-{label}",
                    [readelf, option, "--wide", str(path)], cwd=source)
                   for label, option in (("header", "--file-header"), ("dynamic", "--dynamic"),
                                         ("versions", "--version-info"), ("symbols", "--dyn-syms"))}
        compatibility = inspect_elf(**outputs, policy=policy)
        if compatibility["needed"]:
            dependencies = commands.run(f"{artifact.name}-dependencies",
                ["/lib64/ld-linux-x86-64.so.2", "--list", str(path.resolve())], cwd=source)
            compatibility["loader_resolution"] = dependencies
            closure = loader_closure(dependencies, policy)
            require(set(compatibility["needed"]) <= {entry["soname"] for entry in closure},
                    f"loader did not resolve every required dependency: {artifact.name}")
            for entry in closure:
                resolved = Path(entry["path"]).resolve()
                require(str(resolved).startswith(("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")),
                        f"dependency symlink leaves baseline system libraries: {resolved}")
                entry["resolved_file"] = file_identity(resolved)
            compatibility["resolved_dependencies"] = closure
        if artifact.kind == "executable":
            ok, evidence = doctor._exec_probe(path)
            require(ok, f"native executable probe failed: {artifact.name}: {evidence}")
            probe = {"kind": "executable", "passed": True, "evidence": evidence}
        else:
            # Execute each loader in a fresh process so missing symbols or a
            # bad library cannot turn the enclosing proof into a false pass.
            query = ("import ctypes,sys; p,s,v=sys.argv[1:]; l=ctypes.CDLL(p); "
                     "f=getattr(l,s); f.argtypes=[]; f.restype=ctypes.c_uint32; "
                     "a=int(f()); assert a==int(v),(p,s,a,v); print(a)")
            symbol, version = bridge_assets.library_abi_for(artifact.name)
            answer = commands.run(f"{artifact.name}-abi", [sys.executable, "-I", "-c", query,
                                  str(path.resolve()), symbol, str(version)], cwd=source).strip()
            probe = {"kind": "library", "passed": True, "symbol": symbol,
                     "expected": version, "actual": int(answer)}
        require(file_identity(path) == before, f"native artifact changed during probe: {path}")
        results.append({"artifact": artifact.name, "vendored": artifact.vendored,
                        "file": before, "compatibility": compatibility, "probe": probe})
    return results


def emit_artifacts(output: Path, artifacts: list[dict]) -> list[dict]:
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "qualified output already exists and is nonempty")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-qualified-", dir=output.parent))
    result = []
    for artifact in artifacts:
        origin = Path(artifact["file"]["path"])
        destination = staging / origin.name
        with origin.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        destination.chmod(0o755)
        identity = file_identity(destination)
        require(all(identity[key] == artifact["file"][key] for key in ("bytes", "sha256")),
                f"qualified output copy differs: {destination}")
        result.append({"artifact": artifact["artifact"], "file": identity})
    if output.exists():
        output.rmdir()  # Only the already checked empty directory is admissible.
    staging.rename(output)
    for artifact in result:
        artifact["file"]["path"] = str(output / Path(artifact["file"]["path"]).name)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--rust-toolchain", type=Path, required=True)
    parser.add_argument("--cargo-home", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--reuse-target", action="store_true")
    parser.add_argument("--workspace", action="append", choices=WORKSPACES,
                        help="build and qualify only this workspace's artifacts (repeatable); "
                             "the release cut reuses the others from a build whose inputs are unchanged")
    args = parser.parse_args(argv)
    selected = tuple(args.workspace) if args.workspace else WORKSPACES
    require(args.image == IMAGE, "image identity differs from the pinned compatibility baseline")
    require(re.fullmatch(r"[0-9a-f]{40}", args.revision) is not None, "revision must be a full Git SHA")
    source = args.source.resolve()
    for name in ("rust_toolchain", "cargo_home", "target_dir", "output", "report"):
        setattr(args, name, getattr(args, name).resolve())
        require(not getattr(args, name).is_relative_to(source), f"--{name} must be outside the source checkout")
    require(not args.report.exists(), "report is create-only")
    require(not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir())),
            "qualified output already exists and is nonempty")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    removed = {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_BUILD_TARGET",
               "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTC_BOOTSTRAP", "RUSTUP_TOOLCHAIN",
               "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LIBRARY_PATH", "COMPILER_PATH",
               "GCC_EXEC_PREFIX", "CC", "CXX", "AR", "LD", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"}
    for name in list(environment):
        if name in removed or name.startswith(("CARGO_TARGET_", "CC_", "CXX_", "AR_")):
            environment.pop(name)
    environment.update(PATH=str(args.rust_toolchain / "bin") + os.pathsep + environment.get("PATH", ""),
                       CARGO_HOME=str(args.cargo_home), CARGO_TARGET_DIR=str(args.target_dir),
                       RUSTC=str(args.rust_toolchain / "bin/rustc"),
                       GPUWM_BRIDGE_SOURCE_REV=args.revision,
                       PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", GPUWM_NO_LOCAL_GPU="1",
                       GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0=str(source),
                       GIT_OPTIONAL_LOCKS="0")
    os.environ.clear()
    os.environ.update(environment)
    sys.dont_write_bytecode = True
    commands = Commands(args.report.with_suffix(".logs"), environment)
    report = {"schema": "arwen.manylinux.release-build.v1", "status": "FAILED",
              "image": args.image, "helper": file_identity(Path(__file__).resolve()),
              "image_binding": "The invoking Docker digest or daemonless runtime receipt binds the image; this process verifies its glibc and toolchain.",
              "started_utc": now(), "source_revision": args.revision, "commands": commands.records}
    marker = context = None
    try:
        require(sys.platform == "linux", "this helper must run inside the Linux baseline image")
        glibc = commands.run("glibc-version", ["getconf", "GNU_LIBC_VERSION"], cwd=source).strip()
        require(glibc == "glibc 2.28", f"expected the glibc 2.28 image, found {glibc}")
        report["glibc"] = glibc
        report["source_before"] = source_identity(source, args.revision, commands)
        report["toolchain"] = toolchain_identity(args.rust_toolchain, commands, source)
        policy, report["policy"] = load_policy(commands, source)
        marker, context = target_context(args.target_dir, image=args.image, toolchain=report["toolchain"],
                                         revision=args.revision, reuse=args.reuse_target)
        report["workspaces"] = list(selected)
        for index, workspace in enumerate(selected):
            commands.run(f"build-{index + 1}", [str(args.rust_toolchain / "bin/cargo"),
                         "build", "--release", "--locked", "--offline"], cwd=source / workspace)
        report["artifacts"] = qualify_artifacts(args.target_dir / "release", source,
                                               args.revision, policy, commands, selected)
        report["source_after"] = source_identity(source, args.revision, commands)
        require(report["source_before"] == report["source_after"], "payload source identity changed")
        require(toolchain_identity(args.rust_toolchain, commands, source) == report["toolchain"],
                "copied Rust toolchain changed during build")
        report["qualified_output"] = emit_artifacts(args.output, report["artifacts"])
        # The count travels as its own field: a status literal that carried it
        # went stale the release the bundle grew, and the drivers that compared
        # the literal kept passing over a number that was no longer true.
        report["artifact_count"] = len(report["artifacts"])
        # A subset build says so in its status, so no reader of the report can
        # take it for the whole bundle.
        report["status"] = ("PASS_MANYLINUX_2_28_ALL_NATIVE_ARTIFACTS" if selected == WORKSPACES
                            else "PASS_MANYLINUX_2_28_SELECTED_NATIVE_ARTIFACTS")
        returncode = 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr, flush=True)
        returncode = 1
    finally:
        report["finished_utc"] = now()
        if marker is not None and context is not None:
            context["build_history"][-1].update(finished_utc=now(), status=report["status"])
            write_json(marker, context)
            report["target_context"] = file_identity(marker)
        write_json(args.report, report, exclusive=True)
    print(json.dumps({"status": report["status"], "report": file_identity(args.report)}), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
