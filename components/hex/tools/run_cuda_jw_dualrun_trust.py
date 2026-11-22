#!/usr/bin/env python3
"""Trust-promote one frozen 24-step v8.2.3 CUDA JW dual run.

This launcher intentionally imports no ``woof.hex`` or GPUWM module before
it has validated static authority pins and copied the exact admitted bytes to
a fresh capsule.  The measured child enters through an isolated bootstrap.
Only after the child exits does this launcher import the frozen MPAS validator
and the frozen GPUWM total comparator, validate the exact three-file output,
and publish a separate O_EXCL completion receipt, manifest, and final seal.

The two SHA constants below deliberately retain an unfrozen sentinel until
the shared CUDA source owner announces ``SRC FREEZE``.  Therefore this file is
safe to unit-test now but cannot accidentally promote a preliminary run.
"""

from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FROZEN_TOOL = ROOT / "tools" / "run_cuda_jw_dualrun.py"
ISOLATED_BOOTSTRAP = ROOT / "tools" / "cuda_jw_dualrun_isolated_bootstrap.py"
AUTHORITY_PINS = ROOT / "tools" / "cuda_jw_dualrun_authority_pins.json"
OUTER_LAUNCHER = ROOT / "tools" / "run_cuda_jw_dualrun_promotion.py"

# Set only by the separately audited, literal-pin outer launcher after it has
# validated this payload and every subordinate trust anchor. Direct execution
# of this payload is intentionally non-promotable.
OUTER_LAUNCHER_AUTHORITY: dict[str, Any] | None = None

UNFROZEN_SENTINEL = "SRC_FREEZE_REQUIRED"
FROZEN_TOOL_SHA256 = UNFROZEN_SENTINEL
AUTHORITY_PINS_SHA256 = UNFROZEN_SENTINEL
# Filled after the bootstrap's tests are final.  Unlike the two shared-source
# anchors above, this digest does not depend on the CUDA source freeze.
ISOLATED_BOOTSTRAP_SHA256 = UNFROZEN_SENTINEL

PIN_SCHEMA = "mpas-port.cuda-jw-dualrun-authority-pins/v1"
FREEZE_SCHEMA = "mpas-port.cuda-jw-dualrun-frozen-capsule/v1"
COMPLETION_RECEIPT_SCHEMA = "mpas-port.cuda-jw-dualrun-completion-receipt/v1"
COMPLETION_MANIFEST_SCHEMA = "mpas-port.cuda-jw-dualrun-completion-manifest/v1"
COMPLETION_SEAL_SCHEMA = "mpas-port.cuda-jw-dualrun-completion-seal/v1"
OUTER_AUTHORITY_SCHEMA = "mpas-port.cuda-jw-dualrun-outer-authority/v1"
CAPSULE_SCHEMA = "mpas-port.cuda-dual-run-capsule/v2"
REPORT_SCHEMA = "mpas-port.cuda-dual-run-report/v1"
COMPARISON_SCHEMA = "gpuwm.dual-run-comparison/v1"
COMPARISON_AUTHORITY_SCHEMA = "mpas-port.gpuwm-dualrun-authority/v1"
FTZ_SCHEMA = "mpas-port.cuda-ftz-binding/v1"

TARGET_STEPS = 24
TARGET_DT_SECONDS = 3600.0
TARGET_DURATION_SECONDS = 86_400.0
TARGET_PROFILE = "jw-x1.2562-native-dry-nomix"
TARGET_NAME = "JW x1.2562 native dry no-mix CUDA durability lane"
TARGET_METHOD = (
    "one native JW host preparation reused for two independent device uploads"
)
EXPECTED_OUTPUT_NAMES = (
    "JW-x1.2562-24h-dt3600-arm-a.json",
    "JW-x1.2562-24h-dt3600-arm-b.json",
    "JW-x1.2562-24h-dt3600-comparison.json",
)
EXPECTED_CHILD_PREFIX = (
    "validating FTZ/compiler authority\n"
    "preparing frozen JW inputs once\n"
    "compiling one executable for both CUDA arms\n"
    "running CUDA arm A (24 full steps)\n"
    "running CUDA arm B (24 full steps)\n"
).encode("utf-8")

EXPECTED_CONFIGURATION: dict[str, Any] = {
    "config_apply_lbcs": False,
    "config_apvm_upwinding": 0.5,
    "config_coef_3rd_order": 0.25,
    "config_curvature_terms": False,
    "config_del4u_div_factor": 10.0,
    "config_divergence_damping": False,
    "config_dt": 3600.0,
    "config_dynamics_split_steps": 1,
    "config_epssm": 0.1,
    "config_h_ScaleWithMesh": True,
    "config_h_mom_eddy_visc2": 0.0,
    "config_h_mom_eddy_visc4": 0.0,
    "config_h_theta_eddy_visc2": 0.0,
    "config_h_theta_eddy_visc4": 0.0,
    "config_horiz_mixing": "2d_fixed",
    "config_iau_option": "off",
    "config_len_disp": 0.0,
    "config_moist_physics": False,
    "config_monotonic": True,
    "config_mpas_cam_coef": 0.0,
    "config_number_of_sub_steps": 6,
    "config_physics_suite": "none",
    "config_positive_definite": False,
    "config_rayleigh_damp_u": False,
    "config_scalar_adv_order": 3,
    "config_scalar_advection": True,
    "config_scalar_vadv_order": 3,
    "config_smagorinsky_coef": 0.0,
    "config_smdiv": 0.0,
    "config_split_dynamics_transport": True,
    "config_terrain_following": True,
    "config_time_integration_order": 3,
    "config_v_mom_eddy_visc2": 0.0,
    "config_v_theta_eddy_visc2": 0.0,
    "config_vertical_mixing": False,
    "config_visc4_2dsmag": 0.0,
    "config_xnutr": 0.0,
    "config_zd": 22000.0,
}
EXPECTED_LAYOUT = {
    "logical_order": "[level,entity]",
    "storage": "C-contiguous; entity fastest",
    "launch": "one thread per horizontal owner; ascending level loop",
}
EXPECTED_STEP_CONTRACT_KEYS = {
    "authority_ruler",
    "authority_ruler_sha256",
    "compile_manifest_sha256",
    "configuration",
    "configuration_sha256",
    "d2h_bytes_inside_step",
    "evidence",
    "frozen_source",
    "layout_contract_sha256",
    "stage_acoustic_steps",
    "t0_diagnostics_source",
}
EXPECTED_TOP_KEYS = {
    "configuration",
    "contracts",
    "device",
    "evidence",
    "execution",
    "input_bytes",
    "preparation",
    "profile",
    "schema",
    "trajectory",
}
EXPECTED_SOURCE_PATHS = {
    "cuda_dualrun": "cuda_dualrun.py",
    "cuda_acoustic": "cuda_acoustic.py",
    "cuda_driver": "cuda_driver.py",
    "cuda_fp32": "cuda_fp32.py",
    "cuda_ftz": "cuda_ftz.py",
    "cuda_horizontal": "cuda_horizontal.py",
    "cuda_transport": "cuda_transport.py",
    "cuda_backend_compile_contract": "cuda_backend/compile_contract.py",
    "cuda_backend_containers": "cuda_backend/containers.py",
    "cuda_backend_recovery": "cuda_backend/recovery.py",
    "cuda_backend_runtime": "cuda_backend/runtime.py",
    "host_driver": "driver.py",
    "host_integration": "integration.py",
    "host_mixing": "mixing.py",
    "host_transport": "transport.py",
}
GPUWM_SOURCE_PATHS = (
    "tools/ftz_receipt/probe.py",
    "tools/ftz_receipt/route_inventory.py",
    "woof/core/kernels/ftz_probe.cu",
    "woof/certify/compile_platform.py",
    "woof/certify/dualrun.py",
    "woof/certify/pins.py",
    "woof/gpu_stack_identity.py",
)
RUNTIME_RECEIPT_SCHEMA = "mpas-port.cuda-jw-dualrun-runtime-closure/v1"
_IGNORED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_NONDETERMINISTIC_KEYS = frozenset(
    {
        "cache_directory",
        "compile_seconds",
        "elapsed_seconds",
        "finished_utc",
        "first_launch_ms",
        "first_wall_ms",
        "generated_at_utc",
        "max_launch_ms",
        "mean_launch_ms",
        "min_launch_ms",
        "started_utc",
        "timestamp",
        "wall_seconds",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")


class TrustError(RuntimeError):
    """The dual-run evidence cannot be promoted."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrustError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise TrustError(f"JSON contains non-finite token {value!r}")


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrustError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise TrustError(f"{label} JSON root is not an object")
    return value


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise TrustError(f"cannot read {label} {path}: {error}") from error
    return _strict_json_bytes(payload, label=label), payload


def _file_record(path: Path) -> dict[str, Any]:
    selected = path.expanduser()
    if selected.is_symlink() or not selected.is_file():
        raise TrustError(f"authority input is not a regular file: {selected}")
    before = selected.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = selected.stat(follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or size != after.st_size:
        raise TrustError(f"authority input changed while hashing: {selected}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _tree_inventory(
    root: Path,
    *,
    ignored_parts: frozenset[str] = frozenset(),
    excluded_files: frozenset[str] = frozenset(),
    ignore_git: bool = False,
) -> dict[str, dict[str, Any]]:
    raw = root.expanduser()
    if raw.is_symlink():
        raise TrustError(f"inventory root must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_dir():
        raise TrustError(f"inventory root is not a directory: {selected}")
    result: dict[str, dict[str, Any]] = {}
    for current, directory_names, file_names in os.walk(selected, topdown=True):
        current_path = Path(current)
        retained: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(selected)
            if (ignore_git and name == ".git") or any(
                part in ignored_parts for part in relative.parts
            ):
                continue
            if candidate.is_symlink():
                raise TrustError(f"inventory contains a directory symlink: {candidate}")
            retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(selected)
            relative_text = relative.as_posix()
            if relative_text in excluded_files or any(
                part in ignored_parts for part in relative.parts
            ):
                continue
            result[relative_text] = _file_record(candidate)
    return dict(sorted(result.items()))


def _inventory_summary(files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in sorted(files.items())}
    return {
        "file_count": len(normalized),
        "files_sha256": _canonical_sha256(normalized),
    }


def _runtime_dependency_inventory(raw_entries: object) -> dict[str, Any]:
    """Hash the exact distribution closure named by the static discovery pins.

    ``runtime_capsule_entries`` is generated from module and native-image
    origins observed in an exact isolated discovery run.  The promotion path
    deliberately performs no package discovery of its own: a new or missing
    transitive distribution changes the signed entry map and must be reviewed
    and re-pinned before it can execute.
    """

    if not isinstance(raw_entries, Mapping) or not raw_entries:
        raise TrustError("static runtime capsule entry map is absent or empty")
    entries: dict[str, dict[str, str]] = {}
    casefolded_targets: set[str] = set()
    files: dict[str, dict[str, Any]] = {}
    for target_name, raw_row in sorted(raw_entries.items()):
        if not isinstance(target_name, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", target_name
        ) is None:
            raise TrustError(f"runtime capsule entry target is unsafe: {target_name!r}")
        folded = target_name.casefold()
        if folded in casefolded_targets:
            raise TrustError(f"runtime capsule entry target collides: {target_name!r}")
        casefolded_targets.add(folded)
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "source_path",
            "kind",
        }:
            raise TrustError(f"runtime capsule entry {target_name!r} is malformed")
        source_text = raw_row.get("source_path")
        kind = raw_row.get("kind")
        if not isinstance(source_text, str) or kind not in {"file", "directory"}:
            raise TrustError(f"runtime capsule entry {target_name!r} is malformed")
        raw_source = Path(source_text).expanduser()
        if raw_source.is_symlink():
            raise TrustError(
                f"runtime capsule entry source must not be a symlink: {raw_source}"
            )
        source = raw_source.resolve()
        if (kind == "file" and not source.is_file()) or (
            kind == "directory" and not source.is_dir()
        ):
            raise TrustError(
                f"runtime capsule entry source kind changed: {target_name!r} -> {source}"
            )
        entries[target_name] = {"source_path": str(source), "kind": str(kind)}
        if kind == "file":
            files[target_name] = _file_record(source)
            continue
        inventory = _tree_inventory(source, ignored_parts=_IGNORED_PARTS)
        if not inventory:
            raise TrustError(f"runtime capsule directory is empty: {source}")
        for relative, record in inventory.items():
            files[f"{target_name}/{relative}"] = record
    return {
        "entries": entries,
        "files": dict(sorted(files.items())),
        **_inventory_summary(files),
    }


def _python_runtime_inventory() -> dict[str, Any]:
    """Hash CPython's executable, stdlib, extension modules, and runtime DLLs."""

    roots: list[Path] = []
    for key in ("stdlib", "platstdlib"):
        value = sysconfig.get_path(key)
        if value:
            selected = Path(value).resolve()
            if selected.is_dir() and selected not in roots:
                roots.append(selected)
    dll_root = Path(sys.base_prefix).resolve() / "DLLs"
    if dll_root.is_dir() and dll_root not in roots:
        roots.append(dll_root)
    files: dict[str, dict[str, Any]] = {}
    for root in roots:
        for relative, record in _tree_inventory(
            root,
            ignored_parts=_IGNORED_PARTS | frozenset({"site-packages"}),
        ).items():
            files[str((root / relative).resolve())] = record
    executable = Path(sys.executable).resolve()
    files[str(executable)] = _file_record(executable)
    prefix = Path(sys.base_prefix).resolve()
    for pattern in ("python*.dll", "vcruntime*.dll"):
        for candidate in prefix.glob(pattern):
            if candidate.is_file() and not candidate.is_symlink():
                files[str(candidate.resolve())] = _file_record(candidate.resolve())
    if not files:
        raise TrustError("Python runtime inventory is empty")
    return {"files": dict(sorted(files.items())), **_inventory_summary(files)}


def _native_module_paths() -> tuple[Path, ...]:
    """Return every native image loaded in this process."""

    paths: set[Path] = set()
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        process = ctypes.windll.kernel32.GetCurrentProcess()
        capacity = 4096
        handles = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not ctypes.windll.psapi.EnumProcessModules(
            process,
            handles,
            ctypes.sizeof(handles),
            ctypes.byref(needed),
        ):
            raise TrustError("EnumProcessModules failed for native runtime inventory")
        count = min(capacity, int(needed.value // ctypes.sizeof(wintypes.HMODULE)))
        for handle in handles[:count]:
            buffer = ctypes.create_unicode_buffer(32768)
            if ctypes.windll.psapi.GetModuleFileNameExW(
                process, handle, buffer, len(buffer)
            ):
                candidate = Path(buffer.value).resolve()
                if candidate.is_file():
                    paths.add(candidate)
    elif Path("/proc/self/maps").is_file():
        for line in Path("/proc/self/maps").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            value = line.rsplit(" ", 1)[-1].strip()
            candidate = Path(value)
            if value.startswith("/") and candidate.is_file():
                paths.add(candidate.resolve())
    else:
        raise TrustError("native module inventory is unsupported on this platform")
    return tuple(sorted(paths, key=lambda path: str(path).lower()))


def _native_module_inventory() -> dict[str, Any]:
    files = {str(path): _file_record(path) for path in _native_module_paths()}
    return {"files": files, **_inventory_summary(files)}


def _system_cuda_library_inventory() -> dict[str, Any]:
    """Pin external CUDA/driver images that cannot be capsule-owned."""

    directories: set[Path] = set()
    for value in os.environ.get("PATH", "").split(os.pathsep):
        if value:
            candidate = Path(value).expanduser().resolve()
            if candidate.is_dir():
                directories.add(candidate)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate = (Path(cuda_path).expanduser().resolve() / "bin")
        if candidate.is_dir():
            directories.add(candidate)
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidate = Path(system_root).resolve() / "System32"
        if candidate.is_dir():
            directories.add(candidate)
    pattern = re.compile(
        r"^(?:nvcuda|nvrtc|nvrtc-builtins|cudart|nvjitlink)[^\\/]*\.dll$",
        re.IGNORECASE,
    )
    files: dict[str, dict[str, Any]] = {}
    for directory in sorted(directories, key=lambda path: str(path).lower()):
        try:
            candidates = tuple(directory.iterdir())
        except OSError:
            continue
        for candidate in candidates:
            if pattern.fullmatch(candidate.name) and candidate.is_file() and not candidate.is_symlink():
                selected = candidate.resolve()
                files[str(selected)] = _file_record(selected)
    if not any(Path(path).name.lower() == "nvcuda.dll" for path in files):
        raise TrustError("external NVIDIA driver library nvcuda.dll was not resolved")
    return {"files": dict(sorted(files.items())), **_inventory_summary(files)}


def _git_head(root: Path) -> str:
    selected = root.expanduser().resolve()
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={selected.as_posix()}",
                "-C",
                str(selected),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TrustError(f"cannot resolve GPUWM HEAD at {selected}: {error}") from error
    head = completed.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise TrustError(f"GPUWM HEAD is invalid: {head!r}")
    return head


def _gpuwm_inventory(root: Path) -> dict[str, Any]:
    selected = root.expanduser().resolve()
    if root.expanduser().is_symlink() or not selected.is_dir():
        raise TrustError(f"GPUWM root is not a real directory: {selected}")
    files = {relative: _file_record(selected / relative) for relative in GPUWM_SOURCE_PATHS}
    return {
        "root": str(selected),
        "git_head": _git_head(selected),
        "files": files,
        **_inventory_summary(files),
    }


def _authority_snapshot(
    args: argparse.Namespace, *, pins: Mapping[str, Any]
) -> dict[str, Any]:
    source_files = _tree_inventory(ROOT / "src" / "hexcore", ignored_parts=_IGNORED_PARTS)
    receipt_root = args.gpuwm_probe.expanduser().resolve()
    receipt_files = _tree_inventory(receipt_root)
    expected = pins.get("expected")
    runtime_entries = (
        expected.get("runtime_capsule_entries")
        if isinstance(expected, Mapping)
        else None
    )
    runtime_dependencies = _runtime_dependency_inventory(runtime_entries)
    python_runtime = _python_runtime_inventory()
    system_cuda_libraries = _system_cuda_library_inventory()
    host_native_baseline = _native_module_inventory()
    return {
        "outer_launcher_authority": deepcopy_json(OUTER_LAUNCHER_AUTHORITY),
        "mpas_source": {
            "root": str((ROOT / "src" / "hexcore").resolve()),
            "files": source_files,
            **_inventory_summary(source_files),
        },
        "frozen_tool": {
            "path": str(FROZEN_TOOL.resolve()),
            **_file_record(FROZEN_TOOL),
        },
        "isolated_bootstrap": {
            "path": str(ISOLATED_BOOTSTRAP.resolve()),
            **_file_record(ISOLATED_BOOTSTRAP),
        },
        "authority_pins": {
            "path": str(AUTHORITY_PINS.resolve()),
            **_file_record(AUTHORITY_PINS),
        },
        "inputs": {
            "authority_initial_condition": {
                "path": str(args.initial.expanduser().resolve(strict=True)),
                **_file_record(args.initial.expanduser().resolve(strict=True)),
            },
            "native_internal_t0": {
                "path": str(args.native_t0.expanduser().resolve(strict=True)),
                **_file_record(args.native_t0.expanduser().resolve(strict=True)),
            },
        },
        "ftz_binding": {
            "path": str(args.ftz_binding.expanduser().resolve(strict=True)),
            **_file_record(args.ftz_binding.expanduser().resolve(strict=True)),
        },
        "gpuwm_receipt": {
            "root": str(receipt_root),
            "files": receipt_files,
            **_inventory_summary(receipt_files),
        },
        "gpuwm_sources": _gpuwm_inventory(args.gpuwm_root),
        "runtime_dependencies": runtime_dependencies,
        "python_runtime": python_runtime,
        "system_cuda_libraries": system_cuda_libraries,
        "host_native_baseline": host_native_baseline,
    }


def _pin_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    inputs = snapshot["inputs"]
    return {
        "mpas_source": {
            "file_count": snapshot["mpas_source"]["file_count"],
            "files_sha256": snapshot["mpas_source"]["files_sha256"],
        },
        "frozen_tool": {
            "bytes": snapshot["frozen_tool"]["bytes"],
            "sha256": snapshot["frozen_tool"]["sha256"],
        },
        "inputs": {
            label: {"bytes": row["bytes"], "sha256": row["sha256"]}
            for label, row in inputs.items()
        },
        "ftz_binding": {
            "bytes": snapshot["ftz_binding"]["bytes"],
            "sha256": snapshot["ftz_binding"]["sha256"],
        },
        "gpuwm_receipt": {
            "file_count": snapshot["gpuwm_receipt"]["file_count"],
            "files_sha256": snapshot["gpuwm_receipt"]["files_sha256"],
        },
        "gpuwm_sources": {
            "git_head": snapshot["gpuwm_sources"]["git_head"],
            "file_count": snapshot["gpuwm_sources"]["file_count"],
            "files_sha256": snapshot["gpuwm_sources"]["files_sha256"],
        },
        "runtime_dependencies": {
            "entries": snapshot["runtime_dependencies"]["entries"],
            "file_count": snapshot["runtime_dependencies"]["file_count"],
            "files_sha256": snapshot["runtime_dependencies"]["files_sha256"],
        },
        "python_runtime": {
            "file_count": snapshot["python_runtime"]["file_count"],
            "files_sha256": snapshot["python_runtime"]["files_sha256"],
        },
        "system_cuda_libraries": {
            "file_count": snapshot["system_cuda_libraries"]["file_count"],
            "files_sha256": snapshot["system_cuda_libraries"]["files_sha256"],
        },
        "host_native_baseline": {
            "file_count": snapshot["host_native_baseline"]["file_count"],
            "files_sha256": snapshot["host_native_baseline"]["files_sha256"],
        },
    }


def deepcopy_json(value: object) -> Any:
    """Copy a JSON-shaped value without importing target code."""

    return json.loads(json.dumps(value, sort_keys=True))


def _validate_outer_authority() -> dict[str, Any]:
    authority = OUTER_LAUNCHER_AUTHORITY
    if not isinstance(authority, Mapping) or set(authority) != {
        "schema",
        "outer_launcher",
        "payload",
        "frozen_tool",
        "isolated_bootstrap",
        "authority_pins",
    }:
        raise TrustError("promotion requires the literal-pin outer launcher")
    if authority.get("schema") != OUTER_AUTHORITY_SCHEMA:
        raise TrustError("outer launcher authority schema changed")
    expected_paths = {
        "payload": Path(__file__).resolve(),
        "frozen_tool": FROZEN_TOOL.resolve(),
        "isolated_bootstrap": ISOLATED_BOOTSTRAP.resolve(),
        "authority_pins": AUTHORITY_PINS.resolve(),
    }
    for label, path in expected_paths.items():
        row = authority.get(label)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256"}
            or row.get("path") != str(path)
        ):
            raise TrustError(f"outer launcher {label} path binding changed")
        if {"bytes": row.get("bytes"), "sha256": row.get("sha256")} != _file_record(path):
            raise TrustError(f"outer launcher {label} byte binding changed")
    outer = authority.get("outer_launcher")
    if (
        not isinstance(outer, Mapping)
        or set(outer) != {"path", "bytes", "sha256"}
        or outer.get("path") != str(OUTER_LAUNCHER.resolve())
    ):
        raise TrustError("outer launcher self-record is incomplete")
    if {"bytes": outer.get("bytes"), "sha256": outer.get("sha256")} != _file_record(
        OUTER_LAUNCHER
    ):
        raise TrustError("outer launcher changed after literal-pin admission")
    return deepcopy_json(authority)


def _load_authority_pins() -> dict[str, Any]:
    if FROZEN_TOOL_SHA256 == UNFROZEN_SENTINEL or AUTHORITY_PINS_SHA256 == UNFROZEN_SENTINEL:
        raise TrustError("SRC FREEZE has not been installed; promotion is disabled")
    if _SHA256_RE.fullmatch(FROZEN_TOOL_SHA256) is None:
        raise TrustError("frozen-tool static SHA-256 is invalid")
    if _file_record(FROZEN_TOOL)["sha256"] != FROZEN_TOOL_SHA256:
        raise TrustError("frozen dual-run tool differs from its static SHA-256")
    pin_record = _file_record(AUTHORITY_PINS)
    if _SHA256_RE.fullmatch(AUTHORITY_PINS_SHA256) is None or pin_record["sha256"] != AUTHORITY_PINS_SHA256:
        raise TrustError("authority pin document differs from its static SHA-256")
    pins, _ = _load_json(AUTHORITY_PINS, label="authority pin document")
    if pins.get("schema") != PIN_SCHEMA:
        raise TrustError("authority pin document schema is invalid")
    return pins


def _validate_bootstrap_pin() -> dict[str, Any]:
    if ISOLATED_BOOTSTRAP_SHA256 == UNFROZEN_SENTINEL:
        raise TrustError("isolated bootstrap has not been byte-frozen")
    record = _file_record(ISOLATED_BOOTSTRAP)
    if _SHA256_RE.fullmatch(ISOLATED_BOOTSTRAP_SHA256) is None or record["sha256"] != ISOLATED_BOOTSTRAP_SHA256:
        raise TrustError("isolated bootstrap differs from its static SHA-256")
    return {"path": str(ISOLATED_BOOTSTRAP.resolve()), **record}


def _validate_pre_pins(pins: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if set(pins) != {"schema", "source_release", "authority", "expected"}:
        raise TrustError("authority pin document field set changed")
    if pins.get("schema") != PIN_SCHEMA or pins.get("source_release") != "v8.2.3":
        raise TrustError("authority pins do not identify v8.2.3")
    if pins.get("authority") != _pin_view(snapshot):
        raise TrustError("live source/tool/input/FTZ/GPUWM bytes differ from static pins")
    expected = pins.get("expected")
    if not isinstance(expected, Mapping) or set(expected) != {
        "configuration_sha256",
        "layout_sha256",
        "ftz_binding_canonical_sha256",
        "compile_manifest_canonical_sha256",
        "compile_platform",
        "runtime_capsule_entries",
        "runtime_post_module_count",
        "runtime_post_module_origins_normalized_sha256",
        "runtime_post_native_count",
        "runtime_post_native_modules_normalized_sha256",
    }:
        raise TrustError("static expected contract pins are incomplete")
    if expected["runtime_capsule_entries"] != snapshot["runtime_dependencies"]["entries"]:
        raise TrustError("static runtime capsule entry paths changed")
    binding, _ = _load_json(
        Path(snapshot["ftz_binding"]["path"]), label="pinned FTZ binding"
    )
    if binding.get("schema") != FTZ_SCHEMA:
        raise TrustError("pinned FTZ binding schema changed")
    if expected["configuration_sha256"] != _canonical_sha256(EXPECTED_CONFIGURATION):
        raise TrustError("static configuration pin differs from the fixed 24-step lane")
    if expected["layout_sha256"] != _canonical_sha256(EXPECTED_LAYOUT):
        raise TrustError("static layout pin differs from the fixed CUDA layout")
    if expected["ftz_binding_canonical_sha256"] != _canonical_sha256(binding):
        raise TrustError("static FTZ canonical pin differs from its pinned bytes")
    manifest = binding.get("compile_manifest")
    if not isinstance(manifest, Mapping):
        raise TrustError("pinned FTZ binding has no compile manifest")
    if expected["compile_manifest_canonical_sha256"] != _canonical_sha256(manifest):
        raise TrustError("static compile-manifest pin differs from the FTZ binding")
    platform = manifest.get("compile_platform")
    if expected["compile_platform"] != platform:
        raise TrustError("static compile-platform pin differs from the FTZ binding")
    for label in ("runtime_post_module_count", "runtime_post_native_count"):
        if type(expected[label]) is not int or expected[label] <= 0:
            raise TrustError(f"static {label} pin is invalid")
    for label in (
        "runtime_post_module_origins_normalized_sha256",
        "runtime_post_native_modules_normalized_sha256",
    ):
        if _SHA256_RE.fullmatch(str(expected[label])) is None:
            raise TrustError(f"static {label} pin is invalid")
    return binding


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _resolve_absent_target(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise TrustError(f"{label} must not be a symlink: {raw}")
    selected = raw.resolve()
    if selected.exists():
        raise TrustError(f"{label} must be absent before launch: {selected}")
    parent = selected.parent
    if parent.is_symlink() or not parent.is_dir():
        raise TrustError(f"{label} parent must be a real existing directory: {parent}")
    return selected


def _validate_targets(
    *,
    output_root: Path,
    cache_root: Path,
    completion_root: Path,
    capsule_root: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    targets = {
        "measured output root": _resolve_absent_target(output_root, label="measured output root"),
        "measured cache root": _resolve_absent_target(cache_root, label="measured cache root"),
        "completion root": _resolve_absent_target(completion_root, label="completion root"),
        "frozen capsule root": _resolve_absent_target(capsule_root, label="frozen capsule root"),
    }
    items = list(targets.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if _paths_overlap(left, right):
                raise TrustError(f"{left_name} overlaps {right_name}")
    protected = {
        ROOT.resolve(),
        args.gpuwm_root.expanduser().resolve(),
        args.gpuwm_probe.expanduser().resolve(),
        args.ftz_binding.expanduser().resolve(strict=True),
        args.initial.expanduser().resolve(strict=True),
        args.native_t0.expanduser().resolve(strict=True),
    }
    for label, target in targets.items():
        for authority in protected:
            if _paths_overlap(target, authority):
                raise TrustError(f"{label} overlaps protected authority {authority}")
    return tuple(targets.values())  # type: ignore[return-value]


def _create_root(path: Path, *, label: str) -> Path:
    selected = path.resolve()
    if selected.exists():
        raise TrustError(f"{label} must still be absent: {selected}")
    try:
        os.mkdir(selected)
    except FileExistsError as error:
        raise TrustError(f"{label} lost exclusive-create race: {selected}") from error
    return selected


def _write_json_exclusive(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise TrustError(f"completion artifact already exists: {path}") from error


def _copy_file_exclusive(source: Path, destination: Path, expected: Mapping[str, Any]) -> None:
    if _file_record(source) != dict(expected):
        raise TrustError(f"authority source changed before freeze copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    except FileExistsError as error:
        raise TrustError(f"frozen capsule copy target already exists: {destination}") from error
    measured = {"bytes": size, "sha256": digest.hexdigest()}
    if measured != dict(expected) or _file_record(destination) != measured:
        raise TrustError(f"frozen capsule copy is not byte-exact: {destination}")


def _clone_gpuwm(snapshot: Mapping[str, Any], capsule: Path) -> Path:
    # The source path is deliberately taken from the snapshot root rather than
    # from any child-controlled value.
    original_root = Path(snapshot["gpuwm_sources"].get("root", ""))
    if not original_root.is_dir():
        raise TrustError("GPUWM source snapshot has no real root")
    destination = capsule / "woof"
    try:
        cloned = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-local",
                "--no-checkout",
                str(original_root),
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if cloned.stdout or cloned.stderr:
            raise TrustError("quiet GPUWM clone emitted unexpected output")
        checkout = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={destination.as_posix()}",
                "-C",
                str(destination),
                "checkout",
                "--quiet",
                "HEAD",
                "--",
                *GPUWM_SOURCE_PATHS,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if checkout.stdout or checkout.stderr:
            raise TrustError("quiet GPUWM sparse checkout emitted unexpected output")
    except (OSError, subprocess.SubprocessError) as error:
        raise TrustError(f"cannot create frozen GPUWM checkout: {error}") from error
    measured = _gpuwm_inventory(destination)
    expected = snapshot["gpuwm_sources"]
    if measured["git_head"] != expected["git_head"] or measured["files"] != expected["files"]:
        raise TrustError("frozen GPUWM checkout differs from live pinned sources")
    return destination


def _freeze_capsule(
    target: Path,
    *,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    capsule = _create_root(target, label="frozen capsule root")
    source_root = Path(snapshot["mpas_source"]["root"])
    for relative, record in snapshot["mpas_source"]["files"].items():
        _copy_file_exclusive(
            source_root / relative,
            capsule / "src" / "hexcore" / relative,
            record,
        )
    _copy_file_exclusive(
        Path(snapshot["frozen_tool"]["path"]),
        capsule / "tools" / FROZEN_TOOL.name,
        {key: snapshot["frozen_tool"][key] for key in ("bytes", "sha256")},
    )
    _copy_file_exclusive(
        Path(snapshot["isolated_bootstrap"]["path"]),
        capsule / "tools" / ISOLATED_BOOTSTRAP.name,
        {key: snapshot["isolated_bootstrap"][key] for key in ("bytes", "sha256")},
    )
    _copy_file_exclusive(
        Path(snapshot["authority_pins"]["path"]),
        capsule / "tools" / AUTHORITY_PINS.name,
        {key: snapshot["authority_pins"][key] for key in ("bytes", "sha256")},
    )
    for label, name in (
        ("authority_initial_condition", "authority_init.nc"),
        ("native_internal_t0", "nomix_internal_t0.nc"),
    ):
        row = snapshot["inputs"][label]
        _copy_file_exclusive(
            Path(row["path"]),
            capsule / "inputs" / name,
            {key: row[key] for key in ("bytes", "sha256")},
        )
    _copy_file_exclusive(
        Path(snapshot["ftz_binding"]["path"]),
        capsule / "ftz" / "binding.json",
        {key: snapshot["ftz_binding"][key] for key in ("bytes", "sha256")},
    )
    receipt_root = Path(snapshot["gpuwm_receipt"]["root"])
    for relative, record in snapshot["gpuwm_receipt"]["files"].items():
        _copy_file_exclusive(
            receipt_root / relative,
            capsule / "ftz" / "gpuwm-probe" / relative,
            record,
        )
    runtime_entries = snapshot["runtime_dependencies"]["entries"]
    for relative, record in snapshot["runtime_dependencies"]["files"].items():
        head, separator, tail = relative.partition("/")
        entry = runtime_entries.get(head)
        if not isinstance(entry, Mapping):
            raise TrustError(f"runtime dependency file has no source entry: {relative}")
        source_entry = Path(str(entry["source_path"]))
        source = source_entry / tail if separator else source_entry
        _copy_file_exclusive(source, capsule / "runtime" / relative, record)
    _clone_gpuwm(snapshot, capsule)
    manifest_path = capsule / "freeze-manifest.json"
    capsule_files = _tree_inventory(
        capsule,
        excluded_files=frozenset({manifest_path.name}),
        ignore_git=True,
    )
    external_driver_files = {
        path: record
        for path, record in snapshot["system_cuda_libraries"]["files"].items()
        if Path(path).name.lower() == "nvcuda.dll"
    }
    if not external_driver_files:
        raise TrustError("frozen manifest has no external NVIDIA driver baseline")
    manifest = {
        "schema": FREEZE_SCHEMA,
        "capsule_files": capsule_files,
        "capsule_files_sha256": _canonical_sha256(capsule_files),
        "gpuwm_git_head": snapshot["gpuwm_sources"]["git_head"],
        "runtime_capsule_entries": snapshot["runtime_dependencies"]["entries"],
        "runtime_capsule_entries_sha256": _canonical_sha256(
            snapshot["runtime_dependencies"]["entries"]
        ),
        "runtime_dependency_files_sha256": snapshot["runtime_dependencies"][
            "files_sha256"
        ],
        "external_nvidia_driver_files": external_driver_files,
        "external_nvidia_driver_files_sha256": _canonical_sha256(
            external_driver_files
        ),
        "allowed_python_runtime_files": snapshot["python_runtime"]["files"],
        "allowed_python_runtime_files_sha256": _canonical_sha256(
            snapshot["python_runtime"]["files"]
        ),
        "allowed_external_native_files": dict(
            sorted(
                {
                    **snapshot["host_native_baseline"]["files"],
                    **snapshot["system_cuda_libraries"]["files"],
                    **{
                        path: record
                        for path, record in snapshot["python_runtime"]["files"].items()
                        if Path(path).suffix.lower() in {".dll", ".exe", ".pyd", ".so", ".dylib"}
                    },
                }.items()
            )
        ),
    }
    manifest["allowed_external_native_files_sha256"] = _canonical_sha256(
        manifest["allowed_external_native_files"]
    )
    _write_json_exclusive(manifest_path, manifest)
    return {
        "root": str(capsule),
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_file": _file_record(manifest_path),
    }


def _frozen_snapshot(capsule_record: Mapping[str, Any]) -> dict[str, Any]:
    capsule = Path(capsule_record["root"])
    manifest_path = Path(capsule_record["manifest_path"])
    manifest, _ = _load_json(manifest_path, label="freeze manifest")
    if manifest != capsule_record["manifest"]:
        raise TrustError("freeze manifest changed after capsule creation")
    files = _tree_inventory(
        capsule,
        excluded_files=frozenset({manifest_path.relative_to(capsule).as_posix()}),
        ignore_git=True,
    )
    if files != manifest.get("capsule_files"):
        raise TrustError("frozen capsule changed after creation")
    woof = capsule / "woof"
    if _git_head(woof) != manifest.get("gpuwm_git_head"):
        raise TrustError("frozen GPUWM HEAD changed after capsule creation")
    return {
        "files": files,
        "files_sha256": _canonical_sha256(files),
        "manifest_file": _file_record(manifest_path),
        "gpuwm_git_head": _git_head(woof),
    }


def _nvidia_smi_executable() -> Path:
    selected = shutil.which("nvidia-smi")
    if selected is None:
        raise TrustError("nvidia-smi is required for platform checks")
    executable = Path(selected).resolve()
    if not executable.is_file():
        raise TrustError(f"nvidia-smi is not a regular file: {executable}")
    return executable


def _run_nvidia_smi(executable: Path, query: str, fields: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            [
                str(executable),
                f"--query-{query}=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TrustError(f"nvidia-smi {query} query failed: {error}") from error
    return completed.stdout


def _csv_rows(payload: str) -> list[list[str]]:
    return [
        [value.strip() for value in row]
        for row in csv.reader(io.StringIO(payload))
        if row and any(value.strip() for value in row)
    ]


def _gpu_state(executable: Path) -> dict[str, Any]:
    gpu_fields = (
        "index",
        "uuid",
        "name",
        "pci.bus_id",
        "compute_cap",
        "driver_version",
        "compute_mode",
    )
    rows = _csv_rows(_run_nvidia_smi(executable, "gpu", gpu_fields))
    gpus: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != len(gpu_fields):
            raise TrustError(f"nvidia-smi GPU row is malformed: {row!r}")
        try:
            index = int(row[0])
        except ValueError as error:
            raise TrustError(f"nvidia-smi GPU index is invalid: {row[0]!r}") from error
        if _GPU_UUID_RE.fullmatch(row[1]) is None or re.fullmatch(r"\d+\.\d+", row[4]) is None:
            raise TrustError("nvidia-smi GPU identity is invalid")
        if any(value in {"", "N/A", "[Not Supported]"} for value in row[2:]):
            raise TrustError("nvidia-smi GPU identity is incomplete")
        gpus.append(
            {
                "index": index,
                "uuid": row[1],
                "name": row[2],
                "pci_bus_id": row[3].upper(),
                "compute_capability": row[4],
                "nvidia_driver_version": row[5],
                "compute_mode": row[6],
            }
        )
    gpus.sort(key=lambda item: int(item["index"]))
    if not gpus or len({row["index"] for row in gpus}) != len(gpus):
        raise TrustError("nvidia-smi GPU inventory is empty or duplicated")
    process_fields = ("gpu_uuid", "pid", "process_name")
    process_rows = _csv_rows(_run_nvidia_smi(executable, "compute-apps", process_fields))
    if len(process_rows) == 1 and process_rows[0][0].lower().startswith("no running processes"):
        process_rows = []
    known_uuids = {str(row["uuid"]) for row in gpus}
    processes: list[dict[str, Any]] = []
    for row in process_rows:
        if len(row) != len(process_fields) or row[0] not in known_uuids:
            raise TrustError(f"nvidia-smi process row is invalid: {row!r}")
        try:
            pid = int(row[1])
        except ValueError as error:
            raise TrustError(f"nvidia-smi process PID is invalid: {row[1]!r}") from error
        if not row[2]:
            raise TrustError("nvidia-smi process name is empty")
        processes.append({"gpu_uuid": row[0], "pid": pid, "process_name": row[2]})
    processes.sort(key=lambda row: (str(row["gpu_uuid"]), int(row["pid"])))
    return {
        "gpu_inventory": gpus,
        "reported_processes": processes,
        "physical_gpu_exclusivity_claim": False,
        "wddm_interpretation": (
            "requires an unchanged full PID/UUID/name baseline; does not claim "
            "physical GPU exclusivity"
        ),
    }


def _platform_snapshot() -> dict[str, Any]:
    executable = _nvidia_smi_executable()
    python = Path(sys.executable).resolve()
    return {
        "python": {"path": str(python), **_file_record(python)},
        "nvidia_smi": {"path": str(executable), **_file_record(executable)},
        "gpu_state": _gpu_state(executable),
    }


def _invoke_child(
    argv: Sequence[str], *, cwd: Path, timeout_seconds: int
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.upper().startswith("PYTHON"):
            environment.pop(key, None)
    for key in (
        "CUPY_CACHE_DIR",
        "MPAS_PORT_CUDA_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "WOOF_ROOT",
    ):
        environment.pop(key, None)
    return subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=False,
        timeout=timeout_seconds,
        check=False,
    )


def _parse_child_stdout(payload: bytes) -> dict[str, Any]:
    if not payload.startswith(EXPECTED_CHILD_PREFIX):
        raise TrustError("measured child stdout progress prefix changed or is noisy")
    summary_payload = payload[len(EXPECTED_CHILD_PREFIX) :]
    summary = _strict_json_bytes(summary_payload, label="measured child summary")
    canonical = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if summary_payload != canonical:
        raise TrustError("measured child summary is not one canonical JSON document")
    return summary


def _validate_child_result(child: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    if child.returncode != 0:
        raise TrustError(
            f"frozen dual-run child exited {child.returncode}; no completion allowed"
        )
    if child.stderr != b"":
        raise TrustError("frozen dual-run child wrote stderr; no completion allowed")
    return _parse_child_stdout(child.stdout)


def _reject_nondeterministic_fields(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            selected = str(key)
            child_path = f"{path}.{selected}" if path else selected
            if selected in _NONDETERMINISTIC_KEYS:
                raise TrustError(f"capsule contains nondeterministic field {child_path}")
            _reject_nondeterministic_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nondeterministic_fields(child, f"{path}[{index}]")


def _expected_source_bindings(capsule: Path) -> dict[str, dict[str, str]]:
    source = capsule / "src" / "hexcore"
    return {
        label: {"path": relative, "sha256": _file_record(source / relative)["sha256"]}
        for label, relative in EXPECTED_SOURCE_PATHS.items()
    }


def _validate_with_frozen_source(capsule: Path, arms: Sequence[Mapping[str, Any]]) -> None:
    leaked = sorted(
        name for name in sys.modules if name == "hexcore" or name.startswith("hexcore.")
    )
    if leaked:
        raise TrustError(f"MPAS modules were imported before frozen validation: {leaked}")
    source_root = capsule / "src"
    sys.path.insert(0, str(source_root))
    imported: list[str] = []
    try:
        module = importlib.import_module("woof.hex.cuda_dualrun")
        imported = [
            name for name in sys.modules if name == "hexcore" or name.startswith("hexcore.")
        ]
        module_path = Path(module.__file__).resolve()
        expected_path = (source_root / "hexcore" / "cuda_dualrun.py").resolve()
        if module_path != expected_path:
            raise TrustError("capsule validator was not imported from frozen source")
        for arm in arms:
            module.validate_cuda_capsule(arm)
    except TrustError:
        raise
    except BaseException as error:
        raise TrustError(f"frozen MPAS capsule validator refused output: {error}") from error
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)
        for name in imported:
            sys.modules.pop(name, None)


def _load_frozen_comparator(capsule: Path) -> Any:
    source = (capsule / "woof" / "woof" / "certify" / "dualrun.py").resolve()
    digest = _file_record(source)["sha256"]
    module_name = f"_frozen_gpuwm_dualrun_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise TrustError("cannot load frozen GPUWM comparator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        sys.modules.pop(module_name, None)
        raise TrustError(f"frozen GPUWM comparator import failed: {error}") from error
    for name in ("compare_capsules", "compare_capsule_files", "flatten_capsule"):
        if not callable(getattr(module, name, None)):
            raise TrustError(f"frozen GPUWM comparator lacks {name}()")
    return module


def _validate_capsule_contract(
    arm: Mapping[str, Any],
    *,
    capsule: Path,
    binding: Mapping[str, Any],
    input_records: Mapping[str, Any],
    expected_platform: Mapping[str, Any],
) -> None:
    if set(arm) != EXPECTED_TOP_KEYS or arm.get("schema") != CAPSULE_SCHEMA:
        raise TrustError("CUDA arm top-level schema or field set changed")
    _reject_nondeterministic_fields(arm)
    if arm.get("profile") != TARGET_PROFILE:
        raise TrustError("CUDA arm profile changed")
    configuration = arm.get("configuration")
    config_sha = _canonical_sha256(EXPECTED_CONFIGURATION)
    if configuration != {"value": EXPECTED_CONFIGURATION, "sha256": config_sha}:
        raise TrustError("CUDA arm configuration differs from the fixed 24-step lane")
    if arm.get("input_bytes") != input_records:
        raise TrustError("CUDA arm input-byte binding differs from frozen inputs")
    contracts = arm.get("contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != {
        "comparison_authority",
        "compile_manifest",
        "ftz_binding",
        "implementation_sources",
        "layout",
    }:
        raise TrustError("CUDA arm contract inventory changed")
    comparator_source = capsule / "woof" / "woof" / "certify" / "dualrun.py"
    expected_comparison_authority = {
        "schema": COMPARISON_AUTHORITY_SCHEMA,
        "module": "woof.certify.dualrun",
        "source_path": "woof/certify/dualrun.py",
        "source_sha256": _file_record(comparator_source)["sha256"],
        "functions": ["compare_capsules", "compare_capsule_files"],
        "comparison_schema": COMPARISON_SCHEMA,
        "comparison_scope": "total leaf comparison with no ignore list",
    }
    if contracts.get("comparison_authority") != expected_comparison_authority:
        raise TrustError("CUDA arm comparison authority differs from frozen GPUWM")
    manifest = binding.get("compile_manifest")
    manifest_sha = _canonical_sha256(manifest)
    if contracts.get("compile_manifest") != {"value": manifest, "sha256": manifest_sha}:
        raise TrustError("CUDA arm compile manifest differs from frozen FTZ binding")
    if contracts.get("layout") != {
        "value": EXPECTED_LAYOUT,
        "sha256": _canonical_sha256(EXPECTED_LAYOUT),
    }:
        raise TrustError("CUDA arm layout contract changed")
    frozen_binding_record = _file_record(capsule / "ftz" / "binding.json")
    if contracts.get("ftz_binding") != {
        "schema": FTZ_SCHEMA,
        "artifact_sha256": frozen_binding_record["sha256"],
        "sha256": _canonical_sha256(binding),
        "value": binding,
    }:
        raise TrustError("CUDA arm FTZ binding differs from frozen authority")
    if contracts.get("implementation_sources") != _expected_source_bindings(capsule):
        raise TrustError("CUDA arm implementation-source binding changed")
    device = arm.get("device")
    if not isinstance(device, Mapping) or set(device) != {
        "compute_capability",
        "cupy_version",
        "device_id",
        "driver_version",
        "multiprocessor_count",
        "name",
        "nvrtc_version",
        "runtime_version",
        "sm",
        "total_memory_bytes",
    }:
        raise TrustError("CUDA arm device binding changed")
    fingerprint = expected_platform.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise TrustError("pinned compile platform lacks a fingerprint")
    if (
        device.get("compute_capability") != "12.0"
        or device.get("sm") != "sm_120"
        or str(device.get("driver_version")) != fingerprint.get("cuda_driver_version")
        or device.get("cupy_version") != fingerprint.get("cupy_version")
        or fingerprint.get("device_compute_capability") != "120"
        or device.get("device_id") != 0
        or not isinstance(device.get("name"), str)
        or int(device.get("multiprocessor_count", 0)) <= 0
        or int(device.get("total_memory_bytes", 0)) <= 0
    ):
        raise TrustError("CUDA arm device differs from pinned sm_120 compile platform")

    preparation = arm.get("preparation")
    if not isinstance(preparation, Mapping) or set(preparation) != {
        "configuration_sha256",
        "initial_execution_fingerprint_sha256",
        "initial_snapshot_sha256",
        "input_bytes",
        "method",
        "profile",
        "sha256",
        "target",
    }:
        raise TrustError("CUDA arm preparation binding changed")
    prep_core = {
        "profile": preparation.get("profile"),
        "target": preparation.get("target"),
        "method": preparation.get("method"),
        "configuration_sha256": preparation.get("configuration_sha256"),
        "input_bytes": preparation.get("input_bytes"),
        "initial_snapshot_sha256": preparation.get("initial_snapshot_sha256"),
        "initial_execution_fingerprint_sha256": preparation.get(
            "initial_execution_fingerprint_sha256"
        ),
    }
    if (
        prep_core["profile"] != TARGET_PROFILE
        or prep_core["target"] != TARGET_NAME
        or prep_core["method"] != TARGET_METHOD
        or prep_core["configuration_sha256"] != config_sha
        or prep_core["input_bytes"] != input_records
        or preparation.get("sha256") != _canonical_sha256(prep_core)
    ):
        raise TrustError("CUDA arm shared-preparation binding is false")

    trajectory = arm.get("trajectory")
    if not isinstance(trajectory, Mapping) or set(trajectory) != {
        "dt_seconds",
        "final_snapshot_sha256",
        "initial_execution_fingerprint",
        "initial_snapshot",
        "sha256",
        "step_records",
        "steps",
        "target",
    }:
        raise TrustError("CUDA arm trajectory inventory changed")
    records = trajectory.get("step_records")
    if (
        trajectory.get("steps") != TARGET_STEPS
        or trajectory.get("dt_seconds") != TARGET_DT_SECONDS
        or trajectory.get("target") != TARGET_NAME
        or not isinstance(records, list)
        or len(records) != TARGET_STEPS
    ):
        raise TrustError("CUDA arm is not the exact 24-step/86400-second trajectory")
    if preparation.get("initial_snapshot_sha256") != trajectory.get("initial_snapshot", {}).get("sha256"):
        raise TrustError("CUDA arm preparation/initial-snapshot binding is false")
    if preparation.get("initial_execution_fingerprint_sha256") != trajectory.get(
        "initial_execution_fingerprint", {}
    ).get("sha256"):
        raise TrustError("CUDA arm preparation/execution-input binding is false")
    expected_start = float(trajectory["initial_snapshot"]["model_time_seconds"])
    layout_sha = _canonical_sha256(EXPECTED_LAYOUT)
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping) or set(record) != {
            "step",
            "start_time_seconds",
            "end_time_seconds",
            "snapshot",
            "step_contract",
            "sha256",
        }:
            raise TrustError(f"CUDA trajectory step {index} inventory changed")
        start = record.get("start_time_seconds")
        end = record.get("end_time_seconds")
        if record.get("step") != index or start != expected_start or end != expected_start + TARGET_DT_SECONDS:
            raise TrustError(f"CUDA trajectory time sequence changed at step {index}")
        contract = record.get("step_contract")
        if not isinstance(contract, Mapping) or set(contract) != EXPECTED_STEP_CONTRACT_KEYS:
            raise TrustError(f"CUDA step {index} contract inventory changed")
        if (
            contract.get("authority_ruler") is not None
            or contract.get("authority_ruler_sha256") is not None
            or contract.get("compile_manifest_sha256") != manifest_sha
            or contract.get("configuration") != EXPECTED_CONFIGURATION
            or contract.get("configuration_sha256") != config_sha
            or contract.get("d2h_bytes_inside_step") != 0
            or contract.get("evidence") != "implemented-cuda-dry-rk3-unlinked"
            or contract.get("frozen_source")
            != "MPAS-A v8.2.3 mpas_atm_time_integration.F:638-1224"
            or contract.get("layout_contract_sha256") != layout_sha
            or contract.get("stage_acoustic_steps") != [1, 3, 6]
            or contract.get("t0_diagnostics_source") != "uploaded-exact-sidecar"
        ):
            raise TrustError(f"CUDA step {index} whole-step/no-D2H contract changed")
        step_core = {
            "step": record.get("step"),
            "start_time_seconds": start,
            "end_time_seconds": end,
            "snapshot": record.get("snapshot"),
            "step_contract": contract,
        }
        if record.get("sha256") != _canonical_sha256(step_core):
            raise TrustError(f"CUDA step {index} semantic digest is false")
        expected_start = float(end)
    if expected_start != TARGET_DURATION_SECONDS:
        raise TrustError("CUDA trajectory does not terminate at 86400 seconds")
    if trajectory.get("final_snapshot_sha256") != records[-1]["snapshot"]["sha256"]:
        raise TrustError("CUDA final-snapshot binding is false")
    trajectory_core = {
        "target": trajectory.get("target"),
        "steps": trajectory.get("steps"),
        "dt_seconds": trajectory.get("dt_seconds"),
        "initial_snapshot": trajectory.get("initial_snapshot"),
        "initial_execution_fingerprint": trajectory.get("initial_execution_fingerprint"),
        "step_records": records,
        "final_snapshot_sha256": trajectory.get("final_snapshot_sha256"),
    }
    if trajectory.get("sha256") != _canonical_sha256(trajectory_core):
        raise TrustError("CUDA trajectory semantic digest is false")


def _normalized_runtime_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    capsule: Path,
    module_records: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, row in sorted(records.items()):
        path_text = row.get("path") if module_records else key
        if not isinstance(path_text, str):
            raise TrustError("runtime closure record has no path")
        path = Path(path_text).resolve()
        if path == capsule or capsule in path.parents:
            normalized_path = f"capsule/{path.relative_to(capsule).as_posix()}"
        else:
            normalized_path = str(path)
        core = {
            "path": normalized_path,
            "bytes": row.get("bytes"),
            "sha256": row.get("sha256"),
        }
        result[str(key)] = core if module_records else {
            "bytes": core["bytes"],
            "sha256": core["sha256"],
            "normalized_path": normalized_path,
        }
    return result


def _validate_runtime_receipt(
    *,
    path: Path,
    capsule_record: Mapping[str, Any],
    pins: Mapping[str, Any],
) -> dict[str, Any]:
    receipt, _ = _load_json(path, label="isolated runtime-closure receipt")
    if set(receipt) != {
        "schema",
        "capsule_files_sha256",
        "runtime_capsule_entries_sha256",
        "runtime_dependency_files_sha256",
        "external_nvidia_driver_files_sha256",
        "allowed_python_runtime_files_sha256",
        "allowed_external_native_files_sha256",
        "pre",
        "post",
        "new_module_origins",
        "new_native_modules",
        "all_post_run_origins_within_static_frozen_or_external_pins",
    } or receipt.get("schema") != RUNTIME_RECEIPT_SCHEMA:
        raise TrustError("isolated runtime-closure receipt schema changed")
    manifest = capsule_record["manifest"]
    for key in (
        "capsule_files_sha256",
        "runtime_capsule_entries_sha256",
        "runtime_dependency_files_sha256",
        "external_nvidia_driver_files_sha256",
        "allowed_python_runtime_files_sha256",
        "allowed_external_native_files_sha256",
    ):
        if receipt.get(key) != manifest.get(key):
            raise TrustError(f"runtime receipt {key} differs from frozen manifest")
    if receipt.get("all_post_run_origins_within_static_frozen_or_external_pins") is not True:
        raise TrustError("runtime receipt does not claim complete admitted origins")
    phases: dict[str, dict[str, Any]] = {}
    for phase in ("pre", "post"):
        row = receipt.get(phase)
        if not isinstance(row, Mapping) or set(row) != {
            "module_origins",
            "module_origins_sha256",
            "native_modules",
            "native_modules_sha256",
        }:
            raise TrustError(f"runtime receipt {phase} inventory changed")
        modules = row.get("module_origins")
        natives = row.get("native_modules")
        if not isinstance(modules, Mapping) or not isinstance(natives, Mapping):
            raise TrustError(f"runtime receipt {phase} inventories are not objects")
        if row.get("module_origins_sha256") != _canonical_sha256(modules) or row.get(
            "native_modules_sha256"
        ) != _canonical_sha256(natives):
            raise TrustError(f"runtime receipt {phase} digest is false")
        phases[phase] = {"modules": dict(modules), "natives": dict(natives)}
    if receipt.get("new_module_origins") != sorted(
        set(phases["post"]["modules"]) - set(phases["pre"]["modules"])
    ) or receipt.get("new_native_modules") != sorted(
        set(phases["post"]["natives"]) - set(phases["pre"]["natives"])
    ):
        raise TrustError("runtime receipt new-origin inventories are false")

    capsule = Path(capsule_record["root"])
    capsule_files = manifest["capsule_files"]
    python_files = manifest["allowed_python_runtime_files"]
    external_native = manifest["allowed_external_native_files"]
    post_modules = phases["post"]["modules"]
    for name, record in post_modules.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise TrustError(f"runtime module-origin record {name!r} changed")
        selected = Path(str(record["path"])).resolve()
        core = {"bytes": record["bytes"], "sha256": record["sha256"]}
        if _file_record(selected) != core:
            raise TrustError(f"runtime module-origin bytes drifted: {selected}")
        if selected == capsule or capsule in selected.parents:
            if capsule_files.get(selected.relative_to(capsule).as_posix()) != core:
                raise TrustError(f"runtime module origin is not capsule-frozen: {selected}")
        elif python_files.get(str(selected)) != core:
            raise TrustError(f"runtime module origin escaped pinned Python runtime: {selected}")
    required_modules = {
        "cftime",
        "cupy",
        "cupy_backends.cuda.libs.nvrtc",
        "woof.certify.compile_platform",
        "netCDF4",
        "numpy",
        "hexcore.cuda_dualrun",
        "scipy",
    }
    missing_modules = sorted(required_modules - set(post_modules))
    if missing_modules or not any(
        name.startswith("_mpas_gpuwm_dualrun_") for name in post_modules
    ):
        raise TrustError(
            f"runtime closure is missing required imported modules: {missing_modules}"
        )

    post_natives = phases["post"]["natives"]
    for path_text, record in post_natives.items():
        if not isinstance(record, Mapping) or set(record) != {"bytes", "sha256"}:
            raise TrustError(f"runtime native record changed: {path_text}")
        selected = Path(path_text).resolve()
        core = dict(record)
        if _file_record(selected) != core:
            raise TrustError(f"runtime native bytes drifted: {selected}")
        if selected == capsule or capsule in selected.parents:
            if capsule_files.get(selected.relative_to(capsule).as_posix()) != core:
                raise TrustError(f"runtime native image is not capsule-frozen: {selected}")
        elif external_native.get(str(selected)) != core:
            raise TrustError(
                f"runtime native image escaped pinned external baseline: {selected}"
            )
    if not any(Path(path).name.lower() == "nvcuda.dll" for path in post_natives):
        raise TrustError("runtime closure did not load the pinned NVIDIA driver image")
    native_names = {Path(path).name.lower() for path in post_natives}
    if not any("hdf5" in name for name in native_names):
        raise TrustError("runtime closure did not load the pinned HDF5 native stack")
    if not any(
        marker in name
        for name in native_names
        for marker in ("openblas", "mkl", "fblas")
    ):
        raise TrustError("runtime closure did not load the pinned BLAS native stack")
    expected_nvrtc = pins["expected"]["compile_platform"]["fingerprint"][
        "nvrtc_library_sha256"
    ]
    nvrtc_rows = [
        record
        for path_text, record in post_natives.items()
        if Path(path_text).name.lower().startswith("nvrtc64_")
    ]
    if not nvrtc_rows or not any(row.get("sha256") == expected_nvrtc for row in nvrtc_rows):
        raise TrustError("runtime closure did not load the pinned NVRTC image")

    normalized_modules = _normalized_runtime_records(
        post_modules, capsule=capsule, module_records=True
    )
    normalized_natives = _normalized_runtime_records(
        post_natives, capsule=capsule, module_records=False
    )
    expected = pins["expected"]
    if len(post_modules) != expected["runtime_post_module_count"] or _canonical_sha256(
        normalized_modules
    ) != expected["runtime_post_module_origins_normalized_sha256"]:
        raise TrustError("post-run imported-module closure differs from static pins")
    if len(post_natives) != expected["runtime_post_native_count"] or _canonical_sha256(
        normalized_natives
    ) != expected["runtime_post_native_modules_normalized_sha256"]:
        raise TrustError("post-run loaded-native closure differs from static pins")
    return {
        "path": str(path),
        **_file_record(path),
        "module_count": len(post_modules),
        "module_origins_normalized_sha256": _canonical_sha256(normalized_modules),
        "native_count": len(post_natives),
        "native_modules_normalized_sha256": _canonical_sha256(normalized_natives),
        "new_module_count": len(receipt["new_module_origins"]),
        "new_native_count": len(receipt["new_native_modules"]),
        "all_origins_revalidated": True,
    }


def _validate_outputs(
    *,
    output_root: Path,
    cache_root: Path,
    capsule_record: Mapping[str, Any],
    binding: Mapping[str, Any],
    authority_snapshot: Mapping[str, Any],
    pins: Mapping[str, Any],
    child_summary: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_closure = _validate_runtime_receipt(
        path=cache_root / "runtime-closure.json",
        capsule_record=capsule_record,
        pins=pins,
    )
    output_files = _tree_inventory(output_root)
    if tuple(output_files) != EXPECTED_OUTPUT_NAMES:
        raise TrustError("measured output is not the exact three-file artifact set")
    cache_files = _tree_inventory(cache_root)
    if not cache_files:
        raise TrustError("fresh measured CUDA cache is empty")
    documents: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_OUTPUT_NAMES:
        documents[name], _ = _load_json(output_root / name, label=name)
    arm_a = documents[EXPECTED_OUTPUT_NAMES[0]]
    arm_b = documents[EXPECTED_OUTPUT_NAMES[1]]
    report = documents[EXPECTED_OUTPUT_NAMES[2]]
    if output_files[EXPECTED_OUTPUT_NAMES[0]] != output_files[EXPECTED_OUTPUT_NAMES[1]]:
        raise TrustError("CUDA arm files are not byte-identical")
    capsule = Path(capsule_record["root"])
    _validate_with_frozen_source(capsule, (arm_a, arm_b))
    input_records = {
        label: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for label, row in authority_snapshot["inputs"].items()
    }
    expected_platform = pins["expected"]["compile_platform"]
    for arm in (arm_a, arm_b):
        _validate_capsule_contract(
            arm,
            capsule=capsule,
            binding=binding,
            input_records=input_records,
            expected_platform=expected_platform,
        )
    comparator = _load_frozen_comparator(capsule)
    comparison = comparator.compare_capsules(arm_a, arm_b).report()
    expected_comparison = {
        "schema": COMPARISON_SCHEMA,
        "identical": True,
        "first_divergent_field": None,
        "divergence_count": 0,
        "divergences": [],
    }
    if comparison != expected_comparison:
        raise TrustError("frozen GPUWM total comparison found a divergence")
    comparator_sha = _file_record(
        capsule / "woof" / "woof" / "certify" / "dualrun.py"
    )["sha256"]
    comparison_authority = {
        "schema": COMPARISON_AUTHORITY_SCHEMA,
        "module": "woof.certify.dualrun",
        "source_path": "woof/certify/dualrun.py",
        "source_sha256": comparator_sha,
        "functions": ["compare_capsules", "compare_capsule_files"],
        "comparison_schema": COMPARISON_SCHEMA,
        "comparison_scope": "total leaf comparison with no ignore list",
    }
    expected_report = {
        "schema": REPORT_SCHEMA,
        "capsules": {
            "a": {"sha256": output_files[EXPECTED_OUTPUT_NAMES[0]]["sha256"]},
            "b": {"sha256": output_files[EXPECTED_OUTPUT_NAMES[1]]["sha256"]},
        },
        "comparison_authority": comparison_authority,
        "gpuwm_comparison": expected_comparison,
        "total_comparison": True,
    }
    if report != expected_report:
        raise TrustError("saved comparison report is not the frozen GPUWM total verdict")
    expected_summary = {
        "schema": REPORT_SCHEMA,
        "target_steps": TARGET_STEPS,
        "target_duration_seconds": TARGET_DURATION_SECONDS,
        "identical": True,
        "divergence_count": 0,
        "first_divergent_field": None,
        "capsule_sha256": expected_report["capsules"],
        "comparison_authority_sha256": comparator_sha,
    }
    if child_summary != expected_summary:
        raise TrustError("measured child stdout summary differs from validated artifacts")
    leaf_count = len(comparator.flatten_capsule(arm_a))
    if leaf_count <= 0:
        raise TrustError("GPUWM total comparison exposed no capsule leaves")
    return {
        "output_tree": {
            "root": str(output_root),
            "files": output_files,
            **_inventory_summary(output_files),
        },
        "cache_tree": {
            "root": str(cache_root),
            "files": cache_files,
            **_inventory_summary(cache_files),
        },
        "arm_file_byte_identical": True,
        "arm_file_sha256": output_files[EXPECTED_OUTPUT_NAMES[0]]["sha256"],
        "gpuwm_total_leaf_count": leaf_count,
        "gpuwm_comparison": comparison,
        "comparison_authority": comparison_authority,
        "steps": TARGET_STEPS,
        "duration_seconds": TARGET_DURATION_SECONDS,
        "d2h_bytes_inside_every_step": 0,
        "linked_authority_ruler_claim": False,
        "runtime_closure": runtime_closure,
    }


def _assert_unchanged(expected: Mapping[str, Any], measured: Mapping[str, Any], *, label: str) -> None:
    if dict(expected) != dict(measured):
        raise TrustError(
            f"{label} drifted: {_canonical_sha256(expected)} != {_canonical_sha256(measured)}"
        )


def _checkpoint(
    *,
    args: argparse.Namespace,
    authority_pre: Mapping[str, Any],
    platform_pre: Mapping[str, Any],
    frozen_pre: Mapping[str, Any],
    capsule_record: Mapping[str, Any],
    output_root: Path,
    cache_root: Path,
    measured: Mapping[str, Any],
    pins: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    authority = _authority_snapshot(args, pins=pins)
    platform = _platform_snapshot()
    frozen = _frozen_snapshot(capsule_record)
    output_files = _tree_inventory(output_root)
    cache_files = _tree_inventory(cache_root)
    _assert_unchanged(authority_pre, authority, label=f"authority at {phase}")
    _assert_unchanged(platform_pre, platform, label=f"platform at {phase}")
    _assert_unchanged(frozen_pre, frozen, label=f"frozen capsule at {phase}")
    if output_files != measured["output_tree"]["files"]:
        raise TrustError(f"measured output tree drifted at {phase}")
    if cache_files != measured["cache_tree"]["files"]:
        raise TrustError(f"measured cache tree drifted at {phase}")
    return {
        "phase": phase,
        "authority_sha256": _canonical_sha256(authority),
        "platform_sha256": _canonical_sha256(platform),
        "frozen_capsule_sha256": _canonical_sha256(frozen),
        "output_tree_sha256": _canonical_sha256(output_files),
        "cache_tree_sha256": _canonical_sha256(cache_files),
        "all_match_prevalidated": True,
    }


def build_parser() -> argparse.ArgumentParser:
    default_work = ROOT.parents[1] / "work" / "jw_step"
    parser = argparse.ArgumentParser(
        description="Trust-promote one frozen 24-step v8.2.3 CUDA JW dual run."
    )
    parser.add_argument("--initial", type=Path, default=default_work / "authority_init.nc")
    parser.add_argument("--native-t0", type=Path, default=default_work / "nomix_internal_t0.nc")
    parser.add_argument("--gpuwm-root", type=Path, default=Path.home() / "woof")
    parser.add_argument(
        "--gpuwm-probe",
        type=Path,
        default=ROOT / "receipts" / "cuda-ftz-sm120" / "gpuwm-probe",
    )
    parser.add_argument(
        "--ftz-binding",
        type=Path,
        default=ROOT / "receipts" / "cuda-ftz-sm120" / "binding.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    parser.add_argument("--capsule-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    return parser


def main(argv: list[str] | None = None) -> int:
    outer_authority = _validate_outer_authority()
    args = build_parser().parse_args(argv)
    if type(args.timeout_seconds) is not int or args.timeout_seconds <= 0:
        raise TrustError("--timeout-seconds must be a positive integer")
    pins = _load_authority_pins()
    bootstrap_record = _validate_bootstrap_pin()
    output_root, cache_root, completion_target, capsule_target = _validate_targets(
        output_root=args.output_root,
        cache_root=args.cache_root,
        completion_root=args.completion_root,
        capsule_root=args.capsule_root,
        args=args,
    )
    authority_pre = _authority_snapshot(args, pins=pins)
    binding = _validate_pre_pins(pins, authority_pre)
    launcher_record = _file_record(Path(__file__).resolve())
    python_record = _file_record(Path(sys.executable).resolve())
    platform_pre = _platform_snapshot()
    capsule_record = _freeze_capsule(capsule_target, snapshot=authority_pre)
    frozen_pre = _frozen_snapshot(capsule_record)

    capsule = Path(capsule_record["root"])
    frozen_tool = capsule / "tools" / FROZEN_TOOL.name
    frozen_bootstrap = capsule / "tools" / ISOLATED_BOOTSTRAP.name
    manifest_path = Path(capsule_record["manifest_path"])
    package_roots = (capsule / "runtime",)
    if not package_roots[0].is_dir() or package_roots[0].is_symlink():
        raise TrustError("frozen runtime package root is absent or unsafe")
    child_argv = [
        str(Path(sys.executable).resolve()),
        "-I",
        "-S",
        "-B",
        str(frozen_bootstrap),
        "--capsule-root",
        str(capsule),
        "--freeze-manifest",
        str(manifest_path),
        "--freeze-manifest-sha256",
        capsule_record["manifest_file"]["sha256"],
        "--frozen-tool",
        str(frozen_tool),
        "--frozen-tool-sha256",
        FROZEN_TOOL_SHA256,
        "--gpuwm-root",
        str(capsule / "woof"),
        "--gpuwm-head",
        authority_pre["gpuwm_sources"]["git_head"],
        "--runtime-receipt",
        str(cache_root / "runtime-closure.json"),
    ]
    for package_root in package_roots:
        child_argv.extend(("--package-root", str(package_root)))
    child_argv.extend(
        (
            "--",
            "--initial",
            str(capsule / "inputs" / "authority_init.nc"),
            "--native-t0",
            str(capsule / "inputs" / "nomix_internal_t0.nc"),
            "--gpuwm-root",
            str(capsule / "woof"),
            "--gpuwm-probe",
            str(capsule / "ftz" / "gpuwm-probe"),
            "--ftz-binding",
            str(capsule / "ftz" / "binding.json"),
            "--output-root",
            str(output_root),
            "--cache-root",
            str(cache_root),
            "--dt",
            "3600",
            "--duration",
            "86400",
            "--acoustic-substeps",
            "6",
        )
    )
    try:
        child = _invoke_child(child_argv, cwd=capsule, timeout_seconds=args.timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise TrustError(
            f"frozen dual-run child exceeded {args.timeout_seconds} seconds"
        ) from error
    child_summary = _validate_child_result(child)

    authority_post = _authority_snapshot(args, pins=pins)
    platform_post = _platform_snapshot()
    frozen_post = _frozen_snapshot(capsule_record)
    _assert_unchanged(authority_pre, authority_post, label="post-child authority")
    _assert_unchanged(platform_pre, platform_post, label="post-child platform")
    _assert_unchanged(frozen_pre, frozen_post, label="post-child frozen capsule")
    measured = _validate_outputs(
        output_root=output_root,
        cache_root=cache_root,
        capsule_record=capsule_record,
        binding=binding,
        authority_snapshot=authority_pre,
        pins=pins,
        child_summary=child_summary,
    )

    completion = _create_root(completion_target, label="completion root")
    receipt = {
        "schema": COMPLETION_RECEIPT_SCHEMA,
        "status": "validated-before-publication",
        "source_release": "v8.2.3",
        "weather_authority_claim": False,
        "outer_launcher_authority": outer_authority,
        "launcher": {"path": str(Path(__file__).resolve()), **launcher_record},
        "python": {"path": str(Path(sys.executable).resolve()), **python_record},
        "isolated_bootstrap": bootstrap_record,
        "static_authority_pins": {
            "path": str(AUTHORITY_PINS.resolve()),
            **_file_record(AUTHORITY_PINS),
            "document": pins,
        },
        "frozen_capsule": {**capsule_record, "validated_snapshot": frozen_pre},
        "child": {
            "argv": child_argv,
            "cwd": str(capsule),
            "timeout_seconds": args.timeout_seconds,
            "exit_code": child.returncode,
            "stdout": {
                "bytes": len(child.stdout),
                "sha256": _sha256_bytes(child.stdout),
                "exact_progress_then_one_canonical_json": True,
                "summary": child_summary,
            },
            "stderr": {
                "bytes": len(child.stderr),
                "sha256": _sha256_bytes(child.stderr),
                "empty": True,
            },
            "environment_policy": {
                "isolated_flags": ["-I", "-S", "-B"],
                "python_environment_removed": True,
                "cuda_cache_environment_removed": True,
                "frozen_runtime_roots_added_only_after_capsule_validation": [
                    str(path) for path in package_roots
                ],
                "live_site_package_roots_admitted": False,
            },
        },
        "authority_pre": authority_pre,
        "authority_post_child": authority_post,
        "authority_byte_identical": authority_pre == authority_post,
        "platform_pre": platform_pre,
        "platform_post_child": platform_post,
        "platform_byte_identical": platform_pre == platform_post,
        "measured_evidence": measured,
        "publication": {
            "output_root": str(output_root),
            "cache_root": str(cache_root),
            "capsule_root": str(capsule),
            "completion_root": str(completion),
            "all_roots_fresh_and_nonoverlapping": True,
            "completion_root_exclusively_created": True,
            "completion_writes_use_O_EXCL_and_fsync": True,
            "final_seal_required_for_promotion": True,
        },
    }
    receipt_path = completion / "completion-receipt.json"
    _write_json_exclusive(receipt_path, receipt)
    after_receipt = _checkpoint(
        args=args,
        authority_pre=authority_pre,
        platform_pre=platform_pre,
        frozen_pre=frozen_pre,
        capsule_record=capsule_record,
        output_root=output_root,
        cache_root=cache_root,
        measured=measured,
        pins=pins,
        phase="after-completion-receipt-publication",
    )
    receipt_record = _file_record(receipt_path)
    receipt_reloaded, _ = _load_json(receipt_path, label="completion receipt")
    if receipt_reloaded != receipt:
        raise TrustError("completion receipt changed after publication")

    manifest = {
        "schema": COMPLETION_MANIFEST_SCHEMA,
        "status": "complete",
        "promotion_trust_root": False,
        "final_seal_required_for_promotion": True,
        "source_release": "v8.2.3",
        "outer_launcher_authority": outer_authority,
        "completion_receipt": {
            "path": str(receipt_path),
            **receipt_record,
            "canonical_sha256": _canonical_sha256(receipt),
        },
        "launcher": {"path": str(Path(__file__).resolve()), **launcher_record},
        "static_authority_pins": {
            "path": str(AUTHORITY_PINS.resolve()),
            **_file_record(AUTHORITY_PINS),
        },
        "isolated_bootstrap": bootstrap_record,
        "frozen_tool": {"path": str(FROZEN_TOOL.resolve()), **_file_record(FROZEN_TOOL)},
        "after_receipt_publication_checkpoint": after_receipt,
        "claim": (
            "The exact frozen v8.2.3 JW lane completed two byte-identical "
            "24-step/86400-second resident CUDA trajectories. Every step carries "
            "the exact unlinked whole-step contract and zero internal D2H bytes; "
            "the pinned GPUWM comparator compared every leaf with no ignore list. "
            "This is deterministic durability evidence, not a weather-authority "
            "or physical GPU-exclusivity claim."
        ),
    }
    manifest_path = completion / "completion-manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    pre_seal_files = _tree_inventory(completion)
    if set(pre_seal_files) != {"completion-receipt.json", "completion-manifest.json"}:
        raise TrustError("completion root has unexpected files before final seal")
    if pre_seal_files["completion-receipt.json"] != receipt_record:
        raise TrustError("completion receipt drifted during manifest publication")
    after_manifest = _checkpoint(
        args=args,
        authority_pre=authority_pre,
        platform_pre=platform_pre,
        frozen_pre=frozen_pre,
        capsule_record=capsule_record,
        output_root=output_root,
        cache_root=cache_root,
        measured=measured,
        pins=pins,
        phase="after-completion-manifest-publication",
    )
    receipt_after_manifest = _file_record(receipt_path)
    manifest_record = _file_record(manifest_path)
    manifest_reloaded, _ = _load_json(manifest_path, label="completion manifest")
    if receipt_after_manifest != receipt_record or manifest_reloaded != manifest:
        raise TrustError("completion receipt or manifest drifted before final seal")
    seal = {
        "schema": COMPLETION_SEAL_SCHEMA,
        "status": "complete",
        "promotion_trust_root": True,
        "source_release": "v8.2.3",
        "outer_launcher_authority": outer_authority,
        "finite_trust_boundary": (
            "This seal binds the checkpoint after manifest publication. No "
            "authority-dependent computation or external write follows it; only "
            "O_EXCL+fsync serialization and completion-tree read-back remain."
        ),
        "completion_receipt": {"path": str(receipt_path), **receipt_after_manifest},
        "completion_manifest": {
            "path": str(manifest_path),
            **manifest_record,
            "canonical_sha256": _canonical_sha256(manifest),
        },
        "launcher": {"path": str(Path(__file__).resolve()), **launcher_record},
        "static_authority_pins": {
            "path": str(AUTHORITY_PINS.resolve()),
            **_file_record(AUTHORITY_PINS),
        },
        "isolated_bootstrap": bootstrap_record,
        "frozen_tool": {"path": str(FROZEN_TOOL.resolve()), **_file_record(FROZEN_TOOL)},
        "post_manifest_checkpoint": after_manifest,
        "exact_results": {
            "steps": TARGET_STEPS,
            "duration_seconds": TARGET_DURATION_SECONDS,
            "arm_file_byte_identical": True,
            "arm_file_sha256": measured["arm_file_sha256"],
            "gpuwm_divergence_count": 0,
            "d2h_bytes_inside_every_step": 0,
            "exact_output_file_count": 3,
        },
    }
    seal_path = completion / "completion-seal.json"
    _write_json_exclusive(seal_path, seal)
    seal_record = _file_record(seal_path)
    seal_reloaded, _ = _load_json(seal_path, label="completion seal")
    completion_files = _tree_inventory(completion)
    if seal_reloaded != seal or set(completion_files) != {
        "completion-receipt.json",
        "completion-manifest.json",
        "completion-seal.json",
    }:
        raise TrustError("final completion seal or three-file completion tree is invalid")
    if completion_files["completion-receipt.json"] != receipt_after_manifest or completion_files[
        "completion-manifest.json"
    ] != manifest_record or completion_files["completion-seal.json"] != seal_record:
        raise TrustError("completion tree changed during final-seal publication")
    summary = {
        "schema": COMPLETION_SEAL_SCHEMA,
        "status": "complete",
        "promotion_trust_root": True,
        "completion_root": str(completion),
        "completion_seal": str(seal_path),
        "completion_seal_sha256": seal_record["sha256"],
        "steps": TARGET_STEPS,
        "duration_seconds": TARGET_DURATION_SECONDS,
        "arm_file_sha256": measured["arm_file_sha256"],
        "gpuwm_divergence_count": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrustError as error:
        raise SystemExit(f"trust refusal: {error}") from error
