#!/usr/bin/env python3
"""Isolated bootstrap for the frozen v8.2.3 CUDA JW dual-run child.

The trust launcher starts this file with ``-I -S -B``.  This module uses only
the standard library until it has re-hashed the complete frozen capsule.  It
then exposes the pinned GPUWM checkout as namespace packages, adds the runtime
package directories without processing ``.pth`` files, and executes the
byte-pinned dual-run tool from the frozen capsule.
"""

from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import hashlib
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from types import ModuleType
from typing import Any, Sequence


FREEZE_SCHEMA = "mpas-port.cuda-jw-dualrun-frozen-capsule/v1"
RUNTIME_RECEIPT_SCHEMA = "mpas-port.cuda-jw-dualrun-runtime-closure/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BootstrapError(RuntimeError):
    """The child did not enter through the required frozen boundary."""


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
            raise BootstrapError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise BootstrapError(f"JSON contains non-finite token {value!r}")


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} root must be an object")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"frozen capsule entry is not a regular file: {path}")
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = path.stat(follow_symlinks=False)
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
        raise BootstrapError(f"frozen capsule entry changed while hashing: {path}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _real_directory(value: str, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise BootstrapError(f"{label} must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_dir():
        raise BootstrapError(f"{label} is not a real directory: {selected}")
    return selected


def _real_file(value: str, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise BootstrapError(f"{label} must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_file():
        raise BootstrapError(f"{label} is not a real file: {selected}")
    return selected


def _within(root: Path, path: Path, *, label: str) -> Path:
    selected = path.resolve()
    if selected == root or root not in selected.parents:
        raise BootstrapError(f"{label} escapes the frozen capsule: {selected}")
    return selected


def _tree_inventory(root: Path, *, excluded: frozenset[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for current, directory_names, file_names in os.walk(root, topdown=True):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(root)
            if name == ".git":
                continue
            if candidate.is_symlink():
                raise BootstrapError(
                    f"frozen capsule contains a directory symlink: {candidate}"
                )
            if relative.as_posix() not in excluded:
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if relative in excluded:
                continue
            result[relative] = _file_record(candidate)
    return dict(sorted(result.items()))


def _git_head(root: Path) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BootstrapError(f"cannot resolve frozen GPUWM HEAD: {error}") from error
    head = completed.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise BootstrapError(f"frozen GPUWM HEAD is invalid: {head!r}")
    return head


def _native_module_paths() -> tuple[Path, ...]:
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
            raise BootstrapError("EnumProcessModules failed")
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
        raise BootstrapError("native module inventory is unsupported")
    return tuple(sorted(paths, key=lambda path: str(path).lower()))


def _native_inventory() -> dict[str, dict[str, Any]]:
    return {str(path): _file_record(path) for path in _native_module_paths()}


def _module_origin_inventory() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, module in sorted(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            continue
        candidate = Path(origin).resolve()
        if not candidate.is_file():
            continue
        result[name] = {"path": str(candidate), **_file_record(candidate)}
    return result


def _validate_runtime_closure(
    *,
    capsule: Path,
    manifest: dict[str, Any],
    module_origins: dict[str, dict[str, Any]],
    native_modules: dict[str, dict[str, Any]],
) -> None:
    capsule_files = manifest["capsule_files"]
    python_files = manifest["allowed_python_runtime_files"]
    external_native = manifest["allowed_external_native_files"]
    for name, record in module_origins.items():
        path = Path(record["path"])
        if path == capsule or capsule in path.parents:
            relative = path.relative_to(capsule).as_posix()
            if capsule_files.get(relative) != {
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }:
                raise BootstrapError(
                    f"post-run module {name!r} is not frozen in the capsule: {path}"
                )
        elif python_files.get(str(path)) != {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }:
            raise BootstrapError(
                f"post-run Python module {name!r} escaped pinned runtime: {path}"
            )
    for path_text, record in native_modules.items():
        path = Path(path_text)
        core = {"bytes": record["bytes"], "sha256": record["sha256"]}
        if path == capsule or capsule in path.parents:
            relative = path.relative_to(capsule).as_posix()
            if capsule_files.get(relative) != core:
                raise BootstrapError(
                    f"post-run native image is not frozen in capsule: {path}"
                )
        elif external_native.get(str(path)) != core:
            raise BootstrapError(
                f"post-run native image escaped pinned external baseline: {path}"
            )


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise BootstrapError(f"runtime receipt already exists: {path}") from error


def _assert_isolated_startup() -> None:
    required = {
        "isolated": 1,
        "ignore_environment": 1,
        "no_site": 1,
        "no_user_site": 1,
        "dont_write_bytecode": 1,
        "safe_path": True,
    }
    mismatches = {
        name: (getattr(sys.flags, name, None), expected)
        for name, expected in required.items()
        if getattr(sys.flags, name, None) != expected
    }
    if mismatches:
        raise BootstrapError(f"Python isolated-startup flags are false: {mismatches}")
    loaded_hooks = sorted(
        name for name in ("site", "sitecustomize", "usercustomize") if name in sys.modules
    )
    if loaded_hooks:
        raise BootstrapError(
            f"Python startup customization ran before the bootstrap: {loaded_hooks}"
        )
    leaked = sorted(name for name in sys.modules if name == "hexcore" or name.startswith("hexcore."))
    if leaked:
        raise BootstrapError(f"MPAS modules existed before capsule validation: {leaked}")


def _namespace(name: str, directory: Path) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = None
    module.__loader__ = None
    module.__package__ = name
    module.__path__ = [str(directory)]  # type: ignore[attr-defined]
    spec = ModuleSpec(name=name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    module.__spec__ = spec
    return module


def _install_gpuwm_namespace(root: Path) -> None:
    package = root / "woof"
    certify = package / "certify"
    compile_platform = certify / "compile_platform.py"
    comparator = certify / "dualrun.py"
    if not compile_platform.is_file() or not comparator.is_file():
        raise BootstrapError("frozen GPUWM checkout lacks required certify sources")
    if any(name in sys.modules for name in ("woof", "woof.certify")):
        raise BootstrapError("GPUWM modules existed before namespace installation")
    sys.modules["woof"] = _namespace("woof", package)
    sys.modules["woof.certify"] = _namespace("woof.certify", certify)


def _parse_control(argv: Sequence[str]) -> tuple[dict[str, str], list[Path], list[str]]:
    arguments = list(argv)
    if arguments.count("--") != 1:
        raise BootstrapError("bootstrap argv must contain one exact '--' separator")
    separator = arguments.index("--")
    control = arguments[:separator]
    child = arguments[separator + 1 :]
    if not child:
        raise BootstrapError("frozen child argv is empty")
    values: dict[str, str] = {}
    package_roots: list[Path] = []
    single = {
        "--capsule-root",
        "--freeze-manifest",
        "--freeze-manifest-sha256",
        "--frozen-tool",
        "--frozen-tool-sha256",
        "--gpuwm-root",
        "--gpuwm-head",
        "--runtime-receipt",
    }
    index = 0
    while index < len(control):
        option = control[index]
        if option not in {*single, "--package-root"} or index + 1 >= len(control):
            raise BootstrapError(f"invalid bootstrap control argv near {option!r}")
        value = control[index + 1]
        if option == "--package-root":
            package_roots.append(
                _real_directory(value, label="post-startup package root")
            )
        else:
            if option in values:
                raise BootstrapError(f"duplicate bootstrap option {option}")
            values[option] = value
        index += 2
    if set(values) != single:
        raise BootstrapError("bootstrap control argv is incomplete")
    if not package_roots:
        raise BootstrapError("bootstrap has no post-startup package root")
    return values, package_roots, child


def _single_child_option(arguments: Sequence[str], option: str) -> str:
    indexes = [index for index, value in enumerate(arguments) if value == option]
    if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
        raise BootstrapError(f"frozen child must have one exact {option} value")
    return arguments[indexes[0] + 1]


def main(argv: Sequence[str] | None = None) -> int:
    _assert_isolated_startup()
    values, package_roots, child_argv = _parse_control(
        sys.argv[1:] if argv is None else argv
    )
    capsule = _real_directory(values["--capsule-root"], label="frozen capsule root")
    expected_runtime_root = (capsule / "runtime").resolve()
    if package_roots != [expected_runtime_root]:
        raise BootstrapError(
            "bootstrap admits only the exact frozen capsule/runtime package root"
        )
    manifest_path = _within(
        capsule,
        _real_file(values["--freeze-manifest"], label="freeze manifest"),
        label="freeze manifest",
    )
    manifest_bytes = manifest_path.read_bytes()
    expected_manifest_sha = values["--freeze-manifest-sha256"]
    if _SHA256_RE.fullmatch(expected_manifest_sha) is None:
        raise BootstrapError("freeze-manifest SHA-256 argument is invalid")
    if _sha256_bytes(manifest_bytes) != expected_manifest_sha:
        raise BootstrapError("freeze manifest changed before child execution")
    manifest = _strict_json(manifest_bytes, label="freeze manifest")
    if set(manifest) != {
        "schema",
        "capsule_files",
        "capsule_files_sha256",
        "gpuwm_git_head",
        "runtime_capsule_entries",
        "runtime_capsule_entries_sha256",
        "runtime_dependency_files_sha256",
        "external_nvidia_driver_files",
        "external_nvidia_driver_files_sha256",
        "allowed_python_runtime_files",
        "allowed_python_runtime_files_sha256",
        "allowed_external_native_files",
        "allowed_external_native_files_sha256",
    } or manifest.get("schema") != FREEZE_SCHEMA:
        raise BootstrapError("freeze manifest schema or inventory changed")
    expected_files = manifest.get("capsule_files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise BootstrapError("freeze manifest has no capsule file inventory")
    if manifest.get("capsule_files_sha256") != _canonical_sha256(expected_files):
        raise BootstrapError("freeze manifest file-inventory digest is false")
    runtime_entries = manifest.get("runtime_capsule_entries")
    if (
        not isinstance(runtime_entries, dict)
        or not runtime_entries
        or manifest.get("runtime_capsule_entries_sha256")
        != _canonical_sha256(runtime_entries)
        or not isinstance(manifest.get("runtime_dependency_files_sha256"), str)
        or _SHA256_RE.fullmatch(manifest["runtime_dependency_files_sha256"]) is None
    ):
        raise BootstrapError("freeze manifest runtime distribution closure is false")
    frozen_runtime_files = {
        relative.removeprefix("runtime/"): record
        for relative, record in expected_files.items()
        if relative.startswith("runtime/")
    }
    if (
        not frozen_runtime_files
        or _canonical_sha256(frozen_runtime_files)
        != manifest["runtime_dependency_files_sha256"]
        or any(relative.split("/", 1)[0] not in runtime_entries for relative in frozen_runtime_files)
        or any(
            not any(
                relative == target or relative.startswith(f"{target}/")
                for relative in frozen_runtime_files
            )
            for target in runtime_entries
        )
    ):
        raise BootstrapError("frozen runtime files differ from the distribution closure")
    python_files = manifest.get("allowed_python_runtime_files")
    external_native = manifest.get("allowed_external_native_files")
    external_driver = manifest.get("external_nvidia_driver_files")
    if (
        not isinstance(external_driver, dict)
        or not external_driver
        or manifest.get("external_nvidia_driver_files_sha256")
        != _canonical_sha256(external_driver)
        or not set(external_driver).issubset(external_native or {})
        or not all(Path(path).name.lower() == "nvcuda.dll" for path in external_driver)
        or not isinstance(python_files, dict)
        or manifest.get("allowed_python_runtime_files_sha256")
        != _canonical_sha256(python_files)
        or not isinstance(external_native, dict)
        or manifest.get("allowed_external_native_files_sha256")
        != _canonical_sha256(external_native)
    ):
        raise BootstrapError("freeze manifest external runtime inventories are false")
    measured_files = _tree_inventory(
        capsule,
        excluded=frozenset({manifest_path.relative_to(capsule).as_posix()}),
    )
    if measured_files != expected_files:
        raise BootstrapError("frozen capsule files differ from the freeze manifest")

    frozen_tool = _within(
        capsule,
        _real_file(values["--frozen-tool"], label="frozen dual-run tool"),
        label="frozen dual-run tool",
    )
    tool_sha = values["--frozen-tool-sha256"]
    if _SHA256_RE.fullmatch(tool_sha) is None or _file_record(frozen_tool)["sha256"] != tool_sha:
        raise BootstrapError("frozen dual-run tool SHA-256 changed")
    gpuwm_root = _within(
        capsule,
        _real_directory(values["--gpuwm-root"], label="frozen GPUWM root"),
        label="frozen GPUWM root",
    )
    expected_head = values["--gpuwm-head"].lower()
    if manifest.get("gpuwm_git_head") != expected_head or _git_head(gpuwm_root) != expected_head:
        raise BootstrapError("frozen GPUWM HEAD differs from the freeze manifest")

    child_cache = Path(
        _single_child_option(child_argv, "--cache-root")
    ).expanduser().resolve()
    runtime_receipt = Path(values["--runtime-receipt"]).expanduser().resolve()
    if (
        runtime_receipt != child_cache / "runtime-closure.json"
        or runtime_receipt == capsule
        or capsule in runtime_receipt.parents
        or runtime_receipt.is_symlink()
        or runtime_receipt.exists()
    ):
        raise BootstrapError(
            "runtime receipt must be the fresh child cache runtime-closure.json"
        )

    for package_root in package_roots:
        value = str(package_root)
        if value not in sys.path:
            sys.path.append(value)
    _install_gpuwm_namespace(gpuwm_root)
    module_origins_pre = _module_origin_inventory()
    native_pre = _native_inventory()
    _validate_runtime_closure(
        capsule=capsule,
        manifest=manifest,
        module_origins=module_origins_pre,
        native_modules=native_pre,
    )

    sys.argv = [str(frozen_tool), *child_argv]
    try:
        runpy.run_path(str(frozen_tool), run_name="__main__")
    except SystemExit as error:
        exit_code = 0 if error.code is None else error.code
        if not isinstance(exit_code, int):
            raise BootstrapError(
                f"frozen tool raised non-integer SystemExit: {exit_code!r}"
            ) from error
        if exit_code != 0:
            raise
    module_origins_post = _module_origin_inventory()
    native_post = _native_inventory()
    _validate_runtime_closure(
        capsule=capsule,
        manifest=manifest,
        module_origins=module_origins_post,
        native_modules=native_post,
    )
    receipt = {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "capsule_files_sha256": manifest["capsule_files_sha256"],
        "runtime_capsule_entries_sha256": manifest[
            "runtime_capsule_entries_sha256"
        ],
        "runtime_dependency_files_sha256": manifest[
            "runtime_dependency_files_sha256"
        ],
        "external_nvidia_driver_files_sha256": manifest[
            "external_nvidia_driver_files_sha256"
        ],
        "allowed_python_runtime_files_sha256": manifest[
            "allowed_python_runtime_files_sha256"
        ],
        "allowed_external_native_files_sha256": manifest[
            "allowed_external_native_files_sha256"
        ],
        "pre": {
            "module_origins": module_origins_pre,
            "module_origins_sha256": _canonical_sha256(module_origins_pre),
            "native_modules": native_pre,
            "native_modules_sha256": _canonical_sha256(native_pre),
        },
        "post": {
            "module_origins": module_origins_post,
            "module_origins_sha256": _canonical_sha256(module_origins_post),
            "native_modules": native_post,
            "native_modules_sha256": _canonical_sha256(native_post),
        },
        "new_module_origins": sorted(set(module_origins_post) - set(module_origins_pre)),
        "new_native_modules": sorted(set(native_post) - set(native_pre)),
        "all_post_run_origins_within_static_frozen_or_external_pins": True,
    }
    if runtime_receipt.is_symlink() or runtime_receipt.exists():
        raise BootstrapError(
            f"runtime receipt target must be absent and not a symlink: {runtime_receipt}"
        )
    _write_json_exclusive(runtime_receipt, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
