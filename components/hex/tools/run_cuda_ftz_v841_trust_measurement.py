#!/usr/bin/env python3
"""Trust-anchored launcher for the promoted v8.4.1 sm_120 FTZ measurement.

This launcher deliberately does not import ``woof.hex`` or the measured
runner.  It pins their bytes first, starts the frozen runner in a child
process with absent output/cache roots, independently validates the completed
measurement, then publishes a separate O_EXCL receipt, manifest, and final
seal. The final seal, not the intermediate manifest or child stdout, is the
promotion trust root.
"""

from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import argparse
import csv
import hashlib
from importlib.machinery import PathFinder
import io
import json
import os
from pathlib import Path
import re
import site
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FROZEN_TOOL = ROOT / "tools" / "run_cuda_ftz_contract.py"
AUTHORITY_PINS = ROOT / "tools" / "cuda_ftz_v841_authority_pins.json"
ISOLATED_BOOTSTRAP = ROOT / "tools" / "cuda_ftz_v841_isolated_bootstrap.py"
FROZEN_VALIDATOR = ROOT / "tools" / "cuda_ftz_v841_binding_validator.py"

# These four pin instruments of this repository, not measurements, and they are
# re-derived by tools/repin_source_tables.py.  TWO OF THEM WERE DEAD BEFORE THE
# 0.2.0 RENAME AND NOBODY KNEW: AUTHORITY_PINS_SHA256 arrived with the base
# import at 8a34759 and was killed by 36802f8, which edited the json it
# digests, and it stayed wrong for weeks because nothing that runs without a
# CUDA card ever reads it -- the whole CPU battery is green with this constant
# naming bytes that exist nowhere.  A card-tier gate whose table only a card
# checks is a gate that fails on the node, at the start of a booked run.
# AUTHORITY_PINS_SHA256 is also a fixed point: the json it digests carries
# run_cuda_ftz_contract.py's own digest, so the two settle in that order and
# the generator iterates until a whole pass changes nothing.
FROZEN_TOOL_SHA256 = "e80baa20312e258ac2a3ba2c1e74b3396e8244810355a1535c6b1a0d26a04e75"
AUTHORITY_PINS_SHA256 = (
    "3c0752e9e28b1e673dbb9b1b01d8befe537203d0c51bf20b610a56476b7fde30"
)
ISOLATED_BOOTSTRAP_SHA256 = (
    "41bc68749b80fa50fee0d07cdaf3c2dff8a46d9c128253c42ea3c9df67693c09"
)
FROZEN_VALIDATOR_SHA256 = (
    "c4e28f621d6892cbc3e773fe6c6ea4b37f69e27e1f20e5c80795411d22e0f1c7"
)

PIN_SCHEMA = "mpas-port.cuda-ftz-v841-authority-pins/v1"
COMPLETION_RECEIPT_SCHEMA = "mpas-port.cuda-ftz-v841-completion-receipt/v1"
COMPLETION_MANIFEST_SCHEMA = "mpas-port.cuda-ftz-v841-completion-manifest/v1"
COMPLETION_SEAL_SCHEMA = "mpas-port.cuda-ftz-v841-completion-seal/v1"
CHILD_CAPSULE_SCHEMA = "mpas-port.cuda-ftz-v841-execution-capsule/v1"
COMPILE_MANIFEST_SCHEMA = "mpas-port.cuda-compile-manifest/v1"
KERNEL_AUDIT_SCHEMA = "mpas-port.cuda-ftz-v841-kernel-audit/v2"
BINDING_SCHEMA = "mpas-port.cuda-ftz-binding-v841/v2"
VALIDATION_CHILD_SCHEMA = "mpas-port.cuda-ftz-v841-isolated-live-replay/v1"
VALIDATION_CACHE_SCHEMA = "mpas-port.cuda-ftz-v841-replay-cache/v1"
KERNEL_AUDIT_MEASUREMENT = (
    "direct-production-kernel-launch-sm120-four-pass-with-disabled-fallback-mutation"
)
AUDIT_PASS_SCHEMA = "mpas-port.cuda-ftz-v841-device-pass/v2"
AUDIT_TRANSCRIPT_SCHEMA = "mpas-port.cuda-ftz-v841-transcript/v2"

# THE 0.2.0 RENAME LEFT A CARD-TIER PIN OWED HERE, AND IT IS NOT RE-DERIVABLE
# FROM THIS TREE.  These keys are NVRTC module_key values, they moved from
# mpas_port.* to hexcore.* with the package, and CompileManifest.snapshot()
# keys its "modules" map by exactly them.  So a compile manifest measured on a
# card today canonicalises to a different digest than the sm_120 measurement
# that minted expected_measurement.compile_manifest_canonical_sha256 /
# _file_sha256 in cuda_ftz_v841_authority_pins.json, and _verify_compile_manifest
# will refuse with "measured compile manifest canonical SHA-256 changed".
# Those two values are a MEASUREMENT of a card, not a statement about this
# tree, so tools/repin_source_tables.py deliberately does not touch them:
# re-deriving a card measurement from source would be a fabrication.  They are
# RE-MEASURED by re-running the ftz contract on an sm_120 card, and until that
# happens this gate is red by construction.  NOT MEASURED by the re-pin lane.
EXPECTED_TRANSLATION_UNITS = {
    "hexcore.cuda_acoustic": 1,
    "hexcore.cuda_acoustic_v841": 6,
    "hexcore.cuda_backend.recovery": 3,
    "hexcore.cuda_driver": 12,
    "hexcore.cuda_dynamics_v841": 9,
    "hexcore.cuda_horizontal": 4,
    "hexcore.cuda_horizontal_v841": 6,
    "hexcore.cuda_transport_v841": 5,
}
EXPECTED_TRANSLATION_UNIT_COUNT = 8
EXPECTED_RESOLVED_KERNEL_COUNT = 46
EXPECTED_AUDIT_KERNEL_COUNT = 95
EXPECTED_DISABLED_RED_COUNT = 78
EXPECTED_PROBE_SPEC_SHA256 = (
    "0d731a445f8fc9d0caf805fdf913efdaf4d7bb1f614340ed2b31ef7318046028"
)
EXPECTED_ENABLED_RECORDS_SHA256 = (
    "5bd8bf400ef4713e09e57d916a2f2ccb11e6ceecf849f8bcd2a554bae61c3115"
)
EXPECTED_DISABLED_RECORDS_SHA256 = (
    "80552194e054223ccc9750594f6e7f6c50bb76d201b504737a936f01566516cc"
)
EXPECTED_KERNEL_KEYS_SHA256 = (
    "448b8bd2cf0a38bc7fa8d0732b312f7e4f109321952d64f721a42c58455fc477"
)
EXPECTED_KERNEL_DISPOSITIONS_SHA256 = (
    "73a1ba73f499b88ebe723030c9e718278fac269079a0ab81ff692ca066120f4d"
)
EXPECTED_GPUWM_RECEIPT_SHA256 = (
    "4ae8b69e4081d28bc472f924251d454ad0b687c021e45f4439a1be4a47a668bc"
)

_GPUWM_SOURCE_PATHS = (
    "tools/ftz_receipt/probe.py",
    "tools/ftz_receipt/route_inventory.py",
    "woof/core/kernels/ftz_probe.cu",
    "woof/certify/compile_platform.py",
)
_GPUWM_STARTUP_CLOSURE_PINS = {
    "woof/certify/compile_platform.py": {
        "bytes": 9388,
        "sha256": "2eff8e2e2826e9c5962e3399af280cd859b06acdb58686cca12ff61c61e560fa",
    },
    "woof/certify/pins.py": {
        "bytes": 13929,
        "sha256": "0f41fd16a76a185b94c2ed640f322ed1c5e83d3ffdd828f251079d9c696e32d6",
    },
    "woof/gpu_stack_identity.py": {
        "bytes": 3156,
        "sha256": "86d9b04dac2586637dfb814f2045e57e98afbcc12326ce52a952e746e86c1df2",
    },
}
_IGNORED_AUTHORITY_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
_WDDM_SYSTEM_SENTINEL_PID = 4
_NVIDIA_SMI_INSUFFICIENT_PERMISSIONS = "[Insufficient Permissions]"


class TrustError(RuntimeError):
    """The measurement cannot be promoted from the available evidence."""


def _require_launcher_isolation() -> None:
    required_flags = {
        "isolated": 1,
        "ignore_environment": 1,
        "no_site": 1,
        "no_user_site": 1,
        "dont_write_bytecode": 1,
        "safe_path": True,
    }
    mismatches = {
        name: (getattr(sys.flags, name, None), expected)
        for name, expected in required_flags.items()
        if getattr(sys.flags, name, None) != expected
    }
    if mismatches:
        raise TrustError(
            f"trust launcher must itself run as 'python -I -S -B': {mismatches}"
        )
    loaded_hooks = sorted(
        name for name in ("sitecustomize", "usercustomize") if name in sys.modules
    )
    if loaded_hooks:
        raise TrustError(
            f"trust launcher startup customization already ran: {loaded_hooks}"
        )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    selected = path.expanduser()
    if selected.is_symlink() or not selected.is_file():
        raise TrustError(f"authority input is not a real regular file: {selected}")
    before = selected.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = selected.stat(follow_symlinks=False)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or size != after.st_size:
        raise TrustError(f"authority input changed while hashing: {selected}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _tree_inventory(
    root: Path,
    *,
    ignored_parts: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    raw = root.expanduser()
    if raw.is_symlink():
        raise TrustError(f"inventory root must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_dir():
        raise TrustError(f"inventory root is not a directory: {selected}")
    result: dict[str, dict[str, Any]] = {}
    for candidate in sorted(selected.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(selected)
        if candidate.is_symlink():
            raise TrustError(f"inventory contains a symlink: {candidate}")
        if any(part in ignored_parts for part in relative.parts):
            continue
        if candidate.is_file():
            result[relative.as_posix()] = _file_record(candidate)
    return result


def _inventory_record(
    root: Path,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in sorted(files.items())}
    return {
        "root": str(root.expanduser().resolve()),
        "file_count": len(normalized),
        "files": normalized,
        "files_sha256": _canonical_sha256(normalized),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrustError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise TrustError(f"JSON contains non-finite numeric token {value!r}")


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrustError(
            f"{label} is not one strict UTF-8 JSON document: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TrustError(f"{label} JSON root is not an object")
    return value


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise TrustError(f"cannot read {label} {path}: {error}") from error
    return _strict_json_bytes(payload, label=label), payload


def _parse_exact_child_stdout(payload: bytes) -> dict[str, Any]:
    value = _strict_json_bytes(payload, label="child stdout")
    canonical_lf = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    canonical_crlf = canonical_lf.replace(b"\n", b"\r\n")
    if payload not in {canonical_lf, canonical_crlf}:
        raise TrustError(
            "child stdout must contain exactly one canonical pretty-printed JSON "
            "summary and one terminal LF or CRLF newline, with no noise"
        )
    return value


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
        raise TrustError(f"cannot resolve woof HEAD at {selected}: {error}") from error
    head = completed.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise TrustError(f"woof HEAD is not a full commit id: {head!r}")
    return head


def _mpas_authority_inventory() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative_root in ("src", "tests", "tools"):
        for relative, record in _tree_inventory(
            ROOT / relative_root,
            ignored_parts=_IGNORED_AUTHORITY_PARTS,
        ).items():
            files[f"{relative_root}/{relative}"] = record
    return _inventory_record(ROOT, files)


def _gpuwm_source_inventory(root: Path) -> dict[str, Any]:
    raw = root.expanduser()
    if raw.is_symlink():
        raise TrustError(f"woof root must not be a symlink: {raw}")
    selected = raw.resolve()
    files = {
        relative: _file_record(selected / relative) for relative in _GPUWM_SOURCE_PATHS
    }
    record = _inventory_record(selected, files)
    record["git_head"] = _git_head(selected)
    return record


def _gpuwm_startup_closure_record(root: Path) -> dict[str, Any]:
    raw = root.expanduser()
    if raw.is_symlink():
        raise TrustError(f"woof root must not be a symlink: {raw}")
    selected = raw.resolve()
    files = {
        relative: _file_record(selected / relative)
        for relative in _GPUWM_STARTUP_CLOSURE_PINS
    }
    if files != _GPUWM_STARTUP_CLOSURE_PINS:
        raise TrustError("transitive GPUWM Python startup closure changed")
    return _inventory_record(selected, files)


def _authority_snapshot(gpuwm_root: Path, gpuwm_receipt: Path) -> dict[str, Any]:
    receipt = gpuwm_receipt.expanduser().resolve()
    return {
        "mpas_authority": _mpas_authority_inventory(),
        "gpuwm_sources": _gpuwm_source_inventory(gpuwm_root),
        "gpuwm_receipt": _inventory_record(receipt, _tree_inventory(receipt)),
    }


def _assert_same_authority(
    expected: Mapping[str, Any],
    measured: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    if dict(measured) != dict(expected):
        raise TrustError(
            f"authority drift at {phase}: {_canonical_sha256(expected)} != "
            f"{_canonical_sha256(measured)}"
        )


def _load_authority_pins() -> dict[str, Any]:
    record = _file_record(AUTHORITY_PINS)
    if record["sha256"] != AUTHORITY_PINS_SHA256:
        raise TrustError(
            "static authority pin document changed: "
            f"{record['sha256']} != {AUTHORITY_PINS_SHA256}"
        )
    pins, _ = _load_json(AUTHORITY_PINS, label="authority pin document")
    if pins.get("schema") != PIN_SCHEMA:
        raise TrustError("static authority pin schema is invalid")
    return pins


def _isolated_bootstrap_record() -> dict[str, Any]:
    record = _file_record(ISOLATED_BOOTSTRAP)
    if record["sha256"] != ISOLATED_BOOTSTRAP_SHA256:
        raise TrustError(
            "isolated child bootstrap changed: "
            f"{record['sha256']} != {ISOLATED_BOOTSTRAP_SHA256}"
        )
    return {"path": str(ISOLATED_BOOTSTRAP.resolve()), **record}


def _frozen_validator_record() -> dict[str, Any]:
    record = _file_record(FROZEN_VALIDATOR)
    if record["sha256"] != FROZEN_VALIDATOR_SHA256:
        raise TrustError(
            "isolated live-replay validator changed: "
            f"{record['sha256']} != {FROZEN_VALIDATOR_SHA256}"
        )
    return {"path": str(FROZEN_VALIDATOR.resolve()), **record}


def _runtime_package_roots() -> tuple[Path, ...]:
    python_environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper().startswith("PYTHON")
    }
    previous_user_base = site.USER_BASE
    previous_user_site = site.USER_SITE
    try:
        for key in python_environment:
            os.environ.pop(key, None)
        site.USER_BASE = None
        site.USER_SITE = None
        candidates = [*site.getsitepackages(), site.getusersitepackages()]
    finally:
        site.USER_BASE = previous_user_base
        site.USER_SITE = previous_user_site
        os.environ.update(python_environment)
    roots: list[Path] = []
    for value in candidates:
        raw = Path(value).expanduser()
        if raw.is_symlink():
            raise TrustError(f"runtime package root must not be a symlink: {raw}")
        selected = raw.resolve()
        if selected.is_dir() and selected not in roots:
            roots.append(selected)
    if not roots:
        raise TrustError("no real post-startup Python package root is available")
    for package in ("cupy", "numpy"):
        spec = PathFinder.find_spec(package, [str(root) for root in roots])
        if spec is None or spec.origin is None:
            raise TrustError(f"required runtime package {package!r} is unavailable")
        origin = Path(spec.origin).resolve()
        if not any(origin == root or root in origin.parents for root in roots):
            raise TrustError(
                f"required runtime package {package!r} is outside pinned roots: {origin}"
            )
    return tuple(roots)


def _validate_pre_pins(
    pins: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "frozen_tool",
        "mpas_core_tests",
        "gpuwm_sources",
        "gpuwm_receipt",
        "expected_measurement",
    }
    if set(pins) != expected_keys:
        raise TrustError("static authority pin document fields changed")
    tool_record = _file_record(FROZEN_TOOL)
    if tool_record["sha256"] != FROZEN_TOOL_SHA256:
        raise TrustError(
            f"frozen measured tool changed: {tool_record['sha256']} != "
            f"{FROZEN_TOOL_SHA256}"
        )
    if pins["frozen_tool"] != {
        "path": "tools/run_cuda_ftz_contract.py",
        **tool_record,
    }:
        raise TrustError("frozen tool record disagrees with static authority pins")

    mpas_files = snapshot["mpas_authority"]["files"]
    core_tests = {
        key: value
        for key, value in mpas_files.items()
        if key.startswith("src/") or key.startswith("tests/")
    }
    measured_core_tests = {
        "file_count": len(core_tests),
        "files_sha256": _canonical_sha256(core_tests),
    }
    if pins["mpas_core_tests"] != measured_core_tests:
        raise TrustError("MPAS core/tests bytes disagree with static authority pins")
    # Everything EXCEPT `root`, which is an absolute filesystem path and was
    # never part of what this gate means.  Comparing whole dicts made the
    # check machine-bound: it could only ever pass on the one machine whose
    # woof checkout sat at the absolute path recorded when the pins were
    # written, and it failed on every other machine with a message blaming
    # the SOURCE BYTES -- which were identical.  A gate that reports "your
    # bytes disagree" when the bytes agree is worse than no gate, because it
    # trains a reader to ignore it.
    #
    # The projection is the one its sibling two lines above already uses.
    # Strictness is unchanged where it counts: git_head, file_count, the
    # per-file records and files_sha256 are all still compared exactly, so a
    # single flipped source byte still refuses.
    def _without_root(record: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key != "root"}

    if _without_root(pins["gpuwm_sources"]) != _without_root(
        snapshot["gpuwm_sources"]
    ):
        raise TrustError("woof HEAD/source bytes disagree with static authority pins")
    expected_receipt = pins["gpuwm_receipt"]
    measured_receipt = snapshot["gpuwm_receipt"]
    receipt_projection = {
        "file_count": measured_receipt["file_count"],
        "files_sha256": measured_receipt["files_sha256"],
        "receipt.json": measured_receipt["files"].get("receipt.json"),
        "bitpatterns.csv": measured_receipt["files"].get("bitpatterns.csv"),
    }
    if expected_receipt != receipt_projection:
        raise TrustError("woof FTZ receipt bytes disagree with static authority pins")
    receipt_json = receipt_projection.get("receipt.json")
    if (
        not isinstance(receipt_json, Mapping)
        or receipt_json.get("sha256") != EXPECTED_GPUWM_RECEIPT_SHA256
    ):
        raise TrustError("woof FTZ receipt.json is not the adjudicated receipt")
    _expected_measurement(pins)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _resolve_absent_target(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise TrustError(f"{label} must not be a symlink: {raw}")
    selected = raw.resolve()
    if selected.exists():
        raise TrustError(f"{label} must be absent before launch: {selected}")
    return selected


def _validate_targets(
    *,
    output_root: Path,
    cache_root: Path,
    validation_cache_root: Path,
    completion_root: Path,
    gpuwm_root: Path,
    gpuwm_receipt: Path,
) -> tuple[Path, Path, Path, Path]:
    output = _resolve_absent_target(output_root, label="measured output root")
    cache = _resolve_absent_target(cache_root, label="measured cache root")
    validation_cache = _resolve_absent_target(
        validation_cache_root, label="live-replay validation cache root"
    )
    completion = _resolve_absent_target(
        completion_root, label="completion receipt root"
    )
    targets = {
        "measured output root": output,
        "measured cache root": cache,
        "live-replay validation cache root": validation_cache,
        "completion receipt root": completion,
    }
    for left_name, left in targets.items():
        for right_name, right in targets.items():
            if left_name < right_name and _paths_overlap(left, right):
                raise TrustError(f"{left_name} overlaps {right_name}")
    protected = (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "tools",
        gpuwm_root.expanduser().resolve(),
        gpuwm_receipt.expanduser().resolve(),
    )
    for label, target in targets.items():
        for source in protected:
            if _paths_overlap(target, source.resolve()):
                raise TrustError(f"{label} overlaps authority input {source.resolve()}")
    return output, cache, validation_cache, completion


def _nvidia_smi_executable() -> Path:
    selected = shutil.which("nvidia-smi")
    if selected is None:
        raise TrustError("nvidia-smi is required for launcher GPU identity checks")
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
    gpu_rows = _csv_rows(_run_nvidia_smi(executable, "gpu", gpu_fields))
    gpus: list[dict[str, Any]] = []
    for row in gpu_rows:
        if len(row) != len(gpu_fields):
            raise TrustError(f"nvidia-smi GPU row is malformed: {row!r}")
        try:
            index = int(row[0])
        except ValueError as error:
            raise TrustError(f"nvidia-smi GPU index is invalid: {row[0]!r}") from error
        if _GPU_UUID_RE.fullmatch(row[1]) is None:
            raise TrustError(f"nvidia-smi GPU UUID is invalid: {row[1]!r}")
        if re.fullmatch(r"\d+\.\d+", row[4]) is None:
            raise TrustError("nvidia-smi compute capability is unavailable")
        if any(value in {"", "N/A", "[Not Supported]"} for value in row[2:]):
            raise TrustError(f"nvidia-smi GPU identity is incomplete: {row!r}")
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
        raise TrustError("nvidia-smi GPU inventory is empty or has duplicate indices")

    process_fields = ("gpu_uuid", "pid", "process_name")
    process_rows = _csv_rows(
        _run_nvidia_smi(executable, "compute-apps", process_fields)
    )
    if len(process_rows) == 1 and process_rows[0][0].lower().startswith(
        "no running processes"
    ):
        process_rows = []
    processes: list[dict[str, Any]] = []
    known_uuids = {str(row["uuid"]) for row in gpus}
    for row in process_rows:
        if len(row) != len(process_fields):
            raise TrustError(f"nvidia-smi process row is malformed: {row!r}")
        if row[0] not in known_uuids:
            raise TrustError(f"nvidia-smi process has unknown GPU UUID: {row[0]!r}")
        try:
            pid = int(row[1])
        except ValueError as error:
            raise TrustError(
                f"nvidia-smi process PID is invalid: {row[1]!r}"
            ) from error
        if not row[2]:
            raise TrustError("nvidia-smi process name is empty")
        if (
            sys.platform == "win32"
            and pid == _WDDM_SYSTEM_SENTINEL_PID
            and row[2] == _NVIDIA_SMI_INSUFFICIENT_PERMISSIONS
        ):
            # Match the frozen child: ignore only nvidia-smi's unstable WDDM
            # System PID-4 sentinel, never another PID/UUID/name row.
            continue
        processes.append({"gpu_uuid": row[0], "pid": pid, "process_name": row[2]})
    processes.sort(key=lambda row: (str(row["gpu_uuid"]), int(row["pid"])))
    return {
        "gpu_inventory": gpus,
        "reported_processes": processes,
        "wddm_system_pid4_insufficient_permissions_sentinel_normalized": True,
        "physical_gpu_exclusivity_claim": False,
        "wddm_interpretation": (
            "nvidia-smi can report stable C+G desktop clients on Windows/WDDM; "
            "the launcher requires an unchanged normalized PID/UUID/name baseline, "
            "excluding only the unstable System PID-4 '[Insufficient Permissions]' "
            "telemetry sentinel, and does not claim physical GPU exclusivity"
        ),
    }


def _invoke_child(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.upper().startswith("PYTHON"):
            environment.pop(key, None)
    environment.pop("CUPY_CACHE_DIR", None)
    environment.pop("MPAS_PORT_CUDA_CACHE_DIR", None)
    return subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=False,
        timeout=timeout_seconds,
        check=False,
    )


def _expected_measurement(pins: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = pins.get("expected_measurement")
    if not isinstance(expected, Mapping):
        raise TrustError("static expected measurement pins are absent")
    required = {
        "compile_manifest_canonical_sha256",
        "compile_manifest_file_sha256",
        "translation_unit_count",
        "resolved_kernel_count",
        "audit_kernel_count",
        "disabled_red_count",
        "translation_units",
        "compile_manifest_pin_provenance",
    }
    if set(expected) != required:
        raise TrustError("static expected measurement fields changed")
    exact_counts = {
        "translation_unit_count": EXPECTED_TRANSLATION_UNIT_COUNT,
        "resolved_kernel_count": EXPECTED_RESOLVED_KERNEL_COUNT,
        "audit_kernel_count": EXPECTED_AUDIT_KERNEL_COUNT,
        "disabled_red_count": EXPECTED_DISABLED_RED_COUNT,
    }
    for key, value in exact_counts.items():
        _require_int(expected.get(key), value, label=f"static {key}")
    if expected.get("translation_units") != EXPECTED_TRANSLATION_UNITS:
        raise TrustError("static translation-unit map changed")
    provenance = expected.get("compile_manifest_pin_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "evidence_kind",
        "source_path_at_measurement",
        "measured_file_sha256",
        "measured_canonical_sha256",
    }:
        raise TrustError("compile manifest pin provenance is incomplete")
    if provenance.get("evidence_kind") != "preexisting-work-only-real-sm120-manifest":
        raise TrustError("compile manifest pin has no real sm_120 provenance")
    if provenance.get("measured_file_sha256") != expected.get(
        "compile_manifest_file_sha256"
    ) or provenance.get("measured_canonical_sha256") != expected.get(
        "compile_manifest_canonical_sha256"
    ):
        raise TrustError("compile manifest pin provenance digests are inconsistent")
    return expected


def _require_int(value: Any, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        raise TrustError(f"{label} {value!r} != exact expected integer {expected}")


def _validate_compile_manifest(
    manifest: Mapping[str, Any],
    *,
    file_record: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if manifest.get("schema") != COMPILE_MANIFEST_SCHEMA:
        raise TrustError("measured compile manifest schema is invalid")
    if file_record.get("sha256") != expected["compile_manifest_file_sha256"]:
        raise TrustError("measured compile-manifest.json bytes changed")
    if _canonical_sha256(manifest) != expected["compile_manifest_canonical_sha256"]:
        raise TrustError("measured compile manifest canonical SHA-256 changed")
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or set(modules) != set(
        EXPECTED_TRANSLATION_UNITS
    ):
        raise TrustError("compile manifest does not contain the exact eight TUs")
    counts: dict[str, int] = {}
    for key, expected_count in EXPECTED_TRANSLATION_UNITS.items():
        row = modules[key]
        if not isinstance(row, Mapping):
            raise TrustError(f"compile manifest TU {key!r} is invalid")
        kernels = row.get("resolved_kernels")
        if not isinstance(kernels, list) or len(kernels) != expected_count:
            raise TrustError(f"compile manifest TU {key!r} kernel count changed")
        counts[key] = len(kernels)
    _require_int(len(counts), EXPECTED_TRANSLATION_UNIT_COUNT, label="TU count")
    _require_int(
        sum(counts.values()),
        EXPECTED_RESOLVED_KERNEL_COUNT,
        label="resolved kernel count",
    )
    if counts != dict(expected["translation_units"]):
        raise TrustError("compile manifest TU count map differs from static pins")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TrustError(f"{label} is not one lowercase SHA-256 digest")
    return value


def _validate_compile_relation(
    relation: Mapping[str, Any],
    *,
    compile_manifest: Mapping[str, Any],
    compile_manifest_sha256: str,
) -> set[str]:
    expected_fields = {
        "source_release",
        "compile_manifest_sha256",
        "compile_platform",
        "translation_units",
        "reached_kernel_count",
        "compiled_kernel_count",
        "authority_claim",
    }
    if set(relation) != expected_fields:
        raise TrustError("binding compile relation field set changed")
    if (
        relation.get("source_release") != "v8.4.1"
        or relation.get("compile_manifest_sha256") != compile_manifest_sha256
        or relation.get("compile_platform") != compile_manifest.get("compile_platform")
        or relation.get("authority_claim") is not False
    ):
        raise TrustError("binding compile relation header is false")
    _require_int(
        relation.get("reached_kernel_count"),
        EXPECTED_RESOLVED_KERNEL_COUNT,
        label="binding reached kernel count",
    )
    _require_int(
        relation.get("compiled_kernel_count"),
        EXPECTED_AUDIT_KERNEL_COUNT,
        label="binding compiled kernel count",
    )
    units = relation.get("translation_units")
    modules = compile_manifest.get("modules")
    if (
        not isinstance(units, Mapping)
        or set(units) != set(EXPECTED_TRANSLATION_UNITS)
        or not isinstance(modules, Mapping)
    ):
        raise TrustError("binding compile relation does not contain exact eight TUs")
    unit_fields = {
        "source_sha256",
        "module_cache_key",
        "resolved_kernels",
        "compiled_kernel_surface",
        "compiled_image",
        "effective_terminal_ftz",
        "compile_platform_fingerprint_sha256",
    }
    platform = relation.get("compile_platform")
    if not isinstance(platform, Mapping):
        raise TrustError("binding compile platform is absent")
    platform_sha = _require_sha256(
        platform.get("sha256"), label="binding compile-platform digest"
    )
    compiled_keys: set[str] = set()
    for module_key, reached_count in EXPECTED_TRANSLATION_UNITS.items():
        row = units[module_key]
        manifest_row = modules[module_key]
        if (
            not isinstance(row, Mapping)
            or set(row) != unit_fields
            or not isinstance(manifest_row, Mapping)
        ):
            raise TrustError(f"binding compile relation TU {module_key!r} is incomplete")
        reached = row.get("resolved_kernels")
        compiled = row.get("compiled_kernel_surface")
        effective = manifest_row.get("effective_compile")
        observations = effective.get("observations") if isinstance(effective, Mapping) else None
        authority_image = (
            observations[0].get("compiled_image")
            if isinstance(observations, list)
            and len(observations) == 1
            and isinstance(observations[0], Mapping)
            else None
        )
        if (
            row.get("source_sha256") != manifest_row.get("source_sha256")
            or row.get("module_cache_key") != manifest_row.get("module_cache_key")
            or row.get("compile_platform_fingerprint_sha256") != platform_sha
            or row.get("effective_terminal_ftz") != "-ftz=true"
            or row.get("compiled_image") != authority_image
            or reached != manifest_row.get("resolved_kernels")
            or not isinstance(reached, list)
            or len(reached) != reached_count
            or not isinstance(compiled, list)
            or not compiled
            or any(not isinstance(name, str) or not name for name in compiled)
            or len(set(compiled)) != len(compiled)
        ):
            raise TrustError(f"binding compile relation TU {module_key!r} is false")
        for kernel in compiled:
            key = f"{module_key}::{kernel}"
            if key in compiled_keys:
                raise TrustError(f"duplicate compiled kernel identity {key!r}")
            compiled_keys.add(key)
    if len(compiled_keys) != EXPECTED_AUDIT_KERNEL_COUNT:
        raise TrustError("binding relation does not expose exact 95-kernel surface")
    return compiled_keys


def _validate_pass_translation_units(
    translation_units: Any,
    *,
    mode: str,
    relation: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> None:
    relation_units = relation["translation_units"]
    if not isinstance(translation_units, Mapping) or set(translation_units) != set(
        relation_units
    ):
        raise TrustError(f"{mode} pass does not contain the exact eight TUs")
    module_fields = {
        "source_sha256",
        "requested_options",
        "compile_platform_fingerprint_sha256",
        "module_cache_key",
        "effective_compile",
        "resolved_kernels",
    }
    observation_fields = {
        "source_sha256",
        "effective_flags",
        "include_path_count",
        "include_paths_omitted",
        "compiled_image",
    }
    method = (
        "wrapped cupy.cuda.compiler._compile_using_nvrtc_no_warning at the "
        "NVRTC entry point"
    )
    source_field = (
        "fallback_enabled_source_sha256"
        if mode == "fallback-enabled"
        else "fallback_disabled_source_sha256"
    )
    platform_sha = relation["compile_platform"]["sha256"]
    for module_key, relation_row in relation_units.items():
        row = translation_units[module_key]
        binding = source_bindings[module_key]
        source_sha = binding[source_field]
        effective = row.get("effective_compile") if isinstance(row, Mapping) else None
        observations = effective.get("observations") if isinstance(effective, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != module_fields
            or row.get("source_sha256") != source_sha
            or row.get("requested_options") != ["--std=c++17", "--fmad=false"]
            or row.get("compile_platform_fingerprint_sha256") != platform_sha
            or _SHA256_RE.fullmatch(str(row.get("module_cache_key", ""))) is None
            or row.get("resolved_kernels") != relation_row["compiled_kernel_surface"]
            or not isinstance(effective, Mapping)
            or set(effective) != {"status", "method", "observations"}
            or effective.get("status") != "resolved"
            or effective.get("method") != method
            or not isinstance(observations, list)
            or len(observations) != 1
        ):
            raise TrustError(f"{mode} compile evidence for {module_key!r} is false")
        observation = observations[0]
        image = observation.get("compiled_image") if isinstance(observation, Mapping) else None
        if (
            not isinstance(observation, Mapping)
            or set(observation) != observation_fields
            or observation.get("source_sha256") != source_sha
            or observation.get("effective_flags")
            != ["--std=c++17", "--fmad=false", "-ftz=true"]
            or type(observation.get("include_path_count")) is not int
            or observation.get("include_path_count", -1) < 0
            or observation.get("include_paths_omitted")
            != (
                "-I entries describe this machine's toolkit layout and are "
                "counted but not copied into the arithmetic contract"
            )
            or not isinstance(image, Mapping)
            or set(image) != {"status", "kind", "sha256"}
            or image.get("status") != "resolved"
            or image.get("kind") not in {"cubin", "ptx"}
            or _SHA256_RE.fullmatch(str(image.get("sha256", ""))) is None
        ):
            raise TrustError(f"{mode} compile observation for {module_key!r} is false")
        if mode == "fallback-enabled" and image != relation_row["compiled_image"]:
            raise TrustError(
                f"enabled executable image for {module_key!r} differs from authority"
            )


def _validate_audit(
    audit: Mapping[str, Any],
    *,
    compile_manifest: Mapping[str, Any],
    relation: Mapping[str, Any],
    compiled_keys: set[str],
    runner_source_sha256: str,
) -> None:
    expected_top = {
        "schema",
        "source_release",
        "measurement",
        "device_compute_capability",
        "compile_manifest_sha256",
        "compile_platform_fingerprint_sha256",
        "fallback_verified",
        "dual_run_byte_identical",
        "kernel_count",
        "kernels",
        "measurement_transcript",
        "authority_claim",
    }
    compile_sha = _canonical_sha256(compile_manifest)
    platform_sha = relation["compile_platform"]["sha256"]
    if set(audit) != expected_top:
        raise TrustError("measured kernel audit field set changed")
    if (
        audit.get("schema") != KERNEL_AUDIT_SCHEMA
        or audit.get("source_release") != "v8.4.1"
        or audit.get("measurement") != KERNEL_AUDIT_MEASUREMENT
        or audit.get("device_compute_capability") != "120"
        or audit.get("compile_manifest_sha256") != compile_sha
        or audit.get("compile_platform_fingerprint_sha256") != platform_sha
        or audit.get("fallback_verified") is not True
        or audit.get("dual_run_byte_identical") is not True
        or audit.get("authority_claim") is not False
    ):
        raise TrustError("measured kernel audit header is false")
    _require_int(
        audit.get("kernel_count"),
        EXPECTED_AUDIT_KERNEL_COUNT,
        label="audit kernel_count",
    )
    kernels = audit.get("kernels")
    if not isinstance(kernels, Mapping) or set(kernels) != compiled_keys:
        raise TrustError("kernel audit does not cover the exact compiled surface")

    transcript = audit.get("measurement_transcript")
    transcript_fields = {
        "schema",
        "runner_source_sha256",
        "compile_manifest_sha256",
        "compile_platform_fingerprint_sha256",
        "source_bindings",
        "source_binding_sha256",
        "probe_spec_sha256",
        "runtime",
        "runtime_sha256",
        "enabled_passes",
        "disabled_fallback_passes",
        "enabled_records_sha256",
        "disabled_fallback_records_sha256",
        "transcript_sha256",
    }
    if not isinstance(transcript, Mapping) or set(transcript) != transcript_fields:
        raise TrustError("kernel audit measurement transcript is incomplete")
    if (
        transcript.get("schema") != AUDIT_TRANSCRIPT_SCHEMA
        or transcript.get("runner_source_sha256") != runner_source_sha256
        or transcript.get("compile_manifest_sha256") != compile_sha
        or transcript.get("compile_platform_fingerprint_sha256") != platform_sha
    ):
        raise TrustError("kernel audit transcript source/manifest binding is false")

    source_binding_fields = {
        "production_source_sha256",
        "fallback_enabled_source_sha256",
        "fallback_disabled_source_sha256",
        "reached_kernels",
        "compiled_kernels",
    }
    source_bindings = transcript.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != set(
        relation["translation_units"]
    ):
        raise TrustError("kernel audit source binding lacks exact eight TUs")
    for module_key, relation_row in relation["translation_units"].items():
        row = source_bindings[module_key]
        if (
            not isinstance(row, Mapping)
            or set(row) != source_binding_fields
            or row.get("production_source_sha256") != relation_row["source_sha256"]
            or row.get("fallback_enabled_source_sha256")
            != relation_row["source_sha256"]
            or _SHA256_RE.fullmatch(
                str(row.get("fallback_disabled_source_sha256", ""))
            )
            is None
            or row.get("reached_kernels") != relation_row["resolved_kernels"]
            or row.get("compiled_kernels")
            != relation_row["compiled_kernel_surface"]
        ):
            raise TrustError(f"kernel audit source binding for {module_key!r} is false")
    if transcript.get("source_binding_sha256") != _canonical_sha256(source_bindings):
        raise TrustError("kernel audit source-binding digest is false")

    runtime_fields = {
        "device_id",
        "name",
        "compute_capability",
        "sm",
        "total_memory_bytes",
        "multiprocessor_count",
        "runtime_version",
        "driver_version",
        "nvrtc_version",
        "cupy_version",
    }
    runtime = transcript.get("runtime")
    fingerprint = relation["compile_platform"]["fingerprint"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != runtime_fields
        or runtime.get("compute_capability") != "12.0"
        or runtime.get("sm") != "sm_120"
        or str(runtime.get("driver_version"))
        != fingerprint.get("cuda_driver_version")
        or runtime.get("cupy_version") != fingerprint.get("cupy_version")
        or transcript.get("runtime_sha256") != _canonical_sha256(runtime)
    ):
        raise TrustError("kernel audit runtime/platform binding is false")
    nvrtc_match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", str(fingerprint.get("nvrtc_build", "")))
    nvrtc_pair = runtime.get("nvrtc_version")
    if (
        nvrtc_match is None
        or not isinstance(nvrtc_pair, list)
        or len(nvrtc_pair) != 2
        or any(type(value) is not int for value in nvrtc_pair)
        or nvrtc_pair
        != [int(nvrtc_match.group(1)), int(nvrtc_match.group(2))]
    ):
        raise TrustError("kernel audit NVRTC major/minor binding is false")

    raw_fields = {
        "translation_unit",
        "kernel",
        "classification",
        "lane",
        "expected_bits",
        "observed_bits",
        "matches_expected",
    }
    pass_fields = {
        "schema",
        "mode",
        "ordinal",
        "compile_manifest_sha256",
        "compile_platform_fingerprint_sha256",
        "source_binding_sha256",
        "runner_source_sha256",
        "runtime",
        "runtime_sha256",
        "translation_units_sha256",
        "translation_units",
        "kernel_count",
        "records_sha256",
        "records",
        "pass_sha256",
    }

    def validate_passes(field: str, mode: str) -> list[Mapping[str, Any]]:
        passes = transcript.get(field)
        if not isinstance(passes, list) or len(passes) != 2:
            raise TrustError(f"kernel audit must contain two exact {mode} passes")
        validated: list[Mapping[str, Any]] = []
        for ordinal, one_pass in enumerate(passes, 1):
            if not isinstance(one_pass, Mapping) or set(one_pass) != pass_fields:
                raise TrustError(f"kernel audit {mode} pass {ordinal} is incomplete")
            pass_core = dict(one_pass)
            pass_sha = pass_core.pop("pass_sha256")
            records = one_pass.get("records")
            translation_units = one_pass.get("translation_units")
            if (
                one_pass.get("schema") != AUDIT_PASS_SCHEMA
                or one_pass.get("mode") != mode
                or one_pass.get("ordinal") != ordinal
                or one_pass.get("compile_manifest_sha256") != compile_sha
                or one_pass.get("compile_platform_fingerprint_sha256")
                != platform_sha
                or one_pass.get("source_binding_sha256")
                != transcript["source_binding_sha256"]
                or one_pass.get("runner_source_sha256") != runner_source_sha256
                or one_pass.get("runtime") != runtime
                or one_pass.get("runtime_sha256") != transcript["runtime_sha256"]
                or not isinstance(translation_units, Mapping)
                or one_pass.get("translation_units_sha256")
                != _canonical_sha256(translation_units)
                or one_pass.get("kernel_count") != EXPECTED_AUDIT_KERNEL_COUNT
                or not isinstance(records, Mapping)
                or set(records) != compiled_keys
                or one_pass.get("records_sha256") != _canonical_sha256(records)
                or pass_sha != _canonical_sha256(pass_core)
            ):
                raise TrustError(f"kernel audit {mode} pass {ordinal} is false")
            _validate_pass_translation_units(
                translation_units,
                mode=mode,
                relation=relation,
                source_bindings=source_bindings,
            )
            for key, row in records.items():
                if not isinstance(row, Mapping) or set(row) != raw_fields:
                    raise TrustError(f"kernel audit raw row {key!r} is incomplete")
                module_key, separator, kernel = key.partition("::")
                expected_bits = row.get("expected_bits")
                observed_bits = row.get("observed_bits")
                matches = observed_bits == expected_bits
                if (
                    not separator
                    or row.get("translation_unit") != module_key
                    or row.get("kernel") != kernel
                    or row.get("classification")
                    not in {"guarded_fallback_required", "fallback_invariant"}
                    or not isinstance(row.get("lane"), str)
                    or not row.get("lane", "").strip()
                    or not isinstance(expected_bits, Mapping)
                    or not expected_bits
                    or not isinstance(observed_bits, Mapping)
                    or set(observed_bits) != set(expected_bits)
                    or row.get("matches_expected") is not matches
                ):
                    raise TrustError(f"kernel audit raw row {key!r} is false")
            validated.append(one_pass)
        if validated[0]["records"] != validated[1]["records"]:
            raise TrustError(f"kernel audit {mode} device passes diverged")
        if validated[0]["translation_units"] != validated[1]["translation_units"]:
            raise TrustError(f"kernel audit {mode} compiled images diverged")
        return validated

    enabled = validate_passes("enabled_passes", "fallback-enabled")
    disabled = validate_passes("disabled_fallback_passes", "fallback-disabled")
    enabled_records = enabled[0]["records"]
    disabled_records = disabled[0]["records"]
    enabled_sha = _canonical_sha256(enabled_records)
    disabled_sha = _canonical_sha256(disabled_records)
    if (
        transcript.get("enabled_records_sha256") != enabled_sha
        or transcript.get("disabled_fallback_records_sha256") != disabled_sha
        or enabled_sha != EXPECTED_ENABLED_RECORDS_SHA256
        or disabled_sha != EXPECTED_DISABLED_RECORDS_SHA256
    ):
        raise TrustError("kernel audit raw outcome anchors are false")

    row_fields = {
        "translation_unit",
        "kernel",
        "compiled_source_sha256",
        "reached_by_admitted_step",
        "classification",
        "lane",
        "expected_bits",
        "enabled_observed_bits",
        "disabled_fallback_observed_bits",
        "enabled_matches_expected",
        "disabled_fallback_matches_expected",
        "mutation_red",
    }
    probe_spec: dict[str, Any] = {}
    dispositions: dict[str, Any] = {}
    guarded = 0
    for key in sorted(compiled_keys):
        candidate = enabled_records[key]
        mutation = disabled_records[key]
        if any(
            mutation[field] != candidate[field]
            for field in (
                "translation_unit",
                "kernel",
                "classification",
                "lane",
                "expected_bits",
            )
        ):
            raise TrustError(f"enabled/disabled probe specification differs at {key!r}")
        requires_red = candidate["classification"] == "guarded_fallback_required"
        mutation_red = mutation["matches_expected"] is not True
        guarded += int(requires_red)
        if candidate["matches_expected"] is not True or mutation_red is not requires_red:
            raise TrustError(f"raw mutation disposition is false at {key!r}")
        module_key, kernel = key.split("::", 1)
        summary = {
            "translation_unit": module_key,
            "kernel": kernel,
            "compiled_source_sha256": source_bindings[module_key][
                "production_source_sha256"
            ],
            "reached_by_admitted_step": kernel
            in relation["translation_units"][module_key]["resolved_kernels"],
            "classification": candidate["classification"],
            "lane": candidate["lane"],
            "expected_bits": candidate["expected_bits"],
            "enabled_observed_bits": candidate["observed_bits"],
            "disabled_fallback_observed_bits": mutation["observed_bits"],
            "enabled_matches_expected": True,
            "disabled_fallback_matches_expected": not mutation_red,
            "mutation_red": mutation_red,
        }
        if (
            not isinstance(kernels[key], Mapping)
            or set(kernels[key]) != row_fields
            or kernels[key] != summary
        ):
            raise TrustError(f"kernel summary was not derived from raw row {key!r}")
        probe_spec[key] = {
            field: candidate[field]
            for field in (
                "translation_unit",
                "kernel",
                "classification",
                "lane",
                "expected_bits",
            )
        }
        dispositions[key] = {
            "classification": candidate["classification"],
            "lane": candidate["lane"],
            "mutation_red": mutation_red,
        }
    _require_int(guarded, EXPECTED_DISABLED_RED_COUNT, label="disabled red count")
    if (
        _canonical_sha256(sorted(compiled_keys)) != EXPECTED_KERNEL_KEYS_SHA256
        or _canonical_sha256(dispositions) != EXPECTED_KERNEL_DISPOSITIONS_SHA256
        or transcript.get("probe_spec_sha256") != _canonical_sha256(probe_spec)
        or transcript.get("probe_spec_sha256") != EXPECTED_PROBE_SPEC_SHA256
    ):
        raise TrustError("kernel identities, dispositions, or probe anchor changed")
    transcript_core = dict(transcript)
    transcript_sha = transcript_core.pop("transcript_sha256")
    if transcript_sha != _canonical_sha256(transcript_core):
        raise TrustError("kernel audit transcript digest is false")


def _validate_capsule_authority(
    capsule: Mapping[str, Any],
    *,
    pre_snapshot: Mapping[str, Any],
) -> None:
    if capsule.get("schema") != CHILD_CAPSULE_SCHEMA:
        raise TrustError("child execution capsule schema is invalid")
    runner = capsule.get("runner")
    expected_runner = {
        "path": str(FROZEN_TOOL.resolve()),
        **_file_record(FROZEN_TOOL),
    }
    if runner != expected_runner or runner.get("sha256") != FROZEN_TOOL_SHA256:
        raise TrustError("child execution capsule runner is not the frozen tool")
    authority = capsule.get("authority_inputs")
    if not isinstance(authority, Mapping):
        raise TrustError("child execution capsule authority inventory is missing")
    if authority.get("byte_identical") is not True:
        raise TrustError("child execution capsule reports source drift")
    if authority.get("pre") != pre_snapshot or authority.get("post") != pre_snapshot:
        raise TrustError(
            "child tool authority inventory differs from launcher snapshot"
        )
    expected_digest = _canonical_sha256(pre_snapshot)
    if (
        authority.get("pre_sha256") != expected_digest
        or authority.get("post_sha256") != expected_digest
    ):
        raise TrustError("child authority inventory digest is false")
    gpu = capsule.get("gpu_exclusivity")
    if not isinstance(gpu, Mapping):
        raise TrustError("child GPU process-baseline evidence is absent")
    if gpu.get("physical_gpu_exclusivity_claim") is not False:
        raise TrustError("child made an unsupported physical GPU exclusivity claim")
    if gpu.get("stable_external_process_baseline_enforced") is not True:
        raise TrustError("child did not enforce a stable GPU process baseline")
    if (
        gpu.get(
            "wddm_system_pid4_insufficient_permissions_sentinel_normalized"
        )
        is not True
    ):
        raise TrustError("child did not bind the exact WDDM PID-4 normalization")
    if gpu.get("new_or_drifted_external_process_count") != 0:
        raise TrustError("child observed a new or drifting GPU process")


def _validate_measured_output(
    *,
    output_root: Path,
    cache_root: Path,
    child_summary: Mapping[str, Any],
    pre_snapshot: Mapping[str, Any],
    pins: Mapping[str, Any],
) -> dict[str, Any]:
    if not output_root.is_dir() or output_root.is_symlink():
        raise TrustError("child did not create a real measured output root")
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise TrustError("child did not create a real measured cache root")
    cache_files = _tree_inventory(cache_root)
    if not cache_files:
        raise TrustError("child measured cache root is empty")
    output_files = _tree_inventory(output_root)
    expected_files = {
        "compile-manifest.json",
        "kernel-audit.json",
        "binding.json",
        "execution-capsule.json",
        *(
            f"gpuwm-probe/{relative}"
            for relative in pre_snapshot["gpuwm_receipt"]["files"]
        ),
    }
    if set(output_files) != expected_files:
        raise TrustError("measured output has an extra or missing artifact")

    compile_manifest, _ = _load_json(
        output_root / "compile-manifest.json", label="compile manifest"
    )
    audit, _ = _load_json(output_root / "kernel-audit.json", label="kernel audit")
    binding, _ = _load_json(output_root / "binding.json", label="binding")
    capsule, _ = _load_json(
        output_root / "execution-capsule.json", label="execution capsule"
    )
    expected = _expected_measurement(pins)
    _validate_compile_manifest(
        compile_manifest,
        file_record=output_files["compile-manifest.json"],
        expected=expected,
    )
    compile_sha = str(expected["compile_manifest_canonical_sha256"])
    if binding.get("schema") != BINDING_SCHEMA:
        raise TrustError("measured binding schema is invalid")
    if binding.get("source_release") != "v8.4.1":
        raise TrustError("measured binding source release changed")
    if binding.get("compile_manifest") != compile_manifest:
        raise TrustError("binding compile manifest differs from emitted manifest")
    if binding.get("kernel_audit") != audit:
        raise TrustError("binding kernel audit differs from emitted audit")
    relation = binding.get("compile_relation")
    if not isinstance(relation, Mapping):
        raise TrustError("binding compile relation is missing")
    compiled_keys = _validate_compile_relation(
        relation,
        compile_manifest=compile_manifest,
        compile_manifest_sha256=compile_sha,
    )
    runner_record = pre_snapshot["mpas_authority"]["files"].get(
        "src/hexcore/cuda_ftz_v841.py"
    )
    if not isinstance(runner_record, Mapping):
        raise TrustError("authority snapshot lacks the live FTZ runner source")
    runner_sha = _require_sha256(
        runner_record.get("sha256"), label="live FTZ runner source digest"
    )
    _validate_audit(
        audit,
        compile_manifest=compile_manifest,
        relation=relation,
        compiled_keys=compiled_keys,
        runner_source_sha256=runner_sha,
    )
    probe = binding.get("gpuwm_ftz_probe")
    if (
        not isinstance(probe, Mapping)
        or probe.get("receipt_sha256") != EXPECTED_GPUWM_RECEIPT_SHA256
    ):
        raise TrustError("binding woof FTZ receipt pin changed")
    if binding.get("woof") != {
        "git_head": pins["gpuwm_sources"]["git_head"],
        "sources": {
            {
                "tools/ftz_receipt/probe.py": "generator",
                "tools/ftz_receipt/route_inventory.py": "route_inventory",
                "woof/core/kernels/ftz_probe.cu": "probe_source",
                "woof/certify/compile_platform.py": "compile_platform_source",
            }[path]: {"path": path, "sha256": record["sha256"]}
            for path, record in pins["gpuwm_sources"]["files"].items()
        },
    }:
        raise TrustError("binding woof source pins differ from static authority pins")

    _validate_capsule_authority(capsule, pre_snapshot=pre_snapshot)
    publication = capsule.get("publication")
    if not isinstance(publication, Mapping):
        raise TrustError("child capsule publication record is missing")
    if publication.get("output_root") != str(output_root):
        raise TrustError("child capsule output root differs from requested root")
    if publication.get("output_root_was_absent_and_exclusively_created") is not True:
        raise TrustError("child did not exclusively create its output root")
    if publication.get("all_artifact_writes_used_exclusive_create") is not True:
        raise TrustError("child did not use exclusive artifact writes")
    if publication.get("copied_gpuwm_probe") is not True:
        raise TrustError("child did not copy the pinned woof probe")
    if set(publication.get("expected_files", ())) != expected_files:
        raise TrustError("child capsule expected-file set is false")
    before_capsule_files = {
        key: value
        for key, value in output_files.items()
        if key != "execution-capsule.json"
    }
    bound = publication.get("bound_artifacts_before_capsule")
    if not isinstance(bound, Mapping):
        raise TrustError("child capsule artifact inventory is missing")
    if (
        bound.get("root") != str(output_root)
        or bound.get("file_count") != len(before_capsule_files)
        or bound.get("files") != before_capsule_files
        or bound.get("files_sha256") != _canonical_sha256(before_capsule_files)
    ):
        raise TrustError("child capsule artifact inventory is false")
    runtime = capsule.get("runtime_platform_binding")
    if not isinstance(runtime, Mapping):
        raise TrustError("child capsule runtime/platform binding is missing")
    if runtime.get("kernel_cache_directory") != str(cache_root):
        raise TrustError("child capsule cache path differs from requested fresh cache")
    if runtime.get("compile_platform") != compile_manifest.get("compile_platform"):
        raise TrustError("child capsule platform differs from compile manifest")

    expected_summary_keys = {
        "source_release",
        "binding",
        "execution_capsule",
        "execution_capsule_sha256",
        "compile_manifest_sha256",
        "translation_units",
        "compiled_kernel_audit_count",
        "disabled_fallback_red_count",
        "gpuwm_receipt_sha256",
        "post_capsule_authority_inputs_sha256",
        "post_capsule_nvidia_smi_sha256",
        "post_capsule_gpu_baseline_checkpoint_sha256",
        "sealed_output_tree_sha256",
        "post_capsule_checks_passed",
    }
    if set(child_summary) != expected_summary_keys:
        raise TrustError("child stdout summary has an unexpected field set")
    if child_summary.get("source_release") != "v8.4.1":
        raise TrustError("child stdout source release changed")
    if child_summary.get("binding") != str(output_root / "binding.json"):
        raise TrustError("child stdout binding path is false")
    if child_summary.get("execution_capsule") != str(
        output_root / "execution-capsule.json"
    ):
        raise TrustError("child stdout capsule path is false")
    if (
        child_summary.get("execution_capsule_sha256")
        != output_files["execution-capsule.json"]["sha256"]
    ):
        raise TrustError("child stdout capsule SHA-256 is false")
    if child_summary.get("compile_manifest_sha256") != compile_sha:
        raise TrustError("child stdout compile manifest SHA-256 changed")
    if child_summary.get("translation_units") != EXPECTED_TRANSLATION_UNITS or any(
        type(value) is not int
        for value in child_summary.get("translation_units", {}).values()
    ):
        raise TrustError("child stdout translation-unit count map changed")
    _require_int(
        child_summary.get("compiled_kernel_audit_count"),
        EXPECTED_AUDIT_KERNEL_COUNT,
        label="child stdout audit count",
    )
    _require_int(
        child_summary.get("disabled_fallback_red_count"),
        EXPECTED_DISABLED_RED_COUNT,
        label="child stdout disabled red count",
    )
    if child_summary.get("gpuwm_receipt_sha256") != EXPECTED_GPUWM_RECEIPT_SHA256:
        raise TrustError("child stdout woof receipt SHA-256 changed")
    if child_summary.get("post_capsule_checks_passed") is not True:
        raise TrustError("child post-capsule checks did not pass")
    if child_summary.get("post_capsule_authority_inputs_sha256") != _canonical_sha256(
        pre_snapshot
    ):
        raise TrustError("child post-capsule authority digest is false")
    capsule_smi = capsule["gpu_exclusivity"]["nvidia_smi_executable"]
    if child_summary.get("post_capsule_nvidia_smi_sha256") != capsule_smi.get("sha256"):
        raise TrustError("child post-capsule nvidia-smi digest is false")
    gpu_checkpoint_sha = child_summary.get(
        "post_capsule_gpu_baseline_checkpoint_sha256"
    )
    if (
        not isinstance(gpu_checkpoint_sha, str)
        or _SHA256_RE.fullmatch(gpu_checkpoint_sha) is None
    ):
        raise TrustError("child post-capsule GPU checkpoint digest is invalid")
    if child_summary.get("sealed_output_tree_sha256") != _canonical_sha256(
        output_files
    ):
        raise TrustError("child sealed output tree digest is false")

    return {
        "output_tree": _inventory_record(output_root, output_files),
        "cache_tree": _inventory_record(cache_root, cache_files),
        "compile_manifest": {
            **output_files["compile-manifest.json"],
            "canonical_sha256": _canonical_sha256(compile_manifest),
        },
        "kernel_audit": {
            **output_files["kernel-audit.json"],
            "kernel_count": EXPECTED_AUDIT_KERNEL_COUNT,
            "disabled_red_count": EXPECTED_DISABLED_RED_COUNT,
        },
        "binding": output_files["binding.json"],
        "execution_capsule": output_files["execution-capsule.json"],
        "translation_units": dict(EXPECTED_TRANSLATION_UNITS),
        "translation_unit_count": EXPECTED_TRANSLATION_UNIT_COUNT,
        "resolved_kernel_count": EXPECTED_RESOLVED_KERNEL_COUNT,
    }


def _validate_live_replay_output(
    *,
    binding_path: Path,
    validation_cache_root: Path,
    child_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if not validation_cache_root.is_dir() or validation_cache_root.is_symlink():
        raise TrustError("live-replay child did not create a real fresh cache root")
    cache_files = _tree_inventory(validation_cache_root)
    if set(cache_files) != {"validation-cache.json"}:
        raise TrustError("live-replay cache has an unexpected artifact set")
    cache_record, _ = _load_json(
        validation_cache_root / "validation-cache.json",
        label="live-replay cache record",
    )
    binding, _ = _load_json(binding_path, label="live-replay input binding")
    binding_sha = _canonical_sha256(binding)
    expected_cache_record = {
        "schema": VALIDATION_CACHE_SCHEMA,
        "binding_canonical_sha256": binding_sha,
        "validated_binding_canonical_sha256": binding_sha,
        "temporary_directory_parent": str(validation_cache_root),
        "cache_was_absent_and_exclusively_created": True,
        "four_pass_runner_requires_each_nested_cache_to_be_born_empty": True,
    }
    if cache_record != expected_cache_record:
        raise TrustError("live-replay cache record is false")
    expected_summary_fields = {
        "schema",
        "status",
        "binding",
        "binding_canonical_sha256",
        "validated_binding_canonical_sha256",
        "canonical_binding_equal",
        "validation_cache_directory",
        "validation_cache_record_sha256",
        "kernel_count",
        "disabled_fallback_red_count",
        "four_pass_live_replay",
        "authority_claim",
    }
    if set(child_summary) != expected_summary_fields:
        raise TrustError("live-replay stdout summary field set changed")
    if (
        child_summary.get("schema") != VALIDATION_CHILD_SCHEMA
        or child_summary.get("status") != "live-replay-validated"
        or child_summary.get("binding") != str(binding_path)
        or child_summary.get("binding_canonical_sha256") != binding_sha
        or child_summary.get("validated_binding_canonical_sha256") != binding_sha
        or child_summary.get("canonical_binding_equal") is not True
        or child_summary.get("validation_cache_directory")
        != str(validation_cache_root)
        or child_summary.get("validation_cache_record_sha256")
        != cache_files["validation-cache.json"]["sha256"]
        or child_summary.get("four_pass_live_replay") is not True
        or child_summary.get("authority_claim") is not False
    ):
        raise TrustError("live-replay stdout summary is false")
    _require_int(
        child_summary.get("kernel_count"),
        EXPECTED_AUDIT_KERNEL_COUNT,
        label="live-replay kernel count",
    )
    _require_int(
        child_summary.get("disabled_fallback_red_count"),
        EXPECTED_DISABLED_RED_COUNT,
        label="live-replay disabled red count",
    )
    return {
        "binding_canonical_sha256": binding_sha,
        "cache_tree": _inventory_record(validation_cache_root, cache_files),
        "cache_record": cache_files["validation-cache.json"],
        "canonical_binding_equal": True,
        "four_pass_live_replay": True,
    }


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise TrustError(f"completion artifact already exists: {path}") from error


def _create_completion_root(path: Path) -> Path:
    selected = path.resolve()
    if selected.exists() or selected.is_symlink():
        raise TrustError(f"completion root must still be absent: {selected}")
    selected.parent.mkdir(parents=True, exist_ok=True)
    try:
        selected.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise TrustError(
            f"completion root lost exclusive-create race: {selected}"
        ) from error
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run and independently promote the frozen v8.4.1 sm_120 FTZ "
            "measurement into a separate completion trust root."
        )
    )
    parser.add_argument("--gpuwm-root", type=Path, required=True)
    parser.add_argument("--gpuwm-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_launcher_isolation()
    if type(args.timeout_seconds) is not int or args.timeout_seconds <= 0:
        raise TrustError("--timeout-seconds must be a positive integer")
    pins = _load_authority_pins()
    output_root, cache_root, validation_cache_root, completion_target = _validate_targets(
        output_root=args.output_root,
        cache_root=args.cache_dir,
        validation_cache_root=args.validation_cache_dir,
        completion_root=args.completion_root,
        gpuwm_root=args.gpuwm_root,
        gpuwm_receipt=args.gpuwm_receipt,
    )

    # Load and enforce every static trust anchor before the measured child can
    # create either its output or cache root.
    authority_pre = _authority_snapshot(args.gpuwm_root, args.gpuwm_receipt)
    _validate_pre_pins(pins, authority_pre)
    launcher_record = _file_record(Path(__file__).resolve())
    isolated_bootstrap = _isolated_bootstrap_record()
    frozen_validator = _frozen_validator_record()
    gpuwm_startup_closure = _gpuwm_startup_closure_record(args.gpuwm_root)
    package_roots = _runtime_package_roots()
    python_record = _file_record(Path(sys.executable).resolve())
    nvidia_smi = _nvidia_smi_executable()
    nvidia_smi_pre = _file_record(nvidia_smi)
    gpu_pre = _gpu_state(nvidia_smi)

    child_argv = [
        str(Path(sys.executable).resolve()),
        "-I",
        "-S",
        "-B",
        str(ISOLATED_BOOTSTRAP.resolve()),
        "--frozen-tool",
        str(FROZEN_TOOL.resolve()),
        "--frozen-tool-sha256",
        FROZEN_TOOL_SHA256,
        "--gpuwm-root",
        str(args.gpuwm_root.expanduser().resolve()),
    ]
    for package_root in package_roots:
        child_argv.extend(("--package-root", str(package_root)))
    for relative, record in gpuwm_startup_closure["files"].items():
        child_argv.extend(("--gpuwm-closure", f"{relative}={record['sha256']}"))
    child_argv.extend(
        [
            "--",
            "--source-release",
            "v8.4.1",
            "--gpuwm-root",
            str(args.gpuwm_root.expanduser().resolve()),
            "--gpuwm-receipt",
            str(args.gpuwm_receipt.expanduser().resolve()),
            "--output-root",
            str(output_root),
            "--cache-dir",
            str(cache_root),
        ]
    )
    try:
        child = _invoke_child(
            child_argv,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TrustError(
            f"frozen measured child exceeded {args.timeout_seconds} seconds"
        ) from error
    if child.returncode != 0:
        raise TrustError(
            f"frozen measured child exited {child.returncode}; no completion is allowed"
        )
    if child.stderr != b"":
        raise TrustError("frozen measured child wrote stderr; no completion is allowed")
    child_summary = _parse_exact_child_stdout(child.stdout)

    authority_post = _authority_snapshot(args.gpuwm_root, args.gpuwm_receipt)
    _assert_same_authority(authority_pre, authority_post, phase="post-measurement")
    if _file_record(FROZEN_TOOL)["sha256"] != FROZEN_TOOL_SHA256:
        raise TrustError("frozen measured tool drifted during child execution")
    if _file_record(AUTHORITY_PINS)["sha256"] != AUTHORITY_PINS_SHA256:
        raise TrustError("static authority pins drifted during child execution")
    if _isolated_bootstrap_record() != isolated_bootstrap:
        raise TrustError("isolated child bootstrap drifted during child execution")
    if _gpuwm_startup_closure_record(args.gpuwm_root) != gpuwm_startup_closure:
        raise TrustError(
            "transitive GPUWM startup closure drifted during child execution"
        )
    nvidia_smi_post = _file_record(nvidia_smi)
    if nvidia_smi_post != nvidia_smi_pre:
        raise TrustError("nvidia-smi executable drifted during child execution")
    gpu_post = _gpu_state(nvidia_smi)
    if gpu_post != gpu_pre:
        raise TrustError("GPU identity or WDDM PID/UUID/name baseline drifted")
    measured = _validate_measured_output(
        output_root=output_root,
        cache_root=cache_root,
        child_summary=child_summary,
        pre_snapshot=authority_pre,
        pins=pins,
    )

    validation_argv = [
        str(Path(sys.executable).resolve()),
        "-I",
        "-S",
        "-B",
        str(ISOLATED_BOOTSTRAP.resolve()),
        "--frozen-tool",
        str(FROZEN_VALIDATOR.resolve()),
        "--frozen-tool-sha256",
        FROZEN_VALIDATOR_SHA256,
        "--gpuwm-root",
        str(args.gpuwm_root.expanduser().resolve()),
    ]
    for package_root in package_roots:
        validation_argv.extend(("--package-root", str(package_root)))
    for relative, record in gpuwm_startup_closure["files"].items():
        validation_argv.extend(("--gpuwm-closure", f"{relative}={record['sha256']}"))
    validation_argv.extend(
        [
            "--",
            "--binding",
            str(output_root / "binding.json"),
            "--gpuwm-root",
            str(args.gpuwm_root.expanduser().resolve()),
            "--gpuwm-receipt",
            str(args.gpuwm_receipt.expanduser().resolve()),
            "--cache-dir",
            str(validation_cache_root),
        ]
    )
    try:
        validation_child = _invoke_child(
            validation_argv,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TrustError(
            f"isolated live-replay child exceeded {args.timeout_seconds} seconds"
        ) from error
    if validation_child.returncode != 0:
        raise TrustError(
            "isolated live-replay child exited "
            f"{validation_child.returncode}; no completion is allowed"
        )
    if validation_child.stderr != b"":
        raise TrustError(
            "isolated live-replay child wrote stderr; no completion is allowed"
        )
    validation_summary = _parse_exact_child_stdout(validation_child.stdout)
    replay = _validate_live_replay_output(
        binding_path=output_root / "binding.json",
        validation_cache_root=validation_cache_root,
        child_summary=validation_summary,
    )

    authority_post_replay = _authority_snapshot(args.gpuwm_root, args.gpuwm_receipt)
    _assert_same_authority(authority_pre, authority_post_replay, phase="post-live-replay")
    if _file_record(FROZEN_TOOL)["sha256"] != FROZEN_TOOL_SHA256:
        raise TrustError("frozen measured tool drifted during live replay")
    if _frozen_validator_record() != frozen_validator:
        raise TrustError("isolated live-replay validator drifted during execution")
    if _file_record(AUTHORITY_PINS)["sha256"] != AUTHORITY_PINS_SHA256:
        raise TrustError("static authority pins drifted during live replay")
    if _isolated_bootstrap_record() != isolated_bootstrap:
        raise TrustError("isolated child bootstrap drifted during live replay")
    if _gpuwm_startup_closure_record(args.gpuwm_root) != gpuwm_startup_closure:
        raise TrustError("transitive GPUWM startup closure drifted during live replay")
    nvidia_smi_post_replay = _file_record(nvidia_smi)
    if nvidia_smi_post_replay != nvidia_smi_pre:
        raise TrustError("nvidia-smi executable drifted during live replay")
    gpu_post_replay = _gpu_state(nvidia_smi)
    if gpu_post_replay != gpu_pre:
        raise TrustError(
            "GPU identity or WDDM PID/UUID/name baseline drifted during live replay"
        )
    output_post_replay = _tree_inventory(output_root)
    if output_post_replay != measured["output_tree"]["files"]:
        raise TrustError("measured output tree drifted during live replay")

    completion = _create_completion_root(completion_target)
    receipt = {
        "schema": COMPLETION_RECEIPT_SCHEMA,
        "status": "measurement-validated-before-completion-publication",
        "source_release": "v8.4.1",
        "weather_authority_claim": False,
        "launcher": {
            "path": str(Path(__file__).resolve()),
            **launcher_record,
        },
        "isolated_child_bootstrap": isolated_bootstrap,
        "gpuwm_startup_closure": gpuwm_startup_closure,
        "static_authority_pins": {
            "path": str(AUTHORITY_PINS.resolve()),
            **_file_record(AUTHORITY_PINS),
            "document": pins,
        },
        "frozen_measured_tool": {
            "path": str(FROZEN_TOOL.resolve()),
            **_file_record(FROZEN_TOOL),
        },
        "frozen_live_replay_validator": frozen_validator,
        "python": {
            "path": str(Path(sys.executable).resolve()),
            **python_record,
        },
        "child": {
            "argv": child_argv,
            "cwd": str(ROOT),
            "environment_policy": {
                "isolated_startup_flag": "-I",
                "site_disabled_flag": "-S",
                "bytecode_disabled_flag": "-B",
                "gpuwm_namespace_installed_after_startup": str(
                    args.gpuwm_root.expanduser().resolve()
                ),
                "post_startup_package_roots": [str(path) for path in package_roots],
                "mpas_and_gpuwm_source_only_imports": True,
                "all_PYTHON_environment_variables_removed_before_exec": True,
                "CUPY_CACHE_DIR_removed_before_exec": True,
                "MPAS_PORT_CUDA_CACHE_DIR_removed_before_exec": True,
            },
            "timeout_seconds": args.timeout_seconds,
            "exit_code": child.returncode,
            "stdout": {
                "bytes": len(child.stdout),
                "sha256": _sha256_bytes(child.stdout),
                "exactly_one_canonical_json_document": True,
                "parsed_summary": child_summary,
            },
            "stderr": {
                "bytes": len(child.stderr),
                "sha256": _sha256_bytes(child.stderr),
                "empty": child.stderr == b"",
            },
        },
        "live_replay_child": {
            "argv": validation_argv,
            "cwd": str(ROOT),
            "environment_policy": {
                "isolated_startup_flag": "-I",
                "site_disabled_flag": "-S",
                "bytecode_disabled_flag": "-B",
                "gpuwm_namespace_installed_after_startup": str(
                    args.gpuwm_root.expanduser().resolve()
                ),
                "post_startup_package_roots": [str(path) for path in package_roots],
                "mpas_and_gpuwm_source_only_imports": True,
                "all_PYTHON_environment_variables_removed_before_exec": True,
                "fresh_validation_cache": str(validation_cache_root),
            },
            "timeout_seconds": args.timeout_seconds,
            "exit_code": validation_child.returncode,
            "stdout": {
                "bytes": len(validation_child.stdout),
                "sha256": _sha256_bytes(validation_child.stdout),
                "exactly_one_canonical_json_document": True,
                "parsed_summary": validation_summary,
            },
            "stderr": {
                "bytes": len(validation_child.stderr),
                "sha256": _sha256_bytes(validation_child.stderr),
                "empty": validation_child.stderr == b"",
            },
            "validated_evidence": replay,
        },
        "authority_checks": {
            "pre": authority_pre,
            "post_measurement": authority_post,
            "post_live_replay": authority_post_replay,
            "pre_sha256": _canonical_sha256(authority_pre),
            "post_measurement_sha256": _canonical_sha256(authority_post),
            "post_live_replay_sha256": _canonical_sha256(authority_post_replay),
            "byte_identical": (
                authority_pre == authority_post == authority_post_replay
            ),
        },
        "gpu_process_baseline": {
            "nvidia_smi": {"path": str(nvidia_smi), **nvidia_smi_pre},
            "pre": gpu_pre,
            "post_measurement": gpu_post,
            "post_live_replay": gpu_post_replay,
            "byte_identical": gpu_pre == gpu_post == gpu_post_replay,
            "physical_gpu_exclusivity_claim": False,
        },
        "expected_counts": {
            "translation_units": EXPECTED_TRANSLATION_UNIT_COUNT,
            "resolved_kernels": EXPECTED_RESOLVED_KERNEL_COUNT,
            "audit_kernels": EXPECTED_AUDIT_KERNEL_COUNT,
            "disabled_red": EXPECTED_DISABLED_RED_COUNT,
        },
        "measured_evidence": measured,
        "publication": {
            "measured_output_root": str(output_root),
            "measured_cache_root": str(cache_root),
            "live_replay_validation_cache_root": str(validation_cache_root),
            "completion_root": str(completion),
            "completion_root_was_absent_and_exclusively_created": True,
            "receipt_write_uses_exclusive_create_and_fsync": True,
            "completion_manifest_is_intermediate": True,
            "completion_seal_is_required_for_promotion": True,
        },
    }
    receipt_path = completion / "completion-receipt.json"
    _write_json_exclusive(receipt_path, receipt)

    # These checks occur after the first completion artifact is durable.  Their
    # results are bound into the final completion manifest, which is the trust
    # root consumed by promotion.
    authority_after_publication = _authority_snapshot(
        args.gpuwm_root, args.gpuwm_receipt
    )
    _assert_same_authority(
        authority_pre,
        authority_after_publication,
        phase="after-completion-receipt-publication",
    )
    nvidia_smi_after_publication = _file_record(nvidia_smi)
    if nvidia_smi_after_publication != nvidia_smi_pre:
        raise TrustError("nvidia-smi drifted after completion receipt publication")
    gpu_after_publication = _gpu_state(nvidia_smi)
    if gpu_after_publication != gpu_pre:
        raise TrustError(
            "GPU identity or WDDM process baseline drifted after receipt publication"
        )
    output_after_publication = _tree_inventory(output_root)
    if output_after_publication != measured["output_tree"]["files"]:
        raise TrustError("measured output tree drifted after completion publication")
    replay_cache_after_publication = _tree_inventory(validation_cache_root)
    if replay_cache_after_publication != replay["cache_tree"]["files"]:
        raise TrustError("live-replay cache drifted after completion publication")
    receipt_record = _file_record(receipt_path)
    receipt_reloaded, _ = _load_json(receipt_path, label="completion receipt")
    if receipt_reloaded != receipt:
        raise TrustError(
            "completion receipt bytes do not decode to the published object"
        )
    if _gpuwm_startup_closure_record(args.gpuwm_root) != gpuwm_startup_closure:
        raise TrustError("transitive GPUWM startup closure drifted after receipt")
    if _frozen_validator_record() != frozen_validator:
        raise TrustError("isolated live-replay validator drifted after receipt")

    manifest = {
        "schema": COMPLETION_MANIFEST_SCHEMA,
        "status": "complete",
        "promotion_trust_root": False,
        "final_seal_required_for_promotion": True,
        "source_release": "v8.4.1",
        "completion_receipt": {
            "path": str(receipt_path),
            **receipt_record,
            "canonical_sha256": _canonical_sha256(receipt),
        },
        "launcher": {
            "path": str(Path(__file__).resolve()),
            **launcher_record,
        },
        "isolated_child_bootstrap": isolated_bootstrap,
        "gpuwm_startup_closure": gpuwm_startup_closure,
        "frozen_measured_tool": {
            "path": str(FROZEN_TOOL.resolve()),
            **_file_record(FROZEN_TOOL),
        },
        "frozen_live_replay_validator": frozen_validator,
        "static_authority_pins": {
            "path": str(AUTHORITY_PINS.resolve()),
            **_file_record(AUTHORITY_PINS),
        },
        "after_receipt_publication_checks": {
            "authority": authority_after_publication,
            "authority_sha256": _canonical_sha256(authority_after_publication),
            "authority_matches_pre": authority_after_publication == authority_pre,
            "nvidia_smi": {
                "path": str(nvidia_smi),
                **nvidia_smi_after_publication,
            },
            "gpu_process_baseline": gpu_after_publication,
            "gpu_process_baseline_matches_pre": gpu_after_publication == gpu_pre,
            "physical_gpu_exclusivity_claim": False,
            "measured_output_tree_sha256": _canonical_sha256(output_after_publication),
            "measured_output_tree_matches_validated": (
                output_after_publication == measured["output_tree"]["files"]
            ),
            "live_replay_cache_tree_sha256": _canonical_sha256(
                replay_cache_after_publication
            ),
            "live_replay_cache_tree_matches_validated": (
                replay_cache_after_publication == replay["cache_tree"]["files"]
            ),
        },
        "exact_counts": {
            "translation_units": EXPECTED_TRANSLATION_UNIT_COUNT,
            "resolved_kernels": EXPECTED_RESOLVED_KERNEL_COUNT,
            "audit_kernels": EXPECTED_AUDIT_KERNEL_COUNT,
            "disabled_red": EXPECTED_DISABLED_RED_COUNT,
        },
        "claim": (
            "The frozen v8.4.1 FTZ child exited zero with one exact JSON summary; "
            "its eight-TU/46-resolved-kernel manifest and 95-kernel audit with "
            "78 disabled-fallback red rows match literal outcome/source pins. A "
            "second byte-pinned isolated child rebuilt the binding and performed "
            "a fresh canonically-identical four-pass live replay. "
            "Independent authority, output-tree, nvidia-smi, and stable WDDM "
            "PID/UUID/name checks remained unchanged after receipt publication. "
            "This is an FTZ execution promotion receipt, not weather authority or "
            "a physical GPU-exclusivity claim."
        ),
    }
    manifest_path = completion / "completion-manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    pre_seal_completion_files = _tree_inventory(completion)
    if set(pre_seal_completion_files) != {
        "completion-receipt.json",
        "completion-manifest.json",
    }:
        raise TrustError("completion root has an unexpected file set")
    if pre_seal_completion_files["completion-receipt.json"] != receipt_record:
        raise TrustError("completion receipt changed after manifest publication")

    # Establish one finite outer trust boundary after manifest publication.
    # All authority-dependent work is complete at this checkpoint. The final
    # seal serializes these already-measured values with O_EXCL+fsync; after it
    # is written, only its own bytes and the three-file completion tree are read.
    authority_after_manifest = _authority_snapshot(args.gpuwm_root, args.gpuwm_receipt)
    _assert_same_authority(
        authority_pre,
        authority_after_manifest,
        phase="after-completion-manifest-publication",
    )
    nvidia_smi_after_manifest = _file_record(nvidia_smi)
    if nvidia_smi_after_manifest != nvidia_smi_pre:
        raise TrustError("nvidia-smi drifted after completion manifest publication")
    gpu_after_manifest = _gpu_state(nvidia_smi)
    if gpu_after_manifest != gpu_pre:
        raise TrustError(
            "GPU identity or WDDM process baseline drifted after manifest publication"
        )
    output_after_manifest = _tree_inventory(output_root)
    if output_after_manifest != measured["output_tree"]["files"]:
        raise TrustError("measured output tree drifted after manifest publication")
    replay_cache_after_manifest = _tree_inventory(validation_cache_root)
    if replay_cache_after_manifest != replay["cache_tree"]["files"]:
        raise TrustError("live-replay cache drifted after manifest publication")
    receipt_after_manifest = _file_record(receipt_path)
    if receipt_after_manifest != receipt_record:
        raise TrustError("completion receipt drifted during manifest publication")
    manifest_record = _file_record(manifest_path)
    manifest_reloaded, _ = _load_json(manifest_path, label="completion manifest")
    if manifest_reloaded != manifest:
        raise TrustError(
            "completion manifest bytes do not decode to the published object"
        )
    if _isolated_bootstrap_record() != isolated_bootstrap:
        raise TrustError("isolated child bootstrap drifted before final seal")
    if _gpuwm_startup_closure_record(args.gpuwm_root) != gpuwm_startup_closure:
        raise TrustError("transitive GPUWM startup closure drifted before final seal")
    if _frozen_validator_record() != frozen_validator:
        raise TrustError("isolated live-replay validator drifted before final seal")

    seal = {
        "schema": COMPLETION_SEAL_SCHEMA,
        "status": "complete",
        "promotion_trust_root": True,
        "source_release": "v8.4.1",
        "finite_trust_boundary": (
            "This seal binds the checkpoint measured after completion-manifest "
            "publication. No authority-dependent computation or external write "
            "occurs after this checkpoint; only O_EXCL+fsync serialization of this "
            "object and read-back verification of the completion tree remain."
        ),
        "completion_receipt": {
            "path": str(receipt_path),
            **receipt_after_manifest,
        },
        "completion_manifest": {
            "path": str(manifest_path),
            **manifest_record,
            "canonical_sha256": _canonical_sha256(manifest),
        },
        "launcher": {
            "path": str(Path(__file__).resolve()),
            **launcher_record,
        },
        "isolated_child_bootstrap": isolated_bootstrap,
        "gpuwm_startup_closure": gpuwm_startup_closure,
        "frozen_measured_tool": {
            "path": str(FROZEN_TOOL.resolve()),
            **_file_record(FROZEN_TOOL),
        },
        "frozen_live_replay_validator": frozen_validator,
        "static_authority_pins": {
            "path": str(AUTHORITY_PINS.resolve()),
            **_file_record(AUTHORITY_PINS),
        },
        "post_manifest_checkpoint": {
            "authority": authority_after_manifest,
            "authority_sha256": _canonical_sha256(authority_after_manifest),
            "authority_matches_pre": authority_after_manifest == authority_pre,
            "nvidia_smi": {
                "path": str(nvidia_smi),
                **nvidia_smi_after_manifest,
            },
            "gpu_process_baseline": gpu_after_manifest,
            "gpu_process_baseline_matches_pre": gpu_after_manifest == gpu_pre,
            "physical_gpu_exclusivity_claim": False,
            "measured_output_tree_sha256": _canonical_sha256(output_after_manifest),
            "measured_output_tree_matches_validated": (
                output_after_manifest == measured["output_tree"]["files"]
            ),
            "live_replay_cache_tree_sha256": _canonical_sha256(
                replay_cache_after_manifest
            ),
            "live_replay_cache_tree_matches_validated": (
                replay_cache_after_manifest == replay["cache_tree"]["files"]
            ),
            "pre_seal_completion_tree": _inventory_record(
                completion, pre_seal_completion_files
            ),
        },
        "exact_counts": {
            "translation_units": EXPECTED_TRANSLATION_UNIT_COUNT,
            "resolved_kernels": EXPECTED_RESOLVED_KERNEL_COUNT,
            "audit_kernels": EXPECTED_AUDIT_KERNEL_COUNT,
            "disabled_red": EXPECTED_DISABLED_RED_COUNT,
        },
    }
    seal_path = completion / "completion-seal.json"
    _write_json_exclusive(seal_path, seal)
    seal_reloaded, _ = _load_json(seal_path, label="completion seal")
    if seal_reloaded != seal:
        raise TrustError("completion seal read-back differs from the sealed object")
    completion_files = _tree_inventory(completion)
    if set(completion_files) != {
        "completion-receipt.json",
        "completion-manifest.json",
        "completion-seal.json",
    }:
        raise TrustError("sealed completion root has an unexpected file set")
    if completion_files["completion-receipt.json"] != receipt_after_manifest:
        raise TrustError("completion receipt changed during final seal publication")
    if completion_files["completion-manifest.json"] != manifest_record:
        raise TrustError("completion manifest changed during final seal publication")

    final_summary = {
        "status": "complete",
        "completion_seal": str(seal_path),
        "completion_seal_sha256": completion_files["completion-seal.json"]["sha256"],
        "completion_manifest_sha256": manifest_record["sha256"],
        "completion_receipt_sha256": receipt_record["sha256"],
        "measured_output_tree_sha256": measured["output_tree"]["files_sha256"],
        "physical_gpu_exclusivity_claim": False,
    }
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
