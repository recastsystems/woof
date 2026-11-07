"""Concurrent, process-isolated production forecast orchestration.

The orchestrator resolves physical devices through the supervisor's
``nvidia-smi`` inventory and starts each fresh child with one physical UUID in
``CUDA_VISIBLE_DEVICES``.  Check and prepared-runner wrappers acquire the
existing machine-wide UUID lock before importing their CUDA-facing target.
Inside a masked child, CUDA device ordinal zero is therefore the selected
process-local device; isolation does not depend on the import state of the
calling ``woof.cli`` process.

Each run gets independent output, temporary, CuPy-cache, and driver-JIT-cache
directories.  A versioned JSON summary is published from a fully written
sibling through atomic create-only publication after every child has finished
(or after the parent is interrupted).  Interrupts do not terminate children:
a supervised ``woof run`` may itself own a fresh CUDA worker, so killing only
the visible supervisor could orphan work.  The interrupted summary records
every PID whose completion was not observed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from woof.filesystem_paths import publish_new
from woof.supervisor import (GPUFileLock, GPUIdentity, GPU_LOCK_ROOT_ENV,
                              SHARED_INPUT_AUTHORITY_ROOT_ENV,
                              preflight_exclusive_gpu,
                              priced_reservation_bytes, query_gpus, utc_now)


PLAN_SCHEMA = "gpuwm.multi-run-plan/v1"
SUMMARY_SCHEMA = "gpuwm.multi-run-summary/v1"
PREFLIGHT_MODES = ("estimate", "alloc", "off")
AUTHORITY_MONITOR_POLL_SECONDS = 0.05
SUPPORTED_PREPARED_RUNNERS = frozenset({
    "woof.prepared_domain_tree_forecast",
    "woof.prepared_single_domain_forecast",
})
_RUNNER_PATH_FLAGS = {
    "woof.prepared_domain_tree_forecast": {
        "required": ("--prepared-root", "--experiment-config"),
        "optional": ("--restart",),
    },
    "woof.prepared_single_domain_forecast": {
        "required": (
            "--prepared-root", "--experiment-config", "--wps-namelist"),
        "optional": ("--domain-bundle",),
    },
}
_NON_FORECAST_ARGUMENTS = frozenset({
    "--materialize-authorities", "--show-capabilities",
})
_TOP_LEVEL_KEYS = frozenset({"schema", "summary", "preflight", "run"})
_COMMON_RUN_KEYS = frozenset(
    {"name", "device", "outdir", "scratch", "cache"})
_RUN_KEYS = frozenset(
    {*_COMMON_RUN_KEYS, "config", "module", "args", "inputs"})


@dataclass(frozen=True)
class InputBinding:
    """Pre-launch content authority for one declared read-only input."""

    path: Path
    kind: str
    sha256: str | None
    captured_bytes: bytes | None = field(default=None, repr=False,
                                         compare=False)
    captured_signature: tuple[int, int, int, int, int] | None = field(
        default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "binding": (
                "prelaunch_sha256" if self.sha256 is not None
                else "delegated_to_prepared_runner_receipt"),
            "kind": self.kind,
            "path": str(self.path),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class RunSpec:
    """One validated plan entry before its selector is resolved."""

    name: str
    device_selector: str
    outdir: Path
    scratch_dir: Path
    cache: Path
    config: Path | None
    module: str | None
    arguments: tuple[str, ...]
    inputs: tuple[Path, ...]
    input_bindings: tuple[InputBinding, ...] = ()

    @property
    def log(self) -> Path:
        return self.scratch_dir / "gpuwm-run.log"

    @property
    def preflight_log(self) -> Path:
        return self.scratch_dir / "gpuwm-check.log"

    @property
    def captured_config(self) -> Path:
        """Create-only immutable payload consumed by every config phase."""

        return self.scratch_dir / "captured-config.toml"

    @property
    def is_config_run(self) -> bool:
        return self.config is not None


@dataclass(frozen=True)
class MultiRunPlan:
    """A parsed plan whose paths are absolute and collision-checked."""

    path: Path
    sha256: str
    captured_signature: tuple[int, int, int, int, int]
    summary: Path
    input_authority_store: Path
    preflight: str
    runs: tuple[RunSpec, ...]


@dataclass(frozen=True)
class PlannedRun:
    """A plan entry bound to one physical GPU."""

    spec: RunSpec
    gpu: GPUIdentity
    lock_root: Path | None = None
    input_authority_store: Path | None = None


@dataclass
class ProcessRecord:
    """Observed lifecycle of one check or run child."""

    name: str
    phase: str
    command: tuple[str, ...]
    log: Path
    status: str = "not_started"
    pid: int | None = None
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    error: str | None = None
    _started_monotonic: float | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "command": list(self.command),
            "completed_at_utc": self.completed_at_utc,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "log": str(self.log),
            "pid": self.pid,
            "started_at_utc": self.started_at_utc,
            "status": self.status,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class GroupOutcome:
    """Results from one concurrently launched process group."""

    records: tuple[ProcessRecord, ...]
    interrupted: bool
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    wall_seconds: float | None = None


class GroupExecutionError(RuntimeError):
    """Guaranteed carrier for a partial group ledger and original failure."""

    def __init__(self, phase: str, outcome: GroupOutcome,
                 original: BaseException):
        super().__init__(f"{type(original).__name__}: {original}")
        self.phase = phase
        self.outcome = outcome
        self.original = original


@dataclass
class OrchestrationState:
    """Mutable stage ledger used for accurate interruption receipts."""

    stage: str = "plan_load"
    summary_capable: bool = False
    claimed_directories: list[Path] = field(default_factory=list)
    error: str | None = None


def _orchestration_record(
        state: OrchestrationState, plan: MultiRunPlan,
        preflight: GroupOutcome | None,
        execution: GroupOutcome | None) -> dict[str, object]:
    records = []
    if preflight is not None:
        records.extend(preflight.records)
    if execution is not None:
        records.extend(execution.records)
    known_pids = sorted({
        record.pid for record in records if record.pid is not None})
    unobserved_pids = sorted({
        record.pid for record in records
        if record.pid is not None and record.status in {
            "launch_unobserved", "running_unobserved"}})
    existing_run_directories = []
    for run in plan.runs:
        for path in (
                run.outdir, run.scratch_dir, run.cache,
                run.cache / "cupy", run.cache / "cuda"):
            if path.is_dir():
                existing_run_directories.append(str(path))
    payload: dict[str, object] = {
        "existing_run_directories": sorted(set(existing_run_directories)),
        "known_child_pids": known_pids,
        "shared_input_authority_store": str(plan.input_authority_store),
        "parent_claimed_directories": [
            str(path) for path in state.claimed_directories],
        "stage": state.stage,
        "summary_create_only_capability_proven": state.summary_capable,
        "unobserved_child_pids": unobserved_pids,
    }
    if state.error is not None:
        payload["error"] = state.error
    return payload


def _input_authority_store_record(plan: MultiRunPlan) -> dict[str, object]:
    """Receipt for the plan-wide, SHA-keyed immutable content store."""

    entries = []
    if plan.input_authority_store.is_dir():
        for path in sorted(plan.input_authority_store.iterdir()):
            if (not path.is_file() or len(path.name) != 64
                    or any(c not in "0123456789abcdef" for c in path.name)):
                continue
            entries.append({
                "bytes": path.stat().st_size,
                "path": str(path),
                "sha256": path.name,
            })
    return {
        "content_deduplication": "one_create_only_file_per_sha256",
        "entries": entries,
        "path": str(plan.input_authority_store),
        "status": (
            "MISSING" if not plan.input_authority_store.is_dir()
            else "POPULATED" if entries else "EMPTY"),
    }


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    result = value.strip()
    if any(character in result for character in "\r\n"):
        raise ValueError(f"{label} must fit on one line")
    return result


def _device_selector(value: object, label: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a GPU index or full GPU UUID")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{label} GPU index must be nonnegative")
        return str(value)
    selector = _nonempty_string(value, label)
    if selector.isdigit() or selector.startswith("GPU-"):
        return selector
    raise ValueError(
        f"{label} must be a nonnegative GPU index or full GPU UUID, got "
        f"{selector!r}")


def _module_name(value: object, label: str) -> str:
    module = _nonempty_string(value, label)
    if module not in SUPPORTED_PREPARED_RUNNERS:
        raise ValueError(
            f"{label} must be one of the supported prepared forecast "
            f"runners {sorted(SUPPORTED_PREPARED_RUNNERS)}, got {module!r}")
    return module


def _argument_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a TOML array of argv strings")
    arguments = tuple(
        _nonempty_string(argument, f"{label}[{index}]")
        for index, argument in enumerate(value))
    return arguments


def _canonical_outdir_value(arguments: Sequence[str], label: str) -> str:
    alternates = [
        argument for argument in arguments
        if argument.startswith("--out") and argument != "--outdir"]
    if alternates:
        raise ValueError(
            f"{label} has alternate output option(s) {alternates}; use "
            "exactly '--outdir', followed by its separate value")
    positions = [
        index for index, argument in enumerate(arguments)
        if argument == "--outdir"]
    if len(positions) != 1:
        raise ValueError(
            f"{label} must contain exactly one canonical --outdir option, "
            f"got {len(positions)}")
    index = positions[0]
    if index + 1 >= len(arguments):
        raise ValueError(f"{label} --outdir is missing its value")
    return arguments[index + 1]


def _validate_outdir_template(arguments: Sequence[str], label: str) -> None:
    value = _canonical_outdir_value(arguments, label)
    if value != "{outdir}" or sum(
            argument.count("{outdir}") for argument in arguments) != 1:
        raise ValueError(
            f"{label} must use exactly '--outdir', '{{outdir}}' once; no "
            "fixed, embedded, duplicate, or alternate output path is allowed")


def _validate_rendered_outdir(arguments: Sequence[str], outdir: Path,
                              label: str) -> None:
    value = _canonical_outdir_value(arguments, label)
    if value != str(outdir):
        raise ValueError(
            f"{label} --outdir value {value!r} does not match the "
            f"collision-checked output root {str(outdir)!r}")


def _runner_path_values(module: str, arguments: Sequence[str],
                        label: str) -> dict[str, str]:
    if any(argument in _NON_FORECAST_ARGUMENTS for argument in arguments):
        raise ValueError(
            f"{label} selects a non-forecast mode; multi-run only launches "
            "prepared forecasts")
    specification = _RUNNER_PATH_FLAGS[module]
    required = specification["required"]
    optional = specification["optional"]
    flags = (*required, *optional)
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        for flag in flags:
            if argument.startswith(f"{flag}=") or (
                    option != flag and flag.startswith(option)):
                raise ValueError(
                    f"{label} uses alternate path option {argument!r}; "
                    f"use the complete separate form {flag!r}, VALUE")
    values: dict[str, str] = {}
    for flag in flags:
        positions = [
            index for index, argument in enumerate(arguments)
            if argument == flag]
        maximum = 1
        minimum = 1 if flag in required else 0
        if not minimum <= len(positions) <= maximum:
            expectation = "exactly once" if minimum else "at most once"
            raise ValueError(
                f"{label} must contain {flag} {expectation}, got "
                f"{len(positions)}")
        if not positions:
            continue
        index = positions[0]
        if index + 1 >= len(arguments):
            raise ValueError(f"{label} {flag} is missing its value")
        values[flag] = arguments[index + 1]
    return values


def _validate_runner_input_templates(
        module: str, arguments: Sequence[str], input_count: int,
        label: str) -> None:
    values = _runner_path_values(module, arguments, label)
    used: list[str] = []
    for flag, value in values.items():
        match = re.fullmatch(r"\{input(\d+)\}", value)
        if match is None:
            raise ValueError(
                f"{label} {flag} value must be one exact declared "
                f"{{inputN}} placeholder, got {value!r}")
        index = int(match.group(1))
        if index >= input_count:
            raise ValueError(
                f"{label} {flag} refers to undeclared input {index}")
        used.append(match.group(0))
    occurrences = [
        match.group(0) for argument in arguments
        for match in re.finditer(r"\{input\d+\}", argument)]
    if sorted(occurrences) != sorted(used):
        raise ValueError(
            f"{label} may use {{inputN}} placeholders only as exact values "
            "of the prepared runner's declared path options")
    missing = sorted(
        set(range(input_count))
        - {int(token[6:-1]) for token in used})
    if missing:
        raise ValueError(
            f"{label} does not bind declared input index(es) {missing} to a "
            "prepared-runner path option")


def _validate_rendered_runner_inputs(
        module: str, arguments: Sequence[str], inputs: Sequence[Path],
        label: str) -> None:
    values = _runner_path_values(module, arguments, label)
    keys = [os.path.normcase(str(path)) for path in inputs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains duplicate declared input paths")
    declared = {str(path) for path in inputs}
    used = set(values.values())
    escaped = sorted(used - declared)
    if escaped:
        raise ValueError(
            f"{label} contains path value(s) outside its declared inputs: "
            f"{escaped}")
    missing = sorted(declared - used)
    if missing:
        raise ValueError(
            f"{label} does not bind declared input path(s): {missing}")


def _validate_placeholders(arguments: Sequence[str], input_count: int,
                           label: str) -> None:
    fixed = {"{outdir}", "{scratch}", "{cache}"}
    for argument in arguments:
        recognized = fixed | set(re.findall(r"\{input\d+\}", argument))
        for match in re.finditer(r"\{[A-Za-z_][A-Za-z0-9_]*\}", argument):
            token = match.group(0)
            if token not in recognized:
                raise ValueError(
                    f"{label} argument {argument!r} has unknown "
                    f"placeholder {token!r}")
        for match in re.finditer(r"\{input(\d+)\}", argument):
            if int(match.group(1)) >= input_count:
                raise ValueError(
                    f"{label} argument {argument!r} refers to undeclared "
                    f"input {match.group(1)}")
        remainder = argument
        for token in recognized:
            remainder = remainder.replace(token, "")
        if any(prefix in remainder for prefix in (
                "{outdir", "{scratch", "{cache", "{input")):
            raise ValueError(
                f"{label} argument {argument!r} has a malformed reserved "
                "placeholder")


def _input_paths(base: Path, value: object, label: str) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{label} must be a non-empty TOML array of read-only paths")
    paths = tuple(_plan_path(base, item, f"{label}[{index}]")
                  for index, item in enumerate(value))
    keys = [os.path.normcase(str(path)) for path in paths]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains a duplicate declared input path")
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _authority_signature(path: Path) -> tuple[int, int, int, int, int]:
    metadata = path.stat()
    return (
        int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_size),
        int(metadata.st_mtime_ns), int(metadata.st_ctime_ns))


def _bind_input(path: Path, *, capture_bytes: bool = False) -> InputBinding:
    if path.is_file():
        before = _authority_signature(path)
        payload = path.read_bytes() if capture_bytes else None
        digest = (hashlib.sha256(payload).hexdigest()
                  if payload is not None else _file_sha256(path))
        after = _authority_signature(path)
        if before != after:
            raise ValueError(
                f"declared input changed while it was captured: {path}")
        return InputBinding(path, "file", digest, payload, after)
    if path.is_dir():
        return InputBinding(path, "directory", None)
    raise ValueError(
        f"declared input is neither a regular file nor directory: {path}")


def _plan_path(base: Path, value: object, label: str) -> Path:
    text = _nonempty_string(value, label)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _same_or_nested(first: Path, second: Path) -> bool:
    """Whether two resolved paths alias or one contains the other."""

    if first.exists() and second.exists():
        try:
            if os.path.samefile(first, second):
                return True
        except OSError:
            pass
    first_key = os.path.normcase(str(first))
    second_key = os.path.normcase(str(second))
    try:
        common = os.path.commonpath((first_key, second_key))
    except ValueError:  # different Windows drives
        return False
    return common == first_key or common == second_key


def _validate_path_isolation(plan_path: Path, summary: Path,
                             input_authority_store: Path,
                             runs: Sequence[RunSpec]) -> None:
    mutable: list[tuple[str, Path]] = []
    inputs: list[tuple[str, Path]] = []
    for run in runs:
        mutable.extend((
            (f"run {run.name!r} outdir", run.outdir),
            (f"run {run.name!r} scratch", run.scratch_dir),
            (f"run {run.name!r} cache", run.cache),
        ))
        inputs.extend(
            (f"run {run.name!r} input {index}", path)
            for index, path in enumerate(run.inputs))
    for index, (left_label, left_path) in enumerate(mutable):
        for right_label, right_path in mutable[index + 1:]:
            if _same_or_nested(left_path, right_path):
                raise ValueError(
                    f"{left_label} {left_path} overlaps {right_label} "
                    f"{right_path}; every mutable outdir/scratch/cache root "
                    "must be distinct and non-overlapping")

    if summary == plan_path:
        raise ValueError("multi-run summary must not overwrite its plan")
    if summary.exists():
        raise ValueError(
            f"multi-run summary {summary} already exists; choose a new path "
            "so an earlier production summary is not overwritten")
    if input_authority_store.exists():
        raise ValueError(
            f"multi-run input authority store {input_authority_store} "
            "already exists; choose a new summary so earlier captured "
            "inputs are never mixed into this plan")
    if _same_or_nested(input_authority_store, plan_path):
        raise ValueError(
            f"multi-run input authority store {input_authority_store} "
            f"overlaps the plan {plan_path}")
    if _same_or_nested(input_authority_store, summary):
        raise ValueError(
            f"multi-run input authority store {input_authority_store} "
            f"overlaps the summary {summary}")
    for label, path in mutable:
        if _same_or_nested(plan_path, path):
            raise ValueError(
                f"{label} {path} overlaps the plan {plan_path}")
        if _same_or_nested(summary, path):
            raise ValueError(
                f"{label} {path} overlaps the summary {summary}")
        if _same_or_nested(input_authority_store, path):
            raise ValueError(
                f"multi-run input authority store {input_authority_store} "
                f"overlaps {label} {path}")

    for input_label, input_path in inputs:
        if not input_path.exists():
            raise ValueError(
                f"{input_label} does not exist: {input_path}")
        if _same_or_nested(summary, input_path):
            raise ValueError(
                f"summary {summary} overlaps read-only {input_label} "
                f"{input_path}")
        if _same_or_nested(input_authority_store, input_path):
            raise ValueError(
                f"multi-run input authority store {input_authority_store} "
                f"overlaps read-only {input_label} {input_path}")
        for mutable_label, mutable_path in mutable:
            if _same_or_nested(input_path, mutable_path):
                raise ValueError(
                    f"{mutable_label} {mutable_path} overlaps read-only "
                    f"{input_label} {input_path}")

    for run in runs:
        if run.config is not None and not run.config.is_file():
            raise ValueError(
                f"run {run.name!r} config is not a file: {run.config}")
        for kind, path in (("outdir", run.outdir),
                           ("scratch", run.scratch_dir),
                           ("cache", run.cache)):
            if path.exists():
                raise ValueError(
                    f"run {run.name!r} {kind} already exists: {path}; "
                    "multi-run requires new directories so it cannot mix "
                    "with earlier output or temporary state")


def load_plan(path: str | Path, *, summary_override: str | Path | None = None,
              preflight_override: str | None = None) -> MultiRunPlan:
    """Parse and path-check a versioned TOML plan without touching CUDA."""

    plan_path = Path(path).expanduser().resolve()
    if not plan_path.is_file():
        raise ValueError(f"multi-run plan is not a file: {plan_path}")
    try:
        plan_signature_before = _authority_signature(plan_path)
        plan_bytes = plan_path.read_bytes()
        plan_signature_after = _authority_signature(plan_path)
        if plan_signature_before != plan_signature_after:
            raise ValueError(
                f"multi-run plan changed while it was captured: {plan_path}")
        raw = tomllib.loads(plan_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"multi-run plan {plan_path} is invalid TOML: "
                         f"{error}") from error
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    if not isinstance(raw, dict):
        raise ValueError("multi-run plan must be a TOML table")
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"multi-run plan has unknown top-level keys: "
                         f"{unknown}")
    if raw.get("schema") != PLAN_SCHEMA:
        raise ValueError(
            f"multi-run plan schema must be {PLAN_SCHEMA!r}, got "
            f"{raw.get('schema')!r}")

    base = plan_path.parent
    summary_value: object
    if summary_override is not None:
        summary_value = str(summary_override)
    else:
        summary_value = raw.get(
            "summary", f"{plan_path.stem}.summary.json")
    summary = _plan_path(base, summary_value, "summary")
    input_authority_store = summary.with_name(
        f"{summary.name}.input-authorities").resolve()

    preflight_value = (raw.get("preflight", "estimate")
                       if preflight_override is None
                       else preflight_override)
    preflight = _nonempty_string(preflight_value, "preflight")
    if preflight not in PREFLIGHT_MODES:
        raise ValueError(
            f"preflight must be one of {PREFLIGHT_MODES}, got "
            f"{preflight!r}")

    entries = raw.get("run")
    if not isinstance(entries, list) or not entries:
        raise ValueError("multi-run plan must contain at least one [[run]]")
    runs: list[RunSpec] = []
    names: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        label = f"run {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be a TOML table")
        unknown = sorted(set(entry) - _RUN_KEYS)
        if unknown:
            raise ValueError(f"{label} has unknown keys: {unknown}")
        missing = sorted(_COMMON_RUN_KEYS - set(entry))
        if missing:
            raise ValueError(f"{label} is missing required keys: {missing}")
        has_config = "config" in entry
        has_module = "module" in entry
        if has_config == has_module:
            raise ValueError(
                f"{label} must choose exactly one runner form: config for "
                "woof run, or module plus args and inputs")
        if has_config:
            forbidden = sorted(set(entry) & {"module", "args", "inputs"})
            if forbidden:
                raise ValueError(
                    f"{label} config form cannot also set {forbidden}")
        else:
            missing_module = sorted({"args", "inputs"} - set(entry))
            if missing_module:
                raise ValueError(
                    f"{label} module form is missing required keys: "
                    f"{missing_module}")
        name = _nonempty_string(entry["name"], f"{label} name")
        if name in names:
            raise ValueError(f"multi-run name {name!r} is duplicated")
        names.add(name)
        config = (_plan_path(base, entry["config"], f"run {name!r} config")
                  if has_config else None)
        module = (_module_name(entry["module"], f"run {name!r} module")
                  if has_module else None)
        arguments = (_argument_list(entry["args"], f"run {name!r} args")
                     if has_module else ())
        inputs = (_input_paths(base, entry["inputs"],
                               f"run {name!r} inputs")
                  if has_module else (config,))
        if has_module:
            assert module is not None
            _validate_outdir_template(arguments, f"run {name!r} args")
            _validate_runner_input_templates(
                module, arguments, len(inputs), f"run {name!r} args")
            _validate_placeholders(
                arguments, len(inputs), f"run {name!r} args")
        runs.append(RunSpec(
            name=name,
            device_selector=_device_selector(
                entry["device"], f"run {name!r} device"),
            outdir=_plan_path(base, entry["outdir"],
                              f"run {name!r} outdir"),
            scratch_dir=_plan_path(base, entry["scratch"],
                                   f"run {name!r} scratch"),
            cache=_plan_path(base, entry["cache"],
                             f"run {name!r} cache"),
            config=config,
            module=module,
            arguments=arguments,
            inputs=inputs,
        ))
    _validate_path_isolation(
        plan_path, summary, input_authority_store, runs)
    binding_cache: dict[Path, InputBinding] = {}
    config_paths = {run.config for run in runs if run.config is not None}
    bound_runs = []
    for run in runs:
        bindings = []
        for path in run.inputs:
            if path not in binding_cache:
                binding_cache[path] = _bind_input(
                    path, capture_bytes=path in config_paths)
            bindings.append(binding_cache[path])
        bound_runs.append(replace(run, input_bindings=tuple(bindings)))
    return MultiRunPlan(
        plan_path, plan_sha256, plan_signature_after, summary,
        input_authority_store, preflight, tuple(bound_runs))


def resolve_devices(
        plan: MultiRunPlan, inventory: Sequence[GPUIdentity], *,
        lock_root: Path | None = None) -> tuple[PlannedRun, ...]:
    """Resolve every selector and reject aliases to one physical device."""

    by_index = {str(gpu.index): gpu for gpu in inventory
                if gpu.index is not None}
    by_uuid = {gpu.uuid: gpu for gpu in inventory}
    if len(by_uuid) != len(inventory):
        raise ValueError("nvidia-smi reported duplicate GPU UUIDs")
    indexed = [gpu for gpu in inventory if gpu.index is not None]
    if len(by_index) != len(indexed):
        raise ValueError("nvidia-smi reported duplicate GPU indices")
    resolved: list[PlannedRun] = []
    used: dict[str, str] = {}
    for spec in plan.runs:
        gpu = (by_index.get(spec.device_selector)
               if spec.device_selector.isdigit()
               else by_uuid.get(spec.device_selector))
        if gpu is None:
            available = [
                {"index": item.index, "uuid": item.uuid, "name": item.name}
                for item in inventory]
            raise ValueError(
                f"run {spec.name!r} requested device "
                f"{spec.device_selector!r}, not present in {available}")
        previous = used.get(gpu.uuid)
        if previous is not None:
            raise ValueError(
                f"runs {previous!r} and {spec.name!r} both resolve to GPU "
                f"{gpu.uuid}; every run needs a unique physical device")
        used[gpu.uuid] = spec.name
        resolved.append(PlannedRun(
            spec, gpu, lock_root, plan.input_authority_store))
    return tuple(resolved)


def _parent_lock_root(environment: Mapping[str, str]) -> Path:
    """The supervisor lock root before child temp variables are isolated."""

    configured = environment.get(GPU_LOCK_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = (environment.get("PROGRAMDATA")
                or environment.get("TEMP")
                or environment.get("TMP")
                or environment.get("TMPDIR")
                or str(Path.cwd()))
    else:
        base = (environment.get("TMPDIR")
                or environment.get("TEMP")
                or environment.get("TMP")
                or "/tmp")
    return (Path(base).expanduser().resolve() / "woof" / "locks")


def _validate_lock_root_isolation(
        plan: MultiRunPlan, lock_root: Path) -> Path:
    """Prove the shared UUID-lock namespace is outside every run path."""

    root = lock_root.expanduser().resolve()
    protected: list[tuple[str, Path]] = [
        ("plan", plan.path), ("summary", plan.summary)]
    for run in plan.runs:
        protected.extend((
            (f"run {run.name!r} outdir", run.outdir),
            (f"run {run.name!r} scratch", run.scratch_dir),
            (f"run {run.name!r} cache", run.cache),
        ))
        protected.extend(
            (f"run {run.name!r} input {index}", path)
            for index, path in enumerate(run.inputs))
    protected.append(("shared input authority store",
                      plan.input_authority_store))
    for label, path in protected:
        if _same_or_nested(root, path):
            raise ValueError(
                f"shared GPU lock root {root} overlaps {label} {path}; set "
                f"{GPU_LOCK_ROOT_ENV} to a stable machine-wide directory "
                "outside every plan, input, summary, output, scratch, and "
                "cache path")
    root.mkdir(parents=True, exist_ok=True)
    return root


def child_environment(run: PlannedRun,
                      parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy the parent environment and isolate only device/temp/cache state."""

    from woof.config_authority import (CONFIG_PAYLOAD_ENV,
                                        CONFIG_SHA256_ENV,
                                        CONFIG_SOURCE_ENV)

    environment = dict(os.environ if parent is None else parent)
    for name in (CONFIG_PAYLOAD_ENV, CONFIG_SOURCE_ENV, CONFIG_SHA256_ENV):
        environment.pop(name, None)
    lock_root = (run.lock_root if run.lock_root is not None
                 else _parent_lock_root(environment))
    scratch = str(run.spec.scratch_dir)
    environment.update({
        **__import__("woof.gpu_smi",fromlist=["visibility_environment"]).visibility_environment(run.gpu.uuid),
        "CUDA_CACHE_PATH": str(run.spec.cache / "cuda"),
        "CUPY_CACHE_DIR": str(run.spec.cache / "cupy"),
        GPU_LOCK_ROOT_ENV: str(lock_root),
        "TEMP": scratch,
        "TMP": scratch,
        "TMPDIR": scratch,
    })
    if run.input_authority_store is not None:
        environment[SHARED_INPUT_AUTHORITY_ROOT_ENV] = str(
            run.input_authority_store)
    if run.spec.config is not None:
        from woof.config_authority import authority_environment

        binding = run.spec.input_bindings[0]
        assert binding.sha256 is not None
        environment.update(authority_environment(
            source=run.spec.config,
            payload_path=run.spec.captured_config,
            sha256=binding.sha256))
    return environment


def _check_command(run: PlannedRun, mode: str) -> tuple[str, ...]:
    if run.spec.config is None:
        raise ValueError(
            f"run {run.spec.name!r} uses a production module, which owns "
            "its own preflight")
    return (
        sys.executable, "-m", "woof.multi_run", "check-worker",
        "--gpu-uuid", run.gpu.uuid, "--config", str(run.spec.config),
        "--mode", mode,
    )


def _render_arguments(spec: RunSpec) -> tuple[str, ...]:
    replacements = {
        "{outdir}": str(spec.outdir),
        "{scratch}": str(spec.scratch_dir),
        "{cache}": str(spec.cache),
    }
    replacements.update({
        f"{{input{index}}}": str(path)
        for index, path in enumerate(spec.inputs)
    })
    rendered = []
    for argument in spec.arguments:
        value = argument
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        if "{input" in value:
            raise ValueError(
                f"run {spec.name!r} argument {argument!r} refers to an "
                "input placeholder that was not declared")
        rendered.append(value)
    return tuple(rendered)


def _run_command(run: PlannedRun) -> tuple[str, ...]:
    if run.spec.config is not None:
        return (
            sys.executable, "-m", "woof.cli", "run", str(run.spec.config),
            "--outdir", str(run.spec.outdir), "--gpu-uuid", run.gpu.uuid,
        )
    assert run.spec.module is not None
    return (
        sys.executable, "-m", "woof.multi_run", "worker",
        "--gpu-uuid", run.gpu.uuid,
        "--module", run.spec.module,
        "--outdir", str(run.spec.outdir),
        *(item for path in run.spec.inputs
          for item in ("--input", str(path))),
        "--", *_render_arguments(run.spec),
    )


def _worker_reservation_bytes(module: str,
                              arguments: Sequence[str]) -> int | None:
    """This worker's priced peak envelope, or None when it cannot be priced.

    Read off the prepared runner's own ``--experiment-config`` value, which
    the argument contract above has already pinned to one declared input,
    and priced through :func:`woof.supervisor.priced_reservation_bytes` --
    the same function ``woof run`` and ``woof stream`` price a shared
    card from.  ONE function, every door: a multi-run worker that priced
    its own reservation differently would be admitted onto a card the
    supervisor it wraps refuses, or refused off one that supervisor admits.

    Unpriceable is None, never a number and never a refusal: an unpriced
    run is admitted against the device's measured free memory and says so.

    Priced with the source the runner forecasts from, so the analysed
    hydrometeor tables that source puts on the root's boundary are in the
    reservation (:func:`woof.stage_cli.boundary_pricing_source`: the
    single-domain runner's own ``--source``, else the one its prepared
    root names).  Without it a multi-run of HRRR-forced forecasts reserved
    a shared card short by those tables.
    Verified prepared land cover supplies the urban column counts; an
    unreadable prepared bundle retains the configuration's workspace bound.
    """

    values = _runner_path_values(module, arguments, "worker arguments")
    config = values.get("--experiment-config")
    if config is None:
        return None
    from woof.stage_cli import boundary_pricing_source

    root = values.get("--prepared-root")
    source = _runner_option_value(arguments, "--source")
    pricing = {"source": (source if root is None else
                          boundary_pricing_source(Path(root), source))}
    if root is not None:
        from woof.core.preflight import _prepared_check_inputs

        wps = values.get("--wps-namelist")
        try:
            prepared = _prepared_check_inputs(argparse.Namespace(
                config=Path(config), prepared_root=Path(root),
                wps_namelist=None if wps is None else Path(wps)))
        except (OSError, ValueError, RuntimeError, ImportError):
            # Reservation pricing is advisory. The runner verifies the
            # bundle before restore; unverified land cover keeps the bound.
            prepared = None
        if prepared is not None and prepared.urban_columns is not None:
            pricing["urban_columns"] = prepared.urban_columns
    return priced_reservation_bytes(Path(config), **pricing)


def _runner_option_value(arguments: Sequence[str], flag: str) -> str | None:
    """The value a runner's argument list gives ``flag``, or ``None``."""

    for index, argument in enumerate(arguments):
        if argument == flag and index + 1 < len(arguments):
            return str(arguments[index + 1])
        if argument.startswith(f"{flag}="):
            return argument.split("=", 1)[1]
    return None


def _locked_module_main(*, gpu_uuid: str, module_name: str, outdir: Path,
                        inputs: Sequence[Path],
                        arguments: Sequence[str]) -> int:
    """Run one production module while holding the existing UUID lock."""

    module_name = _module_name(module_name, "worker module")
    _validate_rendered_outdir(arguments, outdir, "worker arguments")
    _validate_rendered_runner_inputs(
        module_name, arguments, inputs, "worker arguments")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != gpu_uuid:
        raise RuntimeError(
            "multi-run worker device mask does not match its physical UUID: "
            f"CUDA_VISIBLE_DEVICES={visible!r}, UUID={gpu_uuid!r}")
    reservation_bytes = _worker_reservation_bytes(module_name, arguments)
    if outdir.exists():
        raise ValueError(
            f"multi-run worker outdir already exists: {outdir}; refusing "
            "to mix production outputs")
    with GPUFileLock(gpu_uuid, run_id=f"multi-run-{os.getpid()}"):
        # Another invocation can finish while this worker waits for the UUID
        # lock.  Recheck inside the lock before importing any CUDA-facing
        # target; prepared runners also retain their own atomic mkdir claim.
        if outdir.exists():
            raise ValueError(
                f"multi-run worker outdir already exists: {outdir}; "
                "refusing to mix production outputs")
        # Priced against THIS worker's own configuration, so a co-tenant
        # that fits beside it is admitted here exactly as it is at
        # ``woof run``, and one that does not is refused at both.
        preflight_exclusive_gpu(
            gpu_uuid, approved_pids={os.getpid()},
            reservation_bytes=reservation_bytes)
        module = importlib.import_module(module_name)
        target = getattr(module, "main", None)
        if not callable(target):
            raise ValueError(
                f"production module {module_name!r} has no callable main")
        result = target(list(arguments))
    return 0 if result is None else int(result)


def _locked_check_main(*, gpu_uuid: str, config: Path, mode: str) -> int:
    """Run the full config check inside its selected UUID lock and mask."""

    if mode not in PREFLIGHT_MODES[:2]:
        raise ValueError(
            f"check-worker mode must be estimate or alloc, got {mode!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != gpu_uuid:
        raise RuntimeError(
            "multi-run check-worker device mask does not match its physical "
            f"UUID: CUDA_VISIBLE_DEVICES={visible!r}, UUID={gpu_uuid!r}")
    reservation_bytes = priced_reservation_bytes(config)
    with GPUFileLock(gpu_uuid, run_id=f"multi-run-check-{os.getpid()}"):
        # Plan review prices the card from the configuration it is
        # reviewing, because a review that cleared a shared card the run
        # door then refused would be a plan review that reviewed a
        # different question than the run.
        preflight_exclusive_gpu(
            gpu_uuid, approved_pids={os.getpid()},
            reservation_bytes=reservation_bytes)
        cli = importlib.import_module("woof.cli")
        command = ["check", str(config)]
        if mode == "alloc":
            command.append("--alloc")
        result = cli.main(command)
    return 0 if result is None else int(result)


def _module_entry(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.multi_run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--gpu-uuid", required=True)
    worker.add_argument("--module", required=True)
    worker.add_argument("--outdir", type=Path, required=True)
    worker.add_argument("--input", type=Path, action="append", required=True)
    worker.add_argument("arguments", nargs=argparse.REMAINDER)
    check_worker = subparsers.add_parser("check-worker")
    check_worker.add_argument("--gpu-uuid", required=True)
    check_worker.add_argument("--config", type=Path, required=True)
    check_worker.add_argument(
        "--mode", choices=PREFLIGHT_MODES[:2], required=True)
    args = parser.parse_args(argv)
    if args.command == "check-worker":
        return _locked_check_main(
            gpu_uuid=args.gpu_uuid, config=args.config.resolve(),
            mode=args.mode)
    arguments = list(args.arguments)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    return _locked_module_main(
        gpu_uuid=args.gpu_uuid, module_name=args.module,
        outdir=args.outdir.resolve(),
        inputs=tuple(path.resolve() for path in args.input),
        arguments=arguments)


def _new_process_group_kwargs() -> dict[str, object]:
    """Keep a terminal interrupt in the parent from signalling children."""

    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _execute_group(
        runs: Sequence[PlannedRun], *, phase: str,
        command_for: Callable[[PlannedRun], tuple[str, ...]],
        log_for: Callable[[PlannedRun], Path],
        popen: Callable[..., subprocess.Popen] | None = None,
        poll_seconds: float = 0.10) -> GroupOutcome:
    """Launch every child before waiting, then report completion live."""

    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be positive")
    group_started_at_utc = utc_now()
    group_started_monotonic = time.monotonic()
    launch = subprocess.Popen if popen is None else popen
    records: list[ProcessRecord] = []
    active: list[tuple[ProcessRecord, subprocess.Popen, object]] = []
    launch_index = -1
    pending_record: ProcessRecord | None = None
    pending_process: subprocess.Popen | None = None
    pending_stream = None
    try:
        for run in runs:
            records.append(ProcessRecord(
                run.spec.name, phase, command_for(run), log_for(run)))
        for launch_index, (run, record) in enumerate(zip(runs, records)):
            pending_record = record
            record.started_at_utc = utc_now()
            record._started_monotonic = time.monotonic()
            stream = None
            try:
                stream = record.log.open("xb")
                pending_stream = stream
                process = launch(
                    list(record.command), env=child_environment(run),
                    stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, close_fds=True,
                    **_new_process_group_kwargs())
                pending_process = process
            except OSError as error:
                if stream is not None:
                    stream.close()
                pending_record = None
                pending_stream = None
                record.status = "launch_failed"
                record.completed_at_utc = utc_now()
                record.duration_seconds = round(
                    time.monotonic() - record._started_monotonic, 6)
                record.error = f"{type(error).__name__}: {error}"
                print(
                    f"multi-run {phase} {record.name}: launch failed; "
                    f"see {record.log}", file=sys.stderr, flush=True)
                continue
            record.pid = process.pid
            record.status = "running"
            active.append((record, process, stream))
            pending_record = None
            pending_process = None
            pending_stream = None
            print(
                f"multi-run {phase} {record.name}: started "
                f"{record.started_at_utc} pid={record.pid} "
                f"device={run.gpu.uuid} log={record.log}", flush=True)

        while active:
            remaining: list[tuple[ProcessRecord, subprocess.Popen, object]] = []
            for record, process, stream in active:
                exit_code = process.poll()
                if exit_code is None:
                    remaining.append((record, process, stream))
                    continue
                record.exit_code = int(exit_code)
                record.status = "succeeded" if exit_code == 0 else "failed"
                record.completed_at_utc = utc_now()
                record.duration_seconds = round(
                    time.monotonic() - record._started_monotonic, 6)
                stream.close()
                print(
                    f"multi-run {phase} {record.name}: completed "
                    f"{record.completed_at_utc} exit={record.exit_code} "
                    f"duration={record.duration_seconds:.3f}s "
                    f"log={record.log}", flush=True)
            active = remaining
            if active:
                time.sleep(poll_seconds)
        group_completed_at_utc = utc_now()
        return GroupOutcome(
            tuple(records), False, group_started_at_utc,
            group_completed_at_utc,
            round(time.monotonic() - group_started_monotonic, 6))
    except KeyboardInterrupt:
        observed_at = utc_now()
        now = time.monotonic()
        if (pending_record is not None
                and any(record is pending_record
                        for record, _process, _stream in active)):
            pending_record = None
            pending_process = None
            pending_stream = None
        if pending_stream is not None:
            pending_stream.close()
        if pending_record is not None:
            pending_record.status = (
                "launch_unobserved" if pending_process is None
                else "running_unobserved")
            pending_record.pid = (None if pending_process is None
                                  else pending_process.pid)
            pending_record.duration_seconds = round(
                now - (pending_record._started_monotonic or now), 6)
        for record, _process, stream in active:
            stream.close()
            record.status = "running_unobserved"
            record.duration_seconds = round(
                now - (record._started_monotonic or now), 6)
        not_started_from = launch_index + 1
        for record in records[not_started_from:]:
            record.status = "not_started"
        pids = [record.pid for record, _process, _stream in active]
        if pending_record is not None:
            pids.append(pending_record.pid)
        print(
            f"multi-run {phase}: interrupted at {observed_at}; child "
            f"processes were not terminated, unobserved pids={pids}",
            file=sys.stderr, flush=True)
        return GroupOutcome(
            tuple(records), True, group_started_at_utc, observed_at,
            round(now - group_started_monotonic, 6))
    except BaseException as error:
        # Preserve everything known when a non-interrupt failure escapes
        # orchestration (including a second Popen raising after the first
        # child was launched).  Child ownership is deliberately untouched:
        # no terminate/kill is safe when a child may itself supervise a
        # forecast worker.
        observed_at = utc_now()
        now = time.monotonic()
        if (pending_record is not None
                and any(record is pending_record
                        for record, _process, _stream in active)):
            pending_record = None
            pending_process = None
            pending_stream = None
        if pending_stream is not None:
            try:
                pending_stream.close()
            except BaseException:
                pass
        if pending_record is not None:
            pending_record.status = (
                "launch_unobserved" if pending_process is None
                else "running_unobserved")
            if pending_process is not None:
                try:
                    pending_record.pid = pending_process.pid
                except BaseException:
                    pending_record.pid = None
            pending_record.duration_seconds = round(
                now - (pending_record._started_monotonic or now), 6)
            pending_record.error = f"{type(error).__name__}: {error}"
        for record, _process, stream in active:
            try:
                stream.close()
            except BaseException:
                pass
            if record.status not in {"succeeded", "failed"}:
                record.status = "running_unobserved"
                record.duration_seconds = round(
                    now - (record._started_monotonic or now), 6)
        for record in records[launch_index + 1:]:
            record.status = "not_started"
        outcome = GroupOutcome(
            tuple(records), False, group_started_at_utc, observed_at,
            round(now - group_started_monotonic, 6))
        pids = [record.pid for record in records
                if record.status == "running_unobserved"
                and record.pid is not None]
        print(
            f"multi-run {phase}: orchestration failed at {observed_at}; "
            f"child processes were not terminated, unobserved pids={pids}",
            file=sys.stderr, flush=True)
        raise GroupExecutionError(phase, outcome, error) from error


def _prepare_directories(
        plan: MultiRunPlan,
        state: OrchestrationState | None = None) -> None:
    """Claim config-run outputs and prepare every isolated runtime root."""

    try:
        plan.input_authority_store.mkdir(parents=True, exist_ok=False)
        if state is not None:
            state.claimed_directories.append(plan.input_authority_store)
    except OSError as error:
        raise ValueError(
            "could not create shared input authority store "
            f"{plan.input_authority_store}: {error}; no child was launched") \
            from error

    for run in plan.runs:
        directories = [
            ("scratch", run.scratch_dir),
            ("cache", run.cache),
            ("CuPy cache", run.cache / "cupy"),
            ("CUDA cache", run.cache / "cuda"),
        ]
        # ``woof run`` accepts an existing output directory, so creating it
        # is an atomic claim. Prepared production runners intentionally claim
        # their own absent --outdir; leave it absent for that contract.
        if run.is_config_run:
            directories.insert(0, ("outdir", run.outdir))
        for kind, path in directories:
            try:
                path.mkdir(parents=True, exist_ok=False)
                if state is not None:
                    state.claimed_directories.append(path)
            except OSError as error:
                raise ValueError(
                    f"could not create run {run.name!r} {kind} {path}: "
                    f"{error}; no child was launched") from error
        if run.is_config_run:
            binding = run.input_bindings[0]
            if binding.captured_bytes is None or binding.sha256 is None:
                raise RuntimeError(
                    f"run {run.name!r} has no captured config authority")
            try:
                with run.captured_config.open("xb") as stream:
                    stream.write(binding.captured_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as error:
                raise ValueError(
                    f"could not publish run {run.name!r} captured config "
                    f"{run.captured_config}: {error}; no child was "
                    "launched") from error


def _prepare_summary_destination(
        path: Path, state: OrchestrationState | None = None) -> None:
    """Prove create-only atomic publication works before any child launch."""

    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed and state is not None:
        state.claimed_directories.append(path.parent)
    if path.exists():
        raise ValueError(
            f"multi-run summary {path} already exists; refusing to "
            "overwrite it")
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    source = path.with_name(f".{path.name}.{token}.link-source")
    destination = path.with_name(f".{path.name}.{token}.link-destination")
    source_created = False
    destination_created = False
    try:
        with source.open("x", encoding="utf-8", newline="\n") as stream:
            source_created = True
            stream.write("woof multi-run create-only publication probe\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_new(source, destination)
        destination_created = True
        if state is not None:
            state.summary_capable = True
    except OSError as error:
        raise ValueError(
            f"multi-run summary destination {path.parent} does not support "
            f"atomic create-only publication: {error}; no child was "
            "launched") from error
    finally:
        if destination_created:
            destination.unlink()
        if source_created:
            # publish_new renames the source where the volume has no hard links.
            source.unlink(missing_ok=True)
    if path.exists():
        raise ValueError(
            f"multi-run summary appeared during destination preparation: "
            f"{path}; refusing to overwrite it")


def _write_summary(path: Path, payload: Mapping[str, object]) -> Path:
    """Durably publish JSON create-only (``publish_new``)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        encoded = json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # The temporary is a fully written sibling on the same volume.
            # Publishing it is atomic and create-only (a hard link, or a
            # no-replace rename where the volume has no links): unlike
            # os.replace, a summary created by a racing invocation is never
            # overwritten.
            publish_new(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"multi-run summary appeared before publication: {path}; "
                "refusing to overwrite it") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


class FileAuthorityMonitor:
    """Sticky periodic mutation detector spanning every launched child."""

    def __init__(self, plan: MultiRunPlan, *, poll_seconds: float):
        if poll_seconds <= 0.0:
            raise ValueError("authority monitor poll_seconds must be positive")
        self.poll_seconds = poll_seconds
        authorities: list[
            tuple[str, Path, str, tuple[int, int, int, int, int] | None]
        ] = [("plan", plan.path, plan.sha256, plan.captured_signature)]
        seen = {os.path.normcase(str(plan.path))}
        for run in plan.runs:
            for binding in run.input_bindings:
                if binding.sha256 is None:
                    continue
                key = os.path.normcase(str(binding.path))
                if key in seen:
                    continue
                seen.add(key)
                authorities.append((
                    "input", binding.path, binding.sha256,
                    binding.captured_signature))
            if run.is_config_run and run.captured_config.is_file():
                binding = run.input_bindings[0]
                assert binding.sha256 is not None
                authorities.append((
                    "captured_config", run.captured_config, binding.sha256,
                    self._signature(run.captured_config)))
        self._records = [{
            "authority": authority,
            "baseline_signature": (
                None if signature is None else list(signature)),
            "expected_sha256": expected,
            "last_signature": signature,
            "observations": [],
            "path": str(path),
        } for authority, path, expected, signature in authorities]
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at_utc: str | None = None
        self._stopped_at_utc: str | None = None
        self._poll_count = 0
        self._monitor_errors: list[dict[str, str]] = []

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int, int, int]:
        metadata = path.stat()
        if not path.is_file():
            raise OSError("path is no longer a regular file")
        return (
            int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_size),
            int(metadata.st_mtime_ns), int(metadata.st_ctime_ns))

    def _record_failure(self, error: BaseException) -> None:
        with self._lock:
            self._monitor_errors.append({
                "error": f"{type(error).__name__}: {error}",
                "observed_at_utc": utc_now(),
            })
        self._failed.set()

    def _observe(self, event: str, *, force_hash: bool) -> None:
        observed_at = utc_now()
        for record in self._records:
            path = Path(record["path"])
            signature = None
            observed_sha256 = None
            error_text = None
            try:
                signature = self._signature(path)
                changed = (record["last_signature"] is not None
                           and signature != record["last_signature"])
                # Content was read exactly once during plan capture.  After
                # that boundary, the original path is metadata-only: polling
                # its sticky inode/size/mtime/ctime signature detects even a
                # same-size mutate/use/restore without reopening mutable
                # bytes.  Children consume the separate captured payload.
            except OSError as error:
                changed = True
                error_text = f"{type(error).__name__}: {error}"
            first = not record["observations"]
            mismatch = False
            if first or changed or force_hash or error_text is not None:
                observation: dict[str, object] = {
                    "event": event if not changed else "metadata_changed",
                    "observed_at_utc": observed_at,
                    "observed_sha256": observed_sha256,
                    "signature": (
                        None if signature is None else list(signature)),
                    "status": (
                        "CHANGED" if changed or mismatch or error_text
                        else "MATCH"),
                }
                if error_text is not None:
                    observation["error"] = error_text
                with self._lock:
                    record["observations"].append(observation)
            if (record["baseline_signature"] is None
                    and signature is not None):
                record["baseline_signature"] = list(signature)
            record["last_signature"] = signature
            if changed or mismatch or error_text is not None:
                self._failed.set()

    def _run(self) -> None:
        self._started_at_utc = utc_now()
        try:
            self._observe("monitor_started", force_hash=True)
        except BaseException as error:  # keep failures observable to parent
            self._record_failure(error)
        finally:
            self._ready.set()
        while not self._stop.wait(self.poll_seconds):
            try:
                self._poll_count += 1
                self._observe("poll", force_hash=False)
            except BaseException as error:
                self._record_failure(error)

    def start(self) -> "FileAuthorityMonitor":
        if self._thread is not None:
            raise RuntimeError("file authority monitor was already started")
        self._thread = threading.Thread(
            target=self._run, name="gpuwm-file-authority-monitor",
            daemon=False)
        self._thread.start()
        self._ready.wait()
        return self

    def wait_for_failure(self, timeout: float) -> bool:
        return self._failed.wait(timeout)

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def stop(self) -> dict[str, object]:
        if self._thread is None:
            raise RuntimeError("file authority monitor was not started")
        self._stop.set()
        self._thread.join()
        try:
            self._observe("monitor_stopped", force_hash=True)
        except BaseException as error:
            self._record_failure(error)
        self._stopped_at_utc = utc_now()
        return self.result()

    def result(self) -> dict[str, object]:
        with self._lock:
            files = []
            for record in self._records:
                observations = list(record["observations"])
                files.append({
                    "authority": record["authority"],
                    "baseline_signature": record["baseline_signature"],
                    "expected_sha256": record["expected_sha256"],
                    "observations": observations,
                    "path": record["path"],
                    "status": (
                        "PASS" if observations
                        and all(item["status"] == "MATCH"
                                for item in observations)
                        else "FAIL"),
                })
            errors = list(self._monitor_errors)
        return {
            "errors": errors,
            "files": files,
            "poll_count": self._poll_count,
            "poll_seconds": self.poll_seconds,
            "schema": "gpuwm.file-authority-monitor/v1",
            "started_at_utc": self._started_at_utc,
            "status": "FAIL" if self.failed else "PASS",
            "stopped_at_utc": self._stopped_at_utc,
        }


def _summary_payload(
        plan: MultiRunPlan, runs: Sequence[PlannedRun], *,
        started_at_utc: str, completed_at_utc: str,
        duration_seconds: float, status: str, exit_code: int,
        preflight: GroupOutcome | None,
        execution: GroupOutcome | None,
        file_verification: Mapping[str, object],
        orchestration: Mapping[str, object]) -> dict[str, object]:
    preflight_by_name = (
        {} if preflight is None
        else {record.name: record for record in preflight.records})
    execution_by_name = (
        {} if execution is None
        else {record.name: record for record in execution.records})
    rows = []
    for run in runs:
        spec = run.spec
        process = execution_by_name.get(spec.name)
        rows.append({
            "arguments": list(spec.arguments),
            "cache": str(spec.cache),
            "config": None if spec.config is None else str(spec.config),
            "config_sha256": (
                None if spec.config is None
                else spec.input_bindings[0].sha256),
            "device": {
                "index": run.gpu.index,
                "name": run.gpu.name,
                "selector": spec.device_selector,
                "uuid": run.gpu.uuid,
            },
            "log": str(spec.log),
            "gpu_lock_root": (
                None if run.lock_root is None else str(run.lock_root)),
            "module": spec.module,
            "name": spec.name,
            "outdir": str(spec.outdir),
            "input_authorities": [
                binding.as_dict() for binding in spec.input_bindings],
            "shared_input_authority_store": str(
                plan.input_authority_store),
            "read_only_inputs": [str(path) for path in spec.inputs],
            "preflight": (
                None if spec.name not in preflight_by_name
                else preflight_by_name[spec.name].as_dict()),
            "process": (
                {"status": "not_started", "exit_code": None,
                 "log": str(spec.log)}
                if process is None else process.as_dict()),
            "scratch": str(spec.scratch_dir),
        })
    execution_timing = _execution_timing(execution)
    if file_verification.get("status") != "PASS":
        execution_timing["overlap_ratio"] = None
        execution_timing["overlap_ratio_basis"] = (
            "unavailable: a declared file authority changed")
    return {
        "completed_at_utc": completed_at_utc,
        "duration_seconds": duration_seconds,
        "execution_timing": execution_timing,
        "exit_code": exit_code,
        "file_authority_verification": dict(file_verification),
        "input_authority_store": _input_authority_store_record(plan),
        "orchestration": dict(orchestration),
        "plan": str(plan.path),
        "plan_sha256": plan.sha256,
        "preflight_mode": plan.preflight,
        "runs": rows,
        "schema": SUMMARY_SCHEMA,
        "started_at_utc": started_at_utc,
        "status": status,
    }


def _execution_timing(
        execution: GroupOutcome | None) -> dict[str, object]:
    """Summarize overlap without claiming speedup from incomplete work."""

    if execution is None:
        return {
            "concurrent_wall_seconds": None,
            "overlap_ratio": None,
            "overlap_ratio_basis": "unavailable: execution not started",
            "observed_child_duration_sum_seconds": 0.0,
            "completed_child_duration_sum_seconds": None,
            "window_completed_at_utc": None,
            "window_started_at_utc": None,
        }
    durations = [
        record.duration_seconds for record in execution.records
        if record.duration_seconds is not None]
    observed_sum = round(sum(durations), 6)
    all_forecast_attempts_completed = bool(execution.records) and all(
        record.status in {"succeeded", "failed"}
        and record.duration_seconds is not None
        for record in execution.records)
    completed_sum = (
        observed_sum if all_forecast_attempts_completed else None)
    all_succeeded = bool(execution.records) and all(
        record.status == "succeeded" and record.exit_code == 0
        for record in execution.records)
    if execution.interrupted:
        overlap_ratio = None
        basis = "unavailable: execution was interrupted"
    elif not all_succeeded:
        overlap_ratio = None
        basis = "unavailable: not every forecast succeeded"
    elif completed_sum is None:
        overlap_ratio = None
        basis = "unavailable: one or more forecast durations were incomplete"
    elif execution.wall_seconds is None or execution.wall_seconds <= 0.0:
        overlap_ratio = None
        basis = "unavailable: concurrent execution window was not measured"
    else:
        overlap_ratio = round(completed_sum / execution.wall_seconds, 6)
        basis = (
            "completed_child_duration_sum_seconds / "
            "concurrent_wall_seconds; this measures process overlap, not "
            "performance against a serial baseline")
    return {
        "concurrent_wall_seconds": execution.wall_seconds,
        "overlap_ratio": overlap_ratio,
        "overlap_ratio_basis": basis,
        "observed_child_duration_sum_seconds": observed_sum,
        "completed_child_duration_sum_seconds": completed_sum,
        "window_completed_at_utc": execution.completed_at_utc,
        "window_started_at_utc": execution.started_at_utc,
    }


def _unstarted_authority_verification(reason: str) -> dict[str, object]:
    """Describe accurately why no continuous authority monitor was started."""

    return {
        "errors": [],
        "files": [],
        "poll_count": 0,
        "poll_seconds": AUTHORITY_MONITOR_POLL_SECONDS,
        "reason": reason,
        "schema": "gpuwm.file-authority-monitor/v1",
        "started_at_utc": None,
        "status": "NOT_STARTED",
        "stopped_at_utc": None,
    }


def multi_run_main(args: argparse.Namespace) -> int:
    """CLI implementation for ``woof multi-run PLAN.toml``."""

    state = OrchestrationState()
    started_at_utc = utc_now()
    started_monotonic = time.monotonic()
    plan: MultiRunPlan | None = None
    runs: tuple[PlannedRun, ...] = ()
    preflight: GroupOutcome | None = None
    execution: GroupOutcome | None = None
    monitor: FileAuthorityMonitor | None = None
    verification: Mapping[str, object] = _unstarted_authority_verification(
        "plan and runtime paths have not yet been prepared")

    def stop_monitor() -> Mapping[str, object]:
        nonlocal verification
        if monitor is None:
            return verification
        if verification.get("stopped_at_utc") is None:
            verification = monitor.stop()
        return verification

    def publish(*, status: str, exit_code: int,
                receipt_stage: str) -> int:
        assert plan is not None
        completed_at_utc = utc_now()
        state.stage = receipt_stage
        payload = _summary_payload(
            plan, runs, started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            duration_seconds=round(
                time.monotonic() - started_monotonic, 6),
            status=status, exit_code=exit_code,
            preflight=preflight, execution=execution,
            file_verification=verification,
            orchestration=_orchestration_record(
                state, plan, preflight, execution))
        state.stage = "summary_publication"
        try:
            _write_summary(plan.summary, payload)
        except (OSError, ValueError) as error:
            print(
                f"multi-run: summary publication failed; any summary "
                f"written by another invocation was preserved: {error}",
                file=sys.stderr, flush=True)
            return 130 if exit_code == 130 else 1
        print(f"multi-run: summary {plan.summary}", flush=True)
        return exit_code

    try:
        state.stage = "plan_load"
        plan = load_plan(
            args.plan, summary_override=getattr(args, "summary", None),
            preflight_override=getattr(args, "preflight", None))

        state.stage = "lock_root_resolution"
        lock_root = _validate_lock_root_isolation(
            plan, _parent_lock_root(os.environ))

        state.stage = "device_discovery"
        try:
            inventory = query_gpus()
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        runs = resolve_devices(plan, inventory, lock_root=lock_root)

        state.stage = "summary_probe"
        _prepare_summary_destination(plan.summary, state)
        state.stage = "directory_claim"
        _prepare_directories(plan, state)

        state.stage = "authority_monitor_start"
        monitor = FileAuthorityMonitor(
            plan, poll_seconds=AUTHORITY_MONITOR_POLL_SECONDS)
        monitor.start()
        verification = monitor.result()
        if monitor.failed:
            verification = stop_monitor()
            print(
                "multi-run: declared file authority changed before any "
                "child was launched", file=sys.stderr, flush=True)
            return publish(
                status="failed", exit_code=1,
                receipt_stage="authority_monitor_start")

        if plan.preflight == "off":
            print(
                "WARNING: multi-run preflight is off; launching without a "
                "woof check sizing report.", file=sys.stderr, flush=True)
        else:
            state.stage = "config_checks"
            checkable = tuple(run for run in runs if run.spec.is_config_run)
            for run in runs:
                if not run.spec.is_config_run:
                    print(
                        f"multi-run check {run.spec.name}: delegated to "
                        f"production module {run.spec.module}; its own input "
                        "and memory gates run inside the locked child.",
                        flush=True)
            if checkable:
                preflight = _execute_group(
                    checkable, phase="check",
                    command_for=lambda run: _check_command(
                        run, plan.preflight),
                    log_for=lambda run: run.spec.preflight_log)
            else:
                preflight = GroupOutcome((), False)
            if preflight.interrupted:
                verification = stop_monitor()
                return publish(
                    status="interrupted", exit_code=130,
                    receipt_stage="config_checks")
            for record in preflight.records:
                if record.status != "succeeded":
                    print(
                        f"WARNING: woof check for {record.name!r} reported "
                        f"{record.status} (exit={record.exit_code}); launch "
                        f"continues because check reports sizing and does "
                        f"not gate woof run. See {record.log}.",
                        file=sys.stderr, flush=True)

        if monitor.failed:
            verification = stop_monitor()
            print(
                "multi-run: declared file authority changed while config "
                "checks ran; no forecast child was launched",
                file=sys.stderr, flush=True)
            return publish(
                status="failed", exit_code=1,
                receipt_stage="config_checks")

        state.stage = "forecast"
        execution = _execute_group(
            runs, phase="run", command_for=_run_command,
            log_for=lambda run: run.spec.log)

        state.stage = "authority_monitor_stop"
        verification = stop_monitor()
        if execution.interrupted:
            exit_code = 130
            status = "interrupted"
        else:
            succeeded = all(
                record.status == "succeeded" and record.exit_code == 0
                for record in execution.records)
            exit_code = 0 if succeeded else 1
            check_warning = (preflight is not None and any(
                record.status != "succeeded"
                for record in preflight.records))
            status = (
                "complete_with_warnings" if succeeded and check_warning
                else "complete" if succeeded else "failed")
        if verification["status"] != "PASS":
            print(
                "multi-run: declared file authority changed while children "
                "ran; receipt marks the execution failed",
                file=sys.stderr, flush=True)
            if exit_code != 130:
                exit_code = 1
                status = "failed"

        published = publish(
            status=status, exit_code=exit_code,
            receipt_stage=("forecast" if exit_code == 130 else "complete"))
        success_count = sum(
            record.status == "succeeded" for record in execution.records)
        print(
            f"multi-run: {success_count}/{len(execution.records)} runs "
            f"completed successfully; aggregate exit={published}; "
            f"summary {plan.summary}", flush=True)
        return published
    except KeyboardInterrupt:
        interrupted_stage = state.stage
        if monitor is not None:
            try:
                verification = stop_monitor()
            except BaseException as error:
                verification = monitor.result()
                verification = dict(verification)
                errors = list(verification.get("errors", []))
                errors.append({
                    "error": (
                        "monitor stop during interruption failed: "
                        f"{type(error).__name__}: {error}"),
                    "observed_at_utc": utc_now(),
                })
                verification["errors"] = errors
                verification["status"] = "FAIL"
        state.stage = interrupted_stage
        print(
            f"multi-run: interrupted during {interrupted_stage}; child "
            "processes were not terminated", file=sys.stderr, flush=True)
        if (not state.summary_capable or plan is None or not runs):
            print(
                "multi-run: no interrupted summary was published because "
                "create-only summary capability had not been established",
                file=sys.stderr, flush=True)
            return 130
        completed_at_utc = utc_now()
        payload = _summary_payload(
            plan, runs, started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            duration_seconds=round(
                time.monotonic() - started_monotonic, 6),
            status="interrupted", exit_code=130,
            preflight=preflight, execution=execution,
            file_verification=verification,
            orchestration=_orchestration_record(
                state, plan, preflight, execution))
        try:
            _write_summary(plan.summary, payload)
        except (OSError, ValueError) as error:
            print(
                "multi-run: interrupted summary publication failed; any "
                f"summary written by another invocation was preserved: "
                f"{error}", file=sys.stderr, flush=True)
        else:
            print(
                f"multi-run: interrupted summary {plan.summary}",
                file=sys.stderr, flush=True)
        return 130
    except BaseException as error:
        failure_stage = state.stage
        reported_error = (
            error.original if isinstance(error, GroupExecutionError)
            else error)
        if isinstance(error, GroupExecutionError):
            if error.phase == "check":
                preflight = error.outcome
            elif error.phase == "run":
                execution = error.outcome
        state.error = f"{type(reported_error).__name__}: {reported_error}"
        if monitor is not None:
            try:
                verification = stop_monitor()
            except BaseException as monitor_error:
                verification = dict(monitor.result())
                errors = list(verification.get("errors", []))
                errors.append({
                    "error": (
                        "monitor stop after orchestration failure failed: "
                        f"{type(monitor_error).__name__}: {monitor_error}"),
                    "observed_at_utc": utc_now(),
                })
                verification["errors"] = errors
                verification["status"] = "FAIL"
        state.stage = failure_stage
        print(
            f"multi-run: failed during {failure_stage}: "
            f"{type(reported_error).__name__}: {reported_error}",
            file=sys.stderr, flush=True)
        if state.summary_capable and plan is not None and runs:
            try:
                completed_at_utc = utc_now()
                payload = _summary_payload(
                    plan, runs, started_at_utc=started_at_utc,
                    completed_at_utc=completed_at_utc,
                    duration_seconds=round(
                        time.monotonic() - started_monotonic, 6),
                    status="failed", exit_code=1,
                    preflight=preflight, execution=execution,
                    file_verification=verification,
                    orchestration=_orchestration_record(
                        state, plan, preflight, execution))
                _write_summary(plan.summary, payload)
            except BaseException as publication_error:
                print(
                    "multi-run: failed-summary publication also failed; any "
                    "summary written by another invocation was preserved: "
                    f"{type(publication_error).__name__}: "
                    f"{publication_error}", file=sys.stderr, flush=True)
            else:
                print(
                    f"multi-run: failed summary {plan.summary}",
                    file=sys.stderr, flush=True)
        raise
    finally:
        # The monitor is deliberately non-daemon: never let an unexpected
        # exception strand it and keep interpreter shutdown alive.  Cleanup
        # errors are reported but cannot replace the active exception.
        if (monitor is not None and monitor._thread is not None
                and verification.get("stopped_at_utc") is None):
            try:
                verification = monitor.stop()
            except BaseException as cleanup_error:
                print(
                    "multi-run: authority-monitor cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                    file=sys.stderr, flush=True)


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Register the process-level multi-GPU orchestration command."""

    parser = subparsers.add_parser(
        "multi-run",
        help="launch independent production forecasts on unique GPUs")
    parser.add_argument(
        "plan", type=Path, metavar="PLAN.toml",
        help="versioned plan with one or more [[run]] entries")
    parser.add_argument(
        "--summary", type=Path, default=None, metavar="SUMMARY.json",
        help="summary path relative to the plan (default PLAN.summary.json)")
    parser.add_argument(
        "--preflight", choices=PREFLIGHT_MODES, default=None,
        help="override the plan's woof check mode: estimate, alloc, or off")
    parser.set_defaults(func=multi_run_main)


if __name__ == "__main__":
    sys.exit(_module_entry())


__all__ = [
    "PLAN_SCHEMA", "PREFLIGHT_MODES", "SUMMARY_SCHEMA",
    "SUPPORTED_PREPARED_RUNNERS", "GroupExecutionError", "GroupOutcome",
    "MultiRunPlan", "PlannedRun", "ProcessRecord", "RunSpec",
    "child_environment", "load_plan", "multi_run_main", "register_cli",
    "resolve_devices",
]
