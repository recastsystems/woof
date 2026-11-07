"""Fresh-process forecast supervision and durable run progress.

Forecast CUDA work runs in a fresh Python worker.  The supervisor owns the
physical-GPU lock, preflights active compute processes through ``nvidia-smi``,
launches that worker with the selected UUID mask, and watches an atomically
published heartbeat.  A CUDA-fatal condition is terminal for that worker
process.  Recovery, when possible, is a new process restored only from the
most recent durable manifest-valid restart; there is no in-process retry path.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import errno
import hashlib
import io
import itertools
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


from woof.certify.capsule import emit_run_capsule
# The refusal half of a layered message.  A capsule records `str(exc)`
# verbatim, sentinel and all, and the sentinel is promised never to
# reach a terminal -- so anything lifted OUT of a capsule and printed
# has to go through this first.
from woof.explain import split as explain_split
from woof.explain import warn
from woof.filesystem_paths import publish_new
# A killed run's only chance to say anything.  Both processes below are
# front doors that own a whole run, and neither had a signal handler:
# SIGTERM killed them at SIG_DFL with nothing printed.
from woof.signal_report import report_on_signal


HEARTBEAT_SCHEMA = "gpuwm.run-progress/v1"
FAILURE_CAPSULE_SCHEMA = "gpuwm.failure-capsule/v3"

# Capsule schema ids this module recognizes.  v2 only adds the optional
# ``config_text``/``input_text`` verbatim small-text captures; v3 only adds
# ``installed`` (the running distribution's version and import path).
# Neither changes anything an earlier capsule already said, so v1 and v2
# capsules keep reading unchanged (the same additive convention as
# ``gpuwm.preserved-input-set/v2``).
#
# The id doubles as a version tell for support: a capsule states the schema
# the CODE THAT RAN emits, so it identifies the running release even when
# ``git_commit`` reports an enclosing checkout the run never executed.
SUPPORTED_FAILURE_CAPSULE_SCHEMAS = (
    "gpuwm.failure-capsule/v1", "gpuwm.failure-capsule/v2",
    FAILURE_CAPSULE_SCHEMA)

# Cap on each verbatim text capture embedded in the failure capsule.  The
# embedded inputs are the run's own small text files (the experiment TOML,
# the WPS namelist, the Vtable); anything larger is cut at the cap and says
# so, keeping a capsule readable rather than multi-megabyte.
FAILURE_CAPSULE_TEXT_CAP_BYTES = 64 * 1024

# Declared-input roles whose bytes are small text a support reader needs
# verbatim.  Forcing GRIBs, orography NetCDFs, and the geography tree stay
# hash-only.
FAILURE_CAPSULE_TEXT_ROLES = frozenset({"vtable", "wps_namelist"})

HEARTBEAT_NAME = "run-progress.json"
FAILURE_CAPSULE_NAME = "failure-capsule.json"

# Where a supervised worker's own output lands, in --outdir, one pair per
# fresh process: 01 is the first launch and the number increments for each
# recovery attempt.  Named here because the WORKER is where a traceback is
# written -- the parent's terminal carries none of it -- so `woof run
# --help` has to be able to say the file name without re-typing it.
WORKER_STDOUT_NAME = "worker-{attempt:02d}.stdout.log"
WORKER_STDERR_NAME = "worker-{attempt:02d}.stderr.log"

COMPUTE_MEMORY_THRESHOLD_MIB = 64
MICROPHYSICS_TRANSITION_RECEIPT_NAME = "microphysics-transitions.json"

# How a declared *directory* input is bound to a run's identity.  Files are
# always content-hashed; a directory is not, because the static geography
# tree is multi-GB and this runs before every launch.  See
# docs/public/DETERMINISM.md for what each mode does and does not detect.
DIRECTORY_HASH_MODES = ("inventory", "content")
DIRECTORY_HASH_DEFAULT = "inventory"
DIRECTORY_HASH_ENV = "WOOF_DIRECTORY_INPUT_HASH"

# ``woof multi-run`` gives every child an isolated TMPDIR.  The physical-GPU
# lock must remain machine-wide rather than following that per-run temp root,
# so the orchestrator pins this to the parent's ordinary lock directory.
# Direct ``woof run`` calls leave it unset and retain the historical path.
GPU_LOCK_ROOT_ENV = "WOOF_GPU_LOCK_ROOT"
INPUT_AUTHORITIES_ENV = "WOOF_INPUT_AUTHORITIES_JSON"
SHARED_INPUT_AUTHORITY_ROOT_ENV = "WOOF_SHARED_INPUT_AUTHORITY_ROOT"

_SNAPSHOT_FILE_ROLES = frozenset({
    "forcing", "vtable", "wps_namelist", "source_orography",
})

# Windows sharing violations are normally transient (the supervisor, an
# editor, or an indexer has the old publication open without FILE_SHARE_DELETE).
# Retry for at most 0.50 s total, then let durable artifacts fail loudly while
# heartbeat callers quarantine their unique temporary and keep the worker up.
_REPLACE_BACKOFF_SECONDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.19)

#: Heartbeat statuses whose bound is sized from the bytes the record
#: declares (:func:`finalization_stale_threshold_seconds`) rather than from
#: the model step alone.
WORK_SIZED_PREFIXES = ("finalizing:", "writing:")
#: A forecast at a seam (or at its start) waiting for a boundary interval
#: that is not there yet, and why: a source lead not posted yet
#: (``waiting:source``) or the preparation still building it
#: (``waiting:preparation``).  A waiting record is refreshed while the wait
#: lasts, so it says the worker is alive; it is not progress, so it never
#: enters the step-wall history and it is bounded by the wait's own limit
#: (:func:`waiting_stop_reason`), not by the step bound.  The breakage this
#: prevents: a seam wait refreshed only ``progress.json``, so the watchdog
#: reading this heartbeat stopped a forecast waiting on a lead that was on
#: schedule once the wait outlasted max(3 x p99 step, 120 s).
WAITING_PREFIX = "waiting:"
WAITING_STATUSES = ("waiting:source", "waiting:preparation")
_PHASE_PREFIXES = ("preparing:",) + WORK_SIZED_PREFIXES + (WAITING_PREFIX,)
#: The ``wait`` record a waiting heartbeat carries, and only it.
WAIT_RECORD_FIELDS = ("on", "lead", "expected_at", "late_at", "since_utc")
#: How long past a source lead's ``late_at`` a ``waiting:source`` record
#: may stand.  The producer's own late check ends the run first, with the
#: lead named (exit 75); this is the backstop for a producer that did not.
SOURCE_WAIT_GRACE_SECONDS = 120.0
#: The shortest silence that stops a waiting record: the producer silence
#: floor (``woof.ingest.boundary_stream.SILENT_FLOOR_SECONDS``), since a
#: waiting worker refreshes its record every few seconds.
WAIT_SILENCE_FLOOR_SECONDS = 120.0
#: A worker ending one attempt and starting the forecast again in the same
#: process (a head-bound run whose terrain clock moved runs again on its
#: sealed preparation).  From this record on, every record carries
#: ``restart`` (the attempt's number and why the last one ended), and step
#: and model time start again from zero; :func:`_heartbeat_regression`
#: takes a higher attempt as a new start and holds the records of one
#: attempt to the usual rules.  Every record carries it, not this one
#: alone, because a watchdog polling the file can miss one record.  The
#: breakage it prevents: the second attempt's first beats (a restore phase,
#: step 1) read as a worker going backward, and ``woof go``'s watchdog
#: stopped a forecast that was recovering exactly as designed.  It is a
#: ``preparing:`` record, so every reader that knows preparation reads it,
#: and preparation has no deadline.  ``RuntimeHeartbeat.restarting``
#: publishes it and :func:`restart_attempt` calls that hook.
RESTART_STATUS = "preparing:restart"
#: The ``restart`` record every record of a restarted attempt carries.
RESTART_RECORD_FIELDS = ("attempt", "reason")
_TEMP_COUNTER = itertools.count()

_HEARTBEAT_FIELDS = frozenset({
    "schema", "run_id", "config_digest", "pid", "started_at_utc",
    "updated_at_utc", "status", "model_elapsed_seconds", "outer_step",
    "last_durable_wrfout", "last_checkpoint",
})
# Present only on a ``finalizing:`` or ``writing:`` record that declares its
# work (``work_bytes``), on a ``waiting:`` record (``wait``), or on the records
# of a restarted attempt (``restart``), so every other record keeps exactly
# the field set above.
_HEARTBEAT_OPTIONAL_FIELDS = frozenset({"work_bytes", "wait", "restart"})
_CUDA_FATAL_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"device(?: |-)lost", r"cudaErrorDeviceLost", r"illegal(?: memory)? address",
    r"cudaErrorIllegalAddress", r"cuda.error.illegal.address",
    r"unspecified launch failure",
    r"cudaErrorLaunchFailure", r"launch failure", r"context is destroyed",
))


class SupervisorError(RuntimeError):
    """The supervised forecast could not safely continue."""


class GPUPreflightError(SupervisorError):
    """GPU identity/process state could not be proven exclusive."""


class GPUAlreadyLockedError(GPUPreflightError):
    """Another woof supervisor owns the UUID-keyed lock."""


class CheckpointValidationError(SupervisorError):
    """A proposed recovery file is not manifest-valid."""


def utc_now() -> str:
    """UTC ISO-8601 timestamp with an explicit ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability on POSIX; Windows has no dir fsync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: str | Path) -> Path:
    """Flush an already closed file through the OS durability boundary."""
    path = Path(path)
    # Windows CRT _commit needs write access; POSIX accepts a read-only
    # descriptor, so archived parent checkpoints need no write permission.
    with path.open("r+b" if os.name == "nt" else "rb") as stream:
        os.fsync(stream.fileno())
    return path


def unique_temp_path(path: str | Path, *, hidden: bool = False) -> Path:
    """Return a per-writer temporary name (PID + process-local counter)."""
    path = Path(path)
    prefix = "." if hidden else ""
    return path.with_name(
        f"{prefix}{path.name}.tmp.{os.getpid()}.{next(_TEMP_COUNTER)}")


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace after bounded Windows sharing-violation backoff (0.50 s)."""
    for delay in (*_REPLACE_BACKOFF_SECONDS, None):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def replace_file_with_retry(source: str | Path,
                            destination: str | Path) -> Path:
    """Fail-loud bounded-retry atomic replace for durable publications."""
    source = Path(source)
    destination = Path(destination)
    _replace_with_retry(source, destination)
    return destination


def atomic_write_json(path: str | Path, payload: dict[str, Any], *,
                      _before_replace: Callable[[Path], None] | None = None,
                      _quarantine_on_permission_error: bool = False,
                      durable: bool = True) -> Path:
    """Publish JSON through tmp + flush + fsync + ``os.replace``.

    ``durable=False`` keeps the atomic replace (a reader sees the old record
    or the new one, never a torn one) and skips both fsyncs.  It is for a
    record the very next step replaces anyway: the per-step heartbeat.

    ``_before_replace`` is a deterministic kill/fault-injection seam for the
    CPU atomicity test.  Production callers never pass it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = unique_temp_path(path)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    with temp.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        if durable:
            os.fsync(stream.fileno())
    if _before_replace is not None:
        _before_replace(temp)
    try:
        replace_file_with_retry(temp, path)
    except PermissionError:
        if not _quarantine_on_permission_error:
            raise
        # A stale heartbeat is safer than terminating a healthy CUDA worker
        # because a reader held run-progress.json open for >0.50 s.
        quarantine_file(temp, reason="heartbeat-sharing-violation")
        return path
    if durable:
        _fsync_directory(path.parent)
    return path


def quarantine_file(path: str | Path, *, reason: str = "incomplete",
                    quarantine_dir: str | Path | None = None) -> Path | None:
    """Atomically move one orphan/incomplete artifact out of publication."""
    path = Path(path)
    if not path.exists():
        return None
    directory = (Path(quarantine_dir) if quarantine_dir is not None
                 else path.parent / ".quarantine")
    directory.mkdir(parents=True, exist_ok=True)
    safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "-", reason).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = directory / f"{path.name}.{safe_reason}.{stamp}.{os.getpid()}"
    replace_file_with_retry(path, target)
    _fsync_directory(directory)
    return target


def atomic_publish_file(
        final_path: str | Path, producer: Callable[[Path], None],
        validator: Callable[[Path], None], *,
        quarantine_dir: str | Path | None = None) -> Path:
    """Produce, durably validate, and atomically publish one artifact.

    A producer or validator exception can leave only a quarantined temporary
    file; the old final remains unchanged and no incomplete final name is
    exposed.  The wrfout handoff patch uses this exact helper.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp = unique_temp_path(final_path, hidden=True)
    try:
        producer(temp)
        fsync_file(temp)
        validator(temp)
        replace_file_with_retry(temp, final_path)
        _fsync_directory(final_path.parent)
    except BaseException:
        if temp.exists():
            quarantine_file(temp, reason="failed-publication",
                            quarantine_dir=quarantine_dir)
        raise
    return final_path


@dataclass(frozen=True)
class Heartbeat:
    schema: str
    run_id: str
    config_digest: str
    pid: int
    started_at_utc: str
    updated_at_utc: str
    status: str
    model_elapsed_seconds: float
    outer_step: int
    last_durable_wrfout: str | None
    last_checkpoint: str | None
    #: Bytes the worker will write or read before its next record, declared
    #: by a ``finalizing:`` record (a frame to hash, queued frames to drain,
    #: a state to digest) or a ``writing:`` record (a history frame or a
    #: checkpoint written between two model steps).  It sizes that phase's
    #: watchdog bound; see :func:`finalization_stale_threshold_seconds`.
    #: ``None`` everywhere else, and then absent from the published record.
    work_bytes: int | None = None
    #: What a ``waiting:`` record waits on: ``{on, lead, expected_at,
    #: late_at, since_utc}`` (:data:`WAIT_RECORD_FIELDS`).  ``on`` is
    #: ``source`` or ``preparation``; a preparation wait names no lead.
    #: ``None`` on every other record, and then absent.
    wait: dict | None = None
    #: The attempt a restarted worker is on (:data:`RESTART_STATUS`):
    #: ``{attempt, reason}`` (:data:`RESTART_RECORD_FIELDS`), its number
    #: (2 after the first restart) and why the last attempt ended.  On every
    #: record from the restart record on; ``None`` on a first attempt's
    #: records, and then absent, so those keep their shape.
    restart: dict | None = None

    def __post_init__(self) -> None:
        if self.schema != HEARTBEAT_SCHEMA:
            raise ValueError(f"unsupported heartbeat schema {self.schema!r}")
        # ``finalizing:<phase>`` is the stretch after the last model step
        # -- drain, device synchronize, trajectory digest, receipts, and
        # a SHA-256 pass over every emitted frame.  It has no model step
        # to beat on, so before it was published a large run went silent
        # there for minutes and the stale-integration watchdog killed a
        # worker that was finishing normally.
        #
        # ``writing:<phase>`` is a history frame or a checkpoint written
        # between two model steps.  The write sits between two step
        # records, so without its own record a large streamed frame (5.24
        # GB, 81 s to write) and the stop-tick work after it were timed as
        # one model step, and a finished run was stopped at its last step.
        if (self.status not in {"integrating", "complete", "failed"}
                and not self.status.startswith(_PHASE_PREFIXES)):
            raise ValueError(f"invalid heartbeat status {self.status!r}")
        if (self.status.startswith(WAITING_PREFIX)
                and self.status not in WAITING_STATUSES):
            raise ValueError(f"invalid heartbeat status {self.status!r}; a "
                             f"wait is one of {list(WAITING_STATUSES)}")
        if self.wait is not None:
            if not self.status.startswith(WAITING_PREFIX):
                raise ValueError(
                    "heartbeat wait belongs to a waiting record, not to "
                    f"status {self.status!r}")
            if (not isinstance(self.wait, dict)
                    or set(self.wait) != set(WAIT_RECORD_FIELDS)):
                raise ValueError(
                    f"heartbeat wait must carry exactly {list(WAIT_RECORD_FIELDS)}, "
                    f"not {self.wait!r}")
            if f"{WAITING_PREFIX}{self.wait['on']}" != self.status:
                raise ValueError(
                    f"heartbeat wait on {self.wait['on']!r} does not match "
                    f"status {self.status!r}")
        if self.status == RESTART_STATUS and self.restart is None:
            raise ValueError(
                f"a {RESTART_STATUS!r} record declares the attempt it "
                "starts (restart)")
        if self.restart is not None:
            if (not isinstance(self.restart, dict)
                    or set(self.restart) != set(RESTART_RECORD_FIELDS)):
                raise ValueError(
                    "heartbeat restart must carry exactly "
                    f"{list(RESTART_RECORD_FIELDS)}, not {self.restart!r}")
            attempt = self.restart["attempt"]
            if (isinstance(attempt, bool) or not isinstance(attempt, int)
                    or attempt < 2):
                raise ValueError(
                    "heartbeat restart attempt must be an integer of at "
                    f"least 2 (the first restart starts attempt 2), not "
                    f"{attempt!r}")
        if self.pid <= 0 or self.outer_step < 0:
            raise ValueError("heartbeat pid must be positive and step nonnegative")
        if (not math.isfinite(self.model_elapsed_seconds)
                or self.model_elapsed_seconds < 0.0):
            raise ValueError("heartbeat model time must be finite and nonnegative")
        if self.work_bytes is not None:
            if (isinstance(self.work_bytes, bool)
                    or not isinstance(self.work_bytes, int)
                    or self.work_bytes < 0):
                raise ValueError(
                    "heartbeat work_bytes must be a nonnegative integer, "
                    f"not {self.work_bytes!r}")
            if not self.status.startswith(WORK_SIZED_PREFIXES):
                raise ValueError(
                    "heartbeat work_bytes belongs to a finalizing or writing "
                    f"record, not to status {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        for optional in ("work_bytes", "wait", "restart"):
            if payload[optional] is None:
                del payload[optional]
        return payload

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "Heartbeat":
        extra = set(payload) - _HEARTBEAT_FIELDS - _HEARTBEAT_OPTIONAL_FIELDS
        missing = _HEARTBEAT_FIELDS - set(payload)
        if extra or missing:
            raise ValueError(
                f"heartbeat fields mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}")
        return cls(**payload)


def read_heartbeat(path: str | Path) -> Heartbeat:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("heartbeat JSON root must be an object")
    return Heartbeat.from_mapping(payload)


def write_heartbeat(path: str | Path, heartbeat: Heartbeat, *,
                    durable: bool = True) -> Path:
    return atomic_write_json(
        path, heartbeat.as_dict(), _quarantine_on_permission_error=True,
        durable=durable)


class RollingStepWall:
    """Bounded rolling wall-time history with a nearest-rank p99."""

    def __init__(self, maxlen: int = 256):
        if maxlen < 1:
            raise ValueError("maxlen must be positive")
        self._values: deque[float] = deque(maxlen=maxlen)

    def add(self, seconds: float) -> None:
        value = float(seconds)
        if math.isfinite(value) and value > 0.0:
            self._values.append(value)

    @property
    def p99(self) -> float:
        if not self._values:
            return 0.0
        values = sorted(self._values)
        index = max(0, math.ceil(0.99 * len(values)) - 1)
        return values[index]

    @property
    def stale_threshold_seconds(self) -> float:
        return max(3.0 * self.p99, 120.0)


def stale_threshold_seconds(step_wall_seconds: list[float] | tuple[float, ...]
                            ) -> float:
    history = RollingStepWall(maxlen=max(1, len(step_wall_seconds)))
    for value in step_wall_seconds:
        history.add(value)
    return history.stale_threshold_seconds


#: The slowest rate at which a healthy worker is taken to move its own
#: bytes after the last model step: write a queued history frame, read it
#: back for its SHA-256, digest a domain's state.  Set for the slowest
#: storage a run folder plausibly sits on (a network share over Wi-Fi, a
#: USB 2 disk), because a floor set above it kills a finishing run there.
#: It prices the finalization bound below and nothing else.
FINALIZATION_FLOOR_BYTES_PER_SECOND = 8 * 1024 * 1024


def finalization_stale_threshold_seconds(step_threshold_seconds: float,
                                         work_bytes: int | None) -> float:
    """How long one ``finalizing:`` or ``writing:`` record may stay newest.

    Finalization is not a model step, so the step bound says nothing about
    it: draining a large tree's queued history frames or hashing a
    multi-GiB frame runs well past ``max(3*p99, 120 s)`` on a healthy
    machine, and timing it as a step killed workers that were finishing.
    It is not unbounded either, because a worker hung in a writer, a
    device synchronize or a dead network share holds the GPU lock until
    something stops it.

    Each finalizing record declares the bytes it will move before the next
    one.  The bound is the step bound, for the fixed cost every phase pays,
    plus that work at :data:`FINALIZATION_FLOOR_BYTES_PER_SECOND`.  A record
    that declares nothing (device synchronize, receipts, the capsule) keeps
    the step bound alone.

    A ``writing:`` record is the same kind of work in the middle of the
    run: a history frame or a checkpoint written between two model steps,
    which is no model step either.
    """
    work = 0 if work_bytes is None else max(0, int(work_bytes))
    return (float(step_threshold_seconds)
            + work / FINALIZATION_FLOOR_BYTES_PER_SECOND)


def _utc_instant(text) -> datetime | None:
    if not isinstance(text, str) or not text:
        return None
    try:
        instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (instant.replace(tzinfo=timezone.utc) if instant.tzinfo is None
            else instant)


def wait_silence_limit_seconds(slowest_build_seconds: float | None = None
                               ) -> float:
    """How long a waiting record may go unrefreshed: the producer silence
    limit, max(120 s, 3 x the slowest forcing-time build seen so far)."""

    try:
        slowest = float(slowest_build_seconds or 0.0)
    except (TypeError, ValueError):
        slowest = 0.0
    if not math.isfinite(slowest) or slowest < 0.0:
        slowest = 0.0
    return max(WAIT_SILENCE_FLOOR_SECONDS, 3.0 * slowest)


def waiting_stop_reason(heartbeat: Heartbeat, *, silent_seconds: float,
                        slowest_build_seconds: float | None = None,
                        now: datetime | None = None) -> str | None:
    """Why a ``waiting:`` record must stop the worker now, or ``None``.

    A waiting worker refreshes its record every few seconds, so silence
    past :func:`wait_silence_limit_seconds` is a hung worker, on either
    cause.  A ``waiting:source`` record is also bounded by its lead's
    ``late_at`` plus :data:`SOURCE_WAIT_GRACE_SECONDS`, refreshed or not:
    the producer ends a late lead's run by name (exit 75), and a record
    still standing past that has lost its producer's late check.
    """

    status = heartbeat.status
    wait = heartbeat.wait or {}
    limit = wait_silence_limit_seconds(slowest_build_seconds)
    if silent_seconds > limit:
        return (f"forecast stalled in {status}: its wait record was not "
                f"refreshed for {silent_seconds:.1f} s (bound {limit:.1f} s, "
                "the producer silence limit)")
    if status == "waiting:source":
        late = _utc_instant(wait.get("late_at"))
        if late is not None:
            current = datetime.now(timezone.utc) if now is None else now
            over = (current - late).total_seconds()
            if over > SOURCE_WAIT_GRACE_SECONDS:
                return (f"forecast waited on source lead {wait.get('lead')} "
                        f"{over:.1f} s past its late time {wait.get('late_at')} "
                        f"(bound {SOURCE_WAIT_GRACE_SECONDS:.0f} s past it); "
                        "the producer's own late check did not end the run")
    return None


def _byte_words(count: int | None) -> str:
    if not count:
        return "no declared work"
    value = float(count)
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            break
        value /= 1024.0
    return (f"{int(value)} bytes" if unit == "bytes"
            else f"{value:.1f} {unit}") + " of declared work"


def config_digest(path: str | Path) -> str:
    return _hash_file(Path(path))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hash_mode(requested: str | None = None) -> str:
    """Resolve the directory-input hash mode: argument, env, then default.

    ``inventory`` binds a directory by relative path, size, and mtime.  It
    is cheap enough to run before every launch on a multi-GB static
    geography tree, and it is the default for that reason.  It has two
    known failure modes, both of which matter to a dual-run comparison:
    a byte-identical copy with fresh mtimes compares *different*, and a
    changed file that preserves path, size, and mtime compares *equal*.

    ``content`` binds the same directory by relative path, size, and the
    SHA-256 of each file's bytes.  It answers "are these the same input
    bytes" instead of "does this look like the same directory listing",
    at the cost of reading every file.
    """
    value = (os.environ.get(DIRECTORY_HASH_ENV, DIRECTORY_HASH_DEFAULT)
             if requested is None else requested)
    if value not in DIRECTORY_HASH_MODES:
        source = ("argument" if requested is not None
                  else f"{DIRECTORY_HASH_ENV} environment variable")
        raise ValueError(
            f"directory hash mode {value!r} from the {source} is not one of "
            f"{list(DIRECTORY_HASH_MODES)}")
    return value


def _hash_directory_manifest(path: Path, *, mode: str = "inventory") -> str:
    """Hash a directory input in ``inventory`` or ``content`` mode.

    The record layout is deliberately unprefixed so that an ``inventory``
    digest recorded by an earlier release still compares equal here; the
    two modes are told apart by the ``algorithm`` label stored beside the
    digest, never by the digest alone.
    """
    if mode not in DIRECTORY_HASH_MODES:
        raise ValueError(
            f"directory hash mode {mode!r} is not one of "
            f"{list(DIRECTORY_HASH_MODES)}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        stat = child.stat()
        third = (_hash_file(child) if mode == "content"
                 else str(stat.st_mtime_ns))
        record = f"{relative}\0{stat.st_size}\0{third}\n".encode("utf-8")
        digest.update(record)
    return digest.hexdigest()


def resolved_input_hashes(
        config_path: str | Path, *, directory_hash: str | None = None,
        config_bytes: bytes | None = None) -> dict[str, Any]:
    """Hash declared case inputs before any device import/allocation.

    Files are always content-hashed.  Directory inputs follow
    ``directory_hash`` (see :func:`directory_hash_mode`), and every record
    carries the algorithm that produced it so two runs can refuse to
    compare digests that were not computed the same way.
    """
    from woof.case_data import (load_experiment_case,
                                 load_experiment_case_bytes)

    mode = directory_hash_mode(directory_hash)
    path = Path(config_path)
    if config_bytes is None:
        _, data = load_experiment_case(path)
    else:
        _, data = load_experiment_case_bytes(
            config_bytes, source=str(path), base_dir=path.parent)
    result: dict[str, Any] = {}
    for record in data.resolved_inputs():
        path = Path(record.path).resolve()
        key = f"{record.role}:{path}"
        identity = {
            "role": record.role,
            "path": str(path),
            "detail": record.detail,
        }
        if path.is_file():
            entry = {"algorithm": "sha256", "digest": _hash_file(path),
                     "detail": record.detail, "identities": [identity]}
        elif path.is_dir():
            entry = {
                "algorithm": f"sha256-directory-{mode}",
                "digest": _hash_directory_manifest(path, mode=mode),
                "detail": record.detail,
                "identities": [identity],
            }
        else:
            raise FileNotFoundError(f"declared input disappeared: {path}")
        previous = result.get(key)
        if previous is None:
            result[key] = entry
            continue
        if (previous.get("algorithm") != entry["algorithm"]
                or previous.get("digest") != entry["digest"]):
            raise SupervisorError(
                "duplicate resolved input identity changed while its parent "
                f"inventory was hashed: {key}")
        previous["identities"].append(identity)
    return result


def _canonical_input_identity(
        role: Any, record_path: Any, detail: Any,
        ) -> tuple[str, str, str]:
    """Return one canonical provenance identity, retaining multiplicity."""

    if (not isinstance(role, str) or not role
            or not isinstance(record_path, (str, os.PathLike))
            or not isinstance(detail, str)):
        raise SupervisorError(
            "worker resolved an input with a malformed provenance identity")
    return role, str(Path(record_path).resolve()), detail


def _resolved_input_identity(record: Any) -> tuple[str, str, str]:
    return _canonical_input_identity(
        getattr(record, "role", None), getattr(record, "path", None),
        getattr(record, "detail", None))


def _parent_resolved_input_inventory(
        input_hashes: dict[str, Any],
        ) -> Counter[tuple[str, str, str]]:
    """Decode the exact role/path/detail multiset bound by the parent."""

    inventory: Counter[tuple[str, str, str]] = Counter()
    for key, entry in input_hashes.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise SupervisorError(
                "parent input-hash inventory contains a malformed entry")
        role, separator, source_text = key.partition(":")
        if not role or not separator or not source_text:
            raise SupervisorError(
                f"parent input-hash inventory key is malformed: {key!r}")
        algorithm = entry.get("algorithm")
        digest = entry.get("digest")
        if (algorithm not in {
                    "sha256", "sha256-directory-inventory",
                    "sha256-directory-content",
                }
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise SupervisorError(
                f"parent input-hash inventory has no valid SHA-256 for {key}")
        identities = entry.get("identities")
        if not isinstance(identities, list) or not identities:
            raise SupervisorError(
                "parent input-hash inventory has no exact identity multiset "
                f"for {key}")
        keyed_path = str(Path(source_text).resolve())
        for identity in identities:
            if (not isinstance(identity, dict)
                    or set(identity) != {"role", "path", "detail"}):
                raise SupervisorError(
                    f"parent input identity is malformed for {key}")
            parsed = _canonical_input_identity(
                identity.get("role"), identity.get("path"),
                identity.get("detail"))
            if parsed[:2] != (role, keyed_path):
                raise SupervisorError(
                    "parent input identity disagrees with its hash key: "
                    f"{key}")
            inventory[parsed] += 1
    return inventory


def _format_input_inventory_delta(
        inventory: Counter[tuple[str, str, str]]) -> str:
    labels: list[str] = []
    for (role, path, detail), count in sorted(inventory.items()):
        label = f"{role}:{path}"
        if detail:
            label += f" [{detail}]"
        if count != 1:
            label += f" x{count}"
        labels.append(label)
    return "[" + "; ".join(labels) + "]"


def _validate_worker_resolved_input_inventory(
        data: Any, input_hashes: dict[str, Any]) -> None:
    """Refuse any worker parse that differs from the parent's exact parse."""

    try:
        records = data.resolved_inputs()
    except (AttributeError, TypeError) as exc:
        raise SupervisorError(
            "worker case data cannot report its resolved input inventory") \
            from exc
    worker = Counter(_resolved_input_identity(record) for record in records)
    parent = _parent_resolved_input_inventory(input_hashes)
    missing = parent - worker
    extra = worker - parent
    if missing or extra:
        raise SupervisorError(
            "worker-resolved input inventory does not match the parent "
            "SHA-256 inventory; missing="
            f"{_format_input_inventory_delta(missing)}; extra="
            f"{_format_input_inventory_delta(extra)}")


def _copy_verified_authority(source: Path, destination: Path,
                             expected_sha256: str) -> None:
    """Copy one file once while proving the snapshot's exact digest."""

    before = source.stat()
    before_signature = (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                digest.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        after = source.stat()
        after_signature = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns)
        observed = digest.hexdigest()
        if before_signature != after_signature:
            raise SupervisorError(
                f"declared input changed while it was snapshotted: {source}")
        if observed != expected_sha256:
            raise SupervisorError(
                "declared input changed between hashing and snapshot: "
                f"{source}; expected {expected_sha256}, observed {observed}")
    except BaseException:
        try:
            if destination.exists():
                destination.chmod(stat.S_IWRITE | stat.S_IREAD)
                destination.unlink()
        except OSError:
            pass
        raise


def _content_authority_path(source: Path, root: Path,
                            expected_sha256: str) -> Path:
    """Publish or reuse one verified SHA-keyed file under a blocking lock."""

    destination = root / expected_sha256
    lock_path = root / ".locks" / f"{expected_sha256}.lock"
    while True:
        lock = GPUFileLock(
            f"input-authority:{expected_sha256}", path=lock_path,
            run_id=f"input-authority-{os.getpid()}")
        try:
            lock.acquire()
            break
        except GPUAlreadyLockedError:
            time.sleep(0.05)
    try:
        if destination.exists():
            observed = _hash_file(destination)
            if observed != expected_sha256:
                raise SupervisorError(
                    "shared input-authority store contains corrupt content: "
                    f"expected {expected_sha256}, observed {observed} at "
                    f"{destination}")
            destination.chmod(stat.S_IREAD)
            return destination
        temporary = root / (
            f".{expected_sha256}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            _copy_verified_authority(source, temporary, expected_sha256)
            try:
                # Atomic and create-only: a racing publisher's digest path
                # can never be replaced by this process.
                publish_new(temporary, destination)
            except FileExistsError:
                observed = _hash_file(destination)
                if observed != expected_sha256:
                    raise SupervisorError(
                        "racing input-authority publisher produced corrupt "
                        f"content: expected {expected_sha256}, observed "
                        f"{observed} at {destination}")
            _fsync_directory(root)
        finally:
            if temporary.exists():
                temporary.unlink()
        destination.chmod(stat.S_IREAD)
        return destination
    finally:
        lock.release()


def snapshot_resolved_input_files(
        config_path: str | Path, *, config_bytes: bytes,
        input_hashes: dict[str, Any], snapshot_root: str | Path,
        ) -> dict[str, dict[str, str]]:
    """Snapshot every resolved file authority into one content store.

    Each distinct SHA-256 is copied at most once, even if the same file fills
    several roles or several declared files are byte-identical.  Geography is
    a directory authority and retains the separately declared directory-hash
    policy; forcing, Vtable, WPS namelist, and source-orography are immutable
    file snapshots consumed by the worker.
    """

    if not input_hashes:
        # Supervisor unit fixtures with no case data intentionally install an
        # empty input inventory.  A real case-data config always resolves at
        # least forcing, Vtable, WPS namelist, and geography records.
        return {}
    from woof.case_data import load_experiment_case_bytes

    config_path = Path(config_path)
    _, data = load_experiment_case_bytes(
        config_bytes, source=str(config_path), base_dir=config_path.parent)
    files = [record for record in data.resolved_inputs()
             if record.role in _SNAPSHOT_FILE_ROLES]
    root = Path(snapshot_root)
    root.mkdir(parents=True, exist_ok=True)
    by_digest: dict[str, Path] = {}
    manifest: dict[str, dict[str, str]] = {}
    for record in files:
        source = Path(record.path).resolve()
        key = f"{record.role}:{source}"
        entry = input_hashes.get(key)
        if not isinstance(entry, dict) or entry.get("algorithm") != "sha256":
            raise SupervisorError(
                f"resolved file authority {key} has no parent SHA-256")
        expected = entry.get("digest")
        if not isinstance(expected, str) or len(expected) != 64:
            raise SupervisorError(
                f"resolved file authority {key} has an invalid SHA-256")
        snapshot = by_digest.get(expected)
        if snapshot is None:
            snapshot = _content_authority_path(source, root, expected)
            by_digest[expected] = snapshot
        previous = manifest.get(str(source))
        authority = {
            "sha256": expected,
            "snapshot": str(snapshot.resolve()),
        }
        if previous is not None and previous != authority:
            raise SupervisorError(
                f"one resolved input path has conflicting authorities: "
                f"{source}")
        manifest[str(source)] = authority
    return manifest


def _validated_worker_input_authorities(
        encoded: str, input_hashes: dict[str, Any],
        ) -> dict[Path, Path]:
    """Validate the parent's manifest and return source-to-snapshot paths."""

    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise SupervisorError(
            "worker received malformed input-authority manifest") from exc
    if not isinstance(decoded, dict):
        raise SupervisorError("worker input-authority manifest must be an object")
    expected_by_source: dict[Path, str] = {}
    for key, entry in input_hashes.items():
        if (not isinstance(key, str) or not isinstance(entry, dict)
                or entry.get("algorithm") != "sha256"):
            continue
        _role, separator, source_text = key.partition(":")
        if not separator:
            continue
        source = Path(source_text).resolve()
        expected = entry.get("digest")
        if source in expected_by_source and expected_by_source[source] != expected:
            raise SupervisorError(
                f"worker input inventory conflicts for {source}")
        expected_by_source[source] = expected
    replacements: dict[Path, Path] = {}
    for source_text, authority in decoded.items():
        if not isinstance(source_text, str) or not isinstance(authority, dict):
            raise SupervisorError("worker input-authority entry is malformed")
        source = Path(source_text).resolve()
        expected = expected_by_source.get(source)
        if expected is None or authority.get("sha256") != expected:
            raise SupervisorError(
                f"worker input authority is not capsule-bound for {source}")
        snapshot_text = authority.get("snapshot")
        if not isinstance(snapshot_text, str):
            raise SupervisorError(
                f"worker input authority has no snapshot path for {source}")
        snapshot = Path(snapshot_text).resolve()
        observed = _hash_file(snapshot)
        if observed != expected:
            raise SupervisorError(
                "worker input snapshot digest mismatch: expected "
                f"{expected}, observed {observed} at {snapshot}")
        replacements[source] = snapshot
    if set(replacements) != set(expected_by_source):
        missing = sorted(str(path) for path in set(expected_by_source)
                         - set(replacements))
        raise SupervisorError(
            f"worker input-authority manifest is incomplete; missing {missing}")
    return replacements


def git_commit() -> str:
    """HEAD of whatever checkout encloses the imported package.

    NOT a statement about the code that is running.  ``git rev-parse``
    walks UP from the package directory, so an install into
    ``<checkout>/.venv/lib/pythonX/site-packages`` reports the ENCLOSING
    CHECKOUT's HEAD while the running bytes are the installed wheel's.
    A reporter who pulls a new tag without reinstalling therefore files a
    capsule that names the new commit and ran the old code.  Pair this
    with :func:`installed_identity`, which reads the running distribution.
    """

    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=False,
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        # A pip-install host with no git binary at all.  The sentinel is
        # the same "unavailable: <why>" form the schema documents; a
        # missing executable must not turn a receipt into a traceback.
        return f"unavailable: {type(error).__name__}: {error}"
    return (result.stdout.strip() if result.returncode == 0
            else f"unavailable: {result.stderr.strip()}")


def installed_identity() -> dict[str, str]:
    """Identify the code that is ACTUALLY executing.

    ``git_commit`` above answers a different question, and the difference
    is not academic: it is how a support report comes in claiming a
    version the reporter never ran.  ``version`` is the installed
    distribution's metadata (what pip resolved), ``package_path`` is the
    directory the running module was imported from -- a ``site-packages``
    path beside a checkout-derived ``git_commit`` is the stale-install
    signature, visible in the capsule without a round trip.
    """

    from woof import DISTRIBUTION_NAME, __version__

    return {
        "distribution": DISTRIBUTION_NAME,
        "version": str(__version__),
        "package_path": str(Path(__file__).resolve().parent),
    }


@dataclass(frozen=True)
class GPUIdentity:
    uuid: str
    driver_version: str
    name: str
    index: int | None = None


@dataclass(frozen=True)
class GPUProcess:
    uuid: str
    pid: int
    process_name: str
    used_gpu_memory_mib: int | None = None
    process_type: str | None = None


def _nvidia_smi_failure(arguments: list[str], reason: str, *,
                       remedy: str, stdout=None, stderr=None) -> GPUPreflightError:
    # NVIDIA tools can put an NVML error on stdout while leaving stderr
    # empty. TimeoutExpired can carry bytes even with text=True. Preserve
    # both streams as diagnostics; neither is evidence of a successful query.
    details = []
    for name, value in (("stderr", stderr), ("stdout", stdout)):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value and value.strip():
            details.append(f"  nvidia-smi {name}: {value.strip()}")
    return GPUPreflightError("\n".join((
        f"GPU preflight failed closed: nvidia-smi {reason}; "
        "the required GPU state checks did not complete.",
        f"  {remedy}",
        f"  Failed query arguments: {arguments!r}",
        *details,
    )))


def _run_nvidia_smi(arguments: list[str]) -> str:
    remedy = ("Run nvidia-smi in the same shell and restore working NVIDIA "
              "driver/runtime access before retrying the forecast.")
    try:
        result = subprocess.run(
            [*__import__('woof.gpu_smi', fromlist=['command']).command(), *arguments], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=20)
    except FileNotFoundError as exc:
        raise _nvidia_smi_failure(
            arguments, "was not found on PATH", remedy=(
                "Make NVIDIA's nvidia-smi available on PATH in this "
                "environment, then run nvidia-smi in the same shell before "
                "retrying the forecast.")) from exc
    except subprocess.TimeoutExpired as exc:
        raise _nvidia_smi_failure(
            arguments, f"timed out after {exc.timeout:g} seconds",
            remedy=remedy, stdout=exc.stdout, stderr=exc.stderr) from exc
    except OSError as exc:
        raise _nvidia_smi_failure(
            arguments, f"could not be started ({exc})", remedy=(
                "Check executable access in this environment. " + remedy)) from exc
    if result.returncode != 0:
        reason = f"exited with status {result.returncode}"
        if os.name == "posix" and result.returncode < 0:
            # POSIX subprocess returns -SIGNAL. Windows return codes are
            # exit/exception statuses and retain their original numeric code.
            try:
                name = signal.Signals(-result.returncode).name
            except ValueError:
                pass
            else:
                reason = (f"terminated by {name} (signal {-result.returncode}, "
                          f"return code {result.returncode})")
        raise _nvidia_smi_failure(
            arguments, reason, remedy=remedy,
            stdout=result.stdout, stderr=result.stderr)
    return result.stdout


def query_gpus() -> tuple[GPUIdentity, ...]:
    output = _run_nvidia_smi([
        "--query-gpu=index,uuid,driver_version,name",
        "--format=csv,noheader,nounits"])
    identities = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if (len(parts) != 4 or not parts[1].startswith("GPU-")
                or not parts[0].isdigit()):
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed GPU row {line!r}")
        identities.append(GPUIdentity(
            parts[1], parts[2], parts[3], int(parts[0])))
    if not identities:
        raise GPUPreflightError("GPU preflight failed closed: no GPU reported")
    return tuple(identities)


def _visible_single_gpu(identities, visibility: str | None) -> GPUIdentity | None:
    """The one card ``CUDA_VISIBLE_DEVICES`` names, or None when it names none.

    Named breakage: on a multi-GPU node (an 8-card H100 or a 4-card RTX
    PRO 6000 box) ``CUDA_VISIBLE_DEVICES=<one card> woof run ...`` was
    refused with "--gpu-uuid is required", because nvidia-smi lists every
    card whatever the mask says, so the standard way to pin a job to one
    card did not count.  A mask that names exactly one card is that
    choice.  A numeric entry is CUDA's ordinal, which equals nvidia-smi's
    index under ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` and on a node of identical
    cards (CUDA's fastest-first order ties them in bus order); on a mixed
    node without PCI_BUS_ID the two can differ, so that case is refused
    with the UUID spelling that cannot.
    """

    entries = [value.strip() for value in (visibility or "").split(",")
               if value.strip()]
    if not entries:
        return None
    if len(entries) != 1:
        raise GPUPreflightError(
            f"CUDA_VISIBLE_DEVICES={visibility!r} names {len(entries)} cards; "
            "a single-card run needs one: set CUDA_VISIBLE_DEVICES to one "
            "card or pass --gpu-uuid: "
            + ", ".join(f"{gpu.index} {gpu.uuid}" for gpu in identities))
    value = entries[0]
    if value.isdigit():
        ordered_by_bus = os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
        if not ordered_by_bus and len({gpu.name for gpu in identities}) > 1:
            raise GPUPreflightError(
                f"CUDA_VISIBLE_DEVICES={value} is a CUDA ordinal on a node of "
                "different card models, where CUDA's default fastest-first "
                "order need not match nvidia-smi's index; name the card by "
                "UUID (CUDA_VISIBLE_DEVICES=GPU-... or --gpu-uuid), or set "
                "CUDA_DEVICE_ORDER=PCI_BUS_ID")
        matches = [gpu for gpu in identities if gpu.index == int(value)]
    elif value.startswith("GPU-"):
        matches = [gpu for gpu in identities if gpu.uuid.startswith(value)]
    else:
        matches = []
    if len(matches) != 1:
        raise GPUPreflightError(
            f"CUDA_VISIBLE_DEVICES={value!r} does not name exactly one card "
            "nvidia-smi reports: "
            + ", ".join(f"{gpu.index} {gpu.uuid}" for gpu in identities))
    return matches[0]


def select_gpu(requested_uuid: str | None = None) -> GPUIdentity:
    identities = query_gpus()
    if requested_uuid is None:
        if len(identities) == 1:
            return identities[0]
        visible = _visible_single_gpu(
            identities, os.environ.get("CUDA_VISIBLE_DEVICES"))
        if visible is None:
            raise GPUPreflightError(
                "multiple GPUs are present; pass --gpu-uuid or set "
                "CUDA_VISIBLE_DEVICES to one card to avoid ambiguous locking: "
                + ", ".join(f"{gpu.index} {gpu.uuid}" for gpu in identities))
        return visible
    matches = [gpu for gpu in identities if gpu.uuid == requested_uuid]
    if len(matches) != 1:
        raise GPUPreflightError(
            f"requested GPU UUID {requested_uuid!r} was not reported by "
            "nvidia-smi")
    return matches[0]


def _memory_mib(value: str) -> int | None:
    normalized = value.strip().strip("[]").strip()
    if normalized.lower() in {"n/a", "-", "not supported"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise GPUPreflightError(
            f"GPU preflight failed closed: malformed memory value {value!r}") from exc
    if parsed < 0:
        raise GPUPreflightError(
            f"GPU preflight failed closed: negative memory value {value!r}")
    return parsed


def parse_compute_apps_output(output: str) -> tuple[GPUProcess, ...]:
    """Parse the WDDM four-column query-compute-apps CSV shape."""
    processes = []
    for row in csv.reader(io.StringIO(output)):
        if not row or not any(part.strip() for part in row):
            continue
        if "no running processes" in ",".join(row).lower():
            continue
        parts = [part.strip() for part in row]
        if len(parts) != 4 or not parts[0].startswith("GPU-"):
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed process row {row!r}")
        try:
            pid = int(parts[1])
        except ValueError as exc:
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed process PID {row!r}") from exc
        if pid <= 0:
            raise GPUPreflightError(
                f"GPU preflight failed closed: nonpositive process PID {row!r}")
        processes.append(GPUProcess(
            parts[0], pid, parts[2], _memory_mib(parts[3])))
    return tuple(processes)


def _parse_pmon_output(output: str) -> dict[int, tuple[str, int | None]]:
    """Return PID -> (C/C+G type, framebuffer MiB) from one pmon sample."""
    result: dict[int, tuple[str, int | None]] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # ``pmon -s um`` emits gpu,pid,type,sm,mem,enc,dec,jpg,ofa,fb,ccpm,name.
        parts = stripped.split(maxsplit=11)
        if len(parts) != 12:
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed pmon row {line!r}")
        if parts[1] == "-" and all(part == "-" for part in parts[1:]):
            # Linux/TCC drivers emit one canonical all-dash row for an idle
            # physical GPU.  It is an explicit absence record, not a process.
            continue
        try:
            pid = int(parts[1])
        except ValueError as exc:
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed pmon PID {line!r}") from exc
        process_type = parts[2]
        if process_type not in {"C", "G", "C+G"}:
            raise GPUPreflightError(
                f"GPU preflight failed closed: unknown pmon type {process_type!r}")
        result[pid] = process_type, _memory_mib(parts[9])
    return result


def query_compute_processes(gpu_uuid: str) -> tuple[GPUProcess, ...]:
    """Return WDDM rows enriched with pmon's compute-vs-graphics type."""
    apps = parse_compute_apps_output(_run_nvidia_smi([
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"]))
    modes = _parse_pmon_output(_run_nvidia_smi([
        "pmon", "-i", gpu_uuid, "-c", "1", "-s", "um"]))
    enriched = []
    for process in apps:
        if process.uuid != gpu_uuid:
            continue
        process_type, pmon_memory = modes.get(process.pid, (None, None))
        memories = [value for value in (
            process.used_gpu_memory_mib, pmon_memory) if value is not None]
        enriched.append(dataclasses.replace(
            process, process_type=process_type,
            used_gpu_memory_mib=(max(memories) if memories else None)))
    return tuple(enriched)


def _gib(value: int | float) -> str:
    return f"{float(value) / float(1024 ** 3):.2f} GiB"


def _cotenant_detail(conflicts) -> str:
    return ", ".join(
        f"pid={process.pid} name={process.process_name!r} "
        f"memory={'unmeasured' if process.used_gpu_memory_mib in (None, 0) else f'{process.used_gpu_memory_mib}MiB'}"
        for process in conflicts)


#: The admission sentences this process has already said.  ``explain.warn``
#: has no de-duplication of its own, and the run doors ask this question
#: more than once -- the supervisor on every recovery attempt, the stream
#: controller before every stage command -- so without this a shared card
#: printed the same sentence per command instead of once.  The key is the
#: sentence itself, so a co-tenant that appears, grows, shrinks or leaves
#: changes the line and IS named again: this suppresses repetition, never
#: news.
_ADMISSION_WARNED: set[str] = set()


def _warn_once(action: str, *, why: str) -> None:
    """Say one admission sentence at most once per process."""

    key = " ".join(str(action).split())
    if key in _ADMISSION_WARNED:
        return
    _ADMISSION_WARNED.add(key)
    warn(action, why=why)


def shared_gpu_admission(gpu_uuid: str, conflicts, reservation_bytes=None, *,
                         decide: bool = True) -> dict:
    """Price one shared card rather than refusing it for being shared.

    Sharing a GPU is not a policy question, it is a VRAM question, and the
    number that answers it is measurable from the same tool the contender
    rows came from: NVML's device total minus the DEVICE-WIDE used figure.
    That figure counts a co-tenant's memory whether or not its per-process
    framebuffer was reported, which is exactly why an unmeasured per-process
    row is priced here instead of being refused: it is the conservative
    recorded basis, and the run is admitted or refused against it with both
    numbers on the page.

    Returns the admission receipt (the two numbers and the basis).  Raises
    :class:`GPUPreflightError` only when the run's PRICED reservation does
    not fit the device's measured free memory, and only when ``decide`` is
    true; a recovery attempt re-measures the device and names what it found
    but can never re-refuse a run that has already started.  nvidia-smi
    itself failing still fails closed, through ``device_wide_used_bytes``.
    """

    from woof.core.preflight import (device_physical_total_bytes,
                                      device_wide_used_bytes)

    detail = _cotenant_detail(conflicts)
    basis = ("NVML device total minus device-wide used, which counts a "
             "co-tenant's memory whether or not its per-process framebuffer "
             "was reported")
    if not conflicts:
        # An exclusive card asks the device nothing: there is no co-tenant
        # to price against, and a query here would be a second nvidia-smi
        # call on every launch for an answer nobody reads.
        return {
            "gpu_uuid": gpu_uuid,
            "cotenants": "",
            "device_total_bytes": None,
            "device_wide_used_bytes": None,
            "device_free_bytes": None,
            "reservation_bytes": (None if reservation_bytes is None
                                  else int(reservation_bytes)),
            "basis": "no CUDA co-tenant on this device",
            "verdict": "exclusive",
        }
    used = device_wide_used_bytes(device_id=gpu_uuid)
    total = device_physical_total_bytes(device_id=gpu_uuid)
    free = None if total is None else max(0, int(total) - int(used))
    receipt = {
        "gpu_uuid": gpu_uuid,
        "cotenants": detail,
        "device_total_bytes": total,
        "device_wide_used_bytes": int(used),
        "device_free_bytes": free,
        "reservation_bytes": (None if reservation_bytes is None
                              else int(reservation_bytes)),
        "basis": basis,
        "verdict": "admitted",
    }
    if free is None:
        _warn_once(
            f"GPU {gpu_uuid} is shared with CUDA compute process(es) "
            f"({detail}) and this device's total memory could not be read, "
            f"so the run is admitted on the co-tenants' own reported "
            f"footprint; NVML reports {_gib(used)} used device-wide",
            why=basis)
        receipt["verdict"] = "admitted-unpriced-device"
        return receipt
    if reservation_bytes is None:
        _warn_once(
            f"GPU {gpu_uuid} is shared with CUDA compute process(es) "
            f"({detail}); this run's reservation could not be priced from "
            f"its configuration, so it is admitted against the "
            f"{_gib(free)} this device reports free",
            why=basis)
        receipt["verdict"] = "admitted-unpriced-run"
        return receipt
    reservation = int(reservation_bytes)
    if reservation <= free or not decide:
        receipt["verdict"] = ("admitted" if reservation <= free
                              else "admitted-already-started")
        _warn_once(
            f"GPU {gpu_uuid} is shared with CUDA compute process(es) "
            f"({detail}); this run's priced reservation of "
            f"{_gib(reservation)} is measured against {_gib(free)} free on "
            f"the device"
            + ("" if reservation <= free else
               ", which it exceeds; the run has already started, so it is "
               "not refused here"),
            why=basis)
        return receipt
    raise GPUPreflightError(
        f"GPU {gpu_uuid} cannot admit this run beside its CUDA co-tenant(s) "
        f"({detail}): the run's priced reservation is {_gib(reservation)} "
        f"and the device reports {_gib(free)} free "
        f"({_gib(total)} total minus {_gib(used)} used device-wide). "
        f"Stop the co-tenant(s), or run this configuration on a device with "
        f"more free memory, or make it smaller. Basis: {basis}")


def preflight_exclusive_gpu(gpu_uuid: str, *,
                            approved_pids: set[int] | None = None,
                            memory_threshold_mib: int =
                            COMPUTE_MEMORY_THRESHOLD_MIB,
                            allow_shared_gpu: bool = False,
                            reservation_bytes: int | None = None,
                            decide: bool = True) -> dict:
    """Verify identity and price the card against this run's reservation.

    The UUID file lock is authoritative for excluding other woof runs.
    WDDM reports desktop graphics contexts in ``query-compute-apps``; pmon
    labels those ``C+G`` and they are explicitly permitted.  A pure ``C``
    process above the small context-noise threshold, or one whose own
    framebuffer figure is unmeasured, is a CO-TENANT: the run is priced
    against the device's measured free memory through
    :func:`shared_gpu_admission` and admitted when it fits, refused with
    both numbers when it does not.  Tool/parse failures still fail closed.

    ``allow_shared_gpu`` is accepted and has no effect: sharing is decided
    by measurement now, so the flag is a workaround for a capability that
    is default-on.  Its argparse help text feeds a generated document that
    is out of this lane's bounds, so it is retired in effect here and its
    wording is deferred.
    """
    if memory_threshold_mib < 0:
        raise ValueError("memory_threshold_mib must be nonnegative")
    # Re-query on every launch so a stale UUID selection cannot silently
    # migrate the worker to a different physical device.
    select_gpu(gpu_uuid)
    approved = set() if approved_pids is None else set(approved_pids)
    # WDDM pmon can report fb=0 for an active pure-C row, so zero is not
    # evidence that the context is harmless; it is an unmeasured PER-PROCESS
    # figure, and the device-wide figure prices it.
    conflicts = [
        process for process in query_compute_processes(gpu_uuid)
        if (process.pid not in approved
            and process.process_type == "C"
            and (process.used_gpu_memory_mib in (None, 0)
                 or process.used_gpu_memory_mib > memory_threshold_mib))
    ]
    if conflicts and allow_shared_gpu:
        _warn_once(
            "--allow-shared-gpu is redundant: a shared GPU is admitted or "
            "refused by measuring this run's priced reservation against "
            "the device's free memory, not by a flag",
            why="")
    return shared_gpu_admission(
        gpu_uuid, conflicts, reservation_bytes, decide=decide)


def default_lock_path(gpu_uuid: str) -> Path:
    digest = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:24]
    configured = os.environ.get(GPU_LOCK_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser().resolve()
    elif os.name == "nt":
        root = (Path(os.environ.get("PROGRAMDATA", tempfile.gettempdir()))
                / "woof" / "locks")
    else:
        root = Path(tempfile.gettempdir()) / "woof" / "locks"
    return root / f"gpu-{digest}.lock"


class GPUFileLock:
    """Cross-process UUID-keyed exclusive lock (Windows byte-range lock)."""

    def __init__(self, gpu_uuid: str, *, path: str | Path | None = None,
                 run_id: str | None = None):
        self.gpu_uuid = gpu_uuid
        self.path = default_lock_path(gpu_uuid) if path is None else Path(path)
        self.run_id = run_id
        self._stream = None

    def acquire(self) -> "GPUFileLock":
        if self._stream is not None:
            raise RuntimeError("GPU lock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            if _is_lock_contention(exc):
                raise GPUAlreadyLockedError(
                    f"GPU {self.gpu_uuid} lock is held: {self.path}") from exc
            raise
        self._stream = stream
        owner = json.dumps({"gpu_uuid": self.gpu_uuid, "pid": os.getpid(),
                            "run_id": self.run_id, "acquired_at_utc": utc_now()},
                           sort_keys=True).encode("utf-8")
        stream.seek(1)
        stream.truncate()
        stream.write(owner)
        stream.flush()
        os.fsync(stream.fileno())
        return self

    def release(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "GPUFileLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _is_lock_contention(exc: OSError) -> bool:
    """Distinguish a held byte-range/flock from disk, ACL, or FD errors."""
    if os.name == "nt":
        return exc.errno in {errno.EACCES, errno.EDEADLK}
    return exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}


def validate_manifest_checkpoint(path: str | Path) -> Path:
    """Read every NPZ member and prove agreement with the restart manifest."""
    from woof.io.restart import read_restart_header

    path = Path(path)
    try:
        header = read_restart_header(path)
        manifest = header.get("array_manifest")
        if not isinstance(manifest, dict):
            raise CheckpointValidationError(
                f"checkpoint {path} has no array_manifest object")
        from woof.io.restart_prefetch import restart_archive
        with restart_archive(path) as archive:
            members = set(archive.files)
            payload_members = {name for name in members
                               if name != "__gpuwm_restart_header__"}
            if payload_members != set(manifest):
                raise CheckpointValidationError(
                    f"checkpoint {path} member set disagrees with manifest")
            for name, expected in manifest.items():
                array = archive[name]
                if (list(array.shape) != expected.get("shape")
                        or str(array.dtype) != expected.get("dtype")):
                    raise CheckpointValidationError(
                        f"checkpoint {path} member {name!r} disagrees with "
                        "its shape/dtype manifest")
                # Accessing the last byte forces lazy zip decompression/read.
                if array.size:
                    array.reshape(-1)[-1]
    except CheckpointValidationError:
        raise
    except Exception as exc:
        raise CheckpointValidationError(
            f"checkpoint {path} is not manifest-valid: {exc}") from exc
    fsync_file(path)
    return path.resolve()


def is_cuda_fatal(value: BaseException | str) -> bool:
    text = f"{type(value).__name__}: {value}" if isinstance(
        value, BaseException) else str(value)
    return any(pattern.search(text) for pattern in _CUDA_FATAL_PATTERNS)


def _capsule_embedded_text(path: Any, *, data: bytes | None = None,
                           cap_bytes: int = FAILURE_CAPSULE_TEXT_CAP_BYTES,
                           ) -> dict[str, Any]:
    """One size-capped verbatim text capture for the failure capsule.

    Total by construction: the capsule writer is the crash reporter, so a
    capture problem (deleted file, permissions, an inventory entry that is
    not a path) degrades to a recorded absence, never a second crash.
    """
    record: dict[str, Any] = {"path": str(path)}
    try:
        if data is None:
            data = Path(path).read_bytes()
        record["size_bytes"] = len(data)
        record["truncated"] = len(data) > cap_bytes
        record["text"] = data[:cap_bytes].decode("utf-8", errors="replace")
    except Exception as exc:
        record["text"] = None
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _capsule_input_text(input_hashes: Any) -> dict[str, Any]:
    """Verbatim captures for the small-text declared inputs.

    Keyed by the same ``role:path`` keys ``input_hashes`` uses, restricted
    to :data:`FAILURE_CAPSULE_TEXT_ROLES`.  Never raises: the inventory may
    be absent, partial, or malformed at the moment of the crash.
    """
    captures: dict[str, Any] = {}
    try:
        entries = dict(input_hashes)
    except Exception:
        return captures
    for key, entry in entries.items():
        try:
            identities = entry.get("identities") or ()
            capture_path = next(
                (identity.get("path") for identity in identities
                 if identity.get("role") in FAILURE_CAPSULE_TEXT_ROLES),
                None)
        except Exception:
            continue
        if capture_path is not None:
            captures[str(key)] = _capsule_embedded_text(capture_path)
    return captures


def write_failure_capsule(
        path: str | Path, *, run_id: str, config_path: str | Path,
        config_sha256: str, input_hashes: dict[str, Any], gpu: GPUIdentity,
        last_phase: str, last_step: int, exception_type: str,
        exception_message: str, exception_traceback: str,
        last_durable_wrfout: str | None, last_checkpoint: str | None,
        worker_pid: int | None = None,
        config_bytes: bytes | None = None) -> Path:
    payload = {
        "schema": FAILURE_CAPSULE_SCHEMA,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256,
        # Verbatim small-text captures (v2): the config the run actually
        # used -- the caller's captured payload bytes when it has them,
        # otherwise a best-effort read of ``config_path`` -- plus the
        # declared small-text inputs.  These are files the user themselves
        # put on disk; embedding them saves the support round trip that
        # asks a reporter to mail back a sub-100-line TOML.
        "config_text": _capsule_embedded_text(config_path, data=config_bytes),
        "input_text": _capsule_input_text(input_hashes),
        "input_hashes": input_hashes,
        "git_commit": git_commit(),
        # The running distribution, which ``git_commit`` does not report:
        # an install inside the checkout makes that field the checkout's
        # HEAD even when the executing bytes are an older wheel.
        "installed": installed_identity(),
        "gpu": dataclasses.asdict(gpu),
        "worker_pid": worker_pid,
        "last_phase": last_phase,
        "last_step": int(last_step),
        "last_durable_wrfout": last_durable_wrfout,
        "last_checkpoint": last_checkpoint,
        "exception": {
            "type": exception_type,
            "message": exception_message,
            "traceback": exception_traceback,
            "cuda_fatal": is_cuda_fatal(
                f"{exception_type}: {exception_message}\n{exception_traceback}"),
        },
    }
    return atomic_write_json(path, payload)


def _file_bytes(path: str | Path) -> int | None:
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return None


class RuntimeHeartbeat:
    """Runtime callback installed by the deferred runtime handoff patch."""

    def __init__(self, path: str | Path, *, run_id: str,
                 config_sha256: str, started_at_utc: str,
                 initial_checkpoint: str | None = None):
        self.path = Path(path)
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.started_at_utc = started_at_utc
        self.last_wrfout: str | None = None
        self.last_checkpoint: str | None = (
            None if initial_checkpoint is None
            else str(Path(initial_checkpoint).resolve()))
        self.last_phase = "preparing:worker-start"
        self.last_step = 0
        self.model_elapsed_seconds = 0.0
        #: The status the latest record published, and the one a
        #: :meth:`writing` record hands back to in :meth:`written`.
        self.last_status: str | None = None
        self._before_write: str | None = None
        #: The status a :meth:`waiting` record interrupted, which
        #: :meth:`waited` publishes again.
        self._before_wait: str | None = None
        #: The ``restart`` record every record carries once this process
        #: has restarted its forecast (:meth:`restarting`); ``None`` on the
        #: first attempt.
        self.restart_record: dict | None = None

    @property
    def attempt(self) -> int:
        """The attempt this process is on: 1 until :meth:`restarting`."""

        return 1 if self.restart_record is None else int(
            self.restart_record["attempt"])

    def _write(self, status: str, *, work_bytes: int | None = None,
               wait: dict | None = None) -> None:
        self.last_status = status
        if not status.startswith("writing:"):
            # Any other record ends a write: there is nothing left for
            # :meth:`written` to hand back.
            self._before_write = None
        if not status.startswith(WAITING_PREFIX):
            self._before_wait = None
        write_heartbeat(self.path, Heartbeat(
            HEARTBEAT_SCHEMA, self.run_id, self.config_sha256, os.getpid(),
            self.started_at_utc, utc_now(), status,
            self.model_elapsed_seconds, self.last_step, self.last_wrfout,
            self.last_checkpoint, work_bytes, wait,
            None if self.restart_record is None
            else dict(self.restart_record)),
            # The per-step beat is liveness, replaced by the next step's: it
            # is published atomically but not fsynced.  Breakage this
            # prevents: two fsyncs (file and directory) on the stepping
            # thread every step, 10-27 ms each on a box whose disk is busy
            # with 16 GB history frames (test host, 8 x RTX 5090, 2026-10-21: the
            # stepping thread sat in xlog_wait_on_iclog, the 26 ms per-step
            # gap the step timer missed).  Every other status stays durable.
            durable=status != "integrating")

    def __call__(self, *, model_elapsed_seconds: float, outer_step: int,
                 last_durable_wrfout: str | Path | None,
                 last_checkpoint: str | Path | None,
                 phase: str = "synchronized-step", **_: Any) -> None:
        if last_durable_wrfout is not None:
            wrfout = Path(last_durable_wrfout)
            resolved_wrfout = str(wrfout.resolve())
            if resolved_wrfout != self.last_wrfout:
                fsync_file(wrfout)
                self.last_wrfout = resolved_wrfout
        if last_checkpoint is not None:
            resolved_checkpoint = str(Path(last_checkpoint).resolve())
            if resolved_checkpoint != self.last_checkpoint:
                # Every member is read back before this step's record, and a
                # checkpoint is the whole state: 10.6 GB of a 1132x906x55
                # streamed domain took 104 s after the stop-tick checkpoint
                # was written, silent, against a 120 s step bound.  The read
                # is its own record, sized from the file; the step record
                # below ends it.
                self.writing("verify-checkpoint",
                             work_bytes=_file_bytes(last_checkpoint))
                self.last_checkpoint = str(validate_manifest_checkpoint(
                    last_checkpoint))
        self.model_elapsed_seconds = float(model_elapsed_seconds)
        self.last_step = int(outer_step)
        self.last_phase = phase
        self._write("integrating")

    def preparing(self, phase: str) -> None:
        """Publish an immediate/named preparation-stage heartbeat."""
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase).strip("-")
        if not normalized:
            raise ValueError("preparation phase must not be empty")
        self.last_phase = f"preparing:{normalized}"
        self._write(self.last_phase)

    def starting(self) -> None:
        """Backward-compatible spelling for the immediate worker heartbeat."""
        self.preparing("worker-start")

    def finalizing(self, phase: str, *, work_bytes: int | None = None) -> None:
        """Publish one named beat from the post-integration stretch.

        Same normalization and same shape as :meth:`preparing`; the
        different prefix is what lets the monitor tell "still working, no
        model steps left" from "stopped answering mid-integration".

        ``work_bytes`` is what this beat will write or read before the
        next one, and it is what the monitor sizes this phase's bound
        from; a beat that moves no bulk data leaves it out.
        """
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase).strip("-")
        if not normalized:
            raise ValueError("finalization phase must not be empty")
        self.last_phase = f"finalizing:{normalized}"
        self._write(self.last_phase,
                    work_bytes=None if work_bytes is None else int(work_bytes))

    def writing(self, phase: str, *, work_bytes: int | None = None) -> None:
        """Publish the start of one write between two model steps.

        A history frame or a checkpoint is written after one step's record
        and before the next one, so without this the monitor times the
        write as part of a model step.  This record names the write and
        declares its bytes, which size the write's bound
        (:func:`finalization_stale_threshold_seconds`).  :meth:`written`
        ends it.
        """
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase).strip("-")
        if not normalized:
            raise ValueError("write phase must not be empty")
        if self._before_write is None:
            self._before_write = self.last_status
        self._write(f"writing:{normalized}",
                    work_bytes=None if work_bytes is None else int(work_bytes))

    def written(self) -> None:
        """End a :meth:`writing` record: publish the status it interrupted.

        The fresh record is the beat after the write, and it carries the
        status from before it, so a write during preparation (the analysis
        frame) goes back to preparation's rules and one during integration
        back to the step bound.
        """
        before, self._before_write = self._before_write, None
        if before is not None:
            self._write(before)

    def waiting(self, on: str, *, since_utc: str, lead: int | None = None,
                expected_at: str | None = None,
                late_at: str | None = None) -> None:
        """Publish (or refresh) one wait for a boundary interval.

        ``on`` is ``source`` (a source lead not posted yet) or
        ``preparation`` (the interval not built yet).  Called again every
        few seconds while the wait lasts, so the record says the worker
        is alive; model time and step stay where the last step left them,
        so the wait is never taken for progress.  :meth:`waited` ends it.
        """

        if self._before_wait is None:
            self._before_wait = self.last_status
        self._write(f"{WAITING_PREFIX}{on}", wait={
            "on": str(on), "lead": None if lead is None else int(lead),
            "expected_at": expected_at, "late_at": late_at,
            "since_utc": str(since_utc)})

    def waited(self) -> None:
        """End a :meth:`waiting` record: publish the status it interrupted."""

        before, self._before_wait = self._before_wait, None
        if before is not None:
            self._write(before)

    def restarting(self, reason: str) -> None:
        """Publish the start of a new attempt in this same process.

        The last attempt's step, model time, frame and checkpoint are
        dropped (its outputs were set aside), so the record and every beat
        after it describe the new attempt from its start.  The record is
        :data:`RESTART_STATUS`, and it and every record after it carry
        ``restart`` = ``{attempt, reason}``, which is what lets a
        supervisor take the step and model time going back to zero as a
        new attempt rather than a regression (:func:`_heartbeat_regression`).
        A write or a wait the last attempt left open ends here.
        """

        self.restart_record = {"attempt": self.attempt + 1,
                               "reason": str(reason)}
        self.last_wrfout = None
        self.last_checkpoint = None
        self.last_step = 0
        self.model_elapsed_seconds = 0.0
        self.last_phase = RESTART_STATUS
        self._write(RESTART_STATUS)

    def complete(self, model_elapsed_seconds: float) -> None:
        self.model_elapsed_seconds = float(model_elapsed_seconds)
        self.last_phase = "complete"
        self._write("complete")

    def failed(self) -> None:
        self._write("failed")


@contextlib.contextmanager
def writing_progress(progress_callback, phase: str, *,
                      work_bytes: int | None = None):
    """Beat before and after one write made between two model steps.

    A history frame and a checkpoint are written after one step's record
    and before the next, so a supervisor that heard nothing timed the write
    as a model step: a 1132x906x55 streamed forecast writing its last 5.24
    GB frame and its stop-tick checkpoint went past the step bound and was
    stopped with every step done.  ``progress_callback.writing`` publishes a
    ``writing:<phase>`` record declaring ``work_bytes``, which sizes that
    write's bound; ``written`` publishes the status the write interrupted.
    Same optional-hook convention as
    :func:`woof.runtime._finalizing_progress`: a callback without them is
    left alone.  Nothing is published when the write raises, because the
    failure record follows.
    """
    reporter = getattr(progress_callback, "writing", None)
    if reporter is None:
        yield
        return
    reporter(phase, work_bytes=None if work_bytes is None else int(work_bytes))
    yield
    done = getattr(progress_callback, "written", None)
    if done is not None:
        done()


def restart_attempt(progress_callback, reason: str) -> None:
    """Declare that a runner starts its forecast again in this process.

    For a runner that ends one attempt and runs again in the same worker:
    the tree runner after a streamed interval moved its terrain clock
    (:mod:`woof.prepared_domain_tree_forecast`), and any single-domain
    head-bound runner that does the same.  Call it as soon as the last
    attempt has ended, before its outputs move aside (a hosting observer
    ends the renders reading them here) and before anything of the new
    one is published.  ``progress_callback.restarting`` publishes the
    restart record (:meth:`RuntimeHeartbeat.restarting`); a callback
    without the hook is left alone, the same convention as
    :func:`writing_progress`.

    What a runner owes after it: the heartbeat file stays in the run
    folder (a watchdog reads it throughout), and every wait before the
    new attempt's first step is published through the callback's
    ``waiting``/``waited`` hooks, as a seam wait is
    (:class:`woof.ingest.boundary_stream.SeamWaits`), so a supervisor
    times the wait by its own bound and a reader sees what it waits on.

    What each supervisor does with it: both take the higher attempt as a
    new start (:func:`_heartbeat_regression`) and time the new attempt's
    preparation as preparation, not by the step bound.
    :class:`woof.forecast_supervisor.ForecastWatchdog` (``woof go``'s
    forecast stage) gives ``preparing:`` no deadline, and its heartbeat
    makes the new attempt's first step beat entry again;
    :func:`supervise_experiment` (``woof run``) times it by its
    ``prep_timeout_seconds``, as it times a launch's preparation.
    """

    hook = getattr(progress_callback, "restarting", None)
    if hook is not None:
        hook(str(reason))


def heartbeat_attempt(heartbeat: Heartbeat) -> int:
    """The attempt a record belongs to: 1 until a restart record says more.

    Read from the ``restart`` field every record of a restarted attempt
    carries (:data:`RESTART_STATUS`), so a supervisor that missed the
    restart record itself still sees the attempt change.
    """

    return 1 if heartbeat.restart is None else int(
        heartbeat.restart["attempt"])


@dataclass(frozen=True)
class SupervisorResult:
    run_id: str
    attempts: int
    heartbeat: Heartbeat
    stdout_logs: tuple[Path, ...]
    stderr_logs: tuple[Path, ...]


#: Worker phases whose answer cannot change between attempts: the
#: checkpoint bytes, the config digest and the reader are all identical,
#: so a refusal raised in one of them is not a transient fault.
RESTORE_PHASES = frozenset({
    "preparing:validate-checkpoint",
    "preparing:restore-checkpoint",
    "preparing:restore-tree-checkpoint",
})


def _terminate_fresh_worker(process: subprocess.Popen, *, timeout: float = 10.0
                            ) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def _tail(path: Path, limit: int = 32_768) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _read_worker_failure_capsule(path: Path, *, run_id: str,
                                 worker_pid: int) -> dict | None:
    """The capsule the worker published FOR THIS EXIT, or ``None``.

    The run/pid binding is the whole point: an unbound read would quote
    a previous attempt's capsule into this attempt's error, which is a
    worse failure than saying nothing.  Every I/O or parse problem is a
    ``None`` -- the capsule reader sits on the crash-reporting path, and
    the house rule there (see ``_capsule_embedded_text``) is that a
    capture problem degrades to a recorded absence, never a second
    crash.
    """

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (payload.get("schema") in SUPPORTED_FAILURE_CAPSULE_SCHEMAS
            and payload.get("run_id") == run_id
            and payload.get("worker_pid") == worker_pid):
        return payload
    return None


#: Capsule exception types the supervisor itself authored.  Their message
#: IS "worker exited with status N", so quoting one back into that same
#: sentence says nothing twice.
_SUPERVISOR_AUTHORED_CAPSULE_TYPES = frozenset({
    "WorkerExit", "WorkerMonitorFailure"})

#: How much of a refusal sentence reaches the console.  Long enough for
#: the refusals this exists to surface (the config loaders' are ~170
#: characters) and short enough that an unbounded message -- a dumped
#: array repr, a wrapped C++ exception -- cannot bury the rest of the
#: SupervisorError under itself.
_CAPSULE_HEADLINE_CAP = 240

#: An absolute path inside a refusal sentence (POSIX ``/a/b`` or Windows
#: ``C:\a\b``), not the ``//`` of a URL and not a relative ``a/b``.
_HEADLINE_PATH = re.compile(r"(?<![\w.:/\\])(?:[A-Za-z]:)?[\\/][^\s'\"]+")


def _shorten_headline_paths(text: str, keep: int = 2) -> str:
    """Each absolute path cut to ``.../<last keep components>``.

    The loaders name the file they refused at the head of their sentence
    and say what is wrong after it, so cutting an over-long headline at its
    end dropped exactly the part the user acts on whenever the path was deep
    (a CI runner's 120-character temporary root cut "must carry at least
    one [[domain]] table" short, 2026-10-15).  The full path stays in the
    capsule the message names.
    """

    def short(match: re.Match) -> str:
        token = match.group(0)
        parts = [part for part in re.split(r"[\\/]", token) if part]
        if len(parts) <= keep + 1:
            return token
        sep = "\\" if "\\" in token else "/"
        return "..." + sep + sep.join(parts[-keep:])

    return _HEADLINE_PATH.sub(short, text)


def _capsule_headline(payload: dict | None) -> str:
    """``"ValueError: the refusal sentence"``, or ``""`` if there is none.

    A guard refusal and a segfault leave the same shell of a
    SupervisorError -- "worker exited with status 1" plus a path -- so
    through 1.8.0 a config the loader deliberately rejected read on the
    console as a crash, and the sentence explaining it sat one file away
    in ``exception.message``.  This lifts the class and the first line of
    that message into the error the CLI actually prints.

    First LINE, not the whole message: layered refusals carry an
    explanation half after the ``[[explain]]`` sentinel, which
    ``explain.split`` removes because that sentinel is promised never to
    reach a terminal, and multi-line messages still keep their headline
    on line one.  The full text stays in the capsule, which the message
    still names.
    """

    if not isinstance(payload, dict):
        return ""
    exception = payload.get("exception")
    if not isinstance(exception, dict):
        return ""
    kind = str(exception.get("type") or "").strip()
    if kind in _SUPERVISOR_AUTHORED_CAPSULE_TYPES:
        return ""
    action, _ = explain_split(str(exception.get("message") or ""))
    first = next((line.strip() for line in action.splitlines()
                  if line.strip()), "")
    # A headline is fused into the middle of a longer sentence, so its
    # own full stop would land as ".; no durable ...".  One trailing
    # period goes; an ellipsis is not a full stop and stays.
    if first.endswith(".") and not first.endswith(".."):
        first = first[:-1]
    if len(first) > _CAPSULE_HEADLINE_CAP:
        first = _shorten_headline_paths(first)
    if len(first) > _CAPSULE_HEADLINE_CAP:
        first = first[:_CAPSULE_HEADLINE_CAP - 3].rstrip() + "..."
    if kind and first:
        return f"{kind}: {first}"
    return kind or first


def _worker_command(
        config_path: Path, config_payload: Path, outdir: Path, *,
        restart: Path | None, health_debug: bool,
        preprocess_backend: str | None = None, members: int | None = None,
        keep_member_files: bool | None = None) -> list[str]:
    command = [sys.executable, "-m", "woof.supervisor", "worker",
               "--config", str(config_path),
               "--config-payload", str(config_payload),
               "--outdir", str(outdir)]
    if restart is not None:
        command.extend(("--restart", str(restart)))
    if health_debug:
        command.append("--health-debug")
    if preprocess_backend is not None:
        command.extend(("--preprocess-backend", preprocess_backend))
    if members is not None:
        command.extend(("--members", str(members)))
    if keep_member_files:
        command.append("--keep-member-files")
    return command


def _capture_config_payload(outdir: Path, run_id: str,
                            payload: bytes) -> Path:
    """Durably create the unique config payload handed to every worker."""

    path = outdir / f"captured-config-{run_id}.toml"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _validated_config_payload_bytes(path: str | Path,
                                    expected_sha256: str) -> bytes:
    """Read once and reject a worker payload that is not parent-bound."""

    payload = Path(path).read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise SupervisorError(
            "captured worker config digest mismatch: expected "
            f"{expected_sha256}, observed {observed}")
    return payload


def _success_run_context(
        config_path: Path, digest: str, input_hashes: dict[str, Any], *,
        restart_interval_seconds: float | None) -> dict[str, Any]:
    """Deterministic capsule pins, excluding transient execution paths."""

    return {
        "config_bytes": {
            "path": str(config_path.resolve()), "sha256": digest},
        "input_artifact_bytes": input_hashes,
        "runner_route_and_io_mode": {
            "route": "supervisor:woof run", "io_mode": "history"},
        "output_and_diagnostic_mode": {
            "io_mode": "history",
            "restart_interval_seconds": restart_interval_seconds,
        },
    }


def _heartbeat_regression(previous: Heartbeat, current: Heartbeat) -> str | None:
    if (previous.status in {"complete", "failed"}
            and current != previous):
        return f"terminal status {previous.status} changed after publication"
    previous_time = datetime.fromisoformat(
        previous.updated_at_utc.replace("Z", "+00:00"))
    current_time = datetime.fromisoformat(
        current.updated_at_utc.replace("Z", "+00:00"))
    if current_time < previous_time:
        return "updated_at_utc moved backward"
    previous_attempt = heartbeat_attempt(previous)
    current_attempt = heartbeat_attempt(current)
    if current_attempt < previous_attempt:
        return (f"attempt moved backward from {previous_attempt} to "
                f"{current_attempt}")
    if current_attempt > previous_attempt:
        # A declared new attempt (RESTART_STATUS): its step and model time
        # start again, from any record that is not terminal.  The records
        # of one attempt are held to each other by the rules below.
        return None
    if current.outer_step < previous.outer_step:
        return (f"outer_step moved backward from {previous.outer_step} to "
                f"{current.outer_step}")
    if current.model_elapsed_seconds < previous.model_elapsed_seconds:
        return ("model_elapsed_seconds moved backward from "
                f"{previous.model_elapsed_seconds} to "
                f"{current.model_elapsed_seconds}")
    if (previous.status == "integrating"
            and current.status.startswith("preparing:")):
        return f"status moved backward from integrating to {current.status}"
    # Finalization is one-way.  Integration is over by the time it is
    # published, so a worker that goes back to integrating, to preparation
    # or to a source wait (which only a seam or the start has) has either
    # restarted inside its own process without saying so (RESTART_STATUS)
    # or is not the worker this attempt launched.  A preparation wait is
    # the one wait finalization has: a head-bound run that has stepped
    # through its last interval waits for the preparation to seal before
    # it binds the seal, and before this was accepted that wait got the
    # worker stopped as a regression under ``woof go``.  It is accepted
    # after any ``finalizing:`` record, not only after the one that
    # announces the seal (``finalizing:bind-prepared-seal``), because a
    # supervisor reads the file on its own poll: the seal wait publishes
    # its first record as soon as it finds no seal, so that announcement
    # stands for well under a poll and the record a poll last read is
    # usually the one before it (a final health or digest phase).  Held to
    # the announcing record alone, the same wait was stopped again.
    if (previous.status.startswith("finalizing:")
            and (current.status == "integrating"
                 or current.status.startswith("preparing:")
                 or (current.status.startswith(WAITING_PREFIX)
                     and current.status != "waiting:preparation"))):
        return (f"status moved backward from {previous.status} to "
                f"{current.status}")
    return None


def _bind_attempt_heartbeat(
        heartbeat: Heartbeat, *, run_id: str, config_digest: str,
        started_at_utc: str, launch_pid: int,
        effective_worker_pid: int | None) -> tuple[int | None, str | None]:
    """Validate one heartbeat and pin the real interpreter PID.

    A Windows venv executable can be a redirector: ``Popen.pid`` belongs to
    the redirector while the Python interpreter that writes heartbeats has a
    descendant PID.  ``preparing:launch`` is the supervisor's provisional
    record.  The first worker-authored status pins the effective PID; later
    PID changes fail closed.  This is correlation and binding within a trusted
    output directory, not authentication against a hostile local writer.
    """
    if heartbeat.run_id != run_id:
        return effective_worker_pid, "run_id does not match this attempt"
    if heartbeat.config_digest != config_digest:
        return effective_worker_pid, "config_digest does not match this attempt"
    if heartbeat.started_at_utc != started_at_utc:
        return effective_worker_pid, "started_at_utc does not match this attempt"
    if heartbeat.status == "preparing:launch":
        if heartbeat.pid != launch_pid:
            return effective_worker_pid, (
                "provisional launch heartbeat PID does not match Popen PID")
        if effective_worker_pid is not None:
            return effective_worker_pid, (
                "heartbeat reverted to provisional launch after worker PID pin")
        return effective_worker_pid, None
    if effective_worker_pid is None:
        return heartbeat.pid, None
    if heartbeat.pid != effective_worker_pid:
        return effective_worker_pid, (
            f"worker heartbeat PID changed from {effective_worker_pid} to "
            f"{heartbeat.pid}")
    return effective_worker_pid, None


def priced_reservation_bytes(configuration, *, source=None,
                             urban_columns=None) -> int | None:
    """This run's priced peak envelope, or None when it cannot be priced.

    The same number plan review prices a configuration from
    (:func:`woof.core.preflight.admission_estimate`), so the run door and
    the review do not invent two answers.  ONE function, every run door:
    :func:`supervise_experiment` prices `woof run` here and
    ``woof.stream``'s controller prices `woof stream` here, because two
    doors that priced a shared card differently would admit a
    configuration at one and refuse it at the other.

    Takes either a configuration path or an already loaded
    :class:`~woof.experiment.ExperimentConfig`, so a caller that has
    already read the file does not read it a second time and risk pricing
    a different object than it runs.

    Unpriceable is None, never a number: an unpriced run is admitted
    against the device's measured free memory and says so, because
    refusing something for being unmeasured is exactly what this admission
    stopped doing.

    ``source`` is the forcing source a door that knows it runs from, so
    the analysed hydrometeor tables that source puts on the boundary are
    in the reservation (``woof stream`` runs HRRR); a configuration read
    from its file alone names none and prices water vapour only.
    ``urban_columns`` is a prepared door's reading of its land cover;
    configuration callers leave it unknown and keep the workspace bound.
    """

    try:
        from woof.core.preflight import (
            admission_estimate, device_memory_probe_subprocess,
            profile_from_device_probe)

        experiment = configuration
        if isinstance(configuration, (str, os.PathLike)):
            from woof.experiment import load_experiment

            experiment = load_experiment(configuration)
        pricing = {"source": source}
        # The supervisor has no CUDA context. Price its selected card from
        # the same short-lived probe as the planning door, rather than
        # charging every device the 170-SM reference card's local store.
        probe = device_memory_probe_subprocess()
        profile = profile_from_device_probe(probe)
        if profile is not None:
            from woof.core.streaming import planner_machine
            pricing["machine"] = planner_machine(
                vram_bytes=probe.get("total_bytes"), name=profile.name,
                device_profile=profile)
        if urban_columns is not None:
            pricing["urban_columns"] = urban_columns
        estimate = admission_estimate(experiment, **pricing)
        value = int(estimate.peak_envelope_bytes)
    except Exception:  # noqa: BLE001 - pricing is advisory, never a gate
        return None
    return value if value > 0 else None


def supervise_experiment(
        config_path: str | Path, outdir: str | Path, *,
        restart: str | Path | None = None, gpu_uuid: str | None = None,
        max_restarts: int = 3, poll_seconds: float = 1.0,
        prep_timeout_seconds: float | None = None,
        health_debug: bool = False, allow_shared_gpu: bool = False,
        lock_path: str | Path | None = None,
        directory_hash: str | None = None,
        on_progress: Callable[[Heartbeat], None] | None = None,
        preprocess_backend: str | None = None, members: int | None = None,
        keep_member_files: bool | None = None) -> SupervisorResult:
    """Run an experiment under exclusive-GPU fresh-process supervision.

    ``preprocess_backend`` (``woof run --preprocess-backend``) reaches
    every worker this run launches, recoveries included, and overrides the
    config's ``[case_data] preprocess_backend``; ``None`` leaves the
    config's own.
    """
    if max_restarts < 0:
        raise ValueError("max_restarts must be nonnegative")
    if not 0.05 <= poll_seconds <= 60.0:
        raise ValueError("poll_seconds must be in [0.05, 60]")
    if (prep_timeout_seconds is not None
            and (not math.isfinite(prep_timeout_seconds)
                 or prep_timeout_seconds <= 0.0)):
        raise ValueError("prep_timeout_seconds must be finite and positive")
    config_path = Path(config_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    from woof.config_authority import read_config_authority

    config_authority = read_config_authority(config_path)
    config_bytes = config_authority.payload
    from woof.ensemble.door import request_for_payload
    ensemble_request = request_for_payload(config_bytes, members=members,
                                          keep_member_files=keep_member_files)
    digest = hashlib.sha256(config_bytes).hexdigest()
    inputs = resolved_input_hashes(
        config_path, directory_hash=directory_hash,
        config_bytes=config_bytes)
    shared_authority_root = os.environ.get(SHARED_INPUT_AUTHORITY_ROOT_ENV)
    input_authorities = None
    if shared_authority_root:
        authority_root = Path(shared_authority_root).expanduser().resolve()
        input_authorities = snapshot_resolved_input_files(
            config_path, config_bytes=config_bytes, input_hashes=inputs,
            snapshot_root=authority_root)
    config_payload = _capture_config_payload(outdir, run_id, config_bytes)
    if ensemble_request is None:
        gpu = select_gpu(gpu_uuid)
        locked_gpus, cuda_mask = (gpu,), gpu.uuid
    else:
        from woof.ensemble.supervised_devices import plan_ensemble_devices
        lease = plan_ensemble_devices(ensemble_request, query_gpus(),
            requested_uuid=gpu_uuid, visibility=os.environ.get("CUDA_VISIBLE_DEVICES"))
        locked_gpus, cuda_mask = lease.locked, lease.cuda_mask
        gpu = locked_gpus[0]
    checkpoint = (None if restart is None
                  else validate_manifest_checkpoint(restart))
    heartbeat_path = outdir / HEARTBEAT_NAME
    capsule_path = outdir / FAILURE_CAPSULE_NAME
    stdout_logs: list[Path] = []
    stderr_logs: list[Path] = []
    attempts = 0

    reservation_bytes = priced_reservation_bytes(config_path)

    with contextlib.ExitStack() as gpu_locks:
        for selected_gpu in sorted(locked_gpus, key=lambda item: item.uuid):
            path = lock_path if len(locked_gpus) == 1 else None
            if lock_path is not None and len(locked_gpus) > 1:
                path = Path(lock_path).with_name(Path(lock_path).name + "." + selected_gpu.uuid)
            gpu_locks.enter_context(GPUFileLock(selected_gpu.uuid, path=path, run_id=run_id))
        # The admission decision is taken ONCE, here, before the first
        # worker exists.  It used to sit inside the recovery loop, where a
        # co-tenant that appeared mid-run could refuse a run that had
        # already produced output.
        for selected_gpu in locked_gpus:
            preflight_exclusive_gpu(
                selected_gpu.uuid, approved_pids={os.getpid()},
                allow_shared_gpu=allow_shared_gpu,
                reservation_bytes=reservation_bytes)
        while True:
            if attempts:
                # A recovery attempt re-measures the device, so a co-tenant
                # that appeared between attempts is still named; it can
                # never re-refuse the run.
                for selected_gpu in locked_gpus:
                    preflight_exclusive_gpu(
                        selected_gpu.uuid, approved_pids={os.getpid()},
                        allow_shared_gpu=allow_shared_gpu,
                        reservation_bytes=reservation_bytes, decide=False)
            attempts += 1
            # Every fresh process gets fresh preparation and step clocks.  A
            # recovery launch can never inherit the dead worker's stale age or
            # p99 history.
            history = RollingStepWall()
            started_at = utc_now()
            stdout_path = outdir / WORKER_STDOUT_NAME.format(attempt=attempts)
            stderr_path = outdir / WORKER_STDERR_NAME.format(attempt=attempts)
            stdout_logs.append(stdout_path)
            stderr_logs.append(stderr_path)
            env = os.environ.copy()
            env.update({
                # ``--gpu-uuid`` used to select and lock a physical card but
                # left every card visible to the worker, whose CuPy code uses
                # process-local device ordinal 0.  Mask before Popen so
                # logical device 0 is the selected UUID before any CUDA import
                # or context can exist in the fresh worker.
                **__import__("woof.gpu_smi",fromlist=["visibility_environment"]).visibility_environment(cuda_mask),
                "WOOF_RUN_ID": run_id,
                "WOOF_CONFIG_DIGEST": digest,
                "WOOF_STARTED_AT_UTC": started_at,
                "WOOF_GPU_UUID": gpu.uuid,
                "WOOF_GPU_DRIVER": gpu.driver_version,
                "WOOF_GPU_NAME": gpu.name,
                "WOOF_INPUT_HASHES_JSON": json.dumps(
                    inputs, sort_keys=True, separators=(",", ":")),
            })
            if input_authorities is not None:
                env[INPUT_AUTHORITIES_ENV] = json.dumps(
                    input_authorities, sort_keys=True, separators=(",", ":"))
            launched_checkpoint = checkpoint
            command = _worker_command(
                config_path, config_payload, outdir, restart=checkpoint,
                health_debug=health_debug,
                preprocess_backend=preprocess_backend, members=members,
                keep_member_files=keep_member_files)
            # The worker dies with this supervisor (woof.parent_death):
            # a `woof run` killed by pid must not leave its worker on the
            # card as an orphan, as a killed `woof go` left its forecast
            # holding 63.9 GB for 43 minutes on a rented box (2026-10-07).  The
            # worker binds itself after exec; no preexec_fn (D-03).
            from woof.parent_death import child_environment

            env.update(child_environment())
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command, cwd=Path(__file__).resolve().parents[1], env=env,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    close_fds=True)
            starting = Heartbeat(
                HEARTBEAT_SCHEMA, run_id, digest, process.pid, started_at,
                utc_now(), "preparing:launch", 0.0, 0, None,
                None if checkpoint is None else str(checkpoint))
            # A tiny fixture worker can publish before Popen returns.  Never
            # overwrite a newer record from this attempt with parent state.
            # On Windows a venv ``python.exe`` may be a redirector whose
            # Popen PID differs from the interpreter PID that publishes the
            # heartbeat.  The run id, config digest, and per-attempt start
            # time correlate the record; the first worker-authored record
            # then pins the effective interpreter PID for the whole attempt.
            try:
                existing = read_heartbeat(heartbeat_path)
            except (OSError, ValueError, json.JSONDecodeError):
                existing = None
            if (existing is None or existing.run_id != run_id
                    or existing.config_digest != digest
                    or existing.started_at_utc != started_at):
                write_heartbeat(heartbeat_path, starting)

            last_heartbeat: Heartbeat | None = None
            last_signal_monotonic = time.monotonic()
            integrating_seen = False
            effective_worker_pid: int | None = None
            monitor_failure: str | None = None
            monitor_failure_kind: str | None = None
            while process.poll() is None:
                time.sleep(poll_seconds)
                try:
                    current = read_heartbeat(heartbeat_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    current = None
                if current is not None:
                    effective_worker_pid, attempt_error = (
                        _bind_attempt_heartbeat(
                            current, run_id=run_id, config_digest=digest,
                            started_at_utc=started_at,
                            launch_pid=process.pid,
                            effective_worker_pid=effective_worker_pid))
                    if attempt_error is not None:
                        monitor_failure_kind = "heartbeat-identity"
                        monitor_failure = (
                            f"worker heartbeat identity violation: {attempt_error}")
                        _terminate_fresh_worker(process)
                        break
                    if last_heartbeat is not None:
                        regression = _heartbeat_regression(
                            last_heartbeat, current)
                        if regression is not None:
                            monitor_failure_kind = "heartbeat-regression"
                            monitor_failure = f"worker heartbeat regression: {regression}"
                            _terminate_fresh_worker(process)
                            break
                    if (last_heartbeat is not None
                            and heartbeat_attempt(current)
                            > heartbeat_attempt(last_heartbeat)):
                        # A new attempt in the same worker (restart_attempt)
                        # prepares again, so until it steps it is timed as
                        # a preparation is.  THE BREAKAGE: left set from
                        # the last attempt, the new attempt's restore was
                        # timed by the step bound and the worker stopped as
                        # a stale integration while it recovered.
                        integrating_seen = False
                    if current != last_heartbeat:
                        if (last_heartbeat is not None
                                and last_heartbeat.status == "integrating"
                                and current.status == "integrating"
                                and current.outer_step
                                > last_heartbeat.outer_step):
                            previous = datetime.fromisoformat(
                                last_heartbeat.updated_at_utc.replace(
                                    "Z", "+00:00")).timestamp()
                            updated = datetime.fromisoformat(
                                current.updated_at_utc.replace(
                                    "Z", "+00:00")).timestamp()
                            history.add(updated - previous)
                        last_heartbeat = current
                        last_signal_monotonic = time.monotonic()
                    integrating_seen |= current.status == "integrating"
                    if on_progress is not None:
                        on_progress(current)
                silent_seconds = time.monotonic() - last_signal_monotonic
                status = (None if last_heartbeat is None
                          else last_heartbeat.status)
                # A published terminal record retires the watchdog.  The
                # worker has said the run is over; what remains is
                # process teardown (CUDA context release, interpreter
                # shutdown), which has no heartbeat and is not
                # integration, so counting its silence as a stalled step
                # killed a finished worker and replayed the completed run
                # as a restart loop.  The exit status still decides
                # success below -- this only stops the SIGTERM.
                finished = status == "complete"
                # Finalization is timed by its own bound, never by the
                # preparation timeout or the step bound: a run resumed at
                # its stop tick finalizes without ever integrating.  A
                # write between two steps (``writing:``) is timed the
                # same way, by the bytes it declares.
                finalizing = (status is not None
                              and status.startswith(WORK_SIZED_PREFIXES))
                waiting = (status is not None
                           and status.startswith(WAITING_PREFIX))
                if (not integrating_seen and not finalizing and not finished
                        and not waiting
                        and prep_timeout_seconds is not None
                        and silent_seconds > prep_timeout_seconds):
                    monitor_failure_kind = "prep-timeout"
                    phase = ("preparing:launch" if last_heartbeat is None
                             else last_heartbeat.status)
                    monitor_failure = (
                        f"worker preparation timed out in {phase} after "
                        f"{silent_seconds:.1f} s without a heartbeat")
                    _terminate_fresh_worker(process)
                    break
                if finalizing:
                    bound = finalization_stale_threshold_seconds(
                        history.stale_threshold_seconds,
                        last_heartbeat.work_bytes)
                    if silent_seconds > bound:
                        finishing = status.startswith("finalizing:")
                        monitor_failure_kind = (
                            "stale-finalization" if finishing
                            else "stale-write")
                        monitor_failure = (
                            f"worker {'finalization' if finishing else 'write'} "
                            f"heartbeat ({status}) "
                            f"became stale after {silent_seconds:.1f} s; "
                            f"its bound was {bound:.1f} s for "
                            f"{_byte_words(last_heartbeat.work_bytes)}")
                        _terminate_fresh_worker(process)
                        break
                elif status is not None and status.startswith(WAITING_PREFIX):
                    reason = waiting_stop_reason(
                        last_heartbeat, silent_seconds=silent_seconds)
                    if reason is not None:
                        monitor_failure_kind = "stale-wait"
                        monitor_failure = f"worker {reason}"
                        _terminate_fresh_worker(process)
                        break
                elif (integrating_seen and not finished
                        and silent_seconds > history.stale_threshold_seconds):
                    monitor_failure_kind = "stale-integration"
                    monitor_failure = (
                        "worker integrating heartbeat became stale after "
                        f"{silent_seconds:.1f} s")
                    _terminate_fresh_worker(process)
                    break
            return_code = process.wait()
            if monitor_failure is None:
                try:
                    current = read_heartbeat(heartbeat_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    current = None
                if current is not None:
                    effective_worker_pid, attempt_error = (
                        _bind_attempt_heartbeat(
                            current, run_id=run_id, config_digest=digest,
                            started_at_utc=started_at,
                            launch_pid=process.pid,
                            effective_worker_pid=effective_worker_pid))
                    if attempt_error is not None:
                        monitor_failure_kind = "heartbeat-identity"
                        monitor_failure = (
                            f"worker heartbeat identity violation: {attempt_error}")
                    else:
                        regression = (
                            None if last_heartbeat is None
                            else _heartbeat_regression(last_heartbeat, current))
                        if regression is not None:
                            monitor_failure_kind = "heartbeat-regression"
                            monitor_failure = (
                                f"worker heartbeat regression: {regression}")
                        else:
                            last_heartbeat = current
            if (monitor_failure is None
                    and return_code == 0 and last_heartbeat is not None
                    and last_heartbeat.run_id == run_id
                    and last_heartbeat.config_digest == digest
                    and last_heartbeat.started_at_utc == started_at
                    and last_heartbeat.status == "complete"):
                return SupervisorResult(
                    run_id, attempts, last_heartbeat, tuple(stdout_logs),
                    tuple(stderr_logs))

            message = (monitor_failure if monitor_failure is not None else
                       f"worker exited with status {return_code}")
            stderr_tail = _tail(stderr_path)
            hb = last_heartbeat
            worker_pid = effective_worker_pid or process.pid
            worker_capsule = _read_worker_failure_capsule(
                capsule_path, run_id=run_id, worker_pid=worker_pid)
            if worker_capsule is None:
                write_failure_capsule(
                    capsule_path, run_id=run_id, config_path=config_path,
                    config_sha256=digest, input_hashes=inputs, gpu=gpu,
                    config_bytes=config_bytes,
                    last_phase=(monitor_failure_kind or "worker-exit"),
                    last_step=0 if hb is None else hb.outer_step,
                    exception_type=("WorkerMonitorFailure"
                                    if monitor_failure is not None
                                    else "WorkerExit"),
                    exception_message=message,
                    exception_traceback=stderr_tail,
                    last_durable_wrfout=(None if hb is None else
                                         hb.last_durable_wrfout),
                    last_checkpoint=(None if hb is None
                                     else hb.last_checkpoint),
                    worker_pid=worker_pid)
            # What the worker itself said, on the line the CLI prints.
            # "worker exited with status 1" plus a path is the same shell
            # for a guard refusal and a segfault, so a config the loader
            # deliberately rejected used to read as a crash; the sentence
            # explaining it was one file away.  The capsule still carries
            # the full text and the message still names it.
            headline = _capsule_headline(worker_capsule)
            if headline:
                message = f"{message}: {headline}"
            if monitor_failure_kind in {
                    "prep-timeout", "heartbeat-identity",
                    "heartbeat-regression"}:
                raise SupervisorError(
                    f"{message}; refusing a deterministic relaunch loop "
                    f"(failure capsule: {capsule_path})")
            proposed = None if hb is None else hb.last_checkpoint
            if proposed is None:
                raise SupervisorError(
                    f"{message}; no durable manifest-valid checkpoint is "
                    f"available (failure capsule: {capsule_path})")
            # A restore the worker REFUSED is deterministic: same config
            # digest, same file, same reader, same answer.  Relaunching it
            # burns fresh processes rediscovering one refusal AND each
            # attempt's failure capsule overwrites the one before it, so
            # the evidence for what killed the first worker is destroyed
            # by the recovery.  A crash anywhere else may well clear on a
            # fresh CUDA context, and still gets its attempts.
            if (worker_capsule is not None
                    and str(worker_capsule.get("last_phase", ""))
                    in RESTORE_PHASES
                    and launched_checkpoint is not None
                    and Path(proposed) == Path(launched_checkpoint)):
                raise SupervisorError(
                    f"{message}; the next attempt would restore from the "
                    "same checkpoint this one refused, so it would refuse "
                    "identically -- refusing a deterministic relaunch loop "
                    "and keeping this attempt's evidence (failure capsule: "
                    f"{capsule_path})")
            checkpoint = validate_manifest_checkpoint(proposed)
            if attempts > max_restarts:
                raise SupervisorError(
                    f"{message}; exhausted {max_restarts} fresh-process "
                    f"restart(s) (failure capsule: {capsule_path})")


def _success_output(summary, *, progress_callback=None) -> dict[str, Any]:
    """The success capsule's ``output`` block, hashing nothing twice.

    ``runtime.run_experiment`` already reads and digests every emitted
    frame for the front-door capsule and hands the records on.  Hashing
    them a second time here doubled the finalization cost of every run
    over the same hundreds of GiB, for an identical answer.  An empty
    record set still hashes: the capsule must never silently ship a run
    with no frames in it, and that pass beats once per frame with the
    frame's size, like the run route's own.
    """
    from woof import runtime

    frames = list(getattr(summary, "frame_records", ()) or ())
    if not frames:
        frames = runtime._frame_records(
            summary.wrfout_paths, progress_callback=progress_callback)
    output = {"frames": frames,
              "trajectory_digest": summary.trajectory_digest}
    ensemble = getattr(summary, "ensemble_manifest", None)
    if ensemble is not None:
        output["ensemble_manifest"] = {
            "path": str(ensemble),
            "sha256": summary.ensemble_manifest_sha256}
    return output


def _success_receipts(outdir: Path, summary) -> dict[str, Any]:
    """The success capsule's ``receipts`` block.

    WHY THE FLOORS ARE HERE.  ``woof run`` supervises unless
    ``--no-supervise`` is passed, so this capsule is what a DEFAULT run
    leaves behind.  The run route writes its own capsule into this same
    directory under the same fixed name first, and this one replaces it:
    a two-domain run through `woof go` stated ``moisture_floors_by_domain``
    for both domains and the identical run through `woof run` stated
    ``run_progress`` alone, so whether a forecast's initial vapour was
    modified on the way in was recorded and then written over.  The run
    route hands its own fragment back on the summary and it is carried
    through here, so the replacing capsule cannot say less than the
    capsule it replaced.
    """

    floors = getattr(summary, "moisture_floor_receipts", None)
    # The hydro decision/cost/products the run route stated, kept for the
    # same reason (an ERA5 run's supervised capsule dropped them).
    hydro = getattr(summary, "hydro_receipts", None)
    return {"run_progress": {"path": str((outdir / HEARTBEAT_NAME).resolve())},
            **(dict(floors) if floors else {}),
            **(dict(hydro) if hydro else {})}


def _worker_main(args: argparse.Namespace) -> int:
    """Fresh CUDA worker entry.  Never called inside the supervisor process."""
    config_path = Path(args.config).resolve()
    config_payload = Path(args.config_payload).resolve()
    outdir = Path(args.outdir).resolve()
    run_id = os.environ["WOOF_RUN_ID"]
    digest = os.environ["WOOF_CONFIG_DIGEST"]
    started_at = os.environ["WOOF_STARTED_AT_UTC"]
    gpu = GPUIdentity(os.environ["WOOF_GPU_UUID"],
                      os.environ["WOOF_GPU_DRIVER"],
                      os.environ["WOOF_GPU_NAME"])
    progress = RuntimeHeartbeat(
        outdir / HEARTBEAT_NAME, run_id=run_id, config_sha256=digest,
        started_at_utc=started_at,
        initial_checkpoint=(None if args.restart is None
                            else str(Path(args.restart).resolve())))
    # Publish before importing the runtime/CuPy-facing module graph.  This is
    # the first executable worker action after environment/path setup.
    progress.preparing("worker-start")
    input_hashes: dict[str, Any] = {}
    config_bytes: bytes | None = None
    try:
        # Validate the one captured read before importing either the config
        # loader or the runtime.  The original source path remains metadata
        # and the relative-path base; its mutable bytes are never reopened.
        progress.preparing("validate-config")
        config_bytes = _validated_config_payload_bytes(
            config_payload, digest)
        encoded_inputs = os.environ.get("WOOF_INPUT_HASHES_JSON", "{}")
        try:
            decoded_inputs = json.loads(encoded_inputs)
        except json.JSONDecodeError as exc:
            raise SupervisorError(
                "worker received malformed parent input-hash inventory") from exc
        if not isinstance(decoded_inputs, dict):
            raise SupervisorError(
                "worker input-hash inventory must be an object")
        input_hashes = decoded_inputs
        progress.preparing("import-runtime")
        from woof.case_data import (load_experiment_case_bytes,
                                     remap_case_data_files)
        from woof import runtime

        progress.preparing("load-config")
        exp, data = load_experiment_case_bytes(
            config_bytes, source=str(config_path),
            base_dir=config_path.parent)
        # The captured TOML is immutable, but its forcing globs are resolved
        # against the original directory.  Seal that parse to the exact
        # parent role/path/detail multiset before accepting any CAS remap or
        # entering runtime; a disappearing, appearing, or renamed match must
        # never make the worker run a subset/superset of the capsule inputs.
        progress.preparing("validate-input-inventory")
        _validate_worker_resolved_input_inventory(data, input_hashes)
        encoded_authorities = os.environ.get(INPUT_AUTHORITIES_ENV)
        if encoded_authorities is not None:
            replacements = _validated_worker_input_authorities(
                encoded_authorities, input_hashes)
            data = remap_case_data_files(data, replacements)
        pinned_backend = getattr(args, "preprocess_backend", None)
        if pinned_backend is not None:
            from dataclasses import replace
            data = replace(data, preprocess_backend=pinned_backend)
        progress.preparing("prepare-case")
        from woof.ensemble.door import request_for_payload, production_run_scope
        request = request_for_payload(config_bytes,
            members=getattr(args, "members", None),
            keep_member_files=getattr(args, "keep_member_files", None))
        with production_run_scope(request, output_directory=outdir):
            summary = runtime.run_experiment(
                exp, data, outdir, restart=args.restart,
                progress_callback=progress, health_debug=args.health_debug)
        # The durable success receipt DETERMINISM.md section 7 records as
        # missing: the failure path has carried the input hashes and the GPU
        # identity all along, and a run that succeeded left only a heartbeat.
        #
        # Published BEFORE the terminal record, so ``complete`` means the
        # worker is done rather than nearly done.  It used to be the
        # other way around, and the minutes this capsule spent re-hashing
        # the output set were minutes the run advertised as finished.
        progress.finalizing("success-capsule")
        emit_run_capsule(
            outdir, emission_site="supervisor:success",
            run_context=_success_run_context(
                config_path, digest, input_hashes,
                restart_interval_seconds=(
                    None if exp.restart_interval_s is None
                    else float(exp.restart_interval_s))),
            input_bytes={"entries": input_hashes},
            run_shape={"route": "supervisor:woof run",
                       "domain_count": len(exp.domains),
                       "run_seconds": float(exp.run_seconds)},
            output=_success_output(summary, progress_callback=progress),
            receipts=_success_receipts(outdir, summary),
        )
        progress.complete(summary.completed_seconds)
        return 0
    except BaseException as exc:
        try:
            progress.failed()
            write_failure_capsule(
                outdir / FAILURE_CAPSULE_NAME, run_id=run_id,
                config_path=config_path, config_sha256=digest,
                input_hashes=input_hashes, gpu=gpu,
                config_bytes=config_bytes,
                last_phase=progress.last_phase, last_step=progress.last_step,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                exception_traceback=traceback.format_exc(),
                last_durable_wrfout=progress.last_wrfout,
                last_checkpoint=progress.last_checkpoint,
                worker_pid=os.getpid())
        finally:
            # Re-raising terminates this CUDA process.  The supervisor alone
            # decides whether a manifest-valid checkpoint permits a NEW one.
            raise


def register_cli(subparsers: argparse._SubParsersAction,
                 command: str = "run") -> None:
    """Attach Task-15 run flags without owning ``woof/cli.py``.

    ``command`` names the already-registered subparser to decorate: the
    ``run`` parser by default, and ``resume`` for the sugar command that
    continues a supervised run and therefore takes the identical
    supervision surface.
    """
    from woof.cli_numbers import nonnegative_int, positive_float

    run = subparsers.choices.get(command)
    if run is None:
        raise ValueError(
            f"register_cli requires the existing {command!r} parser")
    from woof.ensemble.door import add_arguments, add_recipe_arguments
    add_arguments(run)
    if command == "run":
        # Only the door that starts a run takes a member-source recipe.
        # Breakage it prevents: ``resume`` and ``branch`` continue ONE
        # checkpointed trajectory and read neither flag, so registered
        # there they were parsed, dropped, and the run continued as N
        # copies of that trajectory under a command line asking for a
        # time-lagged or multi-model ensemble.
        add_recipe_arguments(run)
    run.add_argument(
        "--no-supervise", action="store_true",
        help="run the experiment in this process (escape hatch; disables "
             "fresh-process recovery and exclusive-GPU supervision)")
    run.add_argument("--gpu-uuid", default=None, metavar="GPU-UUID",
                     help="physical GPU UUID to lock (required on multi-GPU hosts)")
    run.add_argument("--supervisor-max-restarts", type=nonnegative_int, default=3,
                     metavar="N", help="fresh-process recovery attempts (default 3)")
    run.add_argument(
        "--prep-timeout", type=positive_float, default=None, metavar="SECONDS",
        help="optional preparation heartbeat timeout; default is no timeout "
             "until integration begins")
    run.add_argument(
        "--allow-shared-gpu", action="store_true",
        help="UNSUPPORTED: permit another substantial CUDA compute context; "
             "device verification and the GPUWM UUID lock remain enforced")
    run.add_argument("--health-debug", action="store_true",
                     help="enable debug phase health attribution hooks")
    run.add_argument(
        "--no-memory-gate", action="store_true", dest="no_memory_gate",
        help="run a case whose priced peak envelope exceeds this card's "
             "free memory anyway, as `woof go --no-memory-gate` does: the "
             "envelope is an upper bound and the card's own allocation then "
             "decides; a model state too big to build at all is still "
             "refused")
    run.add_argument(
        "--directory-input-hash", dest="directory_input_hash",
        default=None, choices=DIRECTORY_HASH_MODES,
        help="how declared directory inputs (the static geography tree) are "
             "bound to this run's identity: 'inventory' (default) uses "
             "relative path, size, and mtime; 'content' reads every file and "
             "uses its SHA-256. Use 'content' when two runs being compared "
             "for byte identity stage their geography separately, and when "
             "an mtime-preserving change to that tree must not go unnoticed "
             f"(docs/public/DETERMINISM.md). Also settable as "
             f"{DIRECTORY_HASH_ENV}.")


def supervise_from_cli(args: argparse.Namespace) -> int:
    # THE process the shell waits on, and the one whose death the reader
    # sees as a bare `Terminated`.  It is also the process that knows the
    # worker log exists: the forecast's own output is redirected into
    # --outdir and never reaches this terminal, so a reader whose run was
    # killed has evidence they have not been told about.
    # BOTH worker logs.  The traceback is on stderr, but the forcing
    # decode's own line -- how many valid times were decoded and what
    # they cost in host RAM -- is printed on stdout, and on this path
    # that is a file too.  Naming only one of them sends a reader whose
    # run was killed by the host to the half that does not carry the
    # figure.
    from woof.progress import ForecastProgress, format_elapsed, line
    progress = ForecastProgress()
    progress.write(f"Starting forecast. Outputs: {Path(args.outdir).resolve()}")
    with report_on_signal(
            f"woof {getattr(args, 'command', 'run')}",
            heartbeat=Path(args.outdir) / HEARTBEAT_NAME,
            logs=(Path(args.outdir) / WORKER_STDERR_NAME.format(attempt=1),
                  Path(args.outdir) / WORKER_STDOUT_NAME.format(attempt=1))):
        result = supervise_experiment(
            args.config, args.outdir, restart=args.restart,
            gpu_uuid=args.gpu_uuid,
            max_restarts=args.supervisor_max_restarts,
            prep_timeout_seconds=args.prep_timeout,
            allow_shared_gpu=args.allow_shared_gpu,
            health_debug=args.health_debug,
            directory_hash=getattr(args, "directory_input_hash", None),
            on_progress=progress,
            preprocess_backend=getattr(args, "preprocess_backend", None),
            members=getattr(args, "members", None),
            keep_member_files=getattr(args, "keep_member_files", None))
    transition_receipt, _ = _current_transition_receipt(
        args.outdir, result.run_id, result.heartbeat.config_digest)
    heartbeat = result.heartbeat
    line(f"Forecast complete: {format_elapsed(heartbeat.model_elapsed_seconds)} "
         f"simulated ({heartbeat.outer_step} steps).")
    line(f"Outputs: {Path(args.outdir).resolve()}")
    if result.attempts > 1:
        line(f"Completed after {result.attempts - 1} automatic recovery attempt(s).")
    if heartbeat.last_checkpoint is not None:
        line(f"Latest checkpoint: {heartbeat.last_checkpoint}")
    if result.stdout_logs:
        line(f"Detailed log: {result.stdout_logs[-1]}")
    if result.stderr_logs:
        line(f"Diagnostics: {result.stderr_logs[-1]}")
    if transition_receipt is not None:
        line(f"Microphysics transitions: {transition_receipt}")
    return 0


def _current_transition_receipt(
        outdir: str | Path, run_id: str, digest: str
        ) -> tuple[Path | None, str | None]:
    """Return only a receipt bound to this supervised run/config."""

    path = (Path(outdir) / MICROPHYSICS_TRANSITION_RECEIPT_NAME).resolve()
    if not path.is_file():
        return None, None
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError):
        return None, None
    if (not isinstance(payload, dict)
            or payload.get("run_id") != run_id
            or payload.get("config_digest") != digest):
        return None, None
    if (payload.get("schema") != "gpuwm.microphysics-transitions/v1"
            or payload.get("status") != "PASS"
            or not isinstance(payload.get("transitions"), list)):
        raise SupervisorError(
            "current-run microphysics transition receipt is malformed")
    return path, hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m woof.supervisor")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--config-payload", type=Path, required=True)
    worker.add_argument("--outdir", type=Path, required=True)
    worker.add_argument("--restart", type=Path, default=None)
    worker.add_argument("--health-debug", action="store_true")
    worker.add_argument("--preprocess-backend", choices=("cuda", "cpu", "auto"),
                        default=None)
    from woof.ensemble.door import add_arguments
    add_arguments(worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        # Installed at the door rather than inside `_worker_main` so it
        # covers the argument parsing and the environment reads too, and
        # so the report is bounded by the process rather than by a try
        # block.  Descriptor 2 here is worker-NN.stderr.log, which is
        # where the reader is being sent.
        with report_on_signal(
                "woof run worker",
                heartbeat=Path(args.outdir) / HEARTBEAT_NAME, worker=True):
            return _worker_main(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "COMPUTE_MEMORY_THRESHOLD_MIB", "DIRECTORY_HASH_DEFAULT",
    "DIRECTORY_HASH_ENV", "DIRECTORY_HASH_MODES", "FAILURE_CAPSULE_NAME",
    "FINALIZATION_FLOOR_BYTES_PER_SECOND",
    "finalization_stale_threshold_seconds",
    "GPUAlreadyLockedError", "GPUFileLock", "GPUIdentity",
    "GPU_LOCK_ROOT_ENV", "INPUT_AUTHORITIES_ENV",
    "GPUPreflightError", "GPUProcess", "HEARTBEAT_NAME",
    "HEARTBEAT_SCHEMA", "Heartbeat", "RollingStepWall", "RuntimeHeartbeat",
    "SupervisorError", "SupervisorResult", "atomic_publish_file",
    "atomic_write_json", "config_digest", "directory_hash_mode",
    "fsync_file", "is_cuda_fatal",
    "parse_compute_apps_output", "preflight_exclusive_gpu",
    "priced_reservation_bytes", "quarantine_file", "read_heartbeat",
    "register_cli",
    "replace_file_with_retry", "resolved_input_hashes", "select_gpu",
    "shared_gpu_admission",
    "SHARED_INPUT_AUTHORITY_ROOT_ENV", "snapshot_resolved_input_files",
    "stale_threshold_seconds", "supervise_experiment",
    "supervise_from_cli", "utc_now", "validate_manifest_checkpoint",
    "write_failure_capsule", "write_heartbeat", "unique_temp_path",
    "WORKER_STDERR_NAME", "WORKER_STDOUT_NAME",
]
