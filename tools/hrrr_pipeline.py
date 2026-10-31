#!/usr/bin/env python3
"""Fail-closed producer/consumer startup for native HRRR bridge hours."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import threading
import time

from woof.hrrr_forecast import validate_hrrr_source_forecast_hours


#: How much of the producer's own output the parent keeps IN MEMORY.
#:
#: The decoder's stderr used to land only in a log file inside the output
#: tree.  When that tree's filesystem filled -- the exact condition the
#: decoder was failing on -- the log stayed 0 bytes, the atomic
#: ``failure.ready`` could not be renamed into place either, and the
#: consumer raised ``pipeline producer exited 1:`` with nothing after the
#: colon.  Both diagnostic channels were on the filesystem that had just
#: failed.  Memory is the one place a full disk cannot erase, so the
#: parent drains the producer's output through a pipe and keeps the tail
#: here; the log file is written from that same stream, best effort, and
#: is never what a failure report depends on.
PRODUCER_OUTPUT_TAIL_BYTES = 256 * 1024

#: How much of the retained tail a failure message quotes.
_REPORTED_TAIL_BYTES = 8000


class HrrrProducerFailure(RuntimeError):
    """A decoder failure reported with everything the parent still knows.

    Carries the structured evidence as :attr:`diagnostics` so a caller
    can record it, and renders the whole of it in ``str()`` so a caller
    that only prints the exception still gets the exit code, the argv,
    the retained output tail, the signal files that did or did not
    appear, and the free space on every filesystem involved.
    """

    def __init__(self, summary: str, diagnostics: dict[str, object]):
        self.diagnostics = diagnostics
        super().__init__(_render_failure(summary, diagnostics))


def _disk_free(path: Path) -> dict[str, object]:
    """Free/total bytes for one path's filesystem, or why they are unknown.

    ``statvfs``/``GetDiskFreeSpaceEx`` read metadata, so this answers on a
    filesystem with no free blocks left -- which is when it is asked.
    """

    for candidate in (path, *path.parents):
        try:
            usage = shutil.disk_usage(candidate)
        except OSError:
            continue
        return {
            "measured_at": str(candidate),
            "free_bytes": int(usage.free),
            "total_bytes": int(usage.total),
        }
    return {"measured_at": str(path), "free_bytes": None, "total_bytes": None}


def _directory_listing(path: Path, limit: int = 64) -> list[str] | str:
    """``name size`` for each entry, or the reason the listing is missing."""

    try:
        entries = sorted(path.iterdir())
    except OSError as error:
        return f"unreadable: {type(error).__name__}: {error}"
    listing = []
    for entry in entries[:limit]:
        try:
            size = entry.stat().st_size if entry.is_file() else None
        except OSError:
            size = None
        listing.append(
            f"{entry.name} ({size} bytes)" if size is not None
            else f"{entry.name}/")
    if len(entries) > limit:
        listing.append(f"... and {len(entries) - limit} more")
    return listing


def _render_failure(summary: str, diagnostics: dict[str, object]) -> str:
    lines = [summary]
    argv = diagnostics.get("argv")
    if argv:
        lines.append("  producer argv: " + " ".join(
            shlex.quote(str(item)) for item in argv))
    tail = diagnostics.get("output_tail")
    if tail:
        lines.append("  producer output (tail, captured through a pipe so a "
                     "full disk cannot erase it):")
        lines.extend(f"    {line}" for line in str(tail).splitlines()[-40:])
    else:
        lines.append("  producer output: EMPTY -- the process wrote nothing "
                     "to stdout/stderr before exiting")
    log = diagnostics.get("log", {})
    if isinstance(log, dict) and log.get("status") != "written":
        lines.append(f"  producer log {log.get('path')}: {log.get('status')}")
    signals = diagnostics.get("signals", {})
    if isinstance(signals, dict):
        if signals.get("failure_ready") is not None:
            lines.append(f"  failure.ready: {signals['failure_ready']}")
        else:
            lines.append(
                "  failure.ready: ABSENT -- the producer either died before "
                "writing it or could not rename it into place")
        lines.append(f"  signals dir {signals.get('path')}: "
                     f"{signals.get('listing')}")
    staging = diagnostics.get("staging")
    if isinstance(staging, dict):
        for path, listing in sorted(staging.items()):
            lines.append(f"  staging {path}: {listing}")
    elif staging:
        lines.append(f"  staging: {staging}")
    for role in ("output", "signals", "log"):
        usage = diagnostics.get(f"{role}_filesystem")
        if isinstance(usage, dict) and usage.get("free_bytes") is not None:
            lines.append(
                f"  {role} filesystem ({usage['measured_at']}): "
                f"{usage['free_bytes']} bytes free of {usage['total_bytes']}")
    removed = diagnostics.get("removed_after_report")
    if removed:
        lines.append(f"  staging removed AFTER this report: {removed}")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_cgroup_limit(path: str) -> int | None:
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def cgroup_capacity() -> dict[str, float | int | None]:
    cpu_quota = None
    try:
        quota_text, period_text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_text != "max":
            cpu_quota = int(quota_text) / int(period_text)
    except (OSError, ValueError):
        pass
    memory_max = _read_cgroup_limit("/sys/fs/cgroup/memory.max")
    memory_current = _read_cgroup_limit("/sys/fs/cgroup/memory.current")
    memory_available = None
    if memory_max is not None and memory_current is not None:
        memory_available = max(0, memory_max - memory_current)
    return {
        "cpu_quota_cores": cpu_quota,
        "memory_max_bytes": memory_max,
        "memory_current_bytes": memory_current,
        "memory_available_bytes": memory_available,
    }


#: The HRRR decode pipeline's real worker ceiling, and the ONLY place it
#: is stated in Python.  It is not a policy choice here: the sealed
#: decoder binary refuses anything outside 1..13 itself
#: (``tools/grib1_bridge/src/bin/hrrr_grib2_bridge.rs:1227`` and
#: ``:1243``, "WORKERS must be in 1..13"), so a caller that accepts more
#: is only promising something the process it spawns will reject.
#: ``tools/prepare_hrrr_wrf.py`` used to advertise 64 and refused at 14
#: only after the static build had already run -- minutes of work spent
#: to reach a bound the argument parser could have named immediately.
#: Raising the ceiling means rebuilding and re-sealing that binary.
MAX_PIPELINE_WORKERS = 13

#: The auto policy's ceiling and floor.
_AUTO_WORKERS = MAX_PIPELINE_WORKERS
_AUTO_FALLBACK_WORKERS = 8

_WORKER_RANGE_TEXT = (
    f"pipeline workers must be 1..{MAX_PIPELINE_WORKERS} or 'auto'")


#: Operator override for the decoder's worker count, for a shared host.
#: Breakage it prevents: on a shared 8-card box (Vast V8X, 2026-10-27) the
#: HRRR decoder ran 8 workers at ~19-24 GB resident because ``woof go``
#: passes 8 explicitly; the kernel OOM killer took it 20+ times while other
#: jobs held the memory, and every kill threw away a 4-10 min preparation.
#: The value is any number the range accepts, or ``auto``; the receipt
#: records that it came from the environment.
PIPELINE_WORKERS_ENV = "WOOF_PIPELINE_WORKERS"


def resolve_workers(value: str) -> tuple[int, dict[str, object]]:
    capacity = cgroup_capacity()
    override = os.environ.get(PIPELINE_WORKERS_ENV, "").strip()
    # the receipt's "requested" stays the caller's own value: the preparation
    # validator (prepare_hrrr_wrf._validated_worker_receipts) checks it against
    # the count it passed, so an override that rewrote it failed every
    # preparation after ~10 min with "HRRR decoder worker receipt is malformed"
    # (Vast V8X 2026-10-27); the override is recorded beside it.
    requested = str(value).strip().lower()
    if override:
        value = override
    text = str(value).strip().lower()
    if text == "auto":
        cpu = capacity["cpu_quota_cores"]
        available = capacity["memory_available_bytes"]
        enough_cpu = cpu is None or float(cpu) >= 26.0
        enough_memory = available is None or int(available) >= 24 * 1024 ** 3
        workers = (_AUTO_WORKERS if enough_cpu and enough_memory
                   else _AUTO_FALLBACK_WORKERS)
        policy = (f"auto{_AUTO_WORKERS}_if_cpu_ge_26_and_available_memory_"
                  f"ge_24GiB_else{_AUTO_FALLBACK_WORKERS}")
    else:
        try:
            workers = int(text)
        except ValueError as error:
            raise ValueError(_WORKER_RANGE_TEXT) from error
        if workers not in range(1, MAX_PIPELINE_WORKERS + 1):
            raise ValueError(_WORKER_RANGE_TEXT)
        policy = "explicit"
    receipt = {"requested": requested, "selected": workers,
               "policy": policy, "cgroup": capacity}
    if override:
        receipt["policy"] = policy + f" ({PIPELINE_WORKERS_ENV})"
        receipt["override"] = {"env": PIPELINE_WORKERS_ENV, "value": text}
    return workers, receipt


def _parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            digest, relative = raw.split(None, 1)
        except ValueError as error:
            raise ValueError(
                f"malformed source manifest line {line_number}") from error
        relative = relative.removeprefix("*").removeprefix("./")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe source manifest path {relative!r}")
        if relative in entries:
            raise ValueError(f"duplicate source manifest path {relative!r}")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF"
                                   for ch in digest):
            raise ValueError(f"invalid source digest on line {line_number}")
        entries[relative] = digest.lower()
    if not entries:
        raise ValueError("source manifest is empty")
    return entries


def _parse_series(path: Path) -> list[tuple]:
    rows = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) < 3:
            raise ValueError("native series requires HOUR, atmosphere and soil paths")
        from woof.ingest.native_supplements import supplement_bindings
        donors = supplement_bindings(fields[3:], base=path.parent)
        resolve = lambda value: (path.parent / value).resolve()
        rows.append((int(fields[0]), resolve(fields[1]), resolve(fields[2]),
                     *(donor for _, donor in donors)))
    observed = [row[0] for row in rows]
    try:
        validate_hrrr_source_forecast_hours(observed)
    except (TypeError, ValueError) as error:
        raise ValueError(f"pipeline series is invalid: {error}") from error
    return rows


def verify_source_tree(*, source_root: Path, manifest: Path,
                       expected_manifest_sha256: str, series: Path,
                       workers: int) -> dict[str, object]:
    started = time.perf_counter()
    actual_manifest_sha = _sha256(manifest)
    if actual_manifest_sha != expected_manifest_sha256.lower():
        raise ValueError(
            "source manifest hash mismatch: "
            f"expected {expected_manifest_sha256.lower()}, got {actual_manifest_sha}")
    entries = _parse_manifest(manifest)
    paths = [(relative, source_root / relative, digest)
             for relative, digest in entries.items()]
    missing = [str(path) for _, path, _ in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest payloads missing: {missing[:8]}")

    def verify(item):
        relative, path, expected = item
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"source payload hash mismatch for {relative}: "
                f"expected {expected}, got {actual}")
        return path.stat().st_size

    hash_workers = max(1, min(workers, MAX_PIPELINE_WORKERS))
    with ThreadPoolExecutor(max_workers=hash_workers) as pool:
        sizes = list(pool.map(verify, paths))
    series_rows = _parse_series(series)
    declared = {str((source_root / relative).resolve()) for relative in entries}
    for row in series_rows:
        for path in row[1:]:
            if str(path.resolve()) not in declared:
                raise ValueError(f"series input is not bound by source manifest: {path}")
    return {
        "status": "PASS",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": actual_manifest_sha,
        "payload_file_count": len(paths),
        "payload_bytes": int(sum(sizes)),
        "series_sha256": _sha256(series),
        **({"supplement_bindings": [
            {"forecast_hour": row[0], "field": "PMSL", "path": str(path),
             "sha256": next(digest for relative, digest in entries.items()
                            if (source_root / relative).resolve() == path.resolve())}
            for row in series_rows for path in row[3:]]}
           if any(len(row) > 3 for row in series_rows) else {}),
        "forecast_hours": [row[0] for row in series_rows],
        "source_forecast_hours": [row[0] for row in series_rows],
        "model_forcing_hours": list(range(len(series_rows))),
        "wall_seconds": time.perf_counter() - started,
        "hash_workers": hash_workers,
    }


#: The two objects of one posted native HRRR lead, by the role its marker
#: names them under (``woof.fetch_as_posted.legacy_objects``), and the
#: series column each is.
POSTED_LEAD_ROLES = (("atmosphere", 1), ("soil", 2))


def admission_name(hour: int) -> str:
    """The file the decoder waits for before it reads lead ``hour``."""

    return f"f{int(hour):02d}.admitted"


class PostedLeadAdmitter:
    """Admit each lead of a native series to the decoder as it posts (A136 L7c).

    ``posted`` is the preparation's :class:`woof.ingest.boundary_stream.PostedLeads`
    over the as-posted fetch's ``posting/`` folder; ``series`` the native
    series, every row naming the two objects of one lead in the fetch's
    folder.  For each lead in order, :meth:`admit` waits for its posted
    marker, holds both files to the marker (name, bytes, sha256), and
    writes ``ADMIT_DIR/fNN.admitted`` naming the lead's series files, which
    is what ``hrrr_grib2_bridge --series-workers-posted`` waits for before
    it reads that lead.  So the decoder reads no byte a marker did not
    name, in place of :func:`verify_source_tree` hashing every lead up
    front, which a lead not posted yet cannot pass.  :meth:`start` admits
    on its own thread, so the decoder runs ahead of the preparation as
    leads post; a failure (a late lead, a file that is not its marker's) is
    raised to whoever waits on a lead not admitted.
    """

    def __init__(self, *, posted, series: Path, admissions: Path,
                 hash_workers: int = 2):
        self.posted = posted
        self.series = Path(series)
        self.admissions = Path(admissions)
        self.hash_workers = max(1, int(hash_workers))
        self.out = Path(posted.folder).parent.resolve()
        rows = []
        for raw in self.series.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            fields = raw.split("\t")
            if len(fields) != 3:
                # Breakage it prevents: a donor column binds bytes the
                # fetch never posted, and the decoder decodes a donor
                # series whole (--series-workers-ready).
                raise ValueError(
                    "an as-posted native series names each lead's two "
                    "posted objects and nothing else; this one has "
                    f"{len(fields) - 1} columns after the hour")
            hour = int(fields[0])
            paths = (fields[1], fields[2])
            for value in paths:
                if not Path(value).is_absolute() \
                        or Path(value).parent.resolve() != self.out:
                    raise ValueError(
                        f"an as-posted native series reads the fetch's own "
                        f"folder {self.out}; {value} is not in it")
            rows.append((hour, paths))
        validate_hrrr_source_forecast_hours([row[0] for row in rows])
        self.rows = tuple(rows)
        self.markers: dict[int, dict] = {}
        self.leads: dict[int, dict] = {}
        self._admitted = {hour: threading.Event() for hour, _ in rows}
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self.admissions.mkdir(parents=True, exist_ok=False)

    @property
    def hours(self) -> tuple[int, ...]:
        return tuple(hour for hour, _ in self.rows)

    def admit(self, hour: int) -> dict:
        """Wait for lead ``hour``'s marker, hold its files to it, admit it."""

        hour = int(hour)
        paths = dict(self.rows)[hour]
        # The thread's wait ends with the admitter (stop), not only when
        # the lead posts or the fetch fails.
        marker = self.posted.wait(hour, stop=self._stopped)
        objects = {}
        for item in marker.get("objects") or ():
            role = str(item.get("role"))
            if role in objects:
                raise ValueError(
                    f"the posted marker of f{hour:03d} names two {role} "
                    "objects; a native HRRR lead is one of each")
            objects[role] = item
        if set(objects) != {role for role, _ in POSTED_LEAD_ROLES}:
            raise ValueError(
                f"the posted marker of f{hour:03d} names {sorted(objects)}, "
                "not the atmosphere and soil objects a native HRRR lead is")

        def held(role_column):
            role, column = role_column
            path = Path(paths[column - 1])
            item = objects[role]
            if item.get("name") != path.name:
                raise ValueError(
                    f"the posted {role} object of f{hour:03d} is "
                    f"{item.get('name')}, not the series' {path.name}")
            size = path.stat().st_size
            digest = _sha256(path)
            if digest != item.get("sha256") or size != item.get("bytes"):
                raise ValueError(
                    f"{path} is {size} bytes {digest}, not the {role} object "
                    f"f{hour:03d}'s posted marker names "
                    f"({item.get('bytes')} bytes {item.get('sha256')})")
            return {"role": role, "name": path.name, "path": str(path),
                    "bytes": size, "sha256": digest}

        with ThreadPoolExecutor(max_workers=self.hash_workers) as pool:
            files = list(pool.map(held, POSTED_LEAD_ROLES))
        target = self.admissions / admission_name(hour)
        temporary = target.with_suffix(".tmp")
        temporary.write_text("".join((
            f"forecast_hour\t{hour}\n",
            f"atmosphere\t{paths[0]}\n",
            f"soil\t{paths[1]}\n")), encoding="utf-8")
        os.replace(temporary, target)
        with self._lock:
            self.markers[hour] = dict(marker)
            self.leads[hour] = {"forecast_hour": hour, "files": files}
        self._admitted[hour].set()
        return marker

    def _run(self) -> None:
        try:
            for hour in self.hours:
                if self._stopped.is_set():
                    return
                if not self._admitted[hour].is_set():
                    self.admit(hour)
        except BaseException as error:  # noqa: BLE001 - raised by wait()
            with self._lock:
                self._error = error
            for event in self._admitted.values():
                event.set()

    def start(self) -> None:
        """Admit every lead in order on a thread of its own."""

        self._thread = threading.Thread(
            target=self._run, name="hrrr-posted-admitter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """End the admitter: its wait for a lead not posted ends too.

        Breakage this prevents: ``stop`` used to be read only between
        leads, so a thread waiting for a lead that never posted (the
        decoder failed, the preparation gave up) went on polling the
        posting folder for the life of the process; in the public CI
        battery it was still reading a finished test's ``f007.json``
        when a later test recorded every file opened.  The thread is
        joined, so no read of the posting folder or a lead's files
        follows the return: at most one poll and the lead being held to
        its marker finish first.
        """

        self._stopped.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def wait(self, hour: int, *, timeout: float | None = None) -> dict | None:
        """Block until lead ``hour`` is admitted; its marker, or the failure.

        With ``timeout``, ``None`` when the lead is not admitted by then, so
        a caller can look at the decoder between waits.
        """

        hour = int(hour)
        if not self._admitted[hour].wait(timeout):
            return None
        with self._lock:
            if hour in self.markers:
                return self.markers[hour]
            error = self._error
        raise error if error is not None else RuntimeError(
            f"f{hour:02d} was not admitted")

    def failure(self) -> BaseException | None:
        with self._lock:
            return self._error

    def source_manifest_text(self) -> str:
        """The fetch's ``SHA256SUMS`` for these leads, from their markers.

        One ``sha256  name`` row per posted object, sorted by name, as the
        legacy fetch writes it, so it is byte for byte the one-shot fetch's
        manifest of the same window.
        """

        rows = {}
        for hour in self.hours:
            for item in self.wait(hour).get("objects") or ():
                rows[str(item["name"])] = str(item["sha256"])
        return "".join(f"{digest}  {name}\n"
                       for name, digest in sorted(rows.items()))

    def receipt(self, manifest: Path) -> dict[str, object]:
        """What :func:`verify_source_tree` returns, for leads held as they posted."""

        leads = [self.leads[hour] for hour in self.hours]
        return {
            "status": "PASS",
            "mode": "as_posted",
            "manifest": str(Path(manifest).resolve()),
            "manifest_sha256": _sha256(Path(manifest)),
            "payload_file_count": sum(len(lead["files"]) for lead in leads),
            "payload_bytes": int(sum(item["bytes"] for lead in leads
                                     for item in lead["files"])),
            "series_sha256": _sha256(self.series),
            "forecast_hours": list(self.hours),
            "source_forecast_hours": list(self.hours),
            "model_forcing_hours": list(range(len(self.hours))),
            "leads": leads,
        }


def _read_tsv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        key, value = raw.split("\t", 1)
        if key in values:
            raise ValueError(f"duplicate ready-receipt key {key!r}")
        values[key] = value
    return values


def _process_rss_bytes(pid: int) -> int:
    try:
        for raw in Path(f"/proc/{pid}/status").read_text().splitlines():
            if raw.startswith("VmRSS:"):
                return int(raw.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _pid_is_live(pid: int) -> bool:
    if isinstance(pid, bool) or pid < 1:
        raise ValueError("producer PID must be positive")
    if os.name == "nt":
        # os.kill(pid, 0) is not a non-mutating liveness probe on Windows:
        # CPython routes ordinary signal values through TerminateProcess.
        # Query a synchronize handle and sample it without signalling.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        wait_for_single_object = kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        wait_for_single_object.restype = wintypes.DWORD
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL

        synchronize = 0x00100000
        wait_object_0 = 0x00000000
        wait_timeout = 0x00000102
        wait_failed = 0xFFFFFFFF
        handle = open_process(synchronize, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: PID is absent.
                return False
            if error == 5:  # ERROR_ACCESS_DENIED: conservatively live.
                return True
            raise ctypes.WinError(error)
        try:
            result = wait_for_single_object(handle, 0)
            if result == wait_timeout:
                return True
            if result == wait_object_0:
                return False
            if result == wait_failed:
                raise ctypes.WinError(ctypes.get_last_error())
            raise OSError(f"unexpected Windows process wait result: {result}")
        finally:
            close_handle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _path_is_reparse_point(path: Path) -> bool:
    """Recognize links/junctions without relying on Python 3.12 Path APIs."""

    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        return True
    # ``Path.is_junction`` was added in Python 3.12.  Python 3.11 on Windows
    # still exposes both fields below on ``stat_result``; FILE_ATTRIBUTE_* is
    # not guaranteed to exist in ``stat`` on non-Windows hosts, so retain the
    # documented numeric value as the portable inspection fallback.
    file_attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_attribute = int(
        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400))
    reparse_tag = int(getattr(metadata, "st_reparse_tag", 0))
    return bool(file_attributes & reparse_attribute or reparse_tag)


def _owned_producer_tree(output: Path, role: str, pid: int) -> Path:
    if role not in {"partial", "publish"}:
        raise ValueError(f"invalid HRRR producer tree role {role!r}")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ValueError("producer PID must be positive")
    output = Path(output)
    parent = output.parent.resolve()
    stem = output.name
    if not stem or stem in {".", ".."}:
        raise ValueError("pipeline output must have a filename")
    return parent / f".{stem}.{role}-{pid}"


def _remove_owned_producer_tree(
        output: Path, role: str, pid: int) -> str | None:
    """Remove one exact PID-owned directory without following a reparse root."""

    candidate = _owned_producer_tree(output, role, pid)
    if not os.path.lexists(candidate):
        return None
    if _path_is_reparse_point(candidate):
        raise RuntimeError(
            f"unsafe owned HRRR {role} candidate is a symlink/reparse point: "
            f"{candidate}")
    if not candidate.is_dir():
        raise RuntimeError(
            f"owned HRRR {role} candidate is not a directory: {candidate}")
    resolved = str(candidate.resolve())
    # On fd-capable POSIX builds shutil.rmtree uses its symlink-attack-safe
    # implementation.  Python's Windows implementation does not traverse
    # directory junctions; the explicit lstat/reparse rejection above closes
    # the Python-3.11 top-level junction gap before deletion begins.
    shutil.rmtree(candidate)
    return resolved


def _cleanup_stale_producer_trees(output: Path) -> tuple[str, ...]:
    """Remove only dead-PID staging/publish trees for one exact output."""

    parent = output.parent.resolve()
    stem = output.name
    if not stem or stem in {".", ".."}:
        raise ValueError("pipeline output must have a filename")
    removed = []
    for role in ("partial", "publish"):
        prefix = f".{stem}.{role}-"
        for candidate in sorted(parent.glob(prefix + "*")):
            if candidate.parent != parent or _path_is_reparse_point(candidate):
                raise RuntimeError(
                    f"unsafe stale HRRR {role} candidate: {candidate}")
            suffix = candidate.name.removeprefix(prefix)
            if not suffix.isdecimal() or not candidate.is_dir():
                raise RuntimeError(
                    f"malformed stale HRRR {role} candidate: {candidate}")
            pid = int(suffix)
            if _pid_is_live(pid):
                raise RuntimeError(
                    f"live HRRR producer PID {pid} owns {candidate}")
            removed_path = _remove_owned_producer_tree(output, role, pid)
            if removed_path is None:
                raise RuntimeError(
                    f"stale HRRR {role} candidate vanished during cleanup: "
                    f"{candidate}")
            removed.append(removed_path)
    return tuple(removed)


class HrrrPipelineProducer:
    """Own one decoder process and expose only atomically ready hours."""

    def __init__(self, *, decoder: Path, series: Path, output: Path,
                 signals: Path, cycle: str, window: tuple[int, int, int, int],
                 workers: str, log: Path, admissions: Path | None = None,
                 analyzed_aerosol: bool = False, extras: tuple[str, ...] = ()):
        #: As posted (:class:`PostedLeadAdmitter`): the folder the decoder
        #: waits in for each lead's admission.
        self.admissions = admissions
        #: The configuration's preparation reads HRRR's analyzed aerosol
        #: pair (:func:`woof.preparation_assets.analyzed_aerosol_domains`),
        #: so the decoder selects QNWFA/QNIFA when a lead publishes them
        #: (``--analyzed-aerosol``).  Off, it never does, 2.8.6's
        #: inventory: NCEP masks PMTF with a GRIB2 bitmap at some leads of
        #: some cycles, and a pair read unasked at the reference lead
        #: stopped an as-posted series at the first masked later lead.
        self.analyzed_aerosol = bool(analyzed_aerosol)
        #: Records the caller's configuration reads beyond the default
        #: inventory (``--extras``: the decoder's request-only tables, e.g.
        #: HRRR's own number concentrations QNC/QNI/QNR and its 2 m RH for
        #: an HRRR start).  Requested and unpublished is the decoder's refusal.
        self.extras = tuple(str(name) for name in extras)
        self.decoder = decoder
        self.series = series
        series_rows = _parse_series(series)
        self.series_hours = tuple(row[0] for row in series_rows)
        self.hour_payload_files = {
            row[0]: 24 + int(len(row) > 3) for row in series_rows}
        self._base_hour_payload_files = dict(self.hour_payload_files)
        parsed_cycle = datetime.strptime(
            cycle.replace("T", " ").removesuffix("Z"),
            "%Y-%m-%d %H:%M:%S")
        validate_hrrr_source_forecast_hours(
            self.series_hours, cycle=parsed_cycle)
        self.series_count = len(self.series_hours)
        self.output = output
        self.signals = signals
        self.cycle = cycle
        self.window = tuple(map(int, window))
        self.workers, self.worker_receipt = resolve_workers(workers)
        self.log = log
        self.process: subprocess.Popen | None = None
        self.argv: tuple[str, ...] = ()
        self.started = None
        self.preflight = None
        self.producer_ready_seconds: dict[int, float] = {}
        self.consumer_observed_seconds: dict[int, float] = {}
        self.max_rss_bytes = 0
        self.stale_cleanup: tuple[str, ...] = ()
        self.cancel_cleanup: dict[str, str | None] = {}
        #: The producer's own output, retained in this process's memory.
        self._output_tail = bytearray()
        self._output_lock = threading.Lock()
        self._drain: threading.Thread | None = None
        self._log_status = "not opened"
        #: Diagnostics captured at the first failure, BEFORE any unwind
        #: removes the trees they describe.
        self.failure_diagnostics: dict[str, object] | None = None

    # -- producer output ------------------------------------------------
    #
    # Read through a pipe and kept in memory, mirrored to the log file
    # only as a convenience.  A log inside the output tree is exactly as
    # writable as the tree the producer just failed to write, so it is
    # never the channel a failure report depends on.

    def _record_output(self, chunk: bytes) -> None:
        with self._output_lock:
            self._output_tail.extend(chunk)
            excess = len(self._output_tail) - PRODUCER_OUTPUT_TAIL_BYTES
            if excess > 0:
                del self._output_tail[:excess]

    def output_tail(self, limit: int = _REPORTED_TAIL_BYTES) -> str:
        with self._output_lock:
            data = bytes(self._output_tail[-limit:])
        return data.decode("utf-8", errors="replace")

    def _pump(self, stream, log_stream) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                self._record_output(chunk)
                if log_stream is not None:
                    try:
                        log_stream.write(chunk)
                        log_stream.flush()
                    except OSError as error:
                        name = errno.errorcode.get(
                            error.errno, str(error.errno))
                        self._log_status = (
                            f"write failed ({name}): {error}; the tail above "
                            "came from memory instead")
                        try:
                            log_stream.close()
                        except OSError:
                            pass
                        log_stream = None
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass
            if log_stream is not None:
                try:
                    log_stream.close()
                except OSError:
                    pass
                if self._log_status == "not opened":
                    self._log_status = "written"

    def decoder_argv(self) -> tuple[str, ...]:
        """The decoder command this producer launches."""

        i0, i1, j0, j1 = self.window
        request = ("--analyzed-aerosol",) if self.analyzed_aerosol else ()
        if self.extras:
            request += ("--extras", ",".join(self.extras))
        if self.admissions is None:
            return (
                str(self.decoder), *request, "--series-workers-ready",
                str(self.workers), str(self.series), str(self.output),
                str(self.signals), self.cycle,
                str(i0), str(i1), str(j0), str(j1),
            )
        return (
            str(self.decoder), *request, "--series-workers-posted",
            str(self.workers), str(self.series), str(self.output),
            str(self.signals), str(self.admissions), self.cycle,
            str(i0), str(i1), str(j0), str(j1),
        )

    def start(self) -> None:
        for path, label in ((self.output, "output"), (self.signals, "signals")):
            if path.exists():
                raise FileExistsError(f"pipeline {label} already exists: {path}")
        if not self.decoder.is_file() or not os.access(self.decoder, os.X_OK):
            raise FileNotFoundError(f"pipeline decoder is not executable: {self.decoder}")
        self.stale_cleanup = _cleanup_stale_producer_trees(self.output)
        # Best effort, both of them: a log this process cannot open is a
        # missing convenience, not a reason to refuse to run, and it must
        # never be the reason a failure has no message.
        log_stream = None
        try:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            log_stream = self.log.open("wb")
        except OSError as error:
            self._log_status = (
                f"unavailable: {type(error).__name__}: {error}")
        self.argv = self.decoder_argv()
        self.started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                list(self.argv), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except BaseException:
            if log_stream is not None:
                log_stream.close()
            raise
        self._drain = threading.Thread(
            target=self._pump, args=(self.process.stdout, log_stream),
            name="hrrr-producer-output", daemon=True)
        self._drain.start()

    def _sample(self) -> None:
        if self.process is not None:
            self.max_rss_bytes = max(
                self.max_rss_bytes, _process_rss_bytes(self.process.pid))

    # -- failure evidence ------------------------------------------------

    def diagnostics(self) -> dict[str, object]:
        """Everything this process knows about the producer, right now.

        Every reader here is metadata-only or in-memory, so the answer is
        the same on a filesystem with zero free blocks -- which is the
        state it exists to describe.
        """

        failure = self.signals / "failure.ready"
        failure_values: dict[str, str] | str | None = None
        if failure.is_file():
            try:
                failure_values = _read_tsv(failure)
            except (OSError, ValueError) as error:
                failure_values = f"unreadable: {type(error).__name__}: {error}"
        try:
            log_size = self.log.stat().st_size
        except OSError:
            log_size = None
        # The staging tree is what the consumer's unwind removes, so it is
        # described here -- while it still exists -- rather than merely
        # named in a message printed after it is gone.
        staging: dict[str, object] = {}
        if self.process is not None:
            for role in ("partial", "publish"):
                candidate = _owned_producer_tree(self.output, role,
                                                 self.process.pid)
                if os.path.lexists(candidate):
                    staging[str(candidate)] = _directory_listing(candidate)
        return {
            "argv": list(self.argv),
            "staging": staging or "no PID-owned staging/publish tree present",
            "pid": None if self.process is None else self.process.pid,
            "returncode": None if self.process is None else self.process.poll(),
            "output_tail": self.output_tail(),
            "log": {"path": str(self.log), "status": self._log_status,
                    "size_bytes": log_size},
            "signals": {
                "path": str(self.signals),
                "listing": _directory_listing(self.signals),
                "failure_ready": failure_values,
                "failure_tmp_present": (
                    self.signals / "failure.ready.tmp").exists(),
            },
            "output_filesystem": _disk_free(self.output.parent),
            "signals_filesystem": _disk_free(self.signals.parent),
            "log_filesystem": _disk_free(self.log.parent),
        }

    def _capture_failure(self) -> dict[str, object]:
        """Snapshot the evidence once, before any unwind can remove it."""

        if self.failure_diagnostics is None:
            self.failure_diagnostics = self.diagnostics()
        return self.failure_diagnostics

    def _failure(self, summary: str) -> HrrrProducerFailure:
        return HrrrProducerFailure(summary, self._capture_failure())

    def check(self) -> None:
        """Raise the producer's failure now, if it has failed."""

        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self.process is None:
            raise RuntimeError("pipeline producer was not started")
        self._sample()
        failure = self.signals / "failure.ready"
        if failure.is_file():
            raise self._failure("pipeline producer failed")
        returncode = self.process.poll()
        if returncode not in (None, 0):
            raise self._failure(f"pipeline producer exited {returncode}")

    def _validated_staging_root(self) -> Path:
        if self.process is None or self.preflight is None:
            raise RuntimeError("pipeline preflight was not consumed")
        try:
            root = Path(self.preflight["staging_root"])
        except KeyError as error:
            raise ValueError(
                "pipeline preflight omitted the staging root") from error
        expected = (
            self.output.parent.resolve()
            / f".{self.output.name}.partial-{self.process.pid}")
        if (not os.path.lexists(root) or _path_is_reparse_point(root)
                or root.resolve() != expected or not root.is_dir()):
            raise ValueError(
                f"unsafe pipeline staging root: {root}; expected {expected}")
        return root

    def wait_preflight(self, timeout: float = 600.0) -> dict[str, str]:
        deadline = time.monotonic() + timeout
        ready = self.signals / "preflight.ready"
        while not ready.is_file():
            self._raise_if_failed()
            if time.monotonic() >= deadline:
                raise TimeoutError("pipeline inventory preflight timed out")
            time.sleep(0.02)
        values = _read_tsv(ready)
        if (values.get("status") != "PASS"
                or values.get("series_count") != str(self.series_count)):
            raise ValueError(f"invalid pipeline preflight receipt: {values}")
        if values.get("staging_retention") != "until_consumer_finish":
            raise ValueError(
                "pipeline producer does not guarantee a stable staging root")
        canonical = values.get("canonical_output")
        if canonical is None or Path(canonical).resolve() != self.output.resolve():
            raise ValueError(
                "pipeline producer canonical output differs from the request")
        if values.get("workers") != str(self.workers):
            raise ValueError(
                "pipeline producer worker receipt differs from the request")
        producer_seconds = float(values.get("producer_elapsed_seconds", "nan"))
        if not math.isfinite(producer_seconds) or producer_seconds < 0.0:
            raise ValueError(
                f"invalid producer preflight timestamp: {producer_seconds!r}")
        self.preflight = values
        staging_root = self._validated_staging_root()
        from woof.ingest.native_supplements import (
            gate_optional_hybrid_fields, gate_optional_surface_fields,
            gate_soil_surface_fields)
        staged_gate = _read_tsv(staging_root / "gate.txt")
        extra_fields = tuple(name for name in staged_gate.get("extra_fields", "").split(",") if name)
        if sorted(extra_fields) != sorted(self.extras):
            raise ValueError(
                f"pipeline producer gate declares extras {extra_fields}, not the requested {self.extras}")
        soil_surface_count = (len(gate_soil_surface_fields(staged_gate))
                              + len(gate_optional_hybrid_fields(staged_gate))
                              + len(gate_optional_surface_fields(staged_gate))
                              + len(extra_fields))
        self.hour_payload_files = {
            hour: count + soil_surface_count
            for hour, count in self._base_hour_payload_files.items()}
        return values

    def wait_hour(self, hour: int, timeout: float = 600.0) -> Path:
        if self.preflight is None:
            raise RuntimeError("pipeline preflight was not consumed")
        deadline = time.monotonic() + timeout
        ready = self.signals / f"f{hour:02d}.ready"
        while not ready.is_file():
            self._raise_if_failed()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"pipeline f{hour:02d} decode timed out")
            time.sleep(0.02)
        values = _read_tsv(ready)
        if (values.get("status") != "PASS"
                or int(values.get("forecast_hour", -1)) != hour
                or values.get("payload_files") != str(self.hour_payload_files[hour])):
            raise ValueError(f"invalid pipeline f{hour:02d} receipt: {values}")
        producer_seconds = float(values.get("producer_elapsed_seconds", "nan"))
        if not math.isfinite(producer_seconds) or producer_seconds < 0.0:
            raise ValueError(
                f"invalid producer f{hour:02d} timestamp: {producer_seconds!r}")
        self.producer_ready_seconds[hour] = producer_seconds
        self.consumer_observed_seconds[hour] = (
            time.perf_counter() - float(self.started))
        # Hourly READY receipts are consumed only from the stable staging root.
        # The producer retains this tree while atomically publishing a separate
        # hard-linked/copy fallback canonical tree.  Selecting canonical output
        # here reintroduces a root-switch TOCTOU between this return and the
        # caller's subsequent per-field opens.
        root = self._validated_staging_root()
        for role in ("atmosphere", "soil"):
            directory = root / f"{role}-f{hour:02d}"
            if (not os.path.lexists(directory)
                    or _path_is_reparse_point(directory)
                    or not directory.is_dir()):
                raise FileNotFoundError(
                    f"ready f{hour:02d} committed {role} directory vanished")
        return root

    def finish(self, timeout: float = 600.0) -> dict[str, object]:
        if self.process is None:
            raise RuntimeError("pipeline producer was not started")
        try:
            returncode = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            self.cancel()
            raise TimeoutError("pipeline producer completion timed out") from error
        self._sample()
        self._join_drain()
        if returncode != 0:
            raise self._failure(f"pipeline producer exited {returncode}")
        complete = self.signals / "complete.ready"
        if not complete.is_file() or not self.output.is_dir():
            # Exit 0 with no receipt: the same evidence a nonzero exit
            # gets, for the same reason -- a completion receipt that could
            # not be renamed into place is what a full signals filesystem
            # looks like from here.
            raise self._failure(
                "pipeline producer exited 0 but omitted its atomic "
                "completion receipt")
        complete_values = _read_tsv(complete)
        if (complete_values.get("status") != "PASS"
                or complete_values.get("series_count") != str(self.series_count)):
            raise ValueError(
                f"invalid pipeline completion receipt: {complete_values}")
        if complete_values.get("staging_retained") != "true":
            raise ValueError(
                "pipeline producer did not retain staging through consumer finish")
        canonical = complete_values.get("canonical_output")
        if canonical is None or Path(canonical).resolve() != self.output.resolve():
            raise ValueError(
                "pipeline completion canonical output differs from the request")
        producer_wall_seconds = float(
            complete_values.get("producer_elapsed_seconds", "nan"))
        if (not math.isfinite(producer_wall_seconds)
                or producer_wall_seconds < 0.0):
            raise ValueError(
                f"invalid producer completion timestamp: {producer_wall_seconds!r}")
        staging_cleanup = "not_required"
        self._validated_staging_root()
        try:
            removed = _remove_owned_producer_tree(
                self.output, "partial", self.process.pid)
            if removed is None:
                raise FileNotFoundError("retained pipeline staging vanished")
        except (OSError, RuntimeError) as error:
            staging_cleanup = f"retained:{type(error).__name__}:{error}"
        else:
            staging_cleanup = "removed"
        from woof.ingest.hrrr import aerosol_fill_receipt
        return {
            "status": "PASS",
            "aerosol_missing": aerosol_fill_receipt(_read_tsv(self.output / "gate.txt")),
            "source_forecast_hours": list(self.series_hours),
            "model_forcing_hours": list(range(self.series_count)),
            "workers": self.worker_receipt,
            "preflight": self.preflight,
            "producer_ready_seconds_from_decoder_start": {
                f"f{hour:02d}": value
                for hour, value in sorted(self.producer_ready_seconds.items())},
            "consumer_observed_seconds_from_decoder_start": {
                f"f{hour:02d}": value
                for hour, value in sorted(self.consumer_observed_seconds.items())},
            "decoder_wall_seconds": producer_wall_seconds,
            "consumer_finish_observed_seconds_from_decoder_start": (
                time.perf_counter() - float(self.started)),
            "complete_receipt": complete_values,
            "staging_cleanup": staging_cleanup,
            "startup_stale_cleanup": list(self.stale_cleanup),
            "decoder_max_rss_bytes_observed": self.max_rss_bytes,
            "log": str(self.log.resolve()),
            "log_status": self._log_status,
            "argv": list(self.argv),
        }

    def _join_drain(self, timeout: float = 10.0) -> None:
        if self._drain is not None and self._drain.is_alive():
            self._drain.join(timeout)

    def cancel(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                # The producer exited between poll() and terminate(); wait()
                # still reaps it before any owned path is considered.
                pass
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10.0)
        if self.process.poll() is None:
            raise RuntimeError(
                "pipeline producer remained live after cancellation")
        self._join_drain()
        # EVIDENCE BEFORE CLEANUP.  This unwind is what used to delete the
        # staging tree, the signal directory contents and the decoder log
        # while the caller was still trying to explain why the producer
        # died -- so the snapshot is taken here, first, and every path it
        # names is described (size, listing, free space) rather than
        # merely pointed at.  Removal still happens afterwards: on a full
        # filesystem, freeing the space IS the remedy, and the report
        # already says what was in it.
        self._capture_failure()
        # Never remove a tree while its native producer may still be creating
        # or opening fields.  This also covers the common consumer-failure
        # case where the producer exited successfully before cancel() ran and
        # its retained staging would otherwise leak forever.
        cleanup = {}
        for role in ("partial", "publish"):
            try:
                cleanup[role] = _remove_owned_producer_tree(
                    self.output, role, self.process.pid)
            except OSError as error:
                # Consumer failures can leave a mapped field open briefly on
                # Windows even after the native producer has been reaped.  A
                # best-effort cancellation cleanup must not replace that
                # primary failure with PermissionError.  Retain only the exact
                # PID-owned tree and record it for dead-PID startup cleanup.
                cleanup[role] = (
                    f"retained:{type(error).__name__}:{error}")
        self.cancel_cleanup = cleanup
        if self.failure_diagnostics is not None:
            removed = sorted(
                str(path) for path in cleanup.values()
                if isinstance(path, str) and not path.startswith("retained:"))
            if removed:
                self.failure_diagnostics["removed_after_report"] = removed


def write_pipeline_receipt(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
