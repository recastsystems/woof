"""Locate and drive the vendored Rust fetch backbone (``rw_fetch``).

``woof fetch --engine rust`` moves bytes through ``rw_fetch``, the
sixth binary of the vendored ``tools/rustwx`` workspace, instead of
through the stdlib ``urllib`` transports in :mod:`tools`.  What that
buys is machinery WOOF-ArWen would otherwise have to reimplement:

* **Parallel whole-file range GETs** -- one 700 MB serial TCP
  stream becomes a bounded few concurrent ones (``wx-core`` range
  downloads; :data:`STREAMS_ENV` sets how many), each
  resumed from the byte it reached when its body breaks off or stalls;
* **``.idx`` range coalescing** -- a 561-record selection collapses to a
  handful of GETs (``idx.rs:376`` ``byte_ranges`` plus coalescing);
* **the cross-process NOMADS rate governor** (``client.rs:178-232``) --
  a lock file and shared state enforcing a 2.5 s minimum inter-request
  gap and a 15-minute node-wide cooldown when Akamai's malformed
  over-rate-limit response is fingerprinted.  It is shared state, so a
  ``rw_fetch`` subprocess and any other rusty-weather process on the
  box cooperate rather than racing each other into an IP block;
* **a two-tier disk cache** keyed by URL and byte range.

Everything durable stays in Python.  ``rw_fetch`` reports what it did;
:mod:`woof.fetch` decides what that means -- the
``gpuwm-fetch-manifest-v1`` manifest, the request-identity resume
guard, quarantine-not-delete, and the record-count bars.

Like the GRIB bridges and the batch renderer, the pip wheel ships no
compiled Rust: the binary is built once from the vendored workspace
(``cargo build --release --locked --offline``) and then *pointed at*,
with the same resolution ladder as :mod:`woof.bridges`:

1. the ``WOOF_RW_FETCH`` environment variable naming the built file
   (a missing file it names is a hard error, never silently skipped);
2. a source checkout's ``tools/rustwx/target/{release,debug}``;
3. ``<root>/libexec/bridges`` beside the package;
4. ``~/.woof/bridges``.

Nothing here runs cargo.  Resolution has one side effect and one
only: an artifact found in ``~/.woof/bridges`` that is not the one this
release pinned is re-fetched before it is handed to a door
(:func:`woof.bridges.require_release_pin`).  ``woof doctor`` resolves
inside :func:`woof.bridges.inspection_only`, where there is no side
effect at all, so the report still says what the estate IS.
"""

from __future__ import annotations

import codecs
import contextlib
import functools
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Callable

from woof.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           default_bridge_dir,
                           legacy_bridge_candidates, lazy_build_hints,
                           rustwx_build_hint,
                           accept_resolved, executable_name, launchable,
                           packaged_bridge_dir, quiet_loader_errors)

#: Environment variable naming a prebuilt fetch backbone.
FETCH_ENV = "WOOF_RW_FETCH"

#: Executable base name of the vendored fetch backbone.
FETCH_NAME = "rw_fetch"

#: ``CARGO_BUILD_HINT``: the one-liner that builds it, from a checkout
#: root.  Same workspace as the batch renderer, so one cargo invocation
#: produces both.  Spelled for the shell rule when it is read.
__getattr__ = lazy_build_hints(
    __name__, CARGO_BUILD_HINT=RUSTWX_CRATE_RELATIVE)

#: Schema of the document ``rw_fetch fetch`` prints on stdout.
FETCH_RECORD_SCHEMA = "gpuwm-rw-fetch-record-v1"

#: Schema of the document ``rw_fetch probe`` prints on stdout.
PROBE_REPORT_SCHEMA = "gpuwm-rw-fetch-probe-v1"

#: The byte transports ``--mode`` selects between.  ``auto`` is the
#: probe rule: object present and its ``.idx`` absent, malformed, or
#: provably shorter than the object => take the whole file.  No time
#: constants are involved, and both named modes are first-class.
FETCH_MODES = ("auto", "full-file", "idx-subset")

#: ``rw_fetch --abi``.  The exact-ABI marker also compiled into the
#: binary, so a stale build that still answers ``--help`` but no longer
#: emits one of these record keys or lacks the declared commands is caught by
#: ``woof.native_wrf_distribution`` before a distribution ships.
FETCH_ABI_MARKER = (
    "gpuwm-rw-fetch-record-v1\tmode\tmode_reason\tsource\tgrib_url\t"
    "idx_url\tidx_sha256\tidx_record_count\tselected_record_count\t"
    "ranges\tsha256\tcommands=fetch,probe,latest,object,asset,encoded-asset,"
    "hash,unpack,unpack-audit,decode,inventory,objects")

_PROBE_TIMEOUT_S = 20

#: ``rw_fetch`` exit statuses.  2 is a command line it could not act
#: on, 3 a payload transfer the network cut off after the download
#: client's own retries, 1 every other refusal.  A backbone built
#: before the split exits 2 for all three.
EXIT_REFUSED = 1
EXIT_USAGE = 2
EXIT_TRANSFER = 3

#: Where a failure's reason starts on ``rw_fetch``'s stderr.
_REASON_MARK = "rw_fetch: "

#: What ``rw_fetch fetch`` prints on stderr about once a second while an
#: object moves: ``rw_fetch-progress fHHH RECEIVED TOTAL``, with ``-`` for
#: a TOTAL not yet known.  A whole object is assembled in memory and
#: published in one rename, so nothing on disk grows while it moves; this
#: line is the only live count there is.  It never contains
#: ``_REASON_MARK``, so it cannot be read as a failure's reason.
_PROGRESS_MARK = "rw_fetch-progress "

#: The environment variable that bounds how many chunk streams one
#: ``rw_fetch`` keeps open at once (16 when unset).
STREAMS_ENV = "RUSTWX_DOWNLOAD_STREAMS"

# The bundled ureq Config default retains ten idle sockets across origins
# (vendor/crates-io/ureq/src/config.rs). Redirects can fill several host
# pools, so reserving only one idle socket per active stream is unsafe.
LEGACY_IDLE_CONNECTIONS = 10

#: ``(received, total)`` for one ``rw_fetch`` invocation's current object;
#: ``total`` is ``None`` until the backbone knows the size.
ProgressCallback = Callable[[int, "int | None"], None]

#: The line a current ``rw_fetch`` prints after a usage error, and only
#: after one.  A backbone built before the exit statuses were split
#: prints the whole usage text after every failure instead, so exit 2
#: without this line is an older backbone reporting some other failure
#: (a dropped connection included), not a command line it refused.
_USAGE_ERROR_MARK = "rw_fetch --help lists every option"


class RwFetchError(RuntimeError):
    """One ``rw_fetch`` call that printed no record, and why.

    ``str()`` is ``rw_fetch <subcommand>: <reason>``, the line a run's
    ``failed`` event carries.  ``remedy`` is what the reader can do
    about it, or ``None`` when nothing specific is known; the run-plan
    front door relays it.  ``transient`` marks a transfer the network
    cut off, which the caller may ask for again.
    """

    def __init__(self, what: str, reason: str, *, returncode: int,
                 remedy: str | None = None) -> None:
        super().__init__(f"rw_fetch {what}: {reason}")
        self.what = what
        self.reason = reason
        self.returncode = returncode
        self.remedy = remedy

    @property
    def transient(self) -> bool:
        return self.returncode == EXIT_TRANSFER


def failure_reason(stderr: str | None, returncode: int) -> str:
    """The reason a failed ``rw_fetch`` printed, as one line.

    The reason is the text after the LAST ``rw_fetch: `` on stderr,
    wherever that falls on its line.  It used to be the first stderr
    line that was not indented and did not start with ``usage:``, which
    was meant to skip the usage text printed after it.  But wx-core
    draws transfer progress as ``\\r  Downloading chunks N/M...`` with
    no line end, so the reason was glued onto an indented progress
    line, skipped with the usage, and the first usage heading became
    the reason: a dropped connection 38 minutes into an HRRR download
    reached the run's failed event as ``rw_fetch fetch: common
    options``.
    """

    segments = [segment.strip()
                for segment in re.split(r"[\r\n]+", stderr or "")]
    reasons = [segment.split(_REASON_MARK, 1)[1].strip()
               for segment in segments if _REASON_MARK in segment]
    reasons = [reason for reason in reasons if reason]
    if reasons:
        return reasons[-1]
    if returncode < 0:
        return (f"rw_fetch was stopped by signal {-returncode} before it "
                "reported a reason")
    return f"rw_fetch exited {returncode} without reporting a reason"


def failure_remedy(returncode: int, *, binary: Path | str,
                   out: Path | None = None,
                   stderr: str | None = None) -> str | None:
    """What a reader can do about one ``rw_fetch`` failure, if known.

    ``stderr`` separates the two things exit 2 can mean.  A current
    backbone exits 2 only for a command line it could not act on, and
    says so with ``_USAGE_ERROR_MARK``.  A backbone built before the
    split exits 2 for every failure; its probe still passes, because the
    record ABI did not change, so blaming the command line there would
    misstate a dropped connection as a disagreement about options.
    """

    if returncode == EXIT_TRANSFER:
        kept = (f"; the files that did download stay in {out}, and a run "
                "given that folder as its data directory checks and reuses "
                "them instead of downloading them again"
                if out is not None else "")
        return ("the network cut the download off on every retry; try "
                f"again when the connection is steadier{kept}")
    if returncode == EXIT_USAGE:
        if stderr is not None and _USAGE_ERROR_MARK not in stderr:
            return (f"the rw_fetch at {binary} is older than this woof "
                    "and reports every failure the same way, so the "
                    "reason above is all it says about what went wrong; "
                    "reinstall so both come from one release, or rebuild "
                    f"it from a checkout with {rustwx_build_hint()}")
        return (f"this woof and the rw_fetch at {binary} disagree about "
                "its command line, so they come from different releases; "
                "reinstall so both come from one release, or rebuild it "
                f"from a checkout with {rustwx_build_hint()}")
    return None


def crate_dir() -> Path:
    """The vendored Rusty Weather workspace of a source checkout."""

    return Path(__file__).resolve().parent.parent / "tools" / "rustwx"


def fetch_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for the backbone, best first."""

    filename = executable_name(FETCH_NAME)
    candidates: list[Path] = []
    override = os.environ.get(FETCH_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def find_fetch_bin() -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error:
    explicit configuration must fail loudly, not fall through to the
    Python transports and leave the operator wondering why the download
    was slow.
    """

    override = os.environ.get(FETCH_ENV)
    for candidate in fetch_candidates():
        if candidate.is_file():
            return accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{FETCH_ENV} names a missing file: {candidate}")
    return None


def fetch_remedy() -> str:
    """The remedy for a missing backbone, true for THIS install."""

    return artifact_remedy(
        env_var=FETCH_ENV, filename=executable_name(FETCH_NAME),
        subject="the rust fetch backbone",
        crate_relative=RUSTWX_CRATE_RELATIVE,
        one_liner=rustwx_build_hint())


def probe_fetch_bin(path: Path) -> tuple[bool, str]:
    """Launch ``path --version`` once; can this binary execute at all?

    ``rw_fetch --version`` prints its name and version and exits 0.
    That observable separates a runnable executable from an empty,
    truncated, or wrong-platform file, which refuses to launch
    (OSError) or dies with an abnormal status and no output.  The
    ``--abi`` check that follows is the stale-build guard: a binary that
    launches but emits a different record ABI is worse than a missing
    one, because it fails after the download rather than before it.

    The header gate in front of the launch is the same one every probe
    in this package uses, for the same reason: on Windows a corrupt
    image header can hang ``CreateProcess`` itself, where no timeout
    reaches.  See :func:`woof.bridges.native_executable_format`.
    """

    ok, evidence = launchable(path)
    if not ok:
        return False, f"{evidence} -- corrupt, stale, or built for " \
                      "another platform"
    try:
        with quiet_loader_errors():
            probe = subprocess.run(
                [str(path), "--version"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"exists but failed to execute: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"probe invocation did not exit within "
                       f"{_PROBE_TIMEOUT_S} s")
    transcript = f"{probe.stdout or ''}{probe.stderr or ''}"
    if probe.returncode != 0 or not transcript.startswith("rw_fetch "):
        return False, (f"probe --version exited {probe.returncode} without "
                       "its version line -- corrupt, stale, or built for "
                       "another platform")
    try:
        with quiet_loader_errors():
            abi = subprocess.run(
                [str(path), "--abi"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"executes but --abi failed: {error}"
    observed = (abi.stdout or "").strip()
    if abi.returncode != 0 or observed != FETCH_ABI_MARKER:
        missing_commands = (
            "rw_fetch is stale: --abi does not declare the required hash "
            "and object/asset/encoded-asset/inventory/objects fetch commands; "
            if observed == FETCH_ABI_MARKER.split("\tcommands=", 1)[0]
            else "rw_fetch "
        )
        return False, (missing_commands + "executes but reports a different fetch-record ABI "
                       "than this woof expects -- rebuild it: "
                       f"{rustwx_build_hint()}")
    return True, (f"{transcript.strip()} -- --abi matches the "
                  "fetch-record contract")


def parse_progress(segment: str) -> tuple[int, int | None] | None:
    """``(received, total)`` from one stderr segment, or ``None``.

    Found wherever it sits on its segment, the way a failure's reason
    is: wx-core's carriage-return chunk counter can precede it.
    """

    if _PROGRESS_MARK not in segment:
        return None
    fields = segment.split(_PROGRESS_MARK, 1)[1].split()
    if len(fields) < 3:
        return None
    try:
        received = int(fields[1])
        total = None if fields[2] == "-" else int(fields[2])
    except ValueError:
        return None
    return received, total


def _relay(segment: str, on_progress: ProgressCallback) -> None:
    parsed = parse_progress(segment)
    if parsed is None:
        return
    try:
        on_progress(*parsed)
    except Exception:            # noqa: BLE001 - telemetry never fails a fetch
        pass


@functools.lru_cache(maxsize=32)
def _idle_capable(binary, identity):
    try:
        result = subprocess.run([binary, "--help"], capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPUWM_FETCH_NO_IDLE_CONNECTIONS" in result.stdout


def _run(command: list[str], *, what: str,
         out: Path | None = None, env: dict[str, str] | None = None,
         on_progress: ProgressCallback | None = None) -> dict:
    """Reserve box-wide socket slots until every bridge socket is closed."""
    if what not in {"fetch", "object", "asset", "encoded-asset", "probe", "latest"}:
        return _run_process(command, what=what, out=out, env=env,
                            on_progress=on_progress)
    from woof import fetch_connections, fetch_pool
    settings = dict(os.environ if env is None else env)
    cap = fetch_connections.max_connections()
    try:
        stat = Path(command[0]).stat()
        identity = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        identity = None
    idle = 0 if _idle_capable(str(command[0]), identity) else LEGACY_IDLE_CONNECTIONS
    if cap <= idle:
        raise ValueError(f"box fetch cap {cap} cannot bound this older Rust bridge's {idle} idle sockets plus an active socket; rebuild rw_fetch for idle-free transfers or use the Python transport")
    requested = int(settings.get(STREAMS_ENV, 16))
    streams = max(1, min(requested, cap - idle))
    settings["GPUWM_FETCH_NO_IDLE_CONNECTIONS"] = "1"
    # The grant can be smaller than the request when other woof processes on the box hold slots; the
    # bridge opens exactly the granted active sockets (it used to wait, silently, for the full request).
    with fetch_connections.lease(streams + idle, minimum=idle + 1) as granted:
        streams = granted - idle
        settings[STREAMS_ENV] = str(streams)
        record = _run_process(command, what=what, out=out, env=settings,
                              on_progress=on_progress)
    # The native object door reports its selected width after inspecting
    # the object. Small objects and paced hosts can use fewer than the
    # reserved slots. Older artifacts report only the configured limit.
    used = record.get("streams_used", streams)
    if isinstance(used, bool) or not isinstance(used, int) or not 1 <= used <= streams:
        raise ValueError(f"rw_fetch {what} reported invalid stream width {used!r}; reserved {streams} active sockets")
    fetch_pool.record_streams(used, streams + idle)
    return record


def _run_process(command: list[str], *, what: str,
         out: Path | None = None, env: dict[str, str] | None = None,
         on_progress: ProgressCallback | None = None) -> dict:
    """Run one ``rw_fetch`` subcommand and parse its JSON document.

    stderr is read AS IT ARRIVES rather than at exit, so the
    ``rw_fetch-progress`` lines reach ``on_progress`` while the object
    is still moving; the whole of it is kept for the failure reason.

    Run under a fetch pool job (:func:`woof.fetch_pool.current_job`),
    the backbone is not launched once another file has failed the
    request, and one already running is terminated when that happens
    rather than left to pull hundreds of megabytes nobody will use; the
    call then raises :class:`woof.fetch_pool.TransferCancelled`, not a
    failure of its own.
    """

    from woof import fetch_pool

    job = fetch_pool.current_job()
    fetch_pool.raise_if_stopped(job)
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env)
    except OSError as error:
        raise RuntimeError(
            f"rw_fetch failed to launch: {error}") from error
    kept: list[str] = []

    def drain() -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        while True:
            block = process.stderr.read1(65536)
            text = decoder.decode(block, final=not block)
            kept.append(text)
            if on_progress is not None:
                pending += text
                *complete, pending = re.split(r"[\r\n]", pending)
                for segment in complete:
                    _relay(segment, on_progress)
            if not block:
                break
        if on_progress is not None and pending:
            _relay(pending, on_progress)

    reader = threading.Thread(target=drain, name="rw_fetch-stderr",
                              daemon=True)
    reader.start()
    watch = (contextlib.nullcontext() if job is None
             else job.adopt(process))
    try:
        with watch:
            stdout = process.stdout.read()
            returncode = process.wait()
    except BaseException:
        # An interrupt must not leave the backbone downloading on its own.
        process.kill()
        process.wait()
        raise
    finally:
        reader.join()
        process.stdout.close()
        process.stderr.close()
    stderr = "".join(kept)
    if returncode != 0 and job is not None and job.stopped:
        raise fetch_pool.TransferCancelled(
            f"rw_fetch {what}: stopped because another file failed the "
            "request")
    if returncode != 0:
        raise RwFetchError(
            what, failure_reason(stderr, returncode),
            returncode=returncode,
            remedy=failure_remedy(returncode, binary=command[0],
                                  out=out, stderr=stderr))
    try:
        return json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"rw_fetch {what} did not print a JSON record: {error}"
        ) from error


def _window(*, model: str, date: str, cycle: int,
            hours: tuple[int, ...] | None, product: str,
            source: str | None) -> list[str]:
    command = ["--model", model, "--date", date, "--cycle", f"{cycle:02d}",
               "--product", product]
    if hours is not None:
        command += ["--hours", ",".join(str(hour) for hour in hours)]
    if source is not None:
        command += ["--source", source]
    return command


def run_fetch(binary: Path, *, model: str, date: str, cycle: int,
              hours: tuple[int, ...], product: str, out: Path,
              mode: str = "auto", source: str | None = None,
              patterns: tuple[str, ...] = (),
              pattern_file: Path | None = None,
              exclusions: tuple[str, ...] = (),
              cache_dir: Path | None = None,
              keep_idx: bool = False, streams: int | None = None,
              on_progress: ProgressCallback | None = None) -> dict:
    """Download one model-run window; return the parsed fetch record.

    ``streams`` bounds how many chunk streams this one invocation keeps
    open (:data:`STREAMS_ENV`); a caller running several invocations side
    by side divides its own budget.  ``on_progress`` hears the body bytes
    of the object in flight, about once a second, while it moves.

    ``patterns`` are exact ``VAR:LEVEL`` selectors.  A selection of any
    size beyond a handful should go through ``pattern_file`` instead --
    561 selectors on a command line exceeds the Windows argv limit, and
    the file form is what the HRRR lane uses.  ``exclusions`` are
    substrings tested against the whole raw index line, which is how the
    accumulated twin of a surface field (``0-1 hr acc fcst``) is kept
    out of an otherwise exact selection.
    """

    command = [str(binary), "fetch"]
    command += _window(model=model, date=date, cycle=cycle, hours=hours,
                       product=product, source=source)
    command += ["--mode", mode, "--out", str(out)]
    for pattern in patterns:
        command += ["--var-pattern", pattern]
    if pattern_file is not None:
        command += ["--var-pattern-file", str(pattern_file)]
    for excluded in exclusions:
        command += ["--exclude-forecast-contains", excluded]
    if cache_dir is not None:
        command += ["--cache-dir", str(cache_dir)]
    if keep_idx:
        command.append("--keep-idx")
    env = None
    if streams is not None:
        if isinstance(streams, bool) or int(streams) < 1:
            raise ValueError(
                f"streams must be a positive count, not {streams!r}")
        env = {**os.environ, STREAMS_ENV: str(int(streams))}
    record = _run(command, what="fetch", out=out, env=env,
                  on_progress=on_progress)
    if record.get("schema") != FETCH_RECORD_SCHEMA:
        raise RuntimeError(
            f"rw_fetch printed schema {record.get('schema')!r}, expected "
            f"{FETCH_RECORD_SCHEMA!r}")
    return record


def run_probe(binary: Path, *, model: str, date: str, cycle: int,
              hours: tuple[int, ...], product: str,
              mode: str = "auto", source: str | None = None,
              cache_dir: Path | None = None) -> dict:
    """Report the transport decision per hour, moving no payload."""

    command = [str(binary), "probe"]
    command += _window(model=model, date=date, cycle=cycle, hours=hours,
                       product=product, source=source)
    command += ["--mode", mode]
    if cache_dir is not None:
        command += ["--cache-dir", str(cache_dir)]
    report = _run(command, what="probe")
    if report.get("schema") != PROBE_REPORT_SCHEMA:
        raise RuntimeError(
            f"rw_fetch printed schema {report.get('schema')!r}, expected "
            f"{PROBE_REPORT_SCHEMA!r}")
    return report


def write_pattern_file(path: Path, patterns: tuple[str, ...]) -> Path:
    """Write exact selectors one per line for ``--var-pattern-file``."""

    path.write_text(
        "".join(f"{pattern}\n" for pattern in patterns),
        encoding="utf-8", newline="\n")
    return path


__all__ = [
    "CARGO_BUILD_HINT", "EXIT_REFUSED", "EXIT_TRANSFER", "EXIT_USAGE",  # noqa: F822 -- module __getattr__
    "FETCH_ABI_MARKER", "FETCH_ENV", "FETCH_MODES",
    "FETCH_NAME", "FETCH_RECORD_SCHEMA", "PROBE_REPORT_SCHEMA",
    "ProgressCallback", "RwFetchError", "STREAMS_ENV", "crate_dir",
    "failure_reason", "failure_remedy", "fetch_candidates", "fetch_remedy",
    "find_fetch_bin", "parse_progress", "probe_fetch_bin", "run_fetch",
    "run_probe", "write_pattern_file",
]
