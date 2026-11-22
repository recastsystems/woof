"""Schedule native observation verification as reports become available.

Python carries paths, times and receipts. ``rw_verify`` reads and samples the
weather data, computes every score and draws every image. A source outage is
recorded per hour and per quantity so another invocation can finish the run.
"""

from __future__ import annotations

import argparse
import functools
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

REQUEST_SCHEMA = "gpuwm.verify-visuals.request.v1"
STATE_SCHEMA = "gpuwm.verify-visuals.state.v1"
MANIFEST_NAME = "verification.json"
BACKGROUND_NAME = "verification-background.json"
BACKGROUND_SCHEMA = "gpuwm.verification-background.v1"
RETENTION_SCHEMA = "gpuwm.verification-retention.v1"
FINISH_FOREGROUND_SECONDS = 2.0
UTC = timezone.utc
_CALL_DEADLINE = ContextVar("verification_deadline", default=None)
_LOCAL_ONLY = ContextVar("verification_local_only", default=False)
_BACKGROUND_PROCESSES = []
TERMINAL_SOURCE_STATES = frozenset(("ready", "unavailable"))

# Archive posting budgets determine finality, not whether an earlier report
# can be scored. Before the budget ends a station snapshot stays provisional.
OBSERVATION_ROWS = (
    {"key": "stations", "final_minutes": 75},
    {"key": "composite_reflectivity", "product": "MergedReflectivityQCComposite_00.50",
     "final_minutes": 15},
    {"key": "precipitation_1h", "product": "MultiSensor_QPE_01H_Pass2_00.00",
     "final_minutes": 80},
)

FIELD_NAMES = {
    "temperature_2m": "t2_k", "dewpoint_2m": "td2_k",
    "wind_speed_10m": "wind_ms", "composite_reflectivity": "refc_dbz",
    "precipitation_1h": "precip_1h_mm",
}
HISTORY_NAME = re.compile(
    r"wrfout_(d\d+)_(\d{4}-\d{2}-\d{2})[_T](\d{2})[:_](\d{2})[:_](\d{2})(?:\.nc4?|\.cdf)?")


def _time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return _time(datetime.fromisoformat(str(value).replace("Z", "+00:00")))


def _stamp(value: datetime) -> str:
    return _time(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _atomic_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _run_lock(root: Path, *, name=".verification.lock"):
    """A process-owned advisory lock, released even after interruption."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / name).open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def verification_scope(enabled: bool | None = True):
    """Carry a run option to in-process and subprocess finish hooks.

    ``None`` sets nothing and only gives the variable back on exit, so a
    door that sets it from its flag (``woof run --no-verify-visuals``)
    bounds that to its own call when the CLI is driven in-process.
    """
    key = "WOOF_VERIFY_VISUALS"
    previous = os.environ.get(key)
    if enabled is not None:
        os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def add_argument(parser) -> None:
    """``--no-verify-visuals``, one spelling and one meaning on every forecast door.

    ``woof go`` had it; ``woof sim`` and ``woof run`` had only the
    ``WOOF_VERIFY_VISUALS`` variable, so a caller of those doors could not
    find the switch for the CPU check that starts when the forecast ends
    (2026-11-11, an outside runner: the check outlived the card lease).
    """
    parser.add_argument(
        "--no-verify-visuals", action="store_false", dest="verify_visuals", default=None,
        help="skip postforecast observation verification; the physical run is unchanged "
             "(also WOOF_VERIFY_VISUALS=0)")


def requested(args=None) -> bool:
    """The door's ``--no-verify-visuals`` when given, else ``WOOF_VERIFY_VISUALS``."""
    flag = getattr(args, "verify_visuals", None)
    if flag is not None:
        return bool(flag)
    return os.environ.get("WOOF_VERIFY_VISUALS", "1").lower() not in ("0", "false", "off")


@contextmanager
def _deadline_scope(deadline):
    token = _CALL_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _CALL_DEADLINE.reset(token)


@contextmanager
def _local_scope(enabled):
    # Native resolution can refresh an installed artifact. Local work must
    # suppress that network path as well as public observation fetching.
    from woof import bridges

    token = _LOCAL_ONLY.set(enabled)
    try:
        with bridges.inspection_only() if enabled else nullcontext():
            yield
    finally:
        _LOCAL_ONLY.reset(token)


def _verifier_identity():
    """Bind result reuse to the selected native scorer and renderer build."""
    from woof.rustwx import find_verification_binary

    path = find_verification_binary()
    if path is None:
        return None
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


RUN_REPORT_NAME = "report.json"


def _identity_summary(identity, *, origin):
    """Commit, tree and a dirty flag from one provenance identity mapping.

    ``dirty`` counts TRACKED changes only (``??`` rows are run output, not
    code) and is ``None`` when git was not asked, which is a different
    statement from "clean".  When no commit can be named the reason says
    which resolver answered instead, so a reader never mistakes a bare
    content hash for a git id.
    """
    if not isinstance(identity, dict):
        return {"origin": origin, "git_commit": None, "git_tree": None, "dirty": None,
                "identity_source": None, "reason": "no provenance identity was recorded"}
    status = identity.get("git_status_short")
    tracked = None if status is None else [str(line) for line in status
                                           if not str(line).startswith("??")]
    summary = {"origin": origin, "git_commit": identity.get("git_commit"),
               "git_tree": identity.get("git_tree"),
               "dirty": None if tracked is None else bool(tracked),
               "dirty_paths": tracked[:20] if tracked else [],
               "identity_source": identity.get("identity_source")}
    if identity.get("distribution_manifest_sha256"):
        summary["distribution_manifest_sha256"] = identity["distribution_manifest_sha256"]
    content = identity.get("installed_source_content")
    if isinstance(content, dict) and content.get("content_sha256"):
        summary["source_content_sha256"] = content["content_sha256"]
    if not summary["git_commit"]:
        summary["reason"] = (f"resolver {summary['identity_source'] or 'unknown'} names no git commit; "
                             "run from a checkout, an editable install of one, or a sealed distribution")
    return summary


def _forecast_engine_identity(root: Path):
    """The engine that MADE the scored frames, from the run's own report."""
    path = Path(root) / RUN_REPORT_NAME
    report = _read(path, None)
    if not isinstance(report, dict):
        return {"origin": str(path), "git_commit": None, "git_tree": None, "dirty": None,
                "identity_source": None,
                "reason": "the run folder has no forecast report naming its engine"}
    summary = _identity_summary(report.get("runtime_source_identity"), origin=str(path))
    # The run-plan path records its install in a ``provenance`` receipt
    # block (woof.provenance_gate.receipt_block) rather than in
    # ``runtime_source_identity``; its ``git`` mapping names the commit.
    block = report.get("provenance")
    git = block.get("git") if isinstance(block, dict) else None
    if not summary["git_commit"] and isinstance(git, dict) and git.get("commit_full"):
        summary.update(git_commit=git["commit_full"], dirty=git.get("dirty"),
                       identity_source="provenance-receipt-git")
        summary.pop("reason", None)
        if git.get("branch"):
            summary["branch"] = git["branch"]
        if git.get("dirty") is None and git.get("dirty_unknown_reason"):
            summary["dirty_unknown_reason"] = git["dirty_unknown_reason"]
    if report.get("gpuwm_version"):
        summary["gpuwm_version"] = report["gpuwm_version"]
    return summary


@functools.lru_cache(maxsize=1)
def _verifier_engine_identity():
    """The engine doing the scoring, as the package resolver names it."""
    from woof.runtime_manifest import IdentityError, provenance

    root = Path(__file__).resolve().parents[1]
    try:
        identity = provenance(root)
    # The scorer's own name is a receipt line, never a reason to fail the
    # scoring: any resolver error is recorded as the reason instead.
    except (IdentityError, OSError, subprocess.SubprocessError, ValueError, LookupError) as error:
        return {"origin": str(root), "git_commit": None, "git_tree": None, "dirty": None,
                "identity_source": None, "reason": f"{type(error).__name__}: {error}"}
    return _identity_summary(identity, origin=str(root))


def _engine_line(identity) -> str:
    if not identity:
        return "unrecorded"
    commit = identity.get("git_commit")
    if not commit:
        return f"no git commit ({identity.get('reason', 'unrecorded')})"
    state = {True: "dirty", False: "clean", None: "worktree state not asked"}[identity.get("dirty")]
    return f"git {commit} ({state})"


def _signature(request: dict, *, include_metadata=True, include_verifier=True) -> str:
    """Bind result reuse to the request, durable inputs and native artifact."""
    versions = []
    for arm in request["arms"]:
        for key in ("path", "previous_path", "grid_path", "points_path"):
            if arm.get(key):
                path = Path(arm[key])
                stat = path.stat() if path.is_file() else None
                versions.append([str(path), stat.st_size if stat else None,
                                 stat.st_mtime_ns if stat else None])
        if include_metadata and arm.get("kind") == "npz":
            metadata = Path(arm["path"]).with_suffix(".metadata.json")
            if metadata.is_file():
                stat = metadata.stat()
                versions.append([str(metadata), stat.st_size, stat.st_mtime_ns])
    for key in ("stations_path", "station_table_path"):
        if request.get(key):
            path = Path(request[key])
            stat = path.stat()
            versions.append([str(path), stat.st_size, stat.st_mtime_ns])
    for row in request.get("radar", []):
        for key in ("path", "grid_path"):
            path = Path(row[key])
            stat = path.stat()
            versions.append([str(path), stat.st_size, stat.st_mtime_ns])
    identity = [request, versions]
    if include_verifier:
        identity.append(_verifier_identity())
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _first_input_signature(request):
    """Bind an automatic pair to the selected forecast input and valid hour."""
    return _signature({"arms": request["arms"][:1], "domain": request["domain"],
                       "valid_time": request["valid_time"], "cycle": request.get("cycle"),
                       "station_table_path": request.get("station_table_path")}, include_verifier=False)


def _reference_selection(request, entry, reference, reference_dir):
    old = entry.get("reference_selection", {})
    return {"reference": reference,
            "directory": str(reference_dir or request.get("reference_dir") or old.get("directory") or ""),
            "file": str(request.get("reference_file") or old.get("file") or "")}


def _sources_complete(sources):
    return bool(sources) and all(source.get("status") in TERMINAL_SOURCE_STATES
                                 for source in sources.values())


def _reuse_pair(request, entry, selection):
    pair = entry.get("paired_input")
    if not pair or len(pair.get("arms", [])) < 2:
        return request
    managed = len(request["arms"]) == 1 or request["arms"] == pair["arms"]
    if not managed:
        return request
    previous_selection = entry.get("reference_selection")
    if previous_selection is None:
        # Older receipts predate selection bookkeeping. The native model key
        # supplies the table row; an absent model key cannot establish a match.
        previous_selection = {**selection, "reference": pair["arms"][-1].get("model")}
    first_signature = _first_input_signature(request)
    saved_signature = entry.get("paired_first_signature")
    unchanged = (first_signature == saved_signature if saved_signature else
                 first_signature == _first_input_signature(pair)
                 and entry.get("base_signature") in
                 (_signature(pair), _signature(pair, include_verifier=False),
                  _signature(pair, include_metadata=False, include_verifier=False)))
    files_present = all(Path(arm["path"]).is_file() for arm in pair["arms"])
    if selection["reference"] and selection == previous_selection and unchanged and files_present:
        entry["paired_first_signature"] = first_signature
        entry["reference_selection"] = selection
        entry.setdefault("sources", {})["reference"] = {"status": "ready", "input": pair["arms"][-1]["path"]}
        return {**request, **json.loads(json.dumps(pair))}
    # An automatically supplied arm must be replaced when its forecast input
    # or selected reference changes. Explicit multi-arm manifests stay intact.
    request["arms"] = request["arms"][:1]
    entry.pop("paired_input", None)
    entry.pop("paired_first_signature", None)
    entry.setdefault("sources", {}).pop("reference", None)
    return request


def discover_histories(root: Path, *, label="WOOF", paths=None,
                       cycle=None, include_missing=False) -> list[dict]:
    """Discover only path metadata; native inventory checks the field content."""
    frames = []
    for path in sorted(paths if paths is not None else root.rglob("wrfout_d*")):
        path = Path(path)
        # Ready receipts share the frame basename. Importing a trailing JSON
        # receipt as NetCDF would replace a valid retained scheduling request.
        matched = HISTORY_NAME.fullmatch(path.name)
        if not matched or (not include_missing and not path.is_file()):
            continue
        domain, date, hour, minute, second = matched.groups()
        when = _time(f"{date}T{hour}:{minute}:{second}Z")
        frames.append({"schema": REQUEST_SCHEMA, "domain": domain,
                       "valid_time": _stamp(when), "arms": [
                           {"label": label, "kind": "netcdf", "path": str(path.resolve())}],
                       **({"cycle": _stamp(_time(cycle))} if cycle else {})})
    by_time = {(row["domain"], row["valid_time"]): row for row in frames}
    for row in frames:
        previous = by_time.get((row["domain"], _stamp(_time(row["valid_time"]) - timedelta(hours=1))))
        if previous:
            row["arms"][0]["previous_path"] = previous["arms"][0]["path"]
    return frames


def select_run_requests(root: Path, *, cycle=None, domain=None,
                        first_hour=None, last_hour=None) -> list[dict]:
    """A run folder's requests, narrowed to the domain and hours the caller named.

    The same rows :func:`score_available` takes when it is handed none (the
    retained requests of ``verification.json`` plus the frames on disk), then
    only those of ``domain`` and of forecast hours ``first_hour`` to
    ``last_hour``.  Hours count from ``cycle`` when it is given and otherwise
    from the earliest frame of the run.

    THE BREAKAGE THIS PREVENTS (3.0.0 artefact check, 2026-11-10):
    ``woof verify-visuals RUN --domain d02 --first-hour 2 --last-hour 3``
    took the three flags and read none of them on a run folder, so a user who
    asked for two hours of one nest waited for the whole sweep of every domain.
    """
    root = Path(root).resolve()
    previous_state = _read(root / MANIFEST_NAME, {}) or {}
    rows = {key: row["request"] for key, row in previous_state.get("hours", {}).items()
            if row.get("request")}
    for row in discover_histories(root, cycle=cycle):
        rows[f"{row['domain']}/{row['valid_time']}"] = row
    rows = list(rows.values())
    if not rows:
        raise ValueError(f"{root} holds no finished history frame to score")
    start = _time(cycle) if cycle else min(_time(row["valid_time"]) for row in rows)
    kept = []
    for row in rows:
        if domain is not None and row["domain"] != domain:
            continue
        lead = (_time(row["valid_time"]) - start).total_seconds() / 3600.0
        if first_hour is not None and lead < first_hour:
            continue
        if last_hour is not None and lead > last_hour:
            continue
        kept.append(row)
    if not kept:
        named = ", ".join(sorted({row["domain"] for row in rows}))
        last = max((_time(row["valid_time"]) - start).total_seconds() / 3600.0 for row in rows)
        asked = " ".join(f"{flag} {value}" for flag, value in (
            ("--domain", domain), ("--first-hour", first_hour), ("--last-hour", last_hour))
            if value is not None)
        raise ValueError(
            f"no finished frame matches {asked}: this run holds "
            f"{named} through hour {last:g} from {_stamp(start)}")
    return kept


def histories_before_restart(paths, cycle):
    """This run's histories from before a checkpoint restart, beside its frames.

    A run resumed from a checkpoint reports only the frames it wrote after the
    restart. Its earlier hours are already on disk beside them; without them
    those hours got no verification input and the first resumed hour no hourly
    precipitation (2.9.1 audit: a killed ensemble member resumed from 13Z
    lacked its 12Z and 13Z inputs and its 14Z precipitation_1h).
    """
    if cycle is None:
        return []
    start, first, reported = _time(cycle), {}, set()
    for path in map(Path, paths):
        matched = HISTORY_NAME.fullmatch(path.name)
        if matched:
            domain, date, hour, minute, second = matched.groups()
            when = _time(f"{date}T{hour}:{minute}:{second}Z")
            key = (path.parent, domain)
            first[key] = min(first.get(key, when), when)
            reported.add(path.resolve())
    earlier = []
    for (directory, domain), resumed in first.items():
        if resumed <= start or not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"wrfout_{domain}_*")):
            matched = HISTORY_NAME.fullmatch(path.name)
            if not matched or not path.is_file() or path.resolve() in reported:
                continue
            _, date, hour, minute, second = matched.groups()
            if start <= _time(f"{date}T{hour}:{minute}:{second}Z") < resumed:
                earlier.append(path)
    return earlier


def _assignments(values):
    result = {}
    for text in values or ():
        label, separator, path = text.partition("=")
        if not separator or not label.strip() or not path:
            raise ValueError("an arm must be LABEL=PATH")
        result[label.strip()] = path
    return result


def discover_packaged(*, cycle, domain, point_arms=(), field_arms=(),
                      grid=None, station_table=None, hours=range(1, 49)) -> list[dict]:
    """Connect archived paths without decoding point or field payloads."""
    points, fields = _assignments(point_arms), _assignments(field_arms)
    labels = list(dict.fromkeys([*points, *fields]))
    rows = []
    for hour in hours:
        arms = []
        for label in labels:
            point = (Path(points[label]) / f"f{hour:02d}-pts-w0.points.json"
                     if label in points else None)
            field = (Path(fields[label].format(hour=hour)) if label in fields else None)
            if field is not None and field.is_file():
                arm = {"label": label, "kind": "npz", "path": str(field.resolve()),
                       "fields": FIELD_NAMES, "init_time": _stamp(_time(cycle)), "hour": hour}
                if grid:
                    arm["grid_path"] = str(Path(grid).resolve())
                if point is not None and point.is_file():
                    arm["points_path"] = str(point.resolve())
            elif point is not None and point.is_file():
                arm = {"label": label, "kind": "points", "path": str(point.resolve())}
            else:
                continue
            arms.append(arm)
        if arms:
            rows.append({"schema": REQUEST_SCHEMA, "domain": domain,
                         "valid_time": _stamp(_time(cycle) + timedelta(hours=hour)),
                         "cycle": _stamp(_time(cycle)), "arms": arms,
                         **({"station_table_path": str(Path(station_table).resolve())}
                            if station_table else {})})
    return rows


def _door(door, operation, arguments, schema, *, timeout):
    """Bound each public-source call and validate its native metadata receipt."""
    if _LOCAL_ONLY.get():
        raise RuntimeError("local-only verification cannot invoke a public observation source")
    result = subprocess.run([str(door.require()), operation, *arguments],
                            capture_output=True, text=True, errors="replace", timeout=_timeout(timeout))
    if result.returncode:
        raise RuntimeError(f"{door.name} {operation}: {result.stderr.strip()[-1600:]}")
    record = json.loads(result.stdout)
    if record.get("schema") != schema:
        raise RuntimeError(f"{door.name} {operation} returned an incompatible receipt")
    return record


def _timeout(seconds):
    deadline = _CALL_DEADLINE.get()
    if deadline is None:
        return seconds
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("finish-time verification budget ended; rerun the retry command")
    return min(seconds, remaining)


def fetch_stations(request, bbox, folder, *, timeout, refresh=False) -> Path:
    from woof.obs.frontdoor import ASOS
    from woof.obs.surface_networks import networks_for_domain

    folder.mkdir(parents=True, exist_ok=True)
    table = Path(request.get("station_table_path") or folder / "stations.json")
    if not table.is_file():
        if bbox is None:
            raise ValueError("station verification needs a native grid bbox or a frozen station table")
        arguments = ["--networks", ",".join(networks_for_domain(*bbox)),
                     "--bbox", ",".join(map(str, bbox)), "--out", str(table)]
        _door(ASOS, "stations", arguments, "gpuwm-obs.asos-stations.v1", timeout=timeout)
    when = _time(request["valid_time"])
    csv = folder / "observations.csv"
    surface = folder / "surface.json"
    _door(ASOS, "fetch", ["--stations", str(table), "--start", _stamp(when - timedelta(hours=1)),
                         "--end", _stamp(when + timedelta(hours=1)), "--out", str(csv)],
          "gpuwm-obs.asos-fetch.v1", timeout=timeout)
    decoded = _door(ASOS, "decode", ["--stations", str(table), "--obs", str(csv),
                       "--start", _stamp(when), "--end", _stamp(when),
                       "--out", str(surface)], "gpuwm-obs.asos-surface.v2", timeout=timeout)
    if not decoded.get("reports"):
        raise LookupError("the public archive has no quality station reports for this hour")
    return surface


def fetch_radar(request, row, bbox, folder, *, timeout, refresh=False) -> dict:
    from woof.obs.frontdoor import MRMS

    folder.mkdir(parents=True, exist_ok=True)
    product = ["--product", row["product"]]
    nearest = _door(MRMS, "nearest", [*product, "--valid-time", request["valid_time"],
                     "--window-seconds", "240"], "gpuwm-obs.mrms-nearest.v1", timeout=timeout)
    stamp = nearest["frame"]["valid_time"]
    fetched = _door(MRMS, "fetch", [*product, "--start", stamp, "--end", stamp,
                         "--out", str(folder / "objects")], "gpuwm-obs.mrms-fetch.v1", timeout=timeout)
    source = fetched["files"][0]["path"]
    pack, grid = folder / "field.obspack", folder / "grid.geopack"
    crop = ["--bbox", ",".join(map(str, bbox))] if bbox is not None else []
    decoded = _door(MRMS, "decode", [*product, "--file", source, "--out", str(pack), *crop],
                   "gpuwm-obs.mrms-decode.v1", timeout=timeout)
    _door(MRMS, "grid", [*product, "--file", source, "--out", str(grid), *crop],
          "gpuwm-obs.mrms-grid.v1", timeout=timeout)
    _atomic_json(folder / "provenance.json", {"selected": nearest, "fetch": fetched, "decode": decoded})
    return {"quantity": row["key"], "path": str(pack), "grid_path": str(grid)}


def _cached_observation(row, folder):
    """Select already decoded local files; native code validates their data."""
    if row["key"] == "stations":
        path = folder / "surface.json"
        return str(path) if path.is_file() else None
    pack, grid = folder / "field.obspack", folder / "grid.geopack"
    if pack.is_file() and grid.is_file():
        return {"quantity": row["key"], "path": str(pack), "grid_path": str(grid)}
    return None


def _outputs_present(entry):
    artifacts = entry.get("artifacts", [])
    for row in artifacts:
        path = Path(row["path"])
        if not path.is_file() or not path.stat().st_size:
            return False
        if row.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            return False
    return bool(artifacts)


def _preserve_inputs(root, requests, state, *, station_mode, timeout, reference, reference_dir):
    """Preserve all future inputs before spending the observation budget."""
    from woof.rustwx import prepare_verification

    preserved = []
    for original in requests:
        request = json.loads(json.dumps(original))
        request.setdefault("schema", REQUEST_SCHEMA)
        request["out_root"] = str(root)
        request["station_mode"] = station_mode
        domain, when = request["domain"], _time(request["valid_time"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", domain):
            raise ValueError("a verification domain must be one folder name")
        key = f"{domain}/{_stamp(when)}"
        entry = state["hours"].setdefault(key, {"valid_time": _stamp(when), "domain": domain})
        entry["request"] = request
        if any(arm["kind"] in ("netcdf", "grib", "store") for arm in request["arms"]):
            try:
                source_signature = _signature(request, include_verifier=False)
                previous = entry.get("prepared_input")
                if (previous and entry.get("source_signature") == source_signature
                        and all(Path(arm["path"]).is_file() for arm in previous["arms"])
                        and entry.get("input_validated_signature") == _first_input_signature(previous)):
                    request = {**request, **json.loads(json.dumps(previous))}
                else:
                    entry.pop("paired_input", None)
                    cache = root / domain / "verification" / when.strftime("%Y-%m-%d") / "observations" / when.strftime("%H%M%S")
                    request = prepare_verification(request, workdir=cache, timeout=max(timeout, 900))
                    entry["prepared_input"] = json.loads(json.dumps(request))
                    entry["source_signature"] = source_signature
                    entry["input_validated_signature"] = _first_input_signature(request)
                entry["request"] = json.loads(json.dumps(request))
                entry["input_status"] = "ready"
                entry.pop("input_reason", None)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                entry.update(status="pending", input_status="pending", input_reason=str(error), reason=str(error))
        else:
            entry["input_status"] = "ready"
            entry.pop("input_reason", None)
        if entry.get("input_status") == "ready":
            selection = _reference_selection(request, entry, reference, reference_dir)
            request = _reuse_pair(request, entry, selection)
            entry["request"] = json.loads(json.dumps(request))
        preserved.append(request)
        _atomic_json(root / MANIFEST_NAME, state)
    return preserved


def score_available(root: Path, *, requests=None, cycle=None, refresh=False,
                    station_mode="observed", timeout=120, budget_seconds=None,
                    now=None, append_to=None, reference="hrrr", reference_dir=None,
                    local_only=False) -> dict:
    """Score each available source independently; retain retryable pending rows."""
    from woof.rustwx import (verification_inventory, verify_observations,
                             verification_reference, prepare_verification)

    root = Path(root).resolve()
    now = _time(now or datetime.now(UTC))
    if requests is None:
        previous_state = _read(root / MANIFEST_NAME, {}) or {}
        old = {key: row["request"] for key, row in previous_state.get("hours", {}).items()
               if row.get("request")}
        for row in discover_histories(root, cycle=cycle):
            old[f"{row['domain']}/{row['valid_time']}"] = row
        requests = list(old.values())
    requests = list(requests)
    with _run_lock(root), _deadline_scope(None), _local_scope(local_only):
        state = _read(root / MANIFEST_NAME, {}) or {}
        state.update(schema=STATE_SCHEMA, updated_at=_stamp(now),
                     retry_command=f"woof verify-visuals \"{root}\"", run_dir=str(root))
        entries = state.setdefault("hours", {})
        engine_identity = {"forecast": _forecast_engine_identity(root),
                           "verifier": _verifier_engine_identity()}
        state["engine_identity"] = engine_identity
        requests = _preserve_inputs(root, requests, state, station_mode=station_mode, timeout=timeout,
                                   reference=reference, reference_dir=reference_dir)
        deadline = None if budget_seconds is None else time.monotonic() + budget_seconds
        _CALL_DEADLINE.set(deadline)
        for original in requests:
            request = json.loads(json.dumps(original))
            request.setdefault("schema", REQUEST_SCHEMA)
            when = _time(request["valid_time"])
            domain = request["domain"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", domain):
                raise ValueError("a verification domain must be one folder name")
            key = f"{domain}/{_stamp(when)}"
            entry = entries.setdefault(key, {"valid_time": _stamp(when), "domain": domain})
            entry["request"] = original
            sources = entry.setdefault("sources", {})
            output = root / domain / "verification" / when.strftime("%Y-%m-%d")
            cache = output / "observations" / when.strftime("%H%M%S")
            request["out_root"] = str(root)
            request["station_mode"] = station_mode
            if entry.get("input_status") == "pending":
                entry.update(status="pending", reason=entry["input_reason"])
                continue
            if deadline is not None and time.monotonic() >= deadline:
                entry["status"] = "pending"
                entry["reason"] = "finish-time verification budget ended; rerun the retry command"
                continue
            try:
                base_signature = _signature(request)
                if when > now:
                    entry["status"] = "pending"
                    entry["reason"] = "valid time has not occurred"
                    continue
                prepared = entry.get("prepared_request")
                if (not refresh and prepared and _outputs_present(entry)
                        and entry.get("base_signature") == base_signature
                        and entry.get("input_validated_signature") == _first_input_signature(request)
                        and _sources_complete(sources)
                        and entry.get("signature") == _signature(prepared)):
                    entry["status"] = "ready"
                    entry.pop("reason", None)
                    continue
                inventory = verification_inventory(request, workdir=cache, timeout=_timeout(timeout))
                entry["input_validated_signature"] = _first_input_signature(request)
                bbox = inventory.get("bbox")
                if not request.get("dx_km"):
                    request["dx_km"] = inventory.get("dx_km") or 0.0
                if inventory.get("dy_km") is not None:
                    request.setdefault("dy_km", inventory["dy_km"])
                if reference and len(request["arms"]) == 1:
                    try:
                        origin = request.get("cycle") or inventory.get("init_time") or cycle
                        if not origin:
                            raise ValueError("reference cycle is absent from the native history metadata; supply --cycle")
                        origin = _time(origin)
                        lead = (when - origin).total_seconds() / 3600
                        if lead < 0 or not lead.is_integer():
                            raise ValueError("reference time is not a nonnegative whole-hour forecast lead")
                        ref = verification_reference(cache=cache / "reference", reference=reference,
                                                     cycle=origin, hour=int(lead), timeout=_timeout(timeout),
                                                     reference_dir=reference_dir or request.get("reference_dir"),
                                                     reference_file=request.get("reference_file"),
                                                     **({"offline": True} if local_only else {}))
                        request["arms"] = [*request["arms"], {"label": ref["label"], "kind": "grib",
                            "path": ref["path"], "model": ref["model"], "hour": ref["hour"]}]
                        request = prepare_verification(request, workdir=cache / "paired-inputs",
                                                       timeout=_timeout(max(timeout, 900)))
                        entry["request"] = json.loads(json.dumps(request))
                        entry["paired_input"] = json.loads(json.dumps(request))
                        entry["paired_first_signature"] = _first_input_signature(request)
                        entry["reference_selection"] = _reference_selection(request, entry, reference, reference_dir)
                        base_signature = _signature(request)
                        sources["reference"] = {"status": "ready", "input": request["arms"][-1]["path"]}
                        inventory = verification_inventory(request, workdir=cache, timeout=_timeout(timeout))
                        entry["input_validated_signature"] = _first_input_signature(request)
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                        sources["reference"] = {"status": "pending", "reason": str(error)}
                support_sets = [set(arm.get("quantities", [])) for arm in inventory.get("arms", [])]
                supported = set.intersection(*support_sets) if support_sets else set()
                for row in OBSERVATION_ROWS:
                    source_key = row["key"]
                    if support_sets and source_key != "stations" and source_key not in supported:
                        sources[source_key] = {"status": "unavailable", "reason": "forecast input lacks this field"}
                        continue
                    previous = sources.get(source_key, {})
                    final = now >= when + timedelta(minutes=row["final_minutes"])
                    supplied = (request.get("stations_path") if source_key == "stations" else
                                next((item for item in request.get("radar", [])
                                      if item["quantity"] == source_key), None))
                    if supplied:
                        sources[source_key] = {"status": "ready", "input": supplied,
                                               "origin": "supplied observation input"}
                        continue
                    if (local_only or not refresh) and previous.get("status") in (
                            ("ready", "provisional") if local_only else ("ready",)):
                        value = previous.get("input")
                        if value and all(Path(path).is_file() for path in
                                         ([value] if isinstance(value, str) else
                                          [value["path"], value["grid_path"]])):
                            if source_key == "stations":
                                request["stations_path"] = value
                            else:
                                request.setdefault("radar", []).append(value)
                            continue
                    if local_only:
                        value = _cached_observation(row, cache / source_key)
                        if value:
                            sources[source_key] = {"status": "provisional", "input": value,
                                "origin": "local cache; archive finality unverified",
                                "final_after": _stamp(when + timedelta(minutes=row["final_minutes"]))}
                            if source_key == "stations":
                                request["stations_path"] = value
                            else:
                                request.setdefault("radar", []).append(value)
                        else:
                            sources[source_key] = {"status": "pending",
                                "reason": "local-only finish has no cached observation input; rerun the retry command online"}
                        continue
                    if deadline is not None and time.monotonic() >= deadline:
                        sources[source_key] = {"status": "pending", "reason": "finish-time budget ended"}
                        continue
                    try:
                        source_folder = cache / source_key
                        value = (fetch_stations(request, bbox, source_folder, timeout=timeout, refresh=refresh)
                                 if source_key == "stations" else
                                 fetch_radar(request, row, bbox, source_folder, timeout=timeout, refresh=refresh))
                        value = str(value) if isinstance(value, Path) else value
                        sources[source_key] = {"status": "ready" if final else "provisional", "input": value,
                                               "fetched_at": _stamp(now),
                                               "final_after": _stamp(when + timedelta(minutes=row["final_minutes"]))}
                        if source_key == "stations":
                            request["stations_path"] = value
                        else:
                            request.setdefault("radar", []).append(value)
                    except (OSError, ValueError, RuntimeError, LookupError, subprocess.SubprocessError) as error:
                        sources[source_key] = {"status": "pending", "reason": str(error)}
                if not request.get("stations_path") and not request.get("radar"):
                    entry.update(status="pending", reason="no observation source is available")
                    continue
                signature = _signature(request)
                if not refresh and entry.get("signature") == signature and _outputs_present(entry):
                    entry["status"] = "ready" if _sources_complete(sources) else "partial"
                    continue
                receipt_path = output / f"verification_{when.strftime('%H%M%S')}.json"
                card = output / f"scorecard_{when.strftime('%H%M%S')}.png"
                receipt = verify_observations(request, receipt_path=receipt_path, image_path=card,
                                              timeout=_timeout(max(timeout, 900)))
                score_gaps = [{"quantity": row["quantity"], "status": row.get("status")}
                              for row in [*receipt.get("stations", []), *receipt.get("radar", [])]
                              if row.get("status", "ready") not in ("ready", "no-observed-events")]
                entry.update(signature=signature, base_signature=base_signature, receipt_path=str(receipt_path),
                             prepared_request=request,
                             artifacts=receipt.get("artifacts", []),
                             score_gaps=score_gaps,
                             engine_identity=engine_identity,
                             status="ready" if _sources_complete(sources) else "partial")
                entry.pop("reason", None)
            except (OSError, ValueError, RuntimeError, LookupError, subprocess.SubprocessError) as error:
                entry.update(status="pending", reason=str(error))
            finally:
                _atomic_json(root / MANIFEST_NAME, state)
        _atomic_json(root / MANIFEST_NAME, state)
        counts = {status: sum(row.get("status") == status for row in entries.values())
                  for status in ("ready", "partial", "pending")}
        state["counts"] = counts
        state["status"] = ("unavailable" if not entries else
                           "ready" if counts["ready"] == len(entries)
                           else "partial" if counts["ready"] + counts["partial"] else "pending")
        if entries:
            state.pop("reason", None)
        else:
            state["reason"] = "the run retained no history fields or point extracts for verification"
        _atomic_json(root / MANIFEST_NAME, state)
        _release_retention(root, state)
        if append_to:
            append_results(state, Path(append_to))
        return state


def append_results(state, folder: Path):
    """Append native score rows once, retaining each document's byte prefix."""
    folder.mkdir(parents=True, exist_ok=True)
    for filename, key in (("STATIONS.md", "stations"), ("MRMS.md", "radar")):
        path = folder / filename
        prefix = path.read_bytes() if path.is_file() else b""
        chunks = []
        for hour in state.get("hours", {}).values():
            receipt = _read(Path(hour.get("receipt_path", "")), {}) or {}
            rows = receipt.get(key, [])
            if not rows:
                continue
            identity = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:20]
            marker = f"<!-- observation-verification {hour['domain']} {hour['valid_time']} {key} {identity} -->"
            if marker.encode() in prefix:
                continue
            text = f"\n\n{marker}\n\n## Observed verification, {hour['valid_time']}\n\n"
            text += f"Native receipt: `{hour['receipt_path']}`.\n\n"
            engines = hour.get("engine_identity") or state.get("engine_identity") or {}
            text += (f"Forecast engine: {_engine_line(engines.get('forecast'))}. "
                     f"Scoring engine: {_engine_line(engines.get('verifier'))}.\n\n")
            for method_key in ("temperature_method", "station_interpolation", "wind_metric"):
                if receipt.get(method_key):
                    text += f"{method_key.replace('_', ' ').capitalize()}: {receipt[method_key]}.\n\n"
            text += "| Quantity | Units | Threshold | Width km | Status | Model | Bias | RMSE | FSS | Count | Observed events | Winner |\n"
            text += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
            for row in rows:
                for arm in row.get("arms", []):
                    values = [row.get("quantity", ""), row.get("units", ""), row.get("threshold", ""),
                              row.get("width_km", ""), row.get("status", ""),
                              arm.get("label", ""), arm.get("bias", ""), arm.get("rmse", ""),
                              arm.get("fss", ""), arm.get("count", ""),
                              row.get("observed_event_cells", ""), row.get("winner", "")]
                    text += "| " + " | ".join(str(value) if value is not None else "unavailable" for value in values) + " |\n"
            chunks.append(text.encode("utf-8"))
        if chunks:
            with path.open("ab") as stream:
                stream.write(b"".join(chunks))
                stream.flush()
                os.fsync(stream.fileno())
            if not path.read_bytes().startswith(prefix):
                raise RuntimeError(f"the existing report prefix changed: {path}")


def finish_run(root, *, sections=None) -> dict | None:
    """Queue local-only output with a two second foreground cap; OFF is inert."""
    if os.environ.get("WOOF_VERIFY_VISUALS", "1").lower() in ("0", "false", "off"):
        return None
    result = {}

    def launch():
        try:
            result["state"] = _finish_run(root, sections=sections)
        except Exception as error:  # Output failure cannot change a completed forecast verdict.
            result["state"] = {"schema": BACKGROUND_SCHEMA, "status": "pending",
                "reason": f"{type(error).__name__}: {error}",
                "retry_command": f'woof verify-visuals "{root}"'}

    deadline = time.monotonic() + FINISH_FOREGROUND_SECONDS
    launcher = threading.Thread(target=launch, name="verification-launcher", daemon=True)
    try:
        launcher.start()
        launcher.join(timeout=max(0, deadline - time.monotonic()))
    except RuntimeError as error:
        return {"schema": BACKGROUND_SCHEMA, "status": "pending", "reason": str(error),
                "retry_command": f'woof verify-visuals "{root}"'}
    state = result.get("state") or {"schema": BACKGROUND_SCHEMA, "status": "pending",
        "reason": "local verification queue exceeded its two second foreground cap; retain history inputs until compact preparation completes",
        "retry_command": f'woof verify-visuals "{root}"'}
    return state


def finish_line(state) -> str | None:
    """The one line a finished forecast prints about its verification step.

    The step used to be silent: a detached CPU process started after the last
    model step and could outlive the forecast's card lease, and nothing named
    it or its switch (2026-11-11, an outside runner's pool record
    ``cpu-only-tail=2``).
    """
    if not isinstance(state, dict):
        return None
    record = Path(state["run_dir"]) / BACKGROUND_NAME if state.get("run_dir") else None
    if state.get("pid"):
        return (f"verification: CPU check queued as process {state['pid']} (no card), which "
                f"may run on after this forecast exits; record {record}; "
                f"--no-verify-visuals or WOOF_VERIFY_VISUALS=0 skips it, and then the "
                f"history is not made compact")
    if state.get("status") == "pending" and state.get("reason"):
        return f"verification: pending ({state['reason']}); later: {state.get('retry_command')}"
    return None


def _configured_cycle(config):
    """The forecast's start time from its experiment TOML, as the worker reads it."""
    if not config or Path(config).suffix.lower() != ".toml":
        return None
    import tomllib
    metadata = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    return metadata.get("experiment", {}).get("start_time")


def _finish_run(root, *, sections=None) -> dict | None:
    """Only the daemon launcher performs metadata I/O and process creation."""
    root = Path(root).resolve()
    sections = sections or {}
    cycle = sections.get("run_shape", {}).get("start_time")
    config = sections.get("run_context", {}).get("config_bytes", {}).get("path")
    frames = sections.get("output", {}).get("frames", [])
    paths = [str(row["path"]) for row in frames if isinstance(row, dict) and row.get("path")]
    declared_empty = "frames" in sections.get("output", {}) and not paths
    with _run_lock(root, name=".verification-background.lock"):
        previous = _read(root / BACKGROUND_NAME, {}) or {}
        if previous.get("schema") == BACKGROUND_SCHEMA and _background_active(root, previous):
            expected = {item["key"]: item for item in previous.get("expected_inputs", [])}
            expected.update({item["key"]: item for item in _expected_inputs(frames)})
            merged = list(expected.values())
            if merged != previous.get("expected_inputs", []):
                previous.update(expected_inputs=merged, history_paths=[item["path"] for item in merged],
                                lease_token=_ensure_retention(root, previous["token"]),
                                compact_inputs_ready=False, declared_empty=False)
                _atomic_json(root / BACKGROUND_NAME, previous)
            return previous
        token = uuid.uuid4().hex
        lease_token = None if declared_empty else _ensure_retention(root, token)
        job = {"schema": BACKGROUND_SCHEMA, "phase": "queued", "status": "pending",
               "token": token, "lease_token": lease_token, "run_dir": str(root),
               "cycle": _stamp(_time(cycle)) if cycle is not None else None,
               "config_path": str(config) if config else None, "history_paths": paths,
               "expected_inputs": _expected_inputs(frames),
               "declared_empty": declared_empty,
               "local_only": True, "foreground_cap_seconds": FINISH_FOREGROUND_SECONDS,
               "queued_at": _stamp(datetime.now(UTC)),
               "retry_command": f'woof verify-visuals "{root}"'}
        # A run whose hours are all still ahead queues the worker too: it
        # makes the history's compact verification inputs and releases the
        # .keep lease, which is work even when no hour can be scored yet
        # (item 8 review; the worker is CPU only and holds no card).
        _atomic_json(root / BACKGROUND_NAME, job)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1", WOOF_VERIFY_VISUALS="0")
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_root, env.get("PYTHONPATH"))))
        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        try:
            with (root / "verification-background.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, "-m", "woof.verification_visuals",
                    "--finish-worker", str(root), "--job-token", token],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, close_fds=True, **options)
            _BACKGROUND_PROCESSES[:] = [p for p in _BACKGROUND_PROCESSES if p.poll() is None]
            _BACKGROUND_PROCESSES.append(process)
            _atomic_json(_worker_identity_path(root, token), _process_identity(process.pid) or {"pid": process.pid})
            return {**job, "pid": process.pid}
        except Exception as error:
            job.update(phase="failed", reason=f"background launch failed: {error}")
            _atomic_json(root / BACKGROUND_NAME, job)
            raise


def _ensure_retention(root, token):
    lease = {"schema": RETENTION_SCHEMA, "token": token,
             "reason": "native compact verification inputs are not yet ready",
             "retry_command": f'woof verify-visuals "{root}"'}
    keep = root / ".keep"
    try:
        with keep.open("x", encoding="utf-8") as stream:
            json.dump(lease, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return token
    except FileExistsError:
        existing = _read(keep, {}) or {}
        return existing.get("token") if existing.get("schema") == RETENTION_SCHEMA else None


def _expected_inputs(frames):
    expected = []
    for frame in frames:
        if not isinstance(frame, dict) or not frame.get("path"):
            continue
        path = Path(frame["path"])
        matched = HISTORY_NAME.fullmatch(path.name)
        if matched:
            domain, date, hour, minute, second = matched.groups()
            expected.append({"key": f"{domain}/{date}T{hour}:{minute}:{second}Z",
                             "path": str(path.resolve()), "sha256": frame.get("sha256")})
    return expected


def _worker_identity_path(root, token):
    return root / f".verification-worker-{token}.json"


def _process_identity(pid):
    """Query liveness and process birth without sending a signal on Windows."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return {"pid": pid, "birth": None} if ctypes.get_last_error() == 5 else None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            birth = None
            if kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
                birth = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            return {"pid": pid, "birth": birth}
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
            return None if fields[0] == "Z" else {"pid": pid, "birth": fields[19]}
        except FileNotFoundError:
            return None
        except OSError:
            return {"pid": pid, "birth": None}
    try:
        os.kill(int(pid), 0)
        return {"pid": pid, "birth": None}
    except ProcessLookupError:
        return None
    except PermissionError:
        return {"pid": pid, "birth": None}


def _background_active(root, job):
    if job.get("phase") not in ("queued", "running") or not job.get("token"):
        return False
    saved = job.get("process_identity") or _read(_worker_identity_path(root, job["token"]), {})
    if saved and saved.get("pid"):
        current = _process_identity(saved["pid"])
        return current is not None and (not saved.get("birth") or not current.get("birth")
                                        or saved["birth"] == current["birth"])
    queued = job.get("queued_at")
    return bool(queued and datetime.now(UTC) < _time(queued) + timedelta(seconds=30))


def _lineage_matches(request, expected):
    identity = request.get("arms", [{}])[0].get("source_identity")
    while isinstance(identity, dict):
        artifact = identity.get("artifact", {})
        if (artifact.get("path") and str(Path(artifact["path"]).resolve()) == expected["path"]
                and artifact.get("sha256")
                and (not expected.get("sha256") or artifact["sha256"] == expected["sha256"])):
            return True
        identity = identity.get("prepared_from")
    return False


def _background_job_update(root, token, *, expected_inputs_match=None, wait_seconds=0, **updates):
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            with _run_lock(root, name=".verification-background.lock"):
                job = _read(root / BACKGROUND_NAME, {}) or {}
                if job.get("schema") != BACKGROUND_SCHEMA or job.get("token") != token:
                    return None
                if expected_inputs_match is not None and job.get("expected_inputs", []) != expected_inputs_match:
                    return None
                job.update(updates)
                _atomic_json(root / BACKGROUND_NAME, job)
                return job
        except OSError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.05, remaining))


def _release_retention(root, state):
    # A concurrent finish can replace the job token. Never unlink a lease or
    # overwrite that new job using an old online retry's snapshot.
    try:
        with _run_lock(root, name=".verification-background.lock"):
            return _release_retention_locked(root, state)
    except OSError:
        return False


def _release_retention_locked(root, state):
    """Release only this job's lease after all forecast inputs are compact."""
    entries = list(state.get("hours", {}).values())
    if not entries or any(row.get("input_status") != "ready" or
           any(arm.get("kind") in ("netcdf", "grib", "store") or not Path(arm["path"]).is_file()
               for arm in row.get("request", {}).get("arms", [])) or
            row.get("input_validated_signature") != _first_input_signature(row["request"])
            for row in entries):
        return False
    job = _read(root / BACKGROUND_NAME, {}) or {}
    keep = root / ".keep"
    lease = _read(keep, {}) or {}
    if (job.get("schema") != BACKGROUND_SCHEMA or lease.get("schema") != RETENTION_SCHEMA
            or not job.get("lease_token") or job["lease_token"] != lease.get("token")):
        return False
    expected = job.get("expected_inputs", [])
    if not expected or any(item["key"] not in state["hours"] or
            not _lineage_matches(state["hours"][item["key"]]["request"], item) for item in expected):
        return False
    keep.unlink()
    job["compact_inputs_ready"] = True
    _atomic_json(root / BACKGROUND_NAME, job)
    return True


def _finish_worker(root, token, *, coverage_attempt=0):
    """Detached CPU work: retain native inputs and score local caches only."""
    root = Path(root).resolve()
    job = _background_job_update(root, token, phase="running", pid=os.getpid(),
        process_identity=_process_identity(os.getpid()), started_at=_stamp(datetime.now(UTC)), wait_seconds=10)
    if job is None:
        return 2
    try:
        if job.get("declared_empty"):
            with _run_lock(root):
                state = {"schema": STATE_SCHEMA, "status": "unavailable", "hours": {},
                         "counts": {"ready": 0, "partial": 0, "pending": 0}, "run_dir": str(root),
                         "reason": "the completed run explicitly declared no retained history frames",
                         "retry_command": f'woof verify-visuals "{root}"', "updated_at": _stamp(datetime.now(UTC))}
                _atomic_json(root / MANIFEST_NAME, state)
            completed = _background_job_update(root, token, expected_inputs_match=job.get("expected_inputs", []),
                phase="finished", status="unavailable", counts=state["counts"], finished_at=_stamp(datetime.now(UTC)))
            if completed is None:
                current = _read(root / BACKGROUND_NAME, {}) or {}
                if (coverage_attempt < 3 and current.get("token") == token
                        and current.get("expected_inputs") != job.get("expected_inputs")):
                    return _finish_worker(root, token, coverage_attempt=coverage_attempt + 1)
                return 2
            return 0
        cycle = job.get("cycle")
        config = job.get("config_path")
        if cycle is None:
            cycle = _configured_cycle(config)
        paths = [Path(path) for path in job.get("history_paths", [])]
        paths = [*histories_before_restart(paths, cycle), *paths]
        requests = discover_histories(root, paths=paths or None, cycle=cycle, include_missing=bool(paths))
        if not job.get("expected_inputs"):
            job["expected_inputs"] = _expected_inputs([{"path": request["arms"][0]["path"]} for request in requests])
            if _background_job_update(root, token, expected_inputs_match=[], expected_inputs=job["expected_inputs"]) is None:
                return 2
        old = (_read(root / MANIFEST_NAME, {}) or {}).get("hours", {})
        for index, request in enumerate(requests):
            if not Path(request["arms"][0]["path"]).is_file():
                expected = next(item for item in job["expected_inputs"]
                                if item["key"] == f'{request["domain"]}/{request["valid_time"]}')
                saved = old.get(expected["key"], {}).get("request")
                if saved and _lineage_matches(saved, expected):
                    requests[index] = saved
        # Let saved compact requests survive a repeated finish with no raw frames.
        state = score_available(root, requests=requests or None, cycle=cycle, timeout=30,
                                budget_seconds=120, local_only=True)
        completed = _background_job_update(root, token, expected_inputs_match=job.get("expected_inputs", []),
            phase="finished", status=state["status"], counts=state["counts"], finished_at=_stamp(datetime.now(UTC)))
        if completed is None:
            current = _read(root / BACKGROUND_NAME, {}) or {}
            if (coverage_attempt < 3 and current.get("token") == token
                    and current.get("expected_inputs") != job.get("expected_inputs")):
                return _finish_worker(root, token, coverage_attempt=coverage_attempt + 1)
            return 2
        return 0
    except Exception as error:
        _background_job_update(root, token, phase="failed", status="pending", reason=f"{type(error).__name__}: {error}",
                               finished_at=_stamp(datetime.now(UTC)))
        return 2


def command_main(args):
    try:
        if args.manifest:
            document = _read(args.manifest)
            requests = document["hours"] if isinstance(document, dict) else document
            if isinstance(requests, dict):
                requests = [row.get("request", row) for row in requests.values()]
        elif args.point_arm or args.field_arm:
            if not args.cycle:
                raise ValueError("packaged verification needs --cycle to bind each valid hour")
            first = 1 if args.first_hour is None else args.first_hour
            last = 48 if args.last_hour is None else args.last_hour
            requests = discover_packaged(cycle=args.cycle, domain=args.domain or "d01",
                point_arms=args.point_arm, field_arms=args.field_arm, grid=args.grid,
                station_table=args.station_table, hours=range(first, last + 1))
        elif (args.domain is not None or args.first_hour is not None
              or args.last_hour is not None):
            requests = select_run_requests(
                args.run_dir, cycle=args.cycle, domain=args.domain,
                first_hour=args.first_hour, last_hour=args.last_hour)
        else:
            requests = None
        if args.list_pending:
            state = _read(args.run_dir / MANIFEST_NAME, {})
        else:
            state = score_available(args.run_dir, requests=requests, cycle=args.cycle,
                station_mode=args.station_mode, refresh=args.refresh, timeout=args.timeout,
                append_to=args.append_to, reference=args.reference, reference_dir=args.reference_dir)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"verify-visuals: {error}")
        return 2


def domain_id(text: str) -> str:
    """``--domain`` as the frames name it: ``1``, ``d1``, ``D01`` and ``d01`` are all ``d01``.

    THE BREAKAGE THIS PREVENTS (3.0.3 candidate check, 2026-11-15):
    ``woof verify-visuals RUN --domain 1`` exited 2 with "no finished frame
    matches ... this run holds d01", which reads as if the run had no frame,
    when the only fault was the spelling of the domain.
    """
    matched = re.fullmatch(r"\s*[dD]?0*([1-9][0-9]?)\s*", str(text))
    if not matched:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a domain: give its number or its frame name, for example 1 or d01")
    return f"d{int(matched.group(1)):02d}"


def register_cli(subparsers):
    parser = subparsers.add_parser("verify-visuals", help="score and render finished hours against arrived observations")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="JSON list of native per-hour requests, or {hours: [...]} object")
    parser.add_argument("--cycle", help="UTC cycle, YYYY-MM-DDTHH[:MM:SS]Z")
    parser.add_argument("--reference", default="hrrr", help="reference row in the native renderer table")
    parser.add_argument("--reference-dir", type=Path, help="existing native reference files, named by its metadata table")
    parser.add_argument("--domain", default=None, type=domain_id, metavar="N|dNN",
                        help="on a run folder, score this domain only, by number or frame name "
                             "(1 and d01 are the same; default: every domain); "
                             "with packaged arms, the domain they belong to (default d01)")
    parser.add_argument("--point-arm", action="append", default=[], metavar="LABEL=DIR")
    parser.add_argument("--field-arm", action="append", default=[], metavar="LABEL=TEMPLATE",
                        help="NPZ path template, for example LABEL=fields/run-f{hour:02d}.npz")
    parser.add_argument("--grid", type=Path, help="native lat/lon NPZ for archived field arrays")
    parser.add_argument("--station-table", type=Path, help="frozen station table used by point extracts")
    parser.add_argument("--first-hour", type=int, default=None,
                        help="first forecast hour to score, counted from --cycle or, on a run folder "
                             "without it, from the run's first frame (default: 1 for packaged arms, "
                             "every finished hour of a run folder)")
    parser.add_argument("--last-hour", type=int, default=None,
                        help="last forecast hour to score (default: 48 for packaged arms, every "
                             "finished hour of a run folder)")
    parser.add_argument("--station-mode", choices=("observed", "error"), default="observed")
    parser.add_argument("--timeout", type=int, default=120, help="seconds allowed for each public-source command")
    parser.add_argument("--refresh", action="store_true", help="refetch observations and regenerate receipts")
    parser.add_argument("--list-pending", action="store_true", help="print the durable verification state without fetching")
    parser.add_argument("--append-to", type=Path, help="append station and radar rows to STATIONS.md and MRMS.md")
    parser.set_defaults(func=command_main)
    return parser


if __name__ == "__main__":
    worker_parser = argparse.ArgumentParser(description="local-only finished-run verification worker")
    worker_parser.add_argument("--finish-worker", type=Path, required=True)
    worker_parser.add_argument("--job-token", required=True)
    worker_args = worker_parser.parse_args()
    raise SystemExit(_finish_worker(worker_args.finish_worker, worker_args.job_token))
