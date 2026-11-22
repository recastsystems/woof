"""Acquire the existing native ERA5 input contract through authenticated CDS."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

from woof.filesystem_paths import publish_new


_RECEIPT = "era5-acquisition.json"
_SCHEMA = "arwen.era5-acquisition.v1"


def _silent(*args, **kwargs):
    """Do not pass client credentials or raw server diagnostics into job logs."""


def _client(progress):
    try:
        import cdsapi
    except ImportError:
        raise ValueError("Automatic ERA5 retrieval needs cdsapi>=0.7.7 in this WOOF-ArWen Python environment.") from None
    try:
        # Both legacy and modern CDS clients accept these callbacks. quiet
        # alone does not disable an already configured DEBUG logger.
        state = [None]
        def information(message, *args, **kwargs):
            text = str(message).lower()
            for name in ("accepted", "queued", "running", "successful", "completed"):
                if name in text and name != state[0]:
                    state[0] = name
                    progress(f"fetch era5: CDS request {'queued' if name == 'accepted' else name}")
                    break
        return cdsapi.Client(quiet=True, debug=False, progress=False,
            info_callback=information, warning_callback=_silent,
            error_callback=_silent, debug_callback=_silent)
    except Exception as error:
        # The sentence carries the client's own error (key values masked),
        # the file it looked for and whether that file exists, the source
        # it would use and the way out for that case.  One fixed sentence
        # here left a user who had entered a key being told to enter a key.
        from woof.cds_credentials import client_refusal
        raise ValueError(client_refusal(error)) from None


def _validate(path, *, times, area):
    from woof import fetch
    report = fetch.validate_era5_files((path,), expected_times=times, expected_area=area)
    if not report.ok:
        raise ValueError("Retrieved ERA5 did not pass native input validation:\n" + report.format())
    records = fetch.read_grib1_records(path)
    if {row.valid_time for row in records} != set(times):
        raise ValueError("ERA5 valid times differ from the exact requested boundary schedule")
    levels = {row.level for row in records
              if row.level_type == 100 and row.parameter in fetch.ERA5_REQUIRED_PRESSURE}
    if levels != set(fetch.ERA5_PRESSURE_LEVELS_HPA):
        raise ValueError("ERA5 retrieval must contain the full requested 37 pressure levels")
    if any(row.grid is None for row in records):
        raise ValueError("ERA5 retrieval lacks grid metadata needed to verify its requested area")
    # The new request includes an explicit lake-state provider. A complete
    # atmosphere alone cannot establish that the server delivered that state.
    # Keep the ordinary/manual ERA5 validator compatible with older files;
    # enforce the full requested inventory at this acquisition boundary.
    from woof.ingest.grib import _NATIVE_LAKE_SPECS
    reference_grids = {(row.grid, row.grid_definition_sha256) for row in records
                       if row.level_type == 100
                       and row.parameter in fetch.ERA5_REQUIRED_PRESSURE}
    if len(reference_grids) != 1:
        raise ValueError("ERA5 retrieval pressure fields do not share one source grid")
    reference_grid = next(iter(reference_grids))
    if reference_grid[1] is None:
        raise ValueError("ERA5 retrieval lacks native grid-definition identity")
    for moment in times:
        for identity, name in _NATIVE_LAKE_SPECS.items():
            matching = [row for row in records if row.valid_time == moment and
                        (row.center, row.table_version, row.parameter,
                         row.level_type, row.level) == identity]
            if len(matching) != 1:
                raise ValueError(f"ERA5 retrieval must contain exactly one {name} "
                                 f"at {moment.isoformat()} UTC")
            if (matching[0].grid, matching[0].grid_definition_sha256) != reference_grid:
                raise ValueError(f"ERA5 lake-state field {name} differs from "
                                 "the atmospheric source grid")
    return report


def retrieve_era5(*, cycle: datetime | str, hours: int, area,
                  out: str | Path, cadence: int = 6, force: bool = False,
                  product_type: str = "reanalysis", member: int | None = None,
                  progress=print) -> Path:
    """Retrieve, validate and publish one complete ``era5-combined.grib``.

    Inputs follow ``woof.fetch``; area may be its Area object or SWNE string.
    Reuse requires matching request metadata, file digest and fresh native
    validation. ``force`` preserves prior canonical files by quarantine only
    after a new complete response has passed validation.
    """
    from woof import fetch, fetch_guard

    if isinstance(cycle, str):
        cycle = fetch.parse_cycle(cycle, "era5")
    if not isinstance(cycle, datetime):
        raise ValueError("ERA5 cycle must be an explicit UTC date and hour")
    if cycle.tzinfo is not None:
        cycle = cycle.astimezone(timezone.utc).replace(tzinfo=None)
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("ERA5 cycle must fall on an exact UTC hour")
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 0:
        raise ValueError("ERA5 hours must be a nonnegative integer")
    from woof.era5_member import validate_selection
    validate_selection(cadence=cadence)
    if isinstance(area, str):
        area = fetch.parse_area(area)
    if not isinstance(area, fetch.Area):
        raise ValueError("ERA5 needs an explicit geographic area")
    times = fetch._era5_times(cycle, hours, cadence)
    from woof.era5_member import validate_selection, require_bridge, check_member, select_member
    member = validate_selection(product_type=product_type, member=member,
                                cadence=cadence, provider="cds", cycle=cycle)
    # Refuse a stale/missing native selector before submitting any CDS jobs.
    member_bridge = require_bridge() if member is not None else None
    out = Path(out).expanduser().resolve()
    target = out / fetch.ERA5_COMBINED_NAME
    receipt_path = out / _RECEIPT
    template = fetch.era5_request_template(cycle=cycle, hours=hours, area=area,
                                           cadence=cadence, out=out)
    requests = []
    for item in template["requests"]:
        # The template owns the exact per-day schedule. Convert only its
        # date grammar for CDS; never cross it with an independent day loop.
        day = datetime.strptime(item["request"]["date"][0], "%Y-%m-%d")
        selection = {key: value for key, value in item["request"].items()
                     if key not in {"date", "time"}}
        selection.update(product_type=[product_type], year=[f"{day.year:04d}"],
            month=[f"{day.month:02d}"], day=[f"{day.day:02d}"],
            time=item["request"]["time"], data_format="grib", download_format="unarchived")
        requests.append({"dataset": item["dataset"], "request": selection})
    identity = {"source": "era5", "cycle": cycle.isoformat() + "Z", "hours": hours,
                "cadence_hours": cadence, "area": area.as_manifest(), "requests": requests}
    if member is not None:
        identity.update(product_type=product_type, member=member, provider="cds",
                        member_selection_schema="arwen.era5-member-selection.v1")

    from woof import progress as progress_mod
    acquisition = {"schema": "arwen.acquisition-progress.v1", "source": "era5", "provider": "cds",
                   "files_total": len(requests), "requests_total": len(requests), "requests_completed": 0,
                   "forcing_hours": hours, "forcing_times_total": len(times)}
    def publish(phase, **fields):
        acquisition.update(phase=phase, **fields)
        progress_mod.emit_event("fetch_progress", acquisition=dict(acquisition))
    def information(message):
        progress(message)
        for state in ("queued", "running", "successful", "completed"):
            if message == "fetch era5: CDS request " + state:
                publish("cds_" + state)
                break

    with fetch_guard.hold("fetch-out", out, progress=progress):
        for path in (target, receipt_path):
            if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
                raise FileExistsError(f"ERA5 retrieval preserves the non-regular path: {path}")
        if not force and (target.exists() or receipt_path.exists()):
            try:
                prior = json.loads(receipt_path.read_text(encoding="utf-8"))
                artifact = prior["artifact"]
                matches = (prior["schema"] == _SCHEMA and prior["request"] == identity
                           and target.stat().st_size == artifact["bytes"]
                           and fetch.sha256_file(target) == artifact["sha256"])
                if matches:
                    if member is not None:
                        check_member(target, member, bridge=member_bridge)
                    _validate(target, times=times, area=area)
                    if fetch.sha256_file(target) == artifact["sha256"]:
                        progress(f"fetch era5: reused verified inputs in {target}")
                        publish("ready", reused=True, requests_completed=len(requests),
                                files_completed=len(requests), bytes_available=artifact["bytes"],
                                transferred_bytes=0, forcing_times_completed=len(times))
                        return target
            except (OSError, ValueError, KeyError, TypeError):
                pass
            raise FileExistsError(
                f"ERA5 output is not a verified match for this request: {target}. "
                "Choose another output directory or explicitly force a refetch; existing files were preserved.")

        client = _client(information)
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".era5-cds-", dir=out) as temporary:
            stage = Path(temporary)
            parts = []
            with progress_mod.TransferMonitor("fetch era5", interval=0.25) as monitor:
                for index, request in enumerate(requests):
                    part = stage / f"part-{index:04d}.grib"
                    day = request["request"]
                    progress(f"fetch era5: CDS request {index + 1}/{len(requests)} "
                             f"{request['dataset']} {day['year'][0]}-{day['month'][0]}-{day['day'][0]}")
                    publish("requesting", request_index=index + 1, dataset=request["dataset"])
                    try:
                        # The provider prepares the response before any bytes
                        # can move. Its result supplies the actual file size;
                        # CDS/multiurl writes .download while streaming it.
                        result = client.retrieve(request["dataset"], request["request"])
                        expected = getattr(result, "content_length", None)
                        expected = expected if type(expected) is int and expected > 0 else None
                        monitor.start(part.name, path=part, expected_bytes=expected)
                        publish("downloading")
                        result.download(str(part))
                    except Exception as error:
                        monitor.finish(part.name, failed=True)
                        publish("failed")
                        from woof.cds_credentials import retrieval_refusal
                        raise ValueError(retrieval_refusal(error)) from None
                    if not part.is_file() or part.stat().st_size == 0:
                        raise ValueError("CDS returned no ERA5 data file; nothing was published")
                    parts.append(part)
                    monitor.finish(part.name, size=part.stat().st_size)
                    publish("request_completed", requests_completed=index + 1)
            combined = stage / fetch.ERA5_COMBINED_NAME
            with combined.open("xb") as stream:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            selection_report = None
            if member is not None:
                progress(f"fetch era5: verifying all ten EDA identities and selecting member {member}")
                selected = stage / "era5-single-member.grib"
                selection_report = select_member(combined, selected, member, bridge=member_bridge)
                combined = selected
            publish("validating", bytes_available=combined.stat().st_size)
            progress("fetch era5: validating times, fields, levels and geographic coverage")
            report = _validate(combined, times=times, area=area)
            receipt = {"schema": _SCHEMA, "status": "validated", "request": identity,
                "artifact": {"name": target.name, "bytes": combined.stat().st_size,
                             "sha256": fetch.sha256_file(combined)},
                "validation": {"checks": list(report.checks), "failures": []},
                "forecast_started": False}
            if selection_report is not None:
                receipt["member_selection"] = selection_report
            staged_receipt = stage / _RECEIPT
            fetch_guard.atomic_write_text(staged_receipt,
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", tag="era5")
            if force:
                for path in (receipt_path, target):
                    if os.path.lexists(path):
                        if path.is_symlink() or not path.is_file():
                            raise FileExistsError(f"ERA5 retrieval preserves the changed path: {path}")
                        aside = fetch_guard.quarantine(path, tag="era5-refetch")
                        progress(f"fetch era5: preserved previous {path.name} as {aside.name}")
            # Atomic create-only publication cannot overwrite a file that appeared
            # while CDS was working. The receipt is the final commit marker.
            publish_new(combined, target)
            # On a volume without hard links the stage's copy has moved, so the
            # rollback below recognises the published file by its own stat.
            published = target.stat()
            try:
                publish_new(staged_receipt, receipt_path)
            except BaseException:
                if target.is_file() and os.path.samestat(published, target.stat()):
                    target.unlink()
                raise
            fetch_guard._fsync_dir(out)
        publish("ready", files_completed=len(requests), forcing_times_completed=len(times), bytes_available=target.stat().st_size)
        progress(f"fetch era5: published validated inputs in {target}")
        return target
