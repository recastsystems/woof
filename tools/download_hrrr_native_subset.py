"""Download only the HRRR GRIB2 messages required by native initialization.

NOAA's ``.idx`` files are used solely as byte-range transport indexes.  The
native Rust bridge remains the scientific authority: it reopens every
downloaded GRIB message and validates numeric discipline/category/parameter,
surface coordinates, cycle, forecast hour, grid, and required cardinality.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.request import Request

from woof.hrrr_forecast import validate_hrrr_source_forecast_hours
from woof.nomads_governor import governed_workers, paced_urlopen


BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HYBRID_FIELDS = (
    "PRES", "CLMR", "CIMIXR", "RWMR", "SNMR", "GRLE",
    "HGT", "TMP", "SPFH", "UGRD", "VGRD",
)
#: The analyzed water- and ice-friendly aerosol number concentrations,
#: as the public index spells them: NOAA's operational hrrr_vtable binds
#: QNWFA and QNIFA to the local codes 0/13/193 and 0/13/192, which
#: indexes label PMTF and PMTC (woof/authorities/
#: rw-wps-hrrr-native-grib2.provenance.json).  Selected only when the
#: configuration's preparation reads them (an mp=28 domain asking for
#: the analyzed aerosol, woof.preparation_assets.analyzed_aerosol_domains):
#: the native bridge reads the pair whole when a file carries it
#: (OPTIONAL_HYBRID_SPECS in hrrr_grib2_bridge.rs), and a subset without
#: it refused that preparation "missing QNIFA, QNWFA".
ANALYZED_AEROSOL_FIELDS = ("PMTF", "PMTC")
SURFACE_FIELDS = (
    ("PRES", "surface"),
    ("HGT", "surface"),
    ("TMP", "surface"),
    ("WEASD", "surface"),
    ("SNOD", "surface"),
    ("TMP", "2 m above ground"),
    ("SPFH", "2 m above ground"),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
    ("LAND", "surface"),
    ("ICEC", "surface"),
)
SOIL_DEPTHS = ("0", "0.01", "0.04", "0.1", "0.3", "0.6", "1", "1.6", "3")
HYBRID_LEVEL = re.compile(r"([1-9]|[1-4][0-9]|50) hybrid level")

#: Index short names that denote the same physical record.
#:
#: Providers do not always agree on a field's short name.  NOMADS spells
#: HRRR hybrid cloud water ``CLWMR``; AWS S3 spells the same record
#: ``CLMR``, and has done as far back as 2025.  That is a **naming**
#: difference, not an inventory difference: same role, same count, same
#: bytes.  Treating it as drift would hard-fail a whole transport over a
#: spelling, so the role is what the selection keys on and either
#: spelling satisfies it.  A change in the record *count* remains the
#: loud ``--accept-inventory-change`` gate -- alias tolerance never
#: silently absorbs a field appearing or disappearing.
#:
#: Cloud ice is the other row.  HRRRv1 and v2 wrfnat files, published
#: before July 2018, index their hybrid cloud ice as ``CICE`` (GRIB2
#: 0/6/0) on the same 50 levels and carry no ``CIMIXR`` (0/1/82); the
#: archive's 2017-01-19 00Z wrfnatf00 and f01 indexes hold CICE 50 times
#: and CIMIXR never, the 2026-09-27 00Z wrfnatf00 index the reverse.  The
#: bridge reads 0/6/0 only from a file that publishes no 0/1/82, and
#: refuses a file that mixes the two.
FIELD_ALIASES = {
    "CLMR": ("CLMR", "CLWMR"),
    "CIMIXR": ("CIMIXR", "CICE"),
    # The HRRR start's number concentrations are requested by their decoder names (woof.ingest.native_extras);
    # the index spells them NCONCD, NCCICE and SPNCR.
    "QNC": ("NCONCD",),
    "QNI": ("NCCICE",),
    "QNR": ("SPNCR",),
}

#: Request-only single-level records, by decoder name: (index short name, index level).
#:
#: THE BREAKAGE THIS PREVENTS (a polygon hex run from HRRR, 2026-10-29): the HRRR start
#: (fix/hex-hrrr-start-init) asks the decoder for QNC, QNI, QNR and RH2, but the index subset had no selectors for
#: them, so the 561-record download lacked NCONCD/NCCICE/SPNCR and RH at 2 m and the decoder refused
#: "missing required field QNC hybrid level 1": every hex HRRR start without whole GRIB on disk failed.
SURFACE_EXTRA_FIELDS = {
    "RH2": ("RH", "2 m above ground"),
}

#: The single-level records hrrr_grib2_bridge reads from a wrfnat file
#: whenever the file publishes them, unasked (OPTIONAL_SURFACE_SPECS in
#: hrrr_grib2_bridge.rs: the cycle's PBL height, 0/3/18 surface, and its
#: low/middle/high cloud cover, 0/6/3-5 on surfaces 214/224/234), as the
#: index spells them.  They seed the start's PBLH and CLDFRA cover.
#:
#: THE BREAKAGE THIS PREVENTS: a whole wrfnat carries them and an index
#: subset that left them out did not, so the same cycle prepared two
#: different starts (PBLH = 0 everywhere and no cloud-cover seed from a
#: subset), and a series mixing the two byte modes was refused 80-146 s
#: into the preparation.  Selected only where the index lists them, so a
#: cycle that publishes none of them takes exactly the records it always
#: did; four records, about 4.7 MB per hour.
OPTIONAL_SURFACE_FIELDS = (
    ("HPBL", "surface"),
    ("LCDC", "low cloud layer"),
    ("MCDC", "middle cloud layer"),
    ("HCDC", "high cloud layer"),
)


def optional_surface_published(index_text: str) -> tuple[tuple[str, str], ...]:
    """The :data:`OPTIONAL_SURFACE_FIELDS` records this ``.idx`` lists.

    Instantaneous records only (the same ``acc fcst`` exclusion as every
    other selector), in table order.  A field listed twice is returned
    once: the selection then takes both records, and the bridge refuses
    the duplicate exactly as it would in a whole file.
    """

    listed = set()
    for line in index_text.splitlines():
        stripped = line.strip()
        if not stripped or ACCUMULATION_EXCLUSION in stripped:
            continue
        fields = stripped.rstrip(":").split(":")
        if len(fields) >= 5:
            listed.add((fields[3], fields[4]))
    return tuple(field for field in OPTIONAL_SURFACE_FIELDS if field in listed)


def optional_surface_selectors(published) -> tuple[str, ...]:
    """``VAR:LEVEL`` selectors for the optional records an index lists."""

    return tuple(f"{variable}:{level}" for variable, level in published)


def field_spellings(role: str) -> tuple[str, ...]:
    """Every index short name accepted for canonical field ``role``."""

    return FIELD_ALIASES.get(role, (role,))


class IndexInventoryError(ValueError):
    """The published index does not carry the expected inventory.

    Raised only for inventory disagreements -- a missing hybrid level, a
    surface field that is absent or duplicated, a record count that has
    moved.  It is distinct from a transport error so ``woof fetch`` can
    tell "this host publishes something we do not recognise" (try the
    next host) from "the network failed" (retry this one).
    """

#: One published wrfnat atmosphere subset carries exactly the 11 hybrid
#: fields on all 50 levels plus the 11 surface/near-surface records (561).
#: Single source for the completeness bar: the index selection below and
#: ``woof.fetch``'s existing-file re-verification both use these counts.
ATMOSPHERE_RECORD_COUNT = 50 * len(HYBRID_FIELDS) + len(SURFACE_FIELDS)


def atmosphere_hybrid_fields(*, analyzed_aerosol: bool = False,
                             extras=()) -> tuple[str, ...]:
    """The hybrid fields a preparation reads, the aerosol pair when asked.

    ``extras`` are the native extra records a caller requested
    (woof.ingest.native_extras).  Two of them, PMTF and PMTC, are the
    analyzed aerosol pair's own records, and the native bridge reads that
    pair whole or refuses the file (OPTIONAL_HYBRID_SPECS), so asking for
    either one selects the whole pair, each record once.
    """

    from woof.ingest.native_extras import requested_extra_fields
    extras = requested_extra_fields(extras)
    pair = analyzed_aerosol or any(
        name in ANALYZED_AEROSOL_FIELDS for name in extras)
    fields = HYBRID_FIELDS + (ANALYZED_AEROSOL_FIELDS if pair else ())
    return fields + tuple(name for name in extras
                          if name not in fields and name not in SURFACE_EXTRA_FIELDS)


def atmosphere_surface_fields(extras=()) -> tuple[tuple[str, str], ...]:
    """The single-level records of the subset: the fixed set plus the requested single-level extras."""

    from woof.ingest.native_extras import requested_extra_fields
    extras = requested_extra_fields(extras)
    return SURFACE_FIELDS + tuple(SURFACE_EXTRA_FIELDS[name] for name in extras
                                  if name in SURFACE_EXTRA_FIELDS)


def atmosphere_record_count(*, analyzed_aerosol: bool = False,
                            extras=()) -> int:
    """The atmosphere subset's record count for that preparation."""

    return (50 * len(atmosphere_hybrid_fields(
                analyzed_aerosol=analyzed_aerosol, extras=extras))
            + len(atmosphere_surface_fields(extras)))
#: One published soil subset carries TSOIL + SOILW at the nine RUC level
#: depths of the wrfprs file (18).
SOIL_RECORD_COUNT = 2 * len(SOIL_DEPTHS)

#: HRRR publishes an accumulated twin of several surface fields --
#: ``WEASD:surface:0-1 hr acc fcst`` sits beside the instantaneous
#: ``WEASD:surface`` -- and the native bridge selects only the
#: instantaneous one.  Both the index selection below and the Rust
#: fetch backbone's exact selectors filter on this substring.
ACCUMULATION_EXCLUSION = "acc fcst"

#: Soil level strings exactly as the ``.idx`` spells them.
SOIL_LEVELS = tuple(f"{depth}-{depth} m below ground" for depth in SOIL_DEPTHS)


def atmosphere_selectors(extras=(), *, analyzed_aerosol: bool = False) -> tuple[str, ...]:
    """The exact ``VAR:LEVEL`` selectors of the atmosphere subset.

    Same tables the index selection above walks, spelled as the
    ``rw_fetch`` backbone's ``--var-pattern`` wants them, so the Python
    and Rust transports cannot drift apart on what a native HRRR subset
    contains.
    """

    return tuple(
        [f"{'|'.join(field_spellings(name))}:{level} hybrid level"
         for name in atmosphere_hybrid_fields(
             analyzed_aerosol=analyzed_aerosol, extras=extras)
         for level in range(1, 51)]
        + [f"{'|'.join(field_spellings(variable))}:{level}"
           for variable, level in atmosphere_surface_fields(extras)])


def soil_selectors() -> tuple[str, ...]:
    """The exact ``VAR:LEVEL`` selectors of the soil subset."""

    return tuple(f"{'|'.join(field_spellings(variable))}:{level}"
                 for variable in ("TSOIL", "SOILW")
                 for level in SOIL_LEVELS)


@dataclass(frozen=True)
class IndexRow:
    sequence: int
    offset: int
    variable: str
    level: str
    raw: str


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int
    first_row: str
    last_row: str

    @property
    def size(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ProductRequest:
    url: str
    index_url: str
    index_path: Path
    destination: Path
    kind: str
    #: The atmosphere product also selects the analyzed aerosol pair.
    analyzed_aerosol: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cycle(raw: str) -> datetime:
    value = datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S")
    if value.minute or value.second:
        raise ValueError("HRRR cycle must be an exact UTC hour")
    return value


def _hours(raw: str, *, cycle: datetime | None = None) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("forecast hours must be comma-separated integers") from error
    return validate_hrrr_source_forecast_hours(
        values, cycle=cycle, window_flag="--forecast-hours")


def _request_bytes(url: str) -> tuple[bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": "rw-wps-hrrr-range/1"})
    with paced_urlopen(request, timeout=120) as response:
        return response.read(), {key.lower(): value for key, value in response.headers.items()}


def _head(url: str) -> dict[str, str]:
    request = Request(url, method="HEAD", headers={"User-Agent": "rw-wps-hrrr-range/1"})
    with paced_urlopen(request, timeout=120) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
    if "content-length" not in headers or int(headers["content-length"]) <= 0:
        raise ValueError(f"HRRR source omitted a positive Content-Length: {url}")
    if headers.get("accept-ranges", "").lower() != "bytes":
        raise ValueError(f"HRRR source does not advertise byte ranges: {url}")
    return headers


def _publish_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"existing authority differs from NOAA response: {path}")
        return
    temporary = path.with_suffix(
        f"{path.suffix}.{os.getpid()}-{time.time_ns()}.part")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_index(payload: bytes, object_bytes: int) -> tuple[IndexRow, ...]:
    rows = []
    for line_number, raw in enumerate(payload.decode("ascii").splitlines(), 1):
        fields = raw.split(":", 5)
        if len(fields) != 6:
            raise ValueError(f"malformed NOAA index line {line_number}")
        sequence, offset = int(fields[0]), int(fields[1])
        if sequence != line_number or offset < 0:
            raise ValueError(f"non-canonical NOAA index line {line_number}")
        rows.append(IndexRow(sequence, offset, fields[3], fields[4], raw))
    if not rows or rows[0].offset != 0:
        raise ValueError("NOAA index must begin with record 1 at byte zero")
    if any(right.offset <= left.offset for left, right in zip(rows, rows[1:])):
        raise ValueError("NOAA index offsets are not strictly increasing")
    if rows[-1].offset >= object_bytes:
        raise ValueError("NOAA index exceeds its source object")
    return tuple(rows)


def _atmosphere_selection(
    rows: tuple[IndexRow, ...],
    *, expected_count: int | None = -1, analyzed_aerosol: bool = False,
    extras=(),
) -> tuple[int, ...]:
    """Indices of the atmosphere records, refusing an inexact inventory.

    ``expected_count=None`` keeps the structural checks -- every hybrid
    field on every one of the fifty levels, every surface field exactly
    once -- but drops the total-count clause, so a caller that has
    explicitly accepted an upstream inventory change can proceed and
    then record the live count.  The structural checks are never
    optional: a missing level is incompleteness, not a change.
    """

    hybrid = atmosphere_hybrid_fields(analyzed_aerosol=analyzed_aerosol,
                                      extras=extras)
    if expected_count == -1:
        expected_count = atmosphere_record_count(
            analyzed_aerosol=analyzed_aerosol, extras=extras)
    elif expected_count == ATMOSPHERE_RECORD_COUNT:
        # A caller holding the plain census asks for it plus the records
        # its extras add (the AQ request form).
        expected_count += 50 * (len(hybrid) - len(atmosphere_hybrid_fields(
            analyzed_aerosol=analyzed_aerosol)))
        expected_count += len(atmosphere_surface_fields(extras)) - len(SURFACE_FIELDS)
    selected = []
    duplicates = []
    levels = {name: set() for name in hybrid}
    #: index short name -> canonical role, expanded through FIELD_ALIASES
    roles = {spelling: role for role in hybrid
             for spelling in field_spellings(role)}
    surfaces = {key: 0 for key in atmosphere_surface_fields(extras)}
    optional = 0
    for index, row in enumerate(rows):
        match = HYBRID_LEVEL.fullmatch(row.level)
        role = roles.get(row.variable)
        if role is not None and match is not None:
            level = int(match.group(1))
            if level in levels[role]:
                duplicates.append((role, level))
            levels[role].add(level)
            selected.append(index)
            continue
        key = (row.variable, row.level)
        if key in surfaces and "acc fcst" not in row.raw:
            surfaces[key] += 1
            selected.append(index)
        elif key in OPTIONAL_SURFACE_FIELDS and "acc fcst" not in row.raw:
            # What the bridge reads from a whole file when it is there;
            # not part of the count bar, which still guards exactly the
            # required inventory (see OPTIONAL_SURFACE_FIELDS).
            optional += 1
            selected.append(index)
    if expected_count is not None:
        expected_count += optional
    expected_levels = set(range(1, 51))
    bad_levels = {name: sorted(values) for name, values in levels.items()
                  if values != expected_levels}
    bad_surfaces = {str(key): count for key, count in surfaces.items() if count != 1}
    drifted = (expected_count is not None
               and len(selected) != expected_count)
    if bad_levels or bad_surfaces or drifted or duplicates:
        raise IndexInventoryError(
            "HRRR atmosphere index lacks the exact native bridge inventory: "
            f"levels={bad_levels}, surfaces={bad_surfaces}, duplicates={duplicates} (prevents duplicate source levels), "
            f"count={len(selected)} (expected {expected_count})"
            + ("; if the provider has announced this inventory change, "
               "re-run woof fetch with --accept-inventory-change"
               if drifted and not (bad_levels or bad_surfaces) else ""))
    return tuple(selected)


def _soil_selection(
    rows: tuple[IndexRow, ...],
    *, expected_count: int | None = SOIL_RECORD_COUNT,
) -> tuple[int, ...]:
    """Indices of the soil records; see :func:`_atmosphere_selection`."""

    expected = {
        (variable, f"{depth}-{depth} m below ground")
        for variable in ("TSOIL", "SOILW") for depth in SOIL_DEPTHS
    }
    counts = {key: 0 for key in expected}
    selected = []
    for index, row in enumerate(rows):
        key = (row.variable, row.level)
        if key in counts:
            counts[key] += 1
            selected.append(index)
    bad = {str(key): count for key, count in counts.items() if count != 1}
    drifted = expected_count is not None and len(selected) != expected_count
    if bad or drifted:
        raise IndexInventoryError(
            "HRRR pressure-file index lacks the exact soil inventory: "
            f"records={bad}, count={len(selected)} "
            f"(expected {expected_count})"
            + ("; if the provider has announced this inventory change, "
               "re-run woof fetch with --accept-inventory-change"
               if drifted and not bad else ""))
    return tuple(selected)


def _coalesce(
    rows: tuple[IndexRow, ...], selected: tuple[int, ...], object_bytes: int,
) -> tuple[ByteRange, ...]:
    groups: list[tuple[int, int]] = []
    first = previous = selected[0]
    for index in selected[1:]:
        if index != previous + 1:
            groups.append((first, previous))
            first = index
        previous = index
    groups.append((first, previous))
    result = []
    for first, last in groups:
        end = rows[last + 1].offset - 1 if last + 1 < len(rows) else object_bytes - 1
        result.append(ByteRange(
            rows[first].offset, end, rows[first].raw, rows[last].raw))
    return tuple(result)


def _range_retry_delay(error: BaseException, attempt: int) -> float | None:
    """Seconds before the next attempt, or None when asking again cannot help.

    The network faults are the tree's shared classification
    (:func:`woof.fetch_endpoints.retry_delay`), so a 404 or another
    status that will not change is not asked again.  A ``ValueError`` is
    this transport's own: a range answer that did not verify (not 206,
    the wrong Content-Range, short), which a fresh request can repair.
    """

    from woof import fetch_endpoints

    if isinstance(error, ValueError):
        return 2.0 ** attempt
    return fetch_endpoints.retry_delay(
        error, attempt, wait_limit_s=fetch_endpoints.TRANSIENT_WAIT_LIMIT_S)


def _download_range(
    url: str, byte_range: ByteRange, path: Path, retries: int,
) -> None:
    """One byte range, retried on the tree's shared discipline.

    ``retries`` is the caller's attempt budget, held to the shared
    :data:`woof.fetch_endpoints.TRANSIENT_ATTEMPTS`; each wait is
    :func:`_range_retry_delay`'s.  Once another file has failed the
    fetch request this range belongs to, it stops at its next block
    instead of finishing (see :func:`woof.fetch_pool.raise_if_stopped`;
    the caller carries the pool job onto this thread with
    :func:`woof.fetch_pool.working_for`).
    """

    from woof import fetch_endpoints
    from woof.fetch_pool import (TransferCancelled, raise_if_stopped,
                                  sleep_unless_stopped)

    attempts = max(1, min(int(retries), fetch_endpoints.TRANSIENT_ATTEMPTS))
    expected_content_range = f"bytes {byte_range.start}-{byte_range.end}/"
    for attempt in range(1, attempts + 1):
        try:
            raise_if_stopped()
            request = Request(url, headers={
                "User-Agent": "rw-wps-hrrr-range/1",
                "Range": f"bytes={byte_range.start}-{byte_range.end}",
                "Accept-Encoding": "identity",
            })
            with paced_urlopen(request, timeout=300) as response, \
                    path.open("wb") as output:
                if response.status != 206:
                    raise ValueError(f"range request returned HTTP {response.status}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(expected_content_range):
                    raise ValueError(f"unexpected Content-Range {content_range!r}")
                copied = 0
                while block := response.read(1024 * 1024):
                    output.write(block)
                    copied += len(block)
                    raise_if_stopped()
            if copied != byte_range.size:
                raise ValueError(f"short range response: {copied} != {byte_range.size}")
            return
        except Exception as error:
            path.unlink(missing_ok=True)
            if attempt == attempts or isinstance(error, TransferCancelled):
                raise
            delay = _range_retry_delay(error, attempt)
            if delay is None:
                raise
            sleep_unless_stopped(delay, sleep=time.sleep)


def _download_subset(
    *, url: str, index_url: str, index_path: Path, destination: Path,
    kind: str, workers: int, retries: int,
    expected_count: int | None = -1, analyzed_aerosol: bool = False,
    extras=(),
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing subset: {destination}")
    started = time.perf_counter()
    headers = _head(url)
    object_bytes = int(headers["content-length"])
    index_payload, index_headers = _request_bytes(index_url)
    _publish_exact(index_path, index_payload)
    rows = _parse_index(index_payload, object_bytes)
    if expected_count == -1:
        expected_count = (
            atmosphere_record_count(analyzed_aerosol=analyzed_aerosol)
            if kind == "atmosphere" else SOIL_RECORD_COUNT)
    selected = (
        _atmosphere_selection(
            rows, expected_count=expected_count,
            **({"analyzed_aerosol": True} if analyzed_aerosol else {}),
            extras=extras)
        if kind == "atmosphere"
        else _soil_selection(rows, expected_count=expected_count)
    )
    ranges = _coalesce(rows, selected, object_bytes)
    # Taken HERE, on the fetch pool's thread: the ranges below run on
    # threads of their own, which do not carry the pool job.
    from woof.fetch_pool import current_job, working_for
    job = current_job()

    def fetch_range(pair):
        with working_for(job):
            _download_range(url, pair[0], pair[1], retries)

    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.ranges-", dir=destination.parent))
    try:
        chunks = tuple(staging / f"{index:04d}.part" for index in range(len(ranges)))
        # A NOMADS transfer is serialized by the rate governor anyway --
        # eight workers would queue on the same node-wide sentinel for
        # the same 2.5 s gap -- so the pool states that plainly instead
        # of letting the burst shape depend on thread scheduling.  S3
        # keeps the pool the caller asked for.
        with ThreadPoolExecutor(
                max_workers=governed_workers(url, workers)) as pool:
            list(pool.map(
                fetch_range,
                zip(ranges, chunks),
            ))
        # Assembly happens INSIDE the unique staging directory, never at
        # a canonical `<destination>.part`.  A killed run used to leave
        # that fixed name behind, and the next run opened it with `xb`,
        # got FileExistsError, and could neither resume, overwrite, nor
        # quarantine it -- an unrecoverable state reachable by one kill.
        # Staging is unique per attempt and removed in `finally`, so the
        # only thing that ever appears at a canonical name is a complete,
        # framed, fsynced file arriving by atomic rename.
        partial = staging / "assembled.part"
        with partial.open("xb") as output:
            for chunk in chunks:
                with chunk.open("rb") as stream:
                    shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        expected_bytes = sum(item.size for item in ranges)
        if partial.stat().st_size != expected_bytes:
            raise ValueError("assembled HRRR subset byte count differs from ranges")
        with partial.open("rb") as stream:
            signature = stream.read(4)
            stream.seek(-4, os.SEEK_END)
            terminator = stream.read(4)
        if signature != b"GRIB" or terminator != b"7777":
            raise ValueError("assembled HRRR subset is not a complete GRIB2 stream")
        os.replace(partial, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "kind": kind,
        "source_url": url,
        "source_bytes": object_bytes,
        "source_etag": headers.get("etag"),
        "source_last_modified": headers.get("last-modified"),
        "index_url": index_url,
        "index_path": index_path.name,
        "index_bytes": len(index_payload),
        "index_sha256": hashlib.sha256(index_payload).hexdigest(),
        "index_etag": index_headers.get("etag"),
        "selected_record_count": len(selected),
        "range_count": len(ranges),
        "subset_path": destination.name,
        "subset_bytes": destination.stat().st_size,
        "subset_sha256": _sha256(destination),
        "wall_seconds": time.perf_counter() - started,
        "ranges": [
            {
                "start": item.start,
                "end": item.end,
                "bytes": item.size,
                "first_index_row": item.first_row,
                "last_index_row": item.last_row,
            }
            for item in ranges
        ],
    }


def _download_product(
    request: ProductRequest, *, workers: int, retries: int,
    expected_count: int | None = -1, extras=(),
) -> dict[str, object]:
    return _download_subset(
        url=request.url,
        index_url=request.index_url,
        index_path=request.index_path,
        destination=request.destination,
        kind=request.kind,
        workers=workers,
        retries=retries,
        expected_count=expected_count,
        analyzed_aerosol=request.analyzed_aerosol,
        extras=extras,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", default="", help="comma-separated optional hybrid fields")
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--forecast-hours", default="0,1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--file-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument(
        "--analyzed-aerosol", action="store_true",
        help="also select the analyzed aerosol number pair (PMTF/PMTC = "
             "QNWFA/QNIFA) the preparation reads for analyzed mp=28 aerosol")
    args = parser.parse_args(argv)
    cycle = _cycle(args.cycle)
    hours = _hours(args.forecast_hours, cycle=cycle)
    from woof.fetch_endpoints import TRANSIENT_ATTEMPTS

    # A range is asked at most TRANSIENT_ATTEMPTS times (the tree's one
    # budget), so a larger --retries is refused rather than quietly held
    # to it: 6..10 used to be accepted and are no longer honoured.
    if (args.workers not in range(1, 33)
            or args.retries not in range(1, TRANSIENT_ATTEMPTS + 1)):
        raise ValueError(
            "workers must be 1..32 and retries must be "
            f"1..{TRANSIENT_ATTEMPTS} (the shared attempt budget, "
            "woof.fetch_endpoints.TRANSIENT_ATTEMPTS)")
    if args.file_workers not in range(1, 17):
        raise ValueError("file-workers must be 1..16")
    if args.workers * args.file_workers > 64:
        raise ValueError("workers * file-workers may not exceed 64")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "download-receipt.json"
    manifest_path = output / "SHA256SUMS"
    if receipt_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to replace an existing HRRR download receipt")
    prefix = f"{args.base_url.rstrip('/')}/hrrr.{cycle:%Y%m%d}/conus"
    requests = []
    for hour in hours:
        forecast = f"{hour:02d}"
        atmosphere_name = f"hrrr.t{cycle:%H}z.wrfnatf{forecast}.grib2"
        pressure_name = f"hrrr.t{cycle:%H}z.wrfprsf{forecast}.grib2"
        soil_name = f"hrrr.t{cycle:%H}z.soilf{forecast}.grib2"
        requests.append(ProductRequest(
            url=f"{prefix}/{atmosphere_name}",
            index_url=f"{prefix}/{atmosphere_name}.idx",
            index_path=output / f"{atmosphere_name}.idx",
            destination=output / atmosphere_name,
            kind="atmosphere",
            analyzed_aerosol=args.analyzed_aerosol,
        ))
        requests.append(ProductRequest(
            url=f"{prefix}/{pressure_name}",
            index_url=f"{prefix}/{pressure_name}.idx",
            index_path=output / f"{pressure_name}.idx",
            destination=output / soil_name,
            kind="soil",
        ))
    download_options = {}
    if args.extras:
        from woof.ingest.native_extras import requested_extra_fields
        download_options["extras"] = requested_extra_fields(args.extras.split(","))
    download = partial(
        _download_product, workers=args.workers, retries=args.retries,
        **download_options)
    with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        # Executor.map preserves request order, so the receipt is deterministic
        # even when products complete out of order.
        products = list(pool.map(download, requests))
    manifest_records = sorted(
        (item["subset_path"], item["subset_sha256"]) for item in products)
    manifest_path.write_text("".join(
        f"{digest}  {name}\n" for name, digest in manifest_records),
        encoding="ascii",
    )
    receipt = {
        "schema": "rw-wps.hrrr-native-range-download.v1",
        "status": "PASS_TRANSPORT_SUBSET_RUST_SCIENCE_VALIDATION_REQUIRED",
        "cycle": cycle.isoformat() + "Z",
        "source_forecast_hours": list(hours),
        # Kept for compatibility with existing download consumers.
        "forecast_hours": list(hours),
        "workers": args.workers,
        "file_workers": args.file_workers,
        "products": products,
        "payload_bytes": sum(item["subset_bytes"] for item in products),
        "source_bytes": sum(item["source_bytes"] for item in products),
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "payload_bytes": receipt["payload_bytes"],
        "source_bytes": receipt["source_bytes"],
        "manifest": str(manifest_path),
        "manifest_sha256": receipt["manifest_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
