"""Copy the records a table row selects from one published GRIB object.

A publisher that writes an index beside each object (ECMWF's JSON-lines
``.index``, NCEP's ``.idx``) lets a fetch move only the records the
packaged mapping binds.  The selection is table data (the row's
``record_index`` / a donor's ``records``), never code: this module reads
the index, turns the selection into byte ranges, moves them in parallel
and verifies every GRIB message before publishing the file.

The bytes are copied unchanged.  A record the mapping binds is either in
the selection or the decode refuses by name ("lacks required fields"), so
a selection that is too narrow fails loudly; it never changes a value.

Measured basis (2026-10-29, the 2026-10-28 12Z indexes): the IFS 0.1-degree
pair is 451 MB per lead and the mapped records are 165 MB of it; the
0.25-degree object is 97 MB and the mapped records 37 MB; the GFS f000 a
sea-ice donor reads is 513 MB and its ICEC record about 0.1 MB.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import http.client
import json
from pathlib import Path
import re
from http.client import IncompleteRead
import threading
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, getproxies

from woof import fetch_pool
from woof.nomads_governor import is_nomads_url, paced_urlopen

#: Index formats a selection may name.
INDEX_FORMATS = ("ecmwf-jsonl", "ncep-idx")

#: Largest single range request.  A coalesced span longer than this is cut
#: so the streams of one object stay balanced (one 63 MB span would
#: otherwise finish long after the rest).
MAX_PIECE = 32 * 1024 * 1024

#: Concurrent range streams per object.
STREAMS = 8

#: Tries per range before a transient fault (a reset or dropped connection,
#: a timeout, a body cut short) fails the whole object.
PIECE_ATTEMPTS = 3


#: Answers that move the object elsewhere; urllib follows them.
_REDIRECTS = frozenset({301, 302, 303, 307, 308})


def _reusable(url: str, opener) -> bool:
    """Whether the ranges of ``url`` may ride persistent connections.

    Not for NOMADS (the governor paces and logs each request), not through
    a configured proxy (urllib honours it, http.client does not), and not
    for a caller's own opener (a test's fake host, an offline replay).
    """

    scheme = urlsplit(url).scheme
    return (opener is None and scheme in ("http", "https") and not is_nomads_url(url)
            and scheme not in getproxies())


def _transient(error: BaseException) -> bool:
    """A fault a fresh request for the same range can repair."""

    if isinstance(error, HTTPError):
        return False
    if isinstance(error, URLError):
        return isinstance(error.reason, (OSError, TimeoutError))
    return isinstance(error, (ConnectionError, TimeoutError, IncompleteRead))

_MAX_INDEX_BYTES = 16 * 1024 * 1024

#: Index answers that mean "this host keeps no index for the object": the
#: object itself may still be there, so the caller moves it whole rather
#: than failing a healthy object.  A throttle or server fault (429, 5xx)
#: is not one of them: it propagates, so the ladder's retry and throttle
#: rows ask again or ask the next host.
_ABSENT_INDEX = frozenset({403, 404, 410})


class IndexAbsent(LookupError):
    """The host answered that it keeps no index beside this object."""


class SelectionAbsent(IndexAbsent, ValueError):
    """The index lists no record for one of the selection's pairs.

    Named breakage: an upstream index that renames a variable, or an older
    era that lacks one, would otherwise fail a healthy object on every
    endpoint, retried, although a renamed record is still in the object
    under its GRIB identity, which is what the decode reads.  As an
    :class:`IndexAbsent` the table-route transfer moves that object whole
    and says so; the decode then reads the field or refuses it by name, as
    it would have before.  It stays a ``ValueError`` for callers that treat
    a selection that cannot be served as a payload that did not verify.
    """


_CHUNK = 1024 * 1024
_USER_AGENT = "gpuwm-fetch/2.5 (+https://github.com/arwenweather)"


def validate_declaration(declaration: Mapping[str, object], label: str) -> dict:
    """A row's selection, checked at table load.

    ``{"format": "ecmwf-jsonl", "select": [{"levtype": "pl", "param":
    ["t", "u"]}, ...], "why": ...}`` keeps a record when every key of
    ANY clause matches (a list value matches any of its members);
    ``{"format": "ncep-idx", "select": [["ICEC", "surface"]], ...}``
    keeps the records whose VAR and LEVEL match a pair of full-match
    regexes.  Named breakage: an empty or misspelt selection would plan a
    fetch that moves nothing (or everything) and only the decode, minutes
    later, would say so.

    An ecmwf-jsonl selection may also declare ``"prefer": {"param": ["gh",
    "z"]}``: of the kept records that differ only in that key, the one
    whose value comes first in the list is kept and the rest are dropped.
    That is the mapping's own rule (a record answers the first selector it
    matches, so a later one stands in only where no earlier one was
    published), so the selection keeps exactly the record the decode reads
    whichever of them a cycle publishes.
    """
    if not isinstance(declaration, Mapping):
        raise ValueError(f"{label}: a record selection is an object")
    unknown = set(declaration) - {"format", "select", "prefer", "why"}
    if unknown:
        raise ValueError(f"{label}: a record selection declares {sorted(unknown)}; "
                         "it reads format, select, prefer and why")
    form = declaration.get("format")
    if form not in INDEX_FORMATS:
        raise ValueError(f"{label}: record selection format {form!r} is not one of "
                         f"{list(INDEX_FORMATS)}")
    select = declaration.get("select")
    if not isinstance(select, Sequence) or isinstance(select, (str, bytes)) or not select:
        raise ValueError(f"{label}: a record selection needs a non-empty select list")
    if form == "ecmwf-jsonl":
        for clause in select:
            if not isinstance(clause, Mapping) or not clause:
                raise ValueError(f"{label}: each ecmwf-jsonl select clause is a "
                                 "non-empty key-to-value object")
            for key, value in clause.items():
                values = value if isinstance(value, (list, tuple)) else [value]
                if not values or not all(isinstance(item, str) and item for item in values):
                    raise ValueError(f"{label}: select key {key!r} needs a string or a "
                                     "non-empty list of strings")
    else:
        for pair in select:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not all(isinstance(item, str) and item for item in pair)):
                raise ValueError(f"{label}: each ncep-idx select entry is [VAR, LEVEL] "
                                 "regexes")
            for item in pair:
                re.compile(item)
    prefer = declaration.get("prefer")
    if prefer is not None:
        # Named breakage: an NCEP .idx line has no keyed identity to group
        # by, and an empty or one-value order would drop nothing.
        if form != "ecmwf-jsonl":
            raise ValueError(f"{label}: prefer orders ecmwf-jsonl records only")
        if not isinstance(prefer, Mapping) or len(prefer) != 1:
            raise ValueError(f"{label}: prefer names one index key and its value order")
        ((key, order),) = prefer.items()
        if (not isinstance(order, (list, tuple)) or len(order) < 2
                or not all(isinstance(item, str) and item for item in order)
                or len(set(order)) != len(order)):
            raise ValueError(f"{label}: prefer {key!r} needs two or more distinct values")
    result = {"format": form, "select": json.loads(json.dumps(select)),
              "why": str(declaration.get("why", ""))}
    if prefer is not None:
        result["prefer"] = json.loads(json.dumps(prefer))
    return result


def _clause_matches(record: Mapping[str, object], clause: Mapping[str, object]) -> bool:
    for key, value in clause.items():
        values = value if isinstance(value, (list, tuple)) else (value,)
        if str(record.get(key, "")) not in values:
            return False
    return True


def select_ecmwf(index: bytes, select, prefer: Mapping[str, Sequence[str]] | None = None
                 ) -> tuple[tuple[int, int], ...]:
    """``(offset, length)`` of every record a clause keeps, in file order.

    ``prefer`` (``{key: [first, second, ...]}``) keeps, of the kept records
    that agree on every other index key, only the one whose ``key`` comes
    first in the list (see :func:`validate_declaration`).
    """
    entries = []
    rows = []
    for line, text in enumerate(index.decode("utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
            offset, length = row["_offset"], row["_length"]
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"index line {line} is not a complete range record") from error
        if type(offset) is not int or type(length) is not int or offset < 0 or length < 20:
            raise ValueError(f"index line {line} has an invalid offset or GRIB length")
        entries.append([offset, length, any(_clause_matches(row, clause) for clause in select)])
        rows.append(row)
    for key, order in (prefer or {}).items():
        rank = {str(value): position for position, value in enumerate(order)}
        best: dict[str, int] = {}
        ranked = []
        for entry, row in zip(entries, rows):
            value = str(row.get(key, ""))
            if not entry[2] or value not in rank:
                continue
            identity = json.dumps({name: item for name, item in row.items()
                                   if name not in (key, "_offset", "_length")},
                                  sort_keys=True)
            best[identity] = min(best.get(identity, rank[value]), rank[value])
            ranked.append((entry, identity, rank[value]))
        for entry, identity, position in ranked:
            entry[2] = position == best[identity]
    entries = [tuple(entry) for entry in entries]
    entries.sort()
    previous_end = 0
    for offset, length, _keep in entries:
        if offset < previous_end:
            raise ValueError("index ranges overlap, which would duplicate or cut records")
        previous_end = offset + length
    return tuple((offset, length) for offset, length, keep in entries if keep)


def select_ncep(index: bytes, select, object_size: int | None
                ) -> tuple[tuple[tuple[int, int], ...], bool]:
    """The records of an NCEP ``.idx`` whose VAR and LEVEL match a pair.

    Returns ``(ranges, needs_size)``: a record ends where the next distinct
    offset starts; the last record ends at the object size, so when it is
    selected and ``object_size`` is None the caller HEADs and asks again.
    Sub-messages (``12.1``, ``12.2``) share their message's offset and are
    one range.  Every pair must match at least one record.
    """
    rules = [(re.compile(variable), re.compile(level)) for variable, level in select]
    rows = []
    for line, text in enumerate(index.decode("utf-8").splitlines(), 1):
        if not text.strip():
            continue
        fields = text.split(":")
        if len(fields) < 5:
            raise ValueError(f"index line {line} has {len(fields)} fields; an NCEP "
                             "inventory line has at least five")
        try:
            rows.append((int(fields[1]), fields[3], fields[4]))
        except ValueError as error:
            raise ValueError(f"index line {line} has no byte offset") from error
    starts = sorted({offset for offset, _variable, _level in rows})
    following = {start: (starts[k + 1] if k + 1 < len(starts) else None)
                 for k, start in enumerate(starts)}
    matched = [False] * len(rules)
    chosen: set[int] = set()
    for offset, variable, level in rows:
        for k, (want_variable, want_level) in enumerate(rules):
            if want_variable.fullmatch(variable) and want_level.fullmatch(level):
                matched[k] = True
                chosen.add(offset)
    missing = [list(select[k]) for k, hit in enumerate(matched) if not hit]
    if missing:
        raise SelectionAbsent(f"the index has no record matching {missing}")
    ranges = []
    for offset in sorted(chosen):
        end = following[offset]
        if end is None:
            if object_size is None:
                return (), True
            end = object_size
        if end - offset < 20:
            raise ValueError(f"index record at byte {offset} is shorter than a GRIB message")
        ranges.append((offset, end - offset))
    return tuple(ranges), False


def count_ncep(index: bytes, select) -> int:
    """How many GRIB messages an NCEP ``.idx`` selection keeps.

    The census :func:`download_records` holds its file to, read without
    moving the object: one message per distinct offset whose VAR and LEVEL
    match a pair (sub-messages share their message's offset).
    """
    ranges, needs_size = select_ncep(index, select, 1 << 62)
    del needs_size
    return len(ranges)


def _pieces(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Coalesce adjacent records, cut long spans: ``(source, length, dest)``."""
    spans: list[list[int]] = []
    for offset, length in ranges:
        if spans and spans[-1][0] + spans[-1][1] == offset:
            spans[-1][1] += length
        else:
            spans.append([offset, length])
    pieces = []
    written = 0
    for offset, length in spans:
        done = 0
        while done < length:
            size = min(MAX_PIECE, length - done)
            pieces.append((offset + done, size, written + done))
            done += size
        written += length
    return pieces


#: The smallest piece :func:`_spread` cuts a span to.
MIN_SPREAD_PIECE = 4 * 1024 * 1024


def _spread(pieces: list[tuple[int, int, int]], streams: int) -> list[tuple[int, int, int]]:
    """Cut pieces finer so a selection fewer pieces long uses every stream.

    Measured cost this removes: hrrr-native's soil donor selection is one
    27 MB run of contiguous records, which the MAX_PIECE cut left as one
    request on one connection: its 29 MB took 12-27 s from the home nodes
    (2026-10-29), where the whole 381 MB object took 27 s on 12 streams.
    A piece is never cut below MIN_SPREAD_PIECE, so a small selection stays
    one request per span.
    """

    if streams <= 1 or len(pieces) >= streams:
        return pieces
    total = sum(length for _source, length, _dest in pieces)
    target = max(MIN_SPREAD_PIECE, -(-total // streams))
    spread = []
    for source, length, dest in pieces:
        done = 0
        while done < length:
            size = min(target, length - done)
            spread.append((source + done, size, dest + done))
            done += size
    return spread


def grib_messages(path: Path) -> int:
    """Walk the GRIB envelope of ``path``: every message whole, nothing between."""
    count = 0
    with Path(path).open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        offset = 0
        while offset < size:
            handle.seek(offset)
            head = handle.read(16)
            if len(head) < 8 or head[:4] != b"GRIB":
                raise ValueError(f"no GRIB message starts at byte {offset}")
            edition = head[7]
            if edition == 2:
                length = int.from_bytes(head[8:16], "big")
            elif edition == 1:
                length = int.from_bytes(head[4:7], "big")
            else:
                raise ValueError(f"GRIB edition {edition} at byte {offset}")
            if length < 20 or offset + length > size:
                raise ValueError(f"the GRIB message at byte {offset} runs past the file")
            handle.seek(offset + length - 4)
            if handle.read(4) != b"7777":
                raise ValueError(f"the GRIB message at byte {offset} has no 7777 terminator")
            offset += length
            count += 1
    return count


def _object_size(url: str, *, opener, timeout: float) -> int:
    options = {"opener": opener} if opener is not None else {}
    request = Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    with paced_urlopen(request, timeout=timeout, **options) as response:
        return int(response.headers["Content-Length"])


def download_records(url: str, index_url: str, dest: Path, *,
                     declaration: Mapping[str, object], opener=None,
                     timeout: float = 300.0, streams: int = STREAMS) -> dict:
    """Range-fetch the selected records of ``url``, verify, publish ``dest``.

    Every reply must be a 206 for exactly the bytes asked, from one object
    version (the same total size, and the same ETag or Last-Modified when
    the host sends one); the file must walk as whole GRIB messages, one per
    selected record.  Anything else is a ``ValueError`` the ladder treats as
    a payload that did not verify.
    """
    from woof.filesystem_paths import io_path

    from urllib.error import HTTPError

    options = {"opener": opener} if opener is not None else {}
    request = Request(index_url, headers={"User-Agent": _USER_AGENT})
    try:
        with paced_urlopen(request, timeout=timeout, **options) as response:
            index = response.read(_MAX_INDEX_BYTES + 1)
    except HTTPError as error:
        if error.code in _ABSENT_INDEX:
            raise IndexAbsent(f"{index_url}: HTTP {error.code}") from None
        raise
    if len(index) > _MAX_INDEX_BYTES:
        raise ValueError("the index exceeds 16 MiB; refusing an unbounded response")
    form = declaration["format"]
    select = declaration["select"]
    object_size = None
    if form == "ecmwf-jsonl":
        ranges = select_ecmwf(index, select, declaration.get("prefer"))
    else:
        ranges, needs_size = select_ncep(index, select, None)
        if needs_size:
            object_size = _object_size(url, opener=opener, timeout=timeout)
            ranges, _ = select_ncep(index, select, object_size)
    if not ranges:
        raise ValueError(f"the index {index_url} has no record the selection keeps; "
                         "an empty file would decode to a missing field")
    total = sum(length for _offset, length in ranges)
    pieces = _pieces(ranges)
    if not is_nomads_url(url):
        pieces = _spread(pieces, streams)
    dest = io_path(Path(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with part.open("wb") as output:
        output.truncate(total)
    job = fetch_pool.current_job()
    failed = threading.Event()
    seen: dict[str, object] = {}
    lock = threading.Lock()
    width = 1 if is_nomads_url(url) else max(1, min(streams, len(pieces)))
    fetch_pool.record_streams(width)

    # One persistent connection per stream: an HRRR wrfprs selection is 127
    # ranges of about 1.5 MB, and a fresh connection per range (TLS setup and
    # TCP slow start each time) moved them at 7.5-9 MB/s on a home link
    # where the whole object moved at 23 MB/s; on persistent connections the
    # same 127 ranges moved at 23 MB/s (2026-10-29).  NOMADS keeps the
    # governor's paced path, and so does a configured proxy or a caller's
    # opener (a test's fake host, an offline replay).
    reuse = {"on": _reusable(url, opener)}
    parts = urlsplit(url)
    resource = parts.path + (f"?{parts.query}" if parts.query else "")
    connections = threading.local()
    opened: list[http.client.HTTPConnection] = []

    def _connection() -> http.client.HTTPConnection:
        connection = getattr(connections, "connection", None)
        if connection is None:
            kind = (http.client.HTTPSConnection if parts.scheme == "https"
                    else http.client.HTTPConnection)
            connection = kind(parts.hostname, parts.port, timeout=timeout)
            connections.connection = connection
            with lock:
                opened.append(connection)
        return connection

    def _drop() -> None:
        connection = getattr(connections, "connection", None)
        connections.connection = None
        if connection is not None:
            connection.close()

    @contextmanager
    def ranged(headers):
        if reuse["on"]:
            from woof.fetch_connections import lease

            with lease():
                connection = _connection()
                try:
                    connection.request("GET", resource, headers=headers)
                    body = connection.getresponse()
                    if body.status in _REDIRECTS:
                        # A host that moves the object: urllib follows it.
                        body.read()
                        reuse["on"] = False
                        _drop()
                    else:
                        try:
                            yield body
                        finally:
                            # Reuse only a connection whose answer was read to
                            # its end; anything else is closed.
                            if body.will_close or not body.isclosed():
                                _drop()
                        return
                except BaseException:
                    _drop()
                    raise
        with paced_urlopen(Request(url, headers=headers), timeout=timeout,
                           **options) as body:
            yield body

    def transfer_once(piece: tuple[int, int, int], last: bool = True) -> None:
        source, length, target = piece
        end = source + length - 1
        headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "identity",
                   "Range": f"bytes={source}-{end}"}
        try:
            with fetch_pool.working_for(job):
                fetch_pool.raise_if_stopped()
                with ranged(headers) as body:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         body.headers.get("Content-Range", ""))
                    if (getattr(body, "status", None) != 206 or match is None
                            or (int(match.group(1)), int(match.group(2))) != (source, end)
                            or body.headers.get("Content-Encoding", "identity") != "identity"):
                        raise ValueError(f"{dest.name}: the host ignored or changed the "
                                         f"requested byte range {source}-{end}")
                    version = (int(match.group(3)), body.headers.get("ETag"),
                               body.headers.get("Last-Modified"))
                    with lock:
                        first = seen.setdefault("version", version)
                    if (first[0] != version[0]
                            or (first[1] and version[1] and first[1] != version[1])
                            or (not first[1] and first[2] and version[2]
                                and first[2] != version[2])):
                        raise ValueError(f"{dest.name}: the object changed between range "
                                         "requests")
                    if object_size is not None and version[0] != object_size:
                        raise ValueError(f"{dest.name}: the object changed size after its index "
                                         "was read")
                    remaining = length
                    with part.open("r+b") as output:
                        output.seek(target)
                        while remaining:
                            fetch_pool.raise_if_stopped()
                            if failed.is_set():
                                return
                            chunk = body.read(min(_CHUNK, remaining))
                            if not chunk:
                                raise ValueError(f"{dest.name}: range {source}-{end} ended "
                                                 "early")
                            output.write(chunk)
                            remaining -= len(chunk)
                    if body.read(1):
                        raise ValueError(f"{dest.name}: range {source}-{end} ran past its "
                                         "length")
        except BaseException as error:
            if last or not _transient(error):
                failed.set()
            raise

    def transfer(piece: tuple[int, int, int]) -> None:
        # A range whose connection resets or times out is asked again
        # before it fails the object.  Measured cost: one reset among an
        # HRRR wrfprs selection's 127 ranges (2026-10-29) failed the
        # whole object on the archive, and the ladder's next rung (NOMADS,
        # paced) moved it whole: 412 MB instead of 186 MB.  Rewriting a
        # piece from its start is safe: it owns its bytes of the part file.
        for attempt in range(1, PIECE_ATTEMPTS):
            try:
                return transfer_once(piece, last=False)
            except BaseException as error:
                if failed.is_set() or not _transient(error):
                    raise
                with fetch_pool.working_for(job):
                    fetch_pool.sleep_unless_stopped(float(attempt), sleep=time.sleep)
        return transfer_once(piece)

    try:
        try:
            with ThreadPoolExecutor(max_workers=width,
                                    thread_name_prefix="gpuwm-records") as pool:
                list(pool.map(transfer, pieces))
        finally:
            for connection in opened:
                connection.close()
        messages = grib_messages(part)
        if messages != len(ranges):
            raise ValueError(f"{dest.name}: {messages} GRIB messages for {len(ranges)} "
                             "selected records")
        digest = hashlib.sha256()
        with part.open("rb") as payload:
            for chunk in iter(lambda: payload.read(_CHUNK), b""):
                digest.update(chunk)
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    version = seen.get("version") or (object_size, None, None)
    return {
        "name": dest.name, "bytes": total, "sha256": digest.hexdigest(), "url": url,
        "index_url": index_url, "index_sha256": hashlib.sha256(index).hexdigest(),
        "index_format": form, "selected_records": len(ranges), "requests": len(pieces),
        "object_bytes": version[0], "object_etag": version[1],
    }
