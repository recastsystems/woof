"""Copy one member's indexed GRIB ranges without decoding or changing them.

Field decoding and ensemble-octet verification remain in the Rust inventory.
The index is an acquisition hint only, never proof of member identity.

A row may also declare a record selection (``record_index``, the same
ECMWF JSON-lines clauses :mod:`woof.record_index` reads): the member's
records are then narrowed to the ones the packaged mapping binds.  Measured
basis (fix/ecmwf-ens-record-subset, 2026-10-29): an ECMWF ENS member is
177 records and 80-91 MB per lead, of which the mapping binds 85 records,
35.8-38.5 MB.  The ranges are moved in parallel: the 177 single-record
ranges of one member, asked one after another, ran at 2.4 MB/s from a host
that serves one whole object at 38.8 MB/s.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Mapping
from urllib.request import Request

from woof import fetch_pool
from woof.nomads_governor import is_nomads_url, paced_urlopen

#: Concurrent range streams per member object (the record-index transfer's
#: width, so both index transports load a host alike).
STREAMS = 8

_CHUNK = 1024 * 1024


def select_member_ranges(index: bytes, declaration: Mapping[str, object],
                         member_ordinal: int, records: Mapping[str, object] | None = None
                         ) -> tuple[tuple[int, int], ...]:
    """Return nonoverlapping (offset, length) records in publication order.

    ``records`` is a validated ecmwf-jsonl record selection: of the member's
    records, only those a clause keeps are returned.
    """
    required = {"member_key", "offset_key", "length_key"}
    if set(declaration) - (required | {"filters"}) or required - set(declaration):
        raise ValueError("Member index needs member_key, offset_key and length_key only, with optional filters")
    member_key, offset_key, length_key = (
        str(declaration[name]) for name in ("member_key", "offset_key", "length_key"))
    filters = declaration.get("filters", {})
    if not isinstance(filters, Mapping):
        raise ValueError("Member index filters must be a field-to-value mapping")
    clauses = None
    if records is not None:
        if records.get("format") != "ecmwf-jsonl":
            raise ValueError("A member's records are narrowed only by an ecmwf-jsonl selection")
        from woof.record_index import _clause_matches
        clauses = records["select"]
    entries = []
    members = 0
    for line, text in enumerate(index.decode("utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
            offset, length = row[offset_key], row[length_key]
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"Member index line {line} is not a complete range record") from error
        if (type(offset) is not int or type(length) is not int
                or offset < 0 or length < 20):
            raise ValueError(f"Member index line {line} has invalid offset or GRIB length")
        selected = (str(row.get(member_key, "")) == str(member_ordinal)
                    and all(str(row.get(key, "")) == str(value)
                            for key, value in filters.items()))
        members += selected
        if selected and clauses is not None:
            selected = any(_clause_matches(row, clause) for clause in clauses)
        entries.append((offset, length, selected))
    entries.sort()
    previous_end = 0
    for offset, length, _ in entries:
        if offset < previous_end:
            raise ValueError("Member index ranges overlap, which could duplicate fields or mix members")
        previous_end = offset + length
    selected = tuple((offset, length) for offset, length, keep in entries if keep)
    if not members:
        raise ValueError(f"Member index contains no records for member {member_ordinal}")
    if not selected:
        raise ValueError(f"Member index keeps none of member {member_ordinal}'s {members} "
                         "records under the row's record selection; an empty file would "
                         "decode to a missing field")
    return selected


def download_indexed_member(url: str, index_url: str, dest: Path, *,
                            declaration: Mapping[str, object], member_ordinal: int,
                            source: str, member: str, opener=None,
                            timeout: float = 300.0, forecast_hour: int | None = None,
                            records: Mapping[str, object] | None = None,
                            streams: int = STREAMS) -> dict:
    """Range-fetch, verify every GRIB message, then publish one member file.

Servers must return the requested range exactly. A 200 whole-ensemble reply
is refused before its body is read, preventing a multi-gigabyte accidental
download and mixed-member preparation. Object ETags pin all range requests
to one version while an as-posted source may still be publishing: the first
range is asked alone to learn the tag, and the rest, asked in parallel,
carry it as If-Match.  ``records`` narrows the member to the records a
row's selection keeps (JSON-lines member indexes only).
"""
    from woof.member_grammar import load_member_grammar
    from woof.member_prep import verify_member_file
    from woof.record_index import _pieces
    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import packaged_member_grammar
    from woof.filesystem_paths import io_path

    options = {"opener": opener} if opener is not None else {}
    with paced_urlopen(Request(index_url), timeout=timeout, **options) as response:
        index = response.read(16 * 1024 * 1024 + 1)
    if len(index) > 16 * 1024 * 1024:
        raise ValueError("Member index exceeds 16 MiB; refusing an unbounded response")
    initial_etag = None
    object_bytes = None
    if declaration.get("format") == "wgrib2":
        if set(declaration) != {"format"} or forecast_hour is None:
            raise ValueError("wgrib2 lead index requires an explicit forecast hour")
        if records is not None:
            raise ValueError("A wgrib2 lead index selects a lead, not records; the row "
                             "cannot also narrow it by a record selection")
        with paced_urlopen(Request(url, method="HEAD"), timeout=timeout, **options) as response:
            object_bytes = int(response.headers["Content-Length"])
            initial_etag = response.headers.get("ETag")
        from woof.mapped_engine_bridge import require_engine
        import subprocess
        import tempfile
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".idx") as sidecar:
            sidecar.write(index)
            sidecar.flush()
            result = subprocess.run([str(require_engine()), "index-ranges", sidecar.name,
                                     str(forecast_hour), str(object_bytes)],
                                    capture_output=True, text=True, check=True, timeout=60.0)
        selected = json.loads(result.stdout)
        ranges = tuple(tuple(row) for row in selected["ranges"])
    else:
        ranges = select_member_ranges(index, declaration, member_ordinal, records)
    dest = io_path(Path(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    total = sum(length for _offset, length in ranges)
    pieces = _pieces(ranges)
    # The object version every range must come from: (strong ETag, size).
    pinned = {"etag": initial_etag, "size": object_bytes}
    job = fetch_pool.current_job()
    failed = threading.Event()

    def transfer(piece: tuple[int, int, int]) -> None:
        offset, length, target = piece
        end = offset + length - 1
        headers = {"Range": f"bytes={offset}-{end}"}
        if pinned["etag"] is not None:
            headers["If-Match"] = pinned["etag"]
        try:
            with fetch_pool.working_for(job):
                fetch_pool.raise_if_stopped()
                with paced_urlopen(Request(url, headers=headers),
                                   timeout=timeout, **options) as response:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if (response.status != 206 or match is None
                            or tuple(map(int, match.groups()[:2])) != (offset, end)):
                        raise ValueError("Member range reply did not match the exact requested bytes")
                    object_size = int(match.group(3))
                    if object_size <= end:
                        raise ValueError("Member Content-Range extends beyond the declared object size")
                    received_etag = response.headers.get("ETag")
                    if (received_etag is None
                            or re.fullmatch(r'"[\x21\x23-\x7e\x80-\xff]*"', received_etag) is None):
                        raise ValueError("Member range reply needs a quoted strong ETag to pin exact bytes")
                    if pinned["etag"] is not None and (
                            received_etag != pinned["etag"]
                            or pinned["size"] != object_size):
                        raise ValueError("Member object changed between index range requests")
                    pinned["etag"], pinned["size"] = received_etag, object_size
                    remaining = length
                    with part.open("r+b") as target_file:
                        target_file.seek(target)
                        while remaining:
                            fetch_pool.raise_if_stopped()
                            if failed.is_set():
                                return
                            chunk = response.read(min(_CHUNK, remaining))
                            if not chunk:
                                raise ValueError("Member range reply ended before its declared length")
                            target_file.write(chunk)
                            remaining -= len(chunk)
                    if response.read(1):
                        raise ValueError("Member range reply exceeded its declared length")
        except BaseException:
            failed.set()
            raise

    try:
        with part.open("wb") as output:
            output.truncate(total)
        # The first range alone, so every later one is pinned to its tag.
        transfer(pieces[0])
        rest = pieces[1:]
        if rest:
            width = 1 if is_nomads_url(url) else max(1, min(streams, len(rest)))
            fetch_pool.record_streams(width)
            with ThreadPoolExecutor(max_workers=width,
                                    thread_name_prefix="gpuwm-member") as pool:
                list(pool.map(transfer, rest))
        digest = hashlib.sha256()
        with part.open("rb") as payload:
            for chunk in iter(lambda: payload.read(_CHUNK), b""):
                digest.update(chunk)
        grammar_id = get_source_adapter(source).member_set
        grammar = load_member_grammar(packaged_member_grammar(grammar_id))
        evidence = verify_member_file(grammar, member, part)
        if evidence.messages != len(ranges):
            raise ValueError("Selected index record count differs from the verified GRIB message count")
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return {
        "name": dest.name, "bytes": total, "sha256": digest.hexdigest(),
        "url": url, "member": member, "member_ordinal": member_ordinal,
        "index_url": index_url, "index_sha256": hashlib.sha256(index).hexdigest(),
        "selected_records": len(ranges), "requests": len(pieces),
        "record_selection": records is not None, "object_etag": pinned["etag"],
        "object_bytes": pinned["size"], "member_verification": evidence.to_dict(),
    }
