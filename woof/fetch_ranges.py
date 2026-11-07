"""Parallel whole-object HTTP transfers for the shared Python route pool."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
from urllib.request import Request

from woof import fetch_connections, fetch_pool
from woof.nomads_governor import is_nomads_url, paced_urlopen

CHUNK = 1024 * 1024
MIN_PART = 16 * 1024 * 1024


class RangeRefused(ValueError):
    """A host answered a byte-range request with something else.

    A 200 with the whole representation, a 206 whose Content-Range names
    other bytes, or an encoded body.  Named breakage (WOOF incident
    2026-11-04, fix/ecmwf-open-data-range): a customer start from the
    2026-10-21 06Z IFS 0.1-degree cycle failed in 41 s because the one
    mirror that still kept that cycle "ignored or changed the requested
    byte range" on all five rounds, although the whole object was
    there.  The callers move the object whole on one plain GET instead.
    """


def download(url, part, *, user_agent, streams=1, timeout=300, opener=None,
             progress=None, declared_size=None):
    """Write one identity representation, validating every returned range.

    The initial GET doubles as the range-capability probe. Small objects
    and origins without byte ranges or a generation validator keep it.
    Large objects use long disjoint streams, avoiding repeated TCP starts.
    A host that refuses a range mid-transfer (:class:`RangeRefused`) gets
    the object again on one stream without a Range header, so a mirror
    that ignores Range costs one transfer, never the run.
    """
    try:
        return _download(url, part, user_agent=user_agent, streams=streams,
                         timeout=timeout, opener=opener, progress=progress,
                         declared_size=declared_size)
    except RangeRefused:
        if streams <= 1:
            raise
        return _download(url, part, user_agent=user_agent, streams=1,
                         timeout=timeout, opener=opener, progress=progress,
                         declared_size=declared_size)


def _download(url, part, *, user_agent, streams, timeout, opener, progress,
              declared_size):
    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
    kwargs = {} if opener is None else {"opener": opener}
    response = paced_urlopen(Request(url, headers=headers), timeout=timeout, **kwargs)
    with response:
        raw_length = response.headers.get("Content-Length")
        length = None if raw_length is None else int(raw_length)
        if declared_size is not None and length is not None:
            declared_size(part.with_name(part.name.removesuffix(".part")), length)
        etag = response.headers.get("ETag")
        modified = response.headers.get("Last-Modified")
        validator = ({"If-Match": etag} if etag and not etag.startswith("W/") else
                     {"If-Unmodified-Since": modified} if modified else {})
        width = min(streams, fetch_connections.max_connections(),
                    max(1, (length or 0) // MIN_PART))
        if is_nomads_url(url):
            width = 1
        ranged = (width > 1 and validator and
                  response.headers.get("Accept-Ranges", "").lower() == "bytes" and
                  response.headers.get("Content-Encoding", "identity") == "identity")
        if not ranged:
            fetch_pool.record_streams(1)
            digest = hashlib.sha256()
            written = 0
            with part.open("wb") as output:
                while True:
                    fetch_pool.raise_if_stopped()
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(len(chunk))
            if length is not None and written != length:
                raise ValueError(f"{part.name}: the host declared {length} bytes and delivered {written}")
            return written, digest.hexdigest()
    # The probe is closed before any range opens, so it costs no extra slot.
    fetch_pool.record_streams(width)
    job = fetch_pool.current_job()
    failed = threading.Event()
    with part.open("wb") as output:
        output.truncate(length)

    def transfer(index):
        start, stop = length * index // width, length * (index + 1) // width - 1
        request = Request(url, headers={**headers, **validator, "Range": f"bytes={start}-{stop}"})
        try:
            with fetch_pool.working_for(job):
                fetch_pool.raise_if_stopped()
                with paced_urlopen(request, timeout=timeout, **kwargs) as body:
                    if (getattr(body, "status", None) != 206 or
                            body.headers.get("Content-Range") != f"bytes {start}-{stop}/{length}" or
                            body.headers.get("Content-Encoding", "identity") != "identity"):
                        raise RangeRefused(f"{part.name}: host ignored or changed the requested byte range {start}-{stop}")
                    if etag and body.headers.get("ETag") != etag:
                        raise ValueError(f"{part.name}: object ETag changed between range streams")
                    if not etag and modified and body.headers.get("Last-Modified") != modified:
                        raise ValueError(f"{part.name}: object changed between range streams")
                    remaining = stop - start + 1
                    with part.open("r+b") as output:
                        output.seek(start)
                        while remaining:
                            fetch_pool.raise_if_stopped()
                            if failed.is_set():
                                return
                            chunk = body.read(min(CHUNK, remaining))
                            if not chunk:
                                raise ValueError(f"{part.name}: truncated range {start}-{stop}")
                            output.write(chunk)
                            remaining -= len(chunk)
                            if progress is not None:
                                progress(len(chunk))
                        if body.read(1):
                            raise ValueError(f"{part.name}: excess bytes after range {start}-{stop}")
        except BaseException:
            failed.set()
            raise

    with ThreadPoolExecutor(max_workers=width, thread_name_prefix="gpuwm-range") as pool:
        list(pool.map(transfer, range(width)))
    # Verification remains inside each file worker, parallel across files.
    digest = hashlib.sha256()
    with part.open("rb") as payload:
        for chunk in iter(lambda: payload.read(CHUNK), b""):
            fetch_pool.raise_if_stopped()
            digest.update(chunk)
    return length, digest.hexdigest()
