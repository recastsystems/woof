"""Atomic per-hour producer receipts and checked, ordered consumer admission."""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import time

FORMAT = "%Y-%m-%d_%H:%M:%S"
SERIES = "gpuwm-hex.lbc-series/v1"
READY = "gpuwm-hex.lbc.hour/v1"


def file_name(valid_time):
    return f"lbc.{valid_time.replace(':', '.')}.nc"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checked_marker(path, *, schema, valid_time=None):
    """Never admit a partial file, a moved receipt or changed complete bytes."""
    path = Path(path)
    document = json.loads(path.read_text())
    data = Path(document["path"])
    if not data.is_absolute():
        data = path.parent / data
    if data.resolve().parent != path.resolve().parent or data.is_symlink():
        raise ValueError(f"{path}: receipt escapes its producer directory; refuses unrelated input")
    if document.get("schema") != schema:
        raise ValueError(f"{path}: wrong ready schema; partial input cannot be admitted")
    if valid_time is not None and document.get("valid_time") != valid_time:
        raise ValueError(f"{path}: wrong valid_time; admitting it would drive the wrong interval")
    before = data.stat()
    if before.st_size != document.get("bytes") or sha256(data) != document.get("sha256"):
        raise ValueError(f"{path}: complete-file hash mismatch; admitting changed bytes would change the forecast")
    after = data.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"{path}: input changed during admission; refuses a torn file")
    return data


def series_document(directory):
    path = Path(directory) / "lbc-series.json"
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    times = document.get("valid_times", [])
    if document.get("schema") != SERIES or len(times) < 2:
        raise ValueError(f"{path}: invalid boundary schedule; refuses an unbounded or empty series")
    moments = [datetime.strptime(item, FORMAT) for item in times]
    interval = document.get('interval_seconds', 3600)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError(f"{path}: boundary interval_seconds must be a positive integer")
    if any(b - a != timedelta(seconds=interval) for a, b in zip(moments, moments[1:])):
        raise ValueError(f"{path}: boundary schedule differs from its {interval}-second interval; refuses a missing interpolation endpoint")
    return document


def wait_file(directory, valid_time, *, timeout=1800.0, stage="linger.lbc_hour"):
    from . import time_ledger
    directory = Path(directory)
    path = directory / file_name(valid_time)
    marker = path.with_name(path.name + ".ready.json")
    started = time.monotonic()
    waiting = False
    try:
        while not marker.is_file():
            failure = directory / "lbc-failed.json"
            if failure.is_file():
                raise ValueError(f"boundary producer failed: {failure.read_text()}; refuses frozen boundaries")
            if not waiting:
                waiting = True
                import sys
                print(f"forecast: waiting for complete boundary {valid_time}: {marker}",file=sys.stderr,flush=True)
                time_ledger.emit(stage, "begin", attrs={"valid_time": valid_time, "marker": str(marker),
                                                       "reason": "missing_lbc_hour"})
            if time.monotonic() - started >= timeout:
                raise ValueError(f"missing_lbc_hour {valid_time}: timed out waiting for {marker}; refuses frozen boundaries")
            time.sleep(0.05)
        admitted = checked_marker(marker, schema=READY, valid_time=valid_time)
        if admitted != path:
            raise ValueError(f"{marker}: file name differs from the scheduled endpoint")
        return admitted
    finally:
        if waiting:
            import sys
            print(f"forecast: boundary wait ended for {valid_time} after {time.monotonic()-started:.3f} s",file=sys.stderr,flush=True)
            time_ledger.emit(stage, "end", attrs={"valid_time": valid_time, "marker": str(marker)})


def initial_paths(directory):
    document = series_document(directory)
    if document is None:
        return sorted(Path(directory).glob("lbc.*.nc"))
    from .lbc import read_lbc_valid_time
    paths = [wait_file(directory, item, stage="prep.lbc.wait") for item in document["valid_times"][:2]]
    for item, path in zip(document["valid_times"], paths):
        if read_lbc_valid_time(path) != datetime.strptime(item, FORMAT):
            raise ValueError(f"{path}: xtime differs from its ready marker; refuses wrong-time forcing")
    return paths
