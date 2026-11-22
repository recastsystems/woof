"""NEXRAD Level II archive helpers: list, fetch, time and causality of archive objects.

The radar latent-heating window builder (``tools/radar_tten_windows.py``) and
the all-radar grid builder (``tools/obs_radar_grid_build_all.py``) read these.
They lived in the local data assimilation's observation tools
(``tools/conus_da_observations.py`` and ``tools/da_recent_observations.py``),
which gpuwm 3.0.3 removed; the functions are carried here unchanged.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from woof.obs.obspack import sha256_of as sha  # noqa: E402,F401

UTC = dt.timezone.utc
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
RADAR_BUCKET = "unidata-nexrad-level2"
RANGE_KM = 250.0
MAX_ELEVATION_DEG = 20.0
MAX_AGE_S = 900


def utc(value):
    stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def iso(stamp):
    return stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def save(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def object_time(key):
    match = re.search(r"(\d{8})[-_](\d{6})", key)
    if not match:
        return None
    return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def list_objects(bucket, prefix, *, timeout=60):
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if token:
            query["continuation-token"] = token
        url = f"https://{bucket}.s3.amazonaws.com/?" + urllib.parse.urlencode(query)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    tree = ET.fromstring(response.read())
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        for row in tree.findall("s:Contents", NS):
            yield {"bucket": bucket, "key": row.findtext("s:Key", namespaces=NS),
                   "bytes": int(row.findtext("s:Size", namespaces=NS)),
                   "etag": row.findtext("s:ETag", namespaces=NS),
                   "last_modified": row.findtext("s:LastModified", namespaces=NS)}
        token = tree.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            break


def download(row, root, *, timeout=120, cache=None):
    """The object under ``root``; an identical-size copy under the read-only
    ``cache`` (another lane's fetch tree, same bucket/key layout) is used in
    place and never written to."""
    rel = Path(row["bucket"]) / Path(*row["key"].split("/"))
    if cache is not None:
        cached = Path(cache) / rel
        if cached.is_file() and cached.stat().st_size == row["bytes"]:
            return cached, 0.0, False
    path = Path(root) / rel
    if path.is_file() and path.stat().st_size == row["bytes"]:
        return path, 0.0, False
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://{row['bucket']}.s3.amazonaws.com/" + urllib.parse.quote(row["key"])
    started = time.perf_counter()
    for attempt in range(4):
        try:
            tmp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as sink:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    sink.write(block)
            if tmp.stat().st_size != row["bytes"]:
                raise OSError("short download")
            tmp.replace(path)
            return path, time.perf_counter() - started, True
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def _explicit_utc(value):
    """An ISO time that states UTC (Z or +00:00); anything else is refused."""
    stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() != dt.timedelta(0):
        raise ValueError("explicit UTC time required")
    return stamp.astimezone(UTC)


#: Archive re-stamps: a bucket whose objects were re-uploaded wholesale, so
#: their S3 LastModified is the re-upload date and says nothing about when
#: the scan was first published.  One row per re-stamp; adding a bucket or
#: a second migration is a row here, never a code path.
#:
#: ``restamped_from``: LastModified on or after this is the re-upload's.
#: ``scanned_before``: only objects whose KEY scan time is before this were
#: re-stamped; anything scanned later was published after the migration and
#: keeps its LastModified as the publication bound.
#:
#: Breakage this prevents: 2024 archive objects re-stamped 2025-07-15 by
#: bucket migration; every pre-migration replay case refused (every 2024
#: unidata-nexrad-level2 object reads LastModified 2025-07-15T22:59:04Z,
#: e.g. 2024/05/21/KDMX/KDMX20240521_175821_V06, so causal_object refused
#: all of them and radar_tten_windows wrote "no volumes to grid").
ARCHIVE_RESTAMPS = (
    # The migration re-uploaded the archive oldest first over about six weeks, so the row spans it: measured
    # LastModified of whole site-days (2026-10-12) -- 1998-05-03 KTLX 2025-06-10, 2008-05-10 KTLX 2025-06-25,
    # 2015-06-01 KDMX 2025-07-02, 2022-06-13 KIND 2025-07-11, 2024-05-21 KDMX 2025-07-15, 2025-07-16 KTLX
    # 2025-07-18.  The old row (from 2025-07-15) refused every volume of a 2022 replay case as "LastModified is
    # after analysis cutoff", so radar_tten_windows stopped with "no volumes to grid".
    {"bucket": "unidata-nexrad-level2",
     "restamped_from": "2025-06-01T00:00:00Z",
     "scanned_before": "2025-07-17T00:00:00Z",
     "reason": "bucket migration re-uploaded the archive scanned before 2025-07-17, oldest first, 2025-06 to 2025-07-18"},
)


#: ``causal_object``'s return values: which clock bounded publication.
CAUSAL_BY_LAST_MODIFIED = "last_modified"


CAUSAL_BY_SCAN_TIME = "archive-replay-scan-time"


def archive_restamp(row):
    """The ``ARCHIVE_RESTAMPS`` row that re-stamped ``row``, or ``None``.

    An object qualifies only when its bucket has a row, its LastModified is
    on or after that row's re-upload date, and its key scan time is before
    the migration: then the LastModified is the migration's, not the
    publication's.  A post-migration scan uploaded late does not qualify.
    """
    modified, scanned = row.get("last_modified"), object_time(str(row.get("key", "")))
    if modified is None or scanned is None:
        return None
    for restamp in ARCHIVE_RESTAMPS:
        if (row.get("bucket") == restamp["bucket"]
                and _explicit_utc(modified) >= _explicit_utc(restamp["restamped_from"])
                and scanned < _explicit_utc(restamp["scanned_before"])):
            return restamp
    return None


def causal_object(row, analysis):
    """Refuse an object not provably published by ``analysis``.

    LastModified is the publication upper bound, except for an object an
    ``ARCHIVE_RESTAMPS`` row re-stamped: its LastModified is the migration's,
    so causality is judged by the scan time in its key (archive replay).
    Returns the clock used.  A genuinely late object -- LastModified after
    the cutoff and not re-stamped -- is still refused.
    """
    modified = row.get("last_modified")
    if modified is None:
        raise ValueError("radar archive object has no publication upper-bound metadata")
    if _explicit_utc(modified) <= analysis:
        return CAUSAL_BY_LAST_MODIFIED
    if archive_restamp(row) is not None:
        if object_time(str(row["key"])) > analysis:
            raise ValueError("radar archive object scan time is after analysis cutoff")
        return CAUSAL_BY_SCAN_TIME
    raise ValueError("radar archive object LastModified is after analysis cutoff")


def available_memory_bytes():
    """Available host RAM, bounded by the current cgroup's remaining RAM."""
    values = []
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                values.append(int(line.split()[1]) * 1024)
        # A v2 container can have less memory than the host advertises.
        group = next(line.split(':', 2)[2] for line in
                     Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::'))
        mount = Path('/sys/fs/cgroup')
        roots = [mount]
        root = (mount / group.lstrip('/')).resolve()
        while root.is_relative_to(mount) and root != mount:
            roots.append(root)
            root = root.parent
        for root in roots:
            try:
                limit = (root / 'memory.max').read_text().strip()
                if limit != 'max':
                    values.append(max(0, int(limit) - int((root / 'memory.current').read_text())))
            except (OSError, ValueError):
                continue
    except (OSError, StopIteration, ValueError):
        pass
    if not values:
        raise ValueError('cannot measure available RAM; unbounded observation workers could exhaust memory')
    return min(values)


def observation_workers(site_count, slot_count=1, *, requested=None,
                        available_bytes=None, worker_bytes=2 * 2**30,
                        retained_slot_bytes=0):
    """Size native work, including results retained until the ordered merge.

    worker_bytes is a conservative scratch allowance, replaceable by a measured
    peak in callers. Retained outputs are separately reserved for every slot.
    """
    if site_count < 1 or slot_count < 1 or worker_bytes < 1 or retained_slot_bytes < 0:
        raise ValueError('positive site, slot and worker sizes required')
    if requested is not None and requested < 1:
        raise ValueError('workers must be positive; zero would suppress observations')
    available = available_memory_bytes() if available_bytes is None else int(available_bytes)
    budget = available * 4 // 5
    slots = min(slot_count, budget // (retained_slot_bytes + worker_bytes))
    if slots < 1:
        raise ValueError('available RAM cannot hold one observation slot and native worker; prevents OOM and incomplete observations')
    wanted = min(site_count, requested if requested is not None else site_count)
    count = min(wanted, (budget - slots * retained_slot_bytes) // (slots * worker_bytes))
    return {'workers': int(count), 'slot_workers': int(slots),
            'total_workers': int(count * slots),
            'site_count': site_count, 'slot_count': slot_count,
            'available_bytes': available, 'budget_bytes': budget,
            'worker_scratch_bytes': worker_bytes, 'retained_slot_bytes': retained_slot_bytes,
            'reserve_fraction': 0.2}


def native_observation_threads(site_count, slot_count=1, requested=None):
    from woof.obs import superob_bridge
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    per_worker = max(1, int(os.environ.get('RAYON_NUM_THREADS', '2')))
    wanted = min(site_count, requested if requested is not None else site_count)
    return superob_bridge.configure_threads(min(cores, max(1, wanted * slot_count * per_worker)))


def observation_memory(grid, sites, params):
    """Reserve native accumulator and merge buffers from Rust reach windows."""
    from woof.obs.superob import horizontal_window
    cells = []
    for site in sites:
        j0, j1, i0, i1 = horizontal_window(grid, site, params)
        cells.append(grid.nz * (j1 - j0 + 1) * (i1 - i0 + 1))
    # 16 eight-byte accumulator planes plus windowed merge outputs and copies.
    # Full-grid output/writer buffers are also live during the merge.
    retained = 256 * sum(cells) + 256 * grid.nz * grid.ny * grid.nx
    return retained


def require_native_observations():
    from woof.obs import superob_bridge
    if superob_bridge.python_reference_requested():
        raise ValueError('parallel observation preparation requires Rust superob; the Python reference cannot be pooled')
    superob_bridge.library()


def process_tree_rss_bytes():
    """RSS of this process and its children, without reading command lines."""
    pending, seen, total = [os.getpid()], set(), 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        root = Path('/proc') / str(pid)
        try:
            for line in (root / 'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1]) * 1024
            # A subprocess can belong to any thread, not just the main thread.
            for task in (root / 'task').iterdir():
                try:
                    pending.extend(int(value) for value in (task / 'children').read_text().split())
                except (OSError, ValueError):
                    pass
        except (OSError, ValueError):
            pass
    return total


class ObservationRss:
    """Sample the native process tree. Short-lived peaks can fall between samples."""
    def __enter__(self):
        self.before = self.peak = process_tree_rss_bytes()
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.05):
                self.peak = max(self.peak, process_tree_rss_bytes())
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, process_tree_rss_bytes())

    def receipt(self):
        return {'before_tree_rss_bytes': self.before, 'peak_tree_rss_bytes': self.peak,
                'incremental_peak_bytes': max(0, self.peak - self.before),
                'sample_seconds': .05, 'basis': 'sampled RSS; shared pages counted per process'}
