#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1 (ad/phys): add what a finalized hrrrv5 package needs for its replay to equal the ordinary forward.

1. module_initializers: zero-argument kernels launched before step 1 in replayed modules (module_initializers.py).
   Breakage: RUC's ruc_tbq_table (a module global, filled by ruc_tbq_init at physics_init) stayed zero in the
   replay, sea ice read it, 38 allocations wrong at step 1 in every replay.
2. recorded_bases: recording A's base address of EVERY allocation the plan creates (initial-state allocations and
   every `allocate` command), so the runner hashes carried state in the forward's address space. Breakage: with
   initial-state allocations only, a table entry pointing into an allocation made during a step (e.g. TEMPO's
   20-entry table in allocation 102 at step 1 pointing into 408/420/422/445/414) stayed a replay address, and 15
   carried address tables (96 102 104 122 144 154 158 170 190 194 201 373 377 379 423) failed the step-1 hash
   although every byte was right (pkg6 r1t dumps: all 15 equal to recording A once every id is translated).
3. relocations: initial-state words that are recorded device addresses (tables the forward built before step 1).
   Breakage: update_diag_pool faulted (CUDA 700) at its first replayed launch, step 2, on allocation 125's table.
4. held pointers: words of held-slot blobs (held_upload_ranges, milestone-1 radiation and land) that are
   recording-A device addresses of an allocation the replay has live at that command. The runner checks each
   against its recorded base and relocates it. Breakage: held blobs were uploaded verbatim; land's 11-entry
   pointer table (allocation 172, step 1) and 16 words of pool root 347 kept recording addresses, and those that
   fell inside a different replay allocation (0x1000ad32000 and 0x1000ad4ee00 in 172) mistranslated, so 172 and
   347 failed the step-1 hash (pkg6 r1t: both equal to recording A with exactly those words kept as recorded).
5. With a recorder observer folder holding tempo-sched/step-N.json (record_observer.py) and recording A's
   boundary payloads (recorder BINDER_BOUNDARY_PAYLOADS=1): excluded_residue, the TEMPO scheduling words (per-column
   cost from clock64, the column order from an atomic scatter, the bucket starts) and the TEMPO per-warp scratch
   (state, work, PBL work), located in the carried allocations at each step end. They are timing-based and differ
   run to run in the ordinary forward itself, like native-undefined words (the RUC tsnav precedent). The runner
   zeroes exactly those ranges and compares the rest of each allocation with the forward's boundary bytes masked
   the same way (masked_sha256); everything outside the ranges stays compared. The witness (residue_witness.py,
   two forwards) is written to the package and named in every entry. Breakage: allocations 414, 416, 417 and 446
   failed the step hash in every replay.
Usage: p1fix_package.py PACKAGE_DIR CATALOG_A DRIVER_TRACE_A [OBSERVER_A [OBSERVER_B]]   (rewrites PACKAGE_DIR/plan.json)
"""
import bisect
import collections
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_initializers import initializers  # noqa: E402

RESIDUE_REASON = ('declared residue: TEMPO scheduling words (per-column clock64 cycle cost, column order from an '
                  'atomic counting-sort scatter, bucket starts) and TEMPO per-warp scratch (state, work, PBL work) '
                  'whose warp-to-column assignment follows that order; timing-based, they differ run to run in the '
                  'ordinary forward itself, like native-undefined words (RUC tsnav precedent)')


_BOUNDARY_CLOCKS = {}
_T0 = None


def stage(name):
    """One stderr line per stage with wall seconds since start, so a receipt names each stage's time."""
    import time
    global _T0
    now = time.monotonic()
    if _T0 is None:
        _T0 = now
    print(json.dumps({'event': 'p1fix_stage', 'stage': name, 'elapsed_s': round(now - _T0, 2)}), file=sys.stderr, flush=True)


def _cgroup_dirs(proc_cgroup='/proc/self/cgroup', root='/sys/fs/cgroup'):
    """This process's own cgroup directory and every one above it, nearest first (cgroup v2: the '0::' line)."""
    try:
        own = next(line.split('::', 1)[1].strip() for line in open(proc_cgroup) if line.startswith('0::'))
    except (OSError, StopIteration):
        return [root]
    parts = [part for part in own.split('/') if part and part != '..']
    return [os.path.join(root, *parts[:n]) for n in range(len(parts), -1, -1)]


def cgroup_room_bytes(dirs=None):
    """What this process's cgroups still leave it: over its own cgroup and every one above, the smallest of
    (memory.high, memory.max) less what is charged now and cannot be given back (memory.current less the file cache,
    which the kernel drops first). None when no cgroup sets a limit.
    THE BREAKAGE THIS PREVENTS: the limit was read from /sys/fs/cgroup/memory.max alone, the ROOT of the tree, which
    holds no such file on a host, so a job inside a systemd scope or a batch system's cgroup (MemoryHigh, MemoryMax)
    sized its worker pool from the whole machine's free memory. On a 512 GB rented card host, inside a scope of
    MemoryHigh=20G MemoryMax=40G, 16 workers each inflating a step's residue payloads went past the soft limit, where
    the kernel does not kill but stalls every allocation: the packaging step that takes 20 s outside the scope did not
    exit in 7.5 min and the recording's time limit ended it with no package (a 12 km recording on NOAA's hrrrv5 deck,
    2026-11-16 00:53Z to 01:00Z)."""
    room = None
    for d in (_cgroup_dirs() if dirs is None else dirs):
        try:
            current = int(open(os.path.join(d, 'memory.current')).read())
        except (OSError, ValueError):
            continue   # the root cgroup, or a tree that is not mounted here: no limit to read at this level
        cache = 0
        try:
            for line in open(os.path.join(d, 'memory.stat')):
                key, _, value = line.partition(' ')
                if key in ('active_file', 'inactive_file'):
                    cache += int(value)
        except (OSError, ValueError):
            pass
        for name in ('memory.high', 'memory.max'):
            try:
                limit = open(os.path.join(d, name)).read().strip()
            except OSError:
                continue
            if limit and limit != 'max':
                left = max(0, int(limit) - max(0, current - cache))
                room = left if room is None else min(room, left)
    return room


def free_memory_bytes():
    """Memory this process may still take: the smaller of MemAvailable and the headroom of its own cgroups
    (cgroup_room_bytes: memory.high and memory.max of its own cgroup and every one above it), so a scope's or a
    container's limit counts, not only the host's."""
    free = None
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                free = int(line.split()[1]) * 1024
    except OSError:
        pass
    room = cgroup_room_bytes()
    if room is not None:
        free = room if free is None else min(free, room)
    return free


def pool_workers(jobs, peak_bytes, free=None):
    """Worker count for a pool whose jobs each hold up to peak_bytes at once: at most half of free memory in
    flight, at most 16 and the core count, at least 1, capped by P1_PKG_WORKERS when set. Breakage: with
    min(cpu, 16) workers each inflating a ~750 MB held blob plus numpy temporaries, p1fix_package was killed
    without a traceback 7 min in on the 8x5090 box (2026-11-04 19:59:58Z, ~55 GB available beside other jobs);
    the box rerun needed a hand-set P1_PKG_WORKERS=4."""
    n = max(1, min(jobs, os.cpu_count() or 4, 16))
    cap = os.environ.get('P1_PKG_WORKERS')
    if cap:
        n = max(1, min(n, int(cap)))
    free = free_memory_bytes() if free is None else free
    if free is not None and peak_bytes > 0:
        n = max(1, min(n, int(free // 2 // peak_bytes)))
    return n


def _trace_lines(trace):
    """Byte lines of a gzip driver trace. With pigz on PATH the inflate runs in its own process beside the parse."""
    if shutil.which('pigz'):
        proc = subprocess.Popen(['pigz', '-dc', str(trace)], stdout=subprocess.PIPE, bufsize=1 << 22)
        try:
            yield from proc.stdout
        finally:
            proc.stdout.close()
            if proc.wait() != 0:
                sys.exit(f'refusing: pigz -dc {trace} failed (rc {proc.returncode})')
        return
    with gzip.open(trace, 'rb') as stream:
        yield from stream


def boundary_clocks(trace):
    """Every step's Boundary clock in ONE pass over the trace (the first Boundary of each step, as before).
    Breakage: boundary_clock re-read the whole gzip trace from the start for every step, 19 passes for the 18-step
    plan, about 18 min at 100 % of one core on a recording host (2026-11-04 12:00Z) while the card waited."""
    key = str(trace)
    if key not in _BOUNDARY_CLOCKS:
        clocks, masked = {}, {}
        for line in _trace_lines(trace):
            if b'Boundary' not in line:
                continue
            d = json.loads(line)
            if d.get('type') == 'Boundary':
                clocks.setdefault(d.get('a'), d['clock_ns'])
            elif d.get('type') == 'BoundaryMaskedDigest':
                masked[(d['step'], d['base'], d['bytes'])] = d
        _BOUNDARY_CLOCKS[key] = clocks
        _MASKED_DIGESTS[key] = masked
    return _BOUNDARY_CLOCKS[key]


_MASKED_DIGESTS = {}


def masked_digests(trace):
    """The recorder's residue digests (BoundaryMaskedDigest: step, base, bytes, sha256, masked_sha256, ranges), read in
    the same pass as the boundary clocks."""
    boundary_clocks(trace)
    return _MASKED_DIGESTS[str(trace)]


def boundary_clock(trace, step):
    clocks = boundary_clocks(trace)
    if step not in clocks:
        sys.exit(f'refusing: no Boundary {step} in {trace}')
    return clocks[step]


def initial_relocations(data, rows, starts):
    """Initial-state words that are recording-A addresses of a live allocation: (byte offset, id, offset, value).
    Same rule as the per-word loop it replaces (every 8-byte-aligned word, value >= the lowest base, inside
    [base, end] of the allocation found by bisect_right), vectorised: the loop was pure Python over every word."""
    n = len(data) // 8
    if n == 0 or not rows:
        return []
    words = np.frombuffer(data, dtype='<u8', count=n)
    base = np.array(starts, dtype=np.uint64)
    end = np.array([r[1] for r in rows], dtype=np.uint64)
    cand = np.nonzero(words >= base[0])[0]
    vals = words[cand]
    k = np.searchsorted(base, vals, side='right') - 1
    ok = (k >= 0) & (vals <= end[np.where(k >= 0, k, 0)])
    cand, vals, k = cand[ok], vals[ok], k[ok]
    ids = [r[2] for r in rows]
    return [(int(i) * 8, ids[j], int(v) - rows[j][0], int(v)) for i, v, j in zip(cand.tolist(), vals.tolist(), k.tolist())]


def _relocation_job(job):
    path, rows, starts = job
    return initial_relocations(gzip.open(path).read(), rows, starts)


def _values_in_payload(job):
    """pk120: which of the candidate 8-byte values recording B's initial payload holds, at byte alignment 0 or 4."""
    path, cand = job
    data = gzip.open(path).read()
    c = np.array(cand, dtype=np.uint64)
    out = set()
    for start in (0, 4):
        n = (len(data) - start) // 8
        if n <= 0:
            continue
        w = np.frombuffer(data, dtype='<u8', count=n, offset=start)
        out.update(int(v) for v in np.unique(w[np.isin(w, c)]))
    return out


def live_ranges(catalog, clock):
    rows = sorted((a['recorded_base'], a['recorded_base'] + a['bytes'], a['id']) for a in catalog['allocations']
                  if a['allocated_clock_ns'] <= clock and (a['freed_clock_ns'] is None or a['freed_clock_ns'] > clock))
    return rows, [r[0] for r in rows]


def locate(rows, starts, value):
    k = bisect.bisect_right(starts, value) - 1
    if k >= 0 and value <= rows[k][1]:
        return rows[k][2], value - rows[k][0]
    return None


def step_keys(plan):
    return sorted((k for k in plan['blocks'] if k.startswith('step_')), key=lambda k: int(k.split('_')[1]))


def plan_ids(plan):
    """Every allocation id the plan creates, with its size: initial state plus every allocate command."""
    ids = {a['id']: a['bytes'] for a in plan['allocations']}
    for key in step_keys(plan):
        for c in plan['blocks'][key]:
            if c['kind'] == 'allocate':
                if ids.setdefault(c['id'], c['bytes']) != c['bytes']:
                    sys.exit(f'refusing: allocation {c["id"]} allocated with two sizes')
    return ids


def scan_held(job):
    """Address words of one held blob: (blob byte offset, target id, target offset). Words are 8-byte aligned in
    destination space; a word counts only if it lies inside a recording-A allocation the replay has live here."""
    path, ranges, rows = job
    data = gzip.open(path).read()
    starts = np.array([r[0] for r in rows], dtype=np.uint64)
    ends = np.array([r[1] for r in rows], dtype=np.uint64)
    ids = [r[2] for r in rows]
    src = np.array([r['source_offset'] for r in ranges], dtype=np.int64)
    size = np.array([r['bytes'] for r in ranges], dtype=np.int64)
    phase = (src - np.array([r['destination']['offset'] for r in ranges], dtype=np.int64)) % 8
    order = np.argsort(src)
    src, size, phase = src[order], size[order], phase[order]
    lo, hi = int(starts.min()), int(ends.max())
    found = []
    for c in range(8):
        n = (len(data) - c) // 8
        if n <= 0:
            continue
        words = np.frombuffer(data, dtype='<u8', count=n, offset=c)
        cand = np.nonzero((words >= lo) & (words < hi))[0]
        if not len(cand):
            continue
        pos = cand.astype(np.int64) * 8 + c
        k = np.searchsorted(src, pos, side='right') - 1
        ok = (k >= 0)
        k = np.where(ok, k, 0)
        ok &= (pos + 8 <= src[k] + size[k]) & (phase[k] == c)
        pos, vals = pos[ok], words[cand[ok]]
        j = np.searchsorted(starts, vals, side='right') - 1
        hit = (j >= 0) & (vals < ends[np.where(j >= 0, j, 0)])
        for p, v, jj in zip(pos[hit].tolist(), vals[hit].tolist(), j[hit].tolist()):
            found.append((p, ids[jj], v - int(starts[jj])))
    found.sort()
    return found


def split_words(plan, pkg, by_id, trace):
    """Held writes that cover only part of an 8-byte destination word (the tracer records the bytes that changed,
    so a pointer whose high half did not change is held as its 4 low bytes). Uploaded as is, the replay keeps its
    own high half and gets the recording's low half: a word that is neither address. Breakage: allocation 394
    failed the step-2 hash in every replay of the 18-step pair's 6-step plan (a recording host, 2026-11-04): the land hold's
    956-byte range ends 4 bytes into the word at 34918840, which the forward set to 0x7e58069ff800 (allocation
    116), and the replay held 0x7354069ff800 (inside allocation 391); the same for 424, 347, 128 and 145 at step 3.
    Each such word is completed from the forward's own step-end bytes (recording A's boundary payload) when the
    held bytes equal those bytes there (the word is unchanged after the hold) and the whole word is a recording-A
    address of an allocation live here; it is then appended to the blob as one 8-byte range, which scan_held
    lists as a held pointer, so the runner checks it against its recorded base and relocates it. Words that are
    not addresses stay as they are (a partial write of data is already exact).
    Every held command is independent of the others (its own blob, its own live set), so the commands run in a
    process pool. Breakage: the commands ran one after another on one core, the 15 min single-core stretch before the
    held-pointer pool on a recording host (2026-11-04 20:15Z); same output bytes."""
    payloads = Path(trace).parent / 'payloads'
    live = {a['id'] for a in plan['allocations']}
    jobs, commands = [], []
    for key in step_keys(plan):
        step = int(key.split('_')[1])
        for c in plan['blocks'][key]:
            if c['kind'] == 'allocate':
                live.add(c['id'])
            elif c['kind'] == 'free':
                live.discard(c['id'])
            if c['kind'] != 'held_upload_ranges':
                continue
            if all(r['destination']['offset'] % 8 == 0 and (r['destination']['offset'] + r['bytes']) % 8 == 0
                   for r in c['ranges']):
                continue
            rows = sorted((by_id[i]['recorded_base'], by_id[i]['recorded_base'] + by_id[i]['bytes'], i) for i in live)
            jobs.append((str(pkg / c['data']['path']), c['ranges'], step, rows, str(pkg), str(payloads)))
            commands.append(c)
    added, unresolved = 0, collections.Counter()
    # Peak per job: the inflated blob, its extended copy and its compressed output, plus the one forward step-end
    # payload a job holds at a time (an allocation's bytes). Breakage: counting only the blob let 12 workers hold
    # ~3 GB each on the recording host (2026-11-05 00:45Z, 10 GB left on the host) and the run was stopped.
    blob = max((max((r['source_offset'] + r['bytes'] for r in j[1]), default=0) for j in jobs), default=0)
    payload = max((a['bytes'] for a in by_id.values()), default=0)
    with ProcessPoolExecutor(max_workers=pool_workers(len(jobs), 3 * blob + payload)) as pool:
        for c, (data, new_ranges, missed) in zip(commands, pool.map(split_job, jobs)):
            unresolved.update(missed)
            if not new_ranges:
                continue
            c['data'] = data
            c['ranges'] = c['ranges'] + new_ranges
            added += len(new_ranges)
    return added, dict(unresolved)


_REFS = {}


def _forward_refs(pkg):
    if pkg not in _REFS:
        refs = {}
        for line in open(Path(pkg) / 'reference.jsonl'):
            d = json.loads(line)
            if d.get('event') == 'state_sha256' and d.get('storage') != 'host':
                refs[(d['step'], d['allocation'])] = d['sha256']
        _REFS[pkg] = refs
    return _REFS[pkg]


def split_job(job):
    """One held command of split_words: (new data descriptor or None, new 8-byte ranges, unresolved counts)."""
    path, ranges, step, rows, pkg, payloads = job
    refs = _forward_refs(pkg)
    payloads = Path(payloads)
    cache = {}

    def forward(ident):
        if ident not in cache:
            sha = refs.get((step, ident))
            p = payloads / f'{sha}.gz' if sha else None
            # Words are visited in (allocation, offset) order, so one payload at a time is enough.
            cache.clear()
            cache[ident] = gzip.open(p).read() if p is not None and p.exists() else None
        return cache[ident]
    unresolved = collections.Counter()
    blob = None
    cover = collections.defaultdict(dict)   # (allocation, word offset) -> {byte: value}
    for r in ranges:
        d = r['destination']
        lo, hi = d['offset'], d['offset'] + r['bytes']
        if lo % 8 == 0 and hi % 8 == 0:
            continue
        if blob is None:
            blob = gzip.open(path).read()
        for w in {lo - lo % 8, (hi - 1) - (hi - 1) % 8}:
            for b in range(max(w, lo), min(w + 8, hi)):
                cover[(d['allocation'], w)][b - w] = blob[r['source_offset'] + b - lo]
    # whole-word coverage inside one range is already scanned; also add bytes of other ranges of this op
    if not cover:
        return None, [], unresolved
    words_of = collections.defaultdict(list)
    for ident, w in cover:
        words_of[ident].append(w)
    for ws in words_of.values():
        ws.sort()
    for r in ranges:
        d = r['destination']
        ws = words_of.get(d['allocation'])
        if not ws:
            continue
        lo, hi = d['offset'], d['offset'] + r['bytes']
        k = bisect.bisect_left(ws, lo - 7)
        while k < len(ws) and ws[k] < hi:
            w = ws[k]
            got = cover[(d['allocation'], w)]
            for b in range(max(w, lo), min(w + 8, hi)):
                got[b - w] = blob[r['source_offset'] + b - lo]
            k += 1
    starts = [x[0] for x in rows]
    extra = bytearray()
    new_ranges = []
    for (ident, w), got in sorted(cover.items()):
        if len(got) == 8:
            word = bytes(got[i] for i in range(8))
        else:
            data = forward(ident)
            if data is None or w + 8 > len(data):
                unresolved['no forward step-end bytes'] += 1
                continue
            word = bytes(got.get(i, data[w + i]) for i in range(8))
            if any(data[w + i] != v for i, v in got.items()):
                value = int.from_bytes(word, 'little')
                if locate(rows, starts, value) is not None:
                    unresolved['address word changed after the hold'] += 1
                continue
        value = int.from_bytes(word, 'little')
        if locate(rows, starts, value) is None:
            continue
        new_ranges.append({'destination': {'allocation': ident, 'offset': w},
                           'source_offset': len(blob) + len(extra), 'bytes': 8})
        extra += word
    if not new_ranges:
        return None, [], unresolved
    data = bytes(blob) + bytes(extra)
    sha = hashlib.sha256(data).hexdigest()
    out = Path(pkg) / 'split-held'
    out.mkdir(exist_ok=True)
    (out / f'{sha}.gz').write_bytes(gzip.compress(data, 1))
    return {'encoding': 'gzip', 'path': f'split-held/{sha}.gz', 'sha256': sha, 'bytes': len(data)}, new_ranges, unresolved


def held_pointers(plan, pkg, by_id):
    live = {a['id'] for a in plan['allocations']}
    jobs, commands = [], []
    for key in step_keys(plan):
        for c in plan['blocks'][key]:
            if c['kind'] == 'allocate':
                live.add(c['id'])
            elif c['kind'] == 'free':
                live.discard(c['id'])
            elif c['kind'] == 'held_upload_ranges':
                c.pop('pointers', None)
                path = pkg / c['data']['path']
                if not path.exists():
                    sys.exit(f'refusing: held blob {path} is missing')
                rows = sorted((by_id[i]['recorded_base'], by_id[i]['recorded_base'] + by_id[i]['bytes'], i) for i in live)
                jobs.append((str(path), c['ranges'], rows))
                commands.append(c)
    words, by_target = 0, {}
    # peak per job: the inflated blob (its ranges end at most at source_offset + bytes) plus numpy temporaries
    peak = max((max((r['source_offset'] + r['bytes'] for r in ranges), default=0) for _, ranges, _ in jobs), default=0)
    with ProcessPoolExecutor(max_workers=pool_workers(len(jobs), 2 * peak)) as pool:
        for c, found in zip(commands, pool.map(scan_held, jobs)):
            if found:
                c['pointers'] = [{'byte_offset': p, 'target': {'allocation': t, 'offset': o}} for p, t, o in found]
                words += len(found)
                for _, t, _ in found:
                    by_target[t] = by_target.get(t, 0) + 1
    return words, by_target


def residue(plan, catalog, trace, observer, pkg, witness):
    """excluded_residue entries for every step: TEMPO ranges per carried allocation and the forward's masked hash."""
    payloads = Path(trace).parent / 'payloads'
    refs = {}
    for line in open(pkg / 'reference.jsonl'):
        d = json.loads(line)
        if d.get('event') == 'state_sha256' and d.get('storage') != 'host':
            refs[(d['step'], d['allocation'])] = d['sha256']
    carried = set(plan['carried'])
    by_bytes = {a['id']: a['bytes'] for a in catalog['allocations']}
    entries, outside, jobs = [], [], []
    witness_note = (f'{witness["path"]} (sha256 {witness["sha256"]}): two ordinary forwards differ inside these '
                    f'arrays at every recorded boundary ({witness["summary"]}); nothing outside them is excluded, and '
                    'the replay (its own TEMPO timing) must equal the forward in every other carried byte at every '
                    'step boundary (runner excluded_residue: masked_equal, unmasked_equal)')
    for key in step_keys(plan):
        step = int(key.split('_')[1])
        rec = json.load(open(observer / 'tempo-sched' / f'step-{step}.json'))
        rows, starts = live_ranges(catalog, boundary_clock(trace, step + 1))
        groups = {}
        for row in rec['rows']:
            hit = locate(rows, starts, row['pointer'])
            if hit is None:
                sys.exit(f'refusing: TEMPO {row["name"]} at step {step} is in no recorded allocation')
            groups.setdefault(hit[0], []).append({'offset': hit[1], 'bytes': row['bytes'], 'array': row['name']})
        for allocation, ranges in sorted(groups.items()):
            if allocation not in carried:
                outside.append((step, allocation))
                continue
            sha = refs.get((step, allocation))
            if sha is None:
                sys.exit(f'refusing: no reference hash for step {step} allocation {allocation}')
            ranges.sort(key=lambda r: r['offset'])
            jobs.append((step, allocation, ranges, sha, str(payloads / f'{sha}.gz')))
    # lean store: the recorder's own residue digest (BoundaryMaskedDigest) stands for the kept payload, which a lean
    # recording does not keep. Where both exist (a full recording) they must agree, or nothing is sealed.
    digests = masked_digests(trace)
    bases = {a['id']: a['recorded_base'] for a in catalog['allocations']}
    lines = [digests.get((step, bases[allocation], by_bytes[allocation])) for step, allocation, *_ in jobs]
    # Breakage: these payload loads (gzip + sha256 of every carried TEMPO allocation at every step end) ran one
    # after another on one core; they are independent, so they run on all cores. Entries are unchanged (same order, same masked hashes).
    # peak per job: the inflated payload and its masked copy (bytearray), each one allocation long
    loads = [j for j in jobs if Path(j[4]).exists()]
    peak = max((by_bytes.get(j[1], 0) for j in loads), default=0)
    with ProcessPoolExecutor(max_workers=pool_workers(len(loads), 2 * peak)) as pool:
        loaded = dict(zip((j[4] for j in loads), pool.map(masked_payload, loads))) if loads else {}
    results = [recorded_masked(job, line, loaded.get(job[4])) for job, line in zip(jobs, lines)]
    stats = collections.Counter(how for _, _, how in results)
    print(json.dumps({'event': 'p1fix_residue_sources', **stats}), file=sys.stderr, flush=True)
    for (step, allocation, ranges, _sha, _path), (error, masked, _how) in zip(jobs, results):
        if error:
            sys.exit(error)
        entries.append({'allocation': allocation, 'step': step, 'ranges': ranges, 'masked_sha256': masked,
                        'reason': RESIDUE_REASON, 'witness': witness_note})
    return entries, outside


def recorded_masked(job, line, loaded):
    """(refusal or None, masked sha256, source) of one residue entry, from the kept payload (LOADED: masked_payload's
    answer) and/or the recorder's BoundaryMaskedDigest LINE. The line counts only when it is of the same bytes (its
    sha256 is the reference hash) and of the same ranges (offset and length of each, in order)."""
    step, allocation, ranges, sha, path = job
    for a, b in zip(ranges, ranges[1:]):
        if a['offset'] + a['bytes'] > b['offset']:
            return f'refusing: overlapping TEMPO arrays in allocation {allocation} at step {step}', None, 'refused'
    if line is not None:
        if line['sha256'] != sha:
            return (f'refusing: the recorder digested allocation {allocation} at step {step} as {line["sha256"]}, '
                    f'the reference says {sha}'), None, 'refused'
        if [list(r) for r in line['ranges']] != [[r['offset'], r['bytes']] for r in ranges]:
            return (f'refusing: the recorder masked other ranges of allocation {allocation} at step {step} '
                    f'({line["ranges"][:4]} against {[[r["offset"], r["bytes"]] for r in ranges][:4]})'), None, 'refused'
    if loaded is not None:
        error, masked = loaded
        if error or line is None:
            return error, masked, 'payload'
        if masked != line['masked_sha256']:
            return (f'refusing: allocation {allocation} at step {step}: the kept payload masks to {masked}, the '
                    f'recorder\'s residue digest is {line["masked_sha256"]}'), None, 'refused'
        return None, masked, 'payload=recorder'
    if line is None:
        return (f'refusing: forward boundary bytes {path} missing and no recorder residue digest for allocation '
                f'{allocation} at step {step} (record with a recorder that writes BoundaryMaskedDigest, or '
                'WOOF_HEX_RECORD_STORE=full)'), None, 'refused'
    for r in ranges:
        if r['offset'] + r['bytes'] > line['bytes']:
            return f'refusing: TEMPO {r["array"]} runs past allocation {allocation}', None, 'refused'
    return None, line['masked_sha256'], 'recorder'


def masked_payload(job):
    """(refusal or None, masked sha256) of one forward boundary payload with its TEMPO ranges zeroed."""
    step, allocation, ranges, sha, path = job
    path = Path(path)
    if not path.exists():
        return f'refusing: forward boundary bytes {path} missing (record with BINDER_BOUNDARY_PAYLOADS=1)', None
    data = bytearray(gzip.open(path).read())
    if hashlib.sha256(data).hexdigest() != sha:
        return f'refusing: boundary payload {path} does not hash to its name', None
    for a, b in zip(ranges, ranges[1:]):
        if a['offset'] + a['bytes'] > b['offset']:
            return f'refusing: overlapping TEMPO arrays in allocation {allocation} at step {step}', None
    for r in ranges:
        if r['offset'] + r['bytes'] > len(data):
            return f'refusing: TEMPO {r["array"]} runs past allocation {allocation}', None
        data[r['offset']:r['offset'] + r['bytes']] = bytes(r['bytes'])
    return None, hashlib.sha256(data).hexdigest()


def main():
    pkg, catalog_path, trace = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    observer = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    observer_b = Path(sys.argv[5]) if len(sys.argv) > 5 else None
    stage('start')
    plan = json.load(open(pkg / 'plan.json'))
    stage('plan loaded')
    catalog = json.load(open(catalog_path))
    by_id = {a['id']: a for a in catalog['allocations']}
    report = {}
    stage('catalog loaded')
    plan['module_initializers'] = initializers(trace, plan)
    stage('module initializers')
    report['module_initializers'] = [i['function'] for i in plan['module_initializers']]
    ids = plan_ids(plan)
    missing = [i for i, n in ids.items() if i not in by_id or by_id[i]['bytes'] != n]
    if missing:
        sys.exit(f'refusing: plan allocations without a matching recording-A catalog entry: {missing[:10]}')
    plan['recorded_bases'] = [[i, by_id[i]['recorded_base']] for i in sorted(ids)]
    report['recorded_bases'] = len(ids)
    first = boundary_clock(trace, 1)
    stage('boundary clocks')
    rows, starts = live_ranges(catalog, first)
    words = 0
    # G2: one initial-state allocation per pool worker (inflate + vectorised scan), results in plan order. THE BREAKAGE
    # THIS PREVENTS: the serial loop inflated every initial-state payload on one core before any pool started (the
    # 15-min pre-pool stretch at 100 % of one core, a recording host 2026-11-04 20:15Z).
    jobs = []
    for a in plan['allocations']:
        a.pop('relocations', None)
        ini = a.get('initial')
        if ini and ini.get('encoding') == 'gzip':
            jobs.append((a, str(pkg / ini['path'])))
    peak = max((a['bytes'] for a, _ in jobs), default=0)
    with ProcessPoolExecutor(max_workers=pool_workers(len(jobs), 3 * peak)) as pool:
        found = pool.map(_relocation_job, [(path, rows, starts) for _, path in jobs])
        for (a, _), hits in zip(jobs, found):
            relocations = [{'byte_offset': off, 'target': {'allocation': ident, 'offset': inside}, 'recorded_value': value}
                           for off, ident, inside, value in hits]
            if relocations:
                a['relocations'] = relocations
                words += len(relocations)
    # pk120 (x1.40962): a word is a recorded device address of recording A only if it is not data. A value that lies
    # in no allocation live in recording B (B's layout is another one) and that recording B's initial state also holds,
    # as an 8-byte word at either 4-byte alignment, is data: B's process never had that address, so its forward wrote
    # the value as numbers. P1FIX_CATALOG_B / P1FIX_DRIVER_B / P1FIX_RECORDER_B (ad-record passes recording B); unset
    # keeps the A-only rule. Identical A/B layouts (record-mem031) keep every candidate (each lies in a B range).
    # THE BREAKAGE THIS PREVENTS: on x1.40962 (40,962 cells, 122,880 edges) two adjacent int32 cell indices of
    # cells_on_edge and of its physics-pool copies (4,895 and 32,602 = 0x7f5a) made the 8-byte word 0x7f5a0000131f,
    # inside recording A's allocation 359 (base 0x7f5a00000000), and the replay rewrote it to the replay's address of
    # 359: pkg120 (18 steps) died at step 1 (CUDA 700/717, moist_edge_coefficients_v841_f32 / project_edges_v841_f32,
    # the kernels that read those cell indices). Index values never reach 0x7f00 on x1.2562.
    cat_b, drv_b, rec_b = (os.environ.get(k) for k in ('P1FIX_CATALOG_B', 'P1FIX_DRIVER_B', 'P1FIX_RECORDER_B'))
    if cat_b and drv_b and rec_b:
        catalog_b = json.load(open(cat_b))
        rows_b, starts_b = live_ranges(catalog_b, boundary_clock(drv_b, 1))
        cand = sorted({r['recorded_value'] for a in plan['allocations'] for r in a.get('relocations', [])
                       if locate(rows_b, starts_b, r['recorded_value']) is None})
        paths = sorted({f"{rec_b}/payloads/{a['initial_sha256']}.gz" for a in catalog_b['allocations'] if a.get('initial_sha256')})
        found = set()
        if cand:
            with ProcessPoolExecutor(max_workers=pool_workers(len(paths), 3 * max((a['bytes'] for a in catalog_b['allocations']), default=0))) as pool:
                for hit in pool.map(_values_in_payload, [(path, cand) for path in paths]):
                    found.update(hit)
        dropped = []
        for a in plan['allocations']:
            rel = a.get('relocations')
            if not rel:
                continue
            keep = [r for r in rel if r['recorded_value'] not in found]
            if len(keep) != len(rel):
                dropped.append([a['id'], len(rel) - len(keep)])
                words -= len(rel) - len(keep)
                if keep:
                    a['relocations'] = keep
                else:
                    a.pop('relocations')
        report['relocations_dropped_as_data_by_b'] = {'candidates_outside_b': len(cand), 'values_in_b_state': len(found),
                                                      'allocations': dropped}
    report['relocated_words'] = words
    stage('initial relocations')
    report['split_held_words'], report['split_held_unresolved'] = split_words(plan, pkg, by_id, trace)
    stage('split words')
    report['held_pointer_words'], targets = held_pointers(plan, pkg, by_id)
    stage('held pointers')
    report['held_pointer_targets'] = len(targets)
    plan.pop('excluded_residue', None)
    if observer is not None:
        if observer_b is None:
            sys.exit('refusing: excluded residue needs a second forward (OBSERVER_B) for its witness')
        from residue_witness import witness as make_witness
        w = make_witness(observer, observer_b, pkg / 'residue-witness.json')
        entries, outside = residue(plan, catalog, trace, observer, pkg, w)
        plan['excluded_residue'] = entries
        report['excluded_residue'] = {'entries': len(entries), 'allocations': sorted({e['allocation'] for e in entries}),
                                      'bytes_per_step': {e['step']: 0 for e in entries}, 'not_carried': outside,
                                      'witness': w['summary']}
        for e in entries:
            report['excluded_residue']['bytes_per_step'][e['step']] += sum(r['bytes'] for r in e['ranges'])
    stage('residue')
    json.dump(plan, open(pkg / 'plan.json', 'w'))
    stage('plan written')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
