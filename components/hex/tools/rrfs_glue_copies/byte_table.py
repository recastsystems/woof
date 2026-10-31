"""Join measured launch attribution to metadata-only logical-byte records."""
import collections
import json
from pathlib import Path
import sys

records = json.loads(Path(sys.argv[1]).read_text())['records']
sites = json.loads(Path(sys.argv[2]).read_text())
levels = int(sys.argv[4]) if len(sys.argv) > 4 else 59
steps = set(map(int, sys.argv[5].split(','))) if len(sys.argv) > 5 else {2, 3, 4}
if levels != 59:
    grouped = {}
    for row in sites:
        group = grouped.setdefault(row['site'], dict(site=row['site'], kernel=[], launches_per_step=0,
                                                     profile_ms_per_step=0))
        group['kernel'].append(row['kernel'])
        group['launches_per_step'] += row['launches_per_step']
        group['profile_ms_per_step'] += row['profile_ms_per_step']
    sites = [dict(r, kernel=', '.join(r['kernel'])) for r in grouped.values()]
totals = collections.Counter()
scaled = collections.Counter()
for row in records:
    if row['step'] in steps:
        size = row['bytes'] * row['calls'] / len(steps)
        factor = (59 / levels if levels in row['shape'] else
                  60 / (levels+1) if levels+1 in row['shape'] else 1)
        totals[row['kind'], row['site']] += size
        scaled[row['kind'], row['site']] += size * 206726 / 2562 * factor
out = []
for row in sites:
    site = row['site']
    note = 'metadata'
    size = None
    large_size = None
    if site.startswith('plan:'):
        key = site[5:]
        size = totals['plan-copy', key] + totals['plan-fill', key]
        large_size = scaled['plan-copy', key] + scaled['plan-fill', key]
    elif site.startswith('add:'):
        size = totals['add', site[4:]]
        large_size = scaled['add', site[4:]]
    elif site.startswith('copy:'):
        _, kind, key = site.split(':', 2)
        size = totals[kind, key]
        large_size = scaled[kind, key]
        if row['kernel'] in ('cupy_copy__float_float32', 'cupy_copy__int_int32'):
            size /= 2
            large_size /= 2
            note = 'scalar broadcast; destination writes only'
        if 'copy_where' in row['kernel']:
            note = 'upper bound; excludes mask read and predicate-dependent stores'
    elif site.startswith(('c:copy:', 'c:reshape:')):
        _, kind, key = site.split(':', 2)
        size = totals['ndarray.' + kind, key]
        large_size = scaled['ndarray.' + kind, key]
    elif site.startswith('py:mpasphys/resident_output_diagnostics_v842.py:116'):
        size = totals['implicit-current', site[3:]]
        large_size = scaled['implicit-current', site[3:]]
    elif 'gf_mpas_v842.py:24930' in site:
        size = 5 * 2562 * row['launches_per_step']
        note = 'source-derived: three bool to int32 cumsum inputs, gf_mpas_v842.py:24930'
    elif site.startswith('c:astype:mpasphys/gf_mpas_v842.py:24932'):
        size = totals['ndarray.astype', site[len('c:astype:'):]]
        large_size = scaled['ndarray.astype', site[len('c:astype:'):]]
    elif 'deferred_checks_v842.py:' in site and 'int32_int64' in row['kernel']:
        size = (7 if 'bool_int8' in row['kernel'] else 12) * row['launches_per_step']
        note = 'source-derived: int32 to int64 flags (12 bytes); bool to int8 flag (2 bytes), deferred_checks_v842.py:72,98-100'
    elif 'resident_output_diagnostics_v842.py:210' in site:
        size = 8 * levels * 5120
        large_size = 8 * 59 * 5120 * 206726 / 2562
        note = 'source-derived: flush line229, cuda_horizontal_v841.py:1016; x1.2562 vertices=5120'
    elif 'resident_pbl_rrfs.py:311' in site and 'bool_int32' in row['kernel']:
        size = 5 * 2562
        note = 'source-derived: bool to int32; bl_mynnedmf_c6d08709.py:11545'
    out.append(dict(row, bytes_per_step=size,
                    scaled_bytes_per_step=(large_size if large_size is not None else
                                           None if size is None else size * 206726 / 2562),
                    byte_evidence=note))
Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + '\n')
print('| Kernel / call site | Launches/step | Bytes/step | Scaled bytes/step |')
print('|---|---:|---:|---:|')
for row in out:
    label = row['site']
    size = row['bytes_per_step']
    print(f"| {row['kernel']} / `{label}` | {row['launches_per_step']:g} | "
          + ('unknown | unknown |' if size is None else
             f"{size:,.0f} | {row['scaled_bytes_per_step']:,.0f} |"))
