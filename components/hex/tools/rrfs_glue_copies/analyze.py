"""Attribute the copy/add family to its launch-time NVTX stack in plain steps 2-4."""
import collections
import json
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
steps = set(map(int, sys.argv[3].split(','))) if len(sys.argv) > 3 else {2, 3, 4}
names = dict(db.execute('select id,value from StringIds'))
ranges = []
for s, e, text, tid, gid in db.execute(
        'select start,end,text,textId,globalTid from NVTX_EVENTS where end>start'):
    label = text or names.get(tid, '')
    if label.startswith(('step:', 'py:', 'c:', 'copy:', 'plan:', 'add:')):
        ranges.append((s, e, label, gid))
ranges.sort()
apis = []
for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
    try:
        apis += list(db.execute(f'select start,correlationId,globalTid from {table}'))
    except sqlite3.OperationalError:
        pass
stack, paths, i = [], {}, 0
for t, corr, tid in sorted(apis):
    while i < len(ranges) and ranges[i][0] <= t:
        stack.append(ranges[i]); i += 1
    stack = [r for r in stack if r[1] > t]
    path = [r[2] for r in stack if r[3] == tid]
    step = next((int(p.split(':')[1]) for p in reversed(path) if p.startswith('step:')), 0)
    if step in steps:
        paths[corr] = (step, path)
totals = collections.defaultdict(lambda: [0, 0.])
for s, e, n, corr in db.execute(
        'select start,end,shortName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
    name = names.get(n, '')
    if corr not in paths or not any(k in name for k in ('cupy_copy', 'rrfs_glue_copy_plan', 'add_inplace')):
        continue
    step, path = paths[corr]
    label = next((p for p in reversed(path) if p.startswith(('copy:', 'plan:', 'add:'))), path[-1])
    totals[name, label][0] += 1
    totals[name, label][1] += (e-s)/1e6
out = [dict(kernel=k[0], site=k[1], launches_per_step=v[0]/len(steps), profile_ms_per_step=v[1]/len(steps))
       for k, v in totals.items()]
with open(sys.argv[2], 'w') as stream:
    json.dump(sorted(out, key=lambda x: -x['profile_ms_per_step']), stream, indent=2)
print(json.dumps(dict(profile_ms_per_step=sum(x['profile_ms_per_step'] for x in out),
                      launches_per_step=sum(x['launches_per_step'] for x in out))))
