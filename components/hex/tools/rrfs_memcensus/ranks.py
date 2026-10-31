"""1-rank vs N-rank census (per allocating thread). Bytes per column of each file:function at each thread's own peak."""
import json, sys, collections, re
def group(rows):
    g = collections.Counter()
    for r in rows:
        inner = r['site'].split(' < ')[0]; f, ln, fn = inner.split(':')[:3]; g[f'{f}:{fn}'] += r['bytes']
    return g
one = json.load(open(sys.argv[1]))['devices']; many = json.load(open(sys.argv[2]))['devices']
n1 = int(sys.argv[3]); nN = int(sys.argv[4]); R = int(sys.argv[5])
print('1-rank threads (peak MiB, live at exit MiB):', {k: (v['peak_live_bytes']>>20, v['live_now_bytes']>>20) for k, v in one.items() if v['peak_live_bytes'] > 1<<26})
print('N-rank threads:', {k: (v['peak_live_bytes']>>20, v['live_now_bytes']>>20) for k, v in many.items() if v['peak_live_bytes'] > 1<<26})
g1 = collections.Counter(); [g1.update(group(v['sites'])) for v in one.values()]
gN = collections.Counter(); [gN.update(group(v['sites'])) for v in many.values()]
t1 = sum(g1.values()); tN = sum(gN.values())
print(f'sum of thread peaks: 1 rank {t1/2**20:.0f} MiB = {t1/n1:.0f} B/col; {R} ranks {tN/2**20:.0f} MiB = {tN/(R*nN):.0f} B/local col ({tN/n1:.0f} B per global col)')
rows = []
for k in set(g1) | set(gN):
    a = g1.get(k, 0); b = gN.get(k, 0)
    rows.append((b - a, k, a, b))
print('--- grows with ranks: MiB 1 rank | MiB sum of N ranks | delta MiB | B/col 1r | B/local col Nr')
for d_, k, a, b in sorted(rows, reverse=True)[:int(sys.argv[6]) if len(sys.argv) > 6 else 30]:
    print(f'{a/2**20:8.0f} {b/2**20:8.0f} {d_/2**20:+8.0f}  {a/n1:7.0f} {b/(R*nN):7.0f}  {k}')
print('--- shrinks')
for d_, k, a, b in sorted(rows)[:10]:
    print(f'{a/2**20:8.0f} {b/2**20:8.0f} {d_/2**20:+8.0f}  {a/n1:7.0f} {b/(R*nN):7.0f}  {k}')
