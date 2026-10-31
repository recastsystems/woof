"""fit.py A.json B.json cellsA cellsB [top]: per-site fixed + per-cell fit from two census snapshots at peak."""
import json, sys, collections
def load(p):
    r = json.load(open(p)); d = next(iter(r['devices'].values()))
    s = collections.Counter()
    for x in d['sites']:
        s[(x['site'].split(' < ')[0], x['detail'])] += x['bytes']
    return d, s
dA, sA = load(sys.argv[1]); dB, sB = load(sys.argv[2])
nA, nB = int(sys.argv[3]), int(sys.argv[4]); top = int(sys.argv[5]) if len(sys.argv) > 5 else 60
M = 2**20
for tag, d, n in (('A', dA, nA), ('B', dB, nB)):
    print(f"{tag} cells {n}: peak live {d['peak_live_bytes']/M:.0f} MiB, pool {d['snapshot_pool_total_bytes']/M:.0f}, device used {d['snapshot_device_used_bytes']/M:.0f}, non-pool {(d['snapshot_device_used_bytes']-d['snapshot_pool_total_bytes'])/M:.0f}, pool cache {(d['snapshot_pool_total_bytes']-d['snapshot_live_bytes'])/M:.0f}")
def fit(a, b):
    per = (b - a) / (nB - nA); return a - per * nA, per
for name, a, b in (('live', dA['snapshot_live_bytes'], dB['snapshot_live_bytes']),
                   ('pool', dA['snapshot_pool_total_bytes'], dB['snapshot_pool_total_bytes']),
                   ('device', dA['snapshot_device_used_bytes'], dB['snapshot_device_used_bytes'])):
    f, p = fit(a, b); print(f"fit {name}: fixed {f/M:.0f} MiB + {p:.0f} B/cell")
keys = set(sA) | set(sB)
rows = []
for k in keys:
    f, p = fit(sA.get(k, 0), sB.get(k, 0)); rows.append((p, f, k))
rows.sort(key=lambda r: -max(r[0], 0))
print(f"{'B/cell':>8} {'fixed MiB':>9}  site [detail]")
for p, f, k in rows[:top]:
    print(f"{p:8.0f} {f/M:9.1f}  {k[0]} [{k[1]}]")
mod = collections.defaultdict(lambda: [0.0, 0.0])
for p, f, k in rows:
    m = k[0].split(':')[0]; mod[m][0] += p; mod[m][1] += f
print('--- by module')
for m, (p, f) in sorted(mod.items(), key=lambda kv: -kv[1][0]):
    print(f"{p:8.0f} {f/M:9.1f}  {m}")
print('--- fixed big (> 50 MiB)')
for p, f, k in sorted(rows, key=lambda r: -r[1])[:15]:
    print(f"{p:8.0f} {f/M:9.1f}  {k[0]} [{k[1]}]")
