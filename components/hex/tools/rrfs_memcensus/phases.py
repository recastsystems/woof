"""phases.py census.json ncells: per-module local peaks (what is live when each module peaks), resident set at exit."""
import json, sys, collections
r = json.load(open(sys.argv[1])); n = int(sys.argv[2]); d = next(iter(r['devices'].values())); M = 2**20
print(f"peak live {d['peak_live_bytes']/M:.0f} MiB pool {d['snapshot_pool_total_bytes']/M:.0f} used {d['snapshot_device_used_bytes']/M:.0f}  live at exit {d['live_now_bytes']/M:.0f}")
def bymod(sites):
    c = collections.Counter()
    for x in sites: c[x['site'].split(' < ')[0].split(':')[0]] += x['bytes']
    return c
print('== module peaks (total live when that module last raised its own high-water)')
for m, v in sorted(d['module_peaks'].items(), key=lambda kv: -kv[1]['live_bytes'])[:12]:
    c = bymod(v['sites'])
    top = ', '.join(f"{k.split('/')[-1]} {b/M:.0f}" for k, b in c.most_common(7))
    print(f"{v['live_bytes']/M:8.0f} MiB t={v['t_s']:6.1f}s  {m}: {top}")
print('== live at exit by module')
for k, b in bymod(d['live_now_sites']).most_common(25):
    print(f"{b/M:8.1f} MiB {b/n:8.0f} B/cell  {k}")
print('== site maxima (largest ever live per site)')
for x in d['site_max'][:30]:
    print(f"{x['bytes']/M:8.1f} MiB {x['bytes']/n:8.0f} B/cell  {x['site'].split(' < ')[0]} [{x['detail']}]")
