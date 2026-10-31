"""phasediff.py census.json ncells module-substring [top]: what is live at that module's peak beyond the exit resident set."""
import json, sys, collections
r = json.load(open(sys.argv[1])); n = int(sys.argv[2]); key = sys.argv[3]; top = int(sys.argv[4]) if len(sys.argv) > 4 else 40
d = next(iter(r['devices'].values())); M = 2**20
res = collections.Counter()
for x in d['live_now_sites']: res[(x['site'], x['detail'])] += x['bytes']
for m, v in d['module_peaks'].items():
    if key not in m: continue
    ph = collections.Counter()
    for x in v['sites']: ph[(x['site'], x['detail'])] += x['bytes']
    extra = {k: b - res.get(k, 0) for k, b in ph.items() if b - res.get(k, 0) > 0}
    gone = sum(b for k, b in res.items() if k not in ph)
    print(f"== {m}: live {v['live_bytes']/M:.0f} MiB at t={v['t_s']}s; beyond exit-resident {sum(extra.values())/M:.0f} MiB ({sum(extra.values())/n:.0f} B/cell); resident not live then {gone/M:.0f} MiB")
    bym = collections.Counter()
    for k, b in extra.items(): bym[k[0].split(' < ')[0].split(':')[0]] += b
    for k, b in bym.most_common(12): print(f"   {b/M:8.1f} MiB {b/n:7.0f} B/cell  {k}")
    for k, b in sorted(extra.items(), key=lambda kv: -kv[1])[:top]:
        print(f"   {b/M:8.1f} MiB {b/n:7.0f} B/cell  {k[0][:170]} [{k[1]}]")
