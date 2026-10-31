"""estimate.py A.json nA B.json nB [N ...]: per-phase rigid fit (fixed + per cell) and the device need at N local columns.

Elastic sites (sized to free memory, so they shrink to fit): RRTMG SW/LW batch transients and the MYNN-EDMF arena
(min(2 GiB, free/4)). They are reported apart, not extrapolated. Non-pool memory (CUDA context, kernel images) is
taken as measured on an unshared card (base run: 1,383-1,422 MiB)."""
import json, sys, collections, re
M = 2**20
ELASTIC = [r'rrtmg_sw_v842\.py:(690|474|594|606):', r'rrtmg_lw_v842\.py:', r'bl_mynnedmf_c942418\.py:\d+:run\b.*\[name=emis_ant_no\]$']
def elastic(site, detail):
    s = site.split(' < ')[0] + ' [' + detail + ']'
    return any(re.search(p, s) for p in ELASTIC)
def load(p):
    d = next(iter(json.load(open(p))['devices'].values()))
    out = {}
    for m, v in d['module_peaks'].items():
        c = collections.Counter(); e = 0
        for x in v['sites']:
            if elastic(x['site'], x['detail']): e += x['bytes']
            else: c[(x['site'].split(' < ')[0], x['detail'])] += x['bytes']
        out[m] = (c, e, v['live_bytes'])
    res = collections.Counter()
    for x in d['live_now_sites']: res[(x['site'].split(' < ')[0], x['detail'])] += x['bytes']
    return out, res
A, nA, B, nB = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
Ns = [int(x) for x in sys.argv[5:]] or [298826]
pa, ra = load(A); pb, rb = load(B)
NONPOOL = 1422 * M
rows = []
for m in set(pa) & set(pb):
    ca, ea, la = pa[m]; cb, eb, lb = pb[m]
    fa, fb = sum(ca.values()), sum(cb.values())
    per = (fb - fa) / (nB - nA); fix = fa - per * nA
    rows.append((per, fix, m, ea, eb, la, lb))
rows.sort(key=lambda r: -(r[1] + r[0] * Ns[0]))
print(f"{'phase':58s} {'rigid B/cell':>12s} {'rigid fixed MiB':>15s} {'elastic MiB A/B':>16s}  " + "  ".join(f"need@{N} GiB" for N in Ns))
for per, fix, m, ea, eb, la, lb in rows[:10]:
    print(f"{m[-58:]:58s} {per:12.0f} {fix/M:15.0f} {ea/M:7.0f}/{eb/M:<7.0f}  " + "  ".join(f"{(fix + per*N + NONPOOL)/2**30:12.1f}" for N in Ns))
per = (sum(rb.values()) - sum(ra.values())) / (nB - nA); fix = sum(ra.values()) - per * nA
print(f"resident at exit: {per:.0f} B/cell + {fix/M:.0f} MiB fixed")
