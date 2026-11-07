"""Print the branch coverage of a KF-CuP capture: per config, how many
columns are held, take the standard-KF fallback, go deep, go shallow, or
stay clear, and how many rain.  Read the capture, do not trust the table."""
import csv, struct, sys
from collections import Counter

def f(h):
    return struct.unpack('>f', bytes.fromhex(h))[0]

root = sys.argv[1] if len(sys.argv) > 1 else '.'
names = {}
for r in csv.DictReader(open(f'{root}/kfcup-in-surface.csv')):
    names[(r['cfg'], r['case'])] = (r['name'], f(r['nca']))
cfgdt = {r['cfg']: f(r['dt']) for r in csv.DictReader(open(f'{root}/kfcup-config.csv'))}
rows = list(csv.DictReader(open(f'{root}/kfcup-surface.csv')))
per = {}
for r in rows:
    name, nca0 = names[(r['cfg'], r['case'])]
    dt = cfgdt[r['cfg']]
    if nca0 >= 0.5 * dt:
        kind = 'held'
    elif r['cupflag'] == '0':
        kind = 'kf_fallback_rain' if f(r['raincv']) > 0 else 'kf_fallback_dry'
    else:
        kind = {0.0: 'deep', 1.0: 'shallow', 2.0: 'clear'}[f(r['shall'])]
    per.setdefault(r['cfg'], Counter())[kind] += 1
    per[r['cfg']]['rain>0'] += f(r['raincv']) > 0
for cfg, c in sorted(per.items()):
    print(cfg, dict(c))
if '-v' in sys.argv:
    for r in rows:
        name, nca0 = names[(r['cfg'], r['case'])]
        print(r['cfg'], r['case'], name, 'cupflag', r['cupflag'], 'shall', f(r['shall']),
              'rain', f(r['raincv']), 'nca', f(r['nca']), 'bot/top', f(r['cubot']), f(r['cutop']),
              'act', f(r['activefrac']))
