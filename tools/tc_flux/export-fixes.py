"""Translate physical forecast fixes to the shared verification input contract."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--format', choices=['native', 'arwen'], required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--init', required=True, help='ISO UTC initialization')
p.add_argument('--hours', type=int, required=True)
p.add_argument('--cadence-hours', type=int, default=1)
p.add_argument('--arm', help='WOOF product prefix')
p.add_argument('--domain', type=int, default=3)
p.add_argument('--allow-partial', action='store_true')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
init = datetime.fromisoformat(a.init.replace('Z', '+00:00'))
if init.tzinfo is None or init.utcoffset() != timedelta(0):
    p.error('initialization must carry a UTC offset')
if a.hours <= 0 or a.cadence_hours <= 0 or a.hours % a.cadence_hours:
    p.error('positive duration must be divisible by positive saved cadence')
rows = []
if a.format == 'native':
    for row in json.loads(a.source.read_text())['frames']:
        seconds, fix = float(row['model_seconds']), row['fix']
        if seconds % (a.cadence_hours * 3600) or not fix.get('ok'):
            continue
        rows.append((seconds, fix['lat'], fix['lon'], fix['vmax10_ms'], fix['mslp_min_hpa']))
else:
    if not a.arm:
        p.error('--arm is required for WOOF')
    for path in sorted(a.source.glob(f'{a.arm}-d{a.domain:02}-*.json')):
        row = json.loads(path.read_text())
        stamp = row['valid_time']
        # WRF history/ready markers use this documented UTC filename stamp.
        # Do not silently assign UTC to arbitrary offset-free ISO strings.
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}', stamp):
            valid = datetime.strptime(stamp, '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
        else:
            valid = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
        if valid.tzinfo is None:
            raise ValueError('forecast valid_time is missing its UTC offset')
        seconds = (valid - init).total_seconds()
        rows.append((seconds, row['latitude'], row['longitude'], row['maximum_wind150_m_s'], row['minimum_mslp_hpa']))
fixes = {}
for seconds, lat, lon, wind, pressure in rows:
    if not 0 <= seconds <= a.hours * 3600 or seconds % (a.cadence_hours * 3600):
        continue
    if not all(math.isfinite(float(v)) for v in (lat, lon, wind, pressure)):
        raise ValueError(f'nonfinite forecast fix at {seconds}')
    fix = dict(valid=(init + timedelta(seconds=seconds)).isoformat().replace('+00:00', 'Z'),
               lat=lat, lon=lon, vmax_kt=wind * 3600 / 1852, mslp_hpa=pressure)
    if seconds in fixes and fixes[seconds] != fix:
        raise ValueError(f'conflicting segment-boundary fixes at {seconds}')
    fixes[seconds] = fix
expected = set(range(0, a.hours * 3600 + 1, a.cadence_hours * 3600))
missing = sorted(expected - fixes.keys())
if not fixes or (missing and not a.allow_partial):
    raise ValueError(f'incomplete forecast fixes: missing seconds {missing}')
a.out.write_text(json.dumps(dict(fixes=[fixes[t] for t in sorted(fixes)],
    provenance=dict(source=str(a.source), format=a.format, partial=bool(missing),
                    requested_hours=a.hours, cadence_hours=a.cadence_hours,
                    missing_seconds=missing, intensity='10 m model wind; instantaneous, not a 1-minute average')),
    indent=2) + '\n')
