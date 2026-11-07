"""Audit released SFMR inventory against explicit, source-linked flight windows.

This preserves source rows and reports eligibility, never forecast skill.
Unknown flights and known calibration exclusions cannot enter the primary set.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def audit(case, flights):
    path = Path(case['path'])
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != case['sha256']:
            raise ValueError('source SHA256 changed')
    counts, detail = Counter(), {}
    with path.open() as stream:
        rows = csv.DictReader(stream)
        units = next(rows)
        if units['SWS'] != 'm/s' or units['time'] != 'UTC':
            raise ValueError('unrecognized observation units')
        for row in rows:
            fid = row['flight_id']
            flight = flights.get(fid)
            stats = detail.setdefault(fid, Counter())
            reason = 'eligible'
            if row['FLAG'] != '0':
                reason = 'source_flag_' + row['FLAG']
            elif flight is None:
                reason = 'uncatalogued_flight'
            elif flight['storm'] != row['tropical_cyclone_name']:
                raise ValueError(f'flight storm mismatch: {fid}')
            elif not flight['primary']:
                reason = 'flight_excluded'
            else:
                stamp = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
                bounds = [datetime.fromisoformat(flight[k].replace('Z', '+00:00')) for k in ('start', 'end')]
                values = [float(row[k]) for k in ('latitude', 'longitude', 'SWS')]
                if not bounds[0] <= stamp <= bounds[1]:
                    reason = 'outside_flight_window'
                elif not all(math.isfinite(v) for v in values):
                    reason = 'missing_position_or_wind'
                elif not (-90 <= values[0] <= 90 and -180 <= values[1] <= 180 and values[2] >= 0):
                    reason = 'invalid_position_or_wind'
            counts[reason] += 1
            stats[reason] += 1
    return dict(case=case['case'], source_sha256=case['sha256'], counts=counts,
                flights=detail, status='source eligibility only; model matchup pending')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retrieval', type=Path, required=True)
    p.add_argument('--flights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    flights = json.loads(a.flights.read_text())['flights']
    cases = json.loads(a.retrieval.read_text())['cases']
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda case: audit(case, flights), cases))
    result = dict(cases=rows, flight_manifest_sha256=hashlib.sha256(a.flights.read_bytes()).hexdigest())
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([dict(case=r['case'], counts=r['counts']) for r in rows]))
