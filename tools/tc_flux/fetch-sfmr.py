"""Fetch released NOAA SFMR windows with source flags and immutable receipts.

Input JSON: {"cases": [{"storm": "...", "start": "...Z", "end": "...Z"}]}.
This is public observation retrieval, not a DA reader or a matchup operator.
Dataset license: https://erddap.aoml.noaa.gov/hdb/erddap/info/sfmr_2018_2022/index.html
Free use and redistribution; source quality flags remain in every row.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = 'https://erddap.aoml.noaa.gov/hdb/erddap/'
DATASET = 'sfmr_2018_2022'
FIELDS = 'tropical_cyclone_name,flight_id,time,latitude,longitude,SWS,SRR,FLAG,NGC,RALT,RANG,PANG'


def fetch(case, cache):
    # Source metadata gives this table's released date range. Refuse attempts
    # to turn its absence after 2022 into an empty successful observation set.
    if not ('2018-05-25' <= case['start'][:10] <= case['end'][:10] <= '2022-11-10'):
        raise ValueError('window outside the released SFMR table coverage')
    query = (FIELDS + '&tropical_cyclone_name=' + json.dumps(case['storm'])
             + '&time>=' + case['start'] + '&time<=' + case['end'])
    url = ROOT + 'tabledap/' + DATASET + '.csv?' + urllib.parse.quote(query, safe=',&=():-')
    identity = hashlib.sha256(url.encode()).hexdigest()[:20]
    out = cache / case['start'][:10] / (identity + '.csv')
    receipt = out.with_suffix('.json')
    if receipt.exists():
        row = json.loads(receipt.read_text())
        if row['url'] != url or hashlib.sha256(out.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('cached observation receipt does not match data')
        return row
    if out.exists():
        raise FileExistsError(f'unreceipted cache entry: {out}')
    began = time.monotonic()
    with urllib.request.urlopen(url, timeout=180) as response:
        data = response.read()
    rows = csv.DictReader(io.StringIO(data.decode()))
    units = next(rows)
    if units['SWS'] != 'm/s' or units['time'] != 'UTC':
        raise ValueError(f'unexpected source units: {units}')
    count, flags, flights = 0, {}, set()
    for row in rows:
        if row['tropical_cyclone_name'] != case['storm']:
            raise ValueError('server returned another storm')
        count += 1
        flags[row['FLAG']] = flags.get(row['FLAG'], 0) + 1
        flights.add(row['flight_id'])
    if not count:
        raise ValueError('source returned no observations')
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix('.csv.part')
    part.write_bytes(data)
    part.replace(out)
    result = dict(case=case, path=str(out.resolve()), url=url, bytes=len(data), rows=count,
        sha256=hashlib.sha256(data).hexdigest(), units=units, flags=flags,
        flights=sorted(flights), wall_seconds=time.monotonic()-began,
        license='NOAA source metadata: free use and redistribution',
        license_url=ROOT+'info/'+DATASET+'/index.html',
        status='retrieved; flight audit and model matchup pending')
    part = receipt.with_suffix('.json.part')
    part.write_text(json.dumps(result, indent=2)+'\n')
    part.replace(receipt)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    cases = json.loads(a.manifest.read_text())['cases']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda case: fetch(case, a.cache), cases))
    a.receipt.write_text(json.dumps(dict(cases=results), indent=2)+'\n')
    print(json.dumps([dict(rows=r['rows'], bytes=r['bytes'], flags=r['flags']) for r in results]))


if __name__ == '__main__':
    main()
