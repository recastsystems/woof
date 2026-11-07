"""Audit Rust reports against independently produced MET-TC tc_pairs records."""
import argparse
import datetime as dt
import json
from pathlib import Path


def stamp(s):
    return int(dt.datetime.strptime(s, '%Y%m%d_%H%M%S').replace(tzinfo=dt.timezone.utc).timestamp())


def check(report_path, met_path):
    lines = Path(met_path).read_text().splitlines()
    rows = [dict(zip(lines[0].split(), line.split())) for line in lines[1:]]
    met = {(r['AMODEL'], stamp(r['INIT']), stamp(r['VALID'])): r for r in rows}
    report = json.loads(Path(report_path).read_text())
    differences, missing, skipped = {}, [], []
    for pair in report['pairs']:
        aid = 'GFSO' if pair['model'] == 'AVNO' else pair['model']
        key = (aid, pair['init'], pair['valid'])
        ref = met.get(key)
        if ref is None:
            missing.append(key)
            continue
        for ours, theirs, factor in [('met_track_nmi', 'TK_ERR', 1),
                                    ('along_km', 'ALTK_ERR', 1.852),
                                    ('cross_km', 'CRTK_ERR', 1.852)]:
            if ours in pair['errors'] and ref[theirs] != 'NA':
                differences.setdefault(ours, []).append(abs(pair['errors'][ours] - float(ref[theirs])*factor))
            elif ours in pair['errors']:
                skipped.append({'model':aid, 'init':pair['init'], 'lead':pair['lead'], 'metric':ours, 'reason':'MET NA'})
        for ours, af, bf in [('vmax_kt','AMAX_WIND','BMAX_WIND'), ('mslp_hpa','AMSLP','BMSLP')]:
            if ours in pair['errors'] and ref[af] != 'NA' and ref[bf] != 'NA':
                differences.setdefault(ours, []).append(abs(pair['errors'][ours] - (float(ref[af])-float(ref[bf]))))
    tolerances = {'met_track_nmi':0.0005, 'along_km':0.002, 'cross_km':0.002, 'vmax_kt':0, 'mslp_hpa':0}
    metrics = {m:{'n':len(v), 'max_absolute_difference':max(v), 'tolerance':tolerances[m]} for m,v in differences.items()}
    return {'storm':report['case']['storm'], 'matched_pairs':len(report['pairs'])-len(missing),
            'missing':missing, 'not_compared':skipped, 'metrics':metrics,
            'passed':not missing and all(v['max_absolute_difference'] <= v['tolerance'] for v in metrics.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pair', nargs=2, action='append', required=True, metavar=('REPORT_JSON','MET_TCST'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = {'referee':'MET-TC 12.2.2, default ACERR float32; Rust float64', 'cases':[check(*p) for p in args.pair]}
    result['passed'] = all(c['passed'] for c in result['cases'])
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
