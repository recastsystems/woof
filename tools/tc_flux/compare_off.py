"""Require complete whole-forecast field hashes before declaring OFF identical."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


def utc(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return (stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc))


def compare(base, off, expected):
    result = dict(status='unverified', expected_frames=len(expected),
                  base_frames=len(base), off_frames=len(off))
    if set(base) != expected or set(off) != expected:
        result['reason'] = 'frame keys do not exactly cover the requested whole forecast'
        result['missing_base'] = sorted(expected - base.keys())
        result['missing_off'] = sorted(expected - off.keys())
        return result
    differences = []
    count = 0
    for key in sorted(expected):
        left, right = base[key], off[key]
        if not left or not right:
            result['reason'] = f'empty field inventory at {key}'
            return result
        for field in sorted(left.keys() | right.keys()):
            count += 1
            if field not in left or field not in right or left[field] != right[field]:
                differences.append(dict(frame=key, field=field))
    result.update(status='different' if differences else 'bitwise-identical',
                  compared_arrays=count, differences=differences)
    return result


def read(root, arm):
    job = json.loads((root / arm / 'job.json').read_text())
    if job['returncode'] or job['products_returncode']:
        raise ValueError(f'{arm}: forecast or collector failed')
    result = {}
    for file in (root / 'products').glob(f'{arm}-d??-*.json'):
        row = json.loads(file.read_text())
        key = f"d{row['domain']:02}:{utc(row['valid_time']).isoformat()}"
        if key in result:
            raise ValueError(f'duplicate {key}')
        if not row['finite']:
            raise ValueError(f'nonfinite state at {key}')
        result[key] = row['field_hashes']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--hours', type=int, required=True)
    p.add_argument('--history-hours', type=int, default=1)
    p.add_argument('--domains', default='1,2,3')
    p.add_argument('--base', default='base')
    p.add_argument('--off', default='stock')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    t0 = utc(a.start)
    if a.hours <= 0 or a.history_hours <= 0 or a.hours % a.history_hours:
        p.error('positive duration must be divisible by the saved-history cadence')
    expected = {f'd{int(d):02}:{(t0+timedelta(hours=h)).isoformat()}'
                for d in a.domains.split(',') for h in range(0, a.hours+1, a.history_hours)}
    try:
        result = compare(read(a.runs, a.base), read(a.runs, a.off), expected)
    except (OSError, ValueError, KeyError) as error:
        result = dict(status='unverified', reason=str(error))
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'bitwise-identical' else 1


if __name__ == '__main__':
    raise SystemExit(main())
