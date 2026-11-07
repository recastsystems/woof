"""Compare complete native moving-forecast field receipts, refusing partial runs."""
import argparse
import json
from pathlib import Path


def inventory(run, products, exit_receipt, hours):
    producer = json.loads(exit_receipt.read_text())
    if producer['returncode'] or producer.get('reason'):
        raise ValueError(f'{run}: producer did not finish successfully')
    state = json.loads((run / 'controller.json').read_text())
    frames = state.get('frames', [])
    seconds = [float(f['model_seconds']) for f in frames]
    if not seconds or min(seconds) != 0 or max(seconds) != hours * 3600:
        raise ValueError(f'{run}: frames do not span the requested whole forecast')
    cadence = float(state['check_minutes']) * 60
    if cadence <= 0 or set(seconds) != set(range(0, int(hours*3600)+1, int(cadence))):
        raise ValueError(f'{run}: missing scheduled tracker times')
    expected = {(f['segment'], f['model_seconds'], stream)
                for f in frames for stream in ('history', 'diag')}
    result = {}
    for file in products.glob('s*-t*-*/receipt.json'):
        row = json.loads(file.read_text())
        frame = row['frame']
        stream = Path(row['history']).name.split('.')[0]
        key = (frame['segment'], frame['model_seconds'], stream)
        if key in result:
            raise ValueError(f'duplicate stream receipt: {key}')
        fields = row['fields']
        if not fields or any(f.get('nonfinite', 0) for f in fields.values()):
            raise ValueError(f'empty or nonfinite native field inventory: {key}')
        result[key] = {name: {k: value[k] for k in ('sha256', 'shape', 'dtype')}
                       for name, value in fields.items()}
    if set(result) != expected:
        raise ValueError(f'{run}: native stream receipts do not exactly cover controller frames')
    return result, producer['source_commit']


def compare(left, right):
    if left.keys() != right.keys():
        return dict(status='different', reason='moving segment/time inventories differ')
    differences, count = [], 0
    for key in sorted(left):
        for field in sorted(left[key].keys() | right[key].keys()):
            count += 1
            if left[key].get(field) != right[key].get(field):
                differences.append(dict(frame=key, field=field))
    return dict(status='different' if differences else 'bitwise-identical',
                frames=len(left), compared_arrays=count, differences=differences)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arm in ('base', 'off'):
        for kind in ('run', 'products', 'exit'):
            p.add_argument(f'--{arm}-{kind}', type=Path, required=True)
    p.add_argument('--hours', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.hours <= 0:
        p.error('duration must be positive')
    try:
        left, base = inventory(a.base_run, a.base_products, a.base_exit, a.hours)
        right, off = inventory(a.off_run, a.off_products, a.off_exit, a.hours)
        result = compare(left, right)
        result.update(base_commit=base, off_commit=off, hours=a.hours)
    except (OSError, ValueError, KeyError) as error:
        result = dict(status='unverified', reason=str(error))
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return int(result['status'] != 'bitwise-identical')


if __name__ == '__main__':
    raise SystemExit(main())
