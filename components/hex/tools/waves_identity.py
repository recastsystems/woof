"""Compare all completed native atmosphere histories through the Rust word reader."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('off', type=Path)
    parser.add_argument('feedback_off', type=Path)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    completed = []
    for root in (args.off, args.feedback_off):
        receipt = json.loads((root/'native-physics-receipt.json').read_text())
        if not receipt['completed'] or receipt['steps_completed'] != receipt['steps_requested']:
            raise ValueError(f'Incomplete native forecast: {root}')
        completed.append(receipt['steps_completed'])
    if completed[0] != completed[1] or completed[0] <= 0:
        raise ValueError('Atmosphere step counts differ or are empty')
    names = []
    for pattern in ('history.*.nc', 'diag.*.nc'):
        sets = [{p.name for p in root.glob(pattern)} for root in (args.off, args.feedback_off)]
        if sets[0] != sets[1] or len(sets[0]) != args.frames:
            raise ValueError(f'{pattern}: file sets differ or do not contain {args.frames} frames')
        names.extend(sorted(sets[0]))

    def compare(name):
        with (args.output/(name+'.json')).open('w') as out, (args.output/(name+'.log')).open('w') as log:
            process = subprocess.run([str(args.binary), str(args.off/name), str(args.feedback_off/name)],
                                     stdout=out, stderr=log, timeout=900)
        if process.returncode not in (0, 1):
            raise ValueError(f'Native comparison failed: {name}, exit {process.returncode}')
        result = json.loads((args.output/(name+'.json')).read_text())
        return dict(file=name, returncode=process.returncode, **result)

    # Each native reader runs independently because the system NetCDF/HDF5
    # library is not thread-safe. Python schedules, Rust compares field words.
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        results = list(pool.map(compare, names))
    receipt = dict(schema='gpuwm.hex.waves-atmosphere-identity/v1',
        status='PASS' if all(x['all_identical'] for x in results) else 'FAIL',
        steps_per_arm=completed[0], frames_per_stream=args.frames, files=len(results),
        words=sum(x['words'] for x in results), differing_words=sum(x['differing_words'] for x in results),
        floating_words=sum(x['floating_words'] for x in results),
        differing_floating_words=sum(x['differing_floating_words'] for x in results),
        errors=[dict(file=x['file'], errors=x['errors']) for x in results if x['errors']],
        comparator=str(args.binary), comparator_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        off=str(args.off), feedback_off=str(args.feedback_off), wall_seconds=time.perf_counter()-started,
        scope='Every numeric and character field in every native history and diagnostics frame; native word comparison with no conversion, tolerance or field exclusions; global attributes are not compared',
        results=results)
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'results'}))
    if receipt['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
