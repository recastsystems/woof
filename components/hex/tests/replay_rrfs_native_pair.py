"""Compare complete strict native DA/restart executions, without a Hex claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replay(args):
    manifests = []
    for index, root in enumerate((args.left, args.right)):
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        if index == 1 and getattr(args, 'right_transport', False):
            if (manifest.get('schema') != 'rrfs-native-stream-session-transport/v1' or
                    manifest.get('source_native_manifest_sha256') != sha(args.left / 'MANIFEST.json') or
                    manifest.get('transport_completed') is not True):
                raise ValueError('transport must authenticate its strict native source')
        elif (manifest.get('completed') is not True or manifest.get('ranks') != 1 or
                manifest.get('recipe') != 'gnu-strict' or
                manifest.get('fork') != '1d6e370e7b43a332074809f170532a216c2212fb'):
            raise ValueError('complete original strict native fork execution required')
        manifests.append(manifest)
    left_files = {n for n in manifests[0]['outputs'] if n.endswith('.nc')}
    right_files = {n for n in manifests[1]['outputs'] if n.endswith('.nc')}
    membership_passed = left_files <= right_files if args.right_continuous else left_files == right_files
    rows = []
    for name in sorted(left_files):
        if name not in right_files:
            rows.append(dict(file=name, passed=False, comparison=None,
                             error='file absent from right native execution'))
            continue
        paths = [root / name for root in (args.left, args.right)]
        for path, manifest in zip(paths, manifests):
            if sha(path) != manifest['outputs'][name]:
                raise ValueError('native output hash changed: ' + str(path))
        compared = subprocess.run([str(args.comparator.resolve(strict=True)), *map(str, paths)],
                                  text=True, capture_output=True)
        result = json.loads(compared.stdout) if compared.stdout.strip() else {}
        passed = (compared.returncode == 0 and result.get('all_identical') is True and
                  result.get('words', 0) > 0 and result.get('differing_words') == 0 and not result.get('errors'))
        rows.append(dict(file=name, left_sha256=sha(paths[0]), right_sha256=sha(paths[1]),
                         comparator_exit=compared.returncode, comparison=result,
                         stderr=compared.stderr, passed=passed))
    passed = bool(rows) and membership_passed and all(row['passed'] for row in rows)
    receipt = dict(schema='rrfs-strict-native-state-ingress-pair/v1', run_host=os.uname().nodename,
        process_count=1, whole_model_proof=False,
        scope=('Native stream-session transport' if getattr(args, 'right_transport', False)
               else 'Native CPU stand-in state ingress') + '; every primitive stored word, attributes separately transport-tested',
        left_manifest_sha256=sha(args.left / 'MANIFEST.json'),
        reference_labels=manifests[0].get('reference_labels', []),
        init_derivative_authorities=manifests[0].get('init_derivative_authorities', []),
        right_manifest_sha256=sha(args.right / 'MANIFEST.json'),
        compare_resumed_times_only=args.right_continuous,
        file_membership_identical=membership_passed, all_output_words_identical=passed, files=rows)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('left', 'right', 'comparator', 'receipt'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--right-continuous', action='store_true')
    parser.add_argument('--right-transport', action='store_true')
    raise SystemExit(replay(parser.parse_args()))
