#!/usr/bin/env python3
"""Witness literal stream bytes and every field via f0's native Rust comparator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_runs import digest


def compare_outputs(runs, manifests, comparator):
    """Extend f0's full-stream comparison to the native emitted-file inventory."""
    names = [{name for name in m['outputs'] if name.endswith('.nc')} for m in manifests]
    errors = []
    if not names[0]:
        errors.append('reference has no output files')
    if names[0] != names[1]:
        errors.append(dict(missing=sorted(names[0]-names[1]), extra=sorted(names[1]-names[0])))
    for path, manifest, selected in zip(runs, manifests, names):
        for name in selected:
            if Path(name).name != name or digest(path / name) != manifest['outputs'][name]:
                raise ValueError('stream output differs from recorded native run: ' + str(path / name))
    files = {}
    for name in sorted(names[0] & names[1]):
        result = subprocess.run([str(comparator), *(str(path / name) for path in runs)],
                                capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        entry = json.loads(result.stdout)
        entry['sha256'] = [digest(path / name) for path in runs]
        entry['file_byte_identical'] = entry['sha256'][0] == entry['sha256'][1]
        files[name] = entry
    return dict(schema='mpas.native-full-stream-identity/v1', errors=errors, files=files,
        reference=str(runs[0]), other=str(runs[1]),
        all_identical=not errors and bool(files) and all(r['all_identical'] for r in files.values()),
        all_floating_words_identical=not errors and bool(files) and all(
            r.get('all_floating_words_identical', r['all_identical']) for r in files.values()),
        all_file_bytes_identical=not errors and bool(files) and all(r['file_byte_identical'] for r in files.values()))


def witness(reference, other, comparator, mode):
    runs = (reference.resolve(strict=True), other.resolve(strict=True))
    manifests = [json.loads((p / 'MANIFEST.json').read_text()) for p in runs]
    for manifest in manifests:
        if not manifest['completed'] or manifest['exit_status']:
            raise ValueError('identity requires two completed native forecasts')
    for key in ('schema', 'pin_set_hash', 'fork', 'recipe', 'binary_sha256', 'toolchain_sha256',
                'deck', 'deck_index_sha256', 'resolved_configuration', 'runtime', 'selected_table_overrides'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('native run contract differs: ' + key)
    ranks = [m['ranks'] for m in manifests]
    hosts = [m.get('host') for m in manifests]
    if mode == 'cross-host':
        if not all(hosts) or hosts[0] == hosts[1] or ranks[0] != ranks[1]:
            raise ValueError('cross-host comparison requires different recorded hosts and the same rank count')
    elif all(hosts) and hosts[0] != hosts[1]:
        raise ValueError('rerun/rank comparison requires the same host')
    if mode == 'rerun' and ranks[0] != ranks[1]:
        raise ValueError('rerun requires the same rank count')
    if mode == 'ranks' and ranks[0] == ranks[1]:
        raise ValueError('rank comparison requires different rank counts')
    # Rank-specific graph partitions are deployment inputs; all weather inputs
    # and all table bytes must match. Case paths may differ but hashes must not.
    inputs = [{name: value for name, value in m['inputs'].items() if '.graph.info.part.' not in name}
              for m in manifests]
    if inputs[0] != inputs[1]:
        raise ValueError('weather input or table bytes differ')
    result = compare_outputs(runs, manifests, comparator)
    for path, manifest in zip(runs, manifests):
        expected = {name: value for name, value in manifest['outputs'].items() if name.endswith('.nc')}
        actual = {name: digest(path / name) for name in result['files']}
        if expected != actual:
            raise ValueError('stream output differs from recorded native run: ' + str(path))
    result.update(mode=mode, ranks=ranks, hosts=hosts, deck=manifests[0]['deck'],
                  pin_set_hash=manifests[0]['pin_set_hash'],
                  comparator_sha256=digest(comparator),
                  run_manifest_sha256=[digest(p / 'MANIFEST.json') for p in runs])
    result['first_difference'] = next((dict(file=name, field=f['field'], word=f['first_difference'])
        for name, entry in result['files'].items() for f in entry['fields'] if f['differing_words']), None)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('other', type=Path)
    p.add_argument('--comparator', type=Path, required=True)
    p.add_argument('--mode', choices=('rerun', 'ranks', 'cross-host'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = witness(a.reference, a.other, a.comparator, a.mode)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('all_identical', 'all_file_bytes_identical', 'first_difference')}))
    sys.exit(0 if result['all_identical'] else 1)
