#!/usr/bin/env python3
"""Measure whole archived native outputs across two explicitly pinned recipes.

Recipe-generated table differences remain in the contract and measurement.
This is a diagnostic measurement, not a same-recipe identity witness.
Weather decoding/comparison stays in the existing native Rust comparator.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

spec = importlib.util.spec_from_file_location('rrfs_recipe_archives',
                                            Path(__file__).with_name('run_published_decks.py'))
archives = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archives)
sha = archives.replay.load('run_forecast').sha


def validate(manifests, authority):
    for manifest, expected in zip(manifests, authority['recipes'], strict=True):
        if not manifest['completed'] or manifest['exit_status'] or not manifest['source_inputs_unchanged']:
            raise ValueError('recipe measurement requires completed unchanged native forecasts')
        for key in ('recipe', 'binary_sha256', 'toolchain_sha256'):
            if manifest[key] != expected[key]:
                raise ValueError('recipe build authority changed: ' + key)
    if manifests[0]['recipe'] == manifests[1]['recipe']:
        raise ValueError('two distinct recipe authorities required')
    for key in ('schema', 'pin_set_hash', 'fork', 'deck', 'deck_index_sha256',
                'resolved_configuration', 'runtime', 'selected_table_overrides', 'ranks', 'host'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('scientific forecast contract differs: ' + key)
    inputs = [m['inputs'] for m in manifests]
    differences = {name: [values.get(name) for values in inputs]
                   for name in inputs[0].keys() | inputs[1].keys()
                   if inputs[0].get(name) != inputs[1].get(name)}
    if differences != authority['recipe_input_differences']:
        raise ValueError('unwitnessed recipe input difference')
    if any(None in values for values in differences.values()):
        raise ValueError('recipe measurement cannot omit an input')
    return differences


def measure(request, out):
    identity = archives.replay.load('identity')
    root = Path(request['work_root']).resolve(strict=True)
    if out.exists() or not out.resolve().is_relative_to(root):
        raise ValueError('recipe measurement requires new owned scratch')
    if sha(Path(request['comparator'])) != request['comparator_sha256']:
        raise ValueError('native Rust comparator authority changed')
    out.mkdir(parents=True)
    (out / 'REQUEST.json').write_text(json.dumps(request, indent=2) + '\n')
    results = []
    for pair in request['pairs']:
        name = pair['name']
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('simple recipe measurement identifier required')
        while int(subprocess.check_output(['du', '-s', '-B1', str(root)], text=True).split()[0]) + pair['transfer_bytes'] + request['reserve_bytes'] > request['limit_bytes']:
            time.sleep(30)
        directory = out / name
        directory.mkdir()
        paths = [directory / str(i) for i in range(2)]
        manifests = []
        for path, source in zip(paths, pair['sources'], strict=True):
            archives.acquire(request['archive_host'], source['path'], path,
                             source['publication_sha256'], ['MANIFEST.json'])
            if sha(path / 'MANIFEST.json') != source['manifest_sha256']:
                raise ValueError('native forecast manifest authority changed')
            manifests.append(json.loads((path / 'MANIFEST.json').read_text()))
        differences = validate(manifests, pair)
        for path, source, manifest in zip(paths, pair['sources'], manifests, strict=True):
            for native in sorted(n for n in manifest['outputs'] if n.endswith('.nc')):
                subprocess.run(['rsync', '-a', request['archive_host'] + ':' + source['path'] + '/' + native,
                                str(path / native)], check=True)
            names = ['MANIFEST.json', *(n for n in manifest['outputs'] if n.endswith('.nc'))]
            archives.authenticate(path, source['publication_sha256'], names)
        measured = identity.compare_outputs(paths, manifests, Path(request['comparator']))
        measured.update(mode='recipe-measurement', scope=request['scope'],
                        recipes=pair['recipes'], recipe_input_differences=differences,
                        native_sources=pair['sources'], ranks=[m['ranks'] for m in manifests],
                        hosts=[m['host'] for m in manifests], comparator_sha256=request['comparator_sha256'],
                        run_manifest_sha256=[sha(path / 'MANIFEST.json') for path in paths])
        measured['first_difference'] = next((dict(file=n, field=f['field'], word=f['first_difference'])
            for n, entry in measured['files'].items() for f in entry['fields'] if f['differing_words']), None)
        result = directory / 'COMPARISON.json'
        result.write_text(json.dumps(measured, indent=2) + '\n')
        retired = []
        for path, manifest in zip(paths, manifests):
            checked = []
            for native, digest in manifest['outputs'].items():
                if native.endswith('.nc'):
                    p = path / native
                    if sha(p) != digest:
                        raise ValueError('owned native copy changed before retirement')
                    checked.append(dict(path=str(p), sha256=digest, bytes=p.stat().st_size))
            for row in checked:
                Path(row['path']).unlink()
            retired.extend(checked)
        (directory / 'RETIRED.json').write_text(json.dumps(retired, indent=2) + '\n')
        fields = [f for entry in measured['files'].values() for f in entry['fields']]
        row = dict(name=name, comparison_sha256=sha(result), files=len(measured['files']),
                   stored_words=sum(f['words'] for f in fields),
                   differing_words=sum(f['differing_words'] for f in fields),
                   all_identical=measured['all_identical'], first_difference=measured['first_difference'],
                   errors=measured['errors'], retired_copy_bytes=sum(r['bytes'] for r in retired))
        results.append(row)
        (out / 'SUMMARY.json').write_text(json.dumps(dict(scope=request['scope'], requested=len(request['pairs']),
            results=results, all_measured=len(results) == len(request['pairs'])), indent=2) + '\n')
        print(json.dumps(row), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    measure(json.loads(args.request.read_text()), args.out)
