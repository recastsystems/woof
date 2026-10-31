#!/usr/bin/env python3
"""Run a complete native initialization pipeline with bounded endpoint storage.

Requests pin archive endpoints and external inputs. Native init computes every
field; Rust inventories its IEEE words. Each completed stage is authenticated
on the permanent archive host before its private transfer copy is retired.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def load(name):
    spec = importlib.util.spec_from_file_location('rrfs_pipeline_' + name,
                                                Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate(directory, digest):
    cache = load('cache_cases')
    if cache.sha(directory / 'PUBLICATION.json') != digest:
        raise ValueError('native endpoint publication changed')
    publication = json.loads((directory / 'PUBLICATION.json').read_text())
    if any(cache.sha(directory / name) != value for name, value in publication['files'].items()):
        raise ValueError('native endpoint bytes changed')
    return publication


def retire_endpoint(directory, digest, owned_root):
    """Delete only an authenticated, independently copied endpoint directory."""
    if directory.is_symlink() or directory.parent.resolve() != owned_root.resolve():
        raise ValueError('endpoint retirement requires an owned direct child')
    publication = authenticate(directory, digest)
    if {p.name for p in directory.iterdir()} != set(publication['files']) | {'PUBLICATION.json'}:
        raise ValueError('endpoint retirement inventory differs')
    for path in [directory, *directory.rglob('*')]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
    shutil.rmtree(directory)


def pipeline(request, out):
    cache = load('cache_cases')
    finite = load('summarize_finiteness')
    root = Path(request['work_root']).resolve(strict=True)
    out = out.absolute()
    if not out.resolve().is_relative_to(root) or out.exists():
        raise ValueError('native pipeline requires a new owned output directory')
    workflow, build, mpi = (Path(request[key]) for key in ('workflow', 'build', 'mpi'))
    inventory = Path(request['inventory'])
    if cache.sha(inventory) != request['inventory_sha256']:
        raise ValueError('native Rust inventory executable changed')
    if any(cache.sha(workflow / name) != digest for name, digest in request['workflow_templates'].items()):
        raise ValueError('pinned native workflow templates changed')
    if any(cache.sha(Path(name)) != digest for name, digest in request['external_inputs'].items()):
        raise ValueError('pinned native external input changed')
    out.mkdir(parents=True)
    (out / 'REQUEST.json').write_text(json.dumps(request, indent=2) + '\n')
    endpoints = Path(request['endpoint_work']).absolute()
    if not endpoints.resolve().is_relative_to(root) or endpoints.exists() or not request['stages']:
        raise ValueError('native pipeline requires new owned endpoint scratch and explicit stages')
    endpoints.mkdir(parents=True)
    current, current_pin = None, None
    results, produced, retirement = [], {}, []
    host = request['archive_host']
    try:
        for position, item in enumerate(request['stages']):
            row = dict(item['row'])
            endpoint = item['decoded_archive']
            digest = endpoint['publication_sha256']
            if digest != current_pin:
                if current is not None:
                    retire_endpoint(current, current_pin, endpoints)
                    retirement.append(current_pin)
                    current, current_pin = None, None
                usage = int(subprocess.check_output(['du', '-s', '-B1', str(root)], text=True).split()[0])
                if usage + request['reserve_bytes'] >= request['limit_bytes']:
                    raise ValueError('next native stage exceeds the lane storage reservation')
                current = endpoints / digest
                current.mkdir()
                subprocess.run(['rsync', '-a', host + ':' + endpoint['path'] + '/', str(current) + '/'], check=True)
                authenticate(current, digest)
                current_pin = digest
            row['decoded'] = str(current)
            for group in ('inputs', 'runtime_files'):
                for name, source in row.get(group, {}).items():
                    expected = request['external_inputs'].get(source, produced.get(source))
                    if expected is None or cache.sha(Path(source)) != expected:
                        raise ValueError('native stage input is not an authenticated dependency: ' + source)
            while True:
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))) * 1024
                if available >= 16 * 1024**3:
                    break
                time.sleep(15)
            directory = out / row['name']
            error, manifest, publication, summary = None, None, None, None
            try:
                manifest = cache.prepare(row, directory, workflow, build, mpi)
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                if (directory / 'MANIFEST.json').exists():
                    manifest = json.loads((directory / 'MANIFEST.json').read_text())
            if manifest and manifest['completed']:
                inventories = {}
                for name, expected in manifest['outputs'].items():
                    if cache.sha(directory / name) != expected:
                        raise ValueError('native output changed before its Rust inventory')
                    target = directory / (name + '-inventory.json')
                    with target.open('w') as stream:
                        subprocess.run([str(inventory), str(directory / name)], stdout=stream, check=True)
                    inventories[name] = json.loads(target.read_text())
                    produced[str(directory / name)] = expected
                summary = finite.summarize(manifest['outputs'], inventories)
                summary['inventory_binary_sha256'] = cache.sha(inventory)
                (directory / 'FINITENESS.json').write_text(json.dumps(summary, indent=2) + '\n')
                names = sorted(p.name for p in directory.iterdir() if p.is_file() and not p.is_symlink())
                listing = directory / 'transfer-files.txt'
                listing.write_text('\n'.join(names) + '\n')
                stage = Path(request['receive_root']) / row['name']
                subprocess.run(['rsync', '-a', '--mkpath', '--files-from=' + str(listing),
                                str(directory) + '/', host + ':' + str(stage) + '/'], check=True)
                description = row['description'] + '; Rust inventory nonfinite words=' + str(summary['nonfinite_words'])
                argv = ['env', 'TMPDIR=' + request['archive_tmpdir'], request['archive_python'],
                        request['receiver'], '--source', str(stage), '--destination', row['publish'],
                        '--ready', request['ready'], '--build', str(build)]
                if request.get('retire_received_outputs'):
                    argv.extend(['--retire-output-root', request['archive_work_root']])
                response = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, shlex.join(argv)],
                    input=json.dumps(dict(manifest_sha256=cache.sha(directory / 'MANIFEST.json'),
                                          description=description)), capture_output=True, text=True, check=True)
                publication = json.loads(response.stdout)
                # Keep native outputs only while later stages consume them.
                # The permanent receiver authenticated every archived byte.
                needed = {source for later in request['stages'][position+1:]
                          for group in ('inputs', 'runtime_files')
                          for source in later['row'].get(group, {}).values()}
                retired = []
                for name, digest in manifest['outputs'].items():
                    path = directory / name
                    if str(path) in needed:
                        continue
                    if path.is_symlink() or cache.sha(path) != digest:
                        raise ValueError('native output changed before retirement')
                    retired.append(dict(file=name, bytes=path.stat().st_size, sha256=digest))
                    path.unlink()
                (directory / 'OUTPUT-RETIREMENT.json').write_text(json.dumps(
                    dict(publication=publication, retired=retired), indent=2) + '\n')
            result = dict(name=row['name'], error=error, manifest=manifest, publication=publication,
                          finiteness=summary, decoded_publication_sha256=current_pin)
            results.append(result)
            value = dict(schema='mpas.rrfs-native-case-pipeline/v1', scope=request['scope'],
                         requested=len(request['stages']), results=results,
                         all_attempts_finished=len(results) == len(request['stages']),
                         all_complete=len(results) == len(request['stages']) and
                         all(r['manifest'] and r['manifest']['completed'] for r in results))
            (out / 'SUMMARY.json').write_text(json.dumps(value, indent=2) + '\n')
            print(json.dumps(dict(name=row['name'], completed=bool(manifest and manifest['completed']),
                                  error=error, publication=publication)), flush=True)
            if not manifest or not manifest['completed']:
                # Dependent stages cannot run from absent or failed inputs.
                break
        return value
    finally:
        if current is not None and current_pin is not None:
            retire_endpoint(current, current_pin, endpoints)
            retirement.append(current_pin)
        (out / 'ENDPOINT-RETIREMENT.json').write_text(json.dumps(dict(retired_publications=retirement), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = pipeline(json.loads(args.request.read_text()), args.out)
    raise SystemExit(0 if result['all_attempts_finished'] and result['all_complete'] else 1)
