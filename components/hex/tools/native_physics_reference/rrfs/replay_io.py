#!/usr/bin/env python3
"""Complete explicit native cold/restart reruns, publishing every output on the archive host.

All scientific options, clocks, runtime paths and publication destinations come
from the request. Fortran generates fields; the native Rust comparator measures
every stored word of every emitted NetCDF stream. Retirement requires publication
and exact file hashes, and the restart companion inputs remain until both reruns.
"""
from datetime import datetime, timedelta
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess

def load(name):
    spec = importlib.util.spec_from_file_location('rrfs_io_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def replay(request, out):
    runner, overlay, identity = [load(name) for name in ('run_forecast', 'workflow_deck', 'identity')]
    sha = runner.sha
    out.mkdir(parents=True, exist_ok=False)
    (out / 'REQUEST.json').write_text(json.dumps(request, indent=2) + '\n')
    mpi = Path(request['mpi'])
    os.environ['LD_LIBRARY_PATH'] = str(mpi.parent.parent / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    for filename, digest in request['table_hashes'].items():
        if sha(Path(filename)) != digest:
            raise ValueError('native runtime table bytes changed')
    results = []
    retired = []

    def record():
        (out / 'SUMMARY.json').write_text(json.dumps(dict(schema='mpas.rrfs-native-restart-io-complete/v1', scope=request['scope'], results=results, retired=retired, completed=len(results) == len(request['rows']) and all((x.get('completed') for x in results))), indent=2) + '\n')

    def publish(run, row, label):
        manifest = json.loads((run / 'MANIFEST.json').read_text())
        names = sorted({'MANIFEST.json', *manifest['outputs'], *manifest['resolved_configuration']['files']})
        listing = run.parent / (label + '-transfer-files.txt')
        listing.write_text('\n'.join(names) + '\n')
        archive = request['archive']
        stage = Path(archive['staging']) / label
        subprocess.run(['rsync', '-a', '--mkpath', '--files-from=' + str(listing), str(run) + '/', archive['host'] + ':' + str(stage) + '/'], check=True)
        expected = dict(manifest_sha256=sha(run / 'MANIFEST.json'), mesh=row['mesh'], date=row['start'].replace(':', '.'), label=label)
        argv = ['python', archive['receiver'], '--source', str(stage), '--reference', archive['reference'], '--build', request['build']]
        command = 'export TMPDIR=' + shlex.quote(archive['tmpdir']) + '; . ' + shlex.quote(archive['environment']) + '; taskset -c ' + shlex.quote(archive['cpuset']) + ' ' + shlex.join(argv)
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', archive['host'], command], input=json.dumps(expected), capture_output=True, text=True, check=True)
        publication = json.loads(response.stdout)
        (run.parent / (label + '-PUBLICATION.json')).write_text(json.dumps(publication, indent=2) + '\n')
        return publication

    def retire(run, publication, keep=()):
        manifest = json.loads((run / 'MANIFEST.json').read_text())
        for path in sorted(run.glob('*.nc')):
            if path.name in keep:
                continue
            expected = manifest['outputs'].get(path.name) or manifest['inputs'].get(path.name)
            if path.is_symlink() and path.name not in manifest['outputs']:
                if sha(path) != expected:
                    raise ValueError('linked table bytes changed after native run: ' + path.name)
                continue
            if path.is_symlink() or sha(path) != expected:
                raise ValueError('scratch bytes changed before retirement')
            retired.append(dict(path=str(path), sha256=expected, bytes=path.stat().st_size, publication=publication, emitted=path.name in manifest['outputs']))
            record()
            path.unlink()
    for row in request['rows']:
        result = dict(name=row['name'], completed=False)
        results.append(result)
        record()
        if sha(Path(row['source_deck']).parent / 'INDEX.json') != row['source_index_sha256']:
            raise ValueError('source deck index changed')
        original = json.loads(Path(row['case']).read_text())
        files = {name: original['files'][old] for name, old in row['input_mapping'].items()}
        start = datetime.strptime(row['start'], '%Y-%m-%d_%H:%M:%S')
        later = start + timedelta(seconds=request['restart_offset_seconds'])
        stamp = later.strftime('%Y-%m-%d_%H.%M.%S')
        runs = []
        publications = []
        for phase in ['cold', 'resume-1', 'resume-2']:
            label = request['label_prefix'] + row['name'] + '-' + phase
            work = out / label
            available = int(next((line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))) * 1024
            if available < 16 * 1024 ** 3:
                raise RuntimeError('less than16GB available; native not started')
            settings = dict(**request['settings'], config_do_restart=phase != 'cold', config_run_duration=request['phase_duration'], config_start_time=(start if phase == 'cold' else later).strftime('%Y-%m-%d_%H:%M:%S'))
            spec = dict(settings=settings, stream_variables=request['stream_variables'], streams=request['streams'], stream_list_directory=request['stream_list_directory'])
            deck = overlay.generate(Path(row['source_deck']), Path(request['workflow']), Path(request['source']), work / 'decks', spec)
            selected = dict(files)
            if phase != 'cold':
                selected.update({'restart.' + stamp + '.nc': str(runs[0] / ('restart.' + stamp + '.nc')), 'mpasout.' + stamp + '.nc': str(runs[0] / ('mpasout.' + stamp + '.nc'))})
            storage_root = Path(request['storage_root']).resolve(strict=True)
            if not out.resolve().is_relative_to(storage_root):
                raise ValueError('native replay output is outside the explicit work root')
            usage = int(subprocess.check_output(['du', '-s', '-B1', str(storage_root)], text=True).split()[0])
            ingress_bytes = sum((Path(path).stat().st_size for path in selected.values()))
            if usage + ingress_bytes + request['reserve_output_bytes'] > request['work_limit_bytes']:
                raise RuntimeError('native replay exceeds the explicit work-directory storage bound')
            case = dict(files=selected, scope=request['scope'])
            (work / 'case.json').write_text(json.dumps(case, indent=2) + '\n')
            run = work / 'forecast'
            manifest = runner.forecast(Path(request['build']), deck, case, [Path(p) for p in request['tables']], run, mpi, request['ranks'])
            result[phase] = dict(manifest_sha256=sha(run / 'MANIFEST.json'), completed=manifest['completed'], exit_status=manifest['exit_status'], materialized_inputs=manifest['materialized_inputs'])
            record()
            if not manifest['completed']:
                raise RuntimeError('native restart IO stage failed: ' + label)
            publication = publish(run, row, label)
            result[phase]['publication'] = publication
            runs.append(run)
            publications.append(publication)
            record()
            if phase == 'cold':
                retire(run, publication, ['restart.' + stamp + '.nc', 'mpasout.' + stamp + '.nc'])
        comparison = identity.witness(runs[1], runs[2], Path(request['comparator']), 'rerun')
        (out / (row['name'] + '-rerun.json')).write_text(json.dumps(comparison, indent=2) + '\n')
        result['rerun'] = dict(all_identical=comparison['all_identical'], all_file_bytes_identical=comparison['all_file_bytes_identical'], first_difference=comparison['first_difference'], sha256=sha(out / (row['name'] + '-rerun.json')))
        if not comparison['all_identical']:
            raise RuntimeError('native restart rerun words differ')
        for run, publication in zip(runs, publications):
            retire(run, publication)
        result['completed'] = True
        record()
        print(json.dumps(dict(name=row['name'], completed=True, rerun=result['rerun'])), flush=True)
    return json.loads((out / 'SUMMARY.json').read_text())
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = replay(json.loads(args.request.read_text()), args.out)
    print(json.dumps(dict(completed=result['completed'], cases=len(result['results']),
                          manifest=str(args.out / 'SUMMARY.json'))))
