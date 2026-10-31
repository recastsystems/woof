#!/usr/bin/env python3
"""Decode explicit paired hourly GRIB rows with the unchanged native ungrib.

Input transfer, hashing and native process invocation are orchestration only.
Native Fortran performs every weather decode; no fields are changed in Python.
Raw bytes are fetched one pair at a time to bound disposable scratch storage.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import shlex

_spec = importlib.util.spec_from_file_location('rrfs_hourly_case_runner', Path(__file__).with_name('cache_cases.py'))
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
decode, sha = _runner.decode, _runner.sha


def execute(request, work, ungrib, vtable):
    work = work.resolve()
    ungrib = ungrib.resolve(strict=True)
    vtable = vtable.resolve(strict=True)
    work.mkdir(parents=True, exist_ok=False)
    source = request['source']
    if sha(ungrib) != request['ungrib_sha256'] or sha(vtable) != request['vtable_sha256']:
        raise ValueError('native ungrib or source workflow Vtable bytes changed')
    raw = work/'raw'; raw.mkdir()
    results = []
    for row in request['rows']:
        name = row['name']
        if Path(name).name != name:
            raise ValueError('unsafe hourly decode name')
        if 'storage' in request:
            storage=request['storage']
            usage=int(subprocess.check_output(['du','-s','-B1',str(Path(storage['work_root']).resolve(strict=True))],text=True).split()[0])
            needed=2*sum(item['bytes'] for item in row['inputs'])+storage['max_intermediate_bytes']+storage['reserve_bytes']
            if usage+needed>storage['limit_bytes']:
                raise ValueError('hourly decode exceeds explicit work-directory storage bound')
        paths = []
        for item in row['inputs']:
            filename = item['name']
            if Path(filename).name != filename:
                raise ValueError('unsafe raw input filename')
            path = raw/filename
            argv = [*source['rsync_argv'], source['host']+':'+source['directory'].rstrip('/')+'/'+filename, str(path)]
            subprocess.run(argv, check=True)
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('received raw input bytes differ from immutable source: '+filename)
            paths.append(str(path))
        directory = work/name
        decoded = dict(name=name, gribs=[paths], start=row['valid_time'], stop=row['valid_time'], interval_seconds=request['interval_seconds'])
        decode(decoded, directory, ungrib, vtable)
        manifest = json.loads((directory/'MANIFEST.json').read_text())
        manifest['raw_source_publication_sha256'] = request['raw_source_publication_sha256']
        manifest['raw_source_files'] = {item['name']:item['sha256'] for item in row['inputs']}
        (directory/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result=dict(name=name, valid_time=row['valid_time'], manifest_sha256=sha(directory/'MANIFEST.json'), intermediates=manifest['intermediates'])
        if 'storage' in request and sum((directory/p).stat().st_size for p in manifest['intermediates'])>request['storage']['max_intermediate_bytes']:
            raise ValueError('native intermediate exceeded the explicit storage reservation; output retained')
        if 'archive_sink' in request:
            sink=request['archive_sink'];remote=sink['directory'].rstrip('/')
            incoming=remote+'/'+name+'.partial'
            subprocess.run([*sink['rsync_argv'],'--copy-links',str(directory)+'/',sink['host']+':'+incoming+'/'],check=True)
            expected={key:manifest[key] for key in ('ungrib_sha256','vtable_sha256','raw_source_publication_sha256','raw_source_files','intermediates')}
            expected['manifest_sha256']=result['manifest_sha256']
            argv=[*sink['receiver_argv'],'--source',incoming,'--destination',remote+'/'+name,'--ready',sink['ready']]
            received=subprocess.run([*sink['ssh_argv'],sink['host'],shlex.join(argv)],input=json.dumps(expected),capture_output=True,text=True,check=True)
            publication=json.loads(received.stdout)
            if any(publication['files'].get(p)!=h for p,h in manifest['intermediates'].items()):
                raise ValueError('archive host did not authenticate every intermediate')
            result['archive_publication']=publication
        results.append(result)
        (work/'INDEX.json').write_text(json.dumps(dict(schema='mpas.rrfs-native-hourly-decode/v1', request=request, results=results, all_complete=len(results)==len(request['rows'])),indent=2)+'\n')
        for path in paths:
            p=Path(path)
            if p.parent != raw or p.is_symlink():
                raise ValueError('unsafe disposable raw input')
            p.unlink()
        if 'archive_sink' in request:
            for filename,digest in manifest['intermediates'].items():
                path=directory/filename
                if path.is_symlink() or sha(path)!=digest:
                    raise ValueError('disposable native intermediate changed after archive publication')
                path.unlink()
        print(json.dumps(results[-1]),flush=True)
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('request','work','ungrib','vtable'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    execute(json.loads(args.request.read_text()),args.work,args.ungrib,args.vtable)
