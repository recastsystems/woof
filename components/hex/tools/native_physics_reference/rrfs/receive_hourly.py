#!/usr/bin/env python3
"""Seal one original native ungrib endpoint on the permanent archive host."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import seal, sha


def receive(source, destination, ready, expected):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists() or source == destination:
        raise ValueError('hourly archive requires a new destination')
    manifest = source/'MANIFEST.json'
    if sha(manifest) != expected['manifest_sha256']:
        raise ValueError('received hourly native manifest changed')
    receipt = json.loads(manifest.read_text())
    for key in ('ungrib_sha256', 'vtable_sha256', 'raw_source_publication_sha256', 'raw_source_files', 'intermediates'):
        if receipt[key] != expected[key]:
            raise ValueError('received hourly native source contract differs: ' + key)
    if not receipt['intermediates'] or {p.name for p in source.glob('GFS:*')} != set(receipt['intermediates']):
        raise ValueError('received hourly intermediate inventory differs')
    if any(p.is_symlink() or not p.is_file() for p in source.iterdir()):
        raise ValueError('hourly archive cannot retain scratch links or unexpected directories')
    files = {p.name:sha(p) for p in sorted(source.iterdir())}
    if any(files[name] != digest for name, digest in receipt['intermediates'].items()):
        raise ValueError('received hourly intermediate bytes changed')
    if files.get('Vtable') != receipt['vtable_sha256']:
        raise ValueError('received workflow Vtable changed')
    publication = dict(schema='mpas.native-cache-publication/v1',
        description='Original native ungrib endpoint; all raw source/decoder pins verified; no initializer derivative',
        source=str(source), selection=['*'], files=files)
    (source/'PUBLICATION.json').write_text(json.dumps(publication,indent=2)+'\n')
    digest=sha(source/'PUBLICATION.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    source.rename(destination)
    seal(destination)
    with ready.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write(f"Native ungrib endpoint {receipt['row']['start']}: `{destination}`; PUBLICATION.json sha256 `{digest}`; run decode_hourly.py from the parent INDEX.request using original pinned ungrib/Vtable.\n")
    return dict(path=str(destination),publication_sha256=digest,files=files)


def finalize(index_path, destination, ready):
    """Seal a complete hourly index after authenticating every sealed endpoint."""
    index_path = index_path.resolve(strict=True)
    destination = destination.resolve(strict=True)
    if (destination/'PUBLICATION.json').exists() or (destination/'INDEX.json').exists():
        raise ValueError('hourly index publication already exists')
    index = json.loads(index_path.read_text())
    request = index['request']
    rows, results = request['rows'], index['results']
    names = [row['name'] for row in rows]
    if (index['schema'] != 'mpas.rrfs-native-hourly-decode/v1'
            or not index['all_complete'] or not rows or len(rows) != len(results)
            or len(set(names)) != len(names)
            or any(Path(name).name != name for name in names)
            or Path(request['archive_sink']['directory']).resolve() != destination
            or {p.name for p in destination.iterdir()} != set(names)):
        raise ValueError('hourly index requires the complete exact endpoint inventory')
    hours = {}
    for row, result in zip(rows, results):
        directory = destination/row['name']
        publication = directory/'PUBLICATION.json'
        expected = result['archive_publication']
        manifest = directory/'MANIFEST.json'
        if (result['name'] != row['name'] or result['valid_time'] != row['valid_time']
                or sha(publication) != expected['publication_sha256']
                or sha(manifest) != result['manifest_sha256']
                or json.loads(publication.read_text())['files'] != expected['files']):
            raise ValueError('hourly index endpoint publication differs')
        native = json.loads(manifest.read_text())
        if (native['row']['start'] != row['valid_time']
                or native['intermediates'] != result['intermediates']
                or native['raw_source_files'] != {item['name']:item['sha256'] for item in row['inputs']}
                or any(native[key] != request[key] for key in (
                    'ungrib_sha256','vtable_sha256','raw_source_publication_sha256'))):
            raise ValueError('hourly index native input contract differs')
        if (directory.stat().st_mode & 0o222
                or any(p.is_symlink() or not p.is_file() or p.stat().st_mode & 0o222
                       for p in directory.iterdir())
                or {p.name for p in directory.iterdir()} != {*expected['files'],'PUBLICATION.json'}
                or any(sha(directory/name) != digest for name,digest in expected['files'].items())):
            raise ValueError('hourly index endpoint bytes or sealing differ')
        hours[row['name']] = expected['publication_sha256']
    shutil.copyfile(index_path, destination/'INDEX.json')
    publication = dict(schema='mpas.rrfs-hourly-native-index/v1',
        index_sha256=sha(destination/'INDEX.json'), hours=hours, completed=True)
    (destination/'PUBLICATION.json').write_text(json.dumps(publication,indent=2)+'\n')
    digest = sha(destination/'PUBLICATION.json')
    seal(destination)
    with ready.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write(f"Complete original native ungrib hourly endpoints: `{destination}`; PUBLICATION.json sha256 `{digest}`; INDEX.json records all input/decoder/publication pins; run decode_hourly.py from its request.\n")
    return dict(path=str(destination),publication_sha256=digest,endpoints=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--source',type=Path)
    action.add_argument('--index',type=Path)
    for name in ('destination','ready'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=(finalize(args.index,args.destination,args.ready) if args.index
            else receive(args.source,args.destination,args.ready,json.load(sys.stdin)))
    print(json.dumps(result))
