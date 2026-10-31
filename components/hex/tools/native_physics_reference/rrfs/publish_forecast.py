#!/usr/bin/env python3
"""Seal a completed native forecast, retaining every emitted stream and deck."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import sha

spec = importlib.util.spec_from_file_location('rrfs_forecast_sparse_publication',
                                            Path(__file__).with_name('publish_sparse.py'))
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def publish_forecast(source, reference, mesh, date, label):
    source = source.resolve(strict=True)
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    from rrfs.ruled_reference import input_authorities, validate_authority
    init_authorities = input_authorities(manifest['case']) if 'case' in manifest else []
    if init_authorities != manifest.get('init_derivative_authorities',[]):
        raise ValueError('forecast derivative input authority missing or changed')
    corrected = manifest.get('source_correction')
    if corrected:
        validate_authority(corrected,manifest['binary_sha256'])
        if corrected['kind'] != 'gfl':
            raise ValueError('unsupported corrected atmosphere authority')
    if not manifest['completed'] or manifest['exit_status']:
        raise ValueError('only a completed real native forecast can be published')
    names = list(manifest['outputs']) + list(manifest['resolved_configuration']['files'])
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe recorded output/deck name')
        expected = (manifest['outputs'].get(name)
                    or manifest['resolved_configuration']['files'][name])
        if sha(source / name) != expected:
            raise ValueError('forecast output or deck differs from native receipt: ' + name)
    if not any(name.endswith('.nc') for name in manifest['outputs']):
        raise ValueError('completed forecast has no emitted NetCDF stream')
    for value in (mesh, date, label):
        if not value or Path(value).name != value or value in ('.', '..'):
            raise ValueError('simple publication identifiers required')
    root = reference / 'forecasts'
    if corrected:
        root = root/'source-corrected'/corrected['patch_sha256']
    destination = (root / manifest['pin_set_hash'] / manifest['recipe']
                   / mesh / date / manifest['deck'] / ('ranks-' + str(manifest['ranks'])) / label)
    return storage.publish(source, destination, ['MANIFEST.json', *names], reference / 'READY.md',
                   '; '.join([a['label'] for a in init_authorities]+([corrected['label']] if corrected else []))+
                   f'; Native RRFS forecast {mesh} {date} {manifest["deck"]} {manifest["recipe"]}, '
                   f'{manifest["ranks"]} ranks, {manifest["runtime"].get("duration", "source deck duration")}; '
                   'all emitted streams/deck/log hashes in MANIFEST.json; '
                   'rerun with rrfs/run_forecast.py and the recorded immutable binary, inputs and deck; identity pending')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    for name in ('mesh', 'date', 'label'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(publish_forecast(args.source, args.reference, args.mesh, args.date, args.label)))
