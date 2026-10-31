#!/usr/bin/env python3
"""Recompare every stored field of two authenticated native forecast archives.

This executes the native Rust comparator on the actual archived weather bytes.
It does not execute a forecast or establish Hex/module/visual acceptance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from rrfs.identity import witness
from rrfs.ruled_reference import validate_authority, validate_init_authorities
from rrfs.verify_diagnostic_contract import publication
from transport_inputs import sha


def stage_archives(request, destination):
    """Copy unchanged archive bodies into an explicitly staged lane directory."""
    if request['schema'] != 'mpas.rrfs-published-field-recomparison-request/v1':
        raise ValueError('unsupported field recomparison request')
    destination.mkdir(parents=True, exist_ok=False)
    mapping = {}
    for row in request['comparisons']:
        for side in ('reference', 'other'):
            descriptor = row[side]
            original = str(Path(descriptor['path']))
            if original in mapping:
                continue
            source = Path(original).resolve(strict=True)
            if sha(source/'PUBLICATION.json') != descriptor['publication_sha256']:
                raise ValueError('publication digest changed: ' + original)
            published = json.loads((source/'PUBLICATION.json').read_text())
            target = destination/str(len(mapping)); target.mkdir()
            for name in ['PUBLICATION.json', *published['files']]:
                member = (source/name).resolve(strict=True)
                if Path(name).is_absolute() or '..' in Path(name).parts or not member.is_relative_to(source):
                    raise ValueError('publication member escapes its directory')
                if member.stat().st_size >= 200*1024**2:
                    raise ValueError('archive member exceeds bounded host transport')
                output = target/name; output.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['cp', '--reflink=never', '--sparse=always', '--',
                    str(member), str(output)], check=True)
            # Preserve the logical lane path, which the remote runner also uses.
            mapping[original] = str(target.absolute())
    (destination/'REPLICAS.json').write_text(json.dumps(mapping, indent=2)+'\n')
    return mapping


def archive(row, replicas):
    original = str(Path(row['path']))
    root = Path(replicas.get(original, original)).resolve(strict=True)
    published = publication(root / 'PUBLICATION.json', row['publication_sha256'])
    if 'MANIFEST.json' not in published['files']:
        raise ValueError('forecast manifest is not a published member')
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if manifest['schema'] != 'mpas.rrfs-native-forecast/v1':
        raise ValueError('requires a native forecast manifest')
    if not manifest['completed'] or manifest['exit_status'] != 0:
        raise ValueError('requires a completed native forecast')
    for name, digest in manifest['outputs'].items():
        if published['files'].get(name) != digest:
            raise ValueError('emitted stream is not a matching published member: ' + name)
    authorities = validate_init_authorities(manifest.get('init_derivative_authorities', []))
    corrected = manifest.get('source_correction')
    if corrected:
        validate_authority(corrected, manifest['binary_sha256'])
        if corrected['kind'] != 'gfl':
            raise ValueError('unsupported corrected atmosphere authority')
    for value in [*authorities, *([corrected] if corrected else [])]:
        if value['label'] not in manifest.get('reference_labels', []):
            raise ValueError('native derivative authority label is absent')
    return root, manifest, dict(path=original,
        publication_sha256=row['publication_sha256'],
        manifest_sha256=published['files']['MANIFEST.json'])


def recompare(request, comparator, destination, replicas=None):
    if request['schema'] != 'mpas.rrfs-published-field-recomparison-request/v1':
        raise ValueError('unsupported field recomparison request')
    if sha(comparator) != request['comparator_sha256']:
        raise ValueError('native Rust comparator bytes differ')
    rows = request['comparisons']
    ids = [row['id'] for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(
            not value or Path(value).name != value or value in ('.', '..') for value in ids):
        raise ValueError('requires unique simple comparison identifiers')
    if any(row['mode'] not in ('rerun', 'ranks', 'cross-host') for row in rows):
        raise ValueError('unsupported native comparison mode')
    replicas = replicas or {}
    declared = {str(Path(row[side]['path'])) for row in rows for side in ('reference', 'other')}
    if set(replicas) - declared:
        raise ValueError('replica mapping names an undeclared archive')
    destination.mkdir(parents=True, exist_ok=False)
    authenticated = {}
    results = []
    for row in rows:
        pair = []
        for side in ('reference', 'other'):
            descriptor = row[side]
            key = (descriptor['path'], descriptor['publication_sha256'])
            if key not in authenticated:
                authenticated[key] = archive(descriptor, replicas)
            pair.append(authenticated[key])
        for key in ('init_derivative_authorities', 'source_correction', 'reference_labels'):
            if pair[0][1].get(key) != pair[1][1].get(key):
                raise ValueError('native derivative contract differs: ' + key)
        if pair[0][2] == pair[1][2]:
            raise ValueError('comparison requires two distinct native publications')
        result = witness(pair[0][0], pair[1][0], comparator, row['mode'])
        result.update(native_publications=[value[2] for value in pair],
            init_derivative_authorities=pair[0][1].get('init_derivative_authorities', []),
            source_correction=pair[0][1].get('source_correction'),
            reference_labels=pair[0][1].get('reference_labels', []))
        output = destination / (row['id'] + '.json')
        output.write_text(json.dumps(result, indent=2) + '\n')
        fields = {}
        for file in result['files'].values():
            for field in file['fields']:
                if field['differing_words']:
                    name = field['field']
                    fields[name] = fields.get(name, 0) + field['differing_words']
        results.append(dict(id=row['id'], mode=row['mode'], receipt=output.name,
            receipt_sha256=sha(output), native_publications=result['native_publications'],
            ranks=result['ranks'], hosts=result['hosts'], files=len(result['files']),
            stored_words=sum(file['words'] for file in result['files'].values()),
            differing_words=sum(file['differing_words'] for file in result['files'].values()),
            differing_fields=fields, first_difference=result['first_difference'],
            all_identical=result['all_identical'], all_file_bytes_identical=result['all_file_bytes_identical']))
    summary = dict(schema='mpas.rrfs-published-field-recomparison/v1',
        scope='All emitted archived NetCDF fields, raw stored-word comparison; no new forecast or Hex replay',
        comparator_sha256=sha(comparator), comparisons=results,
        all_identical=all(row['all_identical'] for row in results),
        archives_rehashed=True, weather_fields_decoded_by='native Rust comparator',
        forecast_executed=False, hex_replay=False, visual_acceptance=False)
    (destination / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--comparator', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replicas', type=Path)
    parser.add_argument('--stage', action='store_true')
    args = parser.parse_args()
    if args.stage:
        stage_archives(json.loads(args.request.read_text()), args.out)
        sys.exit(0)
    if args.comparator is None:
        parser.error('--comparator is required for numerical recomparison')
    result = recompare(json.loads(args.request.read_text()), args.comparator, args.out,
        json.loads(args.replicas.read_text()) if args.replicas else None)
    print(json.dumps(dict(all_identical=result['all_identical'], comparisons=[
        {key: row[key] for key in ('id', 'files', 'stored_words', 'differing_words')}
        for row in result['comparisons']])))
    sys.exit(0 if result['all_identical'] else 1)
