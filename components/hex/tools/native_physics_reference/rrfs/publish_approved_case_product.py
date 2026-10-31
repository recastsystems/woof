#!/usr/bin/env python3
"""Seal a whole approved input product from completed native pipeline receipts."""
import argparse
import json
from pathlib import Path
import shutil
from ruled_reference import authority, INIT_BINARY, sha
from verify_approved_reference import verify
from publish_cache import publish


def seal(request, stage, destination, ready):
    if stage.exists() or not request['expected_names']:
        raise ValueError('explicit input product and new metadata stage required')
    rows = []
    for item in request['summaries']:
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError('native pipeline summary changed')
        value = json.loads(path.read_text())
        rows.extend(value['results'])
    for item in request.get('recoveries', []):
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError('native transfer recovery receipt changed')
        value = json.loads(path.read_text())
        rows.append(dict(name=value['manifest']['row']['name'], manifest=value['manifest'],
                         finiteness=value['finiteness'], publication=value['publication'], error=None))
    names = [row['name'] for row in rows]
    if len(set(names)) != len(names) or set(names) != set(request['expected_names']):
        raise ValueError('whole native input product membership differs')
    products, entries = [], []
    for row in sorted(rows, key=lambda r:r['name']):
        manifest, finite = row['manifest'], row['finiteness']
        if (row['error'] or not manifest or not row['publication'] or not finite
                or manifest['row']['name'] != row['name'] or not manifest['completed']
                or manifest['exit_status'] or not finite['all_finite'] or finite['nonfinite_words']
                or finite['outputs'] != manifest['outputs']):
            raise ValueError('canonical input product requires complete finite native stages')
        entry = {key:row['publication'][key] for key in ('path', 'publication_sha256')}
        entries.append(entry)
        products.append(dict(name=row['name'], **entry, outputs=manifest['outputs'],
                             floating_words=finite['floating_words'], nonfinite_words=0))
    provenance = dict(builds=request['builds'], cases=entries)
    checked = verify(provenance)
    stage.mkdir(parents=True)
    for position, item in enumerate([*request['summaries'], *request.get('recoveries', [])]):
        shutil.copy2(item['path'], stage / f'native-{position}-receipt.json')
    (stage / 'REQUEST.json').write_text(json.dumps(request, indent=2) + '\n')
    (stage / 'VERIFY-REQUEST.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (stage / 'VERIFIED.json').write_text(json.dumps(checked, indent=2) + '\n')
    index = dict(schema='mpas.rrfs-approved-canonical-input-product/v1',
                 source_correction=authority('init', INIT_BINARY), products=products,
                 stages=len(products), netcdf_files=sum(len(p['outputs']) for p in products),
                 floating_words=sum(p['floating_words'] for p in products), nonfinite_words=0,
                 complete=True, module_replay=False, forecast_identity=False,
                 production_request_sha256=sha(stage / 'REQUEST.json'))
    (stage / 'INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    return publish(stage, destination, ['*'], ready, request['description'] + '; ' +
                   index['source_correction']['label'] + '; native finite input provenance only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'stage', 'destination', 'ready'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(json.loads(args.request.read_text()), args.stage, args.destination, args.ready)))
