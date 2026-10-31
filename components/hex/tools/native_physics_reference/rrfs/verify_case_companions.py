#!/usr/bin/env python3
"""Authenticate complete native initializer companion indexes without running weather.

Original stage bytes, source/recipe authority, static-to-case7 dependencies and
the archived Rust finiteness summaries must agree. This is input provenance,
not rerun, rank, cross-host, physics replay or Hex forecast acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

from verify_diagnostic_contract import publication
from transport_inputs import sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, expected):
    path = Path(path)
    catalog = publication(path, expected)
    require('INDEX.json' in catalog['files'], 'index is not a published member')
    index = json.loads((path.parent / 'INDEX.json').read_text())
    build = Path(index['initializer_build'])
    authority = publication(build / 'PUBLICATION.json', index['initializer_publication_sha256'])
    for name in ('MANIFEST.json', 'SOURCE-PINS.json', 'flag-audit.json', 'init_atmosphere_model'):
        require(name in authority['files'], 'initializer authority missing member: ' + name)
    native = json.loads((build / 'MANIFEST.json').read_text())
    pins = json.loads((build / 'SOURCE-PINS.json').read_text())
    audit = json.loads((build / 'flag-audit.json').read_text())
    require(native['core'] == 'init_atmosphere' and native['reference'] == 'primary'
            and native['recipe'] == 'gnu-strict' and not pins.get('diagnostic'),
            'requires pristine primary strict initializer')
    require(native['pin_set']['precision'] == 'single'
            and native['pin_set']['musica'] is False, 'initializer precision/chemistry mismatch')
    pin_hash = hashlib.sha256(json.dumps(native['pin_set'], sort_keys=True).encode()).hexdigest()
    require(pin_hash == native['pin_set_hash'] == index['pin_set_hash'], 'initializer pin set changed')
    require(all(pins.get(k) == v for k, v in native['pin_set'].items()), 'initializer source pins changed')
    require(audit['all_ok'] is True and not audit['errors']
            and audit['core'] == native['core'] and audit['recipe'] == native['recipe']
            and len(audit['objects']) == native['objects'], 'initializer audit incomplete')
    require(authority['files']['init_atmosphere_model'] == native['binary_sha256']
            and authority['files']['flag-audit.json'] == native['flags_sha256'],
            'initializer binary/audit binding changed')
    # Match copied stage authorities by their digests, independent of directory
    # layout. Dates and mesh names are request/index data, never defaults.
    copies = {}
    for name, digest in catalog['files'].items():
        if Path(name).name == 'ORIGINAL-PUBLICATION.json':
            copies.setdefault(digest, []).append(Path(name).parent)
    total_words = total_files = total_stages = 0
    rows = []
    identities = set()
    for product in index['products']:
        stages = product['stages']
        require(len(stages) == 3 and {r['stage'] for r in stages} == {'static', 'invariant', 'case7'},
                'requires exactly static/invariant/case7 companions')
        manifests = {}
        finite_path = None
        for stage in stages:
            directory = Path(stage['path'])
            original = publication(directory / 'PUBLICATION.json', stage['publication_sha256'])
            require('MANIFEST.json' in original['files'], 'stage manifest is not published')
            matching = copies.get(stage['publication_sha256'], [])
            require(len(matching) == 1, 'stage authority copy missing or ambiguous')
            manifest_name = str(matching[0] / 'MANIFEST.json')
            require(catalog['files'].get(manifest_name) == original['files']['MANIFEST.json'],
                    'archived manifest differs from original stage')
            manifest = json.loads((directory / 'MANIFEST.json').read_text())
            require(manifest['completed'] is True and manifest['exit_status'] == 0
                    and manifest['ranks'] == 1, 'native stage did not complete at stated rank')
            require(manifest['binary_sha256'] == native['binary_sha256']
                    and manifest['audit_sha256'] == native['flags_sha256']
                    and not manifest['source_pins'].get('diagnostic')
                    and all(manifest['source_pins'].get(k) == v for k, v in native['pin_set'].items()),
                    'stage initializer/source authority differs')
            require(manifest['outputs'] == stage['outputs'] and manifest['outputs'],
                    'stage output inventory differs')
            require(all(original['files'].get(k) == v for k, v in stage['outputs'].items()),
                    'native output is not a published member')
            manifests[stage['stage']] = manifest
            total_files += len(manifest['outputs'])
            total_stages += 1
            if stage['stage'] == 'case7':
                finite_path = matching[0].parent / 'FINITENESS.json'
        require(manifests['case7']['inputs'].get('static.nc') == manifests['static']['outputs'].get('static.nc')
                and 'static.nc' in manifests['static']['outputs'], 'static-to-case7 dependency differs')
        variables = manifests['case7']['row']['variables']
        require(variables['init_case'] == 7
                and variables['nsoillevels'] == product['soil_levels']
                and variables['nvertlevels'] == product['model_levels']
                and product['interface_levels'] == product['model_levels'] + 1,
                'case7 soil/vertical companion differs')
        date = product.get('date', index.get('date'))
        require(date is not None and date.replace('.', ':') == variables['start_time'],
                'case7 clock differs from indexed companion')
        require(all(m['host'] == manifests['case7']['host']
                    and all(m['row']['variables'].get(k) == variables[k]
                            for k in ('start_time', 'nsoillevels', 'nvertlevels'))
                    for m in manifests.values()), 'native stage host/clock/levels differ')
        identity = (product['mesh'], variables['start_time'], product['soil_levels'])
        require(identity not in identities, 'duplicate native companion row')
        identities.add(identity)
        require(str(finite_path) in catalog['files'], 'Rust finiteness summary is not published')
        finite = json.loads((path.parent / finite_path).read_text())
        require(finite['schema'] == 'mpas.rrfs-native-finiteness-summary/v1'
                and finite['outputs'] == manifests['case7']['outputs']
                and finite['files'] == len(finite['outputs']), 'finiteness output binding differs')
        fields = finite['fields']
        require(fields and len({(f['file'], f['field']) for f in fields}) == len(fields),
                'finiteness field inventory empty or duplicated')
        require(all(f['file'] in finite['outputs'] and f['words'] > 0 and f['nonfinite'] == 0
                    and f['first_nonfinite'] is None for f in fields)
                and finite['all_finite'] is True and finite['nonfinite_words'] == 0,
                'nonfinite native case7 words')
        require(sum(f['words'] for f in fields) == finite['floating_words'] == product['finite_floating_words'],
                'finiteness word count differs')
        total_words += finite['floating_words']
        rows.append(dict(mesh=product['mesh'], start=variables['start_time'], soil_levels=product['soil_levels'],
                         host=manifests['case7']['host'], ranks=1, finite_words=finite['floating_words']))
    require(rows and total_stages == index['completed_stages'] and total_files == index['nc_files']
            and total_words == index['finite_case7_words'], 'companion aggregate inventory differs')
    return dict(schema='mpas.rrfs-native-case-companion-byte-verification/v1', verified=True,
                publication_sha256=expected, initializer_binary_sha256=native['binary_sha256'],
                pin_set_hash=pin_hash, products=rows, stages=total_stages, files=total_files,
                finite_case7_words=total_words, weather_execution=False, canonical_OFF=False,
                physics_replay=False, hex_identity=False, scope=index['scope'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.publication, args.expected_sha256)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
