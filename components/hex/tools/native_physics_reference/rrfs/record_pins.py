#!/usr/bin/env python3
"""Record only verified immutable RRFS products in the shared pins document."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import sha


def record(reference, document):
    pins = json.loads((Path(__file__).parent / 'pins.json').read_text())
    pin_set = {key: pins[key] for key in ('fork', 'workflow', 'submodules', 'mpas_data', 'precision', 'musica')}
    builds, tables, corrected = {}, {}, {}
    publications = [*(reference / 'native').glob('*/PUBLICATION.json'),
                    *(reference / 'native' / 'derivatives').glob('*/PUBLICATION.json'),
                    *(reference / 'tables').glob('*/PUBLICATION.json')]
    for publication in sorted(publications):
        directory = publication.parent
        published = json.loads(publication.read_text())
        manifest_path = directory / 'MANIFEST.json'
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('schema', '').startswith('mpas.rrfs-'):
            continue
        for name, expected in published['files'].items():
            if sha(directory / name) != expected:
                raise ValueError('published RRFS product changed: ' + str(directory / name))
        if manifest['schema'] == 'mpas.rrfs-native-build/v1':
            correction = manifest.get('source_correction')
            if correction:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from ruled_reference import validate_authority
                validate_authority(correction, manifest['binary_sha256'])
                target = corrected
            else:
                if directory.parent.name == 'derivatives':
                    raise ValueError('source-corrected build lacks its approved authority')
                target = builds
            key = '/'.join((manifest['reference'], manifest['recipe'], manifest['core']))
            if key in target:
                raise ValueError('ambiguous RRFS build recipe: ' + key)
            target[key] = dict(path=str(directory), binary_sha256=manifest['binary_sha256'],
                              toolchain_sha256=manifest['toolchain_sha256'],
                              flags_sha256=manifest['flags_sha256'],
                              publication_sha256=sha(publication),
                              utility_sha256=manifest.get('utility_sha256', {}))
            if correction:
                target[key]['source_correction'] = correction
        elif manifest['schema'] in ('mpas.rrfs-tempo-native-tables/v1',
                                   'mpas.rrfs-thompson-native-tables/v1'):
            if not manifest['completed']:
                raise ValueError('published native tables not completed')
            if not manifest['outputs'] or any(
                    published['files'].get(name) != expected or sha(directory / name) != expected
                    for name, expected in manifest['outputs'].items()):
                raise ValueError('native table outputs differ from publication authority')
            tables[directory.name] = dict(path=str(directory), table_sha256=manifest['outputs'],
                                         publication_sha256=sha(publication))
    data = json.loads(document.read_text())
    data['rrfs'] = dict(schema='mpas.rrfs-native-reference-pins/v1',
                        primary=pin_set, secondary_fork=pins['secondary'],
                        pin_set_hash=hashlib.sha256(json.dumps(pin_set, sort_keys=True).encode()).hexdigest(),
                        production_intel_flags=pins['production_intel_flags'],
                        native_builds=builds, native_tables=tables,
                        source_corrected_builds=corrected,
                        forecast_identity='pending; consult published per-run identity receipts')
    document.write_text(json.dumps(data, indent=1) + '\n')
    return data['rrfs']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--pins-document', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(record(a.reference_root.resolve(), a.pins_document.resolve())))
