#!/usr/bin/env python3
"""Authenticate transferred native forecast receipts on the archive host."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('rrfs_received_forecast', Path(__file__).with_name('publish_forecast.py'))
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def receive(source, reference, build, expected, retire_output_root=None):
    source = source.resolve(strict=True)
    build = build.resolve(strict=True)
    manifest_path = source / 'MANIFEST.json'
    if publisher.sha(manifest_path) != expected['manifest_sha256']:
        raise ValueError('transferred native forecast manifest differs')
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((build / 'flag-audit.json').read_text())
    pins = json.loads((build / 'SOURCE-PINS.json').read_text())
    native = json.loads((build / 'MANIFEST.json').read_text())
    if manifest.get('source_correction') != native.get('source_correction'):
        raise ValueError('transferred forecast source-corrected authority differs')
    if (pins.get('diagnostic') or not audit['all_ok'] or audit['core'] != 'atmosphere'
            or manifest['recipe'] != audit['recipe']
            or manifest['fork'] != pins['fork']
            or manifest['pin_set_hash'] != native['pin_set_hash']
            or manifest['binary_sha256'] != publisher.sha(build / 'atmosphere_model')
            or native['binary_sha256'] != manifest['binary_sha256']
            or native['flags_sha256'] != publisher.sha(build / 'flag-audit.json')
            or manifest['toolchain_sha256'] != publisher.sha(build / 'toolchain.txt')
            or native['toolchain_sha256'] != manifest['toolchain_sha256']):
        raise ValueError('transferred forecast requires the pinned audited native executable')
    if not manifest.get('source_inputs_unchanged'):
        raise ValueError('transferred forecast lacks an unchanged ingress source witness')
    expected_names = {'MANIFEST.json', *manifest['outputs'], *manifest['resolved_configuration']['files']}
    actual = {str(path.relative_to(source)) for path in source.rglob('*') if path.is_file()}
    if actual != expected_names or any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('transferred native output inventory differs')
    publication = publisher.publish_forecast(source, reference, expected['mesh'], expected['date'], expected['label'])
    if retire_output_root is not None:
        spec = importlib.util.spec_from_file_location('rrfs_forecast_transfer_retirement',
                                                    Path(__file__).with_name('receive_case.py'))
        retirement = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retirement)
        receipt = retirement.retire_transferred_outputs(source, Path(publication['path']),
                                                       retire_output_root, publication)
        publication['retired_transfer_bytes'] = sum(row['bytes'] for row in receipt['retired'])
    return publication


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference', 'build'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--retire-output-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(receive(args.source, args.reference, args.build, json.load(sys.stdin),
                            args.retire_output_root)))
