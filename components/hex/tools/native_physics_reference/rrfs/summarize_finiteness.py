#!/usr/bin/env python3
"""Summarize metadata emitted by the native Rust IEEE-word inventory.

This does not read or transform weather arrays. Nonfinite native words are
reported, rather than removed or used to disable a scheme.
"""
import argparse
import json
from pathlib import Path


def summarize(outputs, inventories):
    if not outputs or set(outputs) != set(inventories):
        raise ValueError('Rust inventory must cover every emitted native file')
    rows = []
    for filename, inventory in inventories.items():
        if inventory['schema'] != 'mpas.rrfs-native-field-finiteness/v1' or not inventory['fields']:
            raise ValueError('native Rust floating-word inventory is absent')
        names = set()
        for field in inventory['fields']:
            if field['field'] in names or field['netcdf_type'] not in (5, 6):
                raise ValueError('native Rust field inventory is ambiguous')
            names.add(field['field'])
            counts = [field[key] for key in ('nan', 'positive_infinity', 'negative_infinity')]
            words = field['words']
            if (type(words) is not int or words < 0
                    or any(type(count) is not int or count < 0 for count in counts)
                    or sum(counts) > words
                    or (field['first_nonfinite'] is None) != (sum(counts) == 0)):
                raise ValueError('native Rust nonfinite counts are inconsistent')
            rows.append(dict(file=filename, field=field['field'], words=words,
                             nonfinite=sum(counts), first_nonfinite=field['first_nonfinite']))
    return dict(schema='mpas.rrfs-native-finiteness-summary/v1',
                outputs=outputs, files=len(inventories), fields=rows,
                floating_words=sum(row['words'] for row in rows),
                nonfinite_words=sum(row['nonfinite'] for row in rows),
                all_finite=all(row['nonfinite'] == 0 for row in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inventories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    inventories = {name: json.loads((args.inventories / (name + '-inventory.json')).read_text())
                   for name in manifest['outputs']}
    args.out.write_text(json.dumps(summarize(manifest['outputs'], inventories), indent=2) + '\n')
