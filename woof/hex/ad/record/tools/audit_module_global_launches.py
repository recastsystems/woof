#!/usr/bin/env python3
"""Validate observed LW module-global pointers against byte-exact launches.

This audits only the flushed part of one ordinary recording. It does not
establish a carried-state census, host frame rules, or a replay proof.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

from audit_lw_global_candidates import source_rules
from correlate_v10 import compatible
from source_signatures import _complete_json_rows


def audit(source, observer, driver, driver_images, correlation):
    source_sha, static, rules = source_rules(source)
    globals_, globals_closed = _complete_json_rows(observer / 'module-globals.jsonl.gz')
    modules, modules_closed = _complete_json_rows(observer / 'modules.jsonl.gz')
    launches, launches_closed = _complete_json_rows(observer / 'launches.jsonl.gz')
    driver_rows, driver_closed = _complete_json_rows(driver)
    with gzip.open(correlation, 'rt', encoding='utf-8') as stream:
        matched = {x['observer_ordinal']: x['driver_sequence'] for x in json.load(stream)['matched_pairs']}
    by_sequence, launch_modules, module_images, functions = {}, {}, {}, {}
    for row in driver_rows:
        kind = row.get('type')
        if kind == 'ModuleImage' and row.get('status') == 0:
            module_images[row['module']] = row['sha256']
        elif kind == 'ModuleFunction':
            functions[row['function']] = (module_images.get(row['module']),
                                           bytes.fromhex(row['name_hex']).decode('utf-8'), row['module'])
        elif kind == 'ModuleUnload' and row.get('status') == 0:
            module_images.pop(row['a'], None)
            functions = {key: value for key, value in functions.items() if value[2] != row['a']}
        elif kind == 'LaunchKernel':
            by_sequence[row['sequence']] = row
            launch_modules[row['sequence']] = functions.get(int(row['function'], 16))
    loads = collections.defaultdict(list)
    for event in modules:
        if event.get('action') == 'load':
            loads[event['module']].append(event)
    for events in loads.values():
        events.sort(key=lambda item: item['clock_ns'])
    symbols = collections.defaultdict(list)
    checked_images = {}
    for row in globals_:
        if (row.get('schema') != 'ad.binder.phys-module-global/v1' or row.get('complete') is not False
                or not isinstance(row.get('clock_ns'), int) or not isinstance(row.get('address'), int)
                or not isinstance(row.get('name'), str) or row['address'] <= 0):
            raise ValueError('invalid module-global observation')
        previous = [e for e in loads.get(row['module'], []) if e['clock_ns'] <= row['clock_ns']]
        if not previous or previous[-1]['image_sha256'] != row['image_sha256']:
            raise ValueError('module-global lacks the matching loaded image')
        raw_sha = row['image_sha256']
        if raw_sha not in checked_images:
            image = gzip.decompress((observer / 'images' / (raw_sha + '.gz')).read_bytes())
            if hashlib.sha256(image).hexdigest() != raw_sha:
                raise ValueError('observed image hash mismatch')
            recorded = image + b'\0'
            recorded_sha = hashlib.sha256(recorded).hexdigest()
            if gzip.decompress((driver_images / (recorded_sha + '.gz')).read_bytes()) != recorded:
                raise ValueError('driver image differs from observed image')
            checked_images[raw_sha] = recorded_sha
        symbols[(row['name'], row['address'])].append(row)
    result, failures, bindings = collections.Counter(), collections.Counter(), []
    for launch in launches:
        sequence = matched.get(launch['ordinal'])
        if sequence is None:
            continue
        observed_driver = by_sequence.get(sequence)
        if (not observed_driver or not compatible(launch, observed_driver)
                or bytes.fromhex(observed_driver['name_hex']).decode('utf-8') != launch['kernel']
                or launch['clock_ns'] > observed_driver['clock_ns']):
            raise ValueError('launch correlation is not byte-exact')
        for index, argument in enumerate(launch['arguments']):
            options = rules.get((launch['kernel'], index), set())
            names = {name for name, _ in options}
            if len(names) != 1 or argument.get('kind') != 'device':
                continue
            name = next(iter(names))
            info = static[name]
            address = argument['base'] + argument['offset']
            matching = [row for row in symbols.get(('mpas_static_' + name, address), [])
                        if row['clock_ns'] <= launch['clock_ns']]
            # multirank (L20): rank threads on one card share the module, and each rank looks the same global up, so one
            # global (same name, address, image, module, size) is observed once per rank. One global, not an ambiguity.
            matching = list({(row['image_sha256'], row['module'], row.get('bytes')): row for row in matching}.values())
            if (len(matching) != 1 or argument.get('allocation_id') is not None
                    or argument.get('bytes') != info['bytes'] or argument.get('dtype') != info['dtype']
                    or argument.get('offset') != 0):
                failures['lookup, size, type or identity mismatch'] += 1
                continue
            row = matching[0]
            if row.get('bytes') is not None and row['bytes'] != info['bytes']:
                failures['observed symbol size differs'] += 1
                continue
            image = checked_images[row['image_sha256']]
            if launch_modules.get(sequence) != (image, launch['kernel'], row['module']):
                failures['launch function and lookup owner image differ'] += 1
                continue
            result[(launch['kernel'], index, name, image)] += 1
            bindings.append({'sequence': sequence, 'argument': index,
                             'kernel': launch['kernel'], 'module': image,
                             'symbol': 'mpas_static_' + name, 'expected_bytes': info['bytes']})
    return {'schema': 'ad.binder.phys-module-global-launch-check/v1', 'complete': False,
            'source_sha256': source_sha, 'observed_globals': len(globals_),
            'matched_launch_arguments': sum(result.values()), 'failures': dict(sorted(failures.items())),
            'bindings': bindings,
            'matches': [{'kernel': kernel, 'argument': index, 'symbol': 'mpas_static_' + name,
                         'owner_driver_image_sha256': image, 'observations': count}
                        for (kernel, index, name, image), count in sorted(result.items())],
            'streams_closed': {'globals': globals_closed, 'modules': modules_closed,
                               'launches': launches_closed, 'driver': driver_closed},
            'limitations': ['Lookup observations validate only the same process and flushed launches.',
                            'Live replay must resolve the symbol and validate its size independently.',
                            'Global contents, carried state, and complete replay remain unproved.']}


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'observer', 'driver', 'driver-images', 'correlation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.observer, args.driver, args.driver_images, args.correlation)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ('observed_globals', 'matched_launch_arguments', 'failures', 'streams_closed')}, sort_keys=True))


if __name__ == '__main__':
    main()
