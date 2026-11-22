#!/usr/bin/env python3
"""Produce a deliberately unsealed, relocated physics command list.

Only byte-exact Python/Driver argument correlations are typed. Other launch
arguments remain unresolved, including candidates which happen to lie inside a
physical device allocation. The resulting file is for investigation: it is
not a replay plan and must never be promoted to complete without the requested
allocation, frame, host-boundary and replay proofs.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path
from source_signatures import add_verified, from_cache, from_capture_archive, from_observation


def rows(path):
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)
    except EOFError:
        pass


def observed_empty_null(observed, raw):
    """Recognize a byte-exact null pointer for an observed zero-size array.

    The caller must still prove the live frame rule which produces the null;
    this only distinguishes it from an unrelocated nonzero device address.
    """
    shape = observed.get('shape')
    return (len(raw) == 8 and raw == bytes(8) and observed.get('base') == 0
            and observed.get('offset') == 0 and observed.get('bytes') == 0
            and isinstance(shape, list) and bool(shape)
            and all(isinstance(d, int) and d >= 0 for d in shape) and 0 in shape)


class AllocationLookup:
    """Resolve an address against the entire time-qualified lifetime index.

    A released small allocation can have a greater base than a later, live
    larger allocation containing that address. Looking only at the nearest
    historical base misses the live allocation. Multiple live matches are an
    error: the planner must not guess which root owns an address.
    """
    def __init__(self, allocations):
        self.roots = sorted(allocations, key=lambda root: root['recorded_base'])
        self.bases = [root['recorded_base'] for root in self.roots]
        self.prefix_end = []
        for root in self.roots:
            end = root['recorded_base'] + root['bytes']
            self.prefix_end.append(max(self.prefix_end[-1], end) if self.prefix_end else end)

    def pointer(self, address, clock, size=0):
        if not isinstance(address, int) or not isinstance(size, int) or address < 0 or size < 0:
            raise ValueError('invalid pointer or extent')
        matches = []
        index = bisect.bisect_right(self.bases, address) - 1
        while index >= 0 and self.prefix_end[index] > address:
            root = self.roots[index]
            offset = address - root['recorded_base']
            if (0 <= offset < root['bytes'] and size <= root['bytes'] - offset
                    and root['allocated_clock_ns'] <= clock
                    and (root['freed_clock_ns'] is None or clock < root['freed_clock_ns'])):
                matches.append({'allocation': root['id'], 'offset': offset})
            index -= 1
        if len(matches) > 1:
            raise ValueError('ambiguous live physical allocation')
        return matches[0] if matches else None


def global_binding_map(audit):
    """Index same-recording global checks; this never certifies global contents."""
    if audit.get('failures') or audit.get('schema') != 'ad.binder.phys-module-global-launch-check/v1':
        raise ValueError('module-global audit failed')
    result = {}
    for row in audit['bindings']:
        key = (row['sequence'], row['argument'])
        image, symbol, size = row['module'], row['symbol'], row['expected_bytes']
        if (key in result or not isinstance(key[0], int) or key[0] < 1
                or not isinstance(key[1], int) or key[1] < 0
                or not isinstance(image, str) or len(image) != 64
                or any(c not in '0123456789abcdef' for c in image)
                or not isinstance(symbol, str) or not symbol.startswith('mpas_static_')
                or '\x00' in symbol or not isinstance(size, int) or size < 1):
            raise ValueError('invalid or duplicate module-global binding')
        result[key] = {'kind': 'module_global', 'module': image, 'symbol': symbol,
                       'offset': 0, 'expected_bytes': size}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--observer', type=Path, required=True)
    parser.add_argument('--correlation', type=Path, required=True)
    parser.add_argument('--linearization', type=Path, required=True)
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument('--observed-sources', type=Path)
    parser.add_argument('--observed-driver-images', type=Path)
    parser.add_argument('--capture-archive', type=Path)
    parser.add_argument('--capture-config')
    parser.add_argument('--capture-audit', type=Path)
    parser.add_argument('--same-run-global-source', type=Path,
                        help='audit module globals from this driver and observer, not another recording')
    parser.add_argument('--verified-source-proof', type=Path)
    parser.add_argument('--verified-source-code', type=Path)
    parser.add_argument('--pointer-tables', type=Path)
    parser.add_argument('--event-audit', type=Path)
    parser.add_argument('--asset-prefix', required=True,
                        help='relative path to the recorded physics image and payload directory')
    parser.add_argument('--max-steps', type=int, default=72)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    asset_prefix = Path(args.asset_prefix)
    if asset_prefix.is_absolute() or '..' in asset_prefix.parts or args.max_steps < 1:
        parser.error('invalid relative asset prefix or step count')
    census = json.loads(args.census.read_text())
    signatures, source_images = from_cache(args.source_cache) if args.source_cache else ({}, {})
    if bool(args.observed_sources) != bool(args.observed_driver_images):
        parser.error('observed sources and driver images must be supplied together')
    source_observation = None
    if args.observed_sources:
        found, images, source_observation = from_observation(args.observed_sources, args.observed_driver_images)
        for key, parsed in found.items():
            if key in signatures and signatures[key] != parsed:
                signatures[key] = None
            elif key not in signatures:
                signatures[key] = parsed
        source_images.update(images)
    capture_observation = None
    if any((args.capture_archive, args.capture_config, args.capture_audit)):
        if not all((args.capture_archive, args.capture_config, args.capture_audit,
                    args.observed_sources, args.observed_driver_images)):
            parser.error('capture archive, config and audit require same-run observed sources and images')
        if args.observer.resolve() != (args.observed_sources / 'launches.jsonl.gz').resolve():
            parser.error('capture bindings require this plan\'s own observed launch stream')
        found, images, capture_observation = from_capture_archive(
            args.observed_sources, args.observed_driver_images, args.capture_archive, args.capture_config)
        for key, parsed in found.items():
            if key in signatures and signatures[key] != parsed:
                signatures[key] = None
            elif key not in signatures:
                signatures[key] = parsed
        for key, records in images.items():
            source_images.setdefault(key, {}).setdefault('capture_manifests', []).extend(records)
        args.capture_audit.parent.mkdir(parents=True, exist_ok=True)
        args.capture_audit.write_text(json.dumps(capture_observation, sort_keys=True, indent=2) + '\n')
    if bool(args.verified_source_proof) != bool(args.verified_source_code):
        parser.error('verified source proof and code must be supplied together')
    if args.verified_source_proof:
        add_verified(signatures, source_images, args.verified_source_proof, args.verified_source_code)
    globals_by_argument = {}
    if args.same_run_global_source:
        if not args.observed_driver_images:
            parser.error('same-run globals require this driver recording\'s images')
        from audit_module_global_launches import audit as audit_globals
        checked_globals = audit_globals(args.same_run_global_source, args.observer.parent,
                                        args.driver, args.observed_driver_images, args.correlation)
        globals_by_argument = global_binding_map(checked_globals)
    pointer_tables = (json.loads(args.pointer_tables.read_text())['tables']
                      if args.pointer_tables else [])
    table_patches = {table['upload_sequence']: table['pointers'] for table in pointer_tables}
    events = []
    if args.event_audit:
        event_audit = json.loads(args.event_audit.read_text())
        if event_audit['schema'] != 'ad.binder.event-audit/v1' or event_audit['errors']:
            parser.error('event lifetime audit failed')
        # The initial allocation snapshot synchronizes the CUDA context first.
        # An event recorded before that barrier is complete; a newly created,
        # unrecorded CUDA event has the same wait/synchronize behavior.
        events = [{'id': event['id'], 'flags': event['flags'],
                   'recorded_before_initial_barrier': event['recorded']}
                  for event in event_audit['initial_live_events']]
    graph = json.loads(args.linearization.read_text())
    with gzip.open(args.correlation, 'rt') as f:
        matched = {p['driver_sequence']: p['observer_ordinal'] for p in json.load(f)['matched_pairs']}
    matched_ordinals = set(matched.values())
    visible = {r['ordinal']: r for r in rows(args.observer) if r['ordinal'] in matched_ordinals}
    roots = sorted(census['allocations'], key=lambda a: (a['recorded_base'], a['allocated_clock_ns']))
    by_base = collections.defaultdict(list)
    for root in roots:
        by_base[root['recorded_base']].append(root)
    pointer = AllocationLookup(roots).pointer
    # S14: module-global symbol lookups observed in this recording (observe_module_globals.py get_global_var).
    global_obs = []
    global_path = args.observer.parent / 'module-globals.jsonl.gz'
    if global_path.is_file():
        with gzip.open(global_path, 'rt') as f:
            global_obs = sorted((json.loads(line) for line in f if line.strip()), key=lambda r: r['clock_ns'])

    def module_global_at(address, clock):
        hits = [r for r in global_obs if r['address'] == address and r['clock_ns'] <= clock]
        if not hits:
            return None
        last = hits[-1]
        if any(r['address'] == address and (r['name'], r['image_sha256']) != (last['name'], last['image_sha256'])
               for r in hits[-8:]):
            return None  # ambiguous: the address named two symbols
        return last

    modules = {}
    function_map = {}
    commands = {}
    pending_upload = None
    pending_copy = None
    step = 0
    first_boundary_clock = None
    dynamic_allocations = set()
    counters = collections.Counter()
    missing = []
    slots = []
    slot_stacks = collections.defaultdict(list)
    for row in rows(args.driver):
        kind = row['type']
        seq = row['sequence']
        clock = row.get('clock_ns', 0)
        if kind == 'Boundary':
            step = row['a']
            if step == 1:
                if first_boundary_clock is not None:
                    raise ValueError('duplicate initial boundary')
                first_boundary_clock = clock
            continue
        if kind == 'ModuleImage' and row.get('status') == 0:
            modules[row['module']] = {'id': row['sha256'], 'sha256': row['sha256'],
                                     'path': str(asset_prefix / 'images' / (row['sha256'] + '.gz')),
                                     'compression': 'gzip', 'bytes': row['bytes']}
            continue
        if kind == 'ModuleFunction':
            module = modules.get(row['module'])
            if not module:
                missing.append({'sequence': seq, 'reason': 'module image missing'})
            function_map[row['function']] = (module, bytes.fromhex(row['name_hex']).decode('utf-8', 'replace'))
            continue
        if kind == 'HostUploadBytes':
            pending_upload = row
            continue
        if kind == 'GenericCopyKind':
            pending_copy = row
            continue
        if kind == 'CUPTIFallbackBegin':
            if 1 <= step <= args.max_steps:
                missing.append({'sequence': seq, 'reason':
                    'unwrapped driver call requires host-interval and default-stream ordering proof'})
            continue
        if kind in ('CUPTIFallbackEnd', 'CUPTIFallbackInvalid'):
            if kind == 'CUPTIFallbackInvalid':
                missing.append({'sequence': seq, 'reason': 'unwrapped driver call has invalid parameters'})
            continue
        if kind == 'HostSlot' and 1 <= row.get('step', 0) <= args.max_steps:
            try:
                name = bytes.fromhex(row['operation_hex']).decode('utf-8')
                phase = row['phase']
                thread_id = row['thread_id']
                if not isinstance(thread_id, int) or thread_id <= 0:
                    raise ValueError('slot marker lacks a valid host thread')
                slot_stack = slot_stacks[thread_id]
                if row['step'] != step:
                    raise ValueError('slot step differs from driver boundary')
                if phase == 0:
                    slot_stack.append({'operation': name, 'step': step, 'thread_id': thread_id,
                                       'begin_sequence': seq})
                elif phase in (1, 2) and slot_stack:
                    slot = slot_stack.pop()
                    if slot['operation'] != name or slot['step'] != step:
                        raise ValueError('slot markers do not nest')
                    slot.update({'end_sequence': seq, 'exception': phase == 2})
                    slots.append(slot)
                    if phase == 2:
                        missing.append({'sequence': seq, 'reason': 'host slot raised an exception'})
                else:
                    raise ValueError('unexpected host slot phase')
            except (KeyError, UnicodeError, ValueError) as error:
                missing.append({'sequence': seq, 'reason': 'invalid host slot marker: ' + str(error)})
            continue
        if not 1 <= step <= args.max_steps:
            continue
        command = None
        if kind == 'LaunchKernel':
            function = int(row['function'], 16)
            module, name = function_map.get(function, (None, ''))
            if module is None:
                missing.append({'sequence': seq, 'reason': 'kernel function image missing'})
            seen = visible.get(matched.get(seq))
            signature = signatures.get((module['id'], name)) if module else None
            if signature is not None and (len(signature) != len(row.get('parameters', [])) or
                    any(width != value.get('size') for (_, width), value in zip(signature, row['parameters']))):
                signature = None
            arguments = []
            for index, param in enumerate(row.get('parameters', [])):
                raw = bytes.fromhex(param.get('bytes_hex', ''))
                observed = seen['arguments'][index] if seen else None
                if observed and observed['kind'] == 'device':
                    target = pointer(observed['base'] + observed['offset'], clock)
                    if observed_empty_null(observed, raw):
                        arguments.append({'kind': 'bytes', 'bytes_hex': raw.hex(),
                                          'observed_dtype': observed['dtype'],
                                          'observed_shape': observed['shape'],
                                          'frame_rule': 'unproved'})
                        counters['typed_empty_null_arguments'] += 1
                    elif target is None:
                        global_binding = globals_by_argument.get((seq, index))
                        if global_binding and module and global_binding['module'] == module['id']:
                            arguments.append(global_binding)
                            counters['same_run_module_global_arguments'] += 1
                        else:
                            missing.append({'sequence': seq, 'argument': index, 'reason': 'typed device pointer unmapped'})
                            arguments.append({'kind': 'unresolved_argument', 'bytes_hex': raw.hex()})
                    else:
                        arguments.append({'kind': 'device', 'target': target,
                                          'observed_dtype': observed['dtype'], 'observed_shape': observed['shape']})
                        counters['typed_device_arguments'] += 1
                elif observed and observed['kind'] == 'scalar':
                    arguments.append({'kind': 'bytes', 'bytes_hex': raw.hex(),
                                      'observed_dtype': observed['dtype'],
                                      'frame_rule': 'unproved'})
                    counters['typed_scalar_arguments'] += 1
                elif signature and signature[index][0] in ('carray', 'device_pointer'):
                    target = pointer(int.from_bytes(raw[:8], 'little'), clock)
                    if target is not None:
                        if signature[index][0] == 'carray':
                            arguments.append({'kind': 'patched', 'bytes_hex': raw.hex(),
                                              'pointers': [{'byte_offset': 0, 'target': target}],
                                              'source_image_sha256': module['id'], 'frame_rule': 'unproved'})
                        else:
                            arguments.append({'kind': 'device', 'target': target,
                                              'source_image_sha256': module['id']})
                        counters['source_typed_device_arguments'] += 1
                    elif signature[index][0] == 'device_pointer' and int.from_bytes(raw, 'little') == 0:
                        arguments.append({'kind': 'bytes', 'bytes_hex': raw.hex(),
                                          'source_image_sha256': module['id'], 'frame_rule': 'unproved'})
                        counters['source_typed_null_arguments'] += 1
                    else:
                        arguments.append({'kind': 'unresolved_argument', 'bytes_hex': raw.hex(),
                                          'reason': 'source pointer has no live allocation'})
                        counters['unresolved_arguments'] += 1
                elif signature and signature[index][0] in ('indexer', 'scalar'):
                    arguments.append({'kind': 'bytes', 'bytes_hex': raw.hex(),
                                      'source_image_sha256': module['id'], 'frame_rule': 'unproved'})
                    counters['source_typed_value_arguments'] += 1
                else:
                    candidates = []
                    for off in range(0, len(raw) - 7, 8):
                        target = pointer(int.from_bytes(raw[off:off+8], 'little'), clock)
                        if target:
                            candidates.append({'byte_offset': off, 'target': target})
                    arguments.append({'kind': 'unresolved_argument', 'bytes_hex': raw.hex(),
                                      'pointer_candidates': candidates})
                    counters['unresolved_arguments'] += 1
            command = {'kind': 'kernel', 'module': module['id'] if module else None,
                       'function': name, 'grid': row['grid'], 'block': row['block'],
                       'shared_bytes': row['shared_bytes'], 'arguments': arguments}
            if row.get('extra') or not row.get('parameter_array', False):
                missing.append({'sequence': seq, 'reason': 'launch parameter encoding not handled'})
        elif kind in ('Memcpy', 'MemcpyAsync'):
            if not pending_copy or pending_copy['a'] != 2 or pending_copy['b'] != 2:
                missing.append({'sequence': seq, 'reason': 'generic copy direction unresolved'})
                command = {'kind': 'unresolved_copy', 'bytes': row['count']}
            else:
                command = {'kind': 'device_copy',
                           'destination': pointer(row['a'], clock, row['count']),
                           'source': pointer(row['b'], clock, row['count']), 'bytes': row['count']}
            pending_copy = None
        elif kind.startswith('MemcpyHtoD'):
            if not pending_upload or pending_upload['bytes'] != row['count'] or not pending_upload['payload']:
                missing.append({'sequence': seq, 'reason': 'upload bytes missing'})
                data = None
            else:
                data = {'encoding': 'gzip', 'path': str(asset_prefix / 'payloads' / (pending_upload['sha256'] + '.gz')),
                        'sha256': pending_upload['sha256'], 'bytes': pending_upload['bytes']}
            command = {'kind': 'upload', 'destination': pointer(row['a'], clock, row['count']),
                       'data': data, 'host_value_rule': 'unproved',
                       'embedded_device_pointers': 'not audited'}
            if seq in table_patches:
                command['pointers'] = table_patches[seq]
                command['embedded_device_pointers'] = 'source-identified MPAS rate table'
                counters['source_identified_rate_tables'] += 1
            pending_upload = None
        elif kind.startswith('MemcpyDtoH'):
            command = {'kind': 'device_to_host', 'source': pointer(row['b'], clock, row['count']),
                       'bytes': row['count'], 'asynchronous': kind.endswith('Async'),
                       'recorded_host_destination': row['a'], 'host_consumer': 'unresolved'}
        elif kind.startswith('MemcpyDtoD'):
            command = {'kind': 'device_copy', 'destination': pointer(row['a'], clock, row['count']),
                       'source': pointer(row['b'], clock, row['count']), 'bytes': row['count']}
        elif kind.startswith('MemsetD8'):
            command = {'kind': 'clear', 'destination': pointer(row['a'], clock, row['count']),
                       'bytes': row['count'], 'value': row['b']}
        elif kind.startswith(('MemsetD16', 'MemsetD32')):
            width = 2 if kind.startswith('MemsetD16') else 4
            command = {'kind': 'fill16' if width == 2 else 'fill32',
                       'destination': pointer(row['a'], clock, row['count'] * width),
                       'count': row['count'], 'value': row['b']}
        elif kind in ('EventCreate', 'EventDestroy', 'EventRecord', 'StreamWaitEvent',
                      'EventSynchronize', 'StreamSynchronize', 'CtxSynchronize'):
            command = {'kind': 'synchronization', 'operation': kind,
                       'event_handle_recorded': row['a'], 'flags': row['extra']}
        elif kind in ('MemAlloc', 'MemAllocAsync', 'MemFree', 'MemFreeAsync'):
            if kind.startswith('MemAlloc'):
                found = [root for root in by_base.get(row['a'], [])
                         if root['allocated_clock_ns'] == clock and root['bytes'] == row['count']]
                if len(found) != 1 or row['a'] == 0 or row['count'] == 0:
                    missing.append({'sequence': seq, 'reason': 'allocation lifetime is ambiguous'})
                    continue
                root = found[0]
                dynamic_allocations.add(root['id'])
                command = {'kind': 'allocate', 'id': root['id'], 'bytes': root['bytes'],
                           'asynchronous': kind.endswith('Async'), 'overwritten_before_read': False}
                missing.append({'sequence': seq, 'id': root['id'],
                                'reason': 'allocated storage overwrite-before-read unproved'})
            elif row['a'] == 0:
                command = {'kind': 'free_null', 'asynchronous': kind.endswith('Async')}
            else:
                found = [root for root in by_base.get(row['a'], []) if root['freed_clock_ns'] == clock]
                if len(found) != 1:
                    missing.append({'sequence': seq, 'reason': 'free lifetime is ambiguous'})
                    continue
                command = {'kind': 'free', 'id': found[0]['id'], 'asynchronous': kind.endswith('Async')}
        if (command is not None and command.get('kind') == 'device_copy' and command.get('destination') is None
                and command.get('source') is not None):
            # S14: the destination is a CUDA module global of a recorded image (fast-numerics TEMPO writes its
            # binary32 parameter block tempo_k by a device copy before each launch that reads it: tempo_v303.py
            # _set_scalars). The observer's get_global_var record names the image, symbol and recorded address; the
            # replay looks the symbol up in its own loaded image. THE BREAKAGE THIS PREVENTS: the fast pkg18m2 sealed
            # 36 device copies with destination null, and every replay refused the plan ("invalid type: null,
            # expected struct Slice").
            hit = module_global_at(row['a'], clock)
            # The observer hashes the image CuPy hands the driver; the plan names a module by the driver-recorded
            # image (ModuleImage), so the module is taken through the driver handle both records carry.
            image = modules.get(hit['module']) if hit is not None else None
            if hit is not None and image is not None:
                command = {'kind': 'module_global_copy', 'module': image['id'], 'symbol': hit['name'],
                           'offset': row['a'] - hit['address'], 'source': command['source'], 'bytes': command['bytes']}
                counters['module_global_copies'] += 1
        if command is not None:
            command['stream'] = 0  # target of the topological single-stream order
            command['recorded_sequence'] = seq
            command['step'] = step
            if any(command.get(k, 1) is None for k in ('source', 'destination')):
                missing.append({'sequence': seq, 'reason': 'transfer pointer unmapped'})
            commands[seq] = command
            counters[command['kind']] += 1

    unclosed = sum(len(stack) for stack in slot_stacks.values())
    if unclosed:
        missing.append({'reason': 'unclosed host slot markers', 'count': unclosed})
    if not slots:
        missing.append({'reason': 'no host slot markers observed'})
    blocks = {f'step_{step:02d}': [] for step in range(1, args.max_steps + 1)}
    for node in graph['nodes']:
        command = commands.pop(node['issue'], None)
        if command:
            blocks[f"step_{command.pop('step'):02d}"].append(command)
    if commands:
        missing.append({'reason': 'commands absent from linearization', 'count': len(commands)})
    allocations = []
    if first_boundary_clock is None:
        raise ValueError('initial boundary missing')
    for root in census['allocations']:
        if root['id'] in dynamic_allocations:
            continue
        if (root['allocated_clock_ns'] > first_boundary_clock or
                (root['freed_clock_ns'] is not None and root['freed_clock_ns'] <= first_boundary_clock)):
            continue
        sha = root['initial_sha256']
        allocations.append({'id': root['id'], 'bytes': root['bytes'],
                            'initial': ({'encoding': 'gzip', 'path': str(asset_prefix / 'payloads' / (sha + '.gz')),
                                         'sha256': sha, 'bytes': root['bytes']} if sha else None),
                            'classification': 'unresolved', 'overwritten_before_read': False})
    result = {'schema': 'ad.binder.plan/v1', 'complete': False,
              'source_pin': '16251589f19aec0012fd164d1ea060126208aed1',
              'end_step_barrier_proven': False, 'modules': list({m['id']: m for m in modules.values()}.values()),
              'allocations': allocations, 'streams': [], 'events': events, 'carried': [],
              'blocks': blocks, 'cadence': [[f'step_{step:02d}'] for step in range(1, args.max_steps + 1)],
              'observed_host_slots': slots,
              'capture_errors': missing,
              'limitations': ['unsealed argument, lifetime, access and frame rules',
                              'host consumers and host-library boundaries have not been classified',
                              'source layouts do not prove all host dataflow or frame rules',
                              'census and replay/poison/full-word proofs are incomplete']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, 'wt', encoding='utf-8', compresslevel=1) as f:
        json.dump(result, f, separators=(',', ':'), sort_keys=True)
    print(json.dumps({'complete': False, 'modules': len(result['modules']), 'allocations': len(allocations),
                      'blocks': len(blocks), 'counters': dict(sorted(counters.items())),
                      'capture_errors': len(missing), 'capture_error_examples': missing[:8],
                      'source_images': len(source_images), 'source_observation': source_observation,
                      'capture_observation': ({k: v for k, v in capture_observation.items() if k not in ('records', 'rejected')}
                                              if capture_observation else None),
                      'host_slot_intervals': len(slots)}, sort_keys=True))


if __name__ == '__main__':
    main()
