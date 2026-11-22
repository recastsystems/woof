#!/usr/bin/env python3
"""Audit source-declared longwave globals against one recorded launch trace.

These are candidates, not replay bindings: a same-run module-global lookup
and a completed census are still required before a plan may use them.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import collections
import gzip
import hashlib
import json
from pathlib import Path


def source_rules(path):
    data = path.read_bytes()
    tree = ast.parse(data, filename=str(path))
    static = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_STATIC_ARRAYS' for t in node.targets):
            if static is not None:
                raise ValueError('multiple static-array declarations')
            static = {name: {'bytes': count * 4, 'dtype': '<i4' if integer else '<f4'}
                      for name, count, integer in ast.literal_eval(node.value)}
    if not static:
        raise ValueError('static-array declaration missing')
    candidates = collections.defaultdict(set)
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Call)
                or not isinstance(node.func.func, ast.Name) or node.func.func.id != '_gpu_kernel'
                or not node.func.args or not isinstance(node.func.args[0], ast.Constant)
                or not isinstance(node.func.args[0].value, str) or len(node.args) < 3
                or not isinstance(node.args[2], ast.Tuple)):
            continue
        kernel = node.func.args[0].value
        for index, arg in enumerate(node.args[2].elts):
            if (isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name) and arg.value.id == 'K'
                    and isinstance(arg.slice, ast.Constant) and arg.slice.value in static):
                candidates[(kernel, index)].add((arg.slice.value, node.lineno))
    return hashlib.sha256(data).hexdigest(), static, candidates


def audit(source, plan_path, correlation_path, observer_path):
    sha, static, rules = source_rules(source)
    with gzip.open(plan_path, 'rt', encoding='utf-8') as stream:
        plan = json.load(stream)
    commands = {row['recorded_sequence']: row for block in plan['blocks'].values()
                for row in block if row.get('kind') == 'kernel'}
    errors = [row for row in plan['capture_errors'] if row.get('reason') == 'typed device pointer unmapped']
    with gzip.open(correlation_path, 'rt', encoding='utf-8') as stream:
        pairs = {row['observer_ordinal']: row['driver_sequence'] for row in json.load(stream)['matched_pairs']}
    needed = {row['sequence'] for row in errors}
    observed = {}
    with gzip.open(observer_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            sequence = pairs.get(row['ordinal'])
            if sequence in needed:
                observed[sequence] = row
    counts, rejected = collections.Counter(), collections.Counter()
    for error in errors:
        sequence, index = error['sequence'], error['argument']
        command, launch = commands.get(sequence), observed.get(sequence)
        if not command or not launch:
            rejected['missing command or observer'] += 1
            continue
        options = rules.get((command['function'], index), set())
        names = {name for name, _ in options}
        if len(names) != 1:
            rejected['no unambiguous source rule'] += 1
            continue
        name = next(iter(names))
        info, argument = static[name], launch['arguments'][index]
        raw = bytes.fromhex(command['arguments'][index]['bytes_hex'])
        if (argument.get('kind') != 'device' or argument.get('allocation_id') is not None
                or argument.get('bytes') != info['bytes'] or argument.get('dtype') != info['dtype']
                or len(raw) != 8 or int.from_bytes(raw, 'little') != argument['base'] + argument['offset']
                or argument['offset'] != 0):
            rejected['recorded address, size or dtype mismatch'] += 1
            continue
        lines = sorted(line for option, line in options if option == name)
        counts[(command['function'], index, name, info['bytes'], tuple(lines))] += 1
    entries = [{'kernel': k, 'argument': i, 'symbol': 'mpas_static_' + name, 'bytes': size,
                'source_lines': list(lines), 'observations': count}
               for (k, i, name, size, lines), count in sorted(counts.items())]
    return {'schema': 'ad.binder.phys-lw-global-candidates/v1', 'complete': False,
            'source_sha256': sha, 'source': str(source), 'errors_examined': len(errors),
            'candidate_observations': sum(counts.values()), 'rejected': dict(sorted(rejected.items())),
            'candidates': entries,
            'limitations': ['candidate names come from pinned source and launch metadata',
                            'same-run symbol lookup and replay equivalence remain unproved']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--correlation', type=Path, required=True)
    parser.add_argument('--observer', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.plan, args.correlation, args.observer), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
