#!/usr/bin/env python3
"""Describe, but never seal, native host-call replacements for one recording.

Same-run intervals and allocation candidates make the missing evidence visible.
Observed dimensions and addresses here are not a replay rule or captured output.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def propose(plan, audit, bindings, cub_path, cub_sha, thrust_path, thrust_sha):
    if (plan.get('complete') is not False or audit.get('complete') is not False or
            bindings.get('complete') is not False or audit.get('failures') or bindings.get('binding_errors')):
        raise ValueError('expected error-free observations and an unsealed plan')
    by_id = {}
    for call in audit['calls']:
        key = (call['entry']['id'], call['entry']['kind'])
        if key in by_id:
            raise ValueError('duplicate host-call identity')
        by_id[key] = call
    positions = {}
    for block, commands in plan['blocks'].items():
        for position, command in enumerate(commands):
            sequence = command['recorded_sequence']
            if sequence in positions:
                raise ValueError('duplicate recorded command')
            positions[sequence] = (block, position, command)
    proposals, claimed = [], set()
    # A plan sealed with --max-steps N from a longer recording holds steps 1..N only; the audit and bindings cover
    # the whole recording. Breakage: a 6-step seal of the 18-step hrrrv5 pair (a recording host, 2026-11-04) refused with
    # 'missing, repeated, or empty enclosed command sequence' on the step-7 host calls. Calls past the plan's last
    # step are outside the plan, so they are left out (counted in beyond_plan_steps); every call inside is checked.
    last_step = max((int(k.split('_')[1]) for k in plan['blocks'] if k.startswith('step_')), default=None)
    beyond = 0
    for candidate in bindings['candidates']:
        if last_step is not None and candidate['step'] > last_step:
            beyond += 1
            continue
        call = by_id[(candidate['call_id'], candidate['kind'])]
        entry = call['entry']
        if candidate['binding_errors'] or call['step'] != candidate['step'] or call['slot'] != candidate['slot']:
            raise ValueError('host call and binding disagree')
        enclosed = [x['sequence'] for x in candidate['enclosed_commands']]
        if not enclosed or claimed.intersection(enclosed) or any(x not in positions for x in enclosed):
            raise ValueError('missing, repeated, or empty enclosed command sequence')
        selected = [positions[x] for x in enclosed]
        block, first = selected[0][:2]
        if any(item[0] != block or item[1] != first + i for i, item in enumerate(selected)):
            raise ValueError('host-call commands are not contiguous in proposed order')
        claimed.update(enclosed)
        target = {key: value['target'] for key, value in candidate['bindings'].items()}
        if candidate['kind'] == 'cub_reduce':
            operation = {'kind': 'native_cub_reduce', 'stream': 0, 'library_path': cub_path,
                         'library_sha256': cub_sha, **target, 'count': None,
                         'op': entry['op'], 'dtype': entry['dtype']}
        elif candidate['kind'] == 'cub_segmented_reduce':
            operation = {'kind': 'native_cub_segmented_reduce', 'stream': 0, 'library_path': cub_path,
                         'library_sha256': cub_sha, **target, 'segments': None,
                         'segment_size': None, 'op': entry['op'], 'dtype': entry['dtype']}
        elif candidate['kind'] == 'thrust_argsort':
            # The original allocation strategy has not been tied to a live
            # replay arena; deliberately leave this missing and non-runnable.
            operation = {'kind': 'native_thrust_argsort', 'stream': 0, 'library_path': thrust_path,
                         'library_sha256': thrust_sha, **target, 'workspace': None,
                         'count': None, 'dtype': entry['dtype']}
        else:
            raise ValueError('unsupported native host call')
        unresolved = sum(arg.get('kind') == 'unresolved_argument' for _, _, command in selected
                         for arg in command.get('arguments', []))
        missing_images = sum(error.get('sequence') in enclosed and error.get('reason') == 'kernel function image missing'
                             for error in plan['capture_errors'])
        proposals.append({'step': candidate['step'], 'slot': candidate['slot'], 'block': block,
                          'first_index': first, 'recorded_sequences': enclosed, 'operation': operation,
                          'observed_geometry_for_check_only': {key: entry[key] for key in
                              ('count', 'segments', 'segment_size', 'shape') if key in entry},
                          'unresolved_kernel_arguments_covered': unresolved,
                          'missing_kernel_image_errors_covered': missing_images,
                          'input_and_frame_derivation_proven': False,
                          'workspace_ownership_and_lifetime_proven': False,
                          'single_stream_replacement_proven': False})
    return {'schema': 'ad.binder.phys-library-proposals/v1', 'complete': False,
            'native_physics_replay': False, 'proposals': proposals,
            'counts': dict(sorted(Counter(p['operation']['kind'] for p in proposals).items())),
            'beyond_plan_steps': beyond,
            'unresolved_kernel_arguments_covered': sum(p['unresolved_kernel_arguments_covered'] for p in proposals),
            'missing_kernel_image_errors_covered': sum(p['missing_kernel_image_errors_covered'] for p in proposals),
            'limitations': ['The observed shape, count and buffer candidates must be derived from live state.',
                            'No proposed operation is a certified replay command.',
                            'Thrust arena selection, host checks and the carried-state census remain unresolved.']}


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'audit', 'bindings', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('cub-path', 'cub-sha', 'thrust-path', 'thrust-sha'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    with gzip.open(args.plan, 'rt', encoding='utf-8') as stream:
        plan = json.load(stream)
    result = propose(plan, json.loads(args.audit.read_text()), json.loads(args.bindings.read_text()),
                     args.cub_path, args.cub_sha, args.thrust_path, args.thrust_sha)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('proposals', 'limitations')}, sort_keys=True))


if __name__ == '__main__':
    main()
