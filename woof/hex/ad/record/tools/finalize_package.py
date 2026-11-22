#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1 (ad/phys): turn a stage-3 hrrrv5 plan with an EMPTY stage-2 residual into a replay package.

Refuses if any residual count is nonzero (no rows are dropped or relabelled, unlike make_diag_package.py).
Sets complete=true and end_step_barrier_proven=true: the runner itself synchronises the context at every
step end before it hashes the carried state (finish_step), so the step barrier is the runner's, not an
assumption about the forward. The runner's preflight still refuses any unclassified upload, host copy or
allocation. Assets are linked, not copied: recording/{images,payloads} -> the recorder, held store -> stage2.
Usage: finalize_package.py STAGE2_RESIDUAL STAGE3_PLAN REFERENCE RECORDER OBSERVER OUT
"""
import gzip, json, os, sys
from pathlib import Path

residual_path, plan_path, reference, recorder, observer, out = sys.argv[1:7]
res = json.load(open(residual_path))
counts = res.get('residual_counts') or {}
# Not open items: the two TEMPO radii words are live frame rules whose recorded values are kept for comparison
# An allocation whose first use is a partial clear (the 0.98 GB TEMPO blob clears only the
# blob, leaving a 480-byte pad) is marked overwritten-before-read ONLY as a claim the poison runs must prove:
# the label says so, and P1 does not call the plan proved until both poison runs pass.
INFORMATIONAL = {'radii_recorded_for_check_only'}
POISON_PROVED = {'allocation_first_use_not_whole_write'}
open_items = {k: v for k, v in counts.items() if v and k not in INFORMATIONAL | POISON_PROVED}
# P1_DIAGNOSTIC=1: also accept frame_rule_needed words (carried as recording A's bytes) for a diagnostic replay only;
# the package is then marked diagnostic and is never a proof.
diagnostic = os.environ.get('P1_DIAGNOSTIC') == '1'
if diagnostic: open_items.pop('frame_rule_needed', None)
if open_items:
    sys.exit(f'refusing: stage-2 residual not empty: {open_items}')
poison_claims = {(row[0], row[1]) for row in (res.get('residual') or {}).get('allocation_first_use_not_whole_write', [])}
out = Path(out); (out / 'recording' / 'sources').mkdir(parents=True, exist_ok=True)
p = json.load(gzip.open(plan_path, 'rt'))
base = Path(plan_path).parent
# P1_DIAGNOSTIC: kernel words that differ between the two recordings (frame_rule_needed) carry recording A's bytes
# with an explicit diagnostic class; the runner accepts that class only under AD_BINDER_DIAGNOSTIC=1.
diag_words = []
if diagnostic:
    import hashlib
    tables = sorted(base.glob('*.r2-word-table-a.txt'))
    if len(tables) != 1:
        sys.exit('refusing: diagnostic package needs exactly one R2 word table beside the plan')
    table = tables[0]
    table_sha = hashlib.sha256(table.read_bytes()).hexdigest()
    for name, ops in p['blocks'].items():
        for o in ops:
            if o['kind'] != 'kernel':
                continue
            for j, arg in enumerate(o['arguments']):
                if arg.get('kind') in ('bytes', 'patched') and not arg.get('fixed_source'):
                    arg['fixed_source'] = {'class': 'diagnostic_single_recording', 'source': table.name,
                                           'source_sha256': table_sha, 'line': 1}
                    diag_words.append([name, o['function'], j])
claimed = []
for name, ops in p['blocks'].items():
    for o in ops:
        if o['kind'] == 'allocate' and (name, o['id']) in poison_claims:
            o['overwritten_before_read'] = True
            o['proof'] = 'first use is a partial clear; untouched bytes proved unread by the P1 poison runs'
            claimed.append([name, o['id'], o['bytes']])
if len(claimed) != len(poison_claims):
    sys.exit(f'refusing: poison-claimed allocations not found in the plan: {sorted(poison_claims)} vs {claimed}')
p['complete'] = True
if diagnostic: p['p1_diagnostic'] = 'frame_rule_needed words carried as recorded; not a proof'
p['end_step_barrier_proven'] = True
p['reference'] = 'reference.jsonl'
for name in ('images', 'payloads'):
    link = out / 'recording' / name
    if link.is_symlink() or link.exists(): link.unlink()
    link.symlink_to(Path(recorder, name).resolve())
held = {o['data']['path'] for ops in p['blocks'].values() for o in ops if o['kind'] == 'held_upload_ranges'}
base = Path(plan_path).parent
for rel in sorted(held):
    dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists(): dst.unlink()
    dst.symlink_to((base / rel).resolve())
for f in Path(observer, 'sources').glob('*.cu.gz'):
    (out / 'recording' / 'sources' / f.name[:-3]).write_bytes(gzip.decompress(f.read_bytes()))
(out / 'reference.jsonl').write_bytes(Path(reference).read_bytes())
# Fixed-argument provenance files (the R2 word table) named by the plan, beside the plan, so the runner's
# source-assertion check (verify_fixed_argument_sources) can read and hash them.
fixed_files = set()
for ops in p['blocks'].values():
    for o in ops:
        for arg in o.get('arguments', []) if o['kind'] == 'kernel' else []:
            src = (arg.get('fixed_source') or {}).get('source')
            if src and not src.startswith('recording/'):
                fixed_files.add(src)
for src in sorted(fixed_files):
    if (base / src).is_file():
        (out / src).write_bytes((base / src).read_bytes())
    else:
        sys.exit(f'refusing: fixed-argument source file {src} not found beside the stage-3 plan')
json.dump(p, open(out / 'plan.json', 'w'))
kinds = {}
for ops in p['blocks'].values():
    for o in ops: kinds[o['kind']] = kinds.get(o['kind'], 0) + 1
print(json.dumps({'package': str(out), 'held_files': len(held), 'op_kinds': kinds, 'poison_claims': claimed, 'diagnostic_words': diag_words, 'fixed_source_files': sorted(fixed_files),
                  'modules': len(p['modules']), 'allocations': len(p['allocations'])}))
