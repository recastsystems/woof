#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 3 of the hrrrv5 milestone-1 sealer .

Adds what the runner's physics replay path needs around the stage-2 blocks:
* the live physics frame (`physics`): the frozen hrrrv5 clock and alarm state before step 1,
  taken from the recording's own frame observation (frames.jsonl.gz `alarms_before`, clock_before),
  and one recorded block per step (`step_blocks`);
* the two host boundaries the frame needs in every step: `physics_timetracker_state` at that host
  slot's position (observed_host_slots) and `update_state_time` at the end of the step;
* the frame host owner (`frame_state_id`) in host_carried;
* carried device state and its reference: every allocation that has a boundary digest
  (BoundaryAllocationDigest, written by the tracer at each outer step boundary) at every step is
  carried, and its per-step SHA-256 becomes the reference line. This is the full-state comparison;
  the poison census that may shrink it is a separate proof.
Nothing here certifies the plan; `complete` stays false.
"""
import argparse
import collections
import gzip
import json
from pathlib import Path


def jsonl(p):
    with gzip.open(p, 'rt') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def frame_states_from_observation(events):
    """P1 sealer-output-steps: the ordinary forward's frame snapshot after every recorded step, from the
    recording's own frame observation (frames.jsonl.gz: manager, calendar, state_time, clock_advance per step),
    with the encoding and field choices of runtime/tests/binder_phys_frame_snapshots.py (which made the six-step
    sample). Breakage: stage 3 of the 18-step seal stopped at 'no frame snapshot for step 7' because the only
    snapshots were the six in configs/binder/hrrrv5.frame-state.sample.jsonl."""
    from frame_snapshot import encode, integral
    # multirank (L20): with same-host rank threads every rank's physics driver keeps its own frame (clock, alarms)
    # and emits the four events, so a step has N copies of each (manager x N, calendar x N, then state_time and
    # clock_advance N times). The frames are one model clock: every copy must be byte-equal, and one is kept.
    first = next((i for i, x in enumerate(events) if x.get('event') == 'state_time'), len(events))
    ranks = max(1, sum(1 for x in events[:first] if x.get('event') == 'manager'))
    if ranks > 1:
        if len(events) % (4 * ranks):
            raise SystemExit(f'frame observation is not whole steps of {ranks} ranks x four events')
        kept = []
        for i in range(0, len(events), 4 * ranks):
            window = events[i:i + 4 * ranks]
            by = collections.defaultdict(list)
            for x in window:
                by[x['event']].append(x)
            for kind in ('manager', 'calendar', 'state_time', 'clock_advance'):
                copies = by.get(kind, [])
                if len(copies) != ranks or any(json.dumps(c, sort_keys=True) != json.dumps(copies[0], sort_keys=True)
                                               for c in copies):
                    raise SystemExit(f'rank frames differ or are incomplete at {kind}, window {i // (4 * ranks) + 1}')
                kept.append(copies[0])
        events = kept
    if len(events) % 4:
        raise SystemExit('frame observation is not whole steps of four events')
    result = {}
    for i in range(len(events) // 4):
        manager, calendar, time, advance = events[4 * i:4 * i + 4]
        if [x['event'] for x in (manager, calendar, time, advance)] != ['manager', 'calendar', 'state_time', 'clock_advance']:
            raise SystemExit('unexpected frame observation order')
        step = calendar['step']
        if step != i + 1 or any(x['complete'] for x in (manager, calendar, time, advance)):
            raise SystemExit(f'invalid observed frame at step {i + 1}')
        before, after = advance['before'], advance['after']
        if before != calendar['clock_before'] or before != time['clock']:
            raise SystemExit(f'frame clock observation differs at step {step}')
        if time['clock_advanced'] or time['source_level'] != 1 or time['target_level'] != 1:
            raise SystemExit(f'unexpected state time route at step {step}')
        active = [manager['active'][k] for k in ('radiation_lw', 'radiation_sw', 'convection')]
        result[step] = encode(after, after['elapsed'], step + 1, manager['dt'], integral(manager['time_seconds']),
                              active, manager['correct_rp6'], manager['pbl_due'], manager['alarms_after'],
                              int.from_bytes(bytes.fromhex(time['target_f32_le']), 'little'), time['target_xtime'])
    return result


def block_key(name):
    """Step blocks in step order. THE BREAKAGE THIS PREVENTS: the names are step_01 .. step_NN and, sorted as text, a
    plan of 100 or more steps visits step_100 before step_11 (a 3 km storm window of 15 s steps has 240 in an hour).
    Below 100 steps both orders are the same."""
    head, _, tail = name.partition('_')
    return (0, int(tail), '') if head == 'step' and tail.isdigit() else (1, 0, name)


def require_boundary_state(steps, digests):
    """Every step of the plan must have the recorder's boundary state. THE BREAKAGE THIS PREVENTS: a recorder library
    that stopped snapshotting after step 72 said nothing; the carried set is the intersection over all steps, so one
    step with none empties it, and the plan would then carry and compare no device state at any step."""
    bare = [s for s in steps if not digests.get(s)]
    if bare:
        raise SystemExit(f'no boundary state recorded for steps {bare[0]} to {bare[-1]} ({len(bare)} of {len(steps)} '
                         'steps): the recorder stopped snapshotting. Record again with the current library.')


def name_run_steps(physics, events, steps):
    """A plan sealed as a PREFIX of a longer recording names the recorded run length in physics.run_steps; the replay
    takes the run's last step from it (the model computes the TEMPO effective radii at the RUN's last step, not at the
    plan's). THE BREAKAGE THIS PREVENTS: every 29-step replay of a 29-of-60-step plan refused 'state hash mismatch at
    step 29'. Written only for a prefix, so a whole-recording plan is the bytes it was. The recorded length is the
    number of different steps that have a calendar event."""
    recorded = len({int(r['step']) for r in events if r.get('event') == 'calendar'})
    if recorded > len(steps):
        physics['run_steps'] = recorded
    return physics


def output_seconds(blocks, dt):
    """Seconds between output streams that read the pools, from the steps whose block downloads the history
    stream (stage 2's history_stream_output copies). Empty when no step writes output."""
    steps = sorted(int(n.split('_')[1]) for n, ops in blocks.items()
                   if any(op.get('proof') == 'history_stream_output' for op in ops))
    if not steps:
        return []
    period = steps[0] * dt
    if any((s * dt) % period for s in steps) or [s * dt for s in steps] != [period * (k + 1) for k in range(len(steps))]:
        raise SystemExit(f'history output steps {steps} are not one regular stream')
    return [period]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--driver', required=True)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--frame-states', required=True,
                    help='per-step ordinary-forward frame snapshots (snapshot_hex, frame_snapshot.py)')
    ap.add_argument('--frame-states-from-observation', action='store_true',
                    help='take every step snapshot from --frames; --frame-states then only cross-checks its steps')
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference', required=True)
    a = ap.parse_args()
    plan = json.load(gzip.open(a.plan, 'rt'))
    catalog = json.load(open(a.catalog))
    steps = sorted(int(n.split('_')[1]) for n in plan['blocks'])

    # ---- frame state before step 1 from the recording's own observation
    manager = next(r for r in jsonl(a.frames) if r.get('event') == 'manager')
    calendar = next(r for r in jsonl(a.frames) if r.get('event') == 'calendar')
    cb = calendar['clock_before']
    alarms = {k: {'period': int(v['period']), 'next_ring': int(v['next_ring'])}
              for k, v in manager['alarms_before'].items()}
    active = manager['active']
    frame_id = 900099
    plan['physics'] = {
        'case': 'hrrrv5', 'frame_state_id': frame_id,
        'clock': {'year': cb['year'], 'month': cb['month'], 'day': cb['day'], 'seconds': int(cb['seconds'])},
        'dt_seconds': int(cb['dt']), 'elapsed_seconds': int(cb['elapsed']), 'next_step': 1, 'last_time': None,
        'active': [bool(active['radiation_lw']), bool(active['radiation_sw']), bool(active['convection'])],
        'correct_rp6': bool(manager['correct_rp6']), 'pbl_due': True, 'alarms': alarms,
        'state_time_f32_bits': 0, 'state_xtime': cb['stamp'],
        # P1 sealer-output-steps: the output streams due inside the plan (the 18-step plan writes diagnostics at
        # steps 9 and 18, so 10800 s); empty for the 6-step plan. The last step reads pools in any case.
        'pool_reader_output_seconds': output_seconds(plan['blocks'], int(cb['dt'])),
        'max_steps': len(steps),
        'segments': [{'kind': 'step_blocks', 'blocks': [f'step_{s:02d}' for s in steps]}],
    }
    name_run_steps(plan['physics'], jsonl(a.frames), steps)
    plan.setdefault('host_carried', []).append(frame_id)

    # ---- host boundaries
    tracker = {s['step']: s['begin_sequence'] for s in plan['observed_host_slots']
               if s['operation'] == 'physics_timetracker_state'}
    for name, ops in plan['blocks'].items():
        step = int(name.split('_')[1])
        seq = tracker.get(step)
        idx = 0
        for pos, op in enumerate(ops):
            if seq is not None and op.get('recorded_sequence', 0) and op['recorded_sequence'] < seq:
                idx = pos + 1
        ops.insert(idx, {'kind': 'host_boundary', 'name': 'physics_timetracker_state'})
        ops.append({'kind': 'host_boundary', 'name': 'update_state_time'})

    # ---- events live when step 1 starts (created during setup): created at prepare with their recorded flags
    first_use, created_flags = {}, {}
    for name in sorted(plan['blocks'], key=block_key):  # step order, not text order
        for op in plan['blocks'][name]:
            h = op.get('event_handle_recorded')
            if op.get('kind') == 'synchronization' and h and h not in first_use:
                first_use[h] = op['operation']
    setup_flags = {}
    step1 = min((s['begin_sequence'] for s in plan['observed_host_slots'] if s['step'] == 1), default=None)
    for r in jsonl(a.driver):
        if step1 is not None and r.get('sequence', 0) >= step1:
            break
        if r.get('type') == 'EventCreate' and not r.get('status', 0):
            setup_flags[r['a']] = int(r.get('b', 0))
    plan['events'] = [{'id': h, 'flags': setup_flags.get(h, 0)} for h, first in sorted(first_use.items())
                      if first != 'EventCreate']
    missing_flags = [h for h, first in first_use.items() if first != 'EventCreate' and h not in setup_flags]

    # ---- carried device state and references from the boundary digests
    allocs = catalog['allocations']
    last_clock = 0
    digests = collections.defaultdict(dict)
    for r in jsonl(a.driver):
        if 'clock_ns' in r:
            last_clock = r['clock_ns']
        if r.get('type') != 'BoundaryAllocationDigest':
            continue
        hits = [x for x in allocs if x['recorded_base'] == r['base'] and x['bytes'] == r['bytes']
                and x['allocated_clock_ns'] <= last_clock
                and (x.get('freed_clock_ns') is None or last_clock < x['freed_clock_ns'])]
        if len(hits) != 1:
            raise SystemExit(f'digest base unresolved: {r}')
        digests[r['step']][hits[0]['id']] = (r['bytes'], r['sha256'])
    require_boundary_state(steps, digests)
    common = set.intersection(*(set(digests[s]) for s in steps))
    plan['carried'] = sorted(common)
    # Dynamic allocations made in step 1 are carried too, so the carried set is given per step as views
    # (logical id = the allocation id; whole allocation; main.rs validate_view_schedule).
    sizes = {x['id']: x['bytes'] for x in allocs}
    plan['carried_views'] = [[{'id': ident, 'target': {'allocation': ident, 'offset': 0}, 'bytes': sizes[ident]}
                              for ident in sorted(common)] for _ in steps]
    lines = []
    for s in steps:
        for ident in sorted(common):
            nbytes, sha = digests[s][ident]
            lines.append(json.dumps({'event': 'state_sha256', 'step': s, 'allocation': ident, 'bytes': nbytes, 'sha256': sha}))
    # ---- host carried references: the frame snapshot (ordinary forward, frame_snapshot.py) and each
    # copy-plan cache owner (the forward's last build in or before the step; copy_plan.rs encode_cache layout)
    import hashlib
    import struct
    frames = {}
    for line in open(a.frame_states):
        if line.strip():
            r = json.loads(line)
            frames[r['step']] = bytes.fromhex(r['snapshot_hex'])
    if a.frame_states_from_observation:
        observed = frame_states_from_observation(list(jsonl(a.frames)))
        for step, data in frames.items():
            if step in observed and observed[step] != data:
                raise SystemExit(f'observed frame snapshot differs from {a.frame_states} at step {step}')
        frames = observed
    writes = plan.pop('_copy_plan_cache_writes', [])
    caches = sorted({w['cache'] for w in writes})
    current = {c: bytes(40) for c in caches}
    by_step = collections.defaultdict(list)
    for w in writes:
        by_step[w['step']].append(w)
    for s in steps:
        if s not in frames:
            raise SystemExit(f'no frame snapshot for step {s}')
        lines.append(json.dumps({'event': 'state_sha256', 'step': s, 'allocation': frame_id,
                                 'bytes': len(frames[s]), 'sha256': hashlib.sha256(frames[s]).hexdigest()}))
        for w in by_step.get(s, []):
            current[w['cache']] = (b'ADCP' + struct.pack('<III', w['grid'][0], w['grid'][1], int(w['tiled']))
                                   + struct.pack('<QQQ', w['rows'] * 128, w['table']['allocation'],
                                                 w['table']['offset']))
        for c in caches:
            lines.append(json.dumps({'event': 'state_sha256', 'step': s, 'allocation': c, 'bytes': 40,
                                     'sha256': hashlib.sha256(current[c]).hexdigest()}))
    Path(a.reference).write_text('\n'.join(lines) + '\n')
    plan['reference'] = Path(a.reference).name
    plan['complete'] = False
    with gzip.open(a.out, 'wt') as f:
        json.dump(plan, f, separators=(',', ':'))
    print(json.dumps({'steps': steps, 'carried': len(common), 'setup_events': len(plan['events']),
                      'setup_events_without_recorded_create': len(missing_flags),
                      'digests_per_step': {s: len(digests[s]) for s in steps},
                      'not_carried_every_step': sorted(set().union(*(set(digests[s]) for s in steps)) - common)[:20],
                      'alarms': alarms, 'active': plan['physics']['active']}))


if __name__ == '__main__':
    main()
