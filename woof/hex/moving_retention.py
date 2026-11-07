"""Retire inactive member artifacts after external evidence collection.

Only the controller calls this, once per new tracking frame. Static
fields stay available for later surface/rain rendering. This is opt-in storage
management, never a forecast-step operation or a DA carrier retirement policy.
"""
from pathlib import Path
import json
import os


def artifacts_present(member):
    return all(member.get(key) and Path(member[key]).is_file()
               for key in ('grid', 'static', 'init'))


def rebuild_name(name, members):
    """Keep retired identities and their retained statics immutable.

    A rebuilt grid may have a different cell order. Earlier frame receipts must
    still resolve to their original member, even after a return to its centre.
    Zone members retain their source suffix for the existing cache lookup.
    """
    stem, separator, source = name.partition('x')
    candidate, generation = name, 0
    while members.get(candidate, {}).get('status') == 'retired':
        generation += 1
        candidate = f'{stem}r{generation}{separator}{source}'
    return candidate


def retire_finished_preparations(controller, evidence):
    """Release unused speculation only after the entire forecast has closed.

    Also handles a completed older run before extending it with --resume.
    Used members retain their existing evidence-dependent retirement policy.
    """
    state = controller.state
    if (not evidence or not state.get('finished_utc') or controller.procs
            or controller.prelaunched or state.get('pending_move')
            or state.get('t', 0) < state.get('hours', float('inf')) * 3600
            or any(s.get('rc') != 0 or 'end_seconds' not in s
                   for s in state['segments'])):
        return
    protected = {state['current'], controller.a.start_member}
    protected.update(row['member'] for row in state.get('recovery_carriers', []))
    protected.update(s['member'] for s in state['segments'])
    protected.update(f['member'] for f in state['frames'])
    for name, member in state['members'].items():
        if (name not in protected and member.get('grid') != state.get('source_grid')
                and member['status'] in ('ready', 'prepared', 'preparing', 'stale', 'cancelled')):
            member['status'] = 'cancelled'
    retire_members(controller, evidence)


def speculative_admission(state, limit):
    """Bound retained future members, rather than only simultaneous builders."""
    if not limit:
        return True
    return sum(name != state['current'] and row['status'] in ('preparing', 'prepared', 'ready')
               for name, row in state['members'].items()) < limit


def retire_members(controller, evidence):
    if not evidence:
        return
    state = controller.state
    root = controller.members_dir.resolve()
    # External/shared member caches are deliberately never retired.
    if not root.is_relative_to(controller.out.resolve()):
        raise ValueError('member retirement requires a cache inside this run')
    protected = {state['current'], controller.a.start_member}
    protected.update(row['member'] for row in state.get('recovery_carriers', []))
    if state.get('pending_move'):
        protected.add(state['pending_move']['target'])
    if controller.prelaunched:
        protected.add(controller.prelaunched['target'])
    for name, member in state['members'].items():
        if name in protected or member['status'] not in ('stale', 'cancelled'):
            continue
        if member.get('grid') and member['grid'] == state.get('source_grid'):
            continue  # still required by the current remap/recovery carrier
        if any(name in key.split(':', 1)[-1].split('--') for key in controller.procs):
            continue
        segments = [s for s in state['segments'] if s['member'] == name]
        if any(s.get('rc') != 0 or 'end_seconds' not in s for s in segments):
            continue
        frames = [f for f in state['frames'] if f['member'] == name]
        cancelled = member['status'] == 'cancelled'
        if cancelled and (frames or segments):
            continue  # cancellation is only disposable before any forecast use
        ready = True
        for frame in frames:
            receipt = Path(evidence) / f"s{frame['segment']:03d}-t{int(frame['model_seconds']):07d}" / 'receipt.json'
            if not receipt.is_file() or Path(frame['history']).exists():
                ready = False
                break
            record = json.loads(receipt.read_text())
            same_frame = all(record.get('frame', {}).get(k) == frame[k]
                             for k in ('member', 'segment', 'model_seconds'))
            if (record.get('history') != frame['history'] or not same_frame
                    or record.get('raw_retirement_permitted') is not True):
                ready = False
                break
        if not ready:
            continue
        # Grid/init/centre files are reconstructible from retained source and
        # build receipts. Operators involving a retired grid must be rebuilt.
        # Operators go first: interruption must never leave an old operator
        # available after its member has lost geometry and must be regenerated.
        paths = [p for p in (root / 'operators').glob('*.npz')
                 if name in p.name.removesuffix('.npz').removesuffix('.zone').split('--')]
        if cancelled:
            # An interrupted speculative preparation can leave closed grid and
            # partial static/init stages. It has no frame geometry to retain.
            paths.extend(root.glob(f'{name}.*.nc*'))
        else:
            paths.extend(Path(member[k]) for k in ('grid', 'init'))
            paths.append(root / f'{name}.centres.nc')
        paths = [p for p in paths if p.exists()]
        if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in paths):
            continue
        rows = [dict(path=str(p), bytes=p.stat().st_size, member=name,
                     reason='inactive member; all acknowledged histories collected',
                     evidence=str(Path(evidence).resolve())) for p in paths]
        # Persist the exact deletion ledger before unlinking any artifact.
        with (controller.out / 'retired-members.jsonl').open('a') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        member['status'] = 'retired'
        state['deleted'].extend(rows)
        controller.save()
        for path in paths:
            path.unlink()
        controller.event('member-retired', name=name, bytes=sum(r['bytes'] for r in rows),
                         static_retained=None if cancelled else member['static'])
        controller.save()
