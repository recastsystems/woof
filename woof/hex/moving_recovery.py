"""Durable recovery points are distinct from resident move hand-off receipts."""
from pathlib import Path
import json
import math


def durable_carriers(state):
    """Inventory closed, on-disk carriers without reading their field payloads."""
    rows = []
    for segment in state['segments']:
        folder = Path(segment['path'])
        paths = [folder / 'boundary.pkl', *sorted((folder / 'control').glob('seal-*.pkl'))]
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                continue
            try:
                record = json.loads(path.with_suffix('.pkl.json').read_text())
                seconds = float(record['time_seconds'])
                if not math.isfinite(seconds) or seconds < 0:
                    continue
                if record.get('resident') and (record.get('bytes') is None or record.get('sha256') is None):
                    continue
                if record.get('bytes') is not None and int(record['bytes']) != path.stat().st_size:
                    continue
                if seconds > segment.get('superseded_after_seconds', math.inf):
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            rows.append(dict(path=str(path), time_seconds=seconds, member=segment['member']))
    # At an equal clock prefer the segment's final carrier over its periodic seal.
    return sorted(rows, key=lambda row: (row['time_seconds'], Path(row['path']).name == 'boundary.pkl'),
                  reverse=True)


def resident_receipt_only(path):
    if not path:
        return False
    path = Path(path)
    try:
        record = json.loads(path.with_suffix('.pkl.json').read_text())
    except (OSError, ValueError):
        return False
    return bool(record.get('resident') and (not path.is_file() or record.get('bytes') is None))


def retain_latest_two(controller):
    """Keep current and previous durable clocks, including across resident moves."""
    rows = durable_carriers(controller.state)
    keep, clocks = [], set()
    for row in rows:
        if row['time_seconds'] not in clocks and len(keep) < 2:
            keep.append(row)
            clocks.add(row['time_seconds'])
    protected = {row['path'] for row in keep}
    controller.state['recovery_carriers'] = keep
    for row in rows:
        if row['path'] not in protected:
            controller.delete_owned(row['path'], 'superseded; latest two durable recovery clocks retained')


def recover_lost_resident(controller):
    """After process death, replace its inaccessible hand-off with a real seal."""
    state = controller.state
    if not resident_receipt_only(state.get('boundary')):
        return False
    choices = [r for r in durable_carriers(state) if r['time_seconds'] <= float(state['t'])]
    if not choices:
        raise RuntimeError('resident move carrier was lost with its forecast process and no durable recovery '
                           'seal remains; restore a saved checkpoint or restart from the initial analysis')
    chosen = choices[0]
    seconds, member = chosen['time_seconds'], chosen['member']
    analyses = [r['cycle_analysis']['time_seconds'] for r in state['segments'] if r.get('cycle_analysis')]
    if analyses and seconds < max(analyses):
        raise RuntimeError('the newest durable recovery seal predates a cycle analysis; explicit recovery '
                           'with that analysis is required')
    before = dict(t=state['t'], current=state['current'], boundary=state['boundary'])
    same_start = seconds == float(state['t']) and member == controller.carrier_member()
    state['boundary'] = chosen['path']
    if same_start:
        # The original operator still maps this carrier to the failed segment's member.
        dropped = controller.forget_after(seconds)
    else:
        state.update(t=seconds, current=member, operator=None, source_grid=None)
        state['members'][member]['status'] = 'ready'
        # Observe the recovered carrier's frame again, including any move decided there.
        dropped = controller.forget_after(math.nextafter(seconds, -math.inf))
        state['moves'] = [m for m in state['moves'] if float(m['at_seconds']) < seconds]
        for segment in state['segments']:
            if segment.get('end_seconds', math.inf) > seconds:
                segment['superseded_after_seconds'] = seconds
    state.update(t_seen=seconds, pending_move=None)
    controller.event('recover-lost-resident', previous=before, carrier=chosen,
                     frames_discarded=dropped, preserved_move_mapping=same_start)
    controller.save()
    return True
