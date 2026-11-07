"""Run the real CLI with asynchronous per-stage timing for proof receipts.

No synchronization is inserted between stages. Device events are read only
once the forecast has completed its normal final-state check and output drain.
"""
import functools
import json
import os
import statistics
from pathlib import Path
import time
import cupy as cp
from woof.core import dycore, physics, microphysics, acoustic

records = {}

def timed(original, label):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            wall = time.perf_counter() - started
            end.record()
            records.setdefault(label, []).append((wall, begin, end))
    return wrapped

def instrument(owner, attr, label):
    setattr(owner, attr, timed(getattr(owner, attr), label))

for owner, attr, label in (
    (dycore, 'step', 'full_step'),
    (dycore, '_compute_wrf_smag_tendencies', 'diffusion'),
    (dycore, 'prepare_fixed_tendencies', 'fixed_tendencies'),
    (physics.PhysicsDriver, 'compute', 'surface_radiation_pbl'),
    (dycore, 'apply_microphysics', 'microphysics'),
    (dycore, '_add_slow_tendencies', 'slow_dynamics'),
):
    instrument(owner, attr, label)

original_acoustic_prepare = dycore.prepare_acoustic_substep_launch
def prepare_timed_acoustic(*args, **kwargs):
    launch = original_acoustic_prepare(*args, **kwargs)
    result = timed(launch, 'acoustic')
    result.absorbs = getattr(launch, 'absorbs', frozenset())
    return result
dycore.prepare_acoustic_substep_launch = prepare_timed_acoustic

if __name__ == '__main__':
    os.environ.setdefault('WOOF_KEEP_CHECKPOINTS', '0')
    if os.environ.get('KM5_PREPARED') == '1':
        from woof.prepared_single_domain_forecast import main
    else:
        from woof.cli import main
    try:
        raise SystemExit(main())
    finally:
        result = {'schema': 'km5-asynchronous-stage-timing-v1',
                  'added_stage_synchronizations': 0, 'stages': {}}
        for name, rows in records.items():
            # A launch which failed before the forecast's final drain is
            # not a speed receipt.
            ready = all(cp.cuda.runtime.eventQuery(b.ptr) == 0 for _, _, b in rows)
            result['stages'][name] = dict(calls=len(rows),
                host_wall_seconds=sum(w for w, _, _ in rows),
                device_events_complete=ready,
                device_seconds=sum(cp.cuda.get_elapsed_time(a,b) for _,a,b in rows)/1000 if ready else None,
                warm_host_wall_seconds=sum(w for w,_,_ in rows[1:]),
                warm_device_seconds=sum(cp.cuda.get_elapsed_time(a,b) for _,a,b in rows[1:])/1000 if ready else None,
                median_host_ms=statistics.median(w*1000 for w,_,_ in rows),
                median_device_ms=statistics.median(cp.cuda.get_elapsed_time(a,b) for _,a,b in rows) if ready else None,
                step_host_ms=[w*1000 for w,_,_ in rows] if name == 'full_step' else None,
                step_device_ms=[cp.cuda.get_elapsed_time(a,b) for _,a,b in rows] if ready and name == 'full_step' else None)
        Path(os.environ['KM5_TIMING_OUT']).write_text(json.dumps(result, indent=2)+'\n')
