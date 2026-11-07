"""One real go forecast with interleaved stage samples against NSCV.

The reference uses the same atmospheric inputs in isolated diagnostic/rain
storage. Only the selected stage is coupled into the forecast. Alternate
the call order each step; discard ten startup steps. Run on a quiet card.
"""
import atexit
import copy
import json
import os
from pathlib import Path
import sys
import time
import cupy as cp
from woof.core import shallow_runtime

original = shallow_runtime.before_compose
samples = []
proxy = None
step = 0
start_load = Path('/proc/loadavg').read_text().strip()


def measured(driver, atmosphere, state, cfg):
    global proxy, step
    step += 1
    if proxy is None:
        proxy = copy.copy(driver)
        proxy.fields = dict(driver.fields)
        for name in ('rainsh','hbot','htop','surface_raincv'):
            value = driver.fields.get(name)
            proxy.fields[name] = (cp.zeros(state.p.shape[1:], cp.float32)
                                  if value is None else value.copy())
        proxy._pending_rainbl = driver._pending_rainbl.copy()
        proxy._cam_workspace = None
        proxy._nscv_output = None
        proxy.call_counts = dict(driver.call_counts)
    reference = copy.copy(cfg)
    object.__setattr__(reference, 'shcu_physics', 4)
    order = [('selected',driver,cfg),('nscv',proxy,reference)]
    if step % 2 == 0: order.reverse()
    cp.cuda.get_current_stream().synchronize()
    row = dict(step=step,order=[x[0] for x in order])
    for label, owner, configuration in order:
        begin,end = cp.cuda.Event(),cp.cuda.Event()
        begin.record();started=time.perf_counter()
        original(owner,atmosphere,state,configuration)
        end.record();end.synchronize()
        row[label] = dict(gpu_ms=cp.cuda.get_elapsed_time(begin,end),
                          wall_ms=(time.perf_counter()-started)*1000)
    samples.append(row)


def save():
    dest = os.environ.get('WOOF_SHALLOW_TIMINGS')
    if dest:
        warm = [s for s in samples if s['step'] > 10]
        totals = {name:sum(s[name]['wall_ms'] for s in warm)
                  for name in ('selected','nscv')}
        Path(dest).write_text(json.dumps(dict(samples=samples,warm_totals_ms=totals,
            start_load=start_load,end_load=Path('/proc/loadavg').read_text().strip(),
            ratio=totals['selected']/totals['nscv'] if totals['nscv'] else None),indent=2)+'\n')


if __name__ == '__main__':
    # Parallel prewarm uses spawn: workers must not start another forecast
    # or overwrite the parent's stage receipt when this module is imported.
    atexit.register(save)
    shallow_runtime.before_compose = measured
    from woof.cli import main
    arguments=sys.argv[1:]
    # Cap qualification forecasts at one hour, including older queued arms.
    if '--run-seconds' in arguments:
        duration_index=arguments.index('--run-seconds')+1
        arguments[duration_index]=str(min(3600,int(arguments[duration_index])))
    print('Qualification arguments:',arguments,flush=True)
    raise SystemExit(main(['go',*arguments]))
