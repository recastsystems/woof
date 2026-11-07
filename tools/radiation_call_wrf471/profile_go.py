"""Time radiation and its scheduler inside the complete woof go forecast.

CUDA events are collected without a synchronization on each timestep.
"""
import atexit,functools,json,runpy,sys,time
from pathlib import Path
import numpy as np
import cupy as cp
import woof.core.physics as physics
out=Path(sys.argv.pop(1)); calls=[]; decisions=[]
original=physics.PhysicsDriver._radiation if hasattr(physics.PhysicsDriver,'_radiation') else None
# The stable producer is run_radiation; resolve its actual class spelling below.
method=next((n for n in ('_run_radiation','_apply_radiation','_radiation','radiation') if hasattr(physics.PhysicsDriver,n)),None)
if method is None:
    candidates=[n for n in vars(physics.PhysicsDriver) if 'radiation' in n]
    raise RuntimeError('Identify radiation stage: '+repr(candidates))
orig=getattr(physics.PhysicsDriver,method)
@functools.wraps(orig)
def timed(*args,**kwargs):
    a,b=cp.cuda.Event(),cp.cuda.Event();a.record()
    try:return orig(*args,**kwargs)
    finally:b.record();calls.append((a,b))
setattr(physics.PhysicsDriver,method,timed)
predicate=physics._radiation_step_due
@functools.wraps(predicate)
def schedule(*args,**kwargs):
    t=time.perf_counter_ns();value=predicate(*args,**kwargs)
    elapsed=time.perf_counter_ns()-t
    timings={}
    # Alternate order on successive production steps, keeping both phases
    # warm in the same process. These calls only decide; they launch no GPU work.
    phases=(0,-1) if len(decisions)%2==0 else (-1,0)
    for phase in phases:
        began=time.perf_counter_ns()
        predicate(*args[:3],phase)
        timings[str(phase)]=time.perf_counter_ns()-began
    decisions.append(dict(args=list(args),due=bool(value),cpu_ns=elapsed,interleaved_phase_ns=timings));return value
physics._radiation_step_due=schedule
t0=time.monotonic()
def finish():
    cp.cuda.Device().synchronize()
    ms=[float(cp.cuda.get_elapsed_time(a,b)) for a,b in calls]
    out.write_text(json.dumps({'stage':method,'calls':len(ms),'all_ms':ms,'median_after_first_ms':float(np.median(ms[1:])) if len(ms)>1 else None,'decisions':decisions,'wall_s':time.monotonic()-t0},indent=2)+'\n')
atexit.register(finish)
sys.argv=['woof','go',*sys.argv[1:]]
runpy.run_module('woof',run_name='__main__')
