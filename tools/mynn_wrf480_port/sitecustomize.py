"""Optional forecast-only stage observer; enabled by MYNN_STAGE_LOG.

Records, per MYNN call, the device time of the whole PBL stage and, on the
generated routes, of the column driver (pack + column launches) inside it.
A separate observer run only: the events add host waits.
"""
import os
import sys
if os.environ.get('MYNN_STAGE_LOG'):
 import atexit
 import json
 import time
 import cupy as cp
 from woof.core.physics import PhysicsDriver
 original=PhysicsDriver._run_mynn_pbl
 samples=[]
 inner=[]
 packs=[]
 def observed(self,atmosphere,cfg):
  begin=cp.cuda.Event();end=cp.cuda.Event()
  begin.record();start=time.perf_counter()
  result=original(self,atmosphere,cfg)
  end.record();end.synchronize()
  samples.append(dict(step=len(samples)+1,device_ms=cp.cuda.get_elapsed_time(begin,end),
                      wall_ms=1000*(time.perf_counter()-start),
                      driver_ms=sum(inner) if inner else None,
                      pack_ms=sum(packs) if packs else None,
                      column_ms=sum(columns) if columns else None))
  inner.clear();packs.clear();columns.clear()
  return result
 PhysicsDriver._run_mynn_pbl=observed
 import woof.core.mynn_pbl_runtime as _runtime
 _column=_runtime.mynn_bl_driver_cuda
 columns=[]
 def _timed_column(*args,**kwargs):
  b=cp.cuda.Event();e=cp.cuda.Event();b.record()
  out=_column(*args,**kwargs)
  e.record();e.synchronize();columns.append(cp.cuda.get_elapsed_time(b,e))
  return out
 _runtime.mynn_bl_driver_cuda=_timed_column
 for module_name in ('woof.core.mynn480_gpu','woof.core.mynn471_gpu'):
  try:
   module=__import__(module_name,fromlist=['driver'])
  except Exception:
   continue
  def wrap(fn):
   def timed(*args,**kwargs):
    b=cp.cuda.Event();e=cp.cuda.Event();b.record()
    out=fn(*args,**kwargs)
    e.record();e.synchronize();inner.append(cp.cuda.get_elapsed_time(b,e))
    return out
   return timed
  module.driver=wrap(module.driver)
  if hasattr(module,'pack_inputs'):
   def wrap_pack(fn):
    def timed(*args,**kwargs):
     b=cp.cuda.Event();e=cp.cuda.Event();b.record()
     out=fn(*args,**kwargs)
     e.record();e.synchronize();packs.append(cp.cuda.get_elapsed_time(b,e))
     return out
    return timed
   module.pack_inputs=wrap_pack(module.pack_inputs)
 def save():
  from pathlib import Path
  if not samples:return
  Path(os.environ['MYNN_STAGE_LOG']).write_text(json.dumps(dict(pid=os.getpid(),samples=samples),indent=2)+'\n')
 atexit.register(save)
