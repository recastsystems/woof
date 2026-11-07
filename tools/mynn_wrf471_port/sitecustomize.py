"""Optional forecast-only stage observer; enabled by MYNN_STAGE_LOG."""
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
 def observed(self,atmosphere,cfg):
  begin=cp.cuda.Event();end=cp.cuda.Event()
  begin.record();start=time.perf_counter()
  result=original(self,atmosphere,cfg)
  end.record();end.synchronize()
  samples.append(dict(step=len(samples)+1,device_ms=cp.cuda.get_elapsed_time(begin,end),
                      wall_ms=1000*(time.perf_counter()-start)))
  return result
 PhysicsDriver._run_mynn_pbl=observed
 def save():
  from pathlib import Path
  if not samples:return
  Path(os.environ['MYNN_STAGE_LOG']).write_text(json.dumps(dict(pid=os.getpid(),samples=samples),indent=2)+'\n')
 atexit.register(save)
