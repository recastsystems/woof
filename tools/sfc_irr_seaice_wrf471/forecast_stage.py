"""Record CUDA stage spans while the production go door runs a whole forecast."""
import json,sys,time,os
from pathlib import Path
import cupy as cp
def run():
 from woof.core.physics import PhysicsDriver
 records=[]
 for name in ('_run_noah','_run_noahmp','_run_sfclay','apply_irrigation'):
  original=getattr(PhysicsDriver,name)
  def timed(self,*args,_name=name,_original=original,**kw):
   begin=cp.cuda.Event();end=cp.cuda.Event();host=time.perf_counter();begin.record()
   result=_original(self,*args,**kw);end.record()
   records.append((_name,begin,end,time.perf_counter()-host));return result
  setattr(PhysicsDriver,name,timed)
 from woof.cli import main
 try:
  status=main(sys.argv[2:])
 finally:
  rows=[]
  for name,begin,end,wall in records:
   end.synchronize();rows.append({'stage':name,'gpu_ms':cp.cuda.get_elapsed_time(begin,end),'host_seconds':wall})
  Path(sys.argv[1]).write_text(json.dumps({'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'stages':rows},indent=2)+'\n')
 return status

if __name__ == '__main__':
 raise SystemExit(run())
