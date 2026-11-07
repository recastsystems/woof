import sys
from pathlib import Path
from datetime import datetime
import json,numpy as np,cupy as cp
from woof.eclipse_windows import EclipseGeometry
from woof.core.solar import solar_geometry_device
root=Path(sys.argv[1]).resolve()
from netCDF4 import Dataset
with Dataset(root/'eclipse-prepared/wrf-native-input/wrfinput_d01') as f:
 lat=np.array(f['XLAT'][0]);lon=np.array(f['XLONG'][0])
g=EclipseGeometry(lat,lon);valid=datetime(2024,4,8,19)
def timing(fn):
 for _ in range(3):fn()
 s=cp.cuda.Stream(non_blocking=True)
 with s:
  s.begin_capture()
  for _ in range(100):fn()
  graph=s.end_capture();a,b=cp.cuda.Event(),cp.cuda.Event();a.record();graph.launch(s);b.record();b.synchronize()
 return cp.cuda.get_elapsed_time(a,b)/100
r={'columns':lat.size,'scope':'100-call CUDA graph; production adapters; real eclipse-case geography','eclipse_ms':timing(lambda:g.at(valid)),'solar_geometry_ms':timing(lambda:solar_geometry_device(98.5,420,12,g.latitude,g.longitude))}
r['ratio']=r['eclipse_ms']/r['solar_geometry_ms']
(root/'evidence/resume/eclipse-stage-speed.json').write_text(json.dumps(r,indent=2));print(r)
