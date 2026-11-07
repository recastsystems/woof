from datetime import datetime
from pathlib import Path
import json
import sys
import numpy as np
from woof.table_assets import irrigation_calendar_scalars
from woof.core.swint import calendar_scalars
from woof.core.fp32_ulp import fp32_ulp_distance
root=Path(__file__).parent/'fixtures'
refs=np.loadtxt(root/'clock-oracle.txt')[:,1:].astype(np.float32)
new=[];old=[]
for line in (root/'clock-cases.txt').read_text().splitlines():
    index,start,valid,elapsed=line.split()
    time=datetime.strptime(start,'%Y-%m-%d_%H:%M:%S.%f')
    new.append(irrigation_calendar_scalars(time,float(elapsed)))
    old.append(calendar_scalars(time,float(elapsed)))
actual=np.asarray(new,np.float32); previous=np.asarray(old,np.float32)
x=fp32_ulp_distance(actual,refs); prior=fp32_ulp_distance(previous,refs)
result={'cases':len(refs),'fields':['julian','xtime','gmt'],'max_ulp':x.max(axis=0).tolist(),'previous_solar_helper_max_ulp':prior.max(axis=0).tolist(),'pass':bool((x==0).all())}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if result['pass'] else 1)
