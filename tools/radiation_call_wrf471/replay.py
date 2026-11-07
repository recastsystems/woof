"""Replay every radiation decision from two completed strict real WRF runs.

Native debug messages are emitted inside the selected longwave arm, after
module_radiation_driver.F's run_param decision. No native arithmetic is patched.
"""
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,re
import numpy as np
from woof.core.physics import _radiation_step_due
from woof.ingest.wrfrst import _last_radiation_step
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
rows=[]
for label,offset in [('new',0),('old',-1)]:
    case=a.root/('case-'+label);log=case/'rsl.out.0000';text=log.read_text()
    assert 'SUCCESS COMPLETE WRF' in text, str(log)+' did not complete'
    nml=(case/'namelist.input').read_text()
    value=lambda key:float(re.search(r'(?m)^\s*'+key+r'\s*=\s*([\d.+-]+)',nml)[1])
    dt=value('time_step');radt=value('radt');count=int((value('run_hours')*3600+value('run_minutes')*60+value('run_seconds'))/dt)
    start=datetime(2024,5,25,18);calls=[]
    for t in re.findall(r'd01 (\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d) CALL rrtmg_lw',text):
        elapsed=(datetime.strptime(t,'%Y-%m-%d_%H:%M:%S')-start).total_seconds()
        calls.append(int(elapsed/dt)+1)
    assert len(set(calls))==len(calls),(label,'duplicate native call messages')
    assert calls and all(1<=s<=count for s in calls),(label,calls,count)
    stepra=max(int(np.floor(np.float32(radt)*np.float32(60)/np.float32(dt)+np.float32(.5))),1)
    expected=np.array([int(s in calls) for s in range(1,count+1)],dtype='<i4')
    actual=np.array([int(_radiation_step_due(s,stepra,radt,offset)) for s in range(1,count+1)],dtype='<i4')
    differences=int(np.count_nonzero(expected.view('<u4')!=actual.view('<u4')))
    # A resumed urban/solar driver must hold the sun from the same native call.
    last_checks=[_last_radiation_step(s,stepra,offset)==max(c for c in calls if c<=s) for s in range(1,count+1)]
    expected.tofile(case/'radiation-decision.native.i32');actual.tofile(case/'radiation-decision.port.i32')
    row=dict(offset=offset,steps=count,stepra=stepra,native_calls=calls,words=count,differing_words=differences,restart_last_call_matches=all(last_checks),log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    rows.append(row)
    assert differences==0 and all(last_checks),row
result=dict(scope='radiation-driver decision words and retained solar-call indices; radiation-scheme fields are outside this selector replay',rows=rows,words=sum(r['words'] for r in rows),differing_words=sum(r['differing_words'] for r in rows),undefined_exclusions=[])
(a.root/'radiation-replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
