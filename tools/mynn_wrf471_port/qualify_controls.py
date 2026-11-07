#!/usr/bin/env python3
"""Replay the specific Registry controls required by the port audit."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('work',type=Path);p.add_argument('captures',type=Path)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--start',type=int,default=0)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rows=[('bl_mynn_edmf','0'),('icloud_bl','0'),('tke_budget','1'),
      ('bl_mynn_cloudpdf','0'),('bl_mynn_cloudpdf','1'),
      ('bl_mynn_mixlength','0'),('bl_mynn_edmf_mom','0'),
      ('bl_mynn_edmf_tke','1'),('bl_mynn_cloudmix','0'),
      ('bl_mynn_mixqt','1'),('bl_mynn_output','1'),
      ('bl_mynn_closure','2.5'),('bl_mynn_closure','3.0')]
for name,value in rows[a.start:]:
    row=name+'-'+value.replace('.','p')
    out=a.out/row
    command=[sys.executable,str(Path(__file__).with_name('replay.py')),
             str(a.work),str(a.captures),'--out',str(out),
             '--image',str(a.work/'mynn471-strict.cubin')]
    defaults={'tke_budget':'0','bl_mynn_output':'0'}
    defaults['closure' if name=='bl_mynn_closure' else name]=value
    for key,val in defaults.items():command+=['--set',key+'='+val]
    print(json.dumps(dict(row=row,command=command)),flush=True)
    with (a.out/(row+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((out/'comparison.json').read_text())
    print(json.dumps(dict(row=row,compared=report['compared'],
                         differing=report['differing'])),flush=True)
