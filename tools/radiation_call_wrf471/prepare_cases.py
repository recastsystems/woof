"""Prepare two real-input WRF cases differing only in radiation call phase."""
from pathlib import Path
import re, shutil, sys, json
source, root = map(Path, sys.argv[1:])
for label, offset in [('new',0), ('old',-1)]:
    case=root/('case-'+label); case.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.name in ('namelist.input','namelist.output') or p.name.startswith(('rsl.','wrfout','wrfrst')): continue
        target=case/p.name
        if not target.exists():target.symlink_to(p.resolve())
    text=(source/'namelist.input').read_text()
    for key,value in {'run_hours':0,'run_minutes':12,'end_day':25,'end_hour':18,'end_minute':12,'history_interval':6,'debug_level':100,'radt':1}.items():
        text,n=re.subn(r'(?m)^(\s*'+key+r'\s*=)[^,\n]*',lambda m:m[1]+' '+str(value),text)
        assert n==1,(key,n)
    text=text.replace('&physics\n','&physics\n ra_call_offset = '+str(offset)+',\n')
    (case/'namelist.input').write_text(text)
(root/'case-provenance.json').write_text(json.dumps({'source':str(source),'duration_s':720,'dt_s':12,'radt_minutes':1,'offsets':[0,-1]},indent=2)+'\n')
