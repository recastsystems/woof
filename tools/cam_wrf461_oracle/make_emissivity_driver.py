"""Add isolated emissivity inputs and outputs to the stock column harness."""
from pathlib import Path
import json,re,sys
from make_driver import arguments,runner,INITIALIZE

source=Path(sys.argv[1]).read_text()
support=Path(sys.argv[2]).read_text()
layout=json.loads(Path(sys.argv[3]).read_text())
output=Path(sys.argv[4])
base=output.read_text()
# Each saved module datum is written in the exact shared-layout order.
state='''subroutine dump_state()
use module_ra_cam
implicit none
open(35,file='state.bin',access='stream',form='unformatted',status='replace',convert='little_endian')
'''
for entry in layout:
    state+='write(35) '+entry['name']+'\n'
state+='close(35)\nend subroutine dump_state\n'
base=base.replace('call init()','call init()\ncall dump_state()')
base=base.replace('end program',state+'end program')
# Supply the module constants that the CAM init driver normally sets.
base=base.replace('mwdry=28.966','tmelt=273.16\nlatvap=2.501e6\nlatice=3.336e5\nrair=287.0\ncappa=287.0/1004.5\nzvir=461.6/287.0-1.0\nmwdry=28.966')
start=base.index('subroutine run_lw(')
end=base.index('end subroutine run_lw',start)+len('end subroutine run_lw')
lw=base[start:end]
stages=[('aer_pth',source,{}),('radoz2',source,{'ntoplw':'1'}),
        ('trcpth',support,{'tnm':'tnm','pnm':'pint','qnm':'qnm'}),
        ('aer_trn',source,{}),('radems',source,{'pnm':'pint'})]
# Collect formal declaration identities; map the existing native input names.
known=set(re.findall(r'::\s*(\w+)',lw))|{'pcols','pver','pverp','ncol','lchnk','nbands','bnd_nbr_lw','ntoplw'}
for line in lw.splitlines():
    if '::' in line:
        known.update(re.findall(r'\b\w+\b',line.split('::',1)[1]))
extra=[]; calls=[]
for name,text,alias in stages:
    alias={'plev':'pver','plevp':'pverp',**alias}
    args,declarations=arguments(text,name)
    for arg in args:
        if arg in alias or arg in known:
            continue
        decl=re.sub(r',?\s*intent\(\w+\)','',declarations[arg])
        decl=re.sub(r'\b(?:plev|plevp)\b',lambda m:alias[m[0]],decl)
        extra.append(decl)
        known.add(arg)
    call_args=[alias.get(arg,arg) for arg in args]
    calls += ['call '+name+'(&', ', &\n'.join(','.join(call_args[i:i+6]) for i in range(0,len(call_args),6))+')']
    if name=='radems':
        emit=[]
        for arg in args:
            if arg in ('lchnk','ncol','pcols','pver','pverp'):
                continue
            decl=declarations[arg]
            role='output' if 'intent(out)' in decl else 'input'
            emit.append(f"call dump('radems',icase,'{role}:{arg}',reshape([{alias.get(arg,arg)}],[size([{alias.get(arg,arg)}])]))")
        # Capture inputs before the call and outputs afterwards.
        calls=calls[:-2]+[x for x in emit if "'input:" in x]+calls[-2:]+[x for x in emit if "'output:" in x]
lw=lw.replace('only: radclwmx, radtpl','only: radclwmx, radtpl, aer_pth, radoz2, aer_trn, radems')
lw=lw.replace('only: cldovrlap','only: cldovrlap, trcpth')
lw=lw.replace('naer_groups=7,nbands=2','naer_groups=7,nbands=2,bnd_nbr_lw=7')
lw=lw.replace('group=mod(icase-1,8)','\n'.join(extra)+'\ngroup=mod(icase-1,8)')
lw=lw.replace('call radclwmx(&','\n'.join(calls)+'\ncall radclwmx(&')
base=base[:start]+lw+base[end:]
output.write_text(base)
