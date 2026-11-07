"""Capture the native MM5 fractional wrapper around both real surface calls."""
from pathlib import Path
import re,json,sys
p=Path(sys.argv[1])/'phys/module_surface_driver.F';s=p.read_text();dest=Path(sys.argv[2]);dest.mkdir(exist_ok=True,parents=True)
outputs='znt ust ustm mol hfx qfx qsfc zol regime psim psih fm fh lh u10 v10 th2 t2 q2 chs chs2 cqs2 flhc flqc qgh rmol wspd br gz1oz0 cpm ck cka cd cda'.split()
wait='chs2 chs cpm cqs2 flhc flqc hfx lh qfx qgh qsfc znt'.split()
f2=list(dict.fromkeys(['psfc','tsk','pblh','mavail','xland','lakemask','xice','sst','tsk_sea',*outputs,*[n+'_sea' for n in wait]]))
for name,option in [('sfclayrev_seaice_wrapper',1),('sfclay_seaice_wrapper',91)]:
 m=re.search(r'(?ims)^\s*SUBROUTINE '+name+r'\b.*?^\s*END SUBROUTINE '+name,s);assert m,name
 b=m.group(0);mark='     CALL get_local_ice_tsk';assert mark in b
 declare='''     INTEGER :: cap_unit
     CHARACTER(len=160) :: cap_file
'''
 def dump(arm):
  lines=['     IF(ITIMESTEP.LE.3) THEN',f'      WRITE(cap_file,\'(A,I0,A,I0,A,I0,A)\') "captures/fractional-",ITIMESTEP,"-",ITS,"-",JTS,"-{arm}.bin"','      OPEN(newunit=cap_unit,file=trim(cap_file),access="stream",form="unformatted",status="replace",convert="little_endian")',f'      WRITE(cap_unit) ite-its+1,jte-jts+1,itimestep,isfflx,isftcflx,iz0tlnd,merge(1,0,tice2tsk_if2cold),sf_surface_physics,{option}','      WRITE(cap_unit) XICE_THRESHOLD']
  for n in f2:lines+=['      WRITE(cap_unit) '+n.upper()+'(its:ite,jts:jte)']
  lines+=['      WRITE(cap_unit) '+('DX2D(its:ite,jts:jte)' if option==1 else '0.0*TSK(its:ite,jts:jte)+DX')]
  for n in ('U3D','V3D','T3D','QV3D','P3D','DZ8W'):lines+=['      WRITE(cap_unit) '+n+'(its:ite,kts,jts:jte)']
  lines+=['      CLOSE(cap_unit)','     ENDIF'];return '\n'.join(lines)+'\n'
 b=b.replace(mark,declare+dump('in')+mark,1)
 end=re.search(r'(?im)^\s*END SUBROUTINE '+name,b);b=b[:end.start()]+dump('out')+b[end.start():]
 s=s[:m.start()]+b+s[m.end():]
p.write_text(s);(dest/'fractional-layout.json').write_text(json.dumps({'f2':f2+['dx','u','v','temperature','qv','pressure','dz'],'outputs':outputs,'wait':wait},indent=2)+'\n')
