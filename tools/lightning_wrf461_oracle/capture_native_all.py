"""Passive, bounded driver capture on an owned public-source native tree."""
from pathlib import Path
import json,hashlib
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wrf-source',required=True,type=Path)
parser.add_argument('--receipt',required=True,type=Path)
args=parser.parse_args()
root=args.wrf_source.resolve()
assert hashlib.sha256((root/'LICENSE.txt').read_bytes()).hexdigest()=='b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6'
p=root/'phys/module_lightning_driver.F' 
a=p.read_text()
assert hashlib.sha256(a.encode()).hexdigest()=='97ac20aa241c82d8af11fc9df587c1146fec762d7cd6918d48d3c796adff6980'
decl=''' INTEGER,SAVE :: gc_count=0
 INTEGER :: gc_u,gc_status
 LOGICAL :: gc_active
 CHARACTER(LEN=2048) :: gc_dir,gc_file
'''
inputs=[]
for name in ('xlat','xlon','xland','ht','ktop_deep'):inputs.append((name,'(its:ite,jts:jte)'))
for name in ('t_phy','z','w','refl','dz8w','th_phy','pi_phy'):inputs.append((name,'(its:ite,kts:kte,jts:jte)'))
for name in ('QV','QC','QR','QI','QS','QG'):inputs.append(('moist',f'(its:ite,kts:kte,jts:jte,P_{name})'))
for name in ('iccg_in_num','iccg_in_den'):inputs.append((name,'(its:ite,jts:jte,1)'))
outputs=['ic_flashrate','cg_flashrate','ic_flashcount','cg_flashcount','lpi']
write=lambda rows:''.join(' WRITE(gc_u) '+name+sel+'\n' for name,sel in rows)
pre=''' CALL GET_ENVIRONMENT_VARIABLE('WOOF_LTNG_CAPTURE_DIR',gc_dir,STATUS=gc_status)
 gc_active=gc_status==0 .AND. LEN_TRIM(gc_dir)>0 .AND. gc_count<3
 IF(gc_active) THEN
 gc_count=gc_count+1
 WRITE(gc_file,'(A,"/lightning-",I2.2,"-",I2.2,".bin")') TRIM(gc_dir),lightning_option,gc_count
 OPEN(NEWUNIT=gc_u,FILE=TRIM(gc_file),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
 WRITE(gc_u) 47104,ite-its+1,jte-jts+1,kte-kts+1,lightning_option,iccg_method,cellcount_method
 WRITE(gc_u) dt,dx,dy,lightning_dt,flashrate_factor,iccg_prescribed_num,iccg_prescribed_den,cldtop_adjustment
 WRITE(gc_u) curr_secs,ltngacttime
'''+write(inputs)+write([(name,'(its:ite,jts:jte)') for name in outputs])+ ' END IF\n'
post=' IF(gc_active) THEN\n'+write([(name,'(its:ite,jts:jte)') for name in outputs])+' WRITE(gc_u) ltngacttime\n CLOSE(gc_u)\n END IF\n'
a=a.replace(' LOGICAL :: do_ltng\n',' LOGICAL :: do_ltng\n'+decl,1)
anchor='! This driver performs several steps in order to produce lightning'
a=a.replace(anchor,pre+'\n'+anchor,1)
anchor=" CALL wrf_debug ( 100, ' lightning_driver: returning from')"
assert anchor in a
a=a.replace(anchor,post+anchor,1)
p.write_text(a)
args.receipt.write_text(json.dumps({'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'limit':'first three activated calls, full native tile; one MPI rank required'},indent=2)+'\n')
