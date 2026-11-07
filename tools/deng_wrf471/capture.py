"""Capture Deng's full real native driver state before and after its call."""
import argparse
import json
from pathlib import Path
import re
import translate

p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('base',type=Path);a=p.parse_args()
_,rows=translate.routines();r=rows[0]
meta=[];before=[];after=[]
for name in r['args'][18:]:
    v=r['vars'][name]
    if not v.dims:
        section=name
    elif name=='dsigma':section=name+'(kts:kte+1)'
    elif len(v.dims)==2:section=name+'(its:ite,jts:jte)'
    elif len(v.dims)==3:
        levels='1:100' if name=='ainckfsa' else 'kts:kte+1' if name in ('w','z_at_w') else 'kts:kte'
        section=name+'(its:ite,'+levels+',jts:jte)'
    else:raise ValueError(v)
    meta.append(dict(name=name,type=v.typ,dims=v.dims,intent=v.intent,section=section))
    before.append('write(dcap_unit) '+section)
    if v.intent in ('out','inout'):after.append('write(dcap_unit) '+section)
s=translate.SOURCE.read_text()
decl='''
integer :: dcap_unit,dcap_status
character(512) :: dcap_dir,dcap_file
logical :: dcap_on
'''
start='''
call get_environment_variable('WOOF_DENG_CAPTURE',dcap_dir,status=dcap_status)
dcap_on=dcap_status==0.and.len_trim(dcap_dir)>0.and.(ktau==1.or.ktau==31.or.ktau==61)
if(dcap_on)then
 write(dcap_file,'(A,"/deng-",I5.5,"-",I4.4,"-",I4.4,".bin")') trim(dcap_dir),ktau,its,jts
 open(newunit=dcap_unit,file=trim(dcap_file),access='stream',form='unformatted', &
      status='replace',convert='little_endian')
 write(dcap_unit) ktau,its,ite,jts,jte,kts,kte
 write(dcap_unit) ttab,qstab,the0k,alu,rdpr,rdthk,plutop
'''+ '\n'.join(before)+'\nendif\n'
s=s.replace('   dxsq = dx*dx',decl+start+'\n   dxsq = dx*dx',1)
s=s.replace('  END SUBROUTINE deng_shcu_driver','if(dcap_on)then\n'+'\n'.join(after)+'\nclose(dcap_unit)\nendif\n  END SUBROUTINE deng_shcu_driver',1)
(a.native/'phys/module_shcu_deng.F').write_text(s)
out=a.base.parent/'native-shcu5';out.mkdir(exist_ok=True)
for name in ('wrfinput_d01','wrfbdy_d01'):
    if not (out/name).exists():(out/name).symlink_to(a.base/name)
for f in (a.native/'run').iterdir():
    if f.is_file() and f.name not in ('namelist.input','wrf.exe','real.exe') and not (out/f.name).exists():(out/f.name).symlink_to(f)
s=(a.base/'namelist.input').read_text().replace('mp_physics=21','mp_physics=6').replace('do_radar_ref=1','do_radar_ref=0')
s=s.replace('run_hours=3','run_hours=1').replace('end_hour=15','end_hour=13')
s=s.replace('isfflx=1,','icloud_bl=0, isfflx=1,')
s=s.replace('cu_physics=0,','shcu_physics=5, cu_physics=0,').replace('bl_pbl_physics=1','bl_pbl_physics=2').replace('sf_sfclay_physics=1','sf_sfclay_physics=2')
(out/'namelist.input').write_text(s);(out/'capture').mkdir(exist_ok=True)
(out/'capture-layout.json').write_text(json.dumps(meta,indent=2)+'\n')
