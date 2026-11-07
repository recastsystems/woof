"""Capture real shallow-driver arguments without changing scheme arithmetic."""
from pathlib import Path
import json
import re
import argparse

a = argparse.ArgumentParser(description=__doc__)
a.add_argument("native_root", type=Path)
a.add_argument("--base-input", type=Path, required=True)
args = a.parse_args()

root = args.native_root.resolve()
p = root / 'phys/module_shallowcu_driver.F'
s = p.read_text()
original = s
s = s.replace('   CHARACTER(len=200) :: message', '''   CHARACTER(len=200) :: message
   INTEGER :: shcap_unit, shcap_status
   CHARACTER(len=512) :: shcap_dir, shcap_file
   LOGICAL :: shcap_on''', 1)
before = '''
   CALL get_environment_variable('WOOF_SHCU_CAPTURE',shcap_dir,status=shcap_status)
   shcap_on = shcap_status == 0 .and. len_trim(shcap_dir) > 0 .and. &
       (itimestep == 1 .or. itimestep == 31 .or. itimestep == 61)
   IF (shcap_on) THEN
      WRITE(shcap_file,'(A,"/shcu",I1,"-",I5.5,"-",I4.4,"-",I4.4,".bin")') &
          trim(shcap_dir),shcu_physics,itimestep,its,jts
      OPEN(newunit=shcap_unit,file=trim(shcap_file),access='stream', &
           form='unformatted',status='replace',convert='little_endian')
      WRITE(shcap_unit) shcu_physics,itimestep,its,ite,jts,jte,kts,kte
      WRITE(shcap_unit) dt,dx,g,xlv,r_d,r_v,rcp,p1000mb,cp,cliq,cpv,ep_1,ep_2,cice,xls,psat
      WRITE(shcap_unit) mp_physics,bl_pbl_physics
      WRITE(shcap_unit) znu(kts:kte)
      WRITE(shcap_unit) t(its:ite,kts:kte,jts:jte),qv_curr(its:ite,kts:kte,jts:jte), &
           qc_curr(its:ite,kts:kte,jts:jte),qi_curr(its:ite,kts:kte,jts:jte), &
           p(its:ite,kts:kte,jts:jte),pi(its:ite,kts:kte,jts:jte), &
           rho(its:ite,kts:kte,jts:jte),u(its:ite,kts:kte,jts:jte), &
           v(its:ite,kts:kte,jts:jte),dz8w(its:ite,kts:kte,jts:jte)
      WRITE(shcap_unit) p8w(its:ite,kts:kte+1,jts:jte), &
           z_at_w(its:ite,kts:kte+1,jts:jte),w(its:ite,kts:kte+1,jts:jte)
      WRITE(shcap_unit) raincv(its:ite,jts:jte),xland(its:ite,jts:jte), &
           pblh(its:ite,jts:jte),hfx(its:ite,jts:jte),qfx(its:ite,jts:jte)
      IF (shcu_physics == 3) THEN
         WRITE(shcap_unit) wstar(its:ite,jts:jte),delta(its:ite,jts:jte), &
              kpbl(its:ite,jts:jte)
      ENDIF
   ENDIF
'''
s = s.replace('   scps_select: SELECT CASE(shcu_physics)', before + '\n   scps_select: SELECT CASE(shcu_physics)', 1)
after = '''
   IF (shcap_on) THEN
      WRITE(shcap_unit) rthshten(its:ite,kts:kte,jts:jte), &
           rqvshten(its:ite,kts:kte,jts:jte),rqcshten(its:ite,kts:kte,jts:jte), &
           rqishten(its:ite,kts:kte,jts:jte),rushten(its:ite,kts:kte,jts:jte), &
           rvshten(its:ite,kts:kte,jts:jte),rqrshten(its:ite,kts:kte,jts:jte), &
           rqsshten(its:ite,kts:kte,jts:jte),rqgshten(its:ite,kts:kte,jts:jte)
      WRITE(shcap_unit) tmppratesh(its:ite,jts:jte),hbot(its:ite,jts:jte),htop(its:ite,jts:jte)
      CLOSE(shcap_unit)
   ENDIF
'''
s = s.replace('   END SELECT scps_select', '   END SELECT scps_select\n' + after, 1)
assert s != original and s.count('WOOF_SHCU_CAPTURE') == 1
p.write_text(s)
# Keep every build path owned by this lane. The copied build's absolute
# paths otherwise resolve to the shared source and libraries.
p = root / 'configure.wrf'
s = p.read_text().replace('/workspace/native-wrf471', str(root))
p.write_text(s)

base = args.base_input.resolve()
template = (base/'namelist.input').read_text()
for row in (0,1,3,4):
    d = base.parent/f'native-shcu{row}'
    d.mkdir(exist_ok=True)
    for name in ('wrfinput_d01','wrfbdy_d01'):
        (d/name).symlink_to(base/name)
    for f in (root/'run').iterdir():
        if f.is_file() and f.name not in ('namelist.input','wrf.exe','real.exe'):
            (d/f.name).symlink_to(f)
    text = re.sub(r'mp_physics=21', 'mp_physics=6', template)
    text = text.replace('cu_physics=0,',f'shcu_physics={row}, cu_physics=0,')
    text = text.replace('do_radar_ref=1','do_radar_ref=0')
    (d/'namelist.input').write_text(text)
    (d/'capture').mkdir(exist_ok=True)
