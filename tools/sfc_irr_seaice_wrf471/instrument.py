"""Capture inputs and outputs at WRF's existing driver boundary; no math edits."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]);dest=Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=True)
f2='psfc sfcprs sfctmp qv1 qgh dz8w1 glw swdown rainbl sr chs cqs2 chs2 rib vegfra shdmin shdmax tmn xland xice snoalb embck tsk hfx qfx lh grdflx qsfc canwat snow snowc snowh albedo albbck emiss znt z0 snotime lai smstav smstot sfcrunoff udrunoff acsnow acsnom snopcx potevp noahres reslin chklowq'.split()
forcing={'psfc':'P8W3D(its:ite,1,jts:jte)','sfcprs':'(P8W3D(its:ite,kts+1,jts:jte)+P8W3D(its:ite,kts,jts:jte))*0.5','sfctmp':'T3D(its:ite,1,jts:jte)','qv1':'QV3D(its:ite,1,jts:jte)','dz8w1':'DZ8W(its:ite,1,jts:jte)','reslin':'0.0*TSK(its:ite,jts:jte)'}
def arr(n): return forcing.get(n,n.upper()+'(its:ite,jts:jte)')
# Names order defines a little-endian REAL32 stream. Every tile gets its own file.
p=root/'phys/module_sf_noahdrv.F';s=p.read_text();s=s.replace('   IRRIGATION_CHANNEL =0.0', '''   INTEGER :: cap_unit
   CHARACTER(len=256) :: cap_file
   IRRIGATION_CHANNEL =0.0''',1)
def dump_noah(arm):
 lines=['  IF (ITIMESTEP <= 3) THEN', f'    WRITE(cap_file,\'(A,I0,A,I0,A,I0,A)\') "captures/noah-", ITIMESTEP, "-", ITS, "-", JTS, "-{arm}.bin"', '    OPEN(newunit=cap_unit,file=trim(cap_file),access="stream",form="unformatted",status="replace",convert="little_endian")', '    WRITE(cap_unit) ite-its+1,jte-jts+1,num_soil_layers,ITIMESTEP,ISURBAN,ISICE, &', '      MERGE(1,0,FRPCPN),MERGE(1,0,USEMONALB),MERGE(1,0,RDLAI2D),OPT_THCND,MERGE(1,0,UA_PHYS)', '    WRITE(cap_unit) DT,XICE_THRESHOLD,DZS(1:num_soil_layers)']
 for n in ['ivgtyp','isltyp',*f2]:lines.append('    WRITE(cap_unit) '+arr(n))
 for n in ('smois','tslb','sh2o','smcrel'):lines.append('    WRITE(cap_unit) '+n.upper()+'(its:ite,1:num_soil_layers,jts:jte)')
 for n in ('flx4_2d','fvb_2d','fbur_2d','fgsn_2d'):lines.append('    WRITE(cap_unit) '+n.upper()+'(its:ite,jts:jte)')
 lines+=['    IF (PRESENT(IRR_CHAN)) THEN','      WRITE(cap_unit) IRR_CHAN(its:ite,jts:jte)','    ELSE','      WRITE(cap_unit) 0.0*TSK(its:ite,jts:jte)','    ENDIF','    CLOSE(cap_unit)','  ENDIF']
 return '\n'.join(lines)+'\n'
s=s.replace('      NSOIL=num_soil_layers',dump_noah('in')+'      NSOIL=num_soil_layers',1)
s=s.replace('   END SUBROUTINE lsm\n',dump_noah('out')+'   END SUBROUTINE lsm\n',1);p.write_text(s)
(dest/'noah-layout.json').write_text(json.dumps({'f2':f2,'f3':['smois','tslb','sh2o','smcrel'],'extras':['flx4','fvb','fbur','fgsn','irrigation_channel']},indent=2)+'\n')
# Sea-ice driver has no model-step number. One process owns a tile; keep three calls.
p=root/'phys/module_sf_noah_seaice_drv.F';s=p.read_text();mark='    FDTLIW = DT / ROWLIW'
s=s.replace(mark,'''    INTEGER, SAVE :: cap_call = 0
    INTEGER :: cap_unit
    CHARACTER(len=256) :: cap_file
    cap_call = cap_call + 1
'''+mark,1)
s2='xice psfc sfcprs sfctmp qv1 qgh glw swdown rainbl sr rib chs cqs2 chs2 tsk emiss albedo z0 snow snowc snowh znt lh hfx qfx potevp grdflx qsfc acsnow acsnom snopcx sfcrunoff noahres albsi icedepth snowsi snoalb'.split()
sea_force={**forcing,'rib':'RIB(its:ite,jts:jte)','snoalb':'SNOALB2D(its:ite,jts:jte)','snowh':'SNOWH2D(its:ite,jts:jte)','z0':'Z02D(its:ite,jts:jte)'}
def dump_sea(arm):
 lines=['  IF (cap_call <= 3) THEN',f'    WRITE(cap_file,\'(A,I0,A,I0,A,I0,A)\') "captures/seaice-", cap_call, "-", ITS, "-", JTS, "-{arm}.bin"','    OPEN(newunit=cap_unit,file=trim(cap_file),access="stream",form="unformatted",status="replace",convert="little_endian")','    WRITE(cap_unit) ite-its+1,jte-jts+1,num_soil_layers,MERGE(1,0,FRPCPN), &','      SEAICE_ALBEDO_OPT,SEAICE_THICKNESS_OPT,SEAICE_SNOWDEPTH_OPT','    WRITE(cap_unit) DT,XICE_THRESHOLD,SEAICE_ALBEDO_DEFAULT,SEAICE_THICKNESS_DEFAULT, &','      SEAICE_SNOWDEPTH_MIN,SEAICE_SNOWDEPTH_MAX']
 for n in s2:lines.append('    WRITE(cap_unit) '+sea_force.get(n,n.upper()+'(its:ite,jts:jte)'))
 lines+=['    WRITE(cap_unit) TSLB(its:ite,1:num_soil_layers,jts:jte)','    CLOSE(cap_unit)','  ENDIF']
 return '\n'.join(lines)+'\n'
s=s.replace('    NSOIL  = NUM_SOIL_LAYERS',dump_sea('in')+'    NSOIL  = NUM_SOIL_LAYERS',1)
# Only the whole driver return, after all output assignments.
import re
matches=list(re.finditer(r'(?im)^\s*END SUBROUTINE(?:\s+seaice_noah)?\s*$',s));assert len(matches)==1,len(matches)
a=matches[0].start();s=s[:a]+'\n'+dump_sea('out')+s[a:];p.write_text(s)
(dest/'seaice-layout.json').write_text(json.dumps({'f2':s2,'f3':['tslb']},indent=2)+'\n')
