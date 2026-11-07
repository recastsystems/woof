"""Keep complete native UDM boundary calls, including optional diagnostics."""
from pathlib import Path
import json
import sys
source,out=map(Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=True)
bounds='ids ide jds jde kds kde ims ime jms jme kms kme its ite jts jte kts kte itimestep'.split()
scalars='delt g cpd cpv ccn0 rd rv t0c ep1 ep2 qmin xls xlv0 xlf0 den0 denr cliq cice psat'.split()
volume='th q qc qr qi qs qg qh nn nc nr den pii p delz re_cloud re_ice re_snow'.split()
surface='xland xice rain rainncv sr'.split()
optional_volume=['refl_10cm'];optional_surface='snow snowncv graupel graupelncv hail hailncv'.split()
opts=['diagflag','do_radar_ref'];optional=optional_volume+optional_surface+opts
written='th q qc qr qi qs qg qh nn nc nr re_cloud re_ice re_snow'.split()
v='its:ite,kts:kte,jts:jte';p='its:ite,jts:jte'
def stream(ext):
 s=f'''  IF (udd_capture) THEN
  WRITE(udd_file,'(A,A,I6.6,A,I6.6,A,I4.4,A)') &
    TRIM(udd_dir),'/udm-driver-i',its,'-j',jts,'-s',itimestep,'{ext}'
  OPEN(NEWUNIT=udd_unit,FILE=TRIM(udd_file),ACCESS='STREAM', &
    FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
'''
 if ext=='.input':
  for names in (bounds[:12],bounds[12:],scalars[:10],scalars[10:],['has_reqc','has_reqi','has_reqs']):
   s+='  WRITE(udd_unit) '+','.join(names)+'\n'
  for n in optional:s+=f'  WRITE(udd_unit) MERGE(1,0,PRESENT({n}))\n'
  for n in opts:s+=f'  IF (PRESENT({n})) WRITE(udd_unit) {n}\n'
 for names,dims in ((volume if ext=='.input' else written,v),(surface if ext=='.input' else ['rain','rainncv','sr'],p)):
  for n in names:s+=f'  WRITE(udd_unit) {n}({dims})\n'
 for names,dims in ((optional_volume,v),(optional_surface,p)):
  for n in names:s+=f'  IF (PRESENT({n})) WRITE(udd_unit) {n}({dims})\n'
 return s+'  CLOSE(udd_unit)\n  END IF\n'
s=source.read_text();start=s.index('subroutine udm(');a=s.index('      if (itimestep==1) then',start)
s=s[:a]+'''  INTEGER :: udd_unit
  LOGICAL :: udd_capture
  CHARACTER(LEN=512) :: udd_dir,udd_file
  CALL GET_ENVIRONMENT_VARIABLE('MISC_UDM_DRIVER_CAPTURE_DIR',udd_dir)
  udd_capture=LEN_TRIM(udd_dir)>0.AND. &
    (itimestep==1.OR.itimestep==2.OR.itimestep==30.OR.itimestep==100)
'''+stream('.input')+s[a:]
a=s.index('end subroutine udm\n',start);s=s[:a]+stream('.output')+s[a:]
(out/source.name).write_text(s)
(out/'driver-schema.json').write_text(json.dumps(dict(bounds_i4=bounds,scalars_f4=scalars,flags_i4=['has_reqc','has_reqi','has_reqs'],volume=volume,surface=surface,optional_volume=optional_volume,optional_surface=optional_surface,present_i4=optional,optional_scalars=opts,written_volume=written,written_surface=['rain','rainncv','sr']),indent=2)+'\n')
