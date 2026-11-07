from pathlib import Path
import argparse,hashlib,json
ap=argparse.ArgumentParser(description="Capture unmodified real-run MSKF inputs and outputs")
ap.add_argument('source',type=Path,help='a private copy of the pinned module_cu_mskf.F')
ap.add_argument('--receipt',required=True,type=Path)
ap.add_argument('--nz',default=49,type=int)
ap.add_argument('--steps',default='1,241,481')
args=ap.parse_args()
steps=[int(v) for v in args.steps.split(',')]
if not steps or any(v<1 for v in steps) or args.nz<8:ap.error('positive steps and nz >= 8 required')
path=args.source;s=path.read_text();pin='4bca54b294def08ffc98ef69a30650c943e2c394f34802e75e50c7f1a2896fcb'
assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
begin=s.index('   SUBROUTINE MSKF_CPS(');end=s.index('   END SUBROUTINE MSKF_CPS',begin)
part=s[begin:end]
fields=['RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RUCUTEN','RVCUTEN','cldfra_dp_KF','cldfra_sh_KF','qc_KF','qi_KF','qr_KF','qs_KF','nc_KF','ni_KF','nr_KF','ns_KF','ccn_KF','EFCS','EFIS','EFSS','UDR_KF','DDR_KF','UER_KF','DER_KF']
surface=['RAINCV','PRATEC','NCA','CUTOP','CUBOT','ainc_frac','TIMEC_KF']
decl='''! Lane-only real-run capture. No scientific assignments are changed.
   INTEGER :: lane_unit, lane_ii, lane_jj, lane_i, lane_j
   CHARACTER(LEN=128) :: lane_file
   LOGICAL :: lane_capture
'''
anchor='!JTR: CMT on by default';assert anchor in part
entry='''   lane_capture = (KTAU == 1 .OR. KTAU == 241 .OR. KTAU == 481) .AND. &
                  kte-kts+1 == 49 .AND. aercu_opt == 0
   IF (lane_capture) THEN
      WRITE(lane_file,'(A,I6.6,A,I4.4,A,I4.4,A)') 'mskf-',KTAU,'-',its,'-',jts,'.bin'
      OPEN(newunit=lane_unit,file=TRIM(lane_file),access='stream', &
           form='unformatted',status='new',convert='little_endian')
      WRITE(lane_unit) kte-kts+1,16,trigger,KTAU,STEPCU, &
          MERGE(1,0,warm_rain),MERGE(1,0,flag_qr),MERGE(1,0,flag_qi), &
          MERGE(1,0,flag_qs),aercu_opt,KF_EDRATES,MERGE(1,0,ADAPT_STEP_FLAG)
      WRITE(lane_unit) DT,DX,CUDT,aercu_fct,XLV0,XLV1,XLS0,XLS1,CP,R,G,EP1,EP2, &
          SVP1,SVP2,SVP3,SVPT0
      DO lane_jj=1,4
        DO lane_ii=1,4
          lane_i=its+(ite-its)*lane_ii/5
          lane_j=jts+(jte-jts)*lane_jj/5
          WRITE(lane_unit) lane_i,lane_j
'''
for field in ['U','V','T','QV','Pcps','rho','pi','dz8w','W0AVG']:
 entry+=f'          WRITE(lane_unit) {field}(lane_i,kts:kte,lane_j)\n'
entry+='          WRITE(lane_unit) W(lane_i,kts:kte+1,lane_j)\n'
entry+='          WRITE(lane_unit) HFX(lane_i,lane_j),UST(lane_i,lane_j), &\n              PBLH(lane_i,lane_j),ZOL(lane_i,lane_j)\n'
def outputs():
 result=''
 for field in fields:
  # aercu_opt=0 has no allocated aerosol-package arrays. Never index the
  # native one-element dummy. Zero slots preserve the stream's fixed layout.
  if field in ('qr_KF','qs_KF','nc_KF','ni_KF','nr_KF','ns_KF','ccn_KF','EFCS','EFIS','EFSS'):
   result+='          WRITE(lane_unit) SPREAD(0.0,1,kte-kts+1)\n'
  elif field in ('UDR_KF','DDR_KF','UER_KF','DER_KF'):
   result+=f'          IF (KF_EDRATES == 1) THEN\n            WRITE(lane_unit) {field}(lane_i,kts:kte,lane_j)\n          ELSE\n            WRITE(lane_unit) SPREAD(0.0,1,kte-kts+1)\n          ENDIF\n'
  else:result+=f'          WRITE(lane_unit) {field}(lane_i,kts:kte,lane_j)\n'
 for field in surface:
  if field=='ainc_frac':result+='          WRITE(lane_unit) 0.0\n'
  elif field=='TIMEC_KF':result+=f'          IF (KF_EDRATES == 1) THEN\n            WRITE(lane_unit) {field}(lane_i,lane_j)\n          ELSE\n            WRITE(lane_unit) 0.0\n          ENDIF\n'
  else:result+=f'          WRITE(lane_unit) {field}(lane_i,lane_j)\n'
 result+='          WRITE(lane_unit) REAL(MERGE(1,0,CU_ACT_FLAG(lane_i,lane_j)))\n'
 return result
entry+=outputs()+'        ENDDO\n      ENDDO\n   ENDIF\n'
entry=entry.replace('(KTAU == 1 .OR. KTAU == 241 .OR. KTAU == 481)', '('+' .OR. '.join(f'KTAU == {v}' for v in steps)+')').replace('kte-kts+1 == 49',f'kte-kts+1 == {args.nz}')
part=part.replace(anchor,decl+'\n'+anchor)
anchor='   if (lastdt < 0) then';assert anchor in part;part=part.replace(anchor,entry+'\n'+anchor,1)
exitcode='''   IF (lane_capture) THEN
      DO lane_jj=1,4
        DO lane_ii=1,4
          lane_i=its+(ite-its)*lane_ii/5
          lane_j=jts+(jte-jts)*lane_jj/5
'''+outputs()+'''          WRITE(lane_unit) W0AVG(lane_i,kts:kte,lane_j)
        ENDDO
      ENDDO
      CLOSE(lane_unit)
   ENDIF
'''
# The driver has one RETURN immediately before END SUBROUTINE.
idx=part.rfind('   RETURN')
if idx < 0: idx=len(part)
part=part[:idx]+exitcode+'\n'+part[idx:]
path.write_text(s[:begin]+part+s[end:])
args.receipt.write_text(json.dumps({'native_source_sha256':pin,'instrumented_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'capture_steps':steps,'nz':args.nz,'columns_per_tile':16,'field_order':fields,'surface_order':surface},indent=2)+'\n')
