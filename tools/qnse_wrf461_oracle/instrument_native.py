"""Instrument only the lane's native clone, without changing physical expressions."""
from pathlib import Path
import json, sys
root=Path(sys.argv[1]); manifest={}
specs={
 'sfc':dict(file='module_sf_qnsesfc.F',name='QNSESFC',marker='      DO J=JTS,JTE',scalar=['ITIMESTEP'],s2='HT MAVAIL TSK XLAND Z0BASE QSFC THZ0 QZ0 UZ0 VZ0 USTAR ZNT PBLH RMOL AKHS AKMS RIB CHS CHS2 CQS2 HFX QFX FLX_LH FLHC FLQC QGH CPM CT U10 V10 T02 TH02 TSHLTR TH10 Q02 QSHLTR Q10 PSHLTR U10E V10E'.split(),s3='DZ PMID TH T QV QC U V Q2'.split()),
 'pbl':dict(file='module_bl_qnsepbl.F',name='QNSEPBL',marker='      IMD=(IMS+IME)/2',scalar=['DT','STEPBL'],s2='HT SICE SNOW TSK XLAND HFX AKHS AKMS CT QSFC QZ0 THZ0 USTAR UZ0 VZ0 ZNT CHKLOWQ ELFLX CORF PBLH KPBL'.split(),s3='CWM DZ EXNER PMID QV RHO T TH U V EL_MYJ RQCBLTEN RQVBLTEN RTHBLTEN RUBLTEN RVBLTEN EXCH_H EXCH_M TKE WTHV_MF LM_BL89'.split()),
 'mf':dict(full_tile=True,file='module_bl_mfshconvpbl.F',name='MFSHCONVPBL',marker='DO J=JTS,JTE',scalar=['DT','STEPBL','KRR'],s2='HT HFX QFX'.split(),s3='DZ EXNER PMID QV QC RHO TH U V TKE RQCBLTEN RQVBLTEN RTHBLTEN RUBLTEN RVBLTEN MASSFLUX_EDKF ENTR_EDKF DETR_EDKF THL_UP THV_UP RT_UP RV_UP RC_UP U_UP V_UP FRAC_UP RC_MF WTHV PLM_BL89'.split())
}
for key,s in specs.items():
 if len(sys.argv)>2 and key!=sys.argv[2]:continue
 p=root/'phys'/s['file']; text=p.read_text(); start=text.index('      SUBROUTINE '+s['name']); end=text.upper().index('END SUBROUTINE '+s['name'],start)
 block=text[start:end]; at=block.index(s['marker'])
 decl='      LOGICAL, SAVE :: lane_recorded=.FALSE.\n      LOGICAL :: lane_capture\n      INTEGER :: lane_unit,lane_last\n'
 # Insert declarations before executable marker and its preceding comments.
 block=block[:at]+decl+'''      lane_capture=(ITS==1.AND.JTS==1.AND..NOT.lane_recorded)
      IF (lane_capture) THEN
        lane_last=MIN(ITE,ITS+31)
'''+f"        OPEN(NEWUNIT=lane_unit,FILE='capture-{key}.bin',ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')\n"+('        WRITE(lane_unit) ITE-ITS+1,KTE-KTS+1,JTE-JTS+1\n' if s.get('full_tile') else '        WRITE(lane_unit) lane_last-ITS+1,KTE-KTS+1\n')+''.join('        WRITE(lane_unit) '+v+'\n' for v in s['scalar'])+ '      ENDIF\n'+block[at:]
 def dump():
  if s.get('full_tile'):
   return '      IF (lane_capture) THEN\n'+''.join(f'        WRITE(lane_unit) {v}(ITS:ITE,JTS:JTE)\n' for v in s['s2'])+''.join(f'        WRITE(lane_unit) {v}(ITS:ITE,KTS:KTE,JTS:JTE)\n' for v in s['s3'])+'        WRITE(lane_unit) PINT(ITS:ITE,KTS:KTE+1,JTS:JTE)\n      ENDIF\n'
  return '      IF (lane_capture) THEN\n'+''.join(f'        WRITE(lane_unit) {v}(ITS:lane_last,JTS)\n' for v in s['s2'])+''.join(f'        WRITE(lane_unit) {v}(ITS:lane_last,KTS:KTE,JTS)\n' for v in s['s3'])+'        WRITE(lane_unit) PINT(ITS:lane_last,KTS:KTE+1,JTS)\n      ENDIF\n'
 # Place first dump immediately before original first executable statement.
 block=block.replace(s['marker'],dump()+s['marker'],1)
 block+=dump()+'      IF (lane_capture) THEN\n        CLOSE(lane_unit)\n        lane_recorded=.TRUE.\n      ENDIF\n      '
 p.write_text(text[:start]+block+text[end:]);manifest[key]=s
out=root.parent/'evidence'/'capture-manifest.json'
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():
 old=json.loads(out.read_text());old.update(manifest);manifest=old
out.write_text(json.dumps(manifest,indent=2)+'\n')
