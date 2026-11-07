"""Instrument pinned WRF GFS modules at their real driver call boundaries.

Usage: python instrument_native.py WRF_SOURCE_ROOT OUTPUT_DIRECTORY
Only the first two calls per MPI tile write a 8 by 4 column rectangle.
Build these copies with strict scalar flags and link into an own WRF copy.
The resulting little-endian streams are consumed by replay_native.py.
"""
from pathlib import Path
import sys, json, hashlib
src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
pins = {'bl': 'a3898642ee386d60a74af2ef45867e6b5f7df518eef9543ae91586d00372b71c', 'sf': '10eb3f3b47deb8f33d10e13a5b274db89a59cf5fb8c86a92f353205b9e73440e'}
receipt = []
for scheme, sub in [('bl', 'BL_GFS'), ('sf', 'SF_GFS')]:
    p = src / f'phys/module_{scheme}_gfs.F'
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != pins[scheme]:
        raise ValueError(f'{p}: {actual}, expected WRF v4.6.1 {pins[scheme]}')
    s = p.read_text()
    marker = '   IM=ITE-ITS+1' if scheme == 'bl' else '   if(itimestep.eq.0) then'
    decl = '      INTEGER, SAVE :: cap_calls=0\n      INTEGER :: cap_unit,cap_ix,cap_jy,cap_flqi\n      CHARACTER(LEN=160) :: cap_name\n      LOGICAL :: cap_active\n'
    start = '   cap_calls=cap_calls+1\n   cap_active=cap_calls.le.2\n   IF(cap_active) THEN\n      cap_ix=MIN(ite,its+7)\n      cap_jy=MIN(jte,jts+3)\n'
    start += f"      WRITE(cap_name,'(A,I5.5,A,I5.5,A,I3.3,A)') 'capture-{scheme}-',its,'-',jts,'-',cap_calls,'.bin'\n"
    start += "      OPEN(NEWUNIT=cap_unit,FILE=TRIM(cap_name),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')\n"
    if scheme == 'bl':
        start += '      cap_flqi=0\n      IF(P_QI.ge.P_FIRST_SCALAR) cap_flqi=1\n'
        start += '      WRITE(cap_unit) cap_ix-its+1,kte-kts+1,cap_jy-jts+1,cap_flqi,DT\n'
        vols = ['U3D', 'V3D', 'TH3D', 'T3D', 'QV3D', 'QC3D', 'QI3D', 'P3D', 'PI3D', 'dz8w', 'z']
        surfs = ['PSFC', 'UST', 'PBL', 'PSIM', 'PSIH', 'HFX', 'QFX', 'TSK', 'GZ1OZ0', 'WSPD', 'BR']
        for n in vols:
            start += f'      WRITE(cap_unit) {n}(its:cap_ix,kts:kte,jts:cap_jy)\n'
        for n in surfs:
            start += f'      WRITE(cap_unit) {n}(its:cap_ix,jts:cap_jy)\n'
        outs = ['RUBLTEN', 'RVBLTEN', 'RTHBLTEN', 'RQVBLTEN', 'RQCBLTEN', 'RQIBLTEN']
        end = '   IF(cap_active) THEN\n'
        for n in outs:
            end += f'      WRITE(cap_unit) {n}(its:cap_ix,kts:kte,jts:cap_jy)\n'
        for n in ['UST', 'WSPD', 'PBL', 'KPBL2D']:
            end += f'      WRITE(cap_unit) {n}(its:cap_ix,jts:cap_jy)\n'
    else:
        start += '      WRITE(cap_unit) cap_ix-its+1,cap_jy-jts+1,ISFFLX,itimestep\n'
        start += '      WRITE(cap_unit) CP,ROVCP,R,XLV,EP1,EP2,KARMAN\n'
        for n in ['U3D', 'V3D', 'T3D', 'QV3D', 'P3D']:
            start += f'      WRITE(cap_unit) {n}(its:cap_ix,kts,jts:cap_jy)\n'
        for n in ['PSFC', 'TSK', 'XLAND', 'ZNT', 'UST']:
            start += f'      WRITE(cap_unit) {n}(its:cap_ix,jts:cap_jy)\n'
        end = '   IF(cap_active) THEN\n'
        for n in ['BR', 'CHS', 'CHS2', 'CPM', 'CQS2', 'FLHC', 'FLQC', 'GZ1OZ0', 'HFX', 'LH', 'PSIM', 'PSIH', 'QFX', 'QGH', 'QSFC', 'U10', 'V10', 'WSPD', 'ZNT', 'UST']:
            end += f'      WRITE(cap_unit) {n}(its:cap_ix,jts:cap_jy)\n'
    start += '   ENDIF\n'
    end += '      CLOSE(cap_unit)\n   ENDIF\n\n'
    assert s.count(marker) == 1
    s = s.replace(marker, decl + start + '\n' + marker, 1)
    s = s.replace('   END SUBROUTINE ' + sub, end + '   END SUBROUTINE ' + sub, 1)
    q = out / p.name
    q.write_text(s)
    receipt.append(dict(file=str(q), source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), instrumented_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
(out / 'capture-source.json').write_text(json.dumps(receipt, indent=2) + '\n')
