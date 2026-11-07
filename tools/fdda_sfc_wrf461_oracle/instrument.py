#!/usr/bin/env python3
"""Instrument ONLY an owned copy of the pinned native source for one frame.

Run native WRF with one MPI rank and one tile. This patch writes the second
SFDDAGD call's full domain and then closes its stream, with no numerical edit.
The source is a referee tool, never code for distribution in a forecast.
"""
import argparse
import hashlib
from pathlib import Path


PIN = "1d8aaf15a102ffe30ef25c96663c237cdfc16905b9c002064e9b8b387bc10648"
PIN461 = "021773e75185ffac50242ed68a6426303a8dec66b2429af549e042a68175a335"


def instrument(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() not in (PIN, PIN461):
        raise ValueError("native SFDDAGD source differs from the reviewed v4.6.1/v4.7.1 pins")
    text = data.decode()
    split = text.index("   SUBROUTINE sfddagd(")
    head,body = text[:split],text[split:]
    first = body.index("   iwinds =")
    body = body[:first] + """   INTEGER, SAVE :: sfc_capture_count=0
   INTEGER :: sfc_capture_unit
   LOGICAL :: sfc_capture_this
   REAL, ALLOCATABLE :: sfc_capture_wpbl(:,:,:,:)
""" + body[first:]
    anchor = "   DO j=jts,jtf\n   DO k=kts,ktf\n   DO i=itsu,itf\n     IF( idd == 1 ) THEN"
    if body.count(anchor) != 1:
        raise ValueError("native tendency-loop anchor changed")
    before = """   sfc_capture_count=sfc_capture_count+1
   sfc_capture_this=(sfc_capture_count == 2)
   IF(sfc_capture_this) THEN
     IF(its/=ids .OR. jts/=jds .OR. ite<ide-1 .OR. jte<jde-1) &
       CALL wrf_error_fatal('surface capture requires one rank and one whole-domain tile')
     OPEN(newunit=sfc_capture_unit,file='surface-native-frame.bin', &
          access='stream',form='unformatted',status='new',convert='little_endian')
     WRITE(sfc_capture_unit) 'SFDD471A'
     WRITE(sfc_capture_unit) ide-ids+1,jde-jds+1,kde-kds,fasdas
     WRITE(sfc_capture_unit) dx,rinblw,coef,tfac_sfc,guv_sfc,gt_sfc,gq_sfc
     WRITE(sfc_capture_unit) u3d(ids:ide,kds:kde-1,jds:jde),v3d(ids:ide,kds:kde-1,jds:jde), &
       th3d(ids:ide,kds:kde-1,jds:jde),t3d(ids:ide,kds:kde-1,jds:jde), &
       qv3d(ids:ide,kds:kde-1,jds:jde),p3d(ids:ide,kds:kde-1,jds:jde),z(ids:ide,kds:kde-1,jds:jde)
     WRITE(sfc_capture_unit) ht(ids:ide,jds:jde),regime(ids:ide,jds:jde),znt(ids:ide,jds:jde)
     WRITE(sfc_capture_unit) u10_ndg_old(ids:ide,jds:jde),v10_ndg_old(ids:ide,jds:jde), &
       th2_ndg_old(ids:ide,jds:jde),q2_ndg_old(ids:ide,jds:jde)
     WRITE(sfc_capture_unit) u10_ndg_new(ids:ide,jds:jde),v10_ndg_new(ids:ide,jds:jde), &
       th2_ndg_new(ids:ide,jds:jde),q2_ndg_new(ids:ide,jds:jde)
     WRITE(sfc_capture_unit) odis_ndg_old(ids:ide,jds:jde),odis_ndg_new(ids:ide,jds:jde)
     ALLOCATE(sfc_capture_wpbl(ids:ide,kds:kde-1,jds:jde,4))
     sfc_capture_wpbl=0.
     sfc_capture_wpbl(its:ite,kds:kde-1,jts:jte,:)=wpbl(:,kds:kde-1,:,:)
     WRITE(sfc_capture_unit) sfc_capture_wpbl,wzfac(kds:kde-1,:)
     DEALLOCATE(sfc_capture_wpbl)
     WRITE(sfc_capture_unit) RUNDGDTEN(ids:ide,kds:kde-1,jds:jde),RVNDGDTEN(ids:ide,kds:kde-1,jds:jde), &
       RTHNDGDTEN(ids:ide,kds:kde-1,jds:jde),RQVNDGDTEN(ids:ide,kds:kde-1,jds:jde)
   ENDIF
"""
    body = body.replace(anchor,before+anchor)
    end = body.index("   !BPR BEGIN \n   !Diagnostic print")
    after = """   IF(sfc_capture_this) THEN
     WRITE(sfc_capture_unit) RUNDGDTEN(ids:ide,kds:kde-1,jds:jde),RVNDGDTEN(ids:ide,kds:kde-1,jds:jde), &
       RTHNDGDTEN(ids:ide,kds:kde-1,jds:jde),RQVNDGDTEN(ids:ide,kds:kde-1,jds:jde)
     IF(fasdas==1) WRITE(sfc_capture_unit) SDA_HFX(ids:ide,jds:jde),SDA_QFX(ids:ide,jds:jde)
     CLOSE(sfc_capture_unit)
   ENDIF
"""
    body = body[:end]+after+body[end:]
    path.write_text(head+body)
    print(f"instrumented {path}: {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owned_native_source", type=Path)
    instrument(parser.parse_args().owned_native_source)
