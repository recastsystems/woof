"""Capture real irrigation calls without changing native arithmetic."""
from pathlib import Path
import re,sys
p=Path(sys.argv[1])/'phys/module_irrigation.F';s=p.read_text()
for name,option in [('channel_irrigation',1),('drip_irrigation',2)]:
 m=re.search(r'(?ims)^\s*SUBROUTINE '+name+r'\b.*?^END SUBROUTINE '+name,s);assert m
 body=m.group(0);mark='  IRRIGATION_CHANNEL=0.'
 code='''  INTEGER,SAVE :: cap_count=0,cap_unit
  CHARACTER(len=160) :: cap_file
  LOGICAL :: cap_active
  cap_active=IRRIGATION.GE.0.1 .AND. cap_count.LT.128
  IF(cap_active) THEN
   IF(cap_count.EQ.0) THEN
    WRITE(cap_file,'(A,I0,A)') "captures/irrigation-OPTION-",getpid(),".bin"
    OPEN(newunit=cap_unit,file=trim(cap_file),access="stream",form="unformatted",status="replace",convert="little_endian")
   ENDIF
   cap_count=cap_count+1
   WRITE(cap_unit) OPTION,i,j,irr_start_hour,irr_num_hours,irr_start_julianday,irr_end_julianday,irr_freq,irr_ph,irr_rand_field_val
   WRITE(cap_unit) julian_in,IRRIGATION,irr_daily_amount,gmt,xtime,dt,RAINBL,IRRIGATION_CHANNEL
  ENDIF
'''.replace('OPTION',str(option))
 body=body.replace(mark,code+mark,1)
 body=body.replace('RETURN\n','  IF(cap_active) THEN\n   WRITE(cap_unit) RAINBL,IRRIGATION_CHANNEL\n   WRITE(cap_unit) irr_rand_field_val\n   FLUSH(cap_unit)\n  ENDIF\nRETURN\n',1)
 s=s[:m.start()]+body+s[m.end():]
# The sprinkler is already tiled; keep its three first real calls.
m=re.search(r'(?ims)^SUBROUTINE sprinkler_irrigation\b.*?^END SUBROUTINE sprinkler_irrigation',s);assert m
b=m.group(0);mark='   end_hour=irr_start_hour+irr_num_hours';code='''   INTEGER,SAVE :: cap_count=0
   INTEGER :: cap_unit
   CHARACTER(len=160) :: cap_file
   cap_count=cap_count+1
   IF(cap_count.LE.3) THEN
    WRITE(cap_file,'(A,I0,A,I0,A)') "captures/sprinkler-",getpid(),"-",cap_count,"-in.bin"
    OPEN(newunit=cap_unit,file=trim(cap_file),access="stream",form="unformatted",status="replace",convert="little_endian")
    WRITE(cap_unit) ite-its+1,jte-jts+1,its,jts,irr_start_hour,irr_num_hours,irr_start_julianday,irr_end_julianday,irr_freq,irr_ph
    WRITE(cap_unit) julian_in,irr_daily_amount,gmt,xtime,dt
    WRITE(cap_unit) irrigation(its:ite,jts:jte),irr_rand_field_val(its:ite,jts:jte)
    WRITE(cap_unit) qr_curr(its:ite,kms,jts:jte),rho(its:ite,kms,jts:jte),dz8w(its:ite,kms,jts:jte)
    CLOSE(cap_unit)
   ENDIF
'''
b=b.replace(mark,code+mark,1);b=b.replace('RETURN\n','''   IF(cap_count.LE.3) THEN
    WRITE(cap_file,'(A,I0,A,I0,A)') "captures/sprinkler-",getpid(),"-",cap_count,"-out.bin"
    OPEN(newunit=cap_unit,file=trim(cap_file),access="stream",form="unformatted",status="replace",convert="little_endian")
    WRITE(cap_unit) qr_curr(its:ite,kms,jts:jte),irr_rand_field_val(its:ite,jts:jte)
    CLOSE(cap_unit)
   ENDIF
RETURN
''',1)
s=s[:m.start()]+b+s[m.end():];p.write_text(s)
