"""Build the independent public-domain Fortran routine as a binary oracle.
No forecast code calls this program. Input is ncol,nz, then for each column
11 float32 profiles (T,z,p,rho,qv,qi,qc,qr,qs,qg,w), ht and wdur.
--form picks the generation (translate.FORMS): wrf_461 (default) or
wrf_39_noaa, the operational HRRR v4.1.21 branch's copy.
"""
from pathlib import Path
import argparse
import importlib.util
import sys
p=argparse.ArgumentParser();p.add_argument("output",type=Path)
p.add_argument("--form",default="wrf_461");a=p.parse_args()
spec=importlib.util.spec_from_file_location("hail_translate",Path(__file__).parent/"translate.py")
translate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=translate;spec.loader.exec_module(translate)
form=translate.FORMS[a.form];first,last=form["lines"]
src=form["source"].read_text().splitlines()
body="\n".join(src[first-1:last])
body=body.replace("      TFZL = tk_embryo","      sec=0.\n      D=0.\n      TFZL = tk_embryo")
body=body.replace("      CALL INTERPP(PA, WFZLP, TCA, tk_embryo, IFOUT, nz)",
"""      CALL INTERPP(PA, WFZLP, TCA, tk_embryo, IFOUT, nz)
      IF (IFOUT == 1) THEN
          dhails(i)=0.
          CYCLE
      ENDIF""")
# Same explicit physical-domain correction as the CUDA port, expressed in
# original Fortran independently of the translator. Preserve the stock NaN
# reproduction in the whole-run receipt; do not conceal it by a NaN fallback.
body=body.replace("             TSCELSIUS = TS - 273.16",
                  "             VIMP = MAX(VIMP,0.)\n             TSCELSIUS = TS - 273.16")
driver="""
end module
program oracle
use hail_oracle
implicit none
integer :: nc,nz,c,i
real, allocatable :: x(:,:)
real :: ht,dur,d(5)
character(1024) :: input,output
call get_command_argument(1,input)
call get_command_argument(2,output)
open(11,file=trim(input),access='stream',form='unformatted',status='old')
open(12,file=trim(output),access='stream',form='unformatted',status='replace')
read(11) nc,nz
allocate(x(nz,11))
do c=1,nc
 read(11) x,ht,dur
 d=0.
 call hailstone_driver(x(:,1),x(:,2),ht,x(:,3),x(:,4),x(:,5), &
 x(:,6),x(:,7),x(:,8),x(:,9),x(:,10),x(:,11),dur,nz, &
 d(1),d(2),d(3),d(4),d(5))
 write(12) d
enddo
close(11)
close(12)
end program
"""
assert body.count("dhails(i)=0.")==1 and body.count("VIMP = MAX(VIMP,0.)")==1 and "D=0." in body
a.output.write_text("module hail_oracle\nimplicit none\ncontains\n"+body+"\n"+driver)
