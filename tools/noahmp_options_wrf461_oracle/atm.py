"""Check selected ATM branches against the complete, pinned native procedure.

This constructed boundary referee supplements real captured module replay.
It is not a replacement for that replay or a real forecast.
"""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import subprocess

import numpy as np

from .build import FLAGS, LSM_SHA256, NOAHMP_PIN


def build(lsm, work):
    raw = lsm.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LSM_SHA256:
        raise ValueError("refusing unpinned module_sf_noahmplsm.F")
    text = raw.decode()
    first = text.index("  SUBROUTINE ATM (")
    last = text.index("  END SUBROUTINE ATM", first) + len("  END SUBROUTINE ATM")
    block = text[first:last]
    constants = '\n'.join(line for line in text.splitlines()
        if any(':: ' + key + ' ' in line for key in ('RAIR','CPAIR','TFRZ','HVAP','HSUB')))
    assert len(constants.splitlines()) == 5
    source = '''module native_atm
use iso_c_binding
implicit none
type noahmp_parameters
 integer :: unused
end type
integer :: opt_snf
CONSTANTS
contains
subroutine selected_atm(option,x,y) bind(C)
integer(c_int), value :: option
real(c_float), intent(in) :: x(11)
real(c_float), intent(out) :: y(17)
type(noahmp_parameters) :: p
opt_snf=option
call atm(p,x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10),x(11), &
 y(1),y(2),y(3),y(4),y(5),y(6),y(7:8),y(9:10),y(11),y(12), &
 y(13),y(14),y(15),y(16),y(17))
end subroutine
BLOCK
end module
'''.replace('CONSTANTS',constants).replace('BLOCK',block)
    work.mkdir(parents=True,exist_ok=True)
    path=work/'atm.F90';path.write_text(source)
    library=work/'libatm.so'
    command=['gfortran',*FLAGS,'-J',str(work.resolve()),str(path.resolve()),'-o',str(library.resolve())]
    subprocess.run(command,cwd=work,check=True)
    return library,dict(noahmp_pin=NOAHMP_PIN,lsm_sha256=LSM_SHA256,
        procedure_sha256=hashlib.sha256(block.encode()).hexdigest(),
        source_lines=[text[:first].count('\n')+1,text[:last].count('\n')+1],
        licence='WRF v4 public domain',command=command)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lsm',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    library,provenance=build(args.lsm,args.work)
    fn=ctypes.CDLL(str(library.resolve())).selected_atm
    ptr=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int,ptr,ptr];fn.restype=None
    # Adjacent representable temperatures straddle both native thresholds;
    # wet/dry, cold/hot clamps and independent precipitation species matter.
    f=np.float32
    temps=[]
    for value in (f(273.16),f(f(273.16)+f(2.2))):
        temps.extend([np.nextafter(value,f(-np.inf)),value,np.nextafter(value,f(np.inf))])
    temps.extend(map(f,(210.,230.,260.,270.,280.,290.,330.)))
    rows=[]
    for k,t in enumerate(temps):
        for mix in ((0.,0.,0.,0.,0.,0.),(.001,.003,.0007,.0004,.0009,.0002),
                    (.002,.0001,.0001,.001,.0005,.0002),(.003,0.,.0002,.002,0.,0.)):
            rows.append([75000.+k*1700.,t,.0003+k*.0006,*mix,550.,-.1 if k%2 else .6])
    x=np.asarray(rows,dtype=np.float32)
    import cupy as cp
    from woof.core.noahmp_kernel_sources import selected_kernel
    kernel=selected_kernel('noahmp_leaves','noahmp_leaf_atm_selected')
    actual=cp.empty((len(x),17),dtype=cp.float32);device=cp.asarray(x)
    reports=[]
    for option in (2,3,4,5):
        expected=np.empty((len(x),17),dtype=np.float32)
        for i,row in enumerate(x):fn(option,row,expected[i])
        kernel(((len(x)+127)//128,),(128,),(device,cp.zeros(len(x),dtype=cp.int32),
            actual,np.int32(len(x)),np.int32(option)))
        got=cp.asnumpy(actual);mask=got.view(np.uint32)!=expected.view(np.uint32)
        reports.append(dict(opt_snf=option,cases=len(x),words=expected.size,
            differing_words=int(mask.sum()),first_differences=np.argwhere(mask)[:8].tolist()))
    receipt=dict(scope='constructed full native ATM procedure boundary checks',
        provenance=provenance,results=reports)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
    return int(any(row['differing_words'] for row in reports))


if __name__=='__main__':
    raise SystemExit(main())
