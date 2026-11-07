"""Build WRF's source-bound REAL32 Deng saturation table with strict GNU flags."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import translate

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
a.directory.mkdir(parents=True,exist_ok=True)
s=translate.SOURCE.read_text();assert hashlib.sha256(s.encode()).hexdigest()==translate.PIN
body=s[s.lower().index('      subroutine kf_lutab('):s.index('   END SUBROUTINE KF_LUTAB')+len('   END SUBROUTINE KF_LUTAB')]
head='''module deng_tables
integer,parameter::kfnt=250,kfnp=220
real::ttab(kfnt,kfnp),qstab(kfnt,kfnp),the0k(kfnp),alu(200),rdpr,rdthk,plutop
contains
'''
program='''
end module
program build
use deng_tables
integer::u
call kf_lutab(0.6112,17.67,29.65,273.15)
open(newunit=u,file='deng_tables.bin',access='stream',form='unformatted',status='replace',convert='little_endian')
write(u) ttab,qstab,the0k,alu,rdpr,rdthk,plutop
close(u)
end program
'''
f=a.directory/'tables.f90';f.write_text(head+body+program)
cmd=['gfortran','-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize','-ffree-line-length-none',str(f.resolve()),'-o',str((a.directory/'tables').resolve())]
subprocess.run(cmd,check=True,cwd=a.directory)
subprocess.run(['./tables'],cwd=a.directory,check=True)
b=a.directory/'deng_tables.bin';print(' '.join(cmd));print(b.stat().st_size,hashlib.sha256(b.read_bytes()).hexdigest())
