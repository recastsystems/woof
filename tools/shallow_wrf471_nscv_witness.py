"""Build/run the named native unwritten-output witness, from kept captures."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('captures',type=Path)
p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
s=a.source.read_text()
s=s.replace('MODULE module_shcu_nscv','MODULE module_shcu_nscv\n integer :: witness_seed',1)
s=s.replace('     call nscv2d(', '     kbot(:)=witness_seed;ktop(:)=witness_seed\n     call nscv2d(',1)
assert s.count('kbot(:)=witness_seed')==1
src=a.output/'module_shcu_nscv_witness.f90';src.write_text(s)
exe=(a.output/'witness').resolve()
subprocess.run(['gfortran','-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize',
    '-ffree-line-length-none',str(src.resolve()),
    str(Path(__file__).with_suffix('.F90').resolve()),'-o',str(exe)],cwd=a.output,check=True)
for capture in sorted(a.captures.glob('*.bin')):
    for seed in (-12345,-23456):
        subprocess.run([str(exe),str(capture.resolve()),str((a.output/(capture.stem+f'.{seed}.bin')).resolve()),str(seed)],check=True)
