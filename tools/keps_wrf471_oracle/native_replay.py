#!/usr/bin/env python3
"""Write a native single-call driver for captured real WRF KEPS inputs.

The optional defined mode changes ONLY the two uninitialized native reads.
It is a diagnostic witness, never the unmodified full-run replay authority.
Compile the emitted module and driver together with the recorded strict flags.
"""
from pathlib import Path
import argparse
import hashlib
from instrument import SOURCE_SHA256

MASS = ('u', 'v', 'th', 'qv', 'qc', 'rho', 'dz', 'pi', 'raten')
SURFACE = ('ust', 'hfx', 'qfx', 'br', 'znt', 'psim', 'psih')
STATE = ('tke', 'diss', 'tpe', 'pr', 'tkea', 'dissa', 'tpea')
OUTPUT = ('rub', 'rvb', 'rthb', 'rqvb', 'rqcb', 'wu', 'wv', 'wt', 'exch_h', 'exch_m')

def generate(source, directory, defined=False):
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
    text = raw.decode()
    if defined:
        assert text.count('maxtke < ttke') == text.count('maxtke = ttke') == 1
        text = text.replace('maxtke < ttke', 'maxtke < tke').replace('maxtke = ttke', 'maxtke = tke')
        assert text.count('     Pra=4.') == 1
        text = text.replace('     Pra=4.', '     Pra=4.\n     radsum=0.')
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'module_bl_keps.F').write_text(text)
    m3 = (*MASS, 'p8w', *STATE, *OUTPUT, 'wq', 'bep')
    m2 = (*SURFACE, 'pblh', 'unused')
    lines = ['program replay_real_call', 'use module_bl_keps, only: keps', 'implicit none',
             'integer nz,ny,nx,step,fi,fo,i,j,k', 'real dt,cp,g',
             'character(len=1024) input,output']
    lines += [f'real,allocatable :: {n}(:,:,:)' for n in m3]
    lines += [f'real,allocatable :: {n}(:,:)' for n in m2]
    lines += ['call get_command_argument(1,input)', 'call get_command_argument(2,output)',
              "open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')",
              'read(fi) nz,ny,nx,step', 'read(fi) dt,cp,g']
    lines += [f'allocate({n}(nx,nz+1,ny)); {n}=0.' for n in m3]
    lines += [f'allocate({n}(nx,ny)); {n}=0.' for n in m2]
    def read3(n, top='nz'):return f'read(fi) ((({n}(i,k,j),i=1,nx),j=1,ny),k=1,{top})'
    def put3(n, top='nz'):return f'write(fo) ((({n}(i,k,j),i=1,nx),j=1,ny),k=1,{top})'
    lines += [read3(n) for n in MASS] + [read3('p8w','nz+1')]
    lines += [f'read(fi) {n}' for n in SURFACE] + [read3(n) for n in STATE] + ['read(fi) pblh', 'close(fi)']
    lines += ["open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')",
              'write(fo) nz,ny,nx,step', 'write(fo) dt,cp,g']
    lines += [put3(n) for n in MASS] + [put3('p8w','nz+1')]
    lines += [f'write(fo) {n}' for n in SURFACE] + [put3(n) for n in STATE] + ['write(fo) pblh']
    args = ['unused','unused','step','unused','.false.','dz','dt','u','v','th','pi','raten','p8w','rho','qv','qc','hfx','qfx','ust','cp','g',
            'rub','rvb','rthb','rqvb','rqcb','tke','diss','tpe','tkea','dissa','tpea','pr','wu','wv','wt','wq','exch_h','exch_m','pblh',
            *(['bep']*11), 'br','znt','psim','psih',
            '1','nx+1','1','ny+1','1','nz+1', '1','nx','1','ny','1','nz+1', '1','nx','1','ny','1','nz']
    lines += ['call keps('+', &\n'.join(','.join(args[i:i+8]) for i in range(0,len(args),8))+')']
    lines += [put3(n) for n in (*OUTPUT,*STATE)] + ['write(fo) pblh','close(fo)','end program']
    (directory/'driver.F90').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('directory',type=Path)
    p.add_argument('--defined-undefined',action='store_true')
    a=p.parse_args();generate(a.source,a.directory,a.defined_undefined)
