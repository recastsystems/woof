#!/usr/bin/env python3
"""Export the four fixed native aerosol states after checking their licence.

Only an observer subroutine is added to the private native module. The
physics statements and DATA tables are left unchanged. This is build-time
Fortran, not a forecast dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from build import HERE, MANIFEST, verify


def export(source: Path, licence: Path, out: Path, compiler: str):
    source, licence, out = source.resolve(), licence.resolve(), out.resolve()
    verify(source, licence)
    out.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    marker = '      END MODULE MODULE_MP_NTU'
    if text.upper().count(marker) != 1:
        raise ValueError('native module end marker changed')
    observer = (HERE / 'export_aero.F90.inc').read_text()
    text = text.replace(marker, observer + '\n' + marker)
    module = out / 'module_mp_ntu.F'
    module.write_text(text)
    driver = out / 'export.F90'
    driver.write_text('''program export
use module_mp_ntu, only: lane_export_aero
implicit none
integer :: ccnty
character(len=10) :: path
do ccnty=1,4
write(path,'(A,I1,A)') 'ccnty',ccnty,'.bin'
call lane_export_aero(trim(path),ccnty)
enddo
end program
''')
    flags = ['-O2', '-fno-fast-math', '-ffp-contract=off', '-fno-tree-vectorize',
             '-ffree-form', '-ffree-line-length-none']
    command = [compiler, *flags, '-cpp', '-DRWORDSIZE=4', '-DDWORDSIZE=8',
               str(HERE.parent / 'ntu_wrf461_oracle' / 'stub_wrf_error.F90'),
               str(module), str(driver), '-o', 'export-aero']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run(command, cwd=out, env=env, check=True)
    subprocess.run([str(out/'export-aero')], cwd=out, env=env, check=True)
    bank = dict(schema='gpuwm.ntu-aero-native.v1', source_url=MANIFEST['module']['url'],
                source_sha256=MANIFEST['module']['sha256'], licence='WRF public-domain statement',
                endianness='little', layout_bytes=800,
                compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0], ccnty={})
    for ccnty in range(1,5):
        raw = (out/f'ccnty{ccnty}.bin').read_bytes()
        if len(raw) != 800:
            raise ValueError(f'native aerosol layout changed: {len(raw)} bytes')
        bank['ccnty'][str(ccnty)] = dict(bytes_hex=raw.hex(),sha256=hashlib.sha256(raw).hexdigest())
    (out/'ntu_aero_v471.json').write_text(json.dumps(bank,indent=2)+'\n')
    (out/'export-receipt.json').write_text(json.dumps(dict(command=command,cwd=str(out),
        source=MANIFEST,observer_sha256=hashlib.sha256(observer.encode()).hexdigest(),
        executable_sha256=hashlib.sha256((out/'export-aero').read_bytes()).hexdigest()),indent=2)+'\n')
    return bank


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source','licence','out'):
        ap.add_argument(name,type=Path)
    ap.add_argument('--compiler',default='gfortran')
    args=ap.parse_args()
    export(args.source,args.licence,args.out,args.compiler)
