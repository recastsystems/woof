#!/usr/bin/env python3
"""Witness the native DVIC uninitialized YD read on one constructed column.

This is a development diagnostic, not a WRF forecast or GPU replay. Both
executables contain the same verbatim native procedures. Only the compiler's
local-variable initialization policy differs. No initialization is patched
into the scientific source, and this does not define a production policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess

from .build import LSM_SHA256, NOAHMP_PIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lsm', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    args = parser.parse_args()
    original = args.lsm.read_bytes()
    if hashlib.sha256(original).hexdigest() != LSM_SHA256:
        raise ValueError('refusing unpinned module_sf_noahmplsm.F')
    text = original.decode()
    start = text.index('  SUBROUTINE  DYNAMIC_VIC(')
    end = text.index('  END SUBROUTINE PHILIP_INFIL', start)
    end += len('  END SUBROUTINE PHILIP_INFIL')
    hydraulic_start = text.index('  SUBROUTINE WDFCND2 (')
    hydraulic_end = text.index('  END SUBROUTINE WDFCND2', hydraulic_start)
    hydraulic_end += len('  END SUBROUTINE WDFCND2')
    body = text[start:end]
    hydraulic = text[hydraulic_start:hydraulic_end]
    prefix = '''module native_dvic_probe
implicit none
integer :: opt_infdv=1
type noahmp_parameters
 real :: bbvic, bdvic, gdvic
 real :: smcmax(4), smcwlt(4), dksat(4), dwsat(4), bexp(4)
end type
contains
'''
    main_program = '''
end module
program probe
use native_dvic_probe
implicit none
type(noahmp_parameters) :: parameters
real :: smc(4), sh2o(4), sice(4), zsoil(4), facc, pddum, runsrf
parameters%bbvic=1.0
parameters%bdvic=0.2
parameters%gdvic=0.1
parameters%smcmax=0.45
parameters%smcwlt=0.05
parameters%dksat=1.e-6
parameters%dwsat=1.e-5
parameters%bexp=4.0
smc=0.4
sh2o=smc
sice=0.0
zsoil=[-0.1,-0.4,-1.0,-2.0]
facc=0.001
call dynamic_vic(parameters,60.0,smc,sh2o,sice,0.0,4,zsoil,0.001,facc,pddum,runsrf)
print *, 'native completed: facc,pddum,runsrf=',facc,pddum,runsrf
end program
'''
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = work / 'native_dvic_probe.F90'
    with source.open('x') as stream:
        stream.write(prefix + body + '\n' + hydraulic + '\n' + main_program)
    flags = ['-O0', '-g', '-fbacktrace', '-ffp-contract=off', '-fno-fast-math',
             '-fno-tree-vectorize', '-ffree-line-length-none', '-ffpe-trap=invalid',
             '-fcheck=all']
    runs = []
    for initialization in ('zero', 'snan'):
        executable = work / ('dvic_' + initialization)
        command = ['gfortran', *flags, '-finit-real=' + initialization,
                   str(source), '-o', str(executable)]
        subprocess.run(command, cwd=work, check=True, capture_output=True, text=True)
        run = subprocess.run([str(executable)], cwd=work, capture_output=True, text=True)
        runs.append(dict(initialization=initialization, command=command,
                         exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr))
    report = dict(scope='one constructed native DVIC column; not GPU or real WRF-run proof',
                  noahmp_pin=NOAHMP_PIN, source_sha256=LSM_SHA256,
                  source_lines=[text[:start].count('\n') + 1, text[:end].count('\n') + 1],
                  verbatim_block_sha256=hashlib.sha256(body.encode()).hexdigest(),
                  generated_source=str(source), generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  native_unassigned_read=dict(argument_line=8263, consuming_line=8421,
                                               variable='YD', routine='RR1'),
                  compiler=subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0],
                  config=dict(opt_run=8, opt_infdv=1, dt=60., qinsur=.001,
                              smc=[.4]*4, ice=[0.]*4, facc=.001), runs=runs)
    print(json.dumps(report, indent=2))
    # The backtrace must name the real consuming native routine, not just any
    # floating-point trap. A failure here invalidates the proposed witness.
    assert runs[0]['exit_code'] == 0, 'zero-init control did not complete'
    assert runs[1]['exit_code'] == -signal.SIGFPE, 'sNaN policy did not trap'
    assert '__native_dvic_probe_MOD_rr1' in runs[1]['stderr'], 'trap was outside RR1'


if __name__ == '__main__':
    main()
