#!/usr/bin/env python3
"""Freeze the RRFS fork's framework declarations for the ported (no native library) route.

Same method as tools/framework_port/build_metadata.py, applied to the GSL v8.3.1 fork RRFSx/MPAS-Model
1d6e370e (pins in src/hexcore/data/mpasphys-native/mpas-source-rrfs.json): module variables and allocation
sets of the fork's mpas_atmphys_vars.F and allocate_* routines, every routine signature, the constant words of
mpas_constants.F / mpas_atmphys_constants.F (strict gfortran, single precision), and the fork Registry expanded
with its includes (diagnostics, Registry_noahmp.xml, Registry_tempo.xml, registry.chemistry.xml).

Developer proof tool: installed forecasts read only its small output and never fetch or compile the fork.

    python tools/framework_metadata_rrfs.py --source $MPAS_LANES/ref/rrfs/MPAS-Model-rrfs \
        --output src/hexcore/data/mpasphys-framework-rrfs --scratch /tmp/fw-rrfs
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--manifest-name', default='mpas-source-rrfs.json')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('build_metadata', here / 'framework_port/build_metadata.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    atmosphere = args.source / 'src/core_atmosphere'
    # The atmosphere core's Registry and the files it includes (Registry.xml:4669-4674); the fork checkout
    # also holds other cores' Registries, which the atmosphere build never reads.
    registries = [*sorted(atmosphere.rglob('Registry*.xml')), atmosphere / 'physics/registry.chemistry.xml']
    metadata = tool.build(args.source, args.output, args.scratch, manifest_name=args.manifest_name,
                          schema='mpas-framework-rrfs/v1', registries=registries)
    metadata.update(tempo_constants(args.source, args.scratch / 'tempo', tool))
    (args.output / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True, indent=1) + '\n')
    print(json.dumps(dict(tempo_constants=len(metadata['tempo_constants']))))


DUMP = '''module tempo_dump
 use iso_fortran_env
 implicit none
 interface dump
  module procedure d4, d8, di, a4, a8, ai
 end interface
contains
 subroutine d4(n, x)
  character(*), intent(in) :: n
  real(real32), intent(in) :: x
  write(*,'(A,1X,A,1X,I0,1X,Z8.8)') n, 'float32', 0, transfer(x, 0_int32)
 end subroutine
 subroutine d8(n, x)
  character(*), intent(in) :: n
  real(real64), intent(in) :: x
  write(*,'(A,1X,A,1X,I0,1X,Z16.16)') n, 'float64', 0, transfer(x, 0_int64)
 end subroutine
 subroutine di(n, x)
  character(*), intent(in) :: n
  integer(int32), intent(in) :: x
  write(*,'(A,1X,A,1X,I0,1X,Z8.8)') n, 'int32', 0, transfer(x, 0_int32)
 end subroutine
 subroutine a4(n, x)
  character(*), intent(in) :: n
  real(real32), intent(in) :: x(:)
  integer :: i
  do i = 1, size(x)
   write(*,'(A,1X,A,1X,I0,1X,Z8.8)') n, 'float32', i, transfer(x(i), 0_int32)
  enddo
 end subroutine
 subroutine a8(n, x)
  character(*), intent(in) :: n
  real(real64), intent(in) :: x(:)
  integer :: i
  do i = 1, size(x)
   write(*,'(A,1X,A,1X,I0,1X,Z16.16)') n, 'float64', i, transfer(x(i), 0_int64)
  enddo
 end subroutine
 subroutine ai(n, x)
  character(*), intent(in) :: n
  integer(int32), intent(in) :: x(:)
  integer :: i
  do i = 1, size(x)
   write(*,'(A,1X,A,1X,I0,1X,Z8.8)') n, 'int32', i, transfer(x(i), 0_int32)
  enddo
 end subroutine
end module
'''


def tempo_constants(source, scratch, tool):
    """The TEMPO module words tempo_init starts from: PARAMETERs and declaration values, before any init.

    TEMPO 9adf9ef module_mp_tempo_params.F90 compiled with the strict flags (-Dtempo_mpas, single precision);
    a capture program prints each word. Installed forecasts read these words and never compile TEMPO.
    """
    from woof.hex.mpasphys.rrfs.framework_init_rrfs import tempo_constant_names
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'src/hexcore/data/mpasphys-native'
                           / 'mpas-source-rrfs.json').read_text())
    params = source / 'src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90'
    relative = str(params.relative_to(source))
    digest = hashlib.sha256(params.read_bytes()).hexdigest()
    if digest != manifest['files'][relative]['sha256']:
        raise ValueError('source differs from pin: ' + relative)
    names = tempo_constant_names()
    scratch.mkdir(parents=True, exist_ok=True)
    program = ['program capture', 'use module_mp_tempo_params', 'use tempo_dump', 'implicit none']
    program += [f"call dump('{name}', {name})" for name in names] + ['end program']
    (scratch / 'dump.f90').write_text(DUMP)
    (scratch / 'capture.f90').write_text('\n'.join(program) + '\n')
    command = ['gfortran', '-O3', '-ffp-contract=off', '-fno-tree-vectorize', '-fno-unsafe-math-optimizations',
               '-cpp', '-DSINGLE_PRECISION', '-Dtempo_mpas', '-ffree-form', '-ffree-line-length-none', '-o', 'capture',
               str((source / 'src/framework/mpas_kind_types.F').resolve()), str(params.resolve()),
               'dump.f90', 'capture.f90']
    subprocess.run(command, cwd=scratch, check=True)
    rows = {}
    for line in subprocess.check_output([str((scratch / 'capture').resolve())], text=True).splitlines():
        name, dtype, index, word = line.split()
        row = rows.setdefault(name, dict(dtype=dtype, shape=[], words=[]))
        if int(index):
            row['shape'] = [int(index)]
        row['words'].append(int(word, 16))
    if set(rows) != set(names):
        raise ValueError('TEMPO constant capture missed ' + ', '.join(sorted(set(names) - set(rows))))
    command = [arg.replace(str(source.resolve()), '{source}') for arg in command]
    return dict(tempo_constants=rows, tempo_constants_source={relative: digest},
                tempo_constants_capture_command=command)


if __name__ == '__main__':
    main()
