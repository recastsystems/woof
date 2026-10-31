#!/usr/bin/env python3
"""Export the verified fork/gitlinks offline; patch build plumbing only.

Extends f0 build_reference.sh and native_dry_reference/edit_makefile_flags.py.
No copied .git indirection can write into the read-only reference repositories.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rrfs.diagnostic_flags import DIAGNOSTICS, checked_flags
PINS = json.loads((HERE / 'pins.json').read_text())


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def export(source, destination, commit, expected_head=None):
    if git(source, 'rev-parse', 'HEAD') != (expected_head or commit):
        raise ValueError(f'wrong pin: {source}')
    if git(source, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError(f'tracked source modified: {source}')
    process = subprocess.Popen(['git', '-C', str(source), 'archive', commit], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
        archive.extractall(destination, filter='data')
    process.stdout.close()
    if process.wait():
        raise RuntimeError(f'export failed: {source}')


def add_diagnostic_flags(text, diagnostic='bounds-snan'):
    """Keep C/C++ flags intact while checking every native Fortran object."""
    head, tail = text.split('\ngfortran:', 1)
    block, rest = tail.split('\ngfortran-clang:', 1)
    lines = block.splitlines(keepends=True)
    selected = [i for i, line in enumerate(lines) if '"FFLAGS_OPT = ' in line]
    if len(selected) != 1 or lines[selected[0]].count('-fno-unsafe-math-optimizations ') != 1:
        raise ValueError('changed GNU Fortran diagnostic flag location')
    i = selected[0]
    lines[i] = lines[i].replace('-fno-unsafe-math-optimizations ',
        '-fno-unsafe-math-optimizations ' + ' '.join(checked_flags(diagnostic)) + ' ')
    return head + '\ngfortran:' + ''.join(lines) + '\ngfortran-clang:' + rest


def prepare(source, destination, recipe, tables, reference='primary', diagnostic=None):
    if diagnostic and recipe != 'gnu-strict':
        raise ValueError('native diagnostics require the strict recipe')
    if diagnostic:
        checked_flags(diagnostic)
    commit = PINS['fork'] if reference == 'primary' else PINS['secondary']
    submodules = dict(PINS['submodules'])
    if reference == 'secondary':
        submodules = {line.split()[3]: line.split()[2]
                      for line in git(source, 'ls-tree', '-r', commit).splitlines() if line.startswith('160000 ')}
    destination.mkdir(parents=True, exist_ok=False)
    export(source, destination, commit, PINS['fork'])
    inactive = {}
    for relative, pin in submodules.items():
        module_source = source / relative
        if relative == 'src/core_atmosphere/stochastic_physics' and reference == 'secondary':
            # This secondary receipt is explicitly STOCHASTIC_PHYSICS=false.
            # Its dycore-only comparison never compiles this optional gitlink.
            inactive[relative] = pin
            continue
        export(module_source, destination / relative, pin, PINS['submodules'].get(relative))
    # Use f0's pinned MPAS-Data, not checkout_data_files.sh's network fallback.
    import shutil
    target = destination / 'src/core_atmosphere/physics/physics_wrf/files'
    target.mkdir(exist_ok=True)
    for file in tables.iterdir():
        if file.is_file():
            shutil.copy2(file, target / file.name)
    if '8.2' not in (target / 'COMPATIBILITY').read_text().split():
        raise ValueError('MPAS-Data compatibility mismatch')
    originals = {}
    make = destination / 'Makefile'
    originals[make] = make.read_text()
    if recipe == 'gnu-strict':
        subprocess.run([sys.executable, str(HERE.parents[1] / 'native_dry_reference/edit_makefile_flags.py'),
                        str(make), 'gfortran', '-O3 ',
                        '-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations '], check=True)
        head, tail = make.read_text().split('\ngfortran:', 1)
        block, rest = tail.split('\ngfortran-clang:', 1)
        for name in ('CFLAGS_OPT', 'CXXFLAGS_OPT'):
            old = f'"{name} = -O3"'
            if block.count(old) != 1:
                raise ValueError(f'changed GNU {name}')
            block = block.replace(old, f'"{name} = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations"')
        make.write_text(head + '\ngfortran:' + block + '\ngfortran-clang:' + rest)
    if diagnostic:
        make.write_text(add_diagnostic_flags(make.read_text(), diagnostic))
    physics = destination / 'src/core_atmosphere/physics/Makefile'
    originals[physics] = physics.read_text()
    checkout = '\t./../tools/manage_externals/checkout_externals --externals ./../Externals.cfg\n'
    if originals[physics].count(checkout) != 1:
        raise ValueError('changed external checkout rule')
    physics.write_text(originals[physics].replace(checkout, '\t@echo "Pinned externals exported and verified offline by rrfs/prepare_build.py"\n'))
    # The fork's obsolete atmphys_build_tables_tempo is not part of its build.
    # Compile the pinned TEMPO v3 table program against exactly the MPAS modules
    # used by the atmosphere executable. No old API adapter or arithmetic patch.
    utility = destination / 'src/core_atmosphere/utils/Makefile'
    originals[utility] = utility.read_text()
    utility.write_text(originals[utility].replace('UTILS = build_tables', 'UTILS = build_tables build_tables_tempo_v3') + '''

# RRFS native reference: TEMPO 9adf9ef tools/build_tables.F90, unmodified.
build_tables_tempo_v3: tempo_v3_tables.o tempo_v3_program.o
\t$(LINKER) $(LDFLAGS) -o $@ tempo_v3_program.o tempo_v3_tables.o -L../../framework -L../physics -lphys -lframework $(LIBS) -L../../external/esmf_time_f90 -lesmf_time
\tmv $@ ../../..

tempo_v3_tables.o: ../physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90
\t$(FC) $(CPPFLAGS) $(FFLAGS) -Dtempo_mpas -Dbuild_tables_with_mpi=1 -c $< -o $@ $(FCINCLUDES) -I../../framework -I../physics/physics_noaa/TEMPO

tempo_v3_program.o: ../physics/physics_noaa/TEMPO/tools/build_tables.F90 tempo_v3_tables.o
\t$(FC) $(CPPFLAGS) $(FFLAGS) -Dtempo_mpas -Dbuild_tables_with_mpi=1 -c $< -o $@ $(FCINCLUDES) -I../../framework -I../physics/physics_noaa/TEMPO
''')
    smiol = destination / 'src/external/SMIOL/Makefile'
    originals[smiol] = smiol.read_text()
    rule = '$(FC) $(CPPINCLUDES) $(FFLAGS) -c $<'
    if originals[smiol].count(rule) != 1:
        raise ValueError('changed SMIOL Fortran rule')
    smiol.write_text(originals[smiol].replace(rule, '$(FC) $(CPPFLAGS) $(CPPINCLUDES) $(FFLAGS) -c $<'))
    diff = ''.join(''.join(difflib.unified_diff(old.splitlines(True), path.read_text().splitlines(True),
                   fromfile=str(path.relative_to(destination)), tofile=str(path.relative_to(destination))))
                   for path, old in originals.items())
    (destination / 'build-source.diff').write_text(diff)
    manifest = dict(PINS, fork=commit, reference=reference, submodules=submodules,
                    inactive_submodules=inactive, stochastic_physics=False,
                    recipe=recipe, diagnostic=diagnostic, source=str(source.resolve()),
                    table_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()})
    (destination / 'SOURCE-PINS.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--recipe', choices=('gnu-strict', 'gnu-o3'), required=True)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--reference', choices=('primary', 'secondary'), default='primary')
    parser.add_argument('--diagnostic', choices=tuple(DIAGNOSTICS))
    args = parser.parse_args()
    prepare(args.source, args.destination, args.recipe, args.tables, args.reference, args.diagnostic)
