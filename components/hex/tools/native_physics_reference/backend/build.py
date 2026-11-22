#!/usr/bin/env python3
"""Build pristine MPAS physics and its framework closure from an offline pin.

Run through the build queue; this writes only a private build tree and uses make -j4.
MPAS Makefile: GNU flags; native_dry_reference/build_native_reference.sh: strict flags.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

STRICT = ['-O3', '-ffree-line-length-none',
          '-fconvert=big-endian', '-ffree-form', '-ffp-contract=off',
          '-fno-tree-vectorize', '-fno-unsafe-math-optimizations', '-fPIC']
PIN = 'f34984b2c89353cf31c235b08d3b4acd96b37bb3'
INTRINSIC = {'iso_c_binding', 'iso_fortran_env', 'ieee_arithmetic', 'ieee_exceptions',
             'ieee_features', 'mpi', 'mpi_f08', 'netcdf', 'omp_lib'}
# THE BREAKAGE THIS PREVENTS (no default routes into code that cannot be sold):
# MPAS-A v8.4.2's own sources USE Noah-MP v5.0 (UCAR terms: no sale of the Software or of any work that
# contains it), so a library compiled from the full tree contains it even when no run selects it.
# --licence-stub noahmp compiles the Apache-2.0 interface stub beside this file (make_licence_stubs.py) in
# place of physics_noahmp: MPAS's own sources compile unchanged against it, and any stub routine that is called
# stops the run naming the opt-in.  Without --licence-stub the build is byte-for-byte the build it was before
# stubs existed.  (UGWP and the other NOAA-GSL physics are compiled as always: they
# are treated as US Government code and are not held.)
LICENCE_STUBS = {
    'noahmp': {'file': 'licence_stub_noahmp.F90', 'terms': 'ucar-noahmp',
               'replaces': 'src/core_atmosphere/physics/physics_noahmp'},
}


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def objects(makefile):
    """Read upstream object lists, preserving upstream choice of scheme variants."""
    text = re.sub(r'\\\n', ' ', makefile.read_text())
    return [x for m in re.finditer(r'^OBJS(?:_init)?\s*=\s*(.*)$', text, re.M)
            for x in re.findall(r'[\w.]+\.o', m[1])]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--pnetcdf', type=Path)
    ap.add_argument('--math-object', type=Path)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--probe', action='store_true', help='also build the direct native driver probe')
    ap.add_argument('--tap-transport', type=Path, help='existing f1 mpas_oracle_dump.F for driver capture tests')
    # An installed woof hex builds this library on the user's host from files fetched by SHA-256
    # (woof.hex.mpasphys.native_library_build): that tree has no .git, so its pins are the manifest's
    # per-file SHA-256 instead of the checkouts' HEADs.  The oracle transport is lab tooling (rustc).
    ap.add_argument('--source-manifest', type=Path,
                    help='verify the source tree file by file against this gpuwm-hex.mpas-source-manifest/v1 '
                         'instead of reading its git pins')
    ap.add_argument('--no-oracle-transport', action='store_true',
                    help='do not build libmpas_oracle_transport.so (the f1 capture tooling; needs rustc)')
    ap.add_argument('--static-gnu-runtime', action='store_true',
                    help='link GNU Fortran and GCC runtimes statically for release assets')
    ap.add_argument('--jobs', type=int, default=4, help='make -j (default 4)')
    ap.add_argument('--licence-stub', action='append', default=[], choices=sorted(LICENCE_STUBS),
                    help='compile this component\'s interface stub instead of its upstream code (repeatable)')
    a = ap.parse_args()
    source = a.source.resolve()
    stubbed = {name: LICENCE_STUBS[name] for name in sorted(set(a.licence_stub))}
    replaced = tuple(row['replaces'] for row in stubbed.values())
    external_dirs = [('physics_mmm', 'a4baf7f3243d1db0dbc5f63473f895bdbdc05c30'),
                     ('physics_noaa/UGWP', 'c1c893edcf171af5639af60e3a3a528816f6cc2b')]
    external_dirs = [(d, pin) for d, pin in external_dirs if 'src/core_atmosphere/physics/' + d not in replaced]
    source_verification = {'method': 'git'}
    if a.source_manifest:
        manifest = json.loads(a.source_manifest.read_text())
        if manifest.get('schema') != 'gpuwm-hex.mpas-source-manifest/v1' or manifest['pins']['mpas-model'] != PIN:
            raise SystemExit('source manifest is not the pinned MPAS v8.4.2 manifest')
        for relative, row in manifest['files'].items():
            if replaced and relative.startswith(tuple(r + '/' for r in replaced)):
                continue   # a stubbed component is neither fetched nor compiled
            path = source / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise SystemExit(f'source differs from the pinned manifest: {relative}')
        verified = sum(1 for relative in manifest['files']
                       if not (replaced and relative.startswith(tuple(r + '/' for r in replaced))))
        source_verification = {'method': 'manifest', 'files': verified,
                               'manifest_sha256': hashlib.sha256(a.source_manifest.read_bytes()).hexdigest()}
    elif run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True).stdout.strip() != PIN:
        raise SystemExit('source is not pinned MPAS v8.4.2')
    work = a.work.resolve()
    if work == source or source in work.parents:
        raise SystemExit('build must be outside the read-only source tree')
    work.mkdir(parents=True, exist_ok=True)
    compiler = run(['gfortran', '--version'], capture_output=True).stdout.splitlines()[0]
    mpi_compilers = {name: shutil.which(name) for name in ['mpif90', 'mpicc']}
    external_pins = {}
    for directory, pin in external_dirs:
        if a.source_manifest:
            origin = {'physics_mmm': 'mmm-physics', 'physics_noaa/UGWP': 'ugwp'}[directory]
            if manifest['pins'][origin] != pin:
                raise SystemExit(f'external pin mismatch: {directory}: {manifest["pins"][origin]}')
            external_pins[directory] = pin
            continue
        path = source / 'src/core_atmosphere/physics' / directory
        actual = run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True).stdout.strip()
        if actual != pin:
            raise SystemExit(f'external pin mismatch: {directory}: {actual}')
        external_pins[directory] = actual
    private = work / 'source'
    if not private.exists():
        if replaced:
            # A stubbed component's upstream files never enter the build tree, even from a full checkout.
            held = {(source / r).resolve() for r in replaced}
            shutil.copytree(source / 'src', private / 'src',
                            ignore=lambda folder, names: [n for n in names if (Path(folder) / n).resolve() in held])
        else:
            shutil.copytree(source / 'src', private / 'src')
    for relative in replaced:
        if (private / relative).exists():
            shutil.rmtree(private / relative)
    # Never modify copied Fortran; generated/preprocessed units live separately.
    physics = private / 'src/core_atmosphere/physics'
    framework = private / 'src/framework'
    external = private / 'src/external'
    dirs = [physics, physics / 'physics_wrf', physics / 'physics_mmm',
            physics / 'physics_noaa/UGWP', physics / 'physics_noahmp/utility',
            physics / 'physics_noahmp/src', physics / 'physics_noahmp/drivers/mpas']
    dirs = [d for d in dirs if not any(d == private / r or private / r in d.parents for r in replaced)]
    chosen = []
    for d in dirs:
        mf = d / ('Makefile.mpas' if d.name == 'physics_mmm' else 'Makefile')
        for obj in objects(mf):
            paths = [d / (obj[:-2] + s) for s in ['.F', '.F90', '.f90']]
            chosen.append(next(p for p in paths if p.exists()))
    # Each stub is compiled from a copy in the work tree, checked against the file beside this builder.
    stubs = {}
    for name, row in stubbed.items():
        original = Path(__file__).with_name(row['file'])
        copy = work / 'licence_stubs' / row['file']
        copy.parent.mkdir(exist_ok=True)
        write_changed(copy, original.read_text())
        stubs[copy] = original
        chosen.append(copy)
    candidates = sorted(set(chosen + [private / 'src/core_atmosphere/mpas_atm_dimensions.F'] + list(framework.glob('*.F')) +
                            list((external / 'esmf_time_f90').glob('*.F90'))))
    if a.pnetcdf:
        candidates.append(external / 'SMIOL/smiolf.F90')
    includes = sorted(set(p.parent for p in candidates))
    includes += [private / 'src/core_atmosphere', Path('/usr/include')]
    macros = ['-DSINGLE_PRECISION', '-Dmpas', '-D_MPI', '-DMPAS_USE_MPI_F08',
              '-DCORE_ATMOSPHERE', '-DMPAS_NAMELIST_SUFFIX=atmosphere']
    if a.pnetcdf:
        macros += ['-DMPAS_SMIOL_SUPPORT']
        includes += [a.pnetcdf.resolve() / 'include', external / 'SMIOL']
    incflags = ['-I' + str(p) for p in includes]
    (work / 'preprocessed').mkdir(exist_ok=True)
    (work / 'objects').mkdir(exist_ok=True)
    modules, units = {}, {}
    for i, path in enumerate(candidates):
        flags = macros + (['-DHIDE_MPI'] if 'esmf_time_f90' in path.parts else [])
        result = run(['cpp', '-P', '-traditional', *flags, *incflags, str(path)], capture_output=True)
        pp = work / 'preprocessed' / f'{i:04d}_{path.stem}.f90'
        write_changed(pp, result.stdout)
        provided = re.findall(r'^\s*module\s+(?!procedure\b|function\b|subroutine\b)(\w+)', result.stdout, re.I | re.M)
        used = set(x.lower() for x in re.findall(r'^\s*use\b\s*(?:::)?\s*(\w+)', result.stdout, re.I | re.M))
        units[path] = (pp, used, hashlib.sha256(path.read_bytes()).hexdigest())
        original = stubs[path] if path in stubs else source / path.relative_to(private)
        if hashlib.sha256(original.read_bytes()).hexdigest() != units[path][2]:
            raise SystemExit(f'private source differs from pinned source: {path}')
        for name in provided:
            name = name.lower()
            if name in modules and modules[name] != path:
                raise SystemExit(f'duplicate module {name}: {path} / {modules[name]}')
            modules[name] = path
    selected = set(chosen)
    # External procedures do not appear in USE dependencies.
    selected.update(external / 'esmf_time_f90' / name for name in
                    ['wrf_error_fatal.F90', 'wrf_message.F90'])
    roots = ['mpas_pool_routines', 'mpas_timekeeping', 'mpas_dmpar', 'mpas_log', 'mpas_timer']
    selected.update(modules[x] for x in roots)
    queue = list(selected)
    edges = {}
    while queue:
        path = queue.pop()
        deps = set()
        for name in units[path][1]:
            if name in INTRINSIC:
                continue
            if name not in modules:
                raise SystemExit(f'missing module {name} used by {path}')
            dep = modules[name]
            if dep != path:
                deps.add(dep)
                if dep not in selected:
                    selected.add(dep)
                    queue.append(dep)
        edges[path] = deps
    # Bind(C) wrappers are generated from driver signatures, not hand-typed lists.
    # The read-only Noah-MP IO transport indexes Noah-MP's own IO type, so a Noah-MP stub build has none.
    noahmp_io = 'noahmp' not in stubbed
    run([sys.executable, str(Path(__file__).with_name('generate.py')), '--source', str(source),
         '--output', str(work), *([] if noahmp_io else ['--no-noahmp-io'])])
    wrapper = work / 'native_wrappers.f90'
    obj = {p: 'objects/' + units[p][0].stem + '.o' for p in selected}
    quote = shlex.quote
    fc = 'mpif90'
    ff = ' '.join(quote(x) for x in STRICT + ['-Iobjects', '-I/usr/include'])
    lines = ['.PHONY: all', 'all: libmpasphys_native.so', '']
    cpaths = [framework / (x + '.c') for x in ['pool_hash', 'random_id', 'regex_matching',
              'xml_stream_parser', 'stream_inquiry', 'ptscotch_interface']]
    cpaths.append(external / 'ezxml/ezxml.c')
    if a.pnetcdf:
        cpaths += [external / 'SMIOL/smiol.c', external / 'SMIOL/smiol_utils.c']
    dependency_hash = (hashlib.sha256((a.pnetcdf / 'lib/libpnetcdf.a').read_bytes()).hexdigest()
                       if a.pnetcdf else None)
    recipe = {'flags': STRICT, 'macros': macros, 'static_gnu_runtime': a.static_gnu_runtime, 'compiler': compiler,
              **mpi_compilers, 'includes': [str(p) for p in includes],
              'pnetcdf_sha256': dependency_hash,
              'mpi_wrapper': run(['mpif90', '-show'], capture_output=True).stdout.strip()}
    write_changed(work / 'compile-recipe.json', json.dumps(recipe, sort_keys=True, indent=2) + '\n')
    cobjs = []
    for path in cpaths:
        target = 'objects/c_' + path.stem + '.o'
        cobjs.append(target)
        flags = ['-O3', '-fPIC', '-std=gnu17', '-I' + str(external / 'ezxml'), *macros, *incflags]
        if a.pnetcdf:
            flags += ['-DSMIOL_PNETCDF']
        original = source / path.relative_to(private)
        if hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest():
            raise SystemExit(f'private C source differs from pin: {path}')
        lines += [f'{target}: {quote(str(path))} compile-recipe.json', '\tmpicc ' + ' '.join(quote(x) for x in flags) + f' -c $< -o $@', '']
    for path in sorted(selected):
        pp = units[path][0]
        lines += [f'{obj[path]}: {quote(str(pp))} compile-recipe.json ' + ' '.join(obj[d] for d in sorted(edges[path])),
                  f'\t{fc} {ff} -Jobjects -c $< -o $@', '']
    runtime = Path(__file__).with_name('runtime.f90').resolve()
    table_wrapper = work / 'native_tables.f90'
    from framework import generate_framework
    generate_framework(work)
    framework_wrapper = work / 'native_framework.f90'
    allobjs = list(sorted(obj.values())) + cobjs + ['objects/native_wrappers.o', 'objects/native_runtime.o', 'objects/native_tables.o', 'objects/native_framework.o', 'objects/native_noahmp_io.o']
    if not noahmp_io:
        allobjs.remove('objects/native_noahmp_io.o')
    lines += ['objects/native_wrappers.o: ' + quote(str(wrapper)) + ' ' + ' '.join(sorted(obj.values())),
              f'\t{fc} {ff} -Jobjects -c $< -o $@', '']
    lines += ['objects/native_runtime.o: ' + quote(str(runtime)) + ' ' + ' '.join(sorted(obj.values())),
              f'\t{fc} {ff} -Jobjects -c $< -o $@', '']
    lines += ['objects/native_tables.o: ' + quote(str(table_wrapper)) + ' ' + ' '.join(sorted(obj.values())),
              f'\t{fc} {ff} -Jobjects -c $< -o $@', '']
    lines += ['objects/native_framework.o: ' + quote(str(framework_wrapper)) + ' objects/native_runtime.o',
              f'\t{fc} {ff} -Jobjects -c $< -o $@', '']
    io_wrapper = work / 'native_noahmp_io.f90'
    if noahmp_io:
        lines += ['objects/native_noahmp_io.o: ' + quote(str(io_wrapper)) + ' ' + ' '.join(sorted(obj.values())),
                  f'\t{fc} {ff} -Jobjects -c $< -o $@', '']
    link = ['-lnetcdff', '-lnetcdf', '-lm']
    if a.pnetcdf:
        lib = a.pnetcdf.resolve() / 'lib'
        link = ['-L' + str(lib), '-Wl,-rpath,' + str(lib), '-lpnetcdf', *link]
    if a.static_gnu_runtime:
        link += ['-static-libgfortran', '-static-libgcc']
    extra = [str(a.math_object.resolve())] if a.math_object else []
    lines += ['libmpasphys_native.so: ' + ' '.join(allobjs + [quote(x) for x in extra]),
              f'\t{fc} -shared -Wl,-z,defs -Wl,-Bsymbolic-functions -o $@ ' +
              ' '.join(allobjs + [quote(x) for x in extra + link]), '']
    probe = Path(__file__).with_name('probe_cloudiness.f90').resolve()
    case_columns = Path(__file__).with_name('case_columns.f90').resolve()
    lines += ['objects/case_columns.o: ' + quote(str(case_columns)) + ' libmpasphys_native.so compile-recipe.json',
              f'\t{fc} {ff} -Jobjects -c $< -o $@', '']
    lines += ['probe_cloudiness: ' + quote(str(probe)) + ' objects/case_columns.o libmpasphys_native.so compile-recipe.json',
              f'\t{fc} {ff} -cpp $< objects/case_columns.o -L. -lmpasphys_native -lnetcdff -lnetcdf -Wl,-rpath,' + quote(str(work)) + ' -o $@', '']
    table_probe = Path(__file__).with_name('probe_tables.f90').resolve()
    lines += ['probe_tables: ' + quote(str(table_probe)) + ' libmpasphys_native.so compile-recipe.json',
              f'\t{fc} {ff} $< -L. -lmpasphys_native -Wl,-rpath,' + quote(str(work)) + ' -o $@', '']
    constants_probe = Path(__file__).with_name('probe_constants.f90').resolve()
    lines += ['probe_constants: ' + quote(str(constants_probe)) + ' libmpasphys_native.so compile-recipe.json',
              f'\t{fc} {ff} $< -L. -lmpasphys_native -Wl,-rpath,' + quote(str(work)) + ' -o $@', '']
    surface_probe = Path(__file__).with_name('probe_surface.f90').resolve()
    lines += ['probe_surface: ' + quote(str(surface_probe)) + ' libmpasphys_native.so compile-recipe.json',
              f'\t{fc} {ff} $< -L. -lmpasphys_native -Wl,-rpath,' + quote(str(work)) + ' -o $@', '']
    oracle = Path(__file__).with_name('oracle_transport.rs').resolve()
    lines += ['libmpas_oracle_transport.so: ' + quote(str(oracle)),
              '\trustc --edition=2024 --crate-type cdylib -O $< -o $@', '']
    if a.tap_transport:
        tap = work / 'tap_transport.f90'
        write_changed(tap, a.tap_transport.read_text())
        lines += ['objects/tap_transport.o: ' + quote(str(tap)),
                  f'\t{fc} {ff} -Jobjects -c $< -o $@',
                  'probe_cloudiness_capture: ' + quote(str(probe)) + ' objects/tap_transport.o objects/case_columns.o libmpasphys_native.so',
                  f'\t{fc} {ff} -cpp -DMPAS_NATIVE_ORACLE_PROBE $< objects/tap_transport.o objects/case_columns.o -L. -lmpasphys_native -lnetcdff -lnetcdf -Wl,-rpath,' + quote(str(work)) + ' -o $@', '']
    write_changed(work / 'Makefile', '\n'.join(lines))
    # What the library carries of each restricted or unlicensed component: its upstream code, or its stub.
    licence = {'schema': 'gpuwm-hex.native-licence/v1',
               'stubbed': {name: {'terms': row['terms'], 'replaces': row['replaces'], 'stub_source': row['file'],
                                  'stub_source_sha256': hashlib.sha256(
                                      Path(__file__).with_name(row['file']).read_bytes()).hexdigest()}
                           for name, row in stubbed.items()},
               'compiled': {name: {'terms': row['terms'], 'source': row['replaces']}
                            for name, row in LICENCE_STUBS.items() if name not in stubbed}}
    receipt = {'schema': 'mpas-native-backend-build-v1', 'source_pin': PIN,
               'flags': STRICT, 'macros': macros, 'static_gnu_runtime': a.static_gnu_runtime, 'io': 'SMIOL' if a.pnetcdf else 'none-development-only',
               'pool_error_policy': 'MPAS_POOL_SILENT (pinned non-MPAS_DEBUG framework)',
               'math_object_sha256': hashlib.sha256(a.math_object.read_bytes()).hexdigest() if a.math_object else None,
               'sources': {(f'licence_stubs/{p.name}' if p in stubs else str(p.relative_to(private))): units[p][2]
                           for p in sorted(selected)},
               'c_sources': {str(p.relative_to(private)): hashlib.sha256(p.read_bytes()).hexdigest() for p in cpaths},
               'external_pins': external_pins, 'licence': licence, 'compile_recipe': recipe,
               'compiler': compiler, **mpi_compilers,
               'abi_sha256': hashlib.sha256((work / 'abi.json').read_bytes()).hexdigest(),
               'wrappers_sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest(),
               'runtime_sha256': hashlib.sha256(runtime.read_bytes()).hexdigest(),
               'tables_wrapper_sha256': hashlib.sha256(table_wrapper.read_bytes()).hexdigest(),
               'framework_wrapper_sha256': hashlib.sha256(framework_wrapper.read_bytes()).hexdigest(),
               'noahmp_io_wrapper_sha256': hashlib.sha256(io_wrapper.read_bytes()).hexdigest() if noahmp_io else None,
               'proof': False, 'status': 'prepared'}
    if a.source_manifest:
        receipt['source_verification'] = source_verification
    if not a.prepare_only:
        run(['make', f'-j{a.jobs}', 'all', *([] if a.no_oracle_transport else ['libmpas_oracle_transport.so']),
             *(['probe_cloudiness', 'probe_tables', 'probe_constants', 'probe_surface'] if a.probe else []),
             *(['probe_cloudiness_capture'] if a.tap_transport else [])], cwd=work)
        receipt.update(status='built', library_sha256=hashlib.sha256((work / 'libmpasphys_native.so').read_bytes()).hexdigest())
        if not a.no_oracle_transport:
            receipt['oracle_transport_sha256'] = hashlib.sha256((work / 'libmpas_oracle_transport.so').read_bytes()).hexdigest()
            receipt['oracle_transport_source_sha256'] = hashlib.sha256(oracle.read_bytes()).hexdigest()
            receipt['rust_compiler'] = run(['rustc', '--version'], capture_output=True).stdout.strip()
            receipt['rust_flags'] = ['--edition=2024', '--crate-type', 'cdylib', '-O']
        receipt['probes'] = {}
        if case_columns.is_file():
            receipt['case_columns_source_sha256'] = hashlib.sha256(case_columns.read_bytes()).hexdigest()
        for name, path in [('probe_cloudiness', probe), ('probe_tables', table_probe), ('probe_constants', constants_probe),
                           ('probe_surface', surface_probe), ('probe_cloudiness_capture', probe)]:
            executable = work / name
            if (name == 'probe_cloudiness_capture' and a.tap_transport) or (name != 'probe_cloudiness_capture' and a.probe):
                receipt['probes'][name] = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                          'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest()}
        if a.tap_transport:
            receipt['tap_transport_source_sha256'] = hashlib.sha256(tap.read_bytes()).hexdigest()
    (work / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
