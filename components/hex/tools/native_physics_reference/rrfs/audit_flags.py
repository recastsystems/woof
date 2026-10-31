#!/usr/bin/env python3
"""Extend f0's audit to all fork objects with unambiguous cwd/compiler receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_flags import STRICT
from rrfs.diagnostic_flags import checked_flags


def audit(root, recipe, core):
    pins_file = root / 'SOURCE-PINS.json'
    diagnostic = json.loads(pins_file.read_text()).get('diagnostic') if pins_file.exists() else None
    diagnostic_flags = checked_flags(diagnostic) if diagnostic else ()
    records = [json.loads(line) for line in (root / 'compile-commands.jsonl').read_text().splitlines()]
    commands = {}
    for record in records:
        args = record['argv']
        sources = [a for a in args if Path(a).suffix.lower() in ('.f', '.f90', '.c', '.cc', '.cpp')]
        if len(sources) != 1:
            continue
        output = args[args.index('-o') + 1] if '-o' in args else Path(sources[0]).stem + '.o'
        obj = (Path(record['cwd']) / output).resolve()
        commands.setdefault(obj, []).append(record)
    paths = sorted(root.rglob('*.o'))
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    hashes = {p: sha(p) for p in paths}
    objects, errors = [], []
    for obj in paths:
        history = commands.get(obj.resolve(), [])
        origins = []
        # core_atmosphere/Makefile:45 extracts already compiled objects from
        # libphys.a; the secondary core also extracts its MUSICA-off chemistry
        # stub archive. Prove these copies by basename AND exact bytes,
        # while retaining the original cwd/compiler command. Other uncaptured
        # objects still fail; a shared source stem alone is never evidence.
        archive_sources = {
            root / 'src/core_atmosphere/libphys': root / 'src/core_atmosphere/physics',
            root / 'src/core_atmosphere/libchem': root / 'src/core_atmosphere/chemistry',
        }
        if not history and obj.parent in archive_sources:
            origins = [p for p in paths if p.name == obj.name and hashes[p] == hashes[obj]
                       and p.is_relative_to(archive_sources[obj.parent])
                       and commands.get(p.resolve())]
            history = [commands[p.resolve()][-1] for p in origins]
        # An incremental rebuild replaces the object. The last completed
        # compiler invocation is its provenance, even after an earlier failure.
        matched = history if origins else history[-1:]
        required = ('-O3',) + (STRICT if recipe == 'gnu-strict' else ())
        forbidden = ('-fdefault-real-8', '-fdefault-double-8', '-ffast-math', '-Ofast', '-fopenmp')
        if recipe == 'gnu-strict':
            forbidden += ('-ffp-contract=fast', '-ffp-contract=on', '-ftree-vectorize', '-funsafe-math-optimizations')
        good = bool(matched)
        for command in matched:
            args = command['argv']
            fortran = command['compiler'] in ('mpif90', 'gfortran')
            good &= command['status'] == 0 and all(f in args for f in required)
            good &= not any(f in args for f in forbidden)
            good &= not fortran or '-DSINGLE_PRECISION' in args
            if diagnostic and fortran:
                good &= recipe == 'gnu-strict' and all(flag in args for flag in diagnostic_flags)
                good &= [flag for flag in args if flag.startswith('-fcheck=')][-1:] == ['-fcheck=all']
                good &= [flag for flag in args if flag.startswith('-finit-real=')][-1:] == ['-finit-real=snan']
                for flag in diagnostic_flags:
                    if flag.startswith('-finit-integer='):
                        good &= [a for a in args if a.startswith('-finit-integer=')][-1:] == [flag]
        row = dict(object=str(obj.relative_to(root)), flags_ok=bool(good), commands=matched,
                   earlier_compile_records=0 if origins else max(0, len(history)-1),
                   sha256=hashes[obj], archive_copy_origins=[str(p.relative_to(root)) for p in origins])
        objects.append(row)
        if not good:
            errors.append(row['object'])
    groups = {}
    if core == 'atmosphere':
        names = ['physics_mmm', 'physics_wrf', 'physics_noahmp']
        names += ['physics_noaa/' + n for n in ('TEMPO', 'MYNN-EDMF', 'MYNN-SFC', 'RUCLSM', 'GFL', 'SMOKE', 'UGWP')]
        for name in names:
            groups[name] = sum('/' + name + '/' in row['object'] for row in objects)
            if not groups[name]:
                errors.append('missing group ' + name)
    if not objects:
        errors.append('no objects')
    return dict(schema='mpas.rrfs-compiler-audit/v1', recipe=recipe, core=core,
                diagnostic=diagnostic,
                all_ok=not errors, errors=errors, groups=groups, objects=objects)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--recipe', choices=('gnu-strict', 'gnu-o3'), required=True)
    p.add_argument('--core', choices=('atmosphere', 'init_atmosphere'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root.resolve(), a.recipe, a.core)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('all_ok', 'errors', 'groups')}))
    sys.exit(0 if result['all_ok'] else 1)
