#!/usr/bin/env python3
"""Freeze Registry declarations and constant words from pinned MPAS sources.

This is a developer proof tool. Installed forecasts read its small output and
never fetch or compile MPAS. No table or scheme arithmetic is generated here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    args = ap.parse_args()
    build(args.source, args.output, args.scratch)


def build(source, output, scratch, *, manifest_name='mpas-source-v842.json',
          schema='mpas-framework-v842/v1', registries=None):
    """Freeze one pinned source tree's declarations and constant words.

    ``registries`` lists the Registry files to authenticate (default: every
    Registry*.xml under the source, as the v8.4.2 export carries only the
    atmosphere's). The fork tree is a whole checkout, so its tool names them.
    """
    args = argparse.Namespace(source=source, output=output, scratch=scratch)
    tree = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(tree / 'src'))
    from woof.hex.mpasphys.pool_v842 import registry_text
    spec = importlib.util.spec_from_file_location('native_generator', tree / 'tools/native_physics_reference/backend/generate.py')
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    manifest = json.loads((tree / 'src/hexcore/data/mpasphys-native' / manifest_name).read_text())
    physics = args.source / 'src/core_atmosphere/physics'
    paths = [*sorted(physics.glob('mpas_atmphys_*.F')), physics.parent / 'mpas_atm_dimensions.F']
    provenance = {}
    registries = list(args.source.rglob('Registry*.xml')) if registries is None else list(registries)
    for path in [*paths, *registries,
                 args.source / 'src/framework/mpas_constants.F', args.source / 'src/framework/mpas_kind_types.F']:
        relative = str(path.relative_to(args.source))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest['files'][relative]['sha256']:
            raise ValueError('source differs from pin: ' + relative)
        provenance[relative] = digest
    entries, variables, allocations = {}, {}, {}
    for path in paths:
        for name, entry in parser.routines(path).items():
            entry['source'] = str(path.relative_to(args.source))
            entries[name] = entry
        current = None
        for line, statement in parser.statements(path.read_text()):
            match = re.match(r'subroutine\s+(allocate_\w+)', statement, re.I)
            if match:
                current = match[1].lower()
                allocations[current] = {}
            elif re.match(r'end\s*subroutine', statement, re.I):
                current = None
            if current:
                for name, bound in re.findall(r'\ballocate\s*\(\s*(\w+)\s*\(([^()]*)\)\s*\)', statement, re.I):
                    allocations[current][name.lower()] = [part.strip().lower().split(':') if ':' in part
                        else ['1', part.strip().lower()] for part in parser.split_list(bound)]
    path = physics / 'mpas_atmphys_vars.F'
    for line, statement in parser.statements(parser.module_head(path.read_text())):
        if '::' not in statement:
            continue
        head, tail = statement.lower().split('::', 1)
        if 'parameter' in head or not head.startswith(('real', 'integer', 'logical')):
            continue
        match = re.search(r'dimension\((.*?)\)', head)
        rank = len(parser.split_list(match[1])) if match else 0
        for declaration in parser.split_list(tail):
            name = re.match(r'\s*(\w+)', declaration)[1]
            variables[name] = dict(kind='real' if head.startswith('real') else
                'integer' if head.startswith('integer') else 'logical', rank=rank,
                allocatable='allocatable' in head, line=line,
                source=str(path.relative_to(args.source)))
    allocations = {key: {name: bound for name, bound in rows.items() if name in variables}
                   for key, rows in allocations.items()}
    constant_names = {}
    for module, path in (('mpas_constants', args.source / 'src/framework/mpas_constants.F'),
                         ('mpas_atmphys_constants', physics / 'mpas_atmphys_constants.F')):
        for line, statement in parser.statements(parser.module_head(path.read_text())):
            if re.match(r'real.*parameter.*::', statement, re.I):
                name = statement.split('::')[1].split('=')[0].strip().lower()
                constant_names[name] = dict(module=module, line=line, source=str(path.relative_to(args.source)))
    args.scratch.mkdir(parents=True, exist_ok=True)
    program = ['program capture', 'use iso_fortran_env, only: int32']
    for name, row in constant_names.items():
        program += [f"use {row['module']}, only: constant_{name} => {name}"]
    program += ['implicit none']
    for name in constant_names:
        program += [f"write(*,'(A,1X,Z8.8)') '{name}', transfer(constant_{name},0_int32)"]
    program += ['end program']
    (args.scratch / 'capture.f90').write_text('\n'.join(program) + '\n')
    sources = [args.source / 'src/framework/mpas_kind_types.F',
               args.source / 'src/framework/mpas_constants.F', physics / 'mpas_atmphys_constants.F']
    command = ['gfortran', '-O3', '-ffp-contract=off', '-fno-tree-vectorize',
               '-fno-unsafe-math-optimizations', '-cpp', '-DSINGLE_PRECISION',
               '-ffree-form', '-ffree-line-length-none', '-o', 'capture',
               *(str(p.resolve()) for p in sources), 'capture.f90']
    subprocess.run(command, cwd=args.scratch, check=True)
    words = subprocess.check_output([str((args.scratch / 'capture').resolve())], text=True)
    for row in words.splitlines():
        name, word = row.split()
        constant_names[name]['word'] = int(word, 16)
    # The native ABI exposes the imported dry-air constant using its local alias.
    constant_names['r_d'] = dict(constant_names['rgas'])
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'src/core_atmosphere/Registry.xml'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(registry_text(args.source / 'src/core_atmosphere/Registry.xml') + '\n')
    metadata = dict(schema=schema, source_pin=manifest['pins'],
                    entries=entries, variables=variables, allocations=allocations,
                    constants=constant_names, source_sha256=provenance,
                    constant_capture_command=command)
    # Do not ship absolute developer paths in the portable declaration artifact.
    metadata['constant_capture_command'] = [arg.replace(str(args.source.resolve()), '{source}') for arg in command]
    (args.output / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True, indent=1) + '\n')
    summary = dict(fields_source_bytes=target.stat().st_size,
                   variables=len(variables), entries=len(entries), constants=len(constant_names))
    print(json.dumps(summary))
    return metadata


if __name__ == '__main__':
    main()
