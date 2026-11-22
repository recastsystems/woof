#!/usr/bin/env python3
"""Write the licence stub interface that stands in for Noah-MP v5.0 in the default native library build.

THE BREAKAGE THIS PREVENTS.  MPAS-A v8.4.2 USEs Noah-MP v5.0 (physics_noahmp; UCAR terms: "You shall not
sell ... the Software, or any work that in any manner contains the Software") from its own sources
(mpas_atmphys_vars.F:12 and the Noah-MP driver, init and finalize files).  So every native framework library
compiled from the full tree contains it, even for a run that never selects it: the default ``woof hex
physics-resources build`` put code that cannot be sold into every install.  build.py --licence-stub noahmp
compiles this stub instead, unless the user opts in by name (``--licence-opt-in ucar-noahmp``).  UGWP and the
other NOAA-GSL physics are compiled as always (US Government code, not held); a
component under another restricted licence would be one more COMPONENTS row.

What is written.  Only interface facts that MPAS's own sources need to compile: the module names they USE, the
routine names they call with each dummy argument's name, type, kind, rank, intent and optional attribute, and
the NoahmpIO_type components they reference with each component's type, kind and rank.  No executable upstream
statement, no upstream comment and no default value is copied.  Every stub routine stops the run, naming the
opt-in that builds the real code; none returns.

The interface is derived, not hand-listed: every ``use`` of a Noah-MP module in an MPAS source the
library compiles (outside the replaced directories) is followed to the names it imports or calls, and the
build fails by name (``missing module``) if a stub ever lacks one.

Usage (lab; the stubs are committed beside build.py and shipped with it):
    python make_licence_stubs.py --source MPAS-Model-v8.4.2 --out tools/native_physics_reference/backend
    python make_licence_stubs.py --source MPAS-Model-rrfs --out tools/native_physics_reference/backend/rrfs \
        --compiled-only rrfs --model "the RRFS fork of MPAS-Model (1d6e370e)"

THE BREAKAGE --compiled-only PREVENTS (RRFS-MPAS from a pip install, 2026-10-15).  The RRFS fork's tree carries
non-MPAS copies of its NOAA physics (CCPP drivers, TEMPO's docs, test harnesses) that USE a module named
``machine``, which Noah-MP also defines; scanning every file stopped the generator ("machine: unhandled imported
name kind_dbl_prec") although no file the fork library compiles imports Noah-MP's ``machine``.  With
``--compiled-only rrfs`` the users are exactly the files the fork builder compiles (rrfs/sources.py), the same
rule build.py's module closure applies.  The v8.4.2 stub is generated as before.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess

PHYSICS = 'src/core_atmosphere/physics'
# build.py's macros, so the preprocessed interfaces are the ones the library build sees.
MACROS = ['-DSINGLE_PRECISION', '-Dmpas', '-D_MPI', '-DMPAS_USE_MPI_F08', '-DCORE_ATMOSPHERE',
          '-DMPAS_NAMELIST_SUFFIX=atmosphere', '-DMPAS_SMIOL_SUPPORT']
COMPONENTS = {
    'noahmp': {'terms': 'ucar-noahmp',
               'what': 'Noah-MP v5.0 (UCAR terms: no sale of the Software or of any work that contains it)',
               'directories': ['physics_noahmp']},
}
# The object that holds a component's NoahmpIO-style derived state, and the type behind it.
DERIVED_STATE = {'noahmp': ('mpas_noahmp', 'NoahmpIOVarType', 'NoahmpIO_type')}


def statements(text):
    joined = ''
    for line in text.splitlines():
        line = line.split('!')[0].rstrip()
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        joined += ' ' + line.strip().lstrip('&').rstrip('&').strip()
        if not line.endswith('&'):
            yield joined.strip()
            joined = ''


def split_list(text):
    out, depth, current = [], 0, ''
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        if char == ',' and depth == 0:
            out.append(current.strip())
            current = ''
        else:
            current += char
    if current.strip():
        out.append(current.strip())
    return out


INCLUDES: list[str] = []


def preprocess(path):
    # build.py's per-file macro: TEMPO compiles with -Dtempo_mpas (the RRFS fork; v8.4.2 has no TEMPO).
    extra = ['-Dtempo_mpas'] if 'TEMPO' in Path(path).parts else []
    return subprocess.run(['cpp', '-P', '-traditional', *MACROS, *extra, '-I' + str(Path(path).parent), *INCLUDES,
                           str(path)], check=True, capture_output=True, text=True).stdout


def modules_in(path):
    return [m.group(1) for m in re.finditer(r'^\s*module\s+(?!procedure\b)(\w+)', preprocess(path), re.I | re.M)]


def held_modules(source, component):
    """{lowercase module name: (declared name, file)} for every module the component's directories define."""
    found = {}
    for directory in COMPONENTS[component]['directories']:
        for path in sorted((source / PHYSICS / directory).rglob('*')):
            if path.suffix in ('.F', '.F90', '.f90'):
                for name in modules_in(path):
                    found[name.lower()] = (name, path)
    return found


COMPILED_ONLY: list[Path] = []


def mpas_users(source):
    """Every MPAS physics Fortran source (the only USErs of these modules) outside the held directories."""
    held = [source / PHYSICS / d for c in COMPONENTS.values() for d in c['directories']]
    if COMPILED_ONLY:
        yield from (path for path in COMPILED_ONLY if not any(h in path.parents for h in held))
        return
    for path in sorted((source / PHYSICS).rglob('*')):
        if path.suffix in ('.F', '.F90') and not any(h in path.parents for h in held):
            yield path


def interface(source, component):
    """{module: {'routines': [...], 'types': [...]}}: what MPAS's own sources import from the component."""
    held = held_modules(source, component)
    wanted = {}
    for path in mpas_users(source):
        text = preprocess(path)
        for match in re.finditer(r'^\s*use\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)\s*(,\s*only\s*:\s*([^\n]*))?$', text,
                                 re.I | re.M):
            module = match.group(1).lower()
            if module not in held:
                continue
            row = wanted.setdefault(module, {'names': set(), 'users': set()})
            row['users'].add(str(path.relative_to(source)))
            if match.group(3):
                for item in split_list(match.group(3)):
                    row['names'].add(item.split('=>')[-1].strip().lower())
            else:
                # No only-list: the names this file calls or declares from the module's public routines and types.
                module_text = preprocess(held[module][1])
                public = set(n.lower() for n in re.findall(r'^\s*(?:subroutine|function)\s+(\w+)', module_text,
                                                          re.I | re.M))
                public |= set(n.lower() for n in re.findall(r'^\s*type\s*,\s*public\s*::\s*(\w+)', module_text,
                                                           re.I | re.M))
                used = set(n.lower() for n in re.findall(r'\b(\w+)\b', text))
                row['names'] |= public & used
    return held, wanted


def declarations(text, start, stop):
    """Declaration statements (head, entity) between two statement indexes."""
    rows = []
    for statement in text[start:stop]:
        if '::' not in statement:
            continue
        head, tail = statement.split('::', 1)
        for entity in split_list(tail):
            name = re.match(r'\s*(\w+)', entity).group(1).lower()
            dims = re.match(r'\s*\w+\s*(\([^=]*\))?', entity).group(1)
            init = entity.split('=', 1)[1].strip() if '=' in entity and '=>' not in entity else None
            rows.append((head.strip(), name, dims, init))
    return rows


def attributes(head):
    """Split a declaration head into the type spec and its attributes (kept in order)."""
    parts = split_list(head)
    return parts[0].strip(), [p.strip() for p in parts[1:]]


def normal(head, dims, *, keep_parameter=False):
    base, attrs = attributes(head)
    base = re.sub(r'\s+', '', base).lower().replace('kind_noahmp', 'RKIND')
    kept = []
    for attribute in attrs:
        lowered = attribute.lower().replace(' ', '')
        if lowered in ('save', 'public', 'private') or (lowered == 'parameter' and not keep_parameter):
            continue
        if dims and lowered.startswith('dimension'):
            continue
        kept.append(re.sub(r'\s+', '', attribute))
    if dims:
        kept.append('dimension' + re.sub(r'\s+', '', dims))
    return ', '.join([re.sub(r'\s+', '', base)] + kept)


def routine_interface(module_text, name):
    stmts = list(statements(module_text))
    for index, statement in enumerate(stmts):
        match = re.match(r'(?:\w+\s+)*subroutine\s+(\w+)\s*(?:\((.*)\))?\s*$', statement, re.I)
        if match and match.group(1).lower() == name:
            break
    else:
        raise SystemExit(f'routine {name} not found')
    declared = match.group(1)
    arguments = [a.strip() for a in (match.group(2) or '').split(',') if a.strip()]
    end = next(i for i in range(index + 1, len(stmts)) if re.match(r'end\s*subroutine', stmts[i], re.I))
    rows = declarations(stmts, index + 1, end)
    by_name = {}
    for head, entity, dims, init in rows:
        by_name.setdefault(entity, (head, dims, init))
    lines, needed = [], set()
    for argument in arguments:
        head, dims, _ = by_name[argument.lower()]
        expression = (dims or '') + (re.search(r'dimension\s*(\(.*\))', head, re.I) or [None, ''])[1]
        needed |= set(n.lower() for n in re.findall(r'\b([a-z_]\w*)\b', expression, re.I))
    parameters = []
    for entity in sorted(needed - {a.lower() for a in arguments}):
        if entity in by_name and 'parameter' in by_name[entity][0].lower():
            head, dims, init = by_name[entity]
            parameters.append(f'{normal(head, dims, keep_parameter=True)} :: {entity} = {init}')
    scalars = [a for a in arguments if not (by_name[a.lower()][1] or 'dimension' in by_name[a.lower()][0].lower())]
    arrays = [a for a in arguments if a not in scalars]
    for argument in scalars + arrays:
        head, dims, _ = by_name[argument.lower()]
        lines.append(f'{normal(head, dims)} :: {argument}')
    return declared, arguments, parameters + lines


def type_interface(module_text, type_name, members):
    stmts = list(statements(module_text))
    start = next(i for i, s in enumerate(stmts) if re.match(rf'type\s*,\s*public\s*::\s*{type_name}\s*$', s, re.I))
    end = next(i for i in range(start + 1, len(stmts)) if re.match(r'end\s*type', stmts[i], re.I))
    rows = {entity: (head, dims) for head, entity, dims, _ in declarations(stmts, start + 1, end)}
    missing = sorted(set(members) - set(rows))
    if missing:
        raise SystemExit(f'{type_name} has no component(s) {", ".join(missing)}')
    return [f'{normal(*rows[m])} :: {m}' for m in sorted(members)]


def wrap(prefix, items, suffix, width=100):
    """``prefix item, item, ...suffix`` as free-form lines of at most ``width`` characters."""
    lines, current = [], prefix
    for index, item in enumerate(items):
        piece = item + (', ' if index < len(items) - 1 else '')
        if len(current) + len(piece) > width - 2 and current.strip() != prefix.strip():
            lines.append(current.rstrip() + ' &')
            current = '   '
        current += piece
    return lines + [current + suffix]


MODEL = {'tree': 'MPAS-Model v8.4.2', 'sources': 'MPAS-A v8.4.2'}


def write_component(source, component, out):
    info = COMPONENTS[component]
    held, wanted = interface(source, component)
    stop = f'hex_licence_stub_{component}'
    lines = [
        f'! woof hex licence stub: {info["what"]}.',
        f'! Generated by make_licence_stubs.py from the pinned {MODEL["tree"]} tree. Do not hand edit.',
        '! SPDX-License-Identifier: Apache-2.0',
        '!',
        '! The default `woof hex physics-resources build` compiles this file in place of',
        f'! src/core_atmosphere/physics/{" and ".join(info["directories"])}, so the native library contains none',
        f'! of that code. It carries only the interface that the {MODEL["sources"]} sources themselves USE (module,',
        '! routine and argument names, types, kinds and ranks, and the derived-type components those sources',
        '! reference), and every routine stops the run naming the opt-in:',
        f'!   woof hex physics-resources build --licence-opt-in {info["terms"]}',
        '',
        f'module {stop}',
        ' use iso_fortran_env, only: error_unit',
        ' implicit none',
        ' private',
        f' public :: {stop}_stop',
        ' contains',
        f' subroutine {stop}_stop(routine)',
        '  character(len=*), intent(in) :: routine',
        "  write(error_unit, '(a)') 'woof hex: '//routine//' was called, but this MPAS native library was built '// &",
        f"   'without {info['what']}. '// &",
        f"   'Rebuild it with: woof hex physics-resources build --licence-opt-in {info['terms']}'",
        '  flush(error_unit)',
        f"  error stop 'woof hex licence stub: rebuild with --licence-opt-in {info['terms']}'",
        f' end subroutine {stop}_stop',
        f'end module {stop}',
    ]
    derived = DERIVED_STATE.get(component)
    members = []
    if derived:
        pattern = re.compile(rf'\b{derived[0]}\s*%\s*(\w+)', re.I)
        for path in mpas_users(source):
            members += [m.lower() for m in pattern.findall(path.read_text(errors='replace'))]
    # Modules that define a type the others need come first.
    order = sorted(wanted, key=lambda m: (0 if derived and m == derived[1].lower() else 1, m))
    for module in order:
        declared, path = held[module]
        text = preprocess(path)
        names = wanted[module]['names']
        uses = sorted(wanted[module]['users'])
        lines += ['', '! Used by:'] + [f'!   {user}' for user in uses] + [f'module {declared}']
        routines = [n for n in sorted(names) if re.search(rf'^\s*(?:\w+\s+)*subroutine\s+{n}\b', text, re.I | re.M)]
        types = [n for n in sorted(names) if n not in routines]
        needs_kind = any('kind_phys' in line for r in routines for line in routine_interface(text, r)[2])
        if routines:
            lines.append(f' use {stop}, only: {stop}_stop')
        if needs_kind:
            lines.append(' use ccpp_kind_types, only: kind_phys')
        if derived and module == derived[1].lower():
            lines.append(' use mpas_kind_types, only: RKIND')
        elif derived and routines:
            lines.append(f' use {held[derived[1].lower()][0]}, only: {derived[2]}')
        lines += [' implicit none', ' private']
        public = [held_name for held_name in types] + routines
        for name in types:
            if derived and name == derived[2].lower():
                lines += [f' type, public :: {derived[2]}']
                lines += ['  ' + row for row in type_interface(text, derived[2], set(members))]
                lines += [f' end type {derived[2]}']
            else:
                raise SystemExit(f'{module}: unhandled imported name {name}')
        if routines:
            lines.append(' public :: ' + ', '.join(routine_interface(text, r)[0] for r in routines))
            lines.append(' contains')
            for routine in routines:
                declared_name, arguments, decls = routine_interface(text, routine)
                lines += wrap(f' subroutine {declared_name}(', arguments, ')')
                lines += ['  ' + d for d in decls]
                lines.append(f"  call {stop}_stop('{declared}::{declared_name}')")
                lines.append(f' end subroutine {declared_name}')
        lines.append(f'end module {declared}')
    target = out / f'licence_stub_{component}.F90'
    text = '\n'.join(lines) + '\n'
    target.write_text(text)
    return target, hashlib.sha256(text.encode()).hexdigest(), {m: sorted(wanted[m]['names']) for m in order}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compiled-only', choices=['rrfs'], default=None,
                        help="the users are only the files that pin set's builder compiles (rrfs/sources.py)")
    parser.add_argument('--model', default=None, help='how the stub header names the source tree')
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    if arguments.model:
        MODEL.update(tree=arguments.model, sources=arguments.model)
    if arguments.compiled_only == 'rrfs':
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().with_name('rrfs')))
        from sources import physics_sources
        COMPILED_ONLY[:] = [path for path in physics_sources(source) if path.suffix in ('.F', '.F90')]
    INCLUDES[:] = ['-I' + str(source / d) for d in ('src/core_atmosphere', PHYSICS, 'src/framework')]
    for component in COMPONENTS:
        path, digest, imported = write_component(arguments.source.resolve(), component, arguments.out)
        print(path, digest)
        for module, names in imported.items():
            print('  ', module, names if len(names) < 8 else f'{len(names)} names')


if __name__ == '__main__':
    main()
