#!/usr/bin/env python3
"""Generate a typed bind(C) ABI from pinned MPAS's own declarations.

Generated calls cite their entire native routine. No physics arithmetic is emitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def split_list(text):
    result, start, depth, quote = [], 0, 0, None
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in '([':
            depth += 1
        elif char in ')]':
            depth -= 1
        elif char == ',' and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    result.append(text[start:].strip())
    return result


def write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def module_head(text):
    return re.split(r'^\s*contains\s*$', text, maxsplit=1, flags=re.I | re.M)[0]


def statements(text):
    joined, first = '', 0
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.split('!')[0].strip()
        if not line or line.startswith('#'):
            continue
        if not joined:
            first = lineno
        joined += ' ' + line.lstrip('&').rstrip('&').strip()
        if not line.endswith('&'):
            yield first, joined.strip()
            joined = ''


def routines(path):
    stmts = list(statements(path.read_text()))
    module = next(re.match(r'module\s+(\w+)', s, re.I)[1] for _, s in stmts
                  if re.match(r'module\s+(?!procedure)(\w+)', s, re.I))
    result = {}
    for i, (line, s) in enumerate(stmts):
        m = re.match(r'subroutine\s+(\w+)\s*(?:\((.*?)\))?$', s, re.I)
        if not m:
            continue
        name = m[1].lower()
        args = [x.strip().lower() for x in (m[2] or '').split(',') if x.strip()]
        types = {}
        for endline, decl in stmts[i + 1:]:
            if re.match(r'end\s*subroutine', decl, re.I):
                break
            if '::' not in decl:
                continue
            head, tail = decl.lower().split('::', 1)
            for var in split_list(tail):
                varname = re.match(r'\s*(\w+)', var)
                if varname and varname[1] in args:
                    types[varname[1]] = head.strip()
        result[name] = {'module': module.lower(), 'arguments': args,
                        'types': types, 'line': line, 'end_line': endline}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    # build.py --licence-stub noahmp: Noah-MP is not in the build, so its IO type has nothing to index and its
    # directories have no Makefile to read.
    ap.add_argument('--no-noahmp-io', action='store_true')
    ap.add_argument('--held', action='append', default=[], help='a physics directory a licence stub replaces')
    ap.add_argument('--pin-set', default='rrfs')
    ap.add_argument('--source-pin', default='1d6e370e7b43a332074809f170532a216c2212fb')
    a = ap.parse_args()
    import sources
    sources.HELD.update(a.held)
    p = a.source / 'src/core_atmosphere/physics'
    output = a.output
    output.mkdir(parents=True, exist_ok=True)
    entries = {}
    for path in sorted([*p.glob('mpas_atmphys_*.F'), p.parent / 'mpas_atm_dimensions.F']):
        for name, entry in routines(path).items():
            if (name.startswith(('driver_', 'allocate_', 'deallocate_', 'update_', 'init_')) or
                    name in {'physics_driver', 'ruc_init_formpas', 'physics_init', 'atmphys_finalize', 'mpas_to_physics',
                             'pre_microphysics', 'post_microphysics', 'physics_get_tend',
                             'pre_physics_get_tend', 'post_physics_get_tend', 'mpas_atm_set_dims',
                             'microphysics_from_mpas', 'microphysics_to_mpas',
                             'physics_tables_init', 'physics_registry_init', 'physics_run_init', 'physics_timetracker',
                             'physics_update', 'physics_update_surface', 'physics_update_sst',
                             'physics_update_sstskin', 'physics_update_deepsoiltemp',
                             'atmphys_sfc_diagnostics', 'o3climatology_from_mpas',
                             'rrtmg_initlw_formpas', 'rrtmg_initsw_formpas'}):
                # Only public routines: private conversion helpers remain inside driver_X.
                top = module_head(path.read_text())
                publics = ' '.join(s for _, s in statements(top) if s.lower().startswith('public'))
                if name != 'mpas_atm_set_dims' and re.search(r'\b' + name + r'\b', publics, re.I) is None:
                    continue
                if any('optional' in t for t in entry['types'].values()):
                    continue
                if any('dimension' in t and not (t.startswith('real') and 'dimension(:,:)' in t)
                       for t in entry['types'].values()):
                    continue
                if any(not re.match(r'type\s*\(', t) and 'dimension' not in t and re.search(r'intent\((out|inout)\)', t)
                       for t in entry['types'].values()):
                    raise ValueError(f'scalar output requires a reference ABI: {name}')
                if len(entry['types']) != len(entry['arguments']):
                    raise ValueError(f'unparsed argument in {name}: {entry}')
                entry['source'] = str(path.relative_to(a.source))
                entry['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                if 'exchange_halo_group' in entry['arguments']:
                    entry['source_arguments'] = entry['arguments'].copy()
                    entry['arguments'].remove('exchange_halo_group')
                    entry['native_bound_arguments'] = {'exchange_halo_group': 'mpas_dmpar_exch_group_full_halo_exch'}
                entries[name] = entry
    uses = ['use iso_c_binding', 'use mpas_kind_types', 'use mpas_pool_routines',
            'use mpas_atmphys_vars', 'use mpas_timekeeping', 'use mpas_stream_manager',
            'use mpas_dmpar, only: mpas_dmpar_exch_group_full_halo_exch']
    for module in sorted(set(e['module'] for e in entries.values())):
        names = [n for n, e in entries.items() if e['module'] == module]
        uses.append('use ' + module + ', only: ' + ','.join(names))
    f = [f'! Generated from {a.pin_set} {a.source_pin}. Do not hand edit.', 'module hex_native_wrappers',
         *uses, 'implicit none',
         'type,bind(C) :: native_array_real_2', ' type(c_ptr) :: data',
         ' integer(c_int) :: shape(2)', 'end type', 'contains', '''
function native_abi_version() result(v) bind(C)
 integer(c_int) :: v
 v=2
end function
function native_pool_create() result(handle) bind(C)
 type(c_ptr) :: handle
 type(mpas_pool_type),pointer :: p
 call mpas_pool_create_pool(p)
 handle=c_loc(p)
end function
subroutine native_pool_destroy(handle) bind(C)
 type(c_ptr),value :: handle
 type(mpas_pool_type),pointer :: p
 call c_f_pointer(handle,p)
 call mpas_pool_destroy_pool(p)
end subroutine
function c_string(s) result(v)
 character(c_char),intent(in) :: s(*)
 character(len=StrKIND) :: v
 integer :: i
 v=''
 do i=1,StrKIND
   if(s(i)==c_null_char) exit
   v(i:i)=s(i)
 enddo
end function
subroutine native_dimension(handle,key,count,data,create) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: key(*)
 integer(c_int),value :: count,create
 integer(c_int),intent(in) :: data(*)
 type(mpas_pool_type),pointer :: p
 integer,pointer :: scalar, vector(:)
 call c_f_pointer(handle,p)
 if(count==0) then
   if(create/=0) call mpas_pool_add_dimension(p,trim(c_string(key)),data(1))
   call mpas_pool_get_dimension(p,trim(c_string(key)),scalar)
   scalar=data(1)
 else
   if(create/=0) call mpas_pool_add_dimension(p,trim(c_string(key)),data(1:count))
   call mpas_pool_get_dimension(p,trim(c_string(key)),vector)
   vector=data(1:count)
 endif
end subroutine
subroutine native_field_character_0(handle,key,data,operation) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: key(*)
 character(c_char),intent(inout) :: data(*)
 integer(c_int),value :: operation
 type(mpas_pool_type),pointer :: p
 type(field0DChar),pointer :: field
 integer :: i
 call c_f_pointer(handle,p)
 if(operation==0) then
  allocate(field)
  field%fieldName=c_string(key)
  field%isDecomposed=.false.
  field%hasTimeDimension=.false.
  field%isActive=.true.
  field%isVarArray=.false.
  field%defaultValue=''
  call mpas_pool_add_field(p,trim(c_string(key)),field)
 endif
 call mpas_pool_get_field(p,trim(c_string(key)),field)
 if(operation==2) then
  do i=1,StrKIND
   data(i)=field%scalar(i:i)
  enddo
 else
  field%scalar=c_string(data)
 endif
end subroutine
! MPAS source: MPAS-Model-rrfs/src/framework/mpas_pool_routines.F:2548-2590,4056-4116.
subroutine native_field_character_levels_0(handle,key,nlevels,level,data,operation) bind(C)
 type(c_ptr),value :: handle
 character(c_char),intent(in) :: key(*)
 character(c_char),intent(inout) :: data(*)
 integer(c_int),value :: nlevels,level,operation
 type(mpas_pool_type),pointer :: p
 type(field0DChar),pointer :: fields(:),field
 integer :: i
 call c_f_pointer(handle,p)
 if(operation==0) then
  allocate(fields(nlevels))
  do i=1,nlevels
   fields(i)%fieldName=c_string(key)
   fields(i)%isDecomposed=.false.
   fields(i)%hasTimeDimension=.true.
   fields(i)%isActive=.true.
   fields(i)%isVarArray=.false.
   fields(i)%defaultValue=''
  enddo
  call mpas_pool_add_field(p,trim(c_string(key)),fields)
 endif
 call mpas_pool_get_field(p,trim(c_string(key)),field,timeLevel=level)
 if(operation==2) then
  do i=1,StrKIND
   data(i)=field%scalar(i:i)
  enddo
 else
  field%scalar=c_string(data)
 endif
end subroutine
''']
    for kind, decl in [('real', 'real(c_float)'), ('integer', 'integer(c_int)'),
                       ('logical', 'logical(c_bool)'), ('character', 'character(c_char)')]:
        f += [f'subroutine native_config_{kind}(handle,key,value,create) bind(C)',
              'type(c_ptr),value :: handle', 'character(c_char),intent(in) :: key(*)',
              ('character(c_char),intent(in) :: value(*)' if kind == 'character' else decl + ',value :: value'),
              'integer(c_int),value :: create', 'type(mpas_pool_type),pointer :: p',
              ('character(len=StrKIND),pointer :: v' if kind == 'character' else
               ('logical,pointer :: v' if kind == 'logical' else decl + ',pointer :: v')),
              'call c_f_pointer(handle,p)']
        val = 'c_string(value)' if kind == 'character' else 'logical(value)' if kind == 'logical' else 'value'
        f += [f'if(create/=0) call mpas_pool_add_config(p,trim(c_string(key)),{val})',
              'call mpas_pool_get_config(p,trim(c_string(key)),v)', f'v={val}', 'end subroutine']
    # MPAS source: MPAS-Model-rrfs/src/framework/mpas_pool_routines.F:4761-4791.
    f += ['subroutine native_config_get_real(handle,key,value) bind(C)',
          'type(c_ptr),value :: handle', 'character(c_char),intent(in) :: key(*)',
          'real(c_float),intent(out) :: value', 'type(mpas_pool_type),pointer :: p',
          'real(RKIND),pointer :: v', 'call c_f_pointer(handle,p)',
          'call mpas_pool_get_config(p,trim(c_string(key)),v)', 'value=v', 'end subroutine']
    for kind, decl, native_type, limit in [('real', 'real(c_float)', 'Real', 5),
                                         ('integer', 'integer(c_int)', 'Integer', 3)]:
        for rank in range(limit + 1):
            shape = ','.join(':' for _ in range(rank))
            dims = ','.join(f'dims({i + 1})' for i in range(rank))
            typ = f'field{rank}D{native_type}'
            f += [f'subroutine native_field_{kind}_{rank}(handle,key,dims,nlevels,level,data,operation) bind(C)',
                  'type(c_ptr),value :: handle,data', 'character(c_char),intent(in) :: key(*)',
                  'integer(c_int),intent(in) :: dims(*)',
                  'integer(c_int),value :: nlevels,level,operation',
                  'type(mpas_pool_type),pointer :: p', f'type({typ}),pointer :: fields(:),single',
                  decl + ',pointer :: buffer' + (f'({shape})' if rank else ''),
                  'integer :: i', 'call c_f_pointer(handle,p)',
                  'call c_f_pointer(data,buffer' + (f',dims(1:{rank})' if rank else '') + ')',
                  'if(operation==0) then',
                  ' if(nlevels==1) then', '  allocate(single)',
                  ' else', '  allocate(fields(nlevels))', ' endif',
                  ' do i=1,nlevels', '  if(nlevels/=1) single=>fields(i)',
                  '  single%fieldName=c_string(key)', '  single%isDecomposed=.false.',
                  '  single%hasTimeDimension=nlevels>1', '  single%isActive=.true.',
                  '  single%isVarArray=.false.', '  single%defaultValue=0', '  single%missingValue=0']
            if rank:
                f += [f'  single%dimSizes=dims(1:{rank})', "  single%dimNames=''",
                      '  single%isPersistent=.true.', f'  allocate(single%array({dims}))', '  single%array=0']
            else:
                f += ['  single%scalar=0']
            f += [' enddo', ' if(nlevels==1) then',
                  '  call mpas_pool_add_field(p,trim(c_string(key)),single)', ' else',
                  '  call mpas_pool_add_field(p,trim(c_string(key)),fields)', ' endif', 'endif',
                  'call mpas_pool_get_field(p,trim(c_string(key)),single,timeLevel=level)']
            ref = 'single%array' if rank else 'single%scalar'
            f += ['if(operation==2) then', f' buffer={ref}', 'else', f' {ref}=buffer', 'endif', 'end subroutine']
    for name, e in entries.items():
        args = e['arguments']
        f += [f"! MPAS source: MPAS-Model-rrfs/{e['source']}:{e['line']}-{e['end_line']}",
              f"subroutine native_{name}({','.join(args)}) bind(C)"]
        callargs = []
        e['abi'] = []
        for arg in args:
            t = e['types'][arg]
            if 'dimension(:,:)' in t:
                f += [f'type(native_array_real_2),value :: {arg}', f'real(c_float),pointer :: f_{arg}(:,:)']
                e['abi'].append('real_array2')
                callargs.append('f_' + arg)
            elif re.match(r'type\s*\(', t):
                typ = re.match(r'type\s*\(([^)]+)\)', t)[1]
                f += [f'type(c_ptr),value :: {arg}', f'type({typ}),pointer :: f_{arg}']
                e['abi'].append('handle')
                callargs.append('f_' + arg)
            elif t.startswith('integer'):
                f += [f'integer(c_int),value :: {arg}']
                e['abi'].append('int32')
                callargs.append(arg)
            elif t.startswith('real'):
                f += [f'real(c_float),value :: {arg}']
                e['abi'].append('float32')
                callargs.append(arg)
            elif t.startswith('character'):
                f += [f'character(c_char),intent(in) :: {arg}(*)']
                e['abi'].append('string')
                # Preserve StrKIND padding, as in the v8.4.2 wrapper. The
                # surface date is len=* but monthly_interp_to_date reads
                # fixed-width columns 21:24 (mpas_atmphys_date_time.F).
                # Trimming makes that downstream read exceed its storage.
                callargs.append('c_string(' + arg + ')')
            elif t.startswith('logical'):
                f += [f'logical(c_bool),value :: {arg}']
                e['abi'].append('bool')
                callargs.append('logical(' + arg + ')')
            else:
                raise ValueError(f'unsupported type {name}:{arg}: {t}')
        for arg, abi in zip(args, e['abi']):
            if abi == 'handle':
                f += [f'call c_f_pointer({arg},f_{arg})']
            elif abi == 'real_array2':
                f += [f'call c_f_pointer({arg}%data,f_{arg},{arg}%shape)']
        if 'source_arguments' in e:
            callargs = [dict(zip(args, callargs)).get(arg, 'mpas_dmpar_exch_group_full_halo_exch')
                        for arg in e['source_arguments']]
        f += [f"call {name}({','.join(callargs)})", 'end subroutine']
    # Expose all primitive mutable module variables for level-b replay, preserving lower bounds.
    variables = {}
    path = p / 'mpas_atmphys_vars.F'
    for line, s in statements(module_head(path.read_text())):
        if '::' not in s:
            continue
        head, tail = s.split('::', 1)
        head = head.lower()
        if 'parameter' in head or not head.startswith(('real', 'integer', 'logical')):
            continue
        m = re.search(r'dimension\((.*?)\)', head)
        rank = len(split_list(m[1])) if m else 0
        kind = 'real' if head.startswith('real') else 'integer' if head.startswith('integer') else 'logical'
        for v in split_list(tail):
            name = re.match(r'\s*(\w+)', v)[1].lower()
            variables[name] = {'kind': kind, 'rank': rank, 'allocatable': 'allocatable' in head,
                               'line': line, 'source': str(path.relative_to(a.source))}
    for kind, decl in [('real', 'real(c_float)'), ('integer', 'integer(c_int)'), ('logical', 'integer(c_int)')]:
        ranks = sorted(set(v['rank'] for v in variables.values() if v['kind'] == kind))
        for rank in ranks:
            shape = ','.join(':' for _ in range(rank))
            f += [f'integer(c_int) function native_vars_{kind}_{rank}(key,dims,lower,data,operation) result(status) bind(C)',
                  'character(c_char),intent(in) :: key(*)', 'type(c_ptr),value :: data',
                  'integer(c_int),intent(inout) :: dims(*),lower(*)', 'integer(c_int),value :: operation',
                  decl + ',pointer :: buffer' + (f'({shape})' if rank else ''),
                  'status=0', 'select case(trim(c_string(key)))']
            for name, v in variables.items():
                if v['kind'] != kind or v['rank'] != rank:
                    continue
                f += [f"! MPAS source: MPAS-Model-rrfs/{v['source']}:{v['line']}", f"case('{name}')"]
                if rank:
                    if not v['allocatable']:
                        raise ValueError(f'nonalloc array {name}')
                    f += [' if(operation==3) then', f'  if(.not.allocated({name})) then',
                          '   status=1', '   return', '  endif', f'  dims(1:{rank})=shape({name})',
                          f'  lower(1:{rank})=lbound({name})', '  return', ' endif',
                          f' call c_f_pointer(data,buffer,dims(1:{rank}))', ' if(operation==0) then',
                          f'  if(allocated({name})) deallocate({name})']
                    bounds = ','.join(f'lower({i}):lower({i})+dims({i})-1' for i in range(1, rank + 1))
                    f += [f'  allocate({name}({bounds}))', ' endif', f' if(.not.allocated({name})) then',
                          '  status=1', '  return', ' endif', f' if(any(shape({name})/=dims(1:{rank}))) then',
                          '  status=2', '  return', ' endif']
                else:
                    f += [' if(operation==3) return', ' call c_f_pointer(data,buffer)']
                get = f'merge(1,0,{name})' if kind == 'logical' else name
                val = 'buffer/=0' if kind == 'logical' else 'buffer'
                f += [' if(operation==2) then', f'  buffer={get}', ' else', f'  {name}={val}', ' endif']
            f += ['case default', ' status=-1', 'end select', 'end function']
    f += ['subroutine native_variables_release() bind(C)']
    for name, v in variables.items():
        if v['allocatable']:
            f += [f'if(allocated({name})) deallocate({name})']
    f += ['end subroutine', 'end module']
    from tables import generate_tables
    tables, constants = generate_tables(a.source, output)
    from noahmp_io import generate_noahmp_io
    noahmp_io = {} if a.no_noahmp_io else generate_noahmp_io(a.source, output)
    for file, text in [('native_wrappers.f90', '\n'.join(f) + '\n'),
                       ('abi.json', json.dumps({'version': 2, 'pin_set': a.pin_set, 'source_pin': a.source_pin, 'entries': entries, 'variables': variables,
                                               'tables': tables, 'constants': constants, 'noahmp_io': noahmp_io}, indent=2) + '\n')]:
        destination = output / file
        if not destination.exists() or destination.read_text() != text:
            destination.write_text(text)


if __name__ == '__main__':
    main()
