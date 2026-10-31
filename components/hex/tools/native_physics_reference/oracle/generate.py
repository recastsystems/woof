#!/usr/bin/env python3
"""Inventory and generate an owned MPAS instrumentation overlay.

Extends native_dry_reference/match/instrument_native.py: typed raw streams,
argument names, indexToCellID context, original source hash and line spans.
Never edits the pinned source root. Unresolved sites are acceptance failures.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from fortran_source import routines, call, bind, first_executable
from derived import noahmp_components, primitive_components, has_presence
from pin_sets import pin_set, source_files, provenance


def module_actual(driver, expression, infos):
    """Resolve unsliced mutable globals while retaining native argument aliases.

    Local declarations shadow module storage. Respect USE ONLY and renames;
    ambiguous names fail rather than selecting a module by source scan order.
    """
    from fortran_source import split_top
    if not re.fullmatch(r'\s*[a-z]\w*\s*',expression,re.I):return None
    name=expression.strip().lower()
    if name in driver.variables:return None
    matches=[]
    own=infos.get(driver.module,{})
    for v in own.get('roots',own.get('variables',[])):
        if v['name']==name:matches.append((driver.module,v))
    if not matches:
        uses=own.get('use_statements',[])+[st.text for st in driver.body if re.match(r'use\b',st.text,re.I)]
        for statement in uses:
            use=re.match(r'use\s+(\w+)\s*(?:,\s*(.*))?$',statement,re.I)
            if not use:continue
            module=use[1].lower();tail=use[2] or '';remote=name
            only=re.match(r'only\s*:\s*(.*)',tail,re.I)
            entries=split_top(only[1] if only else tail) if tail else []
            bindings={e.split('=>')[0].strip().lower():e.split('=>')[-1].strip().lower() for e in entries}
            if only and name not in bindings:continue
            if name in bindings:remote=bindings[name]
            elif any('=>' in e and e.split('=>')[-1].strip().lower()==name for e in entries):continue
            for v in infos.get(module,{}).get('roots',infos.get(module,{}).get('variables',[])):
                if v['name']==remote:matches.append((module,v))
    unique={(m,v['name']):(m,v) for m,v in matches}
    if len(unique)>1:raise ValueError(f'ambiguous module actual {name}: {sorted(unique)}')
    if not unique:return None
    module,var=next(iter(unique.values()))
    return {'module':module,'name':var['name'],'attrs':var['attrs'],'type':var['type']}


def inventory(root, pin_set_name="mpas-v842", *, authenticate=True):
    config=pin_set(pin_set_name)
    wrappers={str(p.relative_to(root)) for glob in config["wrapper_globs"] for p in root.glob(glob)}
    from state import modules
    module_info=modules(root, pin_set_name)
    index, all_routines = {}, []
    for path in source_files(root, pin_set_name):
        if path.suffix.lower() not in {'.f', '.f90'} or 'musica' in str(path).lower():
            continue
        for routine in routines(path, root):
            all_routines.append(routine)
            index.setdefault(routine.name, []).append(routine)
    from fnmatch import fnmatch
    sites, unresolved = [], []
    embedded = json.loads((Path(__file__).parent/'embedded_interfaces.json').read_text())
    for driver in all_routines:
        fork_extra=any(fnmatch(driver.path,g) for g in config.get('extra_caller_globs',[]))
        ordinary=driver.name.startswith('driver_') or driver.path in wrappers
        if not (ordinary or fork_extra or driver.name in embedded) or '/physics/' not in driver.path:
            continue
        for st in driver.body:
            c = call(st)
            if not c:
                continue
            name, actuals = c
            extra=embedded.get(driver.name,{}).get(name)
            if not (ordinary or fork_extra) and extra is None:continue
            if (name.startswith(('mpas_', 'wrf_', 'pre_', 'post_')) and name!='mpas_smoke_driver') or '_from_mpas' in name or '_to_mpas' in name:
                continue
            candidates = index.get(name, [])
            if pin_set_name!='mpas-v842' and len(candidates)>1:
                # USE-associated names disambiguate fork and MMM implementations.
                imports=set(module_info.get(driver.module,{}).get('uses',[]))
                imports.update(re.match(r'use\s+(\w+)',st.text,re.I)[1].lower() for st in driver.body if re.match(r'use\s+(\w+)',st.text,re.I))
                associated=[r for r in candidates if r.module in imports]
                if len(associated)==1:candidates=associated
            if len(candidates) != 1:
                local = [x for x in candidates if x.path == driver.path]
                if len(local) == 1:
                    candidates = local
            site = {'scheme': extra.get('scheme',name) if extra else name, 'driver': driver.name, 'path': driver.path, 'line': st.start + 1,
                    'end_line': st.end, 'tap_level': 'wrapper' if driver.path in wrappers else 'driver', 'site': f'{name}__{driver.name}__L{st.start + 1}'}
            if not ordinary and not extra and not any(fnmatch(c.path,g) for c in candidates for g in config.get('extra_callee_path_globs',[])):continue
            if len(candidates) != 1:
                unresolved.append(dict(site, reason=f'callee resolution: {len(candidates)} candidates'))
                continue
            callee = candidates[0]
            try:
                bound = bind(callee, actuals)
            except ValueError as error:
                unresolved.append(dict(site, reason=str(error)))
                continue
            args, issues = [], []
            for dummy, expression in bound.items():
                var = callee.variables.get(dummy)
                try:alias=module_actual(driver,expression,module_info)
                except ValueError as error:issues.append(f'{dummy}: {error}');alias=None
                if var and re.fullmatch(r'type\s*\(\s*mpas_pool_type\s*\)',var.type,re.I):
                    args.append(dict(asdict(var),actual=expression,native_pool=True,module_actual=alias))
                    continue
                if var and re.fullmatch(r'type\s*\(\s*noahmpio_type\s*\)',var.type,re.I):
                    args.append(dict(asdict(var),actual=expression,derived_module='noahmpiovartype',
                                     components=noahmp_components(root,dummy,expression,var.intent),module_actual=alias))
                    continue
                derived=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',var.type,re.I) if var else None
                if derived and derived[1].lower() in config['derived_types']:
                    spec=config['derived_types'][derived[1].lower()]
                    caller=driver.variables.get(expression.lower())
                    guard=f'present({expression})' if caller and caller.optional else ''
                    args.append(dict(asdict(var),actual=expression,derived_module=spec['module'],presence_guard=guard,
                                     components=primitive_components(root,spec,derived[1],dummy,expression,var.intent,var.dims),
                                     entry_storage=var.intent=='out',module_actual=alias))
                    continue
                if not var or not var.supported:
                    issues.append(f'{dummy}: unsupported or missing declaration')
                    continue
                base=re.match(r'^\s*(\w+)(?:\s*\(|\s*$)',expression)
                caller_var=driver.variables.get(base[1].lower()) if base else None
                caller_attrs=caller_var.attrs if caller_var else ''
                if base and not caller_var:
                    name=base[1].lower();info=module_info.get(driver.module,{})
                    own=[v for v in info.get('variables',[]) if v['name']==name]
                    imported=[v for n in info.get('uses',[]) for v in module_info.get(n,{}).get('variables',[]) if v['name']==name]
                    candidates=own or imported
                    storage={('allocatable' in v['attrs'],'pointer' in v['attrs']) for v in candidates}
                    if len(storage)>1:issues.append(f'{dummy}: ambiguous imported actual storage for {name}')
                    if len(storage)==1:caller_attrs=candidates[0]['attrs']
                guard=''
                if 'allocatable' in caller_attrs:guard=f'allocated({base[1]})'
                if 'pointer' in caller_attrs:guard=f'associated({base[1]})'
                argument=dict(asdict(var),actual=expression,presence_guard=guard,
                              entry_storage=var.intent=='out',module_actual=alias)
                # The fork forwards absent optional actuals to required dummies
                # whose selected native branch never reads them. Preserve that
                # interface and NULL storage; observation must not dereference it.
                if caller_var and caller_var.optional:
                    argument.update(actual_optional=True,
                                    actual_presence_guard=f'present({base[1]})')
                args.append(argument)
            if extra:
                for argument in args:
                    axes = extra.get('argument_axes', {}).get(argument['name'])
                    if axes:
                        argument.update(cell_axis=axes[0], level_axis=axes[1])
            site.update(callee={'name': callee.name, 'path': callee.path, 'line': callee.start + 1, 'end_line': callee.end,
                                'module': callee.module}, arguments=args, issues=issues)
            site['omitted_optional_arguments']=[asdict(callee.variables[n]) for n in callee.args if n not in bound and n in callee.variables]
            sites.append(site)
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
              for p in sorted({x['path'] for x in sites + unresolved} | {x['callee']['path'] for x in sites})}
    return {'schema': 'mpas-oracle-inventory-v1', 'sites': sites, 'unresolved': unresolved,
            'scheme_aliases': {**json.loads((Path(__file__).parent/'scheme_aliases.json').read_text()),**config.get('scheme_aliases',{})},
            'scheme_groups': json.loads((Path(__file__).parent/'scheme_groups.json').read_text()),
            'source_sha256': hashes, 'argument_complete': not unresolved and all(not x['issues'] for x in sites),
            'hidden_state_complete': False, 'forecast_proven': False,
            'source_authenticated': authenticate,
            **(provenance(root,pin_set_name) if authenticate else {})}


def dump_call(site, arg, phase):
    if arg.get('native_pool') or arg.get('pool_argument'):
        label=('argument.' if arg.get('native_pool') else '')+arg['name']
        return f"      call oracle_pool_capture('{site['site']}','{phase}','{label}',{arg['actual']},oracle_event)\n"
    if 'components' in arg:
        text=''.join(dump_call(site,f,phase) for f in arg['components'])
        if arg['optional']:
            guard=arg.get('presence_guard') or '.true.'
            text=f"      call oracle_write('{site['site']}','{phase}','{arg['name']}.present',{guard},oracle_event)\n      if({guard}) then\n"+text+'      endif\n'
        return text
    actual = arg['actual']
    rank = len(arg['dims'])
    # WRF layout is (cell, level, tile); other layouts stay unmapped until audited.
    cell = arg.get('cell_axis',next((i + 1 for i, d in enumerate(arg['dims']) if re.search(r'\bims\b|\bits\b', d, re.I)), 0))
    level = arg.get('level_axis',next((i + 1 for i, d in enumerate(arg['dims']) if re.search(r'\bkms\b|\bkts\b', d, re.I)), 0))
    lower = f', lower=lbound({actual})' if rank else ''
    line=(f"      call oracle_write('{site['site']}', '{phase}', '{arg['name']}', &\n"
            f"           {actual}, oracle_event{lower}, cell_axis={cell}, level_axis={level})\n")
    if has_presence(arg):
        guard=arg.get('presence_guard') or '.true.'
        line=f"      call oracle_write('{site['site']}','{phase}','{arg['name']}.present',{guard},oracle_event)\n"+f'      if({guard}) then\n'+line+'      endif\n'
    if arg.get('actual_optional'):
        # Do not combine PRESENT with ALLOCATED/ASSOCIATED using .and.:
        # Fortran does not require short circuit evaluation of those operands.
        caller_guard=arg['actual_presence_guard']
        line=(f'      if({caller_guard}) then\n'+line+
              f"      else\n      call oracle_write('{site['site']}','{phase}','{arg['name']}.present',.false.,oracle_event)\n      endif\n")
    return line


def overlay(root, out, manifest):
    by_path = {}
    for site in manifest['sites']:
        if site['issues']:
            continue
        by_path.setdefault(site['path'], []).append(site)
    patch_paths = []
    for path, sites in by_path.items():
        lines = (root / path).read_text().splitlines(keepends=True)
        inserts = {}
        replacements={}
        rs = list(routines(root / path, root))
        for driver in rs:
            selected = [x for x in sites if x['driver'] == driver.name]
            if not selected:
                continue
            inserts.setdefault(driver.start, []).append('      use mpas_oracle_dump, only: oracle_begin, oracle_write, oracle_scope_begin, oracle_scope_end, oracle_replaying\n')
            # USE must follow a continued signature, and declarations before any executable.
            from fortran_source import statements
            signature = next(st for st in statements(''.join(lines)) if st.start == driver.start)
            inserts[driver.start].pop()
            inserts.setdefault(signature.end, []).append('      use mpas_oracle_dump, only: oracle_begin, oracle_write, oracle_scope_begin, oracle_scope_end, oracle_replaying\n')
            if any(a.get('native_pool') for site in selected for a in site['arguments']):inserts.setdefault(signature.end,[]).append('      use mpas_oracle_pool,only:oracle_pool_capture\n')
            first = first_executable(driver)
            first_exec = first.start if first else None
            if first_exec is None:
                raise ValueError(f'cannot locate specification end: {driver.name}')
            inserts.setdefault(first_exec, []).append('      integer :: oracle_event\n')
            for site in selected:
                pre = f"      if(.not.oracle_replaying()) then\n      call oracle_begin('{site['site']}', oracle_event)\n"
                pre += f"      call oracle_scope_begin('{site['site']}',oracle_event)\n"
                post = '      if(.not.oracle_replaying()) then\n'
                for arg in site['arguments']:
                    if arg['intent'] in {'in', 'inout'} or arg.get('entry_storage'):
                        pre += dump_call(site, arg, 'pre')
                    if arg['intent'] in {'out', 'inout'} or arg.get('native_pool') or arg.get('pool_argument'):
                        post += dump_call(site, arg, 'post')
                pre += '      endif\n'
                pre += f"! oracle_native_call_begin {site['site']}\n"
                post = f"! oracle_native_call_end {site['site']}\n"+post
                post += '      call oracle_scope_end()\n      endif\n'
                original=''.join(lines[site['line']-1:site['end_line']])
                conditional=re.match(r'\s*if\s*(\(.*\))\s*call\b',original.splitlines()[0],re.I)
                if conditional:
                    pre='      if'+conditional[1]+' then\n'+pre
                    post+='      endif\n'
                    replacements[site['line']-1]=re.sub(r'^\s*if\s*\(.*\)\s*(?=call\b)','      ',lines[site['line']-1],count=1,flags=re.I)
                inserts.setdefault(site['line'] - 1, []).append(pre)
                inserts.setdefault(site['end_line'], []).append(post)
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(''.join(''.join(inserts.get(i, [])) + replacements.get(i,line) for i, line in enumerate(lines)) + ''.join(inserts.get(len(lines), [])))
        patch_paths.append(path)
    manifest['overlay_paths'] = patch_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--overlay', action='store_true')
    p.add_argument('--pin-set',choices=['mpas-v842','rrfs','rrfs-44c0849'],default='mpas-v842')
    p.add_argument('--external-observer', action='append', default=[], metavar='SCHEME',
                   help='emit source-derived GNU ABI Rust observation without rebuilding native')
    p.add_argument('--external-hidden', action='append', default=[], metavar='CALLEE:DUMMY',
                   help='capture a native local actual at its internal callee entry')
    p.add_argument('--external-native-replay', action='store_true',
                   help='emit complete standalone native modules for the selected external ABIs')
    p.add_argument('--external-replay-recipe', action='append', choices=['gnu-strict','gnu-o3','coverage'],
                   help='external replay build recipes (default: all three)')
    p.add_argument('--external-preserve-calls',action='store_true',
                   help='disable standalone call inlining/cloning for nested observation; never rebuild production native')
    a = p.parse_args()
    if a.out.resolve() == a.source.resolve() or a.source.resolve() in a.out.resolve().parents:
        p.error('output must be outside the read-only source root')
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = inventory(a.source,a.pin_set)
    if (a.external_hidden or a.external_native_replay or a.external_preserve_calls) and not a.external_observer:
        p.error('external hidden/replay generation requires --external-observer')
    if a.external_observer:
        from external_abi import emit
        emit(a.source, a.out/'external', manifest, a.external_observer, a.external_hidden)
        if a.external_native_replay:
            from external_abi import prepare_native_replay
            for selected in manifest['external_observers']:
                spec=json.loads(Path(selected['spec']).read_text())
                prepare_native_replay(a.source,spec,manifest,a.out/'external-replay'/spec['callee']['name'],
                                      a.external_replay_recipe or ('gnu-strict','gnu-o3','coverage'),
                                      preserve_calls=a.external_preserve_calls)
    from replay.generate_replay import write_replays
    write_replays(manifest, a.out / 'replay')
    if a.overlay:
        overlay(a.source, a.out, manifest)
        from boundaries import generate_boundaries,generate_output_diagnostics
        generate_boundaries(a.source,a.out,manifest)
        generate_output_diagnostics(a.source,a.out,manifest)
    (a.out / 'inventory.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'sites': len(manifest['sites']), 'unresolved': len(manifest['unresolved']),
                      'argument_complete': manifest['argument_complete'], 'hidden_state_complete': False}))


if __name__ == '__main__':
    main()
