"""Translate the pinned WRF FLG column, preserving statement association.

The WRF driver leaves naero at its module default of zero. This column
specializes that constant; it does not expose the offline aerosol interface.
All ice and gas choices remain available to the module replay harness.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
from extract_tables import statements, split_items
from generate_gas_kernel import Expression, FLOAT_GLOBALS, INT_GLOBALS, ROUTINES as GAS
from generate_stream_kernel import declarations, STREAM_FLAGS, ROUTINES as STREAMS

PIN = 'c4977cdf2ddf3f2279c9be7e1fc659ef23bbb37376e5cf11f9a98a73d33b0266'
ROUTINES = ('thicks','rayle2','rayle','planck','ice','ice_new_zf','ice_new_comb',
            'ice_new_trop','ice_new_midlat','ice_98','ice_singleice','ice_new_single',
            'water_fl','rain','graup','comscp_aero_cld','rad_all')


def inventory(source):
    routines = {}; current = None
    for line in statements(source):
        match = re.match(r'subroutine\s+(\w+)\s*\((.*?)\)', line, re.I)
        if match:
            current = match[1].lower()
            routines[current] = {'args':[v.strip().lower() for v in match[2].split(',')], 'lines':[]}
        elif re.match(r'end\s*subroutine',line,re.I): current = None
        elif current:
            routines[current]['lines'].append(re.sub(r'^\d+\s+','',line.lower()).strip())
    for routine in routines.values():
        routine['mutable'] = set()
        for line in routine['lines']:
            for arg in routine['args']:
                if re.search(r'\b'+arg+r'\s*(?:\([^)]*\))?\s*=(?!=)',line) and not re.match(r'(real|integer)',line):
                    routine['mutable'].add(arg)
    changed = True
    while changed:
        changed = False
        for routine in routines.values():
            for line in routine['lines']:
                match = re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',line)
                if not match or match[1] not in routines: continue
                called = routines[match[1]]
                for formal, value in zip(called['args'],split_items(match[2])):
                    if formal not in called['mutable']: continue
                    base = re.match(r'(\w+)', value)
                    if base and base[1] in routine['args'] and base[1] not in routine['mutable']:
                        routine['mutable'].add(base[1]); changed = True
    if not set(ROUTINES).issubset(routines): raise ValueError('column routine inventory differs')
    return routines


def translate(name, routine, routines, manifest):
    variables = {k:(str(v),'int') for k,v in dict(mbs=6,mbir=12,mb=18,nvx=100,nv1x=101,mxat=7,mxac=18,mbx=18,ngroup=3,nc=8).items()}
    variables.update(naero=('0','int'), nice=('gas_switches[27]','int'), ngas=('gas_switches[28]','int'),
                     nsubcld=('((nv-1)/3)','int'))
    arrays = {}; local = []
    table_ids = {key:i for i,key in enumerate(manifest['tables'])}
    for i,key in enumerate(FLOAT_GLOBALS): variables[key] = (f'gas_values[{i}]','float')
    for i,key in enumerate(INT_GLOBALS): variables[key] = (f'gas_switches[{i}]','int')
    for i,key in enumerate(STREAM_FLAGS): variables[key] = (f'gas_switches[{20+i}]','int')
    arrays['itps'] = ('unused_itps',['18'],'int',['1'])
    local.append('const int unused_itps[18]={};')
    for line in routine['lines']:
        if line.startswith('use '):
            module = re.match(r'use\s+(\w+)',line)[1]
            for key, row in manifest['tables'].items():
                if key.startswith(module+'.'):
                    if row['shape']:
                        arrays[key.split('.')[1]] = (f'(tables+table_offsets[{table_ids[key]}])', [str(d) for d in row['shape']])
                    else:
                        variables[key.split('.')[1]] = (f'tables[table_offsets[{table_ids[key]}]]','float')
            for alias,value in re.findall(r'(\w+)\s*=>\s*(\w+)',line): variables[alias] = variables[value]
    decls = declarations(routine)
    def expr(text):
        text=re.sub(r'(\d)(\.(?:and|or|eq|ne|lt|le|gt|ge)\.)',r'\1 \2',text.strip())
        return Expression(text,variables,arrays).complete()
    for var, decl in decls.items():
        key = name+'.'+var
        if decl['dims']:
            lower = []; dims = []
            for d in decl['dims']:
                if ':' in d:
                    lo,hi=d.split(':'); lower.append(lo); dims.append('('+hi+'-'+lo+'+1)')
                else: lower.append('1'); dims.append(d)
            for k,(v,t) in variables.items():
                if t=='int' and re.fullmatch(r'\d+',v): dims=[re.sub(r'\b'+k+r'\b',v,d) for d in dims]
            if key in table_ids:
                base=f'(tables+table_offsets[{table_ids[key]}])'
                # The packed asset stores integer DATA values as float32.
                kind='float'
            else: base=var; kind=decl['kind']
            arrays[var]=(base,dims,kind,lower)
            if var not in routine['args'] and key not in table_ids:
                cap='*'.join('('+re.sub(r'\b(nv1|nv)\b','101',d)+')' for d in dims)
                for k,(v,t) in variables.items():
                    if t=='int' and re.fullmatch(r'\d+',v): cap=re.sub(r'\b'+k+r'\b',v,cap)
                local.append(decl['kind']+' '+var+'['+cap+'];')
        else:
            variables[var]=(var,decl['kind'])
            if var not in routine['args']:
                value='' if decl['init'] is None else '='+expr(decl['init'])
                local.append(('constexpr ' if decl['parameter'] else '')+decl['kind']+' '+var+value+';')
    args=[]
    for arg in routine['args']:
        decl=decls[arg]
        if decl['dims']: args.append(('' if arg in routine['mutable'] else 'const ')+decl['kind']+'* '+arg)
        else: args.append(decl['kind']+('& ' if arg in routine['mutable'] else ' ')+arg)
    args.extend(['const float* tables','const int* table_offsets','const float* gas_values','int* gas_switches','int* error'])
    signature='__device__ void flgc_'+name+'('+','.join(args)+')'
    output=local[:]
    def emit(line):
        if not line or re.match(r'(use |implicit |real\b|integer\b|logical\b|data\b)',line): return
        if line=='return': output.append('return;'); return
        if line in ('endif','end if','enddo','end do'): output.append('}'); return
        if line=='else': output.append('}else{'); return
        if line.startswith(('elseif','else if')):
            output.append('}else if('+expr(re.search(r'\((.*)\)\s*then',line)[1])+'){'); return
        if re.match(r'if\s*\(',line):
            start=line.index('('); depth=0; end=None
            for i in range(start,len(line)):
                if line[i]=='(': depth+=1
                elif line[i]==')':
                    depth-=1
                    if depth==0: end=i; break
            if end is None: raise ValueError('unbalanced IF '+line)
            output.append('if('+expr(line[start+1:end])+'){')
            after=line[end+1:].strip()
            if after!='then': emit(after); output.append('}')
            return
        if line.startswith('do while'):
            output.append('while('+expr(re.search(r'\((.*)\)',line)[1])+'){'); return
        if line.startswith('do '):
            match=re.fullmatch(r'do\s+(\w+)\s*=\s*(.*)',line); bounds=list(split_items(match[2]))
            lower,upper=map(expr,bounds[:2]); step=expr(bounds[2]) if len(bounds)==3 else '1'; var=match[1]
            output.append(f'for({var}={lower};(({step})>0?{var}<={upper}:{var}>={upper});{var}+={step})'+'{'); return
        if line.startswith('write('):
            # Retain negative-depth diagnostics on device without a host sync.
            output.append('*error=3;'); return
        if line.startswith('call '):
            match=re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',line); target=match[1]
            if target in ('aerosol_init','aerosolxy'):
                # Unreachable with WRF's stock naero=0. No substitute calculation.
                output.append('*error=2; return;'); return
            if target not in (*ROUTINES,*GAS,*STREAMS): raise ValueError('unsupported column call '+line)
            called=routines[target]; actual=list(split_items(match[2])); values=[]; cd=declarations(called)
            if len(actual)!=len(called['args']): raise ValueError('argument count differs '+line)
            for formal,text in zip(called['args'],actual):
                value=expr(text)
                if cd[formal]['dims'] and re.match(r'\w+\s*\(',text): value='&('+value+')'
                values.append(value)
            values.extend(['tables','table_offsets','gas_values','gas_switches']+(['error'] if target not in GAS else []))
            prefix='flgc_' if target in ROUTINES else 'flg_' if target in GAS else 'flgs_'
            output.append(prefix+target+'('+','.join(values)+');')
            if target not in GAS: output.append('if(*error) return;')
            return
        if '=' in line:
            left,right=map(str.strip,line.split('=',1))
            if left in arrays:
                dims=arrays[left][1]; bound='*'.join('('+expr(d)+')' for d in dims)
                rhs=arrays[right][0]+'[copy_i]' if right in arrays else expr(right)
                output.append('for(int copy_i=0;copy_i<'+bound+';++copy_i) '+arrays[left][0]+'[copy_i]='+rhs+';')
            elif ':' in left:
                match=re.fullmatch(r'(\w+)\((\d+):(\d+)\)',left)
                if not match: raise ValueError('unsupported slice assignment '+line)
                for i in range(int(match[2]),int(match[3])+1): output.append(expr(match[1]+'('+str(i)+')')+'='+expr(right)+';')
            else: output.append(expr(left)+'='+expr(right)+';')
            return
        raise ValueError('unsupported column statement '+line)
    for line in routine['lines']:
        try: emit(line)
        except Exception as e: raise ValueError(name+': '+line) from e
    return signature,signature+'{\n'+'\n'.join(output)+'\n}\n'


def generate(source,manifest):
    if hashlib.sha256(source.encode()).hexdigest()!=PIN: raise ValueError('unpinned FLG source')
    routines=inventory(source); signatures=[]; bodies=[]
    for name in ROUTINES:
        signature,body=translate(name,routines[name],routines,manifest)
        signatures.append(signature+';'); bodies.append(body)
    return '// Generated from WRF module_ra_flg.F SHA256 '+PIN+'\n'+'\n'.join(signatures)+'\n'+'\n'.join(bodies)

if __name__=='__main__':
    Path(sys.argv[3]).write_text(generate(Path(sys.argv[1]).read_text(),json.loads(Path(sys.argv[2]).read_text())))
