"""Closed statement translation of the callable WRF FLG stream solvers."""
from pathlib import Path
import json,re,sys
from extract_tables import statements,split_items
from generate_gas_kernel import Expression,FLOAT_GLOBALS,INT_GLOBALS

ROUTINES=('coeff1','coeff2','coeff4','coeffl','coefft','coefft0','qcfe','qcfel','adjust','qfts','qfti',
          'cfgts0','cfgts','qccgts','tridag','qftsts','qftisf')
STREAM_FLAGS=('d4s','d2s','d4ir','d2ir','edding','quadra','hemisp')

def inventory(source):
    routines={}; current=None
    for line in statements(source):
        match=re.match(r'subroutine\s+(\w+)\s*\((.*?)\)',line,re.I)
        if match:
            current=match[1].lower()
            if current in ROUTINES: routines[current]={'args':[x.strip().lower() for x in match[2].split(',')],'lines':[]}
        elif re.match(r'end\s*subroutine',line,re.I): current=None
        elif current in ROUTINES: routines[current]['lines'].append(re.sub(r'^\d+\s+','',line.lower()).strip())
    if set(routines)!=set(ROUTINES): raise ValueError('stream routine inventory differs')
    for routine in routines.values():
        routine['mutable']=set()
        for line in routine['lines']:
            for arg in routine['args']:
                if re.search(r'\b'+arg+r'\s*(?:\([^)]*\))?\s*=(?!=)',line) and not re.match(r'(real|integer)',line):
                    routine['mutable'].add(arg)
    changed=True
    while changed:
        changed=False
        for routine in routines.values():
            for line in routine['lines']:
                match=re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',line)
                if not match or match[1] not in routines: continue
                called=routines[match[1]]; actual=list(split_items(match[2]))
                for formal,value in zip(called['args'],actual):
                    if formal not in called['mutable']: continue
                    base=re.match(r'(\w+)',value)
                    if base and base[1] in routine['args'] and base[1] not in routine['mutable']:
                        routine['mutable'].add(base[1]); changed=True
    return routines

def declarations(routine):
    result={}
    for line in routine['lines']:
        match=re.match(r'(real|integer|logical)\b(.*)',line)
        if not match: continue
        kind={'real':'float','integer':'int','logical':'bool'}[match[1]]; spec=match[2]; attrs=''
        if '::' in spec: attrs,spec=spec.split('::',1)
        common=re.search(r'dimension\(([^)]+)\)',attrs)
        for item in split_items(spec):
            match=re.fullmatch(r'(\w+)\s*(?:\(([^)]+)\))?\s*(?:=\s*(.*))?',item)
            if not match: raise ValueError('unsupported declaration '+item)
            dims=match[2] or (common[1] if common else None)
            result[match[1]]={'kind':kind,'dims':None if dims is None else [x.strip() for x in dims.split(',')],
                            'init':match[3],'parameter':'parameter' in attrs}
    return result

def translate(name,routine,routines,manifest):
    vars={'mbs':('6','int'),'mbir':('12','int'),'mb':('18','int')}; arrays={}; local=[]
    table_ids={key:i for i,key in enumerate(manifest['tables'])}
    for i,key in enumerate(FLOAT_GLOBALS): vars[key]=(f'gas_values[{i}]','float')
    for i,key in enumerate(INT_GLOBALS): vars[key]=(f'gas_switches[{i}]','int')
    for i,key in enumerate(STREAM_FLAGS): vars[key]=(f'(gas_switches[{20+i}]!=0)','bool')
    for line in routine['lines']:
        if line.startswith('use numericals'):
            for key,row in manifest['tables'].items():
                if key.startswith('numericals.'):
                    arrays[key.split('.')[1]]=(f'(tables+table_offsets[{table_ids[key]}])',[str(d) for d in row['shape']])
        if line.startswith('use control_para'):
            for alias,value in re.findall(r'(\w+)\s*=>\s*(\w+)',line): vars[alias]=vars[value]
    decls=declarations(routine)
    def expr(text): return Expression(text.strip(),vars,arrays).complete()
    for var,decl in decls.items():
        if decl['dims']:
            key=name+'.'+var
            arrays[var]=((f'(tables+table_offsets[{table_ids[key]}])' if key in table_ids else var),decl['dims'])
            if var not in routine['args'] and key not in table_ids:
                capacity='*'.join('('+re.sub(r'\b(nv1|nv)\b','101',d)+')' for d in decl['dims'])
                local.append(decl['kind']+' '+var+'['+capacity+'];')
        else:
            vars[var]=(var,decl['kind'])
            if var not in routine['args']:
                value='' if decl['init'] is None else '='+expr(decl['init'])
                local.append(('constexpr ' if decl['parameter'] else '')+decl['kind']+' '+var+value+';')
    args=[]
    for arg in routine['args']:
        decl=decls[arg]; mutable=arg in routine['mutable']
        if decl['dims']: args.append(('' if mutable else 'const ')+decl['kind']+'* '+arg)
        else: args.append(decl['kind']+('& ' if mutable else ' ')+arg)
    args.extend(['const float* tables','const int* table_offsets','const float* gas_values','const int* gas_switches','int* error'])
    signature='__device__ void flgs_'+name+'('+','.join(args)+')'
    output=local[:]
    def emit(line):
        if not line or re.match(r'(use |implicit |real\b|integer\b|logical\b|data\b)',line): return
        if line=='return': output.append('return;'); return
        if line in ('endif','end if','enddo','end do'): output.append('}'); return
        if line=='else': output.append('}else{'); return
        if line.startswith(('elseif','else if')):
            condition=re.search(r'\((.*)\)\s*then',line)
            if not condition: raise ValueError('unsupported ELSE IF '+line)
            output.append('}else if('+expr(condition[1])+'){'); return
        if re.match(r'if\s*\(',line):
            start=line.index('('); depth=0; end=None
            for i in range(start,len(line)):
                if line[i]=='(': depth+=1
                elif line[i]==')':
                    depth-=1
                    if depth==0: end=i; break
            if end is None: raise ValueError('unbalanced IF '+line)
            condition=expr(line[start+1:end]); after=line[end+1:].strip()
            output.append('if('+condition+'){')
            if after!='then': emit(after); output.append('}')
            return
        if line.startswith('do while'):
            output.append('while('+expr(re.search(r'\((.*)\)',line)[1])+'){'); return
        if line.startswith('do '):
            match=re.fullmatch(r'do\s+(\w+)\s*=\s*(.*)',line); bounds=list(split_items(match[2]))
            if len(bounds) not in (2,3): raise ValueError('unsupported DO '+line)
            lower,upper=map(expr,bounds[:2]); step=expr(bounds[2]) if len(bounds)==3 else '1'; var=match[1]
            output.append(f'for({var}={lower};(({step})>0?{var}<={upper}:{var}>={upper});{var}+={step})'+'{'); return
        if line.startswith('call wrf_error_fatal'):
            output.extend(['*error=1;','return;']); return
        if line.startswith('call '):
            match=re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',line)
            if match[1] not in routines: raise ValueError('unsupported stream call '+line)
            called=routines[match[1]]; actual=list(split_items(match[2])); values=[]; cd=declarations(called)
            if len(actual)!=len(called['args']): raise ValueError('argument count differs '+line)
            for formal,text in zip(called['args'],actual):
                value=expr(text)
                if cd[formal]['dims'] and re.match(r'\w+\s*\(',text): value='&('+value+')'
                values.append(value)
            values.extend(['tables','table_offsets','gas_values','gas_switches','error'])
            output.append('flgs_'+match[1]+'('+','.join(values)+');'); output.append('if(*error) return;'); return
        if '=' in line:
            left,right=line.split('=',1); output.append(expr(left)+'='+expr(right)+';'); return
        raise ValueError('unsupported stream statement '+line)
    for line in routine['lines']: emit(line)
    return signature,signature+'{\n'+'\n'.join(output)+'\n}\n'

def generate(source,manifest):
    routines=inventory(source); signatures=[]; bodies=[]
    for name in ROUTINES:
        signature,body=translate(name,routines[name],routines,manifest); signatures.append(signature+';'); bodies.append(body)
    prefix='#define FG_AD(a,b) __fadd_rn((a),(b))\n#define FG_SU(a,b) __fsub_rn((a),(b))\n#define FG_MU(a,b) __fmul_rn((a),(b))\n#define FG_DV(a,b) __fdiv_rn((a),(b))\n'
    return '// Generated from pristine WRF FLG stream solver statements.\n'+prefix+'\n'.join(signatures)+'\n'+'\n'.join(bodies)

if __name__=='__main__':
    Path(sys.argv[3]).write_text(generate(Path(sys.argv[1]).read_text(),json.loads(Path(sys.argv[2]).read_text())),encoding='utf-8')
