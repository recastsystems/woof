#!/usr/bin/env python3
"""Offline lowering of pinned WRF 4.7.1 MYNN to a column CUDA kernel.

WRF public-domain numerical source. The native Fortran remains the referee;
this generator does not confer replay qualification. No native runtime is used.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, re, subprocess
from pathlib import Path

_helpers = Path(__file__).parents[1] / 'noahowp_port/lower.py'
_spec = importlib.util.spec_from_file_location('_f77_helpers', _helpers)
_h = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_h)
split, statements = _h.split, _h.statements


def decl(s):
    s = s.replace('real(kind_phys)', 'real').replace('real(kind=kind_phys)', 'real')
    s = re.sub(r'real\s*\(\s*8\s*\)', 'double precision', s)
    return _h.decl(s)


def read_units(source):
    units, globals_, current = [], [], None
    for s in statements(source):
        m = re.match(r'(subroutine|function)\s+(\w+)\s*\((.*?)\)$', s)
        if m:
            current = dict(name=m[2],kind=m[1],args=split(m[3]),body=[])
            units.append(current)
        elif re.match(r'end\s+(subroutine|function)', s): current = None
        elif current is not None: current['body'].append(s)
        elif (d := decl(s)) and 'parameter' in d[1]:
            for name,dims,value in d[2]: globals_.append((d[0],name,dims,value))
    return units, globals_


def fixed(s):
    m = re.match(r'^(\d+)\s+(.*)$', s)
    prefix = m[1].rjust(5)+' ' if m else '      '
    body = m[2] if m else s
    return '\n'.join((prefix if i==0 else '     &')+body[i:i+66] for i in range(0,len(body),66))


def schema(unit):
    result = {}
    for s in unit['body']:
        if d := decl(s):
            for name,dims,val in d[2]:
                if name in unit['args']:
                    result[name] = dict(type=d[0],dims=dims,intent=(re.search(r'intent\((.*?)\)',d[1]) or [None,'inout'])[1])
    return {n:result[n] for n in unit['args']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--f2c',type=Path)
    p.add_argument('--nz',type=int,default=49);p.add_argument('--constants',type=Path)
    p.add_argument('--prepare',action='store_true');a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    src=a.source/'phys/module_bl_mynnedmf.F'
    pinned = {
        'module_bl_mynnedmf.F': '6e4a7d5b35ce46f01591f2c1d58e545380d546e654b4a59ee1bcf99cfbce2d72',
        'module_bl_mynnedmf_common.F': '02af5bd4b97903ce3cd44458bdd218c504fa6034b7d20dfffc983b88bff5038e',
    }
    for filename, expected in pinned.items():
        actual = hashlib.sha256((a.source/'phys'/filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'{filename}: expected pinned WRF 4.7.1 {expected}, got {actual}')
    units,global_=read_units(src)
    _,common=read_units(a.source/'phys/module_bl_mynnedmf_common.F')
    global_=common+global_
    # Constants imported from WRF's actual model-constants module.
    imported='karman g p1000mb cp r_d r_v rcp xlv xlf xls svp1 svp2 svp3 p608 ep_2 rvovrd cpv cliq cice svpt0'.split()
    known={x[1] for x in global_}
    global_=[('real',n,None,None) for n in imported if n not in known]+global_
    fields=schema(units[0]);(a.work/'arguments.json').write_text(json.dumps(fields,indent=2)+'\n')
    names=[n for typ,n,d,v in global_]
    printer=['program constants','use module_bl_mynnedmf','use module_bl_mynnedmf_common','implicit none',
             "open(99,file='constants.bin',access='stream',form='unformatted',status='replace')"]
    for typ,n,d,v in global_:
        printer.append('write(99) '+('merge(1,0,'+n+')' if typ=='logical' else n))
    printer+=['close(99)','end program']
    (a.work/'constants.F90').write_text('\n'.join(printer)+'\n')
    (a.work/'global-constants.json').write_text(json.dumps(global_,indent=2)+'\n')
    if a.prepare: return
    import struct
    values=struct.unpack('<'+'I'*len(names),a.constants.read_bytes())
    constants=[]
    for (typ,n,d,v),bits in zip(global_,values):
        if typ=='real': val=format(struct.unpack('<f',struct.pack('<I',bits))[0],'.30e')
        elif typ=='logical': val='.true.' if bits else '.false.'
        else: val=str(struct.unpack('<i',struct.pack('<I',bits))[0])
        # f2c folds PARAMETER REAL algebra in host double precision and
        # rounds only its final literal. WRF REAL32 rounds each operation.
        # DATA preserves the exact native constant but leaves its arithmetic
        # to the target's correctly typed float operations.
        constants += [typ+' '+n,('data '+n+' /'+val+'/' if typ=='real'
                                else 'parameter ('+n+'='+val+')')]
    funcs=[u['name'] for u in units if u['kind']=='function']
    # Globals are immutable; all routine scratch becomes thread-local.
    meta=[];output=[]
    for u in units:
        arrays={};declarations=[];initializers=[];data=[];executable=[]
        for s in u['body']:
            s=re.sub(r'_kind_phys\b','',s)
            s=re.sub(r',\s*(?:kind\s*=\s*)?kind_phys\b','',s)
            if s.startswith(('implicit ','use ','save','external ')): continue
            if d:=decl(s):
                typ,attrs,entries=d
                for n,dims,value in entries:
                    if typ=='character':
                        if n!='errmsg': raise ValueError(('character',u['name'],s))
                        continue
                    if n in funcs and u['kind']=='function' and n==u['name']: continue
                    if dims:
                        # Specialize only private scratch. Dummy bounds retain ABI.
                        if n not in u['args']:
                            dims=re.sub(r'\bkts\b','1',dims)
                            dims=re.sub(r'\b(kte|n)\b',str(a.nz),dims)
                            dims=re.sub(r'\b(nchem|ndvel)\b','1',dims)
                        arrays[n]=[x.split(':') if ':' in x else ['1',x] for x in split(dims)]
                    declarations.append(typ+' '+n+('('+dims+')' if dims else ''))
                    if value:
                        if 'parameter' in attrs and typ!='real': declarations.append('parameter ('+n+'='+value+')')
                        elif value.startswith('(/'): data.append('data '+n+' /'+value[2:-2]+'/')
                        else: initializers.append(n+'='+value)
                continue
            if s.startswith(('print','write','errmsg','return','stop')) or re.search(r'\b(?:print|write)\s*[*(]',s):
                if s=='return': executable.append(s)
                elif s.startswith('stop'): executable+=['errflg=1','return'] if u['name']=='mynnedmf' else ['return']
                continue
            s=s.replace(', errmsg, errflg',', errflg')
            # Logical optional arguments are all supplied by the column driver.
            executable.append(s)
        # Whole-array assignments and reductions, including rank-two plume arrays.
        def scalarize(expr,idx):
            ranges=[]
            for n,dims in sorted(arrays.items(),key=lambda x:-len(x[0])):
                pattern=r'\b'+n+r'\b(\s*\([^()]*\))?'
                def sub(m):
                    nonlocal ranges
                    subscript=m[1]
                    if subscript and ':' not in subscript: return m[0]
                    subs=split(subscript.strip()[1:-1]) if subscript else [':']*len(dims)
                    if len(subs)!=len(dims): raise ValueError(('rank',u['name'],m[0],dims))
                    addr=[]; ownranges=[]
                    for subscript,(lo,hi) in zip(subs,dims):
                        if ':' in subscript:
                            parts=subscript.split(':')
                            start,end=parts[0].strip() or lo,parts[1].strip() or hi
                            dimidx=idx+str(len(ownranges)+1)
                            ownranges.append((start,end));addr.append('('+start+')+('+dimidx+')-1')
                        else: addr.append(subscript)
                    if len(ownranges)>len(ranges): ranges=ownranges
                    return n+'('+','.join(addr)+')'
                expr=re.sub(pattern,sub,expr)
            return expr,ranges
        def array_walk(ranges,statement):
            # The first Fortran dimension is contiguous and is the inner walk.
            return ([f'do ixarr{k+1}=1,({hi})-({lo})+1'
                     for k,(lo,hi) in reversed(list(enumerate(ranges)))]
                    +[statement]+['enddo']*len(ranges))
        body=[];loops=[];selects=[];label=9000
        for s in initializers+executable:
            s=re.sub(r'^\w+:\s*do','do',s);s=re.sub(r'^end do\s+\w+$','enddo',s)
            # Single-line array IFs need a structured block for expansion.
            single=None
            if re.match(r'^if\s*\(',s) and not s.endswith(('then','&')):
                start=s.find('(');depth=1;end=start+1
                while depth:
                    if s[end]=='(':depth+=1
                    elif s[end]==')':depth-=1
                    end+=1
                tail=s[end:].strip()
                if '=' in tail and not tail.startswith(('goto','go to')):
                    single=s[:end];s=tail;body.append(single+' then')
            if s.startswith('select case'):
                selects.append([s[s.index('(')+1:-1],False]);continue
            if s.startswith('case'):
                sel,seen=selects[-1]
                if s=='case default':body.append('else')
                else:
                    vals=split(s[s.index('(')+1:-1]);cond=' .or. '.join('('+sel+'.eq.'+v+')' for v in vals)
                    body.append(('else if' if seen else 'if')+' ('+cond+') then')
                selects[-1][1]=True;continue
            if s=='end select':selects.pop();body.append('endif');continue
            if s=='do' or s.startswith('do while'):
                label+=3;loops.append((label,label+1,label+1));body.append(str(label)+' continue')
                if s!='do':body.append('if (.not. '+s[s.index('('):]+') goto '+str(label+1))
                continue
            if s.startswith('do '):
                label+=3;loops.append((None,label+1,label+2));body.append('do '+str(label+1)+' '+s[3:]);continue
            if s in ('enddo','end do'):
                start,end,after=loops.pop()
                if start:body.append('goto '+str(start))
                body.append(str(end)+' continue')
                if after!=end:body.append(str(after)+' continue')
                continue
            if re.match(r'^(cycle|exit)(\s+\w+)?$',s):
                body.append('goto '+str(loops[-1][1 if s.startswith('cycle') else 2]));continue
            if s.endswith(' exit'):body.append(s[:-5]+' goto '+str(loops[-1][2]));continue
            redcount=0
            while red:=re.search(r'\b(sum|maxval|minval)\(',s):
                depth=1;end=red.end()
                while depth:
                    if s[end]=='(':depth+=1
                    elif s[end]==')':depth-=1
                    end+=1
                val,ranges=scalarize(s[red.end():end-1],'ixarr')
                if not ranges:raise ValueError(('reduction',u['name'],s))
                redcount+=1;temp=f'red{redcount:03}';lo,hi=ranges[0]
                initial='0.0' if red[1]=='sum' else '-3.402823466e38' if red[1]=='maxval' else '3.402823466e38'
                combine=temp+'+'+val if red[1]=='sum' else ('max' if red[1]=='maxval' else 'min')+'('+temp+','+val+')'
                body += [temp+'='+initial]+array_walk(ranges,temp+'='+combine)
                s=s[:red.start()]+temp+s[end:]
            m=re.match(r'^(\w+(?:\([^=]*\))?)\s*=\s*(.*)$',s)
            if m:
                lhs,rhs=m[1].strip(),m[2].strip();left,ranges=scalarize(lhs,'ixarr')
                if ranges:
                    right,rr=scalarize(rhs,'ixarr');lo,hi=ranges[0]
                    body+=array_walk(ranges,left+'='+right)
                else:body.append(s)
            else:body.append(s)
            if single:body.append('endif')
        if loops or selects:raise ValueError(('unclosed',u['name']))
        # F2C's double constant folding must not alter REAL expressions.
        literals={}
        def literal(m):
            val=m[0]
            if 'd' in val:return val
            if val not in literals:literals[val]=f'rc{len(literals):04}'
            return literals[val]
        body=[re.sub(r'(?<![\w.])(?:\d+\.\d*|\.\d+|\d+[ed][+-]?\d+)(?:[ed][+-]?\d+)?(?![\w.])',literal,s) for s in body]
        args=[n for n in u['args'] if n!='errmsg']
        localnames={n for st in u['body'] if (d:=decl(st)) for n,_,_ in d[2]}
        ownconstants=[v for k in range(0,len(constants),2) if constants[k].split()[1] not in localnames for v in constants[k:k+2]]
        constant_decls=[s for s in ownconstants if not s.startswith('data ')]
        constant_data=[s for s in ownconstants if s.startswith('data ')]
        decls=[('real function' if u['kind']=='function' else 'subroutine')+' '+u['name']+'('+','.join(args)+')','implicit none',*constant_decls,'integer ceiling',
               'integer ixarr1,ixarr2,ixarr3','real red001,red002,red003,red004',
               *['real '+n for n in funcs if n!=u['name']],*declarations,
               *['real '+n for n in literals.values()],*constant_data,*data,
               *['data '+n+' /'+v+'/' for v,n in literals.items()],*body,'end']
        output += [fixed(s) for s in decls]
        meta.append({k:u[k] for k in ('name','kind','args')})
    output += [fixed(s) for s in ['integer function ceiling(x)','real x','ceiling=int(x)','if (x.gt.real(ceiling)) ceiling=ceiling+1','end']]
    (a.work/'model.f').write_text('\n'.join(output)+'\n')
    subprocess.run([str(a.f2c.resolve()),'-a','-R','-r','-u','-w','model.f'],cwd=a.work,check=True)
    (a.work/'schema.json').write_text(json.dumps(dict(nz=a.nz,arguments=fields,routines=meta,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':main()
