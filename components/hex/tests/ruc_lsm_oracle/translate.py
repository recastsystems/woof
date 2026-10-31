"""Development-only translation of pinned MPAS RUC source; never used by forecasts.

Adapted from the KF lane's Fortran subset translator. Emits original statement
locations and rejects every unsupported numerical statement. Diagnostic I/O is
omitted explicitly. Input is gfortran-preprocessed pristine copied source.
"""
from pathlib import Path
import re, json, hashlib
from dataclasses import dataclass

ROOT=Path(__file__).resolve().parents[4]
NATIVE=ROOT/'scratch/native'

def split(s):
    result=[];depth=0;last=0;quote=None
    for i,c in enumerate(s):
        if quote:
            if c==quote:quote=None
            continue
        if c in "'\"":quote=c
        if c=='(':depth+=1
        if c==')':depth-=1
        if c==',' and depth==0:result.append(s[last:i].strip());last=i+1
    result.append(s[last:].strip());return result

def statements(path):
    pending='';start=0;line=0
    for raw in path.read_text().splitlines():
        m=re.match(r'#\s+(\d+)\s+"',raw)
        if m:line=int(m[1])-1;continue
        line+=1
        if raw.startswith('#'):continue
        # Fortran comments cannot begin inside a literal.
        text=re.split(r"!",raw,maxsplit=1)[0].strip()
        if not text:continue
        if not pending:start=line
        pending+=' '+text.lstrip('&').rstrip('&').strip()
        if text.endswith('&'):continue
        # Semicolon statements do not occur inside computational string literals.
        for part in pending.split(';'):
            if part.strip():yield start,part.strip().lower()
        pending=''
    assert not pending

def declaration(s):
    m=re.match(r'(real|integer|logical|character)\b',s)
    if not m:return None
    ty={'real':'float','integer':'int','logical':'int','character':'int'}[m[1]]
    if '::' in s:attrs,vars=s.split('::',1)
    else:
        mm=re.match(r'(?:real|integer|logical|character)(?:\s*\([^)]*\)|\*\d+)?\s+(.*)',s)
        attrs=m[1];vars=mm[1]
    dim=re.search(r'dimension\s*\(',attrs)
    dims=None
    if dim:
        pos=dim.end();depth=1;end=pos
        while depth:
            if attrs[end]=='(':depth+=1
            if attrs[end]==')':depth-=1
            end+=1
        dims=split(attrs[pos:end-1])
    intent=re.search(r'intent\s*\(\s*(\w+)\s*\)',attrs)
    result={}
    for var in split(vars):
        init=None
        if '=' in var:var,init=map(str.strip,var.split('=',1))
        mm=re.fullmatch(r'(\w+)\s*\((.*)\)',var)
        var_dims=split(mm[2]) if mm else dims or []
        var=mm[1] if mm else var
        result[var]={'type':ty,'dim':var_dims,'intent':intent[1] if intent else None,'optional':'optional' in attrs,'init':init,'parameter':'parameter' in attrs,'logical':m[1]=='logical','character':m[1]=='character'}
    return result

routines={};globals={};current=None
for unit in ('land','ice','diags','init','drivers'):
    g={}
    source={'land':'module_ruc_land.F90','ice':'module_ruc_ice.F90','diags':'module_sf_sfcdiags_ruclsm.F','init':'mpas_atmphys_lsm_rucinit.F','drivers':'mpas_atmphys_driver_seaice.F'}[unit]
    for n,s in statements(NATIVE/(unit+'.pp')):
        m=re.match(r'(?:real\s+)?(subroutine|function) (\w+)\s*\((.*)\)',s)
        if m:
            kind,name,args=m.groups()
            routine_source={'correct_tsk_over_seaice':'mpas_atmphys_lsm_shared.F','driver_forcing':'mpas_atmphys_driver_lsm.F'}.get(name,source)
            current={'name':name,'unit':unit,'source':routine_source,'args':split(args),'symbols':{},'data':{},'body':[],'start':n,'function':kind=='function'}
            routines[(unit,name)]=current
        elif s.startswith(('end subroutine','end function')):
            current['end']=n
            bounds={'rucinit':(72,245),'seaice_prepare':(436,442),'seaice_finish':(571,601),'driver_forcing':(606,640)}
            if current['name'] in bounds:current['start'],current['end']=bounds[current['name']]
            current=None
        elif current:
            d=declaration(s)
            if d:current['symbols'].update(d)
            elif s.startswith('data '):
                names,vals,_=s[5:].split('/');current['data'].update(zip(split(names),split(vals)))
            elif not s.startswith(('implicit ','use ')):current['body'].append((n,s))
        elif declaration(s):g.update(declaration(s))
    globals[unit]=g
globals['init']={**globals['land'],**globals['init']}
# These are table input routines and the unused WRF compatibility init, not
# forecast arithmetic. MPAS's rucinit computation is emitted separately.
routines={k:r for k,r in routines.items() if r['name'] not in ('ruclsminit','ruclsm_soilvegparm')}
for (unit,name),r in routines.items():
    if r['function']:
        r['symbols'][name]={'type':'float','dim':[],'intent':None,'optional':False,'init':None,'parameter':False,'logical':False,'character':False}
# The native module imports these from the fork constants, single precision.
constants={'g':'9.80616','rhowater':'1000.0','piconst':'3.141592653589793','ep_1':'0.6083623693379792','ep_2':'0.6217504332755633','r_d':'287.0','r_v':'461.6','cp':'1004.5','xlv':'2.50e6','xls':'2.85e6','xlf':'3.50e5','t00':'273.15','p1000mb':'100000.0','ep_1':'0.6083624362945557'}
for g in globals.values():
    for k,v in constants.items():g[k]={'type':'float','dim':[],'intent':None,'optional':False,'init':v,'parameter':True,'logical':False,'character':False}
# TABLES is rebound per routine; only module globals absent from local scope
# are qualified through the parameter table object.
TABLES={}
STRINGS={'usgs':1,'modis':2,'modified_igbp_modis_noah':3,'usgs-ruc':4,'modi-ruc':5,'stas-ruc':6}

class Expr:
    def __init__(self,s,syms):
        self.syms=syms;self.ts=re.findall(r"'[^']*'|\d+(?:\.(?![a-z]+\.)\d*)?(?:[ed][+-]?\d+)?|\.\d+(?:[ed][+-]?\d+)?|\.\w+\.|\w+|\*\*|>=|<=|==|/=|[()+*/<>,=:-]",s);self.i=0
        self.s=s
    def peek(self):return self.ts[self.i] if self.i<len(self.ts) else ''
    def pop(self):v=self.peek();self.i+=1;return v
    def read(self,prec=0):
        t=self.pop()
        if t in ('+','-','.not.'):
            v,kind=self.read(65 if t!='.not.' else 25);v=f'({"!" if t==".not." else t}{v})'
        elif t=='(':
            v,kind=self.read();assert self.pop()==')',(self.s,self.ts[self.i:])
        elif t.startswith("'"):
            v=str(STRINGS[t[1:-1]]);kind='int'
        elif re.match(r'^(\d|\.\d)',t):
            kind='float' if any(x in t for x in '.ed') else 'int'
            v=t.replace('d','e')+('f' if kind=='float' else '')
        elif t in ('.true.','.false.'):v='true' if t=='.true.' else 'false';kind='bool'
        else:
            assert t in self.syms or self.peek()=='(',('unknown',t,self.s)
            kind=self.syms.get(t,{}).get('type','float');v=('tb.' if t in TABLES and not TABLES[t]['parameter'] else '')+t
            if self.peek()=='(':
                self.pop();args=[];kinds=[]
                if self.peek()!=')':
                    while True:
                        if self.peek()==':':
                            self.pop();a='RRange{0,0}';k='range'
                        else:
                            a,k=self.read()
                            if self.peek()==':':
                                self.pop();hi,_=self.read();a='RRange{'+a+','+hi+'}';k='range'
                        args.append(a);kinds.append(k)
                        if self.peek()!=',':break
                        self.pop()
                assert self.pop()==')',(self.s,self.ts[self.i:])
                if t in self.syms and self.syms[t]['dim']:v+= '('+', '.join(args)+')'
                elif t in ('float','real'):v='float('+args[0]+')';kind='float'
                elif t in ('int','ifix'):v='int('+args[0]+')';kind='int'
                elif t=='nint':v='int(roundf('+args[0]+'))';kind='int'
                elif t=='aint':v='truncf('+args[0]+')';kind='float'
                elif t=='present':v='has_'+args[0];kind='bool'
                elif t=='mod':v=f'({args[0]} % {args[1]})';kind='int'
                elif t in ('exp','alog','log','sqrt'):
                    v={'exp':'gfk_exp_fma','alog':'gfk_log','log':'gfk_log','sqrt':'__fsqrt_rn'}[t]+'('+args[0]+')';kind='float'
                elif t in ('abs',):v=('fabsf' if kinds[0]=='float' else 'abs')+'('+args[0]+')';kind=kinds[0]
                elif t in ('amax1','amin1','min0','max0','min','max'):
                    kind='float' if 'float' in kinds else 'int';fn='ruc_max' if 'max' in t else 'ruc_min'
                    v=args[0]
                    for a in args[1:]:v=f'{fn}<{kind}>({v}, {a})'
                elif t in ('log10','alog10','tanh','cos'):
                    v={'log10':'ruc_log10f_rn','alog10':'ruc_log10f_rn','tanh':'sfire_tanhf','cos':'glibc_cosf'}[t]+'('+args[0]+')';kind='float'
                elif (UNIT,t) in routines:
                    v=UNIT+'_'+t+'(tb, arena, '+', '.join(args)+')';kind='float'
                else:raise Exception(('intrinsic',t,self.s))
        ops={'.or.':10,'.and.':20,'.eq.':30,'.ne.':30,'.lt.':30,'.le.':30,'.gt.':30,'.ge.':30,'==':30,'/=':30,'>':30,'<':30,'>=':30,'<=':30,'+':40,'-':40,'*':50,'/':50,'**':70}
        mapping={'.or.':'||','.and.':'&&','.eq.':'==','.ne.':'!=','.lt.':'<','.le.':'<=','.gt.':'>','.ge.':'>=','/=':'!='}
        while self.peek() in ops and ops[self.peek()]>prec:
            op=self.pop();rhs,rkind=self.read(ops[op]-1 if op=='**' else ops[op]);newkind='float' if 'float' in (kind,rkind) else 'int'
            if op=='**':
                # GNU also lowers small integral REAL literals to products.
                # Native soil.F90:2474 disassembly is subss, mulss, mulss;
                # calling powf for **2. changed two first-call flux words.
                # Strict GNU simplifies real squares/identity powers but
                # preserves powf for real cubes: replacing **3. by products
                # changed 266 soil-moisture words in the full replay.
                integral_real=re.fullmatch(r'([0-2])(?:\.0*)?f',rhs) if rkind=='float' else None
                if (rkind=='int' and re.fullmatch(r'\d+',rhs)) or integral_real:
                    power=int(integral_real[1] if integral_real else rhs)
                    # GNU's small integer powers are balanced products.
                    def powchain(k):
                        if k==0:return '1.0f'
                        if k==1:return v
                        if k%2==0:return f'__fmul_rn({powchain(k//2)}, {powchain(k//2)})'
                        return f'__fmul_rn({v}, {powchain(k-1)})'
                    v=powchain(power)
                else:v=f'gfk_pow({v}, {rhs})'
                kind='float'
            elif ops[op]<=30:v=f'({v} {mapping.get(op,op)} {rhs})';kind='bool'
            elif newkind=='float':
                fn={'+':'__fadd_rn','-':'__fsub_rn','*':'__fmul_rn','/':'__fdiv_rn'}[op]
                v=f'{fn}({v}, {rhs})';kind='float'
            else:v=f'({v} {op} {rhs})';kind='int'
        return v,kind
    def full(self):
        v,k=self.read();assert self.i==len(self.ts),(self.s,self.ts[self.i:]);return v

def symbols(r):
    return {**globals[r['unit']],**r['symbols']}
def signature(r):
    args=['Tables &tb','Arena &arena']
    for a in r['args']:
        d=r['symbols'][a]
        ty=f'FArray<{d["type"]}>' if d['dim'] else d['type']+(' &' if d['intent']!='in' else ' ')
        args.append(ty+' '+a)
    for a in r['args']:
        if r['symbols'][a]['optional']:args.append('bool has_'+a)
    return ('RUCDEV float ' if r['function'] else 'RUCDEV void ')+r['unit']+'_'+r['name']+'('+', '.join(args)+')'
def generate(r, trace=False):
    global TABLES,UNIT
    UNIT=r['unit'];TABLES={k:v for k,v in globals[UNIT].items() if k not in r['symbols']}
    syms=symbols(r);ex=lambda s:Expr(s,syms).full()
    prefix='src/core_atmosphere/physics/' + ('' if r['unit'] in ('init','drivers') else 'physics_noaa/RUCLSM/MPAS/')
    out=[f'// RRFS-MPAS 1d6e370e {prefix}{r["source"]}:{r["start"]}-{r["end"]}',signature(r)+' {','    const int arena_mark = arena.used;','    ArenaScope arena_scope(arena, arena_mark);']
    for name,d in r['symbols'].items():
        if name in r['args']:continue
        if d['dim']:
            bounds=[]
            for dim in d['dim']:
                lo,hi=dim.split(':') if ':' in dim else ('1',dim)
                bounds.extend((ex(lo),ex(hi)))
            out.append(f'    FArray<{d["type"]}> {name} = arena.alloc<{d["type"]}>('+', '.join(bounds)+');')
            if d['init']:
                vals=split(d['init'][2:-2])
                for i,v in enumerate(vals,1):out.append(f'    {name}({i}) = {ex(v)};')
        else:
            init=r['data'].get(name,d['init'])
            if name=='ilnb' and r['name'] in ('sfctmp','icetmp') and not init:
                out.append('    int ilnb = 0; '+"// Native leaves this local undefined (module_ruc_land.F90:1467 / module_ruc_ice.F90 sfctmp/icetmp); snowtemp only sets it for snow at least snth deep and reads it at :4583 for thinner snow, so native's tsnav there comes from stale stack words. Hex fixes it at 0 (the one-layer formula) so the port is deterministic; native-undefined residue, see MATCH.md.");continue
            out.append(f'    {d["type"]} {name}'+(' = '+ex(init) if init else '')+';')
    # A supplied native entry-storage packet restores the original OUT dummy
    # bytes before any numerical statement. No fixed value is selected here.
    # Fork module_ruc_land.F90:3677-3698,3835,3967-3993,4582-4591;
    # module_ruc_ice.F90:1602-1638,1834-1855,2200-2209.
    if r['unit']=='ice' and r['name']=='snowseaice':
        out[2:2]=['#ifdef RUC_ICE_ENTRY_CONTEXT', '    if (arena.snowtemp_calls) ++*arena.snowtemp_calls;', '    if (arena.entry_ilnb) ilnb = *arena.entry_ilnb;', '#endif']
    if r['name']=='snowtemp':
        out.append('    if (arena.snowtemp_calls) ++*arena.snowtemp_calls;')
        out.append('    if (arena.entry_ilnb) ilnb = *arena.entry_ilnb;')
    loops=[];loopid=0
    def statement(s,n):
        nonlocal loopid
        s=re.sub(r'^\w+:\s*(?=if)', '', s)
        label=''
        m=re.match(r'^(\d+)\s+(.*)',s)
        if m:label,s=m.groups()
        if s.startswith(('format','write','print')):return '; /* Native diagnostic I/O omitted; no physics mutation. */'
        if s=='continue':
            if loops and loops[-1]['label']==label:
                l=loops.pop();return f'}} {l["name"]}_exit: ;'
            return ';'
        m=re.match(r'(?:(\w+):\s*)?do\b\s*(?:(\d+)\s+)?(.*)',s)
        if m:
            name,lab,spec=m.groups();loopid+=1;name=name or f'loop{loopid}'
            loops.append({'name':name,'label':lab,'infinite':not spec})
            if not spec:return f'for (;;) {{ {name}_cycle: ;'
            var,bounds=spec.split('=',1);var=var.strip();bs=split(bounds);lo,hi=map(ex,bs[:2]);step=ex(bs[2]) if len(bs)>2 else '1'
            cmp='>=' if step.startswith('(-') else '<='
            return f'for ({var} = {lo}; {var} {cmp} {hi}; {var} += {step}) {{ {name}_cycle: ;'
        if s.startswith(('enddo','end do')):
            l=loops.pop();return f'}} {l["name"]}_exit: ;'
        if s.startswith(('exit','cycle')):
            cmd,*target=s.split();l=next(l for l in reversed(loops) if not target or l['name']==target[0])
            if cmd=='exit':return f'goto {l["name"]}_exit;'
            # Cycle jumps to end of counted-loop body, to top of infinite loop.
            return f'goto {l["name"]}_continue;'
        if s.startswith('else if') or s.startswith('elseif'):
            s=re.sub(r'^else\s*if','if',s);return '} else '+statement(s,n)
        if s in ('else',):return '} else {'
        if s.startswith(('endif','end if')):return '}'
        if re.match(r'if\s*\(',s):
            idx=s.index('(');depth=1;k=idx+1
            while depth:
                if s[k]=='(':depth+=1
                if s[k]==')':depth-=1
                k+=1
            cond=ex(s[idx+1:k-1]);tail=s[k:].strip()
            return f'if ({cond}) '+('{' if tail=='then' else '{ '+statement(tail,n)+' }')
        if s.startswith(('goto ','go to ')):
            return 'goto label_'+re.sub(r'^(?:go to|goto)\s*','',s)+';'
        if s=='return':return 'return '+(r['name'] if r['function'] else '')+';'
        if s=='stop':return 'atomicOr(arena.error, 2); return;'
        if s.startswith('fatal_error') or s.startswith('call physics_error_fatal'):
            return 'atomicOr(arena.error, 1); return;'
        if s.startswith(('call mpas_log_write','call physics_message','call wrf_message')):
            return '; /* Native diagnostic I/O; no numerical state. */'
        if s.startswith('call '):
            name,args=re.match(r'call (\w+)\s*\((.*)\)',s).groups();rr=routines[(UNIT,name)];aa=split(args)
            assert len(aa)==len(rr['args']),(name,len(aa),len(rr['args']))
            rendered=[]
            for a in aa:
                if a in syms and syms[a]['dim']:rendered.append(('tb.' if a in TABLES else '')+a)
                else:rendered.append(ex(a))
            for a in rr['args']:
                if rr['symbols'][a]['optional']:rendered.append('true')
            return UNIT+'_'+name+'(tb, arena, '+', '.join(rendered)+');'
        if '=' in s:
            lhs,rhs=map(str.strip,s.split('=',1))
            base=re.match(r'\w+',lhs)[0]
            array=base in syms and syms[base]['dim'] and ('(' not in lhs or ':' in lhs)
            if array:return 'ruc_assign('+ex(lhs)+', '+ex(rhs)+');'
            code=ex(lhs)+' = '+ex(rhs)+';'
            if trace:
                rid=list(routines).index((r['unit'],r['name']))
                code+=f' ruc_trace({rid}, {n}, '+ex(lhs)+');'
            return code
        raise Exception(('statement',n,s))
    for n,s in r['body']:
        # A separate end label for CYCLE preserves increment semantics.
        isend=s.startswith(('enddo','end do')) or (re.match(r'^\d+\s+continue$',s) and loops and loops[-1]['label']==s.split()[0])
        if isend:out.append('    '+loops[-1]['name']+'_continue: ;')
        label=re.match(r'^(\d+)\s+',s)
        if label and not isend:out.append('    label_'+label[1]+': ;')
        out.append('    '+statement(s,n)+f' // L{n}')
        # Native L501 returns the entire tile, after mask preparation of
        # earlier cells (L505-579). Preserve those mutations, then stop.
        if r['unit']=='land' and r['name']=='ruc_land' and n==581:
            out.append('    if (arena.preparation_only) return; // Native tile return L501; prefix through L582')
    assert not loops,r['name']
    if r['function']:out.append('    return '+r['name']+';')
    out+=['}'];return '\n'.join(out)

HEADER=r'''
#ifndef RUCDEV
#define RUCDEV __device__
#endif
template<class T> RUCDEV T ruc_max(T a,T b) { return a>b?a:b; }
template<class T> RUCDEV T ruc_min(T a,T b) { return a<b?a:b; }
struct RRange { int lo,hi; };
template<class T> struct FArray {
 T *p; int lo1,lo2,lo3,s1,s2,s3,n1=1,n2=1,n3=1;
 RUCDEV T &operator()(int i) const { return p[(i-lo1)*s1]; }
 RUCDEV T &operator()(int i,int j) const { return p[(i-lo1)*s1+(j-lo2)*s2]; }
 RUCDEV T &operator()(int i,int j,int k) const { return p[(i-lo1)*s1+(j-lo2)*s2+(k-lo3)*s3]; }
 RUCDEV FArray operator()(RRange r) const {
  int lo=r.lo?r.lo:lo1,hi=r.hi?r.hi:lo1+n1-1;
  return {p+(lo-lo1)*s1,1,1,1,s1,0,0,hi-lo+1,1,1};
 }
 RUCDEV FArray operator()(int i,RRange r,int k) const {
  int lo=r.lo?r.lo:lo2,hi=r.hi?r.hi:lo2+n2-1;
  return {p+(i-lo1)*s1+(lo-lo2)*s2+(k-lo3)*s3,1,1,1,s2,0,0,hi-lo+1,1,1};
 }
};
template<class T,class U> RUCDEV void ruc_assign(FArray<T> a,U v) {
 for(int k=0;k<a.n3;++k)for(int j=0;j<a.n2;++j)for(int i=0;i<a.n1;++i)a.p[i*a.s1+j*a.s2+k*a.s3]=v;
}
template<class T> RUCDEV void ruc_assign(FArray<T> a,FArray<T> v) {
 for(int k=0;k<a.n3;++k)for(int j=0;j<a.n2;++j)for(int i=0;i<a.n1;++i)a.p[i*a.s1+j*a.s2+k*a.s3]=v.p[i*v.s1+j*v.s2+k*v.s3];
}
struct Arena {
 float *p; int used,stride; int *error; bool preparation_only=false;
 const int *entry_ilnb=nullptr;
 int *snowtemp_calls=nullptr;
 template<class T> RUCDEV FArray<T> alloc(int lo,int hi,int lo2=1,int hi2=1,int lo3=1,int hi3=1) {
  T *v=reinterpret_cast<T*>(p+used*stride);
  int n1=hi-lo+1,n2=hi2-lo2+1;
  used+=n1*n2*(hi3-lo3+1);
  return {v,lo,lo2,lo3,stride,n1*stride,n1*n2*stride,n1,n2,hi3-lo3+1};
 }
};
struct ArenaScope { Arena &a;int mark;RUCDEV ArenaScope(Arena &v,int m):a(v),mark(m){} RUCDEV ~ArenaScope(){a.used=mark;} };
'''

def parameter_source():
    params={k:v for g in globals.values() for k,v in g.items() if v['parameter']}
    out=[]
    for k,d in params.items():
        ex=lambda s:Expr(s,params).full()
        if d['dim']:
            vals=split(d['init'][2:-2]);out.append('__constant__ '+d['type']+' '+k+'_values[] = {'+', '.join(ex(v) for v in vals)+'};')
            out.append('RUCDEV '+d['type']+' '+k+'(int i) {return '+k+'_values[i-1];}')
        else:out.append('const '+d['type']+' '+k+' = '+ex(d['init'])+';')
    return '\n'.join(out)

def table_source():
    layout={};out=['struct Tables {'];offset=0;inits=[]
    for k,d in globals['land'].items():
        if d['parameter'] or k=='err_message':continue
        # land/ice/diagnostics targetcell are all initialized to zero.
        shape=[]
        for dim in d['dim']:
            lo,hi=dim.split(':') if ':' in dim else ('1',dim)
            params={'nlus':50,'nsltype':30,'nslope':30}
            shape.append((int(lo),params.get(hi,int(hi) if hi.isdigit() else 0)))
        count=1
        for lo,hi in shape:count*=hi-lo+1
        layout[k]={'offset':offset,'words':count,**d}
        if shape:
            assert len(shape)==1
            out.append(' FArray<'+d['type']+'> '+k+';')
            inits.append(k+'{reinterpret_cast<'+d['type']+'*>(p+'+str(offset)+'),1,1,1,1,0,0}')
        else:
            out.append(' '+d['type']+' &'+k+';')
            inits.append(k+'(reinterpret_cast<'+d['type']+'*>(p)['+str(offset)+'])')
        offset+=count
    out.append(' RUCDEV Tables(float *p):'+','.join(inits)+' {}\n};')
    return '\n'.join(out),layout,offset

if __name__=='__main__':
    tables,layout,words=table_source()
    source=HEADER+'\n'+parameter_source()+'\n'+tables+'\n'+'\n'.join(signature(r)+';' for r in routines.values())+'\n\n'+'\n\n'.join(generate(r) for r in routines.values())
    (ROOT/'scratch/generated.cu').write_text(source)
    (ROOT/'scratch/routines.json').write_text(json.dumps(list(routines.values()),indent=2))
    (ROOT/'scratch/table-layout.json').write_text(json.dumps(layout,indent=2))
    print('Translated',len(routines),'routines;',len(source),'bytes;',words,'table words')
