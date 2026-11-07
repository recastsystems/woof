"""Translate the pinned public-domain WRF column chain to CUDA.

f2c supplies statement ordering and indices. WOOF supplies device arithmetic.
No f2c runtime or external math source is included in the generated kernel.
"""
import argparse, hashlib, json, re, subprocess
from pathlib import Path

PIN = 'f52c197ed39d12e087d02c50f412d90d418f6186'

def split(s):
    out=[]; start=depth=0
    for i,c in enumerate(s):
        depth += (c=='(')-(c==')')
        if c==',' and depth==0: out.append(s[start:i].strip()); start=i+1
    out.append(s[start:].strip())
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('source'); ap.add_argument('build'); ap.add_argument('f2c'); args=ap.parse_args()
    root=Path(args.source); build=Path(args.build); build.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head!=PIN: raise RuntimeError('WRF revision differs from pinned 4.7.1')
    path=root/'phys/module_ra_gsfcsw.F'
    subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD','--','phys/module_ra_gsfcsw.F'],check=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != 'da257cf060c385c3aa4d6c3a737ec8c10773bd52034786d954bab0a0dc290766':
        raise RuntimeError('GSFC source hash differs from the audited public-domain module')
    src=subprocess.check_output(['gfortran','-E','-cpp','-P','-ffree-form','-DWRF_CHEM=0',str(path)],text=True)
    statements=[]; acc=''
    for lineno,line in enumerate(src.splitlines(),1):
        line=line.split('!',1)[0].strip().lower()
        if not line: continue
        acc += ' '+line.strip('&').strip()
        if not line.endswith('&'): statements.append(acc.strip()); acc=''
    functions={}; current=None
    for s in statements:
        m=re.match(r'subroutine\s+(\w+)\s*\((.*)\)',s)
        if m:
            current=m[1]; functions[current]={'args':split(m[2]),'body':[],'hidden':{}}
        elif s.startswith('end subroutine'): current=None
        elif current: functions[current]['body'].append(s)
    del functions['gsfc_swinit']
    functions['gsfcswrad']['args'].append('center_lat')
    out=[]
    for name,f in functions.items():
        body=[]
        if name=='gsfcswrad': body.extend(['real center_lat','real thresh,co2','parameter (thresh=1.e-9)'])
        if name=='sorad': body.extend(['real co2'])
        for s in f['body']:
            if s=='implicit none': continue
            s=re.sub(r'\bpresent\s*\([^)]*\)', '.true.',s)
            s=re.sub(r'^j_loop:\s*','',s); s=s.replace('enddo j_loop','enddo')
            if s=='qndrop2d(:,kts-1)=0.':
                body += ['do i=its,ite','qndrop2d(i,kts-1)=0.','enddo']; continue
            m=re.match(r'(real|integer|logical|character)(.*)::(.*)',s)
            if m:
                typ,attrs,vs=m.groups()
                if typ=='character': s='character*200 '+vs
                elif 'parameter' in attrs: body.append(typ+' '+','.join(x.split('=')[0] for x in split(vs))); s='parameter ('+vs+')'
                else:
                    dm=re.search(r'dimension\s*\((.*?)\)',attrs)
                    s=typ+' '+','.join(x+('('+dm[1]+')' if dm else '') for x in split(vs))
            m=re.match(r'(real|integer|logical)\s+(.+)',s)
            if m:
                typ,vs=m.groups()
                for v in split(vs):
                    dm=re.match(r'(\w+)\s*\((.*)\)',v)
                    if dm and dm[1] not in f['args'] and re.search(r'\b(m|n|ndim|np|its|ite|kts|kte)\b',dm[2]):
                        # np=75 in the driver is a fixed ozone table dimension.
                        if name=='gsfcswrad' and not re.search(r'\b(its|ite|kts|kte)\b',dm[2]): continue
                        f['hidden'][dm[1]]=(typ,dm[2]); f['args'].append(dm[1])
            body.append(s)
        out.append('subroutine '+name+'('+','.join(f['args'])+')')
        data_statements=[s for s in body if s.startswith('data ')]
        body=[s for s in body if not s.startswith('data ')]
        first=next(i for i,s in enumerate(body) if not re.match(r'^(real |integer |logical |character|parameter |implicit )',s))
        body[first:first]=data_statements + (['co2=300.e-6'] if name in ('gsfcswrad','sorad') else [])
        out+=body; out.append('end')
    fixed=[]
    for s in out:
        label=re.match(r'^(\d+)\s+(.*)',s)
        prefix=(label[1].rjust(5)+' ' if label else '      '); s=label[2] if label else s
        while len(s)>60: fixed.append(prefix+s[:60]); s=s[60:]; prefix='     &'
        fixed.append(prefix+s)
    ff=build/'gsfcsw.f'; ff.write_text('\n'.join(fixed)+'\n')
    subprocess.run([str(Path(args.f2c).resolve()),'-a','-R','-r','-g','-f',ff.name],cwd=build,check=True)
    text=(build/'gsfcsw.c').read_text()
    text=re.sub(r'/\*.*?\*/','',text,flags=re.S)
    text=re.sub(r'^#include.*\n','',text,flags=re.M)
    text=re.sub(r'^#line[^\n]*\n','',text,flags=re.M)
    # Remove local external prototypes. Forward declarations are generated below.
    text=re.sub(r'\s*extern\s+(?:int|doublereal|real)\s+[^;]+;', '',text)
    text=re.sub(r'\s*(?:double|doublereal)\s+(?:sqrt|exp|log|pow|atan|log10)\s*\([^;]+;', '',text)
    text=re.sub(r'\s*(?:double|real|doublereal)\s+(?:r_lg10|pow_rr|r_sign)\s*\([^;]+;', '',text)
    # Hidden adjustable locals give f2c runtime strides without static padding.
    for name,f in functions.items():
        pat=r'int '+name+r'_\((.*?)\)\s*\{'
        m=re.search(pat,text,re.S)
        if not m: raise RuntimeError('missing generated function '+name)
        ps=split(m[1]); count=len(f['hidden'])
        if count: ps=ps[:-count]
        alloc=[]
        for var,(typ,dims) in f['hidden'].items():
            cap=1
            for dim in split(dims):
                dim=re.sub(r'\b(m|n|ndim|its|ite|kts)\b','1',dim)
                dim=re.sub(r'\b(np|kte)\b','129',dim)
                if ':' in dim: lo,hi=dim.split(':'); value=eval('('+hi+')-('+lo+')+1',{'__builtins__':{}})
                else: value=eval(dim,{'__builtins__':{}})
                cap*=value
            var = var + '__' if '_' in var else var
            alloc.append(f'    {typ} {var}_storage[{cap*2+16}]; {typ} *{var} = {var}_storage + {cap+8};')
        replacement='__device__ int '+name+'_('+', '.join(['const real *tables']+ps)+') {\n'+'\n'.join(alloc)
        text=text[:m.start()]+replacement+text[m.end():]
    # Every DATA block becomes a verified package asset, not a CUDA literal.
    values=[]; tables=[]
    def table(match):
        var,size,body=match.groups(); nums=split(body)
        if len(nums)!=int(size): raise RuntimeError('table count mismatch '+var)
        offset=len(values); values.extend(float(x.rstrip('f')) for x in nums)
        tables.append({'name':var,'offset':offset,'count':len(nums)})
        return f'real *{var} = (real *)(tables + {offset});'
    text=re.sub(r'static real (\w+)\[(\d+)\]\s*=\s*\{(.*?)\};',table,text,flags=re.S)
    if 'static struct' in text or re.search(r'static real \w+\[',text): raise RuntimeError('unhandled table layout')
    for name in functions:
        # Definitions already have tables. Calls get that same pointer.
        text=re.sub(r'(?<!int )\b'+name+r'_\((?!const real \*tables)',name+'_(tables, ',text)
    # f2c real intrinsics are mapped to WOOF math and float32 instructions.
    text=re.sub(r'\(doublereal\)','',text)
    text=text.replace('doublereal','real').replace('double ','real ')
    text=re.sub(r'(?<![\w.])(?:\d+\.\d*|\.\d+)(?:e[+-]?\d+)?(?![\w.])',lambda m:m[0]+'f',text)
    text=re.sub(r'^static (integer|real) ',r'__device__ \1 ',text,flags=re.M)
    text=re.sub(r'\bsqrt\(', '__fsqrt_rn(',text)
    text=re.sub(r'\bexp\(', 'gfk_exp(',text)
    text=re.sub(r'\blog\(', 'gfk_log(',text)
    text=re.sub(r'\batan\(', 'atanf(',text)
    text=re.sub(r'\babs\(', 'fabsf(',text)
    prototypes=re.findall(r'__device__ int (\w+_\(.*?\))\s*\{',text,re.S)
    prologue='''// WRF 4.7.1 module_ra_gsfcsw.F, public domain.
// Generated by tools/gsfcsw_wrf461_oracle/generate.py.
// One CUDA thread follows one WRF column in statement order.
// WOOF owns the shared device math routines used here.
using real = float; using integer = int; using logical = int;

#define TRUE_ 1
#define FALSE_ 0
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#endif
__device__ real pow_rr(real *a, real *b) { return gfk_pow(*a,*b); }
__device__ real pow_dd(real *a, real *b) { return gfk_pow(*a,*b); }
__device__ real r_lg10(real *a) {
#ifdef GSFC_HOST_CHECK
    return log10f(*a);
#else
    // Evaluate in binary64 before rounding once to binary32. The older
    // float reduction inherits logf rounding and differs from native WRF
    // linked with correctly rounded log10f (glibc 2.43). This is local to
    // GSFC, and is graded with the actual native column operands.
    return __double2float_rn(log10((double)*a));
#endif
}
'''
    text=prologue+'\n'.join('__device__ int '+s+';' for s in prototypes)+'\n'+text
    import numpy as np
    asset=np.asarray(values,dtype='<f4').tobytes()
    data=Path('woof/data/wrf_radiation/gsfcsw'); data.mkdir(parents=True,exist_ok=True)
    (data/'coefficients.f32').write_bytes(asset)
    manifest={'wrf_commit':PIN,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':len(asset),'sha256':hashlib.sha256(asset).hexdigest(),'tables':tables,
        'wrf_version':'4.7.1','license':'WRF public domain',
        'license_source':'https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/LICENSE.txt',
        'source_file':'phys/module_ra_gsfcsw.F',
        'previous_identical_commit':'d66e442fccc04111067e29274c9f9eaccc3cef28'}
    (data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        from .arena import move_arrays, coalesce_arrays, single_column_fast
    except ImportError:
        from arena import move_arrays, coalesce_arrays, single_column_fast
    text, workspace_words = move_arrays(text)
    text = coalesce_arrays(text)
    manifest['workspace_words_per_column'] = workspace_words
    (data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    source=text + Path(__file__).with_name('entry.cuh').read_text()
    Path('woof/core/kernels/gsfcsw.cu').write_text('\n'.join(line.rstrip() for line in source.splitlines()).rstrip()+'\n')
    fast=single_column_fast(text) + Path(__file__).with_name('entry.cuh').read_text()
    Path('woof/core/kernels/gsfcsw_fast.cuh').write_text('\n'.join(line.rstrip() for line in fast.splitlines()).rstrip()+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
