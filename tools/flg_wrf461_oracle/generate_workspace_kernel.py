"""Move translated FLG automatic arrays into explicit per-column CUDA storage.

No arithmetic is changed. Every routine has a disjoint slice; the closed call
 graph must be acyclic. Component replay headers retain their original ABI.

Storage choices, none of which changes an operation or its order:
- small fixed-size scratch (no vertical extent, at most THREAD_LOCAL_WORDS
  words) lives in thread-local storage instead of the global column workspace;
- aerosol arrays that native rad_all touches only under ``naero>=1`` (emitted
  as ``if((0>=1))``) get no storage at all (NAERO_ONLY, checked below);
- qcfel, the four-stream band solve, keeps its active elimination window and
  its back-substitution values in registers (qcfel_window_template.cuh), and
  qcfe no longer zero-fills band slots the window never reads;
- qftisf's two-angle source-function sweeps share one pass per direction with
  their recurrences in registers (qftisf_tail_template.cuh);
- comscp's per-level phase moments are recomputed only where their inputs can
  change within a band (first g-point, cloudy levels).
"""
from pathlib import Path
import argparse,json,re

THREAD_LOCAL_WORDS=64
# rad_all arrays native RAD_FLG reads only when naero>=1 (stock RAD_FLG: naero=0).
NAERO_ONLY={('flgc_rad_all','a_wlis'),('flgc_rad_all','a_taus'),('flgc_rad_all','aprofs'),
            ('flgc_rad_all','tae'),('flgc_rad_all','wae'),('flgc_rad_all','wwae'),('flgc_rad_all','ctau')}
DEAD_GUARD='if((0>=1)){'
TEMPLATE=Path(__file__).with_name('qcfel_window_template.cuh')
QFTISF_TAIL=Path(__file__).with_name('qftisf_tail_template.cuh')
TEMPLATE_SPLIT='// @@ flgs_qcfel body (forward elimination, final pivot and back substitution)\n'


def _live_lines(body):
    """Lines of a generated routine body outside the static naero>=1 guards."""
    live=[];depth=0
    for line in body.split('\n'):
        if depth:
            depth+=line.count('{')-line.count('}')
            continue
        if line.startswith(DEAD_GUARD):
            depth=line.count('{')-line.count('}')
            continue
        live.append(line)
    return live


def _check_naero_only(bodies):
    """Every NAERO_ONLY array is used only under the dead guard: directly, or
    as an argument whose callee uses it only under the same guard."""
    for routine,name in NAERO_ONLY:
        word=re.compile(r'\b'+name+r'\b')
        for line in _live_lines(bodies[routine]):
            if not word.search(line) or re.fullmatch(r'(float|int) '+name+r'\[.+\];',line):
                continue
            call=re.match(r'(flg\w+)\(.*\);$',line)
            assert call and call[1]=='flgc_comscp_aero_cld',(routine,name,line)
            assert not any(word.search(live) for live in _live_lines(bodies[call[1]])),(name,'used live by callee')


def _qcfel_window(code):
    helpers,body=TEMPLATE.read_text().split(TEMPLATE_SPLIT)
    sig='__device__ void flgs_qcfel(float* flg_workspace'
    a=code.rindex(sig);h=code.index('{',a)+1;e=code.index('\n__device__ void ',h)
    assert 'n3=(n4-1);' in code[h:e]
    code=code[:a]+helpers+code[a:h]+'\n'+body+code[e:]
    zero=('for(i=1;((1)>0?i<=n4:i>=n4);i+=1){\nfor(j=1;((1)>0?j<=13:j>=13);j+=1){\n'
          'ab[(j-1)*(1)+(i-1)*(1*(13))]=0.0f;\n}\n}\n')
    a=code.rindex('__device__ void flgs_qcfe(float* flg_workspace');e=code.index('\n__device__ void ',a+10)
    assert code[a:e].count(zero)==1
    note='// qcfel reads only the band slots assembled below (flgw_assembled).\n'
    return code[:a]+code[a:e].replace(zero,note)+code[e:]


def _qftisf_tail(code):
    a=code.rindex('__device__ void flgs_qftisf(float* flg_workspace')
    e=code.index('\n__device__ void ',a+10)
    q=code.index('flgs_qccgts(flg_workspace,',a)
    start=code.index('if(*error) return;\n',q)+len('if(*error) return;\n')
    assert code.startswith('for(i=1;((1)>0?i<=n:i>=n);i+=1){\nxgy=',start)
    end=code.rindex('return;\n}',start,e)+len('return;\n}')
    return code[:start]+QFTISF_TAIL.read_text().rstrip('\n')+code[end:]


def _if_else_end(code,start):
    """End of the if(...){...}[else{...}] statement that starts at ``start``."""
    i=code.index('{',start);depth=0
    while True:
        c=code[i]
        depth+=c=='{';depth-=c=='}'
        i+=1
        if depth==0:
            if code.startswith('else{',i):
                i+=len('else')
                continue
            return i


def _comscp_band_invariant(code):
    """comscp's four phase-function moments per level depend only on band
    optics, except where cloud adjustment rescales ti/tw in place (levels with
    cldamnt>0). Native recomputes the same words at every g-point; compute them
    at a band's first g-point and at cloudy levels, and keep the stored words
    elsewhere. Both state sums and single-scattering albedos stay per g-point."""
    sig='__device__ void flgc_comscp_aero_cld(float* flg_workspace'
    for k in range(2):  # declaration, then definition
        a=code.index(sig) if k==0 else code.rindex(sig)
        e=code.index('\n',a)
        line=code[a:e]
        assert 'int* gas_switches,int* error)' in line
        code=code[:a]+line.replace('int* gas_switches,int* error)','int* gas_switches,int* error,int flgw_first)')+code[e:]
    a=code.rindex(sig);e=code.index('\n__device__ void ',a+10)
    body=code[a:e]
    p2=body.index('if((fw2<1.0e-20f)){');p2e=_if_else_end(body,p2)
    p1=body.index('if((fw1<1.0e-20f)){',p2e);p1e=_if_else_end(body,p1)
    body=(body[:p2]+'if(flgw_first||(cldamnt[(i-1)*(1)]>0.f)){\n'+body[p2:p2e]+'\n}'
          +body[p2e:p1]+'if(flgw_first){\n'+body[p1:p1e]+'\n}'+body[p1e:])
    code=code[:a]+body+code[e:]
    call=re.compile(r'(\nflgc_comscp_aero_cld\(flg_workspace,[^\n]*,gas_switches,error)\);\n')
    assert len(call.findall(code))==1
    return call.sub(lambda m:m[1]+',(ig1==1&&ig2==1));\n',code)


def generate(root):
    files=['flg_gases_generated.cuh','flg_streams_generated.cuh','flg_column_generated.cuh']
    source='\n'.join((root/name).read_text() for name in files)
    names=set(re.findall(r'__device__ void (flg\w+)\(',source))
    bodies=dict(re.findall(r'__device__ void (flg\w+)\([^\n]+\)\{([\s\S]*?)(?=\n__device__ void |\Z)',source))
    _check_naero_only(bodies)
    scale=0; bias=0; current=None; arrays=[];graph={name:set() for name in names};output=[]
    for line in source.splitlines():
        definition=re.match(r'__device__ void (flg\w+)\(.*\)\{',line)
        if definition:current=definition[1]
        array=re.fullmatch(r'(float|int) (\w+)\[(.+)\];',line)
        if array:
            assert current
            kind,name,extent=array.groups()
            if len(re.findall(r'\b'+name+r'\b',bodies[current]))==1:
                continue  # Native aerosol-only declarations are dead under naero=0.
            extent=re.sub(r'\bnq\b','2',extent)
            assert re.fullmatch(r'[0-9()*+ /-]+',extent),extent
            # All array extents are affine in the native vertical limit.
            sized=lambda nv1:eval(re.sub(r'\b100\b',str(nv1-1),re.sub(r'\b101\b',str(nv1),extent)),{'__builtins__':{}},{})
            words=sized(101);a=sized(3)-sized(2);b=sized(2)-2*a
            assert a*101+b==words and isinstance(words,int) and words>0
            if (current,name) in NAERO_ONLY:
                arrays.append({'routine':current,'array':name,'storage':'none (naero>=1 only)','words_scale':a,'words_bias':b})
                line=f'FLGPtr<{kind}> {name}(nullptr,32);'
            elif a==0 and words<=THREAD_LOCAL_WORDS:
                arrays.append({'routine':current,'array':name,'storage':'thread-local','words_scale':a,'words_bias':b})
                line=f'{kind} {name}_local[{words}]; FLGPtr<{kind}> {name}({name}_local,1);'
            else:
                arrays.append({'routine':current,'array':name,'storage':'workspace','offset_scale':scale,'offset_bias':bias,'words_scale':a,'words_bias':b})
                line=f'FLGPtr<{kind}> {name}(reinterpret_cast<{kind}*>(flg_workspace+32*({scale}*FLG_NV1+{bias})),32);'
                scale+=a;bias+=b
        if current and not line.startswith('__device__'):
            graph[current].update(name for name in re.findall(r'\b(flg\w+)\(',line) if name in names)
        for name in names:line=re.sub(r'\b'+name+r'\(',name+'(flg_workspace,',line)
        if line.startswith('__device__ void '):line=line.replace('(flg_workspace,','(float* flg_workspace,',1)
        # Device spans carry the workspace stride across the native call tree.
        # Tables and scalar controls remain ordinary, contiguous pointers.
        line=re.sub(r'(const )?(float|int)\* (\w+)',lambda m:
            m[0] if m[3] in {'flg_workspace','tables','table_offsets','gas_values','gas_switches','error'}
            else f'FLGPtr<{m[1] or ""}{m[2]}> {m[3]}',line)
        output.append(line)
    def visit(name,path):
        assert name not in path, f'recursive FLG workspace call: {path+[name]}'
        for child in graph[name]:visit(child,path+[name])
    for name in names:visit(name,[])
    return _comscp_band_invariant(_qftisf_tail(_qcfel_window('// Generated explicit FLG column workspace.\n#ifndef FLG_NV1\n#define FLG_NV1 101\n#endif\n'
            +f'#define FLG_WORKSPACE_SCALE {scale}\n#define FLG_WORKSPACE_BIAS {bias}\n#define FLG_WORKSPACE_WORDS ({scale}*FLG_NV1+{bias})\n'
            +'''template<class T> struct FLGPtr {
    T* p; int stride;
    __device__ FLGPtr(T* v=nullptr,int s=1):p(v),stride(s){}
    template<class U> __device__ FLGPtr(const FLGPtr<U>& a):p(a.p),stride(a.stride){}
    __device__ T& operator[](int i) const {return p[i*stride];}
    __device__ FLGPtr operator+(int i) const {return FLGPtr(p+i*stride,stride);}
};
'''+'\n'.join(output)+'\n'))),{'word_scale':scale,'word_bias':bias,'bytes_per_column_at_32_layers':4*(scale*33+bias),'arrays':arrays}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kernel_root',type=Path);p.add_argument('out',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
    code,receipt=generate(a.kernel_root);a.out.write_text(code)
    if a.receipt:a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='arrays'}))
